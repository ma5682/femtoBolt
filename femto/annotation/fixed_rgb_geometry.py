"""Fixed, undistorted reference RGB -> existing W0 mesh. No inferred depth."""
from __future__ import annotations
from typing import Any

import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
import open3d as o3d
from femto.io_utils import imread_unicode
from femto.annotation.scene_annotations import AnnotationStore, region_boundary
from femto.geometry.rgbd_frames import distortion_maps
from femto.reports.scene_scale_validation import read_geometry_safe, resolve_reconstruction


MAX_IMAGE_PIXELS = 8_388_608
MAX_MESH_TRIANGLES = 4_000_000
MAX_MESH_FILE_BYTES = 512 * 1024 * 1024
MAX_ANNOTATIONS = 2000
MAX_BOUNDARY_SAMPLES = 300_000
MAX_OVERLAY_BYTES = 64 * 1024 * 1024
MAX_POLYGON_CORNERS = 512
VISIBILITY_TOLERANCE_M = 1e-5  # Ray float roundoff, not physical accuracy.


class GeometryTaskCancelled(RuntimeError):
    pass


class GeometryBudget:
    """Cooperative budget for one operation, never for a viewer's lifetime."""
    def __init__(self, cancel=None, seconds=120):
        self.cancel = cancel
        self.deadline = time.monotonic() + seconds

    def check(self) -> None:
        if self.cancel is not None and self.cancel.is_set():
            raise GeometryTaskCancelled('Geometry task cancelled')
        if time.monotonic() > self.deadline:
            raise TimeoutError('Geometry operation exceeded its 120 s budget')


def _check_function(check):
    return check if check is not None else GeometryBudget().check


def _mesh_budget_before_load(path):
    """Check uncompressed PLY counts before allocating native mesh buffers."""
    path=Path(path)
    if path.stat().st_size>MAX_MESH_FILE_BYTES:
        raise ValueError('Mesh file exceeds the 512 MiB input budget')
    with path.open('rb') as stream:header=stream.read(65536)
    if not header.startswith(b'ply') or b'end_header' not in header:
        raise ValueError('A bounded PLY header is required for fixed RGB mapping')
    counts={}
    for line in header.split(b'end_header',1)[0].splitlines():
        parts=line.split()
        if len(parts)==3 and parts[0]==b'element' and parts[1] in (b'vertex',b'face'):
            counts[parts[1]]=int(parts[2])
    if any(counts.get(key,0)>MAX_MESH_TRIANGLES for key in (b'vertex',b'face')):
        raise ValueError('Fixed RGB mesh exceeds the 4000000 triangle memory budget')


def region_vertex_summary(mesh: Any, triangle_indices: Any) -> dict[str, Any]:
    """Vertices of the selected mesh, in its unchanged W0 metre coordinates."""
    indices=np.unique(np.asarray(mesh.triangles)[triangle_indices].ravel())
    points=np.asarray(mesh.vertices)[indices]
    distances=np.linalg.norm(points,axis=1)*1000
    return dict(vertex_count=len(indices),minimum_distance_mm=float(distances.min()),
                maximum_distance_mm=float(distances.max()))


def vertex_cloud_view(mesh: Any,projection: Any) -> dict[str, Any]:
    """Display actual mesh vertices supported by the reference nearest surface."""
    points=np.asarray(mesh.vertices)
    uv=projection.project(points)
    finite=np.isfinite(uv).all(axis=1)&np.isfinite(points).all(axis=1)&(points[:,2]>0)
    indices=np.flatnonzero(finite)
    pixel=np.floor(uv[indices]+.5).astype(np.int64)
    h,w=projection.ids.shape
    inside=(pixel[:,0]>=0)&(pixel[:,0]<w)&(pixel[:,1]>=0)&(pixel[:,1]<h)
    indices=indices[inside];pixel=pixel[inside]
    z=projection.z[pixel[:,1],pixel[:,0]]
    indices=indices[np.isfinite(z)&(np.abs(points[indices,2]-z)<=.01)]
    distances=np.linalg.norm(points[indices],axis=1)*1000
    limits=(float(distances.min()),float(distances.max())) if len(indices) else (0.,0.)
    return dict(uv=uv[indices],z=points[indices,2],distance_mm=distances,limits=limits)


def render_vertex_cloud(view: Any,width: int,height: int,scale: Any,offset: Any,distance: bool=False) -> tuple[np.ndarray, np.ndarray]:
    image=np.full((height,width,3),[8,18,28],np.uint8)
    pixels=np.floor(view['uv']*scale+offset+.5).astype(np.int64)
    valid=(pixels[:,0]>=0)&(pixels[:,0]<width)&(pixels[:,1]>=0)&(pixels[:,1]<height)
    ids=np.flatnonzero(valid);pixels=pixels[ids]
    flat=pixels[:,1]*width+pixels[:,0]
    order=np.lexsort((view['z'][ids],flat))
    _,first=np.unique(flat[order],return_index=True)
    chosen=ids[order[first]];pixels=pixels[order[first]]
    colors=np.tile([180,220,235],(len(chosen),1))
    if distance:
        low,high=view['limits'];t=(view['distance_mm'][chosen]-low)/max(high-low,1e-9)
        colors=np.c_[40+215*t,180-80*t,255-215*t]
    image[pixels[:,1],pixels[:,0]]=colors.astype(np.uint8)
    mask=np.zeros((height,width),bool);mask[pixels[:,1],pixels[:,0]]=True
    return image,mask


def selected_surface_mask(projection: Any, triangle_indices: Any, width: int, height: int, scale: Any, offset: Any) -> Any:
    """Viewport mask of nearest supported hits belonging to selected triangles.

    No polygon fill across holes, and no selection of surfaces behind the hit.
    Memory is bounded by the viewport, regardless of zoom.
    """
    x=np.floor((np.arange(width)-offset[0])/scale+.5).astype(np.int64)
    y=np.floor((np.arange(height)-offset[1])/scale+.5).astype(np.int64)
    inside_x=(np.arange(width)>=offset[0])&(x>=0)&(x<projection.ids.shape[1])
    inside_y=(np.arange(height)>=offset[1])&(y>=0)&(y<projection.ids.shape[0])
    hits=projection.ids[np.clip(y,0,projection.ids.shape[0]-1)[:,None],
                        np.clip(x,0,projection.ids.shape[1]-1)[None,:]]
    return np.isin(hits,triangle_indices)&inside_y[:,None]&inside_x[None,:]&(hits>=0)


def polygon_contains(points: Any, polygon: Any) -> np.ndarray:
    polygon = np.asarray(polygon, float)
    if (polygon.ndim != 2 or polygon.shape[1:] != (2,) or not 3 <= len(polygon) <= MAX_POLYGON_CORNERS
            or not np.isfinite(polygon).all()):
        raise ValueError('Invalid polygon')
    x,y = np.asarray(points).T
    inside = np.zeros(len(x),bool)
    for a,b in zip(polygon,np.roll(polygon,-1,axis=0)):
        if a[1]!=b[1]:
            inside ^= ((a[1]>y)!=(b[1]>y)) & (x<(b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0])
    return inside


class FixedProjection:
    """Pixel centres use OpenCV integer coordinates. t_hit is camera Z in metres."""
    def __init__(self, mesh, k, depth_mm, minimum=.25, maximum=2.21, *, check=None):
        check = _check_function(check)
        check()
        self.mesh, self.k = mesh, np.asarray(k,float)
        self.depth = np.asarray(depth_mm)
        self.height,self.width = self.depth.shape
        if self.depth.size > MAX_IMAGE_PIXELS or len(mesh.triangles) > MAX_MESH_TRIANGLES:
            raise ValueError('Fixed RGB mapping exceeds the image / mesh memory budget')
        if self.k.shape!=(3,3) or not np.isfinite(self.k).all() or min(self.k[0,0],self.k[1,1])<=0:
            raise ValueError('Invalid RGB intrinsics')
        self.minimum,self.maximum=minimum,maximum
        if not np.isfinite([minimum,maximum]).all() or not 0<minimum<maximum:
            raise ValueError('Invalid reference depth range')
        self.scene=o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        self.z=np.full(self.depth.shape,np.nan,np.float32)
        self.ids=np.full(self.depth.shape,-1,np.int64)
        # Reasons remain separate from coordinates: rejected pixels never get XYZ.
        self.reasons=np.full(self.depth.shape,1,np.uint8)
        for top in range(0,self.height,64):
            check()
            yy,xx=np.mgrid[top:min(top+64,self.height),:self.width]
            rays=np.zeros((*xx.shape,6),np.float32)
            rays[...,3]=(xx-self.k[0,2])/self.k[0,0]
            rays[...,4]=(yy-self.k[1,2])/self.k[1,1];rays[...,5]=1
            result=self.scene.cast_rays(o3d.core.Tensor(rays))
            z=result['t_hit'].numpy();ids=result['primitive_ids'].numpy()
            raw=self.depth[top:top+len(z)].astype(float)/1000
            measured=(raw>=minimum)&(raw<=maximum)
            good=np.isfinite(z)&(z>=minimum)&(z<=maximum)
            # A visible RGB foreground must not select an inconsistent deeper mesh.
            good &= ~measured | (np.abs(z-raw)<=.01)
            reasons=np.where(np.isfinite(z),2,1).astype(np.uint8)
            reasons[np.isfinite(z)&(z>=minimum)&(z<=maximum)&measured&(np.abs(z-raw)>.01)]=3
            reasons[good]=0
            self.reasons[top:top+len(z)]=reasons
            self.z[top:top+len(z)]=np.where(good,z,np.nan)
            self.ids[top:top+len(z)]=np.where(good,ids.astype(np.int64),-1)
        self.valid=self.ids>=0
        self.new_support=self.valid & ~((self.depth>=minimum*1000)&(self.depth<=maximum*1000))
        check()

    def pixel_reason(self,x: Any,y: Any) -> Any:
        if not np.isfinite([x,y]).all():return 'invalid_pixel'
        if not (0<=x<self.width and 0<=y<self.height):return 'outside_image'
        ix,iy=int(np.floor(x+.5)),int(np.floor(y+.5))
        if ix>=self.width or iy>=self.height:return 'outside_image'
        if hasattr(self,'image_valid') and not self.image_valid[iy,ix]:return 'no_source_pixel'
        return {0:'supported',1:'no_surface',2:'outside_depth_range',3:'depth_conflict',4:'no_source_pixel'}[int(self.reasons[iy,ix])]

    def pixel(self,x: Any,y: Any) -> dict[str, Any] | None:
        if self.pixel_reason(x,y)!='supported':return None
        if not np.isfinite([x,y]).all() or not (0<=x<self.width and 0<=y<self.height):return None
        ix,iy=int(np.floor(x+.5)),int(np.floor(y+.5))
        if ix>=self.width or iy>=self.height or not self.valid[iy,ix]:return None
        z=float(self.z[iy,ix])
        return dict(point_world_m=[(ix-self.k[0,2])*z/self.k[0,0],(iy-self.k[1,2])*z/self.k[1,1],z],
            triangle_index=int(self.ids[iy,ix]),coordinate_frame='world_rdf',unit='metre',
            source='fixed_reference_rgb_mesh_ray',reference_pixel=[ix,iy],
            scale_validation='not_physically_validated')

    def project(self,xyz: Any) -> Any:
        p=np.asarray(xyz,float)
        with np.errstate(divide='ignore',invalid='ignore'):
            return np.column_stack((p[:,0]*self.k[0,0]/p[:,2]+self.k[0,2],
                                    p[:,1]*self.k[1,1]/p[:,2]+self.k[1,2]))

    def visible(self,xyz: Any,*,check: Any=None) -> Any:
        """Occlusion uses the first continuous ray hit, not the 10 mm depth gate."""
        check = _check_function(check)
        points=np.asarray(xyz,float);uv=self.project(points)
        valid=(points[:,2]>0)&np.isfinite(points).all(axis=1)&np.isfinite(uv).all(axis=1)
        valid &= (uv[:,0]>=0)&(uv[:,0]<self.width-.5)&(uv[:,1]>=0)&(uv[:,1]<self.height-.5)
        ids=np.flatnonzero(valid)
        pix=np.floor(uv[ids]+.5).astype(int)
        if hasattr(self,'image_valid'):valid[ids] &= self.image_valid[pix[:,1],pix[:,0]]
        ids=np.flatnonzero(valid)
        for start in range(0,len(ids),32768):
            check()
            group=ids[start:start+32768]
            p=points[group]
            rays=np.c_[np.zeros((len(group),3)),p/p[:,2:3]].astype(np.float32)
            z=self.scene.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy()
            pix=np.floor(uv[group]+.5).astype(int)
            raw=self.depth[pix[:,1],pix[:,0]].astype(float)/1000
            measured=(raw>=self.minimum)&(raw<=self.maximum)
            valid[group] &= (np.isfinite(z)&(z>=self.minimum)&(z<=self.maximum)
                             &(np.abs(z-p[:,2])<=VISIBILITY_TOLERANCE_M)
                             &(~measured | (np.abs(z-raw)<=.01)))
        return valid

    def region(self,polygon: Any,*,check: Any=None,mask: Any=None) -> Any:
        """Selected whole triangles: visible centroid inside the polygon, or inside `mask` when given.

        The mask is the physician's authoritative pixel selection; the polygon path is the
        legacy rule kept for older records. Both apply the same occlusion and depth gates."""
        check = _check_function(check)
        check()
        centers=np.asarray(self.mesh.vertices)[np.asarray(self.mesh.triangles)].mean(axis=1)
        uv=self.project(centers)
        good=(centers[:,2]>0)&np.isfinite(uv).all(axis=1)
        good &= (uv[:,0]>=0)&(uv[:,0]<self.width-1)&(uv[:,1]>=0)&(uv[:,1]<self.height-1)
        ids=np.flatnonzero(good)
        if mask is not None:
            mask=np.asarray(mask,bool)
            if mask.shape!=(self.height,self.width):raise ValueError('Selection mask does not match the reference image')
            centre_pixels=np.floor(uv[ids]+.5).astype(int)
            ids=ids[mask[centre_pixels[:,1],centre_pixels[:,0]]]
        else:
            ids=ids[polygon_contains(uv[ids],polygon)]
        if not len(ids):return []
        # Exact centroid rays avoid assigning background triangles at an edge pixel.
        rays=np.c_[np.zeros((len(ids),3)), centers[ids]/centers[ids,2:3]].astype(np.float32)
        hit_ids=[];hit_z=[]
        for start in range(0,len(rays),32768):
            check()
            hits=self.scene.cast_rays(o3d.core.Tensor(rays[start:start+32768]))
            hit_ids.append(hits['primitive_ids'].numpy());hit_z.append(hits['t_hit'].numpy())
        pix=np.floor(uv[ids]+.5).astype(int)
        z=np.concatenate(hit_z);raw=self.depth[pix[:,1],pix[:,0]].astype(float)/1000
        measured=(raw>=self.minimum)&(raw<=self.maximum)
        valid=(np.concatenate(hit_ids)==ids)&np.isfinite(z)&(z>=self.minimum)&(z<=self.maximum)
        valid &= ~measured | (np.abs(z-raw)<=.01)
        if hasattr(self,'image_valid'):valid &= self.image_valid[pix[:,1],pix[:,0]]
        return ids[valid].tolist()

    def region_support(self,polygon: Any,*,check: Any=None,mask: Any=None) -> dict[str, Any]:
        """Count integer reference-image pixels. This is not a surface area."""
        check = _check_function(check)
        if mask is not None:
            mask=np.asarray(mask,bool)
            if mask.shape!=(self.height,self.width):raise ValueError('Selection mask does not match the reference image')
        else:
            polygon_contains(np.empty((0,2)),polygon)
        requested=source=supported=missing_depth=0
        for top in range(0,self.height,64):
            check()
            yy,xx=np.mgrid[top:min(top+64,self.height),:self.width]
            inside=(mask[top:top+len(yy)] if mask is not None else
                    polygon_contains(np.c_[xx.ravel(),yy.ravel()],polygon).reshape(xx.shape))
            image_valid=getattr(self,'image_valid',None)
            src=np.ones_like(inside) if image_valid is None else image_valid[top:top+len(inside)]
            good=inside&src&self.valid[top:top+len(inside)]
            requested+=int(inside.sum());source+=int((inside&src).sum());supported+=int(good.sum())
            missing_depth+=int((good&self.new_support[top:top+len(inside)]).sum())
        return dict(schema='femto-fixed-rgb-region-support-v1',unit='pixel',
                    coordinate_frame='undistorted_reference_rgb',
                    source='bound_mesh_first_hit_and_reference_raw_depth',
                    interpretation='image_pixel_support_not_surface_area',
                    selection_geometry=('selection_mask' if mask is not None else 'screen_polygon'),
                    requested_pixels=requested,source_pixels=source,
                    geometry_supported_pixels=supported,unsupported_pixels=requested-supported,
                    reference_depth_missing_supported_pixels=missing_depth,
                    support_ratio=supported/requested if requested else None)

    def visible_boundary(self,boundary: Any,*,check: Any=None) -> Any:
        """Clip display-only boundary lines at occlusion; stored W0 edges stay intact.

        Samples are at most one reference-image pixel apart. A run is drawn only
        between consecutive supported samples, never across a rejected middle.
        """
        check = _check_function(check)
        segments=np.asarray(boundary,float)
        if not segments.size:return np.empty((0,2,2),dtype=np.float64)
        output=[];samples_used=0
        for segment in segments:
            check()
            if not np.isfinite(segment).all() or np.any(segment[:,2]<=0):continue
            uv=self.project(segment)
            if not np.isfinite(uv).all():continue
            direction=uv[1]-uv[0];low,high=0.,1.
            for axis,limit in ((0,self.width-1),(1,self.height-1)):
                if abs(direction[axis])<1e-12:
                    if not 0<=uv[0,axis]<=limit:high=-1.;break
                else:
                    a,b=(0-uv[0,axis])/direction[axis],(limit-uv[0,axis])/direction[axis]
                    low=max(low,min(a,b));high=min(high,max(a,b))
            if low>high:continue
            count=max(2,int(np.ceil(np.max(np.abs(direction))*(high-low)))+1)
            samples_used+=count
            if samples_used>MAX_BOUNDARY_SAMPLES:
                raise ValueError('Region boundary exceeds the 300000 pixel display-sample budget')
            t=np.linspace(low,high,count)
            pixels=uv[0]+t[:,None]*direction
            # A 3D line projects linearly in u/v but depth is perspective-correct.
            z=1/((1-t)/segment[0,2]+t/segment[1,2])
            points=np.c_[(pixels[:,0]-self.k[0,2])*z/self.k[0,0],
                         (pixels[:,1]-self.k[1,2])*z/self.k[1,1],z]
            valid=self.visible(points,check=check)
            starts=np.flatnonzero(valid&~np.r_[False,valid[:-1]])
            ends=np.flatnonzero(valid&~np.r_[valid[1:],False])
            output.extend([pixels[a].tolist(),pixels[b].tolist()] for a,b in zip(starts,ends) if b>a)
        return np.asarray(output,dtype=np.float64).reshape(-1,2,2)


class FixedRGBModel:
    def __init__(self,result,*,cancel=None):
        budget=GeometryBudget(cancel);budget.check()
        self.reconstruction=resolve_reconstruction(Path(result))
        mesh_path=self.reconstruction/'scene_tsdf_mesh_m.ply'
        _mesh_budget_before_load(mesh_path)
        self.mesh=read_geometry_safe(mesh_path,mesh=True)
        budget.check()
        if len(self.mesh.triangles)>MAX_MESH_TRIANGLES:
            raise ValueError('Fixed RGB mesh exceeds the 4000000 triangle memory budget')
        self.store=AnnotationStore(self.mesh,mesh_path)
        reference=self.store.origin_reference
        if reference['kind']!='initial_color_camera':
            raise ValueError('Initial color-camera origin cannot be verified; use the advanced viewer for this model.')
        dataset=Path(reference['source_dataset'])
        rows=[json.loads(s) for s in (dataset/'frames.jsonl').read_text(encoding='utf-8-sig').splitlines() if s.strip()]
        frame=next(r for r in rows if r['frame']==reference['reference_frame_id'])
        intrinsic=json.loads((dataset/'calibration/color_intrinsic.json').read_text(encoding='utf-8-sig'))
        width,height=int(intrinsic['width']),int(intrinsic['height'])
        if min(width,height)<=0 or width*height>MAX_IMAGE_PIXELS:
            raise ValueError('Fixed RGB mapping exceeds the image / mesh memory budget')
        mx,my,k,w,h=distortion_maps(dataset)
        rgb_path,depth_path=dataset/frame['rgb'],dataset/frame['depth']
        bgr=imread_unicode(rgb_path,cv2.IMREAD_COLOR);depth=imread_unicode(depth_path,cv2.IMREAD_UNCHANGED)
        if bgr.shape!=(h,w,3) or depth.shape!=(h,w):raise ValueError('Reference image dimensions mismatch')
        self.rgb=cv2.remap(bgr,mx,my,cv2.INTER_LINEAR)[:,:,::-1].copy()
        depth=cv2.remap(depth,mx,my,cv2.INTER_NEAREST)
        report=json.loads((self.reconstruction/'reconstruction_report.json').read_text(encoding='utf-8-sig'))
        params=report['parameters']
        self.projection=FixedProjection(self.mesh,k,depth,params['depth_min_m'],params['depth_max_m'],check=budget.check)
        self.projection.image_valid=(mx>=0)&(mx<=w-1)&(my>=0)&(my<=h-1)
        invalid=~self.projection.image_valid
        self.projection.z[invalid]=np.nan;self.projection.ids[invalid]=-1
        self.projection.valid[invalid]=False;self.projection.new_support[invalid]=False
        self.projection.reasons[invalid]=4
        tint=self.rgb.astype(np.float32)
        valid,new=self.projection.valid,self.projection.new_support
        tint[valid]=tint[valid]*.6+np.array([40,220,90],dtype=np.float32)*.4
        tint[new]=self.rgb[new]*.5+np.array([40,210,240],dtype=np.float32)*.5
        self.support_rgb=tint.astype(np.uint8)
        self.view_reference=dict(reference_frame_id=frame['frame'],session_id=reference['session_id'],
            rgb_sha256=hashlib.sha256(rgb_path.read_bytes()).hexdigest(),depth_sha256=hashlib.sha256(depth_path.read_bytes()).hexdigest(),
            undistorted_intrinsic=k.tolist(),image_size=[w,h],view='initial_color_camera_locked',
            mapping='undistorted RGB pixel -> nearest supported W0 mesh intersection')
        poses=[np.asarray(json.loads((dataset/r['pose']).read_text(encoding='utf-8-sig'))['T_WC']) for r in rows]
        self.max_radius_mm=max(float(np.linalg.norm(t[:3,3])*1000) for t in poses)
        self._overlay_cache={}
        budget.check()

    def _overlay(self,kind,item,check=None):
        if kind=='point':
            p=[item['point_world_m']]
            return self.projection.project(p)[0].tolist() if self.projection.visible(p,check=check)[0] else None
        return self.projection.visible_boundary(item['boundary_world_m'],check=check)

    def display_for(self,kind: str,item: Any) -> Any:
        key=(kind,item['annotation_id'])
        if not hasattr(self,'_overlay_cache'):self._overlay_cache={}
        if key not in self._overlay_cache and kind=='point':
            self._overlay_cache[key]=self._overlay(kind,item)
        return self._overlay_cache.get(key)

    def _current_cache(self):
        keys={(kind,item['annotation_id']) for kind,items in (('point',self.store.points),('region',self.store.regions)) for item in items}
        return {key:value for key,value in getattr(self,'_overlay_cache',{}).items() if key in keys}

    @staticmethod
    def _check_overlay_budget(cache,extra=None):
        total=sum(np.asarray(value).nbytes+256 for value in cache.values() if value is not None)
        if extra is not None:total+=np.asarray(extra).nbytes+256
        if total>MAX_OVERLAY_BYTES:raise ValueError('Annotation overlays exceed the 64 MiB display memory budget')

    def prepare_display(self,*,cancel: Any=None) -> None:
        budget=GeometryBudget(cancel);cache=self._current_cache()
        for kind,items in (('point',self.store.points),('region',self.store.regions)):
            for item in items:
                budget.check();key=(kind,item['annotation_id'])
                if key not in cache:cache[key]=self._overlay(kind,item,budget.check)
                self._check_overlay_budget(cache)
        budget.check();self._overlay_cache=cache

    def pick(self,x: Any,y: Any,name: str,note: str) -> Any:
        if len(self.store.points)+len(self.store.regions)>=MAX_ANNOTATIONS:
            raise ValueError('Annotation count exceeds the 2000 item memory budget')
        hit=self.projection.pixel(x,y)
        if hit is None:return None
        cache=self._current_cache()
        overlay=self.projection.project([hit['point_world_m']])[0].tolist()
        self._check_overlay_budget(cache,overlay)
        support_kind=('reference_depth_missing_mesh_supported' if self.projection.new_support[int(y),int(x)]
                      else 'reference_depth_consistent_mesh')
        item=self.store.add_point(dict(hit,view_reference=self.view_reference,
            measurement_support=dict(kind=support_kind, representation='ray_TSDF_intersection',
                physical_accuracy_verified=False)),name=name,note=note)
        cache[('point',item['annotation_id'])]=overlay;self._overlay_cache=cache
        return item

    def region(self,polygon: Any,name: str,note: str,*,cancel: Any=None,replace_existing: bool=False,
               selection: Any=None) -> Any:
        """Create a region; `selection` = dict(mask, forbidden_mask, method) records physician intent.

        With a selection mask the triangles are chosen by centroid pixel inside that mask and the
        record carries the authoritative intent plus a per-pixel geometry-support diagnosis.
        Without it the legacy polygon rule applies and no intent is claimed."""
        budget=GeometryBudget(cancel)
        if len(self.store.points)+len(self.store.regions)>=MAX_ANNOTATIONS:
            raise ValueError('Annotation count exceeds the 2000 item memory budget')
        mask=None
        if selection is not None:
            from femto.annotation.selection_intent import build_selection_intent,build_selection_support,SELECTION_RULE
            mask=np.asarray(selection['mask'],bool)
            intent=build_selection_intent(mask,forbidden_mask=selection.get('forbidden_mask'),method=selection['method'])
            budget.check()
            support_record=build_selection_support(mask,self.projection)
        ids=self.projection.region(polygon,check=budget.check,mask=mask)
        if not ids:return None
        support=self.projection.region_support(polygon,check=budget.check,mask=mask)
        overlay=self.projection.visible_boundary(region_boundary(self.store.mesh,ids),check=budget.check)
        cache=self._current_cache();self._check_overlay_budget(cache,overlay)
        budget.check()
        record=dict(triangle_indices=ids,name=name,note=note,
            screen_polygon=polygon,view_reference=self.view_reference,
            screen_polygon_unit='pixel',screen_polygon_frame='undistorted_reference_rgb',
            geometry_support=support,
            selection_rule='visible_triangle_centroid_in_fixed_reference_polygon')
        if selection is not None:
            record.update(selection_rule=SELECTION_RULE,selection_intent=intent,selection_support=support_record)
        item=self.store.add_region(record,replace_existing=replace_existing)
        cache[('region',item['annotation_id'])]=overlay;self._overlay_cache=cache
        return item

    def load(self,path: str | Path,*,cancel: Any=None) -> Any:
        budget=GeometryBudget(cancel);budget.check()
        if Path(path).stat().st_size>64*1024*1024:
            raise ValueError('Annotation file exceeds the 64 MiB input budget')
        data=json.loads(Path(path).read_text(encoding='utf-8-sig'))
        if not isinstance(data,dict):raise ValueError('Invalid annotation document')
        if len(data.get('points',[]))+len(data.get('regions',[]))>MAX_ANNOTATIONS:
            raise ValueError('Annotation count exceeds the 2000 item memory budget')
        for item in data.get('points',[])+data.get('regions',[]):
            if 'view_reference' in item and item['view_reference']!=self.view_reference:
                raise ValueError('Reference RGB evidence mismatch')
        # Validate and build derived UI state separately. Failed/cancelled import
        # leaves the current annotations, saved baseline and undo history intact.
        candidate=AnnotationStore(self.store.mesh,self.store.source)
        result=candidate.load(path)
        overlays={}
        for kind,items in (('point',candidate.points),('region',candidate.regions)):
            for item in items:
                budget.check()
                if kind=='region':
                    from femto.annotation.selection_intent import SELECTION_RULE,build_selection_support,intent_mask
                    if (item.get('view_reference')==self.view_reference and 'screen_polygon' in item
                            and item.get('selection_rule') in ('visible_triangle_centroid_in_fixed_reference_polygon',SELECTION_RULE)
                            and hasattr(self,'projection')):
                        if (item.get('screen_polygon_unit','pixel')!='pixel' or
                                item.get('screen_polygon_frame','undistorted_reference_rgb')!='undistorted_reference_rgb'):
                            raise ValueError('Region image coordinates / units mismatch')
                        if item.get('selection_rule')==SELECTION_RULE:
                            if item.get('selection_intent') is None:raise ValueError('Mask-selected region lost its selection intent')
                            # Derived diagnosis is recomputed from the bound evidence; the intent mask is not.
                            mask=intent_mask(item['selection_intent'])
                            item['geometry_support']=self.projection.region_support(item['screen_polygon'],check=budget.check,mask=mask)
                            item['selection_support']=build_selection_support(mask,self.projection)
                        else:
                            item['geometry_support']=self.projection.region_support(item['screen_polygon'],check=budget.check)
                    else:
                        item.pop('geometry_support',None)
                if hasattr(self,'projection'):
                    overlays[(kind,item['annotation_id'])]=self._overlay(kind,item,budget.check)
                    self._check_overlay_budget(overlays)
        budget.check()
        candidate.mark_saved()
        self.store=candidate;self._overlay_cache=overlays
        result.update(points=candidate.points,regions=candidate.regions)
        return result
