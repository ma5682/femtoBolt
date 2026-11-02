"""Deterministic patch-density and seam-alignment proposals, not coverage proof.

Boundary distance is measured in the undistorted selection image and mapped to
an approximate local metric scale. It is deliberately a planning heuristic: it
does not certify head clearance, full coverage, or any physical motion.
"""
from __future__ import annotations

import math
import cv2
import numpy as np

VERSION='research-adaptive-metric-floor-and-seam-v1'
MINIMUM_FRACTION=.65
MAXIMUM_FRACTION=.8
NORMAL_SCALE_DEG=35.
ALIGN_ANGLE_DEG=10.
MAX_SEAM_ANCHORS=512
MAX_SELECTION_PIXELS=8_388_608


def selection_distance(allowed,check=lambda:None):
    """Distance to an unselected pixel, with the image edge also excluded."""
    if allowed is None:return None
    mask=np.asarray(allowed)
    if mask.ndim!=2 or mask.dtype!=np.bool_ or not mask.size or mask.size>MAX_SELECTION_PIXELS:
        raise ValueError('自适应排布的选择区域无效')
    check();padded=np.pad(mask,1,constant_values=False).astype(np.uint8)
    result=cv2.distanceTransform(padded,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)[1:-1,1:-1]
    check();result.setflags(write=False)
    return result


def choose_fraction(vertices,faces,face_ids,frame,metric_span_m,width_mm,distance,intrinsic,check=lambda:None):
    """Choose .65-.8 from measured normals, transverse span and image margin."""
    vertices=np.asarray(vertices,dtype=float);faces=np.asarray(faces);ids=np.asarray(face_ids)
    k=np.asarray(intrinsic,dtype=float)
    if (vertices.ndim!=2 or vertices.shape[1]!=3 or faces.ndim!=2 or faces.shape[1]!=3
            or ids.ndim!=1 or not len(ids) or ids.dtype.kind not in 'iu'
            or np.any(ids<0) or np.any(ids>=len(faces)) or not np.isfinite(vertices).all()
            or k.shape!=(3,3) or not np.isfinite(k).all() or k[0,0]<=0 or k[1,1]<=0
            or isinstance(width_mm,bool) or not isinstance(width_mm,(float,int)) or not math.isfinite(width_mm) or width_mm<=0
            or isinstance(metric_span_m,bool) or not isinstance(metric_span_m,(float,int))
            or not math.isfinite(metric_span_m) or metric_span_m<0):
        raise ValueError('自适应排布几何或宽度无效')
    face_array=faces[ids]
    if face_array.dtype.kind not in 'iu' or np.any(face_array<0) or np.any(face_array>=len(vertices)):
        raise ValueError('自适应排布面编号无效')
    check();triangles=vertices[face_array]
    cross=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    areas=np.linalg.norm(cross,axis=1);valid=areas>0
    normal=np.asarray(frame.get('n') if isinstance(frame,dict) else None,dtype=float)
    if not valid.any() or normal.shape!=(3,) or not np.isfinite(normal).all() or np.linalg.norm(normal)<=0:
        raise ValueError('自适应排布缺少有效表面方向')
    normal=normal/np.linalg.norm(normal);unit=cross[valid]/areas[valid,None]
    max_angle=float(np.degrees(np.arccos(np.clip(np.abs(unit@normal),0.,1.))).max())
    flat=float(np.clip(1.-max_angle/NORMAL_SCALE_DEG,0.,1.))
    width_m=float(width_mm)/1000.;wide=float(np.clip((metric_span_m/width_m-1.)/3.,0.,1.))
    margin=None;boundary_factor=0.
    if distance is not None:
        distance=np.asarray(distance,dtype=float)
        if distance.ndim!=2 or not distance.size or not np.isfinite(distance).all() or np.any(distance<0):
            raise ValueError('自适应排布边界距离无效')
        centers=triangles[valid].mean(axis=1);z=centers[:,2]
        if np.all(z>0):
            uv=centers@k.T;u=uv[:,0]/uv[:,2];v=uv[:,1]/uv[:,2]
            if np.isfinite(u).all() and np.isfinite(v).all():
                # A sample outside the captured image is treated as zero margin.
                # Match the selection component's nearest-pixel convention.
                x=np.floor(u+.5).astype(np.int64);y=np.floor(v+.5).astype(np.int64)
                inside=(x>=0)&(x<distance.shape[1])&(y>=0)&(y<distance.shape[0])
                metric=np.zeros(len(x),dtype=float)
                metric[inside]=distance[y[inside],x[inside]]*z[inside]/max(k[0,0],k[1,1])
                margin=round(float(metric.min()),9);boundary_factor=float(np.clip(margin/width_m,0.,1.))
    fraction=round(float(MINIMUM_FRACTION+(MAXIMUM_FRACTION-MINIMUM_FRACTION)*flat*wide*boundary_factor),6)
    check()
    return dict(version=VERSION,minimum_fraction=fraction,max_normal_angle_deg=max_angle,
        transverse_metric_span_m=float(metric_span_m),width_mm=float(width_mm),
        image_boundary_margin_proxy_m=margin,flat_factor=flat,width_factor=wide,boundary_factor=boundary_factor,
        semantics='proposal_only_image_margin_not_physical_clearance_or_coverage')


def seam_anchors(patch,frame,previous,first,last,check=lambda:None):
    """Project earlier adjacent, near-parallel rows onto this patch's axis."""
    if (not isinstance(patch,dict) or not isinstance(frame,dict) or not isinstance(previous,list)
            or not all(isinstance(x,(float,int)) and not isinstance(x,bool) and math.isfinite(x) for x in (first,last))
            or last<first):raise ValueError('分区接缝对齐输入无效')
    seams={tuple(edge) for edge in patch.get('seam_edges',[])}
    origin=np.asarray(frame.get('origin'),dtype=float);axis=np.asarray(frame.get('e2'),dtype=float)
    if origin.shape!=(3,) or axis.shape!=(3,) or not np.isfinite([origin,axis]).all() or np.linalg.norm(axis)<=0:
        raise ValueError('分区接缝对齐方向无效')
    axis=axis/np.linalg.norm(axis);anchors=[];sources=[]
    for prior in previous:
        check()
        if not seams.intersection(map(tuple,prior.get('seam_edges',[]))):continue
        ref=prior.get('frame') or {};old_origin=np.asarray(ref.get('origin'),dtype=float);old_axis=np.asarray(ref.get('e2'),dtype=float)
        if (old_origin.shape!=(3,) or old_axis.shape!=(3,) or not np.isfinite([old_origin,old_axis]).all()
                or np.linalg.norm(old_axis)<=0):raise ValueError('相邻分区扫描方向无效')
        old_axis=old_axis/np.linalg.norm(old_axis)
        if abs(float(axis@old_axis))<math.cos(math.radians(ALIGN_ANGLE_DEG)):continue
        levels=np.asarray(prior.get('positions_m',[]),dtype=float)
        if levels.ndim!=1 or not np.isfinite(levels).all():raise ValueError('相邻分区扫描位置无效')
        candidates=(old_origin-origin)@axis+levels*(old_axis@axis)
        found=sorted(set(float(v) for v in candidates if first<v<last))
        if len(anchors)+len(found)>MAX_SEAM_ANCHORS:raise ValueError('相邻分区接缝候选超过预算')
        if found:sources.append(dict(seed_face=prior.get('seed_face'),anchors=len(found)));anchors.extend(found)
    return sorted(set(anchors)),sources
