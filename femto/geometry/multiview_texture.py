"""Original-resolution photo atlas for a fused mesh, without changing geometry.

One calibrated source image per triangle; depth-consistency rejects occluded
or unsupported projections. Unassigned faces explicitly retain fused color.
"""
import hashlib
import json
import math
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import trimesh

from femto.io_utils import imread_unicode, write_json


def project_visible(vertices, faces, pose, K, distortion, depth, tolerance_m=.015):
    """Return raw-image projections and a score for faces supported at 4 samples."""
    camera = (vertices - pose[:3, 3]) @ pose[:3, :3]
    centers = camera[faces].mean(axis=1)
    samples = np.concatenate((camera, centers))
    projected, _ = cv2.projectPoints(samples, np.zeros(3), np.zeros(3), K, distortion)
    xy = projected.reshape(-1,2)
    h,w = depth.shape
    good = np.isfinite(xy).all(axis=1) & (samples[:,2] > 0)
    good &= (xy[:,0] >= 1) & (xy[:,0] <= w-2) & (xy[:,1] >= 1) & (xy[:,1] <= h-2)
    pixels = np.rint(np.nan_to_num(xy, nan=0, posinf=0, neginf=0)).clip(-1e8,1e8).astype(np.int64)
    measured = depth[pixels[:,1].clip(0,h-1),pixels[:,0].clip(0,w-1)] / 1000.
    good &= (measured > 0) & (np.abs(measured-samples[:,2]) <= tolerance_m)
    valid = good[:len(vertices)][faces].all(axis=1) & good[len(vertices):]
    edges = camera[faces[:,1]]-camera[faces[:,0]], camera[faces[:,2]]-camera[faces[:,0]]
    normal = np.cross(*edges)
    cosine = -np.einsum('ij,ij->i',normal,centers) / np.maximum(np.linalg.norm(normal,axis=1)*np.linalg.norm(centers,axis=1),1e-12)
    # No back-face photo transfer. Favor frontal and nearby supported views.
    score = np.where(valid & (cosine > .15), cosine**2 / np.maximum(centers[:,2]**2,1e-8), -1.)
    return xy[:len(vertices)][faces], score


def export_multiview_texture(mesh, dataset, frames, poses, output, exports, max_views=12):
    intr = json.loads((dataset/'calibration/color_intrinsic.json').read_text(encoding='utf-8'))
    dist = json.loads((dataset/'calibration/distortion.json').read_text(encoding='utf-8'))['rgb']
    if 'BROWN_CONRADY_K6' not in dist['model']:
        raise ValueError('Texture projection requires calibrated RGB K6 distortion')
    K = np.array([[intr['fx'],0,intr['cx']],[0,intr['fy'],intr['cy']],[0,0,1]],float)
    coeff = np.array([dist[k] for k in ('k1','k2','p1','p2','k3','k4','k5','k6')])
    w,h = intr['width'],intr['height']
    # Representative views across the trajectory; no resizing of chosen photos.
    indices = np.unique(np.linspace(0,len(frames)-1,min(max_views,len(frames)),dtype=int))
    columns = int(math.ceil(math.sqrt(len(indices))))
    rows = int(math.ceil(len(indices)/columns))
    vertices,faces = np.asarray(mesh.vertices),np.asarray(mesh.triangles)
    choice = np.full(len(faces),-1,np.int32)
    best = np.full(len(faces),-1.)
    pixels = np.zeros((len(faces),3,2))
    atlas_w = columns*w
    fallback_rows = math.ceil(len(faces)/atlas_w)
    atlas_h = rows*h + fallback_rows + 2
    if max(atlas_w,atlas_h) > 8192:
        raise ValueError('Full-resolution atlas exceeds 8192 pixels; reduce max_views')
    atlas = np.zeros((atlas_h,atlas_w,3),np.uint8)
    sources = []
    for slot,ordinal in enumerate(indices):
        frame = frames[ordinal]
        rgb = imread_unicode(dataset/frame['rgb'],cv2.IMREAD_COLOR)
        depth = imread_unicode(dataset/frame['depth'],cv2.IMREAD_UNCHANGED)
        if rgb.shape != (h,w,3) or depth.shape != (h,w):
            raise ValueError('Texture RGB-D does not match calibration')
        projected,score = project_visible(vertices,faces,poses[ordinal],K,coeff,depth)
        update = score > best
        best[update],choice[update],pixels[update] = score[update],slot,projected[update]
        x,y = (slot%columns)*w,(slot//columns)*h
        atlas[y:y+h,x:x+w] = rgb[:,:,::-1]
        sources.append(dict(frame_ordinal=int(ordinal),rgb=str((dataset/frame['rgb']).resolve()),
                            tile_origin_px=[x,y],rgb_sha256=hashlib.sha256((dataset/frame['rgb']).read_bytes()).hexdigest()))
    uv = np.zeros((len(faces),3,2),np.float64)
    for slot in range(len(indices)):
        mask = choice == slot
        xy = pixels[mask] + [(slot%columns)*w+.5,(slot//columns)*h+.5]
        uv[mask,:,0],uv[mask,:,1] = xy[:,:,0]/atlas_w,1-xy[:,:,1]/atlas_h
    fallback = np.flatnonzero(choice < 0)
    colors = np.asarray(mesh.vertex_colors)
    face_colors = np.rint(np.clip(colors[faces].mean(axis=1),0,1)*255).astype(np.uint8) if colors.shape == vertices.shape else np.full((len(faces),3),128,np.uint8)
    for ids in np.array_split(fallback,max(1,math.ceil(len(fallback)/100000))):
        x,y = ids%atlas_w,rows*h+1+ids//atlas_w
        atlas[y,x] = face_colors[ids]
        uv[ids,:,0] = ((x+.5)/atlas_w)[:,None]
        uv[ids,:,1] = (1-(y+.5)/atlas_h)[:,None]
    texture = output/'texture_multiview.png'
    Image.fromarray(atlas).save(texture)
    uv_path,choice_path = output/'mesh_triangle_uv.npy',output/'mesh_texture_source_slot.npy'
    np.save(uv_path,uv)
    np.save(choice_path,choice)
    # PLY retains its original topology and triangle indices for annotations.
    # GLB splits texture seams; OBJ has separate vertex and corner-UV indices.
    obj,glb = Path(exports['obj']),Path(exports['glb'])
    mtl = obj.with_suffix('.mtl')
    mtl.write_text('newmtl multiview_rgb\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nd 1\nillum 1\nmap_Kd '+texture.name+'\n',encoding='ascii')
    with obj.open('w',encoding='ascii') as f:
        f.write('mtllib '+mtl.name+'\n')
        np.savetxt(f,vertices,fmt='v %.9f %.9f %.9f')
        np.savetxt(f,uv.reshape(-1,2),fmt='vt %.9f %.9f')
        f.write('usemtl multiview_rgb\n')
        for start in range(0,len(faces),100000):
            chunk = faces[start:start+100000]
            rows_out = np.empty((len(chunk),6),np.int64)
            rows_out[:,::2] = chunk+1
            rows_out[:,1::2] = np.arange(start*3,(start+len(chunk))*3).reshape(-1,3)+1
            np.savetxt(f,rows_out,fmt='f %d/%d %d/%d %d/%d')
    material = trimesh.visual.material.PBRMaterial(baseColorTexture=Image.fromarray(atlas),metallicFactor=0,roughnessFactor=1,doubleSided=True)
    visual = trimesh.visual.texture.TextureVisuals(uv=uv.reshape(-1,2),material=material)
    tm = trimesh.Trimesh(vertices=vertices[faces].reshape(-1,3),faces=np.arange(len(faces)*3).reshape(-1,3),visual=visual,process=False)
    tm.export(glb)
    report = dict(method='per_triangle_calibrated_original_rgb_atlas',source_dataset=str(dataset),sources=sources,
                  source_resolution=[w,h],atlas_resolution=[atlas_w,atlas_h],image_resize=False,
                  textured_triangles=int((choice>=0).sum()),fallback_triangles=len(fallback),
                  textured_fraction=float((choice>=0).mean()),depth_consistency_tolerance_m=.015,
                  triangle_source_slots=str(choice_path),unit='metre',coordinate_frame='world_rdf',
                  limitations=['Per-face view switches may leave seams; no exposure correction or pose refinement.',
                               'Depth consistency checks vertices and centroid; sub-triangle occlusion may remain.',
                               'Unassigned faces retain fused color. Texture sharpness is not verified geometric accuracy.'])
    report_path = output/'multiview_texture_report.json'
    write_json(report_path, report)
    return dict(exports,texture=str(texture),triangle_uv=str(uv_path),mtl=str(mtl),texture_report=str(report_path))
