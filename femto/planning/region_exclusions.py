"""Reference-image exclusion polygons and conservative whole-face removal."""
import cv2
import numpy as np


def validate_exclusions(polygons):
    if not isinstance(polygons,list) or len(polygons)>32:raise ValueError('最多支持32个禁止区域')
    for polygon in polygons:
        p=np.asarray(polygon,float)
        if p.ndim!=2 or p.shape[1]!=2 or not 3<=len(p)<=512 or not np.isfinite(p).all():
            raise ValueError('禁止区域轮廓无效')
        if np.max(np.abs(p))>100000 or abs(cv2.contourArea(p.astype(np.float32)))<1:
            raise ValueError('禁止区域面积过小或坐标无效')
    return polygons


def allowed_region(projection,mesh,region,check=lambda:None):
    polygons=validate_exclusions(region.get('exclusion_polygons',[]))
    if not polygons:return region,[]
    mask=np.zeros((projection.height,projection.width),np.uint8)
    for polygon in polygons:cv2.fillPoly(mask,[np.rint(polygon).astype(np.int32)],1)
    mask=cv2.dilate(mask,np.ones((3,3),np.uint8))
    vertices=np.asarray(mesh.vertices);faces=np.asarray(mesh.triangles)
    kept=[];removed=[]
    for i,tid in enumerate(region['triangle_indices']):
        if i%256==0:check()
        uv=projection.project(vertices[faces[tid]])
        if not np.isfinite(uv).all():removed.append(tid);continue
        low=np.maximum(0,np.floor(uv.min(axis=0)).astype(int));high=np.minimum([projection.width-1,projection.height-1],np.ceil(uv.max(axis=0)).astype(int))
        if np.any(high<low):removed.append(tid);continue
        if np.prod(high-low+1)>20_000_000:raise ValueError('Forbidden-region projection budget exceeded')
        tile=np.zeros((high[1]-low[1]+1,high[0]-low[0]+1),np.uint8)
        cv2.fillConvexPoly(tile,np.rint(uv-low).astype(np.int32),1)
        blocked=np.any(tile & mask[low[1]:high[1]+1,low[0]:high[0]+1])
        (removed if blocked else kept).append(tid)
    if not kept:raise ValueError('禁止区域已覆盖全部有效表面，无法生成路线。')
    return dict(region,triangle_indices=kept),removed
