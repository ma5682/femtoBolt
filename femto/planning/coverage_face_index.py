"""Bounded face broad phase for nominal-coverage search ordering.

The candidate provider must be a conservative spatial index: every point in
the expanded face box must be returned. This only selects work; it never proves
containment, connectivity, exclusion or physical coverage.
"""
from __future__ import annotations

import math
import numpy as np

MAX_SEEDS=20_000


def relevant_faces(vertices, faces, face_ids, points, radius, candidates_for, *,
                   charge=lambda count:None, check=lambda:None):
    """Return sorted face IDs whose outward expanded AABB contains a seed."""
    vertices=np.asarray(vertices);faces=np.asarray(faces);face_ids=np.asarray(face_ids);points=np.asarray(points)
    if (vertices.ndim!=2 or vertices.shape[1:]!=(3,) or not np.issubdtype(vertices.dtype,np.floating)
            or not np.isfinite(vertices).all() or faces.ndim!=2 or faces.shape[1:]!=(3,)
            or not np.issubdtype(faces.dtype,np.integer) or (faces.size and (faces.min()<0 or faces.max()>=len(vertices)))
            or face_ids.ndim!=1 or not np.issubdtype(face_ids.dtype,np.integer)
            or (face_ids.size and (face_ids.min()<0 or face_ids.max()>=len(faces)))
            or points.ndim!=2 or points.shape[1:]!=(3,) or len(points)>MAX_SEEDS
            or not np.issubdtype(points.dtype,np.floating) or not np.isfinite(points).all()
            or isinstance(radius,bool) or not isinstance(radius,(float,int)) or not math.isfinite(radius) or radius<=0
            or not callable(candidates_for) or not callable(charge) or not callable(check)):
        raise ValueError('覆盖空间候选索引输入无效')
    result=[]
    for face in np.unique(face_ids):
        check();face=int(face)
        raw=candidates_for(face)
        if not isinstance(raw,(tuple,list,np.ndarray)) or len(raw)>MAX_SEEDS:
            raise ValueError('覆盖空间候选点索引无效')
        candidates=np.asarray(raw)
        if candidates.ndim!=1 or (len(candidates) and (not np.issubdtype(candidates.dtype,np.integer)
                or candidates.min()<0 or candidates.max()>=len(points))):
            raise ValueError('覆盖空间候选点索引无效')
        charge(len(candidates))
        if not len(candidates):continue
        triangle=vertices[faces[face]]
        low=np.nextafter(triangle.min(axis=0)-radius,-np.inf)
        high=np.nextafter(triangle.max(axis=0)+radius,np.inf)
        if np.any(np.all((points[candidates]>=low)&(points[candidates]<=high),axis=1)):
            result.append(face)
    return np.asarray(result,dtype='<i8')
