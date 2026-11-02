"""Research-only float proximity screening for optional midpoint proposals.

The result is a priority hint, not a coverage decision. Omitting a record can
lose useful evidence; inclusion never confirms a cell, path, or treatment.
Callers must independently bind, construct and replay any claimed proof.
"""
from __future__ import annotations

import math
import numpy as np

VERSION='midpoint-pending-centroid-priority-v1'
MAX_COMPARISONS=10_000_000
MAX_POINTS=20_000
MAX_CENTERS=200_000
POINT_BATCH=256
CENTER_BATCH=64


def select(records,centers,radius_m,*,max_comparisons=MAX_COMPARISONS,check=lambda:None):
    """Return stable midpoint indices near supplied approximate centroids.

    No midpoint or original target is modified. Float comparisons are used
    only to order or omit experimental candidates; they are not predicates for
    the coverage engine. The total pair budget and temporary arrays are bounded.
    """
    if (not isinstance(records,list) or len(records)>MAX_POINTS
            or isinstance(radius_m,bool) or not isinstance(radius_m,(int,float))
            or not math.isfinite(radius_m) or radius_m<=0 or not math.isfinite(radius_m*radius_m)
            or type(max_comparisons) is not int or not 0<=max_comparisons<=MAX_COMPARISONS):
        raise ValueError('中点优先筛选参数无效')
    centers=np.asarray(centers,dtype=float)
    if centers.ndim!=2 or centers.shape[1:]!=(3,) or len(centers)>MAX_CENTERS or not np.isfinite(centers).all():
        raise ValueError('待确认中心坐标无效')
    if len(records)*len(centers)>max_comparisons:raise ValueError('中点优先筛选超过比较预算')
    if any(not isinstance(record,dict) or record.get('role')!='scan' for record in records):
        raise ValueError('中点优先筛选记录无效')
    points=np.asarray([p.get('xyz_m') for p in records],dtype=float).reshape(-1,3)
    if len(points)!=len(records) or not np.isfinite(points).all():raise ValueError('中点候选坐标无效')
    selected=np.zeros(len(points),dtype=bool);radius2=radius_m*radius_m
    for first in range(0,len(points),POINT_BATCH):
        check();group=points[first:first+POINT_BATCH];found=np.zeros(len(group),bool)
        for start in range(0,len(centers),CENTER_BATCH):
            check()
            with np.errstate(over='ignore',invalid='ignore'):
                delta=group[:,None,:]-centers[None,start:start+CENTER_BATCH,:]
                distances=np.einsum('ijk,ijk->ij',delta,delta)
            if not np.isfinite(distances).all():raise ValueError('中点优先筛选距离溢出')
            found|=(distances<=radius2).any(axis=1)
        selected[first:first+len(group)]=found
    check();indices=np.flatnonzero(selected).astype('<i8')
    return indices,dict(version=VERSION,radius_m=float(radius_m),midpoint_count=len(records),
        centroid_count=len(centers),comparison_count=len(records)*len(centers),selected_count=len(indices),
        semantics='float_priority_only_not_coverage_or_physical_distance')
