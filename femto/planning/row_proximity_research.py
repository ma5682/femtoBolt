"""Bounded sampled 3-D cross-row proximity, never a coverage certificate."""
from __future__ import annotations

import math
import numpy as np
import open3d as o3d
from femto.planning.planning_revision import digest
from femto.planning.surface_trajectory import ScanSegment

VERSION='research-sampled-cross-row-proximity-v1'
MAX_EDGES=200_000
MAX_SAMPLES=500_000
MAX_NEIGHBORS=5_000_000


def measure(trajectory,*,radii_m=(.001,.002),sample_step_m=.0005,check=lambda:None):
    """Measure weighted samples near a different row in the same work block.

    This is Euclidean proximity of discrete samples, not surface geodesic
    distance, physical treatment overlap, or a route/coverage acceptance test.
    """
    if (not isinstance(radii_m,tuple) or not 1<=len(radii_m)<=4
            or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<v<=.01 for v in radii_m)
            or tuple(sorted(set(radii_m)))!=radii_m
            or isinstance(sample_step_m,bool) or not isinstance(sample_step_m,(int,float))
            or not math.isfinite(sample_step_m) or not 0<sample_step_m<=.01):
        raise ValueError('扫描行邻近采样参数无效')
    check();before=digest([item.to_dict() for item in trajectory.ordered_items])
    xyz=[];rows=[];weights=[];edges=0
    for segment in trajectory.ordered_items:
        if not isinstance(segment,ScanSegment):continue
        if type(segment.block_index) is not int:raise ValueError('扫描工作块索引无效')
        for left,right in zip(segment.points,segment.points[1:]):
            edges+=1
            if edges%256==0:check()
            if edges>MAX_EDGES:raise ValueError('扫描行邻近边数超过预算')
            role=right.get('role')
            if role not in (None,'scan','connector'):raise ValueError('扫描行角色无效')
            if role=='connector':continue
            if type(left.get('row_index')) is not int or type(right.get('row_index')) is not int:
                raise ValueError('扫描行索引无效')
            if left['row_index']!=right['row_index']:continue
            a,b=np.asarray(left.get('xyz_m'),float),np.asarray(right.get('xyz_m'),float)
            if a.shape!=(3,) or b.shape!=(3,) or not np.isfinite([a,b]).all():raise ValueError('扫描行坐标无效')
            length=float(np.linalg.norm(b-a))
            if not math.isfinite(length):raise ValueError('扫描行长度无效')
            if not length:continue
            count=max(1,int(math.ceil(length/sample_step_m)))
            if len(xyz)+count>MAX_SAMPLES:raise ValueError('扫描行邻近采样超过预算')
            t=(np.arange(count)+.5)/count
            xyz.extend(a+(b-a)*t[:,None]);rows.extend([(segment.block_index,right['row_index'])]*count)
            weights.extend([length/count]*count)
    values=[];neighbor_count=0
    if xyz:
        xyz=np.asarray(xyz);rows=np.asarray(rows);weights=np.asarray(weights)
        cloud=o3d.geometry.PointCloud();cloud.points=o3d.utility.Vector3dVector(xyz)
        tree=o3d.geometry.KDTreeFlann(cloud);total=float(weights.sum())
        for radius in radii_m:
            close=np.zeros(len(xyz),bool)
            for index,point in enumerate(xyz):
                if index%128==0:check()
                _,found,_=tree.search_radius_vector_3d(point,radius)
                neighbor_count+=len(found)
                if neighbor_count>MAX_NEIGHBORS:raise ValueError('扫描行邻近搜索超过预算')
                close[index]=any(rows[j,0]==rows[index,0] and rows[j,1]!=rows[index,1] for j in found)
            near=float(weights[close].sum())
            values.append(dict(radius_m=float(radius),near_sample_m=near,near_fraction=near/total))
    else:
        total=0.
        values=[dict(radius_m=float(radius),near_sample_m=0.,near_fraction=None) for radius in radii_m]
    if digest([item.to_dict() for item in trajectory.ordered_items])!=before:
        raise ValueError('扫描行邻近采样期间路线发生变化')
    check()
    return dict(version=VERSION,route_items_sha256=before,sample_step_m=float(sample_step_m),
        sampled_points=len(xyz),sampled_scan_length_m=total,proximity=values,
        semantics='sampled_euclidean_proximity_only_not_geodesic_coverage_or_physical_motion')
