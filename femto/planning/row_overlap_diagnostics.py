"""Attribute sampled cross-row proximity to rows and their nearest peers.

Diagnostic only: Euclidean proximity is not surface distance, coverage,
evidence that lines can be removed, or a physical treatment assessment.
"""
from __future__ import annotations

from collections import defaultdict
import math
import numpy as np
import open3d as o3d

from femto.planning import row_proximity_research as common
from femto.planning.planning_revision import digest
from femto.planning.surface_trajectory import ScanSegment

VERSION='research-cross-row-overlap-attribution-v1'


def inspect(trajectory,*,supplement_rows=(),radius_m=.001,sample_step_m=.0005,include_edges=False,check=lambda:None):
    """Attribute each nearby sample once, using the nearest different row.

    Only comparisons within the same work block are considered. The optional
    row IDs identify supplements for diagnostics; no route is modified.
    """
    if (type(include_edges) is not bool or not isinstance(supplement_rows,(tuple,list)) or len(supplement_rows)>common.MAX_EDGES
            or any(type(value) is not int or value<0 for value in supplement_rows)
            or len(set(supplement_rows))!=len(supplement_rows)
            or isinstance(radius_m,bool) or not isinstance(radius_m,(int,float)) or not math.isfinite(radius_m) or not 0<radius_m<=.01
            or isinstance(sample_step_m,bool) or not isinstance(sample_step_m,(int,float)) or not math.isfinite(sample_step_m)
            or not 0<sample_step_m<=.01):raise ValueError('扫描行邻近归因参数无效')
    check();before=digest([item.to_dict() for item in trajectory.ordered_items]);supplements=set(supplement_rows)
    xyz=[];keys=[];weights=[];edge_keys=[];edge_count=0
    for segment in trajectory.ordered_items:
        if not isinstance(segment,ScanSegment):continue
        if type(segment.block_index) is not int:raise ValueError('扫描工作块索引无效')
        for left,right in zip(segment.points,segment.points[1:]):
            edge_count+=1
            if edge_count%256==0:check()
            if edge_count>common.MAX_EDGES:raise ValueError('扫描行邻近边数超过预算')
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
            if len(xyz)+count>common.MAX_SAMPLES:raise ValueError('扫描行邻近采样超过预算')
            t=(np.arange(count)+.5)/count;xyz.extend(a+(b-a)*t[:,None])
            keys.extend([(segment.block_index,right['row_index'])]*count);weights.extend([length/count]*count)
            if include_edges:
                edge_keys.extend([(segment.block_index,right['row_index'],*tuple(sorted((tuple(a),tuple(b)))))]*count)
    row_total=defaultdict(float);row_near=defaultdict(float);pairs=defaultdict(float);groups=defaultdict(float);edge_near=defaultdict(float)
    count_neighbors=0
    if xyz:
        cloud=o3d.geometry.PointCloud();cloud.points=o3d.utility.Vector3dVector(np.asarray(xyz));tree=o3d.geometry.KDTreeFlann(cloud)
        for index,(point,key,weight) in enumerate(zip(xyz,keys,weights)):
            if index%128==0:check()
            row_total[key]+=weight
            _,found,distances=tree.search_radius_vector_3d(point,radius_m);count_neighbors+=len(found)
            if count_neighbors>common.MAX_NEIGHBORS:raise ValueError('扫描行邻近搜索超过预算')
            peers=[(float(distance),keys[j],j) for j,distance in zip(found,distances)
                   if keys[j][0]==key[0] and keys[j][1]!=key[1]]
            if not peers:continue
            other=min(peers)[1];row_near[key]+=weight;pairs[tuple(sorted((key,other)))]+=weight
            if include_edges:edge_near[edge_keys[index]]+=weight
            category=('both_supplement' if key[1] in supplements and other[1] in supplements else
                      'one_supplement' if (key[1] in supplements)!=(other[1] in supplements) else 'neither_supplement')
            groups[category]+=weight
    if digest([item.to_dict() for item in trajectory.ordered_items])!=before:
        raise ValueError('扫描行邻近归因期间路线发生变化')
    check();total=float(sum(row_total.values()));near=float(sum(row_near.values()))
    result=dict(version=VERSION,route_items_sha256=before,radius_m=float(radius_m),sample_step_m=float(sample_step_m),
        sampled_points=len(xyz),sampled_scan_length_m=total,near_sample_m=near,near_fraction=near/total if total else None,
        groups={name:float(groups[name]) for name in ('neither_supplement','one_supplement','both_supplement')},
        rows=[dict(block=key[0],row=key[1],supplement=key[1] in supplements,total_m=float(row_total[key]),near_m=float(value))
              for key,value in sorted(row_near.items(),key=lambda pair:(-pair[1],pair[0]))],
        pairs=[dict(block=left[0],row_a=left[1],row_b=right[1],near_m=float(value))
               for (left,right),value in sorted(pairs.items(),key=lambda pair:(-pair[1],pair[0]))],
        semantics='nearest_different_row_attribution_only_not_surface_distance_or_coverage')
    if include_edges:
        result['edges']=[dict(block=key[0],row=key[1],endpoints=[[float(v) for v in key[2]],[float(v) for v in key[3]]],near_m=float(value))
                         for key,value in sorted(edge_near.items(),key=lambda pair:(-pair[1],pair[0]))]
    return result
