"""Propose bounded trimming of near-row scan ends, with proof-seed protection.

The proximity score is a sampled 3-D heuristic. Callers must independently
validate the route, target and previous proof before calling, then reconnect,
recheck geometry and reprove the entire resulting route. No coverage or robot
motion claim is made here.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import replace
import math
import numpy as np
import open3d as o3d
from femto.planning import coverage_certificate as coverage,scan_fraction_seeds
from femto.planning.coverage_domain import scan_endpoints
from femto.planning.planning_revision import array_record,digest
from femto.planning.surface_trajectory import ScanSegment

VERSION='research-proof-seed-protected-terminal-trim-v1'
MAX_EDGES=200_000
MAX_SAMPLES=500_000
MAX_SEGMENTS=5000
MAX_TRIMS=16


def _key(block,row,a,b):return (int(block),int(row),*tuple(sorted((tuple(a['xyz_m']),tuple(b['xyz_m'])))))


def propose(trajectory,base_segments,original_faces,requests,certificate,*,check=lambda:None,
            radius_m=.001,sample_step_m=.0005,max_edge_m=.0025,min_near_m=.0002,max_trims=MAX_TRIMS):
    """Return detached base segments and a deterministic terminal-edge proposal."""
    check();coverage._checked_assets(certificate,check)
    values=(radius_m,sample_step_m,max_edge_m,min_near_m)
    if (not isinstance(base_segments,list) or len(base_segments)>MAX_SEGMENTS
            or any(not isinstance(s,ScanSegment) for s in base_segments)
            or type(max_trims) is not int or not 0<=max_trims<=MAX_TRIMS
            or any(isinstance(v,bool) or not isinstance(v,(float,int)) or not math.isfinite(v) or v<=0 for v in values)
            or radius_m>.01 or sample_step_m>.01 or max_edge_m>.01):
        raise ValueError('扫描端部筛选参数无效或超过预算')
    base_snapshot=digest([s.to_dict() for s in base_segments]);request_snapshot=digest(requests)
    faces_snapshot=array_record(original_faces)
    raw=[item.to_dict() for item in trajectory.ordered_items]
    binding=certificate.metadata['inputs']['route']
    if binding.get('schema')!='research-checked-supplement-route-v1' or binding.get('items_sha256')!=digest(raw):
        raise ValueError('扫描端部筛选证据与路线不一致')
    positions,_=scan_fraction_seeds.propose(trajectory,original_faces,requests,check=check)
    seeds=scan_endpoints(trajectory,original_faces)+positions
    if digest(seeds)!=certificate.metadata['inputs']['seeds_sha256']:raise ValueError('扫描端部筛选种子与证据不一致')
    segments={}
    for item in trajectory.ordered_items:
        if not isinstance(item,ScanSegment):continue
        if type(item.segment_index) is not int or item.segment_index in segments:raise ValueError('扫描段索引无效或重复')
        segments[item.segment_index]=item
    arrays=certificate.arrays;proof_seeds=arrays['proof_seed']
    if len(proof_seeds) and (proof_seeds.min()<0 or proof_seeds.max()>=len(arrays['seed_input_index'])):
        raise ValueError('证明种子索引无效')
    protected=set()
    for count,index in enumerate(np.unique(proof_seeds)):
        if count%256==0:check()
        source=int(arrays['seed_input_index'][index])
        if not 0<=source<len(seeds):raise ValueError('证明种子输入索引无效')
        record=seeds[source];segment=segments.get(record['segment_index']);edge=record['edge_index']
        if segment is None or not 0<=edge<len(segment.points)-1:raise ValueError('证明种子扫描边无效')
        a,b=segment.points[edge:edge+2]
        if b.get('role')=='connector' or a['xyz_m']==b['xyz_m']:raise ValueError('证明种子不在活动扫描边')
        protected.add(_key(segment.block_index,b['row_index'],a,b))
    xyz=[];rows=[];weights=[];keys=[];examined=0
    for segment in segments.values():
        for a,b in zip(segment.points,segment.points[1:]):
            examined+=1
            if examined%256==0:check()
            if examined>MAX_EDGES:raise ValueError('扫描端部筛选路线边超过预算')
            role=b.get('role')
            if role not in (None,'scan','connector'):raise ValueError('扫描边角色无效')
            if role=='connector' or a['row_index']!=b['row_index']:continue
            left,right=np.asarray(a['xyz_m'],float),np.asarray(b['xyz_m'],float)
            if left.shape!=(3,) or right.shape!=(3,) or not np.isfinite([left,right]).all():raise ValueError('扫描坐标无效')
            length=float(np.linalg.norm(right-left))
            if not length:continue
            count=max(1,int(math.ceil(length/sample_step_m)))
            if len(xyz)+count>MAX_SAMPLES:raise ValueError('扫描端部筛选采样超过预算')
            t=(np.arange(count)+.5)/count;xyz.extend(left+(right-left)*t[:,None])
            rows.extend([(segment.block_index,b['row_index'])]*count)
            weights.extend([length/count]*count);keys.extend([_key(segment.block_index,b['row_index'],a,b)]*count)
    near=defaultdict(float)
    if xyz:
        xyz=np.asarray(xyz);rows=np.asarray(rows);cloud=o3d.geometry.PointCloud();cloud.points=o3d.utility.Vector3dVector(xyz)
        tree=o3d.geometry.KDTreeFlann(cloud)
        for i,point in enumerate(xyz):
            if i%256==0:check()
            _,found,_=tree.search_radius_vector_3d(point,radius_m)
            if any(rows[j,0]==rows[i,0] and rows[j,1]!=rows[i,1] for j in found):near[keys[i]]+=weights[i]
    candidates=[]
    for index,segment in enumerate(base_segments):
        check()
        if len(segment.points)<3:continue
        for side,edge in (('start',0),('end',len(segment.points)-2)):
            a,b=segment.points[edge:edge+2]
            if b.get('role')=='connector' or a['row_index']!=b['row_index']:continue
            marker=_key(segment.block_index,b['row_index'],a,b);length=float(np.linalg.norm(np.asarray(a['xyz_m'])-b['xyz_m']))
            if marker in protected or not 0<length<=max_edge_m or near.get(marker,0.)<min_near_m:continue
            candidates.append(dict(base_segment=index,side=side,row=b['row_index'],length_m=length,near_sample_m=float(near[marker])))
    selected=[];by_segment={}
    for item in sorted(candidates,key=lambda c:(-c['near_sample_m'],c['base_segment'],c['side'])):
        if len(selected)>=max_trims:break
        index=item['base_segment'];sides=by_segment.setdefault(index,set())
        if len(base_segments[index].points)-len(sides)-1<2:continue
        sides.add(item['side']);selected.append(item)
    result=[]
    for index,segment in enumerate(base_segments):
        sides=by_segment.get(index,set());left=int('start' in sides);right=int('end' in sides)
        points=deepcopy(segment.points[left:len(segment.points)-right if right else None])
        result.append(replace(segment,points=points))
    report=dict(version=VERSION,selected=selected,protected_proof_seeds=len(np.unique(proof_seeds)),
        protected_edge_keys=len(protected),sampled_points=len(xyz),route_items_sha256=binding['items_sha256'],
        semantics='proposal_only_requires_route_geometry_and_new_coverage_replay')
    if (digest([s.to_dict() for s in base_segments])!=base_snapshot or digest(requests)!=request_snapshot
            or array_record(original_faces)!=faces_snapshot
            or digest([item.to_dict() for item in trajectory.ordered_items])!=binding['items_sha256']):
        raise ValueError('扫描端部筛选期间输入发生变化')
    coverage._checked_assets(certificate,check)
    check();return result,report
