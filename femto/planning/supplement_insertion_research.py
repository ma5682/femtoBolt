"""Bounded insertion of original-edge supplements into a validated scan order.

The ordering scores are heuristics. Callers must check input and final edges
against the current target, preserve provenance, and independently reprove
coverage before considering any route for use or durable publication.
"""
from __future__ import annotations

from copy import deepcopy
import math,random
import numpy as np
import open3d as o3d
from femto.planning import scan_visit_order
from femto.planning.endpoint_preserving_detour import _signature

VERSION='research-verified-graph-supplement-insertion-v1'
MAX_SEGMENTS=300
MAX_QUERY=20_000
MAX_SCANNED_POINTS=200_000
MAX_ATTEMPTS=64
MAX_RELOCATIONS=3


def propose(graph,segments,base,extra,base_visits,spacing_m,*,check=lambda:None,budget=200_000,
            attempts=24,penalty_m=.005,revisit_weight=1.,seed=0):
    """Propose an insertion using checked graph paths, never a coverage claim."""
    if (not isinstance(segments,list) or not isinstance(base,list) or not isinstance(extra,list)
            or not base or len(segments)>MAX_SEGMENTS or len(base)+len(extra)!=len(segments)
            or type(attempts) is not int or not 1<=attempts<=MAX_ATTEMPTS
            or type(budget) is not int or budget<1 or type(seed) is not int or not 0<=seed<=2**32-1
            or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in (spacing_m,penalty_m,revisit_weight))
            or spacing_m<=0 or not 0<=penalty_m<=.1 or not 0<=revisit_weight<=10):
        raise ValueError('补线插入参数无效或超过预算')
    identities=[id(s) for s in segments]
    if len(set(identities))!=len(identities) or set(identities)!={id(s) for s in base+extra}:
        raise ValueError('补线与原扫描段清单不一致')
    scan_visit_order.checked_visits(base,base_visits)
    indices={id(s):i for i,s in enumerate(segments)}
    base_sequence=[(indices[id(base[i])],reverse) for i,reverse in base_visits]
    blocks={block:[v for v in base_sequence if segments[v[0]].block_index==block]
            for block in dict.fromkeys(s.block_index for s in segments)}
    if any(not seq for seq in blocks.values()):raise ValueError('补线不能引入新的工作块')
    scans=set();seen=0
    for segment in segments:
        check()
        for a,b in zip(segment.points,segment.points[1:]):
            seen+=1
            if seen%256==0:check()
            if seen>MAX_SCANNED_POINTS:raise ValueError('补线插入扫描边超过预算')
            if b.get('role')=='connector' or a['xyz_m']==b['xyz_m']:continue
            for point in (a,b):scans.add((*point['xyz_m'],b['row_index']))
    if not scans:raise ValueError('补线插入没有可用扫描边')
    keys=sorted(scans);xyz=np.asarray([p[:3] for p in keys],float);rows=np.asarray([p[3] for p in keys],int)
    if not np.isfinite(xyz).all():raise ValueError('补线插入坐标无效')
    tree=o3d.geometry.KDTreeFlann(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz)))
    radius=spacing_m/2;cached={};queries=0;relocations=0
    def point(v,tail):
        segment=segments[v[0]];index=(1-int(v[1])) if tail else int(v[1])
        return np.asarray(segment.points[-1 if index else 0]['xyz_m'])
    def row(v,tail):
        segment=segments[v[0]];index=(1-int(v[1])) if tail else int(v[1])
        return segment.points[-1 if index else 0]['row_index']
    def detour_delta(left,candidate,right):
        value=int(row(left,True)==row(candidate,False))
        if right is not None:value+=int(row(candidate,True)==row(right,False))-int(row(left,True)==row(right,False))
        return value
    def route_cost(a,b):
        nonlocal queries
        check();key=(a,b)
        if key not in cached:
            queries+=1
            if queries>MAX_QUERY:raise ValueError('补线连接搜索超过预算')
            samples=graph.route(a[0],b[0],left_reverse=a[1],right_reverse=b[1])
            if not samples:cached[key]=(float('inf'),float('inf'))
            else:
                pts=np.asarray([point(a,True)]+[p['xyz_m'] for p in samples]);lengths=np.linalg.norm(np.diff(pts,axis=0),axis=1)
                joined={row(a,True),row(b,False)};revisit=0.
                for i,p in enumerate(pts[1:]):
                    if i%256==0:check()
                    _,found,_=tree.search_radius_vector_3d(p,radius)
                    if any(int(rows[x]) not in joined for x in found):revisit+=float(lengths[i])
                cached[key]=(float(lengths.sum()),revisit)
        return cached[key]
    def rank_positions(sequence,index):
        ranked=[]
        for pos in range(1,len(sequence)+1):
            left=sequence[pos-1];right=sequence[pos] if pos<len(sequence) else None
            for reverse in (False,True):
                v=(index,reverse);estimate=float(np.linalg.norm(point(left,True)-point(v,False)))
                if right is not None:estimate+=float(np.linalg.norm(point(v,True)-point(right,False))-np.linalg.norm(point(left,True)-point(right,False)))
                ranked.append((estimate+penalty_m*detour_delta(left,v,right),pos,reverse))
        return sorted(ranked)[:attempts]
    chosen=list(extra);random.Random(seed).shuffle(chosen)
    for segment in chosen:
        check();seq=blocks[segment.block_index];index=indices[id(segment)];best=None
        for _,pos,reverse in rank_positions(seq,index):
            left=seq[pos-1];right=seq[pos] if pos<len(seq) else None;v=(index,reverse)
            one,one_revisit=route_cost(left,v)
            if not np.isfinite(one):continue
            delta=one;revisit=one_revisit
            if right is not None:
                old,old_revisit=route_cost(left,right);two,two_revisit=route_cost(v,right)
                if not np.isfinite(old) or not np.isfinite(two):continue
                delta+=two-old;revisit+=two_revisit-old_revisit
            rank=(delta+penalty_m*detour_delta(left,v,right)+revisit_weight*revisit,pos,reverse)
            if best is None or rank<best:best=rank
        if best is None:raise ValueError('补线不能通过已验证图连接')
        seq.insert(best[1],(index,best[2]))
    def metric(a,b):
        if b is None:return (0.,0.,0)
        length,revisit=route_cost(a,b)
        return length,revisit,int(row(a,True)==row(b,False))
    def combine(*terms):return tuple(sum(sign*values[i] for sign,values in terms) for i in range(3))
    for _ in range(MAX_RELOCATIONS):
        changed=False
        for segment in extra:
            check();seq=blocks[segment.block_index];index=indices[id(segment)]
            loc=next(i for i,v in enumerate(seq) if v[0]==index)
            if loc==0:continue
            left=seq[loc-1];moved=seq[loc];right=seq[loc+1] if loc+1<len(seq) else None
            removal=combine((1,metric(left,right)),(-1,metric(left,moved)),(-1,metric(moved,right)))
            if not np.all(np.isfinite(removal)):continue
            reduced=seq[:loc]+seq[loc+1:];best=None
            for _,pos,reverse in rank_positions(reduced,index):
                a=reduced[pos-1];b=reduced[pos] if pos<len(reduced) else None;v=(index,reverse)
                delta=combine((1,removal),(1,metric(a,v)),(1,metric(v,b)),(-1,metric(a,b)))
                if not np.all(np.isfinite(delta)) or delta[2]>0:continue
                score=delta[0]+revisit_weight*delta[1]+penalty_m*delta[2];rank=(score,delta[0],pos,reverse)
                if score < -1e-10 and (best is None or rank<best):best=rank
            if best is not None:
                reduced.insert(best[2],(index,best[3]));blocks[segment.block_index]=reduced;relocations+=1;changed=True
        if not changed:break
    visits=[v for seq in blocks.values() for v in seq]
    links=scan_visit_order.connect(graph,segments,visits,check,budget)
    chains,added,objective=scan_visit_order.reconstruct(segments,visits,links,check,preserve_endpoint_faces=True,budget=budget)
    if objective[0]:raise ValueError('补线产生未连接的工作块内扫描段')
    if _signature(chains,check)!=_signature([(s.block_index,s.points) for s in segments],check):
        raise ValueError('补线插入改变原扫描边或来源面')
    report=dict(version=VERSION,visits=[list(v) for v in visits],path_queries=queries,relocations=relocations,
        selected_objective=list(objective),seed=seed,semantics='candidate_requires_final_geometry_and_coverage_checks')
    return chains,deepcopy(links),added,report
