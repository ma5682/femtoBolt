"""Bounded research proposal to retain old scan-edge witnesses.

The caller must independently replay the old proof and validate its original
route and target before calling. Returned edges are candidates, not a route,
proof of current coverage, physical coverage, or a robot trajectory.
"""
from __future__ import annotations

import math
import numpy as np
from femto.planning import coverage_certificate as coverage
from femto.planning.coverage_domain import scan_endpoints
from femto.planning.planning_revision import digest
from femto.planning.surface_trajectory import ScanSegment

VERSION='research-existing-witness-scan-supplement-v1'
MAX_PENDING=5000
MAX_PROOFS=2_000_000
MAX_SEEDS=100_000
MAX_EDGE_VISITS=200_000


def propose(trajectory, region_faces, certificate, root_area_m2, pending_leaves, *,
            representative_faces=None, gap_m=.008, check=lambda:None):
    """Select old scan edges with witnesses for the requested old proof leaves.

    The weighted greedy set cover is deterministic but not globally optimal.
    Joining a short gap is restricted to the same old scanning row. Neither
    selection nor coalescing bypasses the caller's final whole-edge checks.
    """
    check();coverage._checked_assets(certificate,check)
    if (not isinstance(pending_leaves,(list,tuple,np.ndarray)) or len(pending_leaves)>MAX_PENDING
            or isinstance(gap_m,bool) or not isinstance(gap_m,(int,float))
            or not math.isfinite(gap_m) or not 0<=gap_m<=.1):
        raise ValueError('补线候选参数无效或超过预算')
    pending=[]
    for leaf in pending_leaves:
        if isinstance(leaf,(bool,np.bool_)) or not isinstance(leaf,(int,np.integer)):
            raise ValueError('待确认旧单元索引无效')
        pending.append(int(leaf))
    if len(set(pending))!=len(pending):raise ValueError('待确认旧单元索引重复')
    pending.sort();arrays=certificate.arrays
    if (any(i<0 or i>=len(arrays['root']) or arrays['status'][i]!=1 for i in pending)
            or len(arrays['proof_leaf'])>MAX_PROOFS or len(arrays['seed_face'])>MAX_SEEDS):
        raise ValueError('待确认旧单元或证据超过预算')
    areas=np.asarray(root_area_m2)
    roots=arrays['root'][pending];depth=arrays['depth'][pending]
    if (areas.ndim!=1 or areas.dtype.kind not in 'fiu' or not np.isfinite(areas).all()
            or (len(roots) and (roots.min()<0 or roots.max()>=len(areas)))
            or np.any(areas<0) or np.any(depth>30)):
        raise ValueError('旧目标名义面积无效')
    weights={leaf:float(areas[root])*math.ldexp(1.,-2*int(d)) for leaf,root,d in zip(pending,roots,depth)}
    if any(not math.isfinite(w) or w<=0 for w in weights.values()):raise ValueError('旧目标名义权重无效')
    # The representative map, if needed for coincident mesh faces, must be
    # taken from the same independently validated old target domain.
    if representative_faces is not None:
        representatives=np.asarray(representative_faces)
        if representatives.ndim!=1 or representatives.dtype.kind not in 'iu':raise ValueError('原面代表映射无效')
    else:representatives=None
    segments={}
    examined=0
    for segment in trajectory.ordered_items:
        if not isinstance(segment,ScanSegment):continue
        if type(segment.segment_index) is not int or segment.segment_index in segments:
            raise ValueError('旧路线扫描段索引无效或重复')
        segments[segment.segment_index]=segment
        examined+=max(0,len(segment.points)-1)
        check()
        if examined>MAX_EDGE_VISITS:raise ValueError('旧路线扫描边超过预算')
    records=scan_endpoints(trajectory,region_faces)
    check()
    if digest(records)!=certificate.metadata['inputs']['seeds_sha256']:
        raise ValueError('旧路线扫描点与证据不一致')
    pending_set=set(pending);by_seed={};by_leaf={leaf:set() for leaf in pending}
    for i,(leaf,seed) in enumerate(zip(arrays['proof_leaf'],arrays['proof_seed'])):
        if i%1024==0:check()
        leaf=int(leaf);seed=int(seed)
        if leaf not in pending_set:continue
        if not 0<=seed<len(arrays['seed_face']):raise ValueError('旧证据见证索引无效')
        source=int(arrays['seed_input_index'][seed])
        if not 0<=source<len(records):raise ValueError('旧证据扫描点索引无效')
        record=records[source]
        face=record['source_face']
        if representatives is not None:
            if not 0<=face<len(representatives):raise ValueError('扫描点原面代表映射无效')
            face=int(representatives[face])
        if (int(arrays['seed_face'][seed])!=face
                or not np.array_equal(arrays['seed_xyz'][seed],record['xyz_m'])):
            raise ValueError('旧证据见证与扫描点不一致')
        by_seed.setdefault(seed,set()).add(leaf);by_leaf[leaf].add(seed)
    gains={seed:sum(weights[leaf] for leaf in sorted(leaves)) for seed,leaves in by_seed.items()}
    remaining=set(pending);chosen=[]
    while remaining:
        check()
        if not gains:raise ValueError('待确认单元缺少原扫描点见证')
        seed=max(gains,key=lambda value:(gains[value],-value))
        if gains[seed]<=0:raise ValueError('待确认单元缺少有效原扫描点见证')
        chosen.append(seed)
        for leaf in sorted(by_seed[seed]&remaining):
            remaining.remove(leaf)
            for other in by_leaf[leaf]:gains[other]=max(0.,gains[other]-weights[leaf])
        gains[seed]=0.
    selected=set()
    for seed in chosen:
        record=records[int(arrays['seed_input_index'][seed])]
        selected.add((record['segment_index'],record['edge_index']))
    by_segment={}
    for segment,edge in selected:by_segment.setdefault(segment,[]).append(edge)
    for segment,indices in sorted(by_segment.items()):
        check();points=segments[segment].points
        for a,b in zip(sorted(indices),sorted(indices)[1:]):
            between=range(a+1,b)
            if any(points[e+1].get('role','scan')!='scan' or points[e+1]['row_index']!=points[a+1]['row_index'] for e in between):continue
            length=sum(float(np.linalg.norm(np.asarray(points[e]['xyz_m'])-points[e+1]['xyz_m'])) for e in between)
            if not math.isfinite(length):raise ValueError('旧扫描边长度无效')
            if length<=gap_m:selected.update((segment,e) for e in between)
    edges=sorted(selected)
    length=sum(float(np.linalg.norm(np.asarray(segments[s].points[e]['xyz_m'])-segments[s].points[e+1]['xyz_m'])) for s,e in edges)
    if not math.isfinite(length):raise ValueError('旧扫描边长度无效')
    groups=sum((s,e-1) not in selected or segments[s].points[e]['row_index']!=segments[s].points[e+1]['row_index'] for s,e in edges)
    check()
    return edges,dict(version=VERSION,pending_leaves=len(pending),selected_seeds=len(chosen),
        selected_edges=len(edges),scan_length_m=length,contiguous_groups=groups,
        semantics='candidate_only_not_coverage_or_physical_motion')
