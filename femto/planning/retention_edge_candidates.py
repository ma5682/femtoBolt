"""Bounded proposals for retaining old confirmed leaves with old scan edges.

Only call with replayed old proofs, original accepted scan records, and a
pending set obtained from a target-bound comparison. Results are proposals:
they have not been reconnected, checked on the current route, or certified.
"""
from __future__ import annotations

import heapq
import math
import numpy as np

from femto.planning.surface_trajectory import ScanSegment

VERSION='old-proof-scan-edge-set-cover-v1'
MAX_LEAVES=220_000
MAX_PROOFS=1_000_000
MAX_EDGES=200_000
MAX_SELECTED=512
MAX_UPDATES=1_000_000


def select_edges(leaf_weights, edge_leaves, edge_lengths, *, penalty_m, max_selected=MAX_SELECTED,
                 max_updates=MAX_UPDATES, check=lambda:None):
    """Deterministic greedy proposals; every uncovered leaf is returned.

    Gain is nominal old-leaf weight per candidate edge length plus a fixed
    fragmentation penalty. It is a heuristic, not a minimum-distance proof.
    """
    if (not isinstance(leaf_weights,dict) or len(leaf_weights)>MAX_LEAVES
            or not isinstance(edge_leaves,dict) or len(edge_leaves)>MAX_EDGES
            or not isinstance(edge_lengths,dict) or set(edge_lengths)!=set(edge_leaves)
            or type(max_selected) is not int or not 0<=max_selected<=MAX_SELECTED
            or type(max_updates) is not int or not 0<=max_updates<=MAX_UPDATES
            or isinstance(penalty_m,bool) or not isinstance(penalty_m,(int,float))
            or not math.isfinite(penalty_m) or penalty_m<0):
        raise ValueError('补段候选预算或输入无效')
    for leaf,weight in leaf_weights.items():
        if type(leaf) is not int or leaf<0 or isinstance(weight,bool) or not isinstance(weight,(float,int)) or not math.isfinite(weight) or weight<0:
            raise ValueError('补段候选叶权重无效')
    if any(not isinstance(edge,tuple) or len(edge)!=2 or any(type(i) is not int or i<0 for i in edge)
           for edge in edge_leaves):
        raise ValueError('补段候选扫描边无效')
    remaining=set(leaf_weights);by_leaf={key:[] for key in remaining}
    gains={};versions={};heap=[];associations=0
    for edge in sorted(edge_leaves):
        check()
        if (isinstance(edge_lengths[edge],bool) or not isinstance(edge_lengths[edge],(int,float))
                or not math.isfinite(edge_lengths[edge]) or edge_lengths[edge]<=0):
            raise ValueError('补段候选扫描边无效')
        leaves=edge_leaves[edge]
        if not isinstance(leaves,(set,frozenset)) or not leaves or not leaves.issubset(remaining):
            raise ValueError('补段候选叶关联无效')
        associations+=len(leaves)
        if associations>MAX_PROOFS:raise ValueError('补段候选关联超过预算')
        for leaf in sorted(leaves):by_leaf[leaf].append(edge)
        gains[edge]=math.fsum(max(float(leaf_weights[leaf]),1e-30) for leaf in sorted(leaves))
        versions[edge]=0
        heapq.heappush(heap,(-gains[edge]/(edge_lengths[edge]+penalty_m),edge,0))
    chosen=[];updates=0;stop='complete'
    while remaining and heap:
        check()
        if len(chosen)>=max_selected:stop='selection_budget';break
        _,edge,version=heapq.heappop(heap)
        if version!=versions[edge] or gains[edge]<=0:continue
        useful=sorted(edge_leaves[edge]&remaining)
        if not useful:continue
        chosen.append(edge);remaining.difference_update(useful)
        touched=set()
        for leaf in useful:
            for other in by_leaf[leaf]:
                if updates>=max_updates:stop='update_budget';break
                updates+=1;gains[other]=max(0.,gains[other]-max(float(leaf_weights[leaf]),1e-30))
                touched.add(other)
            if stop!='complete':break
        if stop!='complete':break
        for other in sorted(touched):
            versions[other]+=1
            if gains[other]>0:
                heapq.heappush(heap,(-gains[other]/(edge_lengths[other]+penalty_m),other,versions[other]))
    if remaining and stop=='complete':stop='no_candidate'
    return dict(selected_edges=chosen,unresolved_leaf_indices=sorted(remaining),
                selected_length_m=math.fsum(edge_lengths[e] for e in chosen),
                associations=associations,updates=updates,stop_reason=stop,
                all_pending_selected=not remaining)


def propose(previous_certificate, previous_seeds, previous_trajectory, domain, pending_leaves, width_mm, *,
            max_selected=MAX_SELECTED,max_updates=MAX_UPDATES,check=lambda:None):
    """Map rechecked old proof witnesses to their original accepted scan edges."""
    if (not isinstance(pending_leaves,(list,tuple)) or len(pending_leaves)>MAX_LEAVES
            or any(type(i) is not int or i<0 for i in pending_leaves)
            or list(pending_leaves)!=sorted(set(pending_leaves))
            or isinstance(width_mm,bool) or not isinstance(width_mm,(int,float))
            or not math.isfinite(width_mm) or not 0<width_mm<=1000):
        raise ValueError('补段待确认范围或作用宽度无效')
    arrays=previous_certificate.arrays
    if len(arrays['proof_leaf'])>MAX_PROOFS or len(arrays['seed_face'])>20_000:
        raise ValueError('补段旧证明超过预算')
    if any(i>=len(arrays['status']) or int(arrays['status'][i])!=1 for i in pending_leaves):
        raise ValueError('补段范围不是旧证明中的确认单元')
    segments={s.segment_index:s for s in previous_trajectory.ordered_items if isinstance(s,ScanSegment)}
    edge_by_seed={};lengths={}
    for seed_index,source_index in enumerate(arrays['seed_input_index']):
        if seed_index%1024==0:check()
        index=int(source_index)
        if not 0<=index<len(previous_seeds):raise ValueError('旧证明种子与原扫描边不一致')
        record=previous_seeds[index];face=int(record['source_face']);xyz=record['xyz_m']
        if (record.get('role')!='scan' or not 0<=face<len(domain.representative_faces)
                or int(domain.representative_faces[face])!=int(arrays['seed_face'][seed_index])
                or tuple(xyz)!=tuple(arrays['seed_xyz'][seed_index])):
            raise ValueError('旧证明种子与原扫描坐标或支持面不一致')
        sid,edge_index,end=(record.get(k) for k in ('segment_index','edge_index','endpoint'))
        if (type(sid) is not int or type(edge_index) is not int or type(end) is not int or end not in (0,1)
                or sid not in segments or not 0<=edge_index<len(segments[sid].points)-1):
            raise ValueError('旧证明种子扫描边引用无效')
        points=segments[sid].points;left,right=points[edge_index:edge_index+2]
        if right.get('role','scan')!='scan' or tuple(points[edge_index+end]['xyz_m'])!=tuple(xyz):
            raise ValueError('旧证明种子没有对应原扫描边端点')
        length=float(np.linalg.norm(np.asarray(left['xyz_m'])-np.asarray(right['xyz_m'])))
        if not math.isfinite(length) or length<=0:raise ValueError('旧证明原扫描边长度无效')
        edge=(sid,edge_index);edge_by_seed[seed_index]=edge;lengths[edge]=length
    pending=set(pending_leaves);edge_leaves={}
    for index,(leaf,seed) in enumerate(zip(arrays['proof_leaf'],arrays['proof_seed'])):
        if index%1024==0:check()
        leaf,seed=int(leaf),int(seed)
        if leaf in pending:
            if seed not in edge_by_seed:raise ValueError('待确认单元没有可定位的旧证明扫描边')
            edge_leaves.setdefault(edge_by_seed[seed],set()).add(leaf)
    weights={i:float(domain.areas[int(arrays['root'][i])])/(4**int(arrays['depth'][i])) for i in pending_leaves}
    result=select_edges(weights,edge_leaves,{e:lengths[e] for e in edge_leaves},
                        penalty_m=width_mm*.25/1000,max_selected=max_selected,max_updates=max_updates,check=check)
    return dict(schema=VERSION,**result,penalty_m=width_mm*.25/1000,
                pending_leaf_count=len(pending),candidate_edge_count=len(edge_leaves),
                semantics='proposal_only_requires_current_route_geometry_connection_and_proof_replay',
                installed_route=False,installed_certificate=False,execution_ready=False,physical_accuracy_verified=False)
