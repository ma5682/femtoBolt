"""Bounded checked-graph two-opt for supplement routes, not a coverage proof."""
from __future__ import annotations

from copy import deepcopy
import math
import numpy as np
from femto.planning import balanced_visit_improvement,scan_visit_order
from femto.planning.endpoint_preserving_detour import _signature

VERSION='research-supplement-checked-two-opt-v1'
MAX_SEGMENTS=300
MAX_PASSES=64
MAX_ATTEMPTS=256
MAX_QUERIES=20_000


def _anchored(segments,visits,links,check,budget):
    """Keep both source-face records, including equal-XYZ handoffs."""
    _,added,objective=scan_visit_order.reconstruct(segments,visits,links,check,preserve_endpoint_faces=True,budget=budget)
    first=scan_visit_order.oriented(segments[visits[0][0]],visits[0][1]);chains=[(first.block_index,first.points)]
    count=len(first.points)
    for (_, _),(right,reverse),link in zip(visits,visits[1:],links):
        check();following=scan_visit_order.oriented(segments[right],reverse)
        if link['connected']:
            chains[-1][1].extend(deepcopy(link['samples']));chains[-1][1].extend(following.points)
            count+=len(link['samples'])+len(following.points)
        else:
            chains.append((following.block_index,following.points));count+=len(following.points)
        if count>budget:raise ValueError('补线换序采样点超过预算')
    return chains,added,objective


def propose(graph,segments,visits,links,spacing_m,*,max_detours,max_revisit_m,check=lambda:None,
            budget=200_000,passes=32,attempts=24):
    """Optimize checked links under explicit limits; caller rechecks every edge."""
    if (not segments or len(segments)>MAX_SEGMENTS or type(passes) is not int or not 1<=passes<=MAX_PASSES
            or type(attempts) is not int or not 1<=attempts<=MAX_ATTEMPTS
            or type(max_detours) is not int or max_detours<0 or type(budget) is not int or budget<1
            or any(isinstance(v,bool) or not isinstance(v,(float,int)) or not math.isfinite(v) for v in (spacing_m,max_revisit_m))
            or spacing_m<=0 or max_revisit_m<0):raise ValueError('补线换序参数无效或超过预算')
    original_visits=scan_visit_order.checked_visits(segments,visits);current=list(original_visits)
    original_links=deepcopy(links);original=_anchored(segments,current,original_links,check,budget)
    if original[2][0]:raise ValueError('补线换序需要工作块内已经连接')
    baseline=balanced_visit_improvement._metrics(original[0],spacing_m,check)
    if baseline['detours']>max_detours:raise ValueError('补线初始绕行超过上限')
    signature=_signature(original[0],check)
    def row(v,tail):
        segment=segments[v[0]];index=(1-int(v[1])) if tail else int(v[1])
        return segment.points[-1 if index else 0]['row_index']
    def detours(seq):
        return sum(segments[a[0]].block_index==segments[b[0]].block_index and row(a,True)==row(b,False) for a,b in zip(seq,seq[1:]))
    def length(a,reverse,samples):
        if not samples:return float('inf')
        start=graph.nodes[graph.endpoints[a][int(not reverse)]]
        path=np.asarray([start]+[p['xyz_m'] for p in samples])
        return float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum())
    costs=[0. if segments[a[0]].block_index!=segments[b[0]].block_index else length(a[0],a[1],link['samples'])
           for a,b,link in zip(current,current[1:],original_links)]
    cache={};queries=0;steps=[];current_detours=detours(current)
    def checked(a,ar,b,br):
        nonlocal queries
        check();key=(a,ar,b,br)
        if key not in cache:
            queries+=1
            if queries>MAX_QUERIES:raise ValueError('补线换序搜索超过预算')
            samples=graph.route(a,b,left_reverse=ar,right_reverse=br);cache[key]=length(a,ar,samples)
        return cache[key]
    for _ in range(passes):
        check();choices=[]
        for i in range(1,len(current)):
            check();a,ar=current[i-1];b,br=current[i]
            if segments[a].block_index!=segments[b].block_index:continue
            left=graph.nodes[graph.endpoints[a][int(not ar)]];head_b=graph.nodes[graph.endpoints[b][int(br)]]
            for j in range(i,len(current)):
                c,cr=current[j]
                if segments[c].block_index!=segments[a].block_index:break
                tail_c=graph.nodes[graph.endpoints[c][int(not cr)]]
                optimistic=float(np.linalg.norm(left-tail_c))-costs[i-1]
                change=int(row(current[i-1],True)==row(current[j],True))-int(row(current[i-1],True)==row(current[i],False))
                if j+1<len(current) and segments[current[j+1][0]].block_index==segments[a].block_index:
                    d,dr=current[j+1];head_d=graph.nodes[graph.endpoints[d][int(dr)]]
                    optimistic+=float(np.linalg.norm(head_b-head_d))-costs[j]
                    change+=int(row(current[i],False)==row(current[j+1],False))-int(row(current[j],True)==row(current[j+1],False))
                if current_detours+change<=max_detours and optimistic < -1e-8:choices.append((optimistic,i,j))
        best=None
        for _,i,j in sorted(choices)[:attempts]:
            a,ar=current[i-1];b,br=current[i];c,cr=current[j]
            one=checked(a,ar,c,not cr);two=0.;delta=one-costs[i-1]
            same_next=j+1<len(current) and segments[current[j+1][0]].block_index==segments[a].block_index
            if same_next:
                d,dr=current[j+1];two=checked(b,not br,d,dr);delta+=two-costs[j]
            if np.isfinite(delta) and delta < -1e-8 and (best is None or (delta,i,j)<best[:3]):best=(delta,i,j,one,two,same_next)
        if best is None:break
        _,i,j,one,two,same_next=best
        current[i:j+1]=[(k,not reverse) for k,reverse in reversed(current[i:j+1])]
        current_detours=detours(current)
        if current_detours>max_detours:raise ValueError('补线换序绕行计算不一致')
        costs[i:j]=reversed(costs[i:j]);costs[i-1]=one
        if same_next:costs[j]=two
        steps.append(dict(first=i,last=j))
    candidate_links=scan_visit_order.connect(graph,segments,current,check,budget)
    candidate=_anchored(segments,current,candidate_links,check,budget)
    measured=balanced_visit_improvement._metrics(candidate[0],spacing_m,check)
    accepted=(bool(steps) and candidate[2][0]==0 and candidate[2][1]<=original[2][1]+1e-12
        and measured['detours']<=max_detours and measured['revisit_connector_m']<=max_revisit_m+1e-12
        and _signature(candidate[0],check)==signature)
    if not accepted:current=original_visits;candidate_links=original_links;candidate=original;measured=baseline
    report=dict(version=VERSION,adopted=accepted,visits=[list(v) for v in current],steps=steps if accepted else [],
        evaluated_queries=queries,baseline=baseline,selected=measured,selected_objective=list(candidate[2]),
        semantics='candidate_requires_final_geometry_and_coverage_checks')
    return candidate[0],deepcopy(candidate_links),candidate[1],report
