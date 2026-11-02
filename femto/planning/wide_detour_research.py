"""Bounded, repeatable scan-order search; proposals require final edge checks.

Nominal revisit is a heuristic. This module does not certify coverage, build a
saved route, or establish physical/robot safety. No old route version uses it.
"""
from __future__ import annotations

from copy import deepcopy
import math
import numpy as np
from femto.planning import balanced_visit_improvement,scan_visit_order
from femto.planning.endpoint_preserving_detour import _signature

VERSION='research-bounded-wide-detour-order-v1'
MAX_SEGMENTS=300
MAX_ROUNDS=8
MAX_TRIALS=256
MAX_SPAN=20
MAX_CACHE_ENTRIES=8192
MAX_CACHE_POINTS=100_000


def _ranked(segments,visits,links,check):
    def point(v,tail):
        segment=segments[v[0]]
        return segment.points[-1 if (1-int(v[1]) if tail else int(v[1])) else 0]
    heads=np.asarray([point(v,False)['xyz_m'] for v in visits]);tails=np.asarray([point(v,True)['xyz_m'] for v in visits])
    head_rows=np.asarray([point(v,False)['row_index'] for v in visits]);tail_rows=np.asarray([point(v,True)['row_index'] for v in visits])
    blocks=np.asarray([segments[v[0]].block_index for v in visits])
    missing=np.r_[0,np.cumsum([not x['connected'] for x in links])]
    result=[]
    for first in range(1,len(visits)-1):
        check();last=np.arange(first,min(len(visits)-1,first+MAX_SPAN+1))
        valid=(blocks[first-1]==blocks[last+1])&(missing[last+1]==missing[first-1])
        old=int(tail_rows[first-1]==head_rows[first])+(tail_rows[last]==head_rows[last+1]).astype(int)
        new=(tail_rows[first-1]==tail_rows[last]).astype(int)+(head_rows[first]==head_rows[last+1]).astype(int)
        valid &= new<old;last=last[valid];delta=(new-old)[valid]
        if not len(last):continue
        diff=(np.linalg.norm(tails[first-1]-tails[last],axis=1)+np.linalg.norm(heads[first]-heads[last+1],axis=1)
              -np.linalg.norm(tails[first-1]-heads[first])-np.linalg.norm(tails[last]-heads[last+1],axis=1))
        result.extend((int(d),float(distance),first,int(end)) for d,distance,end in zip(delta,diff,last))
    return sorted(result)


class _CachedGraph:
    def __init__(self,source):self.source=source;self.cache={};self.points=0
    def route(self,left,right,*,left_reverse=False,right_reverse=False):
        key=(left,right,left_reverse,right_reverse)
        if key in self.cache:return deepcopy(self.cache[key])
        result=self.source.route(left,right,left_reverse=left_reverse,right_reverse=right_reverse)
        size=len(result) if result else 0
        if len(self.cache)<MAX_CACHE_ENTRIES and self.points+size<=MAX_CACHE_POINTS:
            self.cache[key]=deepcopy(result);self.points+=size
        return result


def propose(graph,segments,visits,links,spacing_m,*,max_connector_m,max_revisit_m,
            check=lambda:None,budget=200_000,rounds=MAX_ROUNDS,trials=MAX_TRIALS):
    """Reorder complete scans under fixed gap, connection and revisit limits.

    Input graph must already apply the existing original geometry predicates.
    Output remains a proposal: caller must rerun all final edge checks.
    """
    values=(spacing_m,max_connector_m,max_revisit_m)
    if (not segments or len(segments)>MAX_SEGMENTS or type(rounds) is not int or not 1<=rounds<=MAX_ROUNDS
            or type(trials) is not int or not 1<=trials<=MAX_TRIALS or type(budget) is not int or budget<1
            or any(isinstance(v,bool) or not isinstance(v,(float,int)) or not math.isfinite(v) for v in values)
            or spacing_m<=0 or max_connector_m<0 or max_revisit_m<0):
        raise ValueError('有界换序参数无效或超过预算')
    current=scan_visit_order.checked_visits(segments,visits);current_links=deepcopy(links)
    original=scan_visit_order.reconstruct(segments,current,current_links,check,preserve_endpoint_faces=True,budget=budget)
    baseline=balanced_visit_improvement._metrics(original[0],spacing_m,check)
    original_signature=_signature(original[0],check);cached=_CachedGraph(graph)
    accepted=[];evaluated=0
    for iteration in range(rounds):
        check();current_route=scan_visit_order.reconstruct(segments,current,current_links,check,preserve_endpoint_faces=True,budget=budget)
        metrics=balanced_visit_improvement._metrics(current_route[0],spacing_m,check);best=None
        for _,_,first,last in _ranked(segments,current,current_links,check)[:trials]:
            check();candidate=list(current)
            candidate[first:last+1]=[(index,not reverse) for index,reverse in reversed(candidate[first:last+1])]
            candidate_links=scan_visit_order.connect(cached,segments,candidate,check,budget);evaluated+=1
            objective=scan_visit_order.anchored_objective(segments,candidate,candidate_links,check,budget=budget)
            if objective[0]>original[2][0] or objective[1]>max_connector_m+1e-12:continue
            proposed=scan_visit_order.reconstruct(segments,candidate,candidate_links,check,preserve_endpoint_faces=True,budget=budget)
            if proposed[2]!=objective:raise ValueError('有界换序连接统计无法重现')
            measured=balanced_visit_improvement._metrics(proposed[0],spacing_m,check)
            if (measured['detours']>=metrics['detours'] or measured['revisit_connector_m']>max_revisit_m+1e-12
                    or _signature(proposed[0],check)!=original_signature):continue
            rank=(measured['detours'],measured['revisit_connector_m'],objective[1])
            if best is None or rank<best[0]:best=(rank,candidate,candidate_links,first,last)
        if best is None:break
        rank,current,current_links,first,last=best
        accepted.append(dict(first=first,last=last,detours=rank[0],revisit_connector_m=rank[1],connector_m=rank[2]))
    result=scan_visit_order.reconstruct(segments,current,current_links,check,preserve_endpoint_faces=True,budget=budget)
    if _signature(result[0],check)!=original_signature:raise ValueError('有界换序改变原扫描边或来源面')
    report=dict(version=VERSION,adopted=bool(accepted),baseline=baseline,steps=accepted,
        visits=[list(v) for v in current],evaluated=evaluated,max_connector_m=max_connector_m,max_revisit_m=max_revisit_m,
        selected=balanced_visit_improvement._metrics(result[0],spacing_m,check),selected_objective=list(result[2]),
        semantics='proposal_requires_final_edge_and_target_checks')
    return result[0],deepcopy(current_links),result[1],report
