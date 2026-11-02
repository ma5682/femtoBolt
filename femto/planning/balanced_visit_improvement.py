"""Versioned proposal for jointly limiting detours and connector revisits.

Every candidate is reconnected by the existing verified graph. The nominal
revisit and detour metrics are planning heuristics, not coverage or dose.
Acceptance does not constitute a coverage certificate.
"""
from __future__ import annotations

from copy import deepcopy
import math
from femto.planning.scan_visit_order import checked_visits,connect,reconstruct
from femto.planning.surface_trajectory import ScanSegment
from femto.planning.verified_visit_improvement import improve

VERSION='verified-visit-balanced-prefix-proposal-v1'


def _legacy_metrics(chains, spacing_m, check):
    from femto.planning.selection_blocks import _mark_revisits,_walk_connectors
    # Revisit marking changes only top-level point keys. Clone each point
    # dictionary, while sharing read-only XYZ and other nested values, instead
    # of duplicating the full coordinate graph for every candidate.
    marked=[(block,[dict(point) for point in points]) for block,points in chains]
    _mark_revisits([points for _,points in marked],spacing_m,check)
    segments=[]
    for index,(block,points) in enumerate(marked):
        points[0]['role']='scan';points[0].pop('revisit',None)
        segments.append(ScanSegment(index,points[0]['row_index'],points,block))
    _,detours,_,revisit_m=_walk_connectors(segments)
    return dict(detours=detours,revisit_connector_m=revisit_m)


def _metrics(chains, spacing_m, check):
    from femto.planning.nominal_revisit_metrics import measure
    value=measure(chains,spacing_m,check)
    return _legacy_metrics(chains,spacing_m,check) if value is None else value


def propose(graph,segments,visits,links,spacing_m,check,budget,*,search=None):
    """Keep a shorter verified prefix only if it does not worsen both metrics."""
    if isinstance(spacing_m,bool) or not isinstance(spacing_m,(int,float)) or not 0 < spacing_m < float('inf'):
        raise ValueError('重访判定行距无效')
    original_visits=checked_visits(segments,visits)
    original=reconstruct(segments,original_visits,links,check)
    baseline=_metrics(original[0],spacing_m,check)
    full_chains,full_links,full_added,trial=improve(graph,segments,original_visits,links,check,budget,
        **({'search':search} if search is not None else {}))
    if not trial['adopted']:
        return original[0],deepcopy(links),original[1],dict(version=VERSION,adopted=False,baseline=baseline,attempts=[])
    steps=trial['rounds'];states=[list(original_visits)]
    for step in steps:
        check();current=list(states[-1]);first,last=step['first'],step['last']
        current[first:last+1]=[(index,not reverse) for index,reverse in reversed(current[first:last+1])]
        states.append(checked_visits(segments,current))
    attempts=[]
    for count in range(len(steps),0,-1):
        if search is not None and not search.take('balanced_prefix'):break
        check();selected_visits=states[count]
        if count==len(steps):selected_links=full_links;candidate=(full_chains,full_added,tuple(trial['selected_objective']))
        else:
            selected_links=connect(graph,segments,selected_visits,check,budget)
            candidate=reconstruct(segments,selected_visits,selected_links,check)
        metrics=_metrics(candidate[0],spacing_m,check)
        accepted=(candidate[2][0]<=original[2][0] and candidate[2][1]<original[2][1]-1e-12
                  and metrics['detours']<=baseline['detours']
                  and metrics['revisit_connector_m']<=baseline['revisit_connector_m']+1e-12)
        attempts.append(dict(round_count=count,metrics=metrics,objective=list(candidate[2]),accepted=accepted))
        if accepted:
            return candidate[0],selected_links,candidate[1],dict(version=VERSION,adopted=True,baseline=baseline,
                selected=metrics,baseline_objective=list(original[2]),selected_objective=list(candidate[2]),
                visits=[list(v) for v in selected_visits],rounds=deepcopy(steps[:count]),attempts=attempts)
    return original[0],deepcopy(links),original[1],dict(version=VERSION,adopted=False,baseline=baseline,attempts=attempts)


def validate_record(record,segments,visits,initial_links,final_links,spacing_m,check):
    """Check bound routes and metrics; independent route replay reruns search."""
    from femto.planning.verified_visit_improvement import validate_record as validate_order,VERSION as ORDER_VERSION,MAX_ROUNDS
    if (not isinstance(record,dict) or record.get('version')!=VERSION or type(record.get('adopted')) is not bool
            or record.get('baseline_visits')!=visits or record.get('baseline_connections')!=initial_links
            or not isinstance(record.get('attempts'),list) or len(record['attempts'])>MAX_ROUNDS):
        raise ValueError('绕行与重访优化记录无效')
    skipped=record.get('skipped')
    if skipped not in (None,'geometry_budget') or (skipped and (record['adopted'] or record['attempts'])):
        raise ValueError('绕行与重访优化预算记录无效')
    original=reconstruct(segments,visits,initial_links,check)
    baseline=_metrics(original[0],spacing_m,check)
    if record.get('baseline')!=baseline:raise ValueError('绕行与重访优化基准不能重算')
    steps=record.get('rounds',[])
    order=dict(version=ORDER_VERSION,adopted=record['adopted'],rounds=steps,
               baseline_visits=visits,baseline_connections=initial_links)
    if record['adopted']:
        for key in ('visits','baseline_objective','selected_objective'):order[key]=record.get(key)
    final_visits=validate_order(order,segments,visits,initial_links,final_links,check)
    attempts=record['attempts']
    previous=None
    for position,attempt in enumerate(attempts):
        if (not isinstance(attempt,dict) or set(attempt)!={'round_count','metrics','objective','accepted'}
                or type(attempt['round_count']) is not int or not 1<=attempt['round_count']<=MAX_ROUNDS
                or (previous is not None and attempt['round_count']!=previous-1)
                or type(attempt['accepted']) is not bool or (attempt['accepted'] and position!=len(attempts)-1)):
            raise ValueError('绕行与重访优化尝试记录无效')
        metrics,objective=attempt['metrics'],attempt['objective']
        if (not isinstance(metrics,dict) or set(metrics)!={'detours','revisit_connector_m'}
                or type(metrics['detours']) is not int or metrics['detours']<0
                or isinstance(metrics['revisit_connector_m'],bool) or not isinstance(metrics['revisit_connector_m'],(float,int))
                or not math.isfinite(metrics['revisit_connector_m']) or metrics['revisit_connector_m']<0
                or not isinstance(objective,list) or len(objective)!=2 or type(objective[0]) is not int or objective[0]<0
                or isinstance(objective[1],bool) or not isinstance(objective[1],(float,int))
                or not math.isfinite(objective[1]) or objective[1]<0):
            raise ValueError('绕行与重访优化尝试指标无效')
        eligible=(objective[0]<=original[2][0] and objective[1]<original[2][1]-1e-12
                  and metrics['detours']<=baseline['detours']
                  and metrics['revisit_connector_m']<=baseline['revisit_connector_m']+1e-12)
        if attempt['accepted']!=eligible:raise ValueError('绕行与重访优化尝试判断无效')
        previous=attempt['round_count']
    if record['adopted']:
        selected=reconstruct(segments,final_visits,final_links,check)
        metrics=_metrics(selected[0],spacing_m,check)
        if (not attempts or not attempts[-1]['accepted'] or attempts[-1]['round_count']!=len(steps)
                or attempts[-1]['metrics']!=metrics or attempts[-1]['objective']!=list(selected[2])
                or record.get('selected')!=metrics or metrics['detours']>baseline['detours']
                or metrics['revisit_connector_m']>baseline['revisit_connector_m']+1e-12):
            raise ValueError('绕行与重访优化不满足保留约束')
    elif any(attempt['accepted'] for attempt in attempts):raise ValueError('未采用的绕行优化包含成功记录')
    return final_visits
