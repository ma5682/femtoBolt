"""Research-only, bounded detour proposal preserving scan-edge identities.

Links must come from the caller's verified surface graph. This module alone
does not validate mesh edges, prove coverage, or produce a product archive.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import math
import numpy as np

from femto.planning.scan_visit_order import anchored_objective,checked_visits,connect,reconstruct
from femto.planning.scan_edge_revisit import mark
from femto.planning.selection_blocks import _walk_connectors
from femto.planning.surface_trajectory import ScanSegment

VERSION='experimental-preserved-endpoint-detour-v1'
SHORT_VERSION='experimental-preserved-endpoint-short-detour-v2'
MAX_SEGMENTS=512
MAX_ROUNDS=4
MAX_TRIALS_PER_ROUND=4
MAX_CONNECTOR_INCREASE=0.04


def _anchored(segments,visits,links,check,budget):
    """Retain source-face anchor when graph endpoint uses another face."""
    return reconstruct(segments,visits,links,check,preserve_endpoint_faces=True,budget=budget)


def _signature(chains,check):
    """Keep each active edge and each endpoint's original local face/row."""
    edges=Counter();points=set();count=0
    for block,chain in chains:
        for a,b in zip(chain,chain[1:]):
            count+=1
            if count%256==0:check()
            if b.get('role')=='connector' or a['xyz_m']==b['xyz_m']:continue
            x,y=tuple(a['xyz_m']),tuple(b['xyz_m']);row=b['row_index']
            edges[(block,row,*sorted((x,y)))]+=1
            points.add((block,row,a['triangle_index'],*x))
            points.add((block,row,b['triangle_index'],*y))
    check();return edges,points


def _metrics(chains,spacing,check):
    copied=[(block,[dict(point) for point in points]) for block,points in chains]
    result=mark([points for _,points in copied],spacing,check)
    segments=[ScanSegment(i,points[0]['row_index'],points,block)
              for i,(block,points) in enumerate(copied)]
    _,detours,_,_=_walk_connectors(segments)
    return dict(detours=detours,revisit_connector_m=result['connector_length_m'])


def _ranked(segments,visits,links,check):
    """Deterministic optimistic ranking; distances never verify a link."""
    def endpoint(v,tail):
        segment=segments[v[0]]
        return segment.points[-1 if (1-int(v[1]) if tail else int(v[1])) else 0]
    heads=[endpoint(v,False) for v in visits];tails=[endpoint(v,True) for v in visits]
    missing=[0]
    for link in links:missing.append(missing[-1]+int(not link['connected']))
    candidates=[]
    for first in range(1,len(visits)-1):
        check()
        for last in range(first,len(visits)-1):
            if (segments[visits[first-1][0]].block_index!=segments[visits[last+1][0]].block_index
                    or missing[last+1]!=missing[first-1]):continue
            old=int(tails[first-1]['row_index']==heads[first]['row_index'])+int(tails[last]['row_index']==heads[last+1]['row_index'])
            new=int(tails[first-1]['row_index']==tails[last]['row_index'])+int(heads[first]['row_index']==heads[last+1]['row_index'])
            delta=new-old
            if delta>=0:continue
            a,b,c,d=(np.asarray(p['xyz_m'],dtype=float) for p in (tails[first-1],heads[first],tails[last],heads[last+1]))
            distance=float(np.linalg.norm(a-c)+np.linalg.norm(b-d)-np.linalg.norm(a-b)-np.linalg.norm(c-d))
            candidates.append((delta,distance,first,last))
    return sorted(candidates)


def propose(graph,segments,visits,links,spacing,check,budget,*,short_candidates=False,search=None,motion_limits=None):
    """Propose a deterministic verified-graph order, falling back unchanged.

    The caller must independently verify all baseline and emitted edges. The
    bounded increase is relative to the input connector length, never a per-
    iteration allowance. Neither output nor record is a coverage certificate.
    """
    if type(short_candidates) is not bool:raise ValueError('绕行候选策略无效')
    if not segments:raise ValueError('扫描段不能为空')
    if isinstance(spacing,bool) or not isinstance(spacing,(float,int)) or not math.isfinite(spacing) or spacing<=0:
        raise ValueError('重访统计行距无效')
    initial=checked_visits(segments,visits)
    original=reconstruct(segments,initial,links,check)
    baseline=_metrics(original[0],spacing,check)
    signature=_signature(original[0],check)
    record=dict(version=SHORT_VERSION if short_candidates else VERSION,adopted=False,baseline=baseline,baseline_objective=list(original[2]),steps=[])
    if len(initial)>MAX_SEGMENTS:return original[0],deepcopy(links),original[1],record
    current=list(initial);current_links=deepcopy(links);current_result=original;current_metrics=baseline
    ceiling=original[2][1]*(1+MAX_CONNECTOR_INCREASE)
    if motion_limits is not None:ceiling=min(ceiling,motion_limits['connector_length_m'])
    for _ in range(MAX_ROUNDS):
        if search is not None and not search.can_try('detour'):break
        check();best=None
        candidates=_ranked(segments,current,current_links,check)
        candidates=([candidate for candidate in candidates if candidate[3]-candidate[2]<=2][:8]
                    if short_candidates else candidates[:MAX_TRIALS_PER_ROUND])
        for _,_,first,last in candidates:
            if search is not None and not search.take('detour'):break
            check();candidate=list(current)
            candidate[first:last+1]=[(i,not reverse) for i,reverse in reversed(candidate[first:last+1])]
            try:
                candidate_links=connect(graph,segments,candidate,check,budget)
                objective=anchored_objective(segments,candidate,candidate_links,check,budget=budget)
                if objective[0]>original[2][0] or objective[1]>ceiling+1e-12:continue
                candidate_result=_anchored(segments,candidate,candidate_links,check,budget)
            except ValueError as error:
                if str(error)=='连续路线采样点超过预算，请缩小选区':continue
                if str(error)=='连续路线整边搜索超过预算，请缩小选区':
                    record.update(adopted=False,steps=[],skipped='geometry_budget')
                    return original[0],deepcopy(links),original[1],record
                raise
            if candidate_result[2]!=objective:raise ValueError('保留端点绕行连接成本不一致')
            if _signature(candidate_result[0],check)!=signature:continue
            score=_metrics(candidate_result[0],spacing,check)
            if score['detours']>=current_metrics['detours'] or score['revisit_connector_m']>baseline['revisit_connector_m']+1e-12:continue
            if motion_limits is not None:
                from femto.planning.balanced_visit_improvement import _metrics as global_metrics
                measured=global_metrics(candidate_result[0],spacing,check)
                if (measured['detours']>motion_limits['detours']
                        or measured['revisit_connector_m']>motion_limits['revisit_connector_m']+1e-12):continue
            rank=(score['detours'],score['revisit_connector_m'],candidate_result[2][1],first,last)
            if best is None or rank<best[0]:best=(rank,candidate,candidate_links,candidate_result,score)
        if best is None:break
        rank,current,current_links,current_result,current_metrics=best
        record['steps'].append(dict(first=rank[3],last=rank[4]))
    if not record['steps']:return original[0],deepcopy(links),original[1],record
    record.update(adopted=True,visits=[list(v) for v in current],selected=current_metrics,
                  selected_objective=list(current_result[2]))
    return current_result[0],current_links,current_result[1],record


def validate_record(record,segments,visits,initial_links,final_links,spacing,check,budget,*,short_candidates=False):
    """Rebuild outcome constraints; caller still checks all graph/mesh edges.

    This does not independently rerun the heuristic search, inspect a coverage
    certificate, or establish robot readiness. Unadopted records must retain
    their original links and cannot include speculative steps.
    """
    if type(short_candidates) is not bool:raise ValueError('绕行候选策略无效')
    if not segments:raise ValueError('扫描段不能为空')
    keys={'version','adopted','baseline','baseline_objective','steps'}
    skipped=record.get('skipped') if isinstance(record,dict) else None
    if skipped is not None:keys.add('skipped')
    if (not isinstance(record,dict) or type(record.get('adopted')) is not bool
            or record.get('version')!=(SHORT_VERSION if short_candidates else VERSION)
            or set(record)!=keys|({'visits','selected','selected_objective'} if record.get('adopted') else set())
            or not isinstance(record.get('steps'),list) or len(record['steps'])>MAX_ROUNDS):
        raise ValueError('保留端点绕行记录无效')
    if skipped not in (None,'geometry_budget') or (skipped and (record['adopted'] or record['steps'])):
        raise ValueError('保留端点绕行预算记录无效')
    if isinstance(spacing,bool) or not isinstance(spacing,(float,int)) or not math.isfinite(spacing) or spacing<=0:
        raise ValueError('重访统计行距无效')
    current=checked_visits(segments,visits)
    baseline=reconstruct(segments,current,initial_links,check)
    baseline_metrics=_metrics(baseline[0],spacing,check)
    if record['baseline']!=baseline_metrics or record['baseline_objective']!=list(baseline[2]):
        raise ValueError('保留端点绕行基准不能重算')
    if not record['adopted']:
        if record['steps'] or final_links!=initial_links:raise ValueError('未采用的绕行记录改变了路线')
        return baseline[0],baseline[1]
    if not record['steps'] or len(current)>MAX_SEGMENTS:raise ValueError('保留端点绕行步骤无效')
    for step in record['steps']:
        check()
        if (not isinstance(step,dict) or set(step)!={'first','last'}
                or type(step['first']) is not int or type(step['last']) is not int
                or not 1<=step['first']<=step['last']<len(current)-1
                or (short_candidates and step['last']-step['first']>2)):
            raise ValueError('保留端点绕行步骤无效')
        first,last=step['first'],step['last']
        current[first:last+1]=[(i,not reverse) for i,reverse in reversed(current[first:last+1])]
        current=checked_visits(segments,current)
    if record['visits']!=[list(v) for v in current]:raise ValueError('保留端点绕行顺序与步骤不一致')
    result=_anchored(segments,current,final_links,check,budget)
    selected=_metrics(result[0],spacing,check)
    if (_signature(result[0],check)!=_signature(baseline[0],check)
            or result[2][0]>baseline[2][0]
            or result[2][1]>baseline[2][1]*(1+MAX_CONNECTOR_INCREASE)+1e-12
            or selected['detours']>=baseline_metrics['detours']
            or selected['revisit_connector_m']>baseline_metrics['revisit_connector_m']+1e-12
            or record['selected']!=selected or record['selected_objective']!=list(result[2])):
        raise ValueError('保留端点绕行结果不满足扫描与连接约束')
    return result[0],result[1]
