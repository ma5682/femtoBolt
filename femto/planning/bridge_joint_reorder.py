"""Experimental two-step visit proposal with strict final route constraints.

An intermediate order may exceed a nominal statistic and is never emitted.
Graph paths are not cached. The caller must check all baseline and final
geometry; this is neither a registered product version nor a coverage proof.
"""
from __future__ import annotations

from copy import deepcopy
from femto.planning import bounded_joint_reorder as base
from femto.planning.scan_visit_order import checked_visits,connect,reconstruct
from femto.planning.connector_revisit_index import ConnectorRevisitIndex

VERSION='experimental-bridge-joint-reorder-v1'
MAX_ROUNDS=3
MAX_TRIALS=24
MAX_HISTORY_EXCESS_FRACTION=.01


def propose(graph,segments,visits,links,spacing,check,budget):
    """Consider one bounded provisional move, then require a valid final route."""
    if not segments:raise ValueError('扫描段不能为空')
    initial=checked_visits(segments,visits)
    original=reconstruct(segments,initial,links,check,preserve_endpoint_faces=True,budget=budget)
    before=base._values(original[0],spacing,check);signature=base._signature(original[0],check)
    record=dict(version=VERSION,adopted=False,steps=[],baseline=before,baseline_objective=list(original[2]))
    if len(initial)>base.MAX_SEGMENTS:return original[0],deepcopy(links),original[1],record
    index=ConnectorRevisitIndex(segments,spacing,check);cache=base._ScoreCache(index,check)
    current=list(initial);current_links=deepcopy(links);selected=original;values=before;bridge_step=None
    for iteration in range(MAX_ROUNDS):
        best=None;bridge=None
        for _,_,first,last in base._ranked(segments,current,current_links,check)[:MAX_TRIALS]:
            check();candidate=list(current)
            candidate[first:last+1]=[(i,not reverse) for i,reverse in reversed(candidate[first:last+1])]
            try:
                checked_visits(segments,candidate)
                candidate_links=connect(graph,segments,candidate,check,budget)
                gaps=detours=0;length=history=edge=0.
                for left,right,link in zip(candidate,candidate[1:],candidate_links):
                    check()
                    if not link['connected']:
                        gaps+=int(segments[left[0]].block_index==segments[right[0]].block_index);continue
                    tail=base._endpoint(segments,left,True);head=base._endpoint(segments,right,False)
                    measured=cache.measure((left,right),tail,head,link['samples'])
                    length+=measured['length_m'];history+=measured['history_revisit_m'];edge+=measured['edge_revisit_m']
                    detours+=int(tail['row_index']==head['row_index'])
                if (gaps>original[2][0] or length>original[2][1]+base.EPS or detours>before['detours']
                        or edge>before['edge_revisit_m']+base.EPS):continue
                temporary=(history>before['history_revisit_m']+base.EPS)
                if temporary and (iteration!=0 or record['steps'] or history>before['history_revisit_m']*(1+MAX_HISTORY_EXCESS_FRACTION)+base.EPS):continue
                result=base._full_anchors(segments,candidate,candidate_links,check,budget)
            except ValueError as error:
                if str(error)=='连续路线采样点超过预算，请缩小选区':continue
                if str(error)=='连续路线整边搜索超过预算，请缩小选区':
                    record.update(skipped='geometry_budget',steps=[])
                    return original[0],deepcopy(links),original[1],record
                raise
            actual=base._values(result[0],spacing,check)
            if (abs(length-result[2][1])>base.EPS or abs(history-actual['history_revisit_m'])>base.EPS
                    or abs(edge-actual['edge_revisit_m'])>base.EPS or detours!=actual['detours']):
                raise ValueError('联合换序统计与完整路线不一致')
            if (base._signature(result[0],check)!=signature or result[2][0]>original[2][0]
                    or result[2][1]>original[2][1]+base.EPS or actual['detours']>before['detours']
                    or actual['edge_revisit_m']>before['edge_revisit_m']+base.EPS):continue
            if temporary:
                rank=(actual['history_revisit_m']-before['history_revisit_m'],actual['detours'],result[2][1],first,last)
                if bridge is None or rank<bridge[0]:bridge=(rank,candidate,candidate_links)
                continue
            if actual['history_revisit_m']>before['history_revisit_m']+base.EPS:continue
            rank=(actual['detours'],actual['history_revisit_m'],actual['edge_revisit_m'],result[2][1],first,last)
            prior=(values['detours'],values['history_revisit_m'],values['edge_revisit_m'],selected[2][1])
            if rank[:4]>=prior:continue
            if best is None or rank<best[0]:best=(rank,candidate,candidate_links,result,actual)
        if best is None:
            if iteration==0 and bridge is not None:
                rank,current,current_links=bridge;bridge_step=dict(first=rank[3],last=rank[4]);continue
            break
        rank,current,current_links,selected,values=best
        if bridge_step is not None:record['steps'].append(bridge_step);bridge_step=None
        record['steps'].append(dict(first=rank[4],last=rank[5]))
    if not record['steps']:return original[0],deepcopy(links),original[1],record
    record.update(adopted=True,visits=[list(v) for v in current],selected=values,selected_objective=list(selected[2]))
    return selected[0],current_links,selected[1],record


def validate_record(record,segments,visits,initial_links,final_links,spacing,check,budget):
    """Rebuild and verify final constraints; caller independently checks edges."""
    if (not isinstance(record,dict) or record.get('version')!=VERSION
            or not isinstance(record.get('steps'),list) or len(record['steps'])>MAX_ROUNDS):
        raise ValueError('两步联合换序版本或步骤无效')
    converted=dict(record,version=base.VERSION)
    return base.validate_record(converted,segments,visits,initial_links,final_links,spacing,check,budget)
