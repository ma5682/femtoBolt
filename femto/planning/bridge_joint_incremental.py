"""Experimental two-step visit proposal with strict final route constraints.

An intermediate order may exceed a nominal statistic and is never emitted.
Unchanged links may be borrowed from a detached local snapshot while changed
links are searched again. Returned links are independent copies. The caller
must independently check all baseline and final
geometry; the proposal alone is not a coverage proof.
"""
from __future__ import annotations

from copy import deepcopy
from femto.planning import bounded_joint_reorder as base
from femto.planning.scan_visit_order import checked_visits,reconstruct
from femto.planning.connector_revisit_index import ConnectorRevisitIndex

VERSION='experimental-bridge-joint-incremental-v1'
MAX_ROUNDS=3
MAX_TRIALS=24
MAX_HISTORY_EXCESS_FRACTION=.01


def _connect_reusing(graph,segments,candidate,current,current_links,check,budget,*,copy_unchanged=True):
    """Reuse unchanged links; default results remain independent of inputs.

    The proposal's private search may borrow links from its own detached
    snapshot. Such borrowed records must never be mutated or returned to a
    caller without first copying them.
    """
    if type(copy_unchanged) is not bool:raise ValueError('连接复用拷贝策略无效')
    candidate=checked_visits(segments,candidate);current=checked_visits(segments,current)
    if not isinstance(current_links,list) or len(current_links)!=len(current)-1:
        raise ValueError('访问顺序的连接记录缺失')
    previous={}
    for left,right,link in zip(current,current[1:],current_links):
        check()
        if (not isinstance(link,dict) or link.get('from_segment')!=left[0] or link.get('to_segment')!=right[0]
                or type(link.get('connected')) is not bool or not isinstance(link.get('samples'),list)
                or bool(link['samples'])!=link['connected'] or link.get('added_edges')!=len(link['samples'])):
            raise ValueError('访问顺序的连接索引或点数不一致')
        previous[(left,right)]=link
    total=sum(len(s.points) for s in segments);links=[]
    for left,right in zip(candidate,candidate[1:]):
        check();same=segments[left[0]].block_index==segments[right[0]].block_index
        old=previous.get((left,right))
        if old is not None:link=deepcopy(old) if copy_unchanged else old
        else:
            route=graph.route(left[0],right[0],left_reverse=left[1],right_reverse=right[1]) if same else None
            link=dict(from_segment=left[0],to_segment=right[0],connected=bool(route),
                      added_edges=len(route) if route else 0,samples=route or [],
                      reason='verified_surface_path' if route else 'no_verified_graph_path' if same else 'separate_work_blocks')
        total+=len(link['samples'])
        if total>budget:raise ValueError('连续路线采样点超过预算，请缩小选区')
        links.append(link)
    return links


def propose(graph,segments,visits,links,spacing,check,budget,*,search=None,motion_limits=None):
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
        if search is not None and not search.can_try('joint'):break
        best=None;bridge=None
        for _,_,first,last in base._ranked(segments,current,current_links,check)[:MAX_TRIALS]:
            if search is not None and not search.take('joint'):break
            check();candidate=list(current)
            candidate[first:last+1]=[(i,not reverse) for i,reverse in reversed(candidate[first:last+1])]
            try:
                checked_visits(segments,candidate)
                candidate_links=_connect_reusing(graph,segments,candidate,current,current_links,check,budget,copy_unchanged=False)
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
            if motion_limits is not None:
                from femto.planning.balanced_visit_improvement import _metrics as global_metrics
                measured=global_metrics(result[0],spacing,check)
                if (result[2][1]>motion_limits['connector_length_m']+base.EPS
                        or measured['detours']>motion_limits['detours']
                        or measured['revisit_connector_m']>motion_limits['revisit_connector_m']+base.EPS):continue
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
    # Candidate search can borrow from an intermediate snapshot. Detach the
    # accepted result once, so callers cannot mutate that snapshot or inputs.
    return selected[0],deepcopy(current_links),selected[1],record


def validate_record(record,segments,visits,initial_links,final_links,spacing,check,budget):
    """Rebuild and verify final constraints; caller independently checks edges."""
    if (not isinstance(record,dict) or record.get('version')!=VERSION
            or not isinstance(record.get('steps'),list) or len(record['steps'])>MAX_ROUNDS):
        raise ValueError('两步联合换序版本或步骤无效')
    converted=dict(record,version=base.VERSION)
    return base.validate_record(converted,segments,visits,initial_links,final_links,spacing,check,budget)
