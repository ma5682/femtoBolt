"""Experimental single-budget relocation with incremental link pre-scoring.

The index scores a superset of scan points: a reconnect can omit a scan head
when its incoming connector ends on the same face. Its revisit scores are thus
ranking hints only. Every shortlisted candidate is fully rebuilt and measured;
the caller must independently check its surface edges, selection and coverage.
Versioned routing callers retain their own budgets and independent replay.
"""
from __future__ import annotations

from copy import deepcopy
import math
from numbers import Real

from femto.planning import balanced_visit_improvement as balanced, scan_visit_order as order
from femto.planning import verified_relocation_research as legacy
from femto.planning.bounded_joint_reorder import _ScoreCache
from femto.planning.connector_revisit_index import ConnectorRevisitIndex
from femto.planning.endpoint_preserving_detour import _signature

VERSION='research-budgeted-incremental-relocation-v1'
TOTAL_ATTEMPTS=512
MAX_ROUNDS=8
PER_ROUND=128
FULL_PER_ROUND=32
MIN_GAIN_M=.0001
MAX_CONNECTOR_INCREASE_M=.005
MAX_REVISIT_INCREASE_M=.002
MAX_RELATIVE_INCREASE=.02
EPS=1e-12


def _endpoint(segments,visit,tail):
    segment=segments[visit[0]]
    return segment.points[-1 if bool(tail)!=visit[1] else 0]


def _changes(visits,i,j,reverse,remaining=None):
    """Return the exact changed adjacencies for a single recorded move."""
    value=visits[i];left=visits[i-1]
    right=visits[i+1] if i+1<len(visits) else None
    if remaining is None:remaining=visits[:i]+visits[i+1:]
    before=remaining[j-1];after=remaining[j] if j<len(remaining) else None
    moved=(value[0],reverse)
    old={(left,value),(value,right),(before,after)}
    new={(left,right),(before,moved),(moved,after)}
    old={edge for edge in old if None not in edge}
    new={edge for edge in new if None not in edge}
    return sorted(old-new),sorted(new-old)


class _EdgeScorer:
    """Bounded per-proposal link arithmetic; never certifies a graph edge."""
    def __init__(self,segments,graph,spacing,check):
        self.segments,self.graph,self.check=segments,graph,check
        self.cache=_ScoreCache(ConnectorRevisitIndex(segments,spacing,check),check)

    def score(self,edge):
        self.check();a,b=edge;left,right=self.segments[a[0]],self.segments[b[0]]
        if left.block_index!=right.block_index:return (0,0.,0.)
        samples=self.graph.route(a[0],b[0],left_reverse=a[1],right_reverse=b[1])
        if not samples:return None
        tail,head=_endpoint(self.segments,a,True),_endpoint(self.segments,b,False)
        value=self.cache.measure(edge,tail,head,samples)
        return (int(tail['row_index']==head['row_index']),value['history_revisit_m'],value['length_m'])


def _ranked(segments,visits,scorer,check,cache):
    """Prioritize existing detours, long links and nominal repeat crossings."""
    original={edge:scorer.score(edge) for edge in zip(visits,visits[1:])}
    if any(value is None for value in original.values()):
        raise ValueError('有界移动需要工作块内已连接')
    candidates=[];remaining_by_index={}
    for estimate,i,j,reverse in legacy._candidates(segments,visits,check,cache):
        remaining=remaining_by_index.get(i)
        if remaining is None:
            remaining=visits[:i]+visits[i+1:];remaining_by_index[i]=remaining
        removed,added=_changes(visits,i,j,reverse,remaining)
        # Only arithmetic on already verified input edges is used at this step.
        bad=[original[edge] for edge in removed]
        key=(-sum(v[0] for v in bad),-sum(v[1] for v in bad),
             -sum(v[2] for v in bad),estimate,i,j,reverse)
        candidates.append((key,removed,added))
    return sorted(candidates,key=lambda value:value[0]),original


def _work_queue(candidates,count):
    """Keep distance-ranked exploration while reserving attention to bad links."""
    by_distance=sorted(candidates,key=lambda item:(item[0][3],*item[0][-3:]))
    result=[];seen=set();distance_quota=(count*7)//8
    for candidate in by_distance[:distance_quota]+candidates+by_distance[distance_quota:]:
        key=candidate[0][-3:]
        if key in seen:continue
        seen.add(key);result.append(candidate)
        if len(result)>=count:break
    return result


def _queue(segments,visits,scorer,count,check,cache):
    """Keep legacy decisions; batch only the arithmetic and queue selection."""
    from femto.planning import relocation_candidate_batch as batch
    if not batch.supported(segments,visits):
        candidates,original=_ranked(segments,visits,scorer,check,cache)
        return _work_queue(candidates,count),original
    original={edge:scorer.score(edge) for edge in zip(visits,visits[1:])}
    if any(value is None for value in original.values()):
        raise ValueError('有界移动需要工作块内已连接')
    return batch.legacy_queue(segments,visits,original,count,check,cache),original


def _limits_from(baseline,values):
    return dict(detours=values['detours'],
        connector_length_m=baseline[2][1]+min(MAX_CONNECTOR_INCREASE_M,baseline[2][1]*MAX_RELATIVE_INCREASE),
        revisit_connector_m=values['revisit_connector_m']+min(MAX_REVISIT_INCREASE_M,values['revisit_connector_m']*MAX_RELATIVE_INCREASE))


def limits_for(segments,visits,links,spacing_m,*,check=lambda:None,budget=200_000):
    """Bound tradeoffs against the verified stage baseline, in metres."""
    baseline=order.reconstruct(segments,visits,links,check,preserve_endpoint_faces=True,budget=budget)
    if baseline[2][0]:raise ValueError('有界移动需要工作块内已连接')
    return _limits_from(baseline,balanced._metrics(baseline[0],spacing_m,check))


def _motion_improves(values,length,prior,prior_length):
    """A small reduction in revisit alone cannot buy a longer connector."""
    if values['detours']!=prior['detours']:return values['detours']<prior['detours']
    return (values['revisit_connector_m']<=prior['revisit_connector_m']+EPS and length<=prior_length+EPS
            and (values['revisit_connector_m']<prior['revisit_connector_m']-EPS or length<prior_length-EPS))


def propose(graph,segments,visits,links,spacing_m,*,motion_limits,check=lambda:None,budget=200_000,
            total_attempts=TOTAL_ATTEMPTS,per_round=PER_ROUND,rounds=MAX_ROUNDS,full_per_round=FULL_PER_ROUND,search=None):
    """Produce a bounded multi-objective candidate; final geometry is external."""
    if (not isinstance(segments,list) or not segments or len(segments)>legacy.MAX_SEGMENTS
            or isinstance(spacing_m,bool) or not isinstance(spacing_m,Real) or not math.isfinite(spacing_m) or spacing_m<=0
            or type(budget) is not int or not 1<=budget<=200_000
            or type(total_attempts) is not int or not 1<=total_attempts<=legacy.MAX_TRIALS
            or type(per_round) is not int or not 1<=per_round<=total_attempts
            or type(rounds) is not int or not 1<=rounds<=MAX_ROUNDS
            or type(full_per_round) is not int or not 1<=full_per_round<=per_round
            or not isinstance(motion_limits,dict) or set(motion_limits)!={'detours','connector_length_m','revisit_connector_m'}
            or type(motion_limits.get('detours')) is not int or motion_limits['detours']<0
            or any(isinstance(motion_limits.get(k),bool) or not isinstance(motion_limits.get(k),Real)
                   or not math.isfinite(motion_limits[k]) or motion_limits[k]<0
                   for k in ('connector_length_m','revisit_connector_m'))):
        raise ValueError('有界移动参数无效')
    current=order.checked_visits(segments,visits)
    initial=order.reconstruct(segments,current,links,check,preserve_endpoint_faces=True,budget=budget)
    if initial[2][0]:raise ValueError('有界移动需要工作块内已连接')
    initial_values=balanced._metrics(initial[0],spacing_m,check)
    signature=_signature(initial[0],check)
    if signature!=_signature([(s.block_index,s.points) for s in segments],check):
        raise ValueError('有界移动基准改变了原扫描边')
    maximum=_limits_from(initial,initial_values)
    if any(motion_limits[k]>maximum[k]+EPS for k in maximum):
        raise ValueError('有界移动限制超过允许的基准增量')
    local=legacy._LocalGraph(graph,current,links,check)
    scorer=_EdgeScorer(segments,local,spacing_m,check)
    preview=legacy._Preview(segments,check);lengths=legacy._ConnectionLengths();rank_cache={}
    selected=initial;current_links=deepcopy(links);values=initial_values
    steps=[];tested=full_evaluations=0
    for round_index in range(rounds):
        if search is not None and not search.can_try('relocation'):break
        check();requested=per_round if round_index==0 else max(1,per_round//2)
        candidates,original=_queue(segments,current,scorer,min(requested,total_attempts-tested),check,rank_cache)
        proxy_base=tuple(sum(value[k] for value in original.values()) for k in range(3))
        shortlist=[]
        for key,removed,added in candidates:
            if search is not None and not search.take('relocation'):break
            check();tested+=1
            plus=[scorer.score(edge) for edge in added]
            if any(value is None for value in plus):continue
            minus=[original[edge] for edge in removed]
            proxy=tuple(proxy_base[k]+sum(v[k] for v in plus)-sum(v[k] for v in minus) for k in range(3))
            shortlist.append((proxy,key))
        best=None
        for _,key in sorted(shortlist,key=lambda item:(item[0],item[1]))[:full_per_round]:
            check();full_evaluations+=1
            i,j,reverse=key[-3:];value=current[i];remaining=current[:i]+current[i+1:]
            proposal=remaining[:j]+[(value[0],reverse)]+remaining[j:]
            try:
                candidate_links=order.connect(local,segments,proposal,check,budget)
                objective=order.anchored_objective(segments,proposal,candidate_links,check,budget=budget,_lengths=lengths)
            except ValueError as error:
                if str(error)=='连续路线采样点超过预算，请缩小选区':continue
                raise
            if objective[0]>selected[2][0] or objective[1]>motion_limits['connector_length_m']+EPS:continue
            measured=balanced._metrics(preview.chains(proposal,candidate_links),spacing_m,check)
            if (measured['detours']>motion_limits['detours']
                    or measured['revisit_connector_m']>motion_limits['revisit_connector_m']+EPS
                    or not _motion_improves(measured,objective[1],values,selected[2][1])):continue
            rank=(measured['detours'],measured['revisit_connector_m'],objective[1],*key[-3:])
            if best is None or rank<best[0]:
                candidate=order.reconstruct(segments,proposal,candidate_links,check,preserve_endpoint_faces=True,budget=budget)
                verified=balanced._metrics(candidate[0],spacing_m,check)
                if verified!=measured or candidate[2][0]!=objective[0] or abs(candidate[2][1]-objective[1])>EPS:
                    raise ValueError('有界移动预筛与完整重建不一致')
                if _signature(candidate[0],check)!=signature:raise ValueError('有界移动改变原扫描边')
                best=(rank,proposal,candidate_links,candidate,measured)
        if best is None:break
        rank,current,current_links,new_selected,new_values=best
        small=(new_values['detours']==values['detours']
               and values['revisit_connector_m']-new_values['revisit_connector_m']<MIN_GAIN_M
               and selected[2][1]-new_selected[2][1]<MIN_GAIN_M)
        selected,values=new_selected,new_values
        steps.append(dict(source=rank[-3],destination=rank[-2],reverse=rank[-1]))
        if small or tested>=total_attempts:break
    record=dict(version=VERSION,adopted=bool(steps),visits=[list(v) for v in current],steps=steps,
        tested=tested,full_evaluations=full_evaluations,
        budget=dict(total_attempts=total_attempts,per_round=per_round,rounds=rounds,full_per_round=full_per_round),
        baseline=initial_values,selected=values,baseline_objective=list(initial[2]),selected_objective=list(selected[2]),
        motion_limits=deepcopy(motion_limits),semantics='candidate_requires_geometry_and_coverage_replay')
    return selected[0],deepcopy(current_links),selected[1],record


def validate_record(record,segments,baseline_visits,baseline_links,final_links,spacing_m,*,motion_limits,
                    check=lambda:None,budget=200_000,total_attempts=TOTAL_ATTEMPTS,per_round=PER_ROUND,
                    rounds=MAX_ROUNDS,full_per_round=FULL_PER_ROUND):
    expected=dict(total_attempts=total_attempts,per_round=per_round,rounds=rounds,full_per_round=full_per_round)
    if (not isinstance(record,dict) or record.get('version')!=VERSION or record.get('budget')!=expected
            or type(record.get('tested')) is not int or record['tested']>total_attempts
            or type(record.get('full_evaluations')) is not int
            or not 0<=record['full_evaluations']<=min(record['tested'],full_per_round*rounds)
            or not isinstance(record.get('steps'),list) or len(record['steps'])>record['full_evaluations']):
        raise ValueError('有界移动预算记录无效')
    maximum=limits_for(segments,baseline_visits,baseline_links,spacing_m,check=check,budget=budget)
    if any(motion_limits[k]>maximum[k]+EPS for k in maximum):
        raise ValueError('有界移动限制超过允许的基准增量')
    legacy_record={k:v for k,v in record.items() if k not in ('budget','full_evaluations')}
    legacy_record['version']=legacy.LIMITED_DETOUR_VERSION
    result=legacy.validate_extended_record(legacy_record,segments,baseline_visits,baseline_links,final_links,spacing_m,
        motion_limits=motion_limits,attempts=per_round,rounds=rounds,check=check,budget=budget)
    if (record['adopted'] and record['selected']['detours']==record['baseline']['detours']
            and not _motion_improves(record['selected'],record['selected_objective'][1],
                                     record['baseline'],record['baseline_objective'][1])):
        raise ValueError('绕行不变时连接或重访指标变差')
    return result
