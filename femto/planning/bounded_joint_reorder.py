"""Bounded joint route proposal; final edges require independent verification.

This module does not select or validate a physical treatment path, establish
coverage, cache graph routes, or register an archive replay algorithm.
"""
from __future__ import annotations

from copy import deepcopy
import math
import numpy as np

from femto.planning.connector_revisit_index import ConnectorRevisitIndex
from femto.planning.endpoint_preserving_detour import _metrics,_signature
from femto.planning.scan_visit_order import anchored_objective,checked_visits,connect,oriented,reconstruct,_clone_points
from femto.planning.selection_blocks import _mark_revisits
from femto.planning.surface_trajectory import canonical_json_bytes

MAX_SEGMENTS=512
MAX_ROUNDS=4
MAX_TRIALS=24
MAX_SPAN=20
MAX_SCORE_CACHE_ENTRIES=4096
MAX_SCORE_CACHE_POINTS=100_000
MAX_SCORE_CACHE_BYTES=32*1024*1024
EPS=1e-10
VERSION='experimental-bounded-joint-reorder-v1'


def _full_anchors(segments,visits,links,check,budget):
    """Validate links and retain every scan head, including same-face heads."""
    objective=anchored_objective(segments,visits,links,check,budget=budget)
    first=oriented(segments[visits[0][0]],visits[0][1]);chains=[(first.block_index,first.points)]
    count=len(first.points);added=0
    for (_, _),(right,reverse),link in zip(visits,visits[1:],links):
        check();following=oriented(segments[right],reverse)
        count+=len(following.points)+len(link['samples'])
        if count>budget:raise ValueError('连续路线采样点超过预算，请缩小选区')
        if link['connected']:
            added+=len(link['samples'])
            chains[-1][1].extend(_clone_points(link['samples']));chains[-1][1].extend(following.points)
        else:chains.append((following.block_index,following.points))
    return chains,added,objective


def _values(chains,spacing,check):
    edge=_metrics(chains,spacing,check)
    historic=_mark_revisits([[dict(p) for p in points] for _,points in chains],spacing,check)
    return dict(detours=edge['detours'],edge_revisit_m=edge['revisit_connector_m'],
                history_revisit_m=historic['connector_length_m'])


def _endpoint(segments,visit,tail):
    segment=segments[visit[0]]
    return segment.points[-1 if (1-int(visit[1]) if tail else int(visit[1])) else 0]


class _ScoreCache:
    """Reuse statistics only for equal returned link records.

    The graph is still called for every candidate. Entries contain independent
    snapshots and cannot bypass graph budgets, cancellations, or final checks.
    """
    def __init__(self,index,check):
        self.index=index;self.check=check;self.entries={};self.points=0;self.bytes=0

    def measure(self,key,tail,head,samples):
        if not MAX_SCORE_CACHE_ENTRIES or not MAX_SCORE_CACHE_POINTS or not MAX_SCORE_CACHE_BYTES:
            return self.index.measure(tail,head,samples)
        if not isinstance(samples,list) or len(samples)>MAX_SCORE_CACHE_POINTS:
            return self.index.measure(tail,head,samples)
        self.check()
        try:fingerprint=canonical_json_bytes((tail,head,samples))
        except (TypeError,ValueError,RecursionError):fingerprint=None
        self.check();previous=self.entries.get(key)
        if fingerprint is not None and previous is not None and fingerprint==previous[0]:
            return dict(previous[1])
        value=self.index.measure(tail,head,samples)
        if previous is not None:
            self.points-=previous[2];self.bytes-=len(previous[0]);del self.entries[key]
        if (fingerprint is not None and len(self.entries)<MAX_SCORE_CACHE_ENTRIES
                and self.points+len(samples)<=MAX_SCORE_CACHE_POINTS
                and self.bytes+len(fingerprint)<=MAX_SCORE_CACHE_BYTES):
            self.entries[key]=(fingerprint,dict(value),len(samples))
            self.points+=len(samples)
            self.bytes+=len(fingerprint)
        return value


def _ranked(segments,visits,links,check):
    candidates=[]
    for first in range(1,len(visits)):
        check();left=visits[first-1];start=visits[first]
        if segments[left[0]].block_index!=segments[start[0]].block_index:continue
        for last in range(first,min(len(visits),first+MAX_SPAN+1)):
            end=visits[last]
            if segments[end[0]].block_index!=segments[start[0]].block_index:break
            if any(not x['connected'] for x in links[first-1:last]):break
            old_left=_endpoint(segments,start,False);old_right=_endpoint(segments,end,True)
            before=_endpoint(segments,left,True)
            old=int(before['row_index']==old_left['row_index'])
            new=int(before['row_index']==old_right['row_index'])
            estimate=float(np.linalg.norm(np.asarray(before['xyz_m'])-old_right['xyz_m'])-
                           np.linalg.norm(np.asarray(before['xyz_m'])-old_left['xyz_m']))
            if last+1<len(visits) and segments[visits[last+1][0]].block_index==segments[start[0]].block_index:
                if not links[last]['connected']:continue
                after=_endpoint(segments,visits[last+1],False)
                old+=int(old_right['row_index']==after['row_index'])
                new+=int(old_left['row_index']==after['row_index'])
                estimate+=float(np.linalg.norm(np.asarray(old_left['xyz_m'])-after['xyz_m'])-
                                np.linalg.norm(np.asarray(old_right['xyz_m'])-after['xyz_m']))
            candidates.append((new-old,estimate,first,last))
    # Evaluate both detour-reducing and length-reducing moves; otherwise a
    # large group of infeasible detour proposals can consume the whole budget.
    negative=sorted((x for x in candidates if x[0]<0),key=lambda x:(x[0],x[1],x[2],x[3]))
    neutral=sorted((x for x in candidates if x[0]==0),key=lambda x:(x[1],x[2],x[3]))
    result=[]
    for i in range(max(len(negative),len(neutral))):
        if i<len(negative):result.append(negative[i])
        if i<len(neutral):result.append(neutral[i])
    return result


def propose(graph,segments,visits,links,spacing,check,budget):
    """Propose bounded non-worsening visit order; caller must check all edges.

    Only complete reconstructed winners are eligible. Both nominal revisit
    rules are bounded by the original route, as are detours, gaps and length.
    The return record is diagnostic and is not a persisted replay record.
    """
    if not segments:raise ValueError('扫描段不能为空')
    current=checked_visits(segments,visits)
    baseline=reconstruct(segments,current,links,check,preserve_endpoint_faces=True,budget=budget)
    original_values=_values(baseline[0],spacing,check)
    signature=_signature(baseline[0],check)
    record=dict(version=VERSION,adopted=False,steps=[],baseline=original_values,baseline_objective=list(baseline[2]))
    if len(current)>MAX_SEGMENTS:return baseline[0],deepcopy(links),baseline[1],record
    index=ConnectorRevisitIndex(segments,spacing,check)
    score_cache=_ScoreCache(index,check)
    current_links=deepcopy(links);best_result=baseline;best_values=original_values
    for _ in range(MAX_ROUNDS):
        best=None
        for _,_,first,last in _ranked(segments,current,current_links,check)[:MAX_TRIALS]:
            check();candidate=list(current)
            candidate[first:last+1]=[(i,not reverse) for i,reverse in reversed(candidate[first:last+1])]
            try:
                checked_visits(segments,candidate)
                candidate_links=connect(graph,segments,candidate,check,budget)
                gaps=0;length=history=edge=0.
                for left,right,link in zip(candidate,candidate[1:],candidate_links):
                    check()
                    if not link['connected']:
                        gaps+=int(segments[left[0]].block_index==segments[right[0]].block_index);continue
                    score=score_cache.measure((left,right),_endpoint(segments,left,True),
                                              _endpoint(segments,right,False),link['samples'])
                    length+=score['length_m'];history+=score['history_revisit_m'];edge+=score['edge_revisit_m']
                if (gaps>baseline[2][0] or length>baseline[2][1]+EPS
                        or history>original_values['history_revisit_m']+EPS or edge>original_values['edge_revisit_m']+EPS):continue
                result=_full_anchors(segments,candidate,candidate_links,check,budget)
            except ValueError as error:
                if str(error)=='连续路线采样点超过预算，请缩小选区':continue
                if str(error)=='连续路线整边搜索超过预算，请缩小选区':
                    record.update(skipped='geometry_budget',steps=[])
                    return baseline[0],deepcopy(links),baseline[1],record
                raise
            values=_values(result[0],spacing,check)
            if (abs(length-result[2][1])>EPS or abs(history-values['history_revisit_m'])>EPS
                    or abs(edge-values['edge_revisit_m'])>EPS):raise ValueError('连接重访统计与完整路线不一致')
            if (_signature(result[0],check)!=signature or result[2][0]>baseline[2][0]
                    or result[2][1]>baseline[2][1]+EPS or values['detours']>original_values['detours']
                    or values['history_revisit_m']>original_values['history_revisit_m']+EPS
                    or values['edge_revisit_m']>original_values['edge_revisit_m']+EPS):continue
            rank=(values['detours'],values['history_revisit_m'],values['edge_revisit_m'],result[2][1],first,last)
            current_rank=(best_values['detours'],best_values['history_revisit_m'],best_values['edge_revisit_m'],best_result[2][1])
            if rank[:4]>=current_rank:continue
            if best is None or rank<best[0]:best=(rank,candidate,candidate_links,result,values)
        if best is None:break
        rank,current,current_links,best_result,best_values=best
        record['steps'].append(dict(first=rank[4],last=rank[5]))
    if not record['steps']:return baseline[0],deepcopy(links),baseline[1],record
    record.update(adopted=True,visits=[list(v) for v in current],selected=best_values,selected_objective=list(best_result[2]))
    return best_result[0],current_links,best_result[1],record


def validate_record(record,segments,visits,initial_links,final_links,spacing,check,budget):
    """Reconstruct recorded steps and bounds; caller independently checks edges.

    This does not rerun the search, prove graph provenance or establish a
    coverage certificate. The caller owns archive and geometry validation.
    """
    if not segments:raise ValueError('扫描段不能为空')
    if not isinstance(record,dict) or type(record.get('adopted')) is not bool:
        raise ValueError('联合换序记录无效')
    keys={'version','adopted','steps','baseline','baseline_objective'}
    skipped=record.get('skipped')
    if skipped is not None:keys.add('skipped')
    if record['adopted']:keys.update(('visits','selected','selected_objective'))
    if (set(record)!=keys or record['version']!=VERSION or not isinstance(record['steps'],list)
            or len(record['steps'])>MAX_ROUNDS or skipped not in (None,'geometry_budget')
            or (skipped is not None and (record['adopted'] or record['steps']))):
        raise ValueError('联合换序记录版本或步骤无效')
    def valid_values(value):
        return (isinstance(value,dict) and set(value)=={'detours','history_revisit_m','edge_revisit_m'}
                and type(value['detours']) is int and value['detours']>=0
                and all(type(value[k]) is float and math.isfinite(value[k]) and value[k]>=0
                        for k in ('history_revisit_m','edge_revisit_m')))
    def valid_objective(value):
        return (isinstance(value,list) and len(value)==2 and type(value[0]) is int and value[0]>=0
                and type(value[1]) is float and math.isfinite(value[1]) and value[1]>=0)
    if (not valid_values(record['baseline']) or not valid_objective(record['baseline_objective'])
            or (record['adopted'] and (not valid_values(record['selected'])
                                      or not valid_objective(record['selected_objective'])))):
        raise ValueError('联合换序记录指标无效')
    current=checked_visits(segments,visits)
    baseline=reconstruct(segments,current,initial_links,check,preserve_endpoint_faces=True,budget=budget)
    initial_values=_values(baseline[0],spacing,check)
    if record['baseline']!=initial_values or record['baseline_objective']!=list(baseline[2]):
        raise ValueError('联合换序基准不能重算')
    if not record['adopted']:
        if record['steps'] or final_links!=initial_links:raise ValueError('未采用的联合换序改变了路线')
        return baseline[0],baseline[1]
    if not record['steps'] or len(current)>MAX_SEGMENTS:raise ValueError('联合换序采用记录无效')
    for step in record['steps']:
        check()
        if (not isinstance(step,dict) or set(step)!={'first','last'}
                or type(step['first']) is not int or type(step['last']) is not int
                or not 1<=step['first']<=step['last']<len(current)
                or step['last']-step['first']>MAX_SPAN):
            raise ValueError('联合换序步骤无效')
        first,last=step['first'],step['last']
        current[first:last+1]=[(i,not reverse) for i,reverse in reversed(current[first:last+1])]
        current=checked_visits(segments,current)
    recorded=checked_visits(segments,record['visits'])
    if recorded!=current:raise ValueError('联合换序顺序与步骤不一致')
    result=_full_anchors(segments,current,final_links,check,budget)
    selected=_values(result[0],spacing,check)
    rank=(selected['detours'],selected['history_revisit_m'],selected['edge_revisit_m'],result[2][1])
    original_rank=(initial_values['detours'],initial_values['history_revisit_m'],initial_values['edge_revisit_m'],baseline[2][1])
    if (_signature(result[0],check)!=_signature(baseline[0],check)
            or result[2][0]>baseline[2][0] or result[2][1]>baseline[2][1]+EPS
            or selected['detours']>initial_values['detours']
            or selected['history_revisit_m']>initial_values['history_revisit_m']+EPS
            or selected['edge_revisit_m']>initial_values['edge_revisit_m']+EPS
            or rank>=original_rank or record['selected']!=selected
            or record['selected_objective']!=list(result[2])):
        raise ValueError('联合换序结果不满足扫描与连接约束')
    return result[0],result[1]
