"""Bounded ordering over already certified polyline edges; no new surface shortcuts."""
from __future__ import annotations
from typing import Any, Callable

import copy,heapq
import numpy as np
from femto.planning.planning_state import versioned
from femto.planning.route_quality import quality
from femto.planning.coverage_inspection import inspected_serpentine_plan,inspect_coverage,check_document_budget

class OptimizationBudget(ValueError):pass

class Budget:
    def __init__(self,check,limit=2000000):self.check=check;self.limit=limit;self.expanded=0
    def step(self) -> None:
        self.expanded+=1
        if self.expanded%256==0:self.check()
        if self.expanded>self.limit:raise OptimizationBudget('优化搜索超过200万节点展开预算；停止搜索，保留已核验折线')


def key(p: dict[str, Any]) -> tuple[Any, ...]:return (*p['xyz_m'],*p['pixel'],p['triangle_index'])

def reorder_chain(chain: list[dict[str, Any]],budget: Budget,greedy: bool) -> list[dict[str, Any]]:
    nodes={};samples=[];graph=[]
    def node(p):
        k=key(p)
        if k not in nodes:nodes[k]=len(samples);samples.append(p);graph.append({})
        return nodes[k]
    for a,b in zip(chain,chain[1:]):
        i,j=node(a),node(b);d=float(np.linalg.norm(np.subtract(a['xyz_m'],b['xyz_m'])))
        graph[i][j]=d;graph[j][i]=d
    runs=[];run=[]
    for a,b in zip(chain,chain[1:]):
        if b.get('motion_kind','scan')=='scan':
            if not run:run=[node(a)]
            run.append(node(b))
        elif run:runs.append(run);run=[]
    if run:runs.append(run)
    if not runs:return copy.deepcopy(chain)
    if len(runs)>512:raise OptimizationBudget('单链扫描段超过512段排序预算；停止搜索，保留已核验折线')
    def shortest(start,targets):
        distances={start:0.};parents={};queue=[(0.,start)]
        while queue:
            d,i=heapq.heappop(queue)
            if d!=distances[i]:continue
            budget.step()
            if i in targets:
                path=[i]
                while path[-1]!=start:path.append(parents[path[-1]])
                return i,path[::-1]
            for j,w in sorted(graph[i].items()):
                nd=d+w
                if nd<distances.get(j,float('inf')):
                    distances[j]=nd;parents[j]=i;heapq.heappush(queue,(nd,j))
        raise ValueError('原链认证边图不连通，拒绝替代路线')
    output=[copy.deepcopy(chain[0])];current=node(chain[0]);pending=list(range(len(runs)))
    while pending:
        budget.check()
        # Keep the first scan orientation to honor the requested starting preference.
        targets={}
        for index in (pending if greedy and len(pending)!=len(runs) else pending[:1]):
            targets.setdefault(runs[index][0],(index,False))
            if greedy and len(pending)!=len(runs):targets.setdefault(runs[index][-1],(index,True))
        goal,route=shortest(current,targets);index,reverse=targets[goal]
        for i in route[1:]:output.append(dict(samples[i],motion_kind='connector'))
        scan=runs[index][::-1] if reverse else runs[index]
        for i in scan[1:]:output.append(dict(samples[i],motion_kind='scan'))
        current=scan[-1];pending.remove(index)
    _,tail=shortest(current,{node(chain[-1])})
    for i in tail[1:]:output.append(dict(samples[i],motion_kind='connector'))
    return output


def restore_support_edges(original: list[dict[str, Any]],candidate: list[dict[str, Any]],budget: Budget) -> list[dict[str, Any]]:
    """Restore omitted certified edges via explicit round trips, then compare costs."""
    samples={key(p):p for p in original};missing={}
    used={tuple(sorted((key(a),key(b)))) for a,b in zip(candidate,candidate[1:])}
    for a,b in zip(original,original[1:]):
        ka,kb=key(a),key(b)
        if tuple(sorted((ka,kb))) not in used:
            missing.setdefault(ka,set()).add(kb);missing.setdefault(kb,set()).add(ka)
    output=[]
    for p in candidate:
        output.append(p);start=key(p)
        if not missing.get(start):continue
        stack=[start]
        while stack:
            budget.step();node=stack[-1]
            if missing.get(node):
                nxt=min(missing[node]);missing[node].remove(nxt);missing[nxt].remove(node)
                output.append(dict(samples[nxt],motion_kind='connector'));stack.append(nxt)
            else:
                stack.pop()
                if stack:output.append(dict(samples[stack[-1]],motion_kind='connector'))
    if any(missing.values()):raise ValueError('无法在原认证边图中恢复覆盖支持')
    return output


def metrics_not_worse(before: Any,after: Any) -> Any:
    return (abs(before['scan_length_m']-after['scan_length_m'])<=1e-9 and
        before['chain_count']==after['chain_count'] and
        after['connector_length_m']<=before['connector_length_m']+1e-9 and
        after['repeated_edge_length_m']<=before['repeated_edge_length_m']+1e-9 and
        after['total_length_m']<before['total_length_m']-1e-9)


def coverage_not_worse(old: Any,new: Any) -> bool:
    if old['denominator']!=new['denominator'] or len(old['samples'])!=len(new['samples']):return False
    return all((a['triangle_index'],a['sample_index'],a['eligible'])==(b['triangle_index'],b['sample_index'],b['eligible']) and
               (not a['covered'] or b['covered']) for a,b in zip(old['samples'],new['samples']))


@versioned
def optimized_serpentine_plan(projection: Any,mesh: Any,region: Any,width_mm: float=10.,overlap_percent: float=20.,check: Callable[..., Any]=lambda:None,*,boundary_margin_mm: float=1.,direction: str='horizontal',start_corner: str='top_left') -> Any:
    from femto.planning.region_path_guard import constrain_path
    from femto.planning.region_exclusions import allowed_region
    baseline=inspected_serpentine_plan(projection,mesh,region,width_mm,overlap_percent,check,
        boundary_margin_mm=boundary_margin_mm,direction=direction,start_corner=start_corner)
    before,_=quality(baseline,check);chosen=baseline;attempts=[];budget=Budget(check)
    allowed,_=allowed_region(projection,mesh,region,check)
    for name,greedy in [('weighted_connections',False),('nearest_scan_endpoint',True),('nearest_preserve_support',True)]:
        try:
            check();candidate=copy.deepcopy(baseline)
            candidate['segments']=[reorder_chain(c,budget,greedy) for c in baseline['segments']]
            if name=='nearest_preserve_support':
                candidate['segments']=[restore_support_edges(a,b,budget) for a,b in zip(baseline['segments'],candidate['segments'])]
            if sum(map(len,candidate['segments']))>200000:raise OptimizationBudget('候选超过20万点预算；停止搜索，保留已核验折线')
            after,_=quality(candidate,check)
            after['sampled_coverage_percent']=None  # Candidate coverage is not yet recomputed.
            if not metrics_not_worse(before,after):
                attempts.append(dict(method=name,status='not_improved',metrics=after));continue
            verification=copy.deepcopy(candidate)
            guard=constrain_path(projection,mesh,allowed,verification,width_mm,boundary_margin_mm,check)
            if guard['rejected_edges'] or verification['segments']!=candidate['segments']:
                attempts.append(dict(method=name,status='footprint_rejected'));continue
            candidate['boundary_check']=dict(guard,grid_precheck=baseline['boundary_check']['grid_precheck'],accepted_footprint_checked=True)
            candidate['route_order'].update(connector_edges=sum(p.get('motion_kind')=='connector' for c in candidate['segments'] for p in c[1:]),
                optimization=name)
            candidate['coverage'],candidate['coverage_inspection']=inspect_coverage(projection,mesh,region,allowed,candidate,check)
            after,_=quality(candidate,check)
            if not coverage_not_worse(baseline['coverage_inspection'],candidate['coverage_inspection']):
                attempts.append(dict(method=name,status='coverage_regression_rejected',metrics=after,lost_covered_samples=sum(a['covered'] and not b['covered'] for a,b in zip(baseline['coverage_inspection']['samples'],candidate['coverage_inspection']['samples']))));continue
            after,_=quality(candidate,check)
            attempts.append(dict(method=name,status='validated',metrics=after))
            if after['total_length_m']<quality(chosen,check)[0]['total_length_m']-1e-9:chosen=candidate
        except OptimizationBudget as error:
            attempts.append(dict(method=name,status='budget_exceeded',reason=str(error)));break
        except ValueError as error:
            attempts.append(dict(method=name,status='validation_failed',reason=str(error)))
    result=copy.deepcopy(chosen);result['coverage_parameters']['pattern']='serpentine_v4'
    after,turns=quality(result,check)
    result['route_quality']=dict(schema='femto-route-quality-v1',baseline_algorithm='serpentine-v3',
        before=before,after=after,attempts=attempts,status='improved' if chosen is not baseline else 'baseline_retained',
        constraints='same_original_chains_start_end_and_scan_edges; only_traverse_previously_certified_edges; full_footprint_and_each_covered_sample_rechecked',
        budget=dict(max_expansions=budget.limit,expansions=budget.expanded,max_scan_runs_per_chain=512,candidate_methods=3),
        sharp_turns=turns,turn_semantics='deflection_between_adjacent_3d_chords_ge_60_degrees; not_curvature_or_acceleration',
        smoothing=dict(applied=False,reason='本批只标注急转角，未增加曲线平滑；保留经过验证的折线'),
        timing='wall_clock_generation_time_is_recorded_in_external_same_input_benchmark_not_deterministic_binding',execution_ready=False)
    check_document_budget(result,check)
    return result
