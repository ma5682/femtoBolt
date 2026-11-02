"""Proof-protected subsets of scan-end trims for an already checked route.

All results are isolated research candidates. They do not establish physical
coverage, clinical suitability, robot execution or an installable plan.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import math
import numpy as np

from femto.planning import (auto_trim_supplement_research, continuous_quality, coverage_certificate as coverage,
    coverage_partition, coverage_workflow, effective_domain, fraction_footprint, planning_revision,
    row_proximity_research, scan_visit_order, supplement_route_validation, surface_workspace,
    target_surface, terminal_overlap_research)
from femto.planning.coverage_domain import from_plan, scan_endpoints
from femto.planning.surface_trajectory import ScanSegment, SurfaceTrajectory

VERSION='research-protected-trim-subset-v3'
REPLAY_VERSION='research-protected-trim-subset-replay-v1'
MAX_CANDIDATES=32
MAX_PROOFS=4
# Only used within a bounded 10 m coordinate domain and 200k-point budget.
# The slack is deliberately much larger than floating-point roundoff there;
# an inconclusive bound merely leaves the candidate for normal graph checks.
_BOUND_COORDINATE_M=10.
_BOUND_SLACK_M=1e-6


def _route(domain,binding,geometry):
    return dict(schema='research-checked-supplement-route-v1',computation_sha256=domain.revision['sha256'],
        source_route=binding,items_sha256=geometry['items_sha256'],execution_ready=False)


def _prepare(model,region,plan,old_proof,visits,links,prior_proof,check,budget):
    """Replay both source proofs and original/relocated geometry."""
    domain,seeds,footprint,_=from_plan(model,region,plan,check)
    if footprint.get('schema')!='femto-nominal-fixed-ball-v1':raise ValueError('端部子集研究需要固定名义作用宽度')
    base=SurfaceTrajectory.from_dict(plan['surface_trajectory'],check=check)
    binding=coverage_workflow.route_binding(base)
    coverage.validate(old_proof,domain,seeds,footprint,route_binding=binding,check=check)
    reconnection=base.gap_report.get('reconnection') or {}
    segments=[ScanSegment.from_dict(item) for item in reconnection.get('base_items',[]) if item['kind']=='scan_segment']
    if not segments:raise ValueError('端部子集研究缺少原扫描段')
    visits=scan_visit_order.checked_visits(segments,visits)
    mesh=surface_workspace.canonical_region_mesh(model.store.mesh,region)
    items,stats,blocks,geometry=supplement_route_validation.reconstruct_relocated_and_validate(
        base,segments,visits,links,mesh,region,check=check,budget=budget)
    prior=replace(base,ordered_items=items,statistics=stats,blocks=blocks)
    nominal=fraction_footprint.nominal(footprint['effective_width_mm'])
    prior_seeds=scan_endpoints(prior,domain.ledger.arrays['region_faces'])
    coverage.validate(prior_proof,domain,prior_seeds,nominal,route_binding=_route(domain,binding,geometry),guide=old_proof,check=check)
    if not _motion_ok(stats,base.statistics):raise ValueError('端部子集来源路线不满足原路线运动上限')
    return dict(domain=domain,base=base,binding=binding,segments=segments,visits=visits,links=links,mesh=mesh,
                prior=prior,prior_stats=stats,footprint=nominal)


def _motion_ok(candidate,baseline):
    return (candidate['connected_detours']<=baseline['connected_detours']
        and candidate['connector_length_m']<=baseline['connector_length_m']+1e-12
        and candidate['revisit_connector_length_m']<=baseline['revisit_connector_length_m']+1e-12)


def _density_not_worse(candidate,baseline):
    values=candidate['proximity'];reference=baseline['proximity']
    if (not values or len(values)!=len(reference)
            or any(a['radius_m']!=b['radius_m'] for a,b in zip(values,reference))):
        raise ValueError('端部子集近邻采样配置不一致')
    return all(a['near_fraction'] is not None and b['near_fraction'] is not None
               and a['near_fraction']<=b['near_fraction']+1e-12 for a,b in zip(values,reference))


def _viable_rank(value):
    stats=value[6]
    return (stats['connected_detours'],stats['revisit_connector_length_m'],stats['total_length_m'],
            stats['connector_length_m'],value[2])


def _pair_subsets(selected,anchor,trials):
    """Rank partners by their observed connector estimate; this is not admission."""
    def score(item):
        trial=next((value for value in trials if value.get('trim')==[item]),{})
        return trial.get('connector_lower_bound_m',trial.get('connector_length_m',
               trial.get('geometry',{}).get('connector_length_m',math.inf)))
    partners=[(score(item),index,item) for index,item in enumerate(selected) if item!=anchor]
    return [[anchor,item] for _,_,item in sorted(partners,key=lambda x:(x[0],x[1]))]


def _apply(segments,selected,subset):
    if not isinstance(subset,list) or not subset or len(subset)>terminal_overlap_research.MAX_TRIMS:
        raise ValueError('端部删短子集无效')
    first={next(i for i,s in enumerate(segments) if s.block_index==block) for block in dict.fromkeys(s.block_index for s in segments)}
    sides={}
    for entry in subset:
        if entry not in selected or not isinstance(entry,dict):raise ValueError('端部删短不是当前受保护候选')
        index=entry['base_segment'];side=entry['side']
        if index in first and side=='start':raise ValueError('端部删短不能改变工作块起点')
        used=sides.setdefault(index,set())
        if side in used:raise ValueError('端部删短子集包含重复项')
        used.add(side)
    result=[]
    for index,segment in enumerate(segments):
        chosen=sides.get(index,set());left=int('start' in chosen);right=int('end' in chosen)
        points=deepcopy(segment.points[left:len(segment.points)-right if right else None])
        if len(points)<2:raise ValueError('端部删短后扫描段不足两点')
        result.append(replace(segment,points=points))
    return result


def _candidate(state,region,trimmed,visits,links,check,budget):
    base=state['base'];domain=state['domain']
    items,stats,blocks,geometry=supplement_route_validation.reconstruct_relocated_and_validate(
        base,trimmed,visits,links,state['mesh'],region,check=check,budget=budget)
    trajectory=replace(base,ordered_items=items,statistics=stats,blocks=blocks)
    return trajectory,stats,geometry,row_proximity_research.measure(trajectory,check=check)


def _unchanged_endpoints(segments,trimmed,a,ar,b,br):
    oa,na=segments[a].points[0 if ar else -1],trimmed[a].points[0 if ar else -1]
    ob,nb=segments[b].points[-1 if br else 0],trimmed[b].points[-1 if br else 0]
    return all(oa.get(k)==na.get(k) and ob.get(k)==nb.get(k) for k in ('xyz_m','triangle_index','row_index'))


def _prior_link_lengths(segments,visits,links,check):
    """Same per-link arithmetic as anchored_objective; source is validated."""
    result=[]
    for (a,ar),(_, _),link in zip(visits,visits[1:],links):
        check()
        if not link['connected']:result.append(0.);continue
        tail=segments[a].points[0 if ar else -1]
        xyz=np.asarray([tail['xyz_m']]+[point['xyz_m'] for point in link['samples']])
        result.append(float(np.linalg.norm(np.diff(xyz,axis=0),axis=1).sum()))
    return result


def _connector_lower_bound(segments,trimmed,visits,links,lengths,mesh,check):
    """Lower bound using unchanged paths and conservative axis displacement.

    No graph or acceptance decision is established here. Out-of-domain data
    returns None and follows the complete path. The 1 µm slack per changed link
    covers floating-point arithmetic in the bounded coordinate/point domain.
    """
    vertices=np.asarray(mesh.vertices)
    if not np.isfinite(vertices).all() or np.any(np.abs(vertices)>_BOUND_COORDINATE_M):return None
    lower=0.
    for (a,ar),(b,br),link,length in zip(visits,visits[1:],links,lengths):
        check()
        if not link['connected']:continue
        if _unchanged_endpoints(segments,trimmed,a,ar,b,br):part=length
        else:
            tail=np.asarray(trimmed[a].points[0 if ar else -1]['xyz_m'],float)
            head=np.asarray(trimmed[b].points[-1 if br else 0]['xyz_m'],float)
            if (tail.shape!=(3,) or head.shape!=(3,) or not np.isfinite([tail,head]).all()
                    or np.any(np.abs(tail)>_BOUND_COORDINATE_M) or np.any(np.abs(head)>_BOUND_COORDINATE_M)):
                return None
            part=max(0.,float(np.max(np.abs(tail-head)))-_BOUND_SLACK_M)
        if not math.isfinite(part):return None
        lower=max(0.,math.nextafter(lower+part,-math.inf))
    return lower


def _check_end(model,region,plan,visits,links,domain,proofs,snapshot,check):
    if planning_revision.digest(dict(plan=plan,visits=visits,links=links))!=snapshot:
        raise ValueError('端部子集研究期间源路线或连接发生变化')
    planning_revision.require_current(model,region,plan['treatment_head'],plan['planning_context']['revision'])
    params=plan['surface_trajectory']['planner']['parameters']
    target_surface.validate(domain.ledger,plan['planning_context']['revision'],evidence=params['quality_evidence'],
                            domain=params['effective_domain'],check=check,reproduce=False)
    for proof in proofs:coverage._checked_assets(proof,check)
    check()


def _check_budget(max_candidates,max_proofs,motion_prefilter,budget):
    if (type(max_candidates) is not int or not 1<=max_candidates<=MAX_CANDIDATES
            or type(max_proofs) is not int or not 1<=max_proofs<=MAX_PROOFS or type(motion_prefilter) is not bool
            or type(budget) is not int or not 1<=budget<=200_000):raise ValueError('端部子集预算无效')


def build(model,region,plan,old_proof,prior_visits,prior_links,prior_proof,*,max_candidates=17,max_proofs=3,
          proof_budget=None,motion_prefilter=True,check=lambda:None,budget=200_000):
    """Try bounded single trims and prefixes, then independently reprove winners."""
    _check_budget(max_candidates,max_proofs,motion_prefilter,budget)
    snapshot=planning_revision.digest(dict(plan=plan,visits=prior_visits,links=prior_links));check()
    state=_prepare(model,region,plan,old_proof,prior_visits,prior_links,prior_proof,check,budget)
    return _build_prepared(model,region,plan,old_proof,prior_visits,prior_links,prior_proof,state,snapshot,
        max_candidates=max_candidates,max_proofs=max_proofs,proof_budget=proof_budget,
        motion_prefilter=motion_prefilter,check=check,budget=budget)


def _build_prepared(model,region,plan,old_proof,prior_visits,prior_links,prior_proof,state,snapshot,*,
                    max_candidates,max_proofs,proof_budget,motion_prefilter,check,budget):
    """Use only freshly replayed state from the same synchronous operation."""
    _check_budget(max_candidates,max_proofs,motion_prefilter,budget)
    if planning_revision.digest(dict(plan=plan,visits=prior_visits,links=prior_links))!=snapshot:
        raise ValueError('端部子集研究来源在处理前发生变化')
    check()
    domain=state['domain'];base=state['base'];segments=state['segments'];visits=state['visits'];links=state['links']
    prior_density=row_proximity_research.measure(state['prior'],check=check)
    base_density=row_proximity_research.measure(base,check=check)
    _,decision=terminal_overlap_research.propose(state['prior'],segments,domain.ledger.arrays['region_faces'],[],prior_proof,check=check)
    anchors={next(i for i,s in enumerate(segments) if s.block_index==block) for block in dict.fromkeys(s.block_index for s in segments)}
    selected=[item for item in decision['selected'] if not(item['base_segment'] in anchors and item['side']=='start')]
    subsets=[[item] for item in selected]
    prefixes=[selected[:n] for n in (2,3,4,6,8,11,16) if n<=len(selected)]
    params=base.planner['parameters'];width=params['effective_width_mm'];evidence=params['quality_evidence']
    allowed=effective_domain.validate(params['effective_domain'],evidence,width,state['mesh'],region,check)
    prior_lengths=_prior_link_lengths(segments,visits,links,check) if motion_prefilter else None
    viable=[];trials=[];cursor=0;expanded=False;seen=set()
    while len(trials)<max_candidates:
        if cursor>=len(subsets):
            if expanded:break
            expanded=True
            singletons=[value for value in viable if len(value[3])==1]
            if singletons:
                anchor=min(singletons,key=_viable_rank)[3][0]
                subsets.extend(_pair_subsets(selected,anchor,trials))
            subsets.extend(prefixes)
            if cursor>=len(subsets):break
        subset=subsets[cursor];cursor+=1
        key=tuple(sorted((item['base_segment'],item['side']) for item in subset))
        if key in seen:continue
        seen.add(key)
        check();trimmed=_apply(segments,decision['selected'],subset)
        limit=min(state['prior_stats']['connector_length_m'],base.statistics['connector_length_m'])
        if motion_prefilter and limit<=_BOUND_COORDINATE_M:
            lower=_connector_lower_bound(segments,trimmed,visits,links,prior_lengths,state['mesh'],check)
            if lower is not None and lower>limit+1e-12:
                trials.append(dict(trim=deepcopy(subset),reason='connector_lower_bound',connector_lower_bound_m=lower));continue
        class Graph(continuous_quality.LazyVerifiedSurfaceGraph):quarter_portals=True
        graph=Graph(state['mesh'],region,evidence,width,allowed,replace(base,ordered_items=trimmed),check)
        class SafeGraph:
            def __init__(self):
                self.cache={}
                for (a,ar),(b,br),link in zip(visits,visits[1:],links):
                    if not link['connected']:continue
                    if _unchanged_endpoints(segments,trimmed,a,ar,b,br):
                        self.cache[(a,b,ar,br)]=link['samples']
            def route(self,a,b,*,left_reverse=False,right_reverse=False):
                check();key=(a,b,left_reverse,right_reverse)
                return self.cache[key] if key in self.cache else graph.route(a,b,left_reverse=left_reverse,right_reverse=right_reverse)
        candidate_links=scan_visit_order.connect(SafeGraph(),trimmed,visits,check,budget)
        if any(not link['connected'] and segments[a[0]].block_index==segments[b[0]].block_index
               for a,b,link in zip(visits,visits[1:],candidate_links)):
            trials.append(dict(trim=deepcopy(subset),reason='internal_gap'));continue
        if motion_prefilter:
            objective=scan_visit_order.anchored_objective(trimmed,visits,candidate_links,check,budget=budget)
            if objective[0] or objective[1]>limit+1e-12:
                trials.append(dict(trim=deepcopy(subset),reason='motion_prefilter',connector_length_m=objective[1]));continue
        trajectory,stats,geometry,density=_candidate(state,region,trimmed,visits,candidate_links,check,budget)
        improved,motion=auto_trim_supplement_research._assess(prior_density,density,state['prior_stats'],stats,0.)
        base_ok=_density_not_worse(density,base_density)
        accepted=improved and motion and base_ok and _motion_ok(stats,base.statistics)
        trials.append(dict(trim=deepcopy(subset),density_improved=improved,motion_constraints_passed=motion,
                           baseline_constraints_passed=base_ok and _motion_ok(stats,base.statistics),geometry=geometry))
        if accepted:viable.append((stats['connector_length_m'],stats['revisit_connector_length_m'],len(trials),
                                   deepcopy(subset),candidate_links,trajectory,stats,geometry,density))
    result=dict(schema=VERSION,adopted=False,reason='no_qualifying_subset',trim=decision,trials=trials,
                prior_statistics=state['prior_stats'],prior_density=prior_density,installed_route=False,
                installed_certificate=False,full_target_coverage_claim=False,execution_ready=False,physical_accuracy_verified=False)
    selected_proof=None
    for _,_,_,subset,candidate_links,trajectory,stats,geometry,density in sorted(viable,key=_viable_rank)[:max_proofs]:
        check();seeds=scan_endpoints(trajectory,domain.ledger.arrays['region_faces']);route=_route(domain,state['binding'],geometry)
        algorithm=(coverage.RESEARCH_PAGED_FRACTION_REPROOF_ALGORITHM if coverage.checked_page_binding(domain.binding)
                   else coverage.RESEARCH_FRACTION_REPROOF_ALGORITHM)
        proof=coverage.evaluate(domain,seeds,state['footprint'],route_binding=route,guide=old_proof,algorithm=algorithm,budget=proof_budget,check=check)
        coverage.validate(proof,domain,seeds,state['footprint'],route_binding=route,guide=old_proof,check=check)
        baseline=coverage_partition.compare_union(old_proof,[proof],domain.areas,check)
        previous=coverage_partition.compare(prior_proof,proof,domain.areas,check)
        if baseline['retained_subset'] and previous['retained_subset']:
            selected_proof=proof
            result.update(adopted=True,reason='research_candidate_validated',candidate_statistics=stats,candidate_density=density,
                geometry=geometry,baseline_comparison=baseline,previous_comparison=previous,coverage_summary=proof.summary,
                recipe=dict(trim=subset,visits=[list(v) for v in visits],links=deepcopy(candidate_links)))
            break
        result['reason']='proof_retention_not_confirmed'
    _check_end(model,region,plan,prior_visits,prior_links,domain,
               (old_proof,prior_proof,*((selected_proof,) if selected_proof else ())),snapshot,check)
    return result,selected_proof


def replay(model,region,plan,old_proof,prior_visits,prior_links,prior_proof,recipe,proof,*,check=lambda:None,budget=200_000):
    """Independently rederive the trim and replay its single final certificate."""
    if (not isinstance(recipe,dict) or set(recipe)!={'trim','visits','links'}
            or type(budget) is not int or not 1<=budget<=200_000):raise ValueError('端部子集重放参数无效')
    snapshot=planning_revision.digest(dict(plan=plan,visits=prior_visits,links=prior_links))
    recipe_snapshot=planning_revision.digest(recipe);check()
    state=_prepare(model,region,plan,old_proof,prior_visits,prior_links,prior_proof,check,budget)
    if recipe['visits']!=[list(v) for v in state['visits']]:raise ValueError('端部子集改变了已确认访问顺序')
    domain=state['domain'];base=state['base'];segments=state['segments']
    _,decision=terminal_overlap_research.propose(state['prior'],segments,domain.ledger.arrays['region_faces'],[],prior_proof,check=check)
    trimmed=_apply(segments,decision['selected'],recipe['trim'])
    trajectory,stats,geometry,density=_candidate(state,region,trimmed,recipe['visits'],recipe['links'],check,budget)
    before=row_proximity_research.measure(state['prior'],check=check);base_density=row_proximity_research.measure(base,check=check)
    improved,motion=auto_trim_supplement_research._assess(before,density,state['prior_stats'],stats,0.)
    if not improved or not motion or not _motion_ok(stats,base.statistics) or not _density_not_worse(density,base_density):
        raise ValueError('端部子集未满足密度或正式路线运动约束')
    seeds=scan_endpoints(trajectory,domain.ledger.arrays['region_faces']);route=_route(domain,state['binding'],geometry)
    coverage.validate(proof,domain,seeds,state['footprint'],route_binding=route,guide=old_proof,check=check)
    baseline=coverage_partition.compare_union(old_proof,[proof],domain.areas,check)
    previous=coverage_partition.compare(prior_proof,proof,domain.areas,check)
    if not baseline['retained_subset'] or not previous['retained_subset']:raise ValueError('端部子集不能重新确认已确认集合')
    if planning_revision.digest(recipe)!=recipe_snapshot:raise ValueError('端部子集重放期间配方发生变化')
    _check_end(model,region,plan,prior_visits,prior_links,domain,(old_proof,prior_proof,proof),snapshot,check)
    return dict(schema=REPLAY_VERSION,geometry=geometry,statistics=stats,density=density,baseline_comparison=baseline,
                previous_comparison=previous,coverage_summary=proof.summary,installed_route=False,
                installed_certificate=False,full_target_coverage_claim=False,execution_ready=False,physical_accuracy_verified=False)
