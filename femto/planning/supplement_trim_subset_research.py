"""Bounded proof-protected end-trim subsets of a checked supplement route.

This is an isolated research proposal. Geometry and both earlier nominal
coverage sets are independently replayed; neither the heuristic nor its
result establishes physical coverage or an executable robot trajectory.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from itertools import combinations

from femto.planning import (auto_trim_supplement_research, continuous_quality,
    coverage_certificate as coverage, coverage_partition, coverage_workflow,
    effective_domain, fraction_footprint, fraction_transfer, planning_revision,
    row_proximity_research, scan_visit_order, supplement_coverage_research,
    supplement_route_validation, surface_workspace, target_surface,
    terminal_overlap_research, trim_subset_research, verified_relocation_research)
from femto.planning.coverage_domain import scan_endpoints

VERSION='research-supplement-protected-trim-subset-v1'
REPLAY_VERSION='research-supplement-protected-trim-subset-replay-v1'
MAX_CANDIDATES=32
MAX_PROOFS=4
MAX_RELOCATIONS=16
RELOCATION_ATTEMPTS=32


def _limits(max_candidates,max_proofs,require_baseline_density,budget):
    if (type(max_candidates) is not int or not 1<=max_candidates<=MAX_CANDIDATES
            or type(max_proofs) is not int or not 1<=max_proofs<=MAX_PROOFS
            or type(require_baseline_density) is not bool
            or type(budget) is not int or not 1<=budget<=200_000):
        raise ValueError('补线端部子集研究预算无效')


def _snapshot(old_plan,base_plan,recipe,requests):
    return planning_revision.digest(dict(old=old_plan,base=base_plan,recipe=recipe,requests=requests))


def _prepare(model,region,old_plan,old_certificate,base_plan,recipe,requests,prior_certificate,check,budget):
    if not isinstance(recipe,dict) or set(recipe)!={'edges','visits','links'}:
        raise ValueError('补线端部子集来源配方无效')
    report,state=supplement_coverage_research._replay_with_state(model,region,old_plan,old_certificate,
        base_plan,recipe['edges'],recipe['visits'],recipe['links'],requests,prior_certificate,check=check,budget=budget)
    if not report['comparison']['retained_subset']:
        raise ValueError('补线端部子集来源未保留原已确认集合')
    base=state['base'];prior=replace(base,ordered_items=state['items'])
    original=state['old_trajectory'];segments=sorted(state['original']+state['extra'],key=lambda item:item.block_index)
    _,decision=terminal_overlap_research.propose(prior,segments,state['domain'].ledger.arrays['region_faces'],
                                                requests,prior_certificate,check=check)
    visits=scan_visit_order.checked_visits(segments,recipe['visits'])
    return dict(state=state,report=report,base=base,prior=prior,original=original,segments=segments,
                visits=visits,decision=decision,mesh=surface_workspace.canonical_region_mesh(model.store.mesh,region),
                prior_density=row_proximity_research.measure(prior,check=check),
                old_density=row_proximity_research.measure(original,check=check))


def _candidate(prepared,region,subset,visits,links,check,budget):
    segments=trim_subset_research._apply(prepared['segments'],prepared['decision']['selected'],subset)
    base=prepared['base']
    items,stats,blocks,geometry=supplement_route_validation.reconstruct_and_validate(
        base,segments,visits,links,prepared['mesh'],region,check=check,budget=budget)
    trajectory=replace(base,ordered_items=items,statistics=stats,blocks=blocks)
    return trajectory,stats,geometry,row_proximity_research.measure(trajectory,check=check)


def _acceptable(prepared,stats,density,require_baseline_density):
    improved,motion=auto_trim_supplement_research._assess(prepared['prior_density'],density,
                                                          prepared['report']['statistics'],stats,0.)
    baseline_motion=trim_subset_research._motion_ok(stats,prepared['original'].statistics)
    baseline_density=trim_subset_research._density_not_worse(density,prepared['old_density'])
    return improved and motion and baseline_motion and (baseline_density or not require_baseline_density),dict(
        density_improved=improved,prior_motion_passed=motion,baseline_motion_passed=baseline_motion,
        baseline_density_passed=baseline_density)


def _end(model,region,old_plan,base_plan,recipe,requests,old_certificate,prior_certificate,final_proof,state,snapshot,check):
    if _snapshot(old_plan,base_plan,recipe,requests)!=snapshot:raise ValueError('补线端部子集研究期间输入发生变化')
    for plan,domain in ((old_plan,state['old_domain']),(base_plan,state['domain'])):
        planning_revision.require_current(model,region,plan['treatment_head'],plan['planning_context']['revision'])
        params=plan['surface_trajectory']['planner']['parameters']
        target_surface.validate(domain.ledger,plan['planning_context']['revision'],evidence=params['quality_evidence'],
                                domain=params['effective_domain'],check=check,reproduce=False)
    for proof in (old_certificate,prior_certificate,*((final_proof,) if final_proof is not None else ())):
        coverage._checked_assets(proof,check)
    check()


def build(model,region,old_plan,old_certificate,base_plan,previous_recipe,previous_requests,previous_certificate,*,
          max_candidates=16,max_proofs=2,max_relocations=0,require_baseline_density=True,proof_budget=None,check=lambda:None,budget=200_000):
    """Try protected singles and pairs, then reprove only the best candidates."""
    _limits(max_candidates,max_proofs,require_baseline_density,budget)
    if type(max_relocations) is not int or not 0<=max_relocations<=MAX_RELOCATIONS:
        raise ValueError('补线端部换序预算无效')
    check();snapshot=_snapshot(old_plan,base_plan,previous_recipe,previous_requests)
    prepared=_prepare(model,region,old_plan,old_certificate,base_plan,previous_recipe,previous_requests,previous_certificate,check,budget)
    state=prepared['state'];segments=prepared['segments'];visits=prepared['visits'];links=previous_recipe['links']
    selected=prepared['decision']['selected']
    anchors={next(i for i,s in enumerate(segments) if s.block_index==block) for block in dict.fromkeys(s.block_index for s in segments)}
    eligible=[item for item in selected if not(item['base_segment'] in anchors and item['side']=='start')]
    subsets=[[x] for x in eligible]+[list(x) for x in combinations(eligible,2)]
    params=prepared['base'].planner['parameters'];width=params['effective_width_mm'];evidence=params['quality_evidence']
    allowed=effective_domain.validate(params['effective_domain'],evidence,width,prepared['mesh'],region,check)
    lengths=trim_subset_research._prior_link_lengths(segments,visits,links,check)
    maximum=min(prepared['report']['statistics']['connector_length_m'],prepared['original'].statistics['connector_length_m'])
    trials=[];viable=[];relocations=0
    for subset in subsets[:max_candidates]:
        check();trimmed=trim_subset_research._apply(segments,selected,subset)
        lower=trim_subset_research._connector_lower_bound(segments,trimmed,visits,links,lengths,prepared['mesh'],check)
        can_relocate=relocations<max_relocations and len(segments)<=verified_relocation_research.MAX_SEGMENTS
        if lower is not None and lower>maximum+1e-12 and not can_relocate:
            trials.append(dict(trim=deepcopy(subset),reason='connector_lower_bound',connector_lower_bound_m=lower));continue
        class Graph(continuous_quality.LazyVerifiedSurfaceGraph):quarter_portals=True
        graph=Graph(prepared['mesh'],region,evidence,width,allowed,replace(prepared['base'],ordered_items=trimmed),check)
        class SafeGraph:
            def __init__(self):
                self.cache={(a,b,ar,br):link['samples'] for (a,ar),(b,br),link in zip(visits,visits[1:],links)
                            if link['connected'] and trim_subset_research._unchanged_endpoints(segments,trimmed,a,ar,b,br)}
            def route(self,a,b,*,left_reverse=False,right_reverse=False):
                check();key=(a,b,left_reverse,right_reverse)
                return self.cache[key] if key in self.cache else graph.route(a,b,left_reverse=left_reverse,right_reverse=right_reverse)
        safe=SafeGraph();initial_links=scan_visit_order.connect(safe,trimmed,visits,check,budget)
        objective=scan_visit_order.anchored_objective(trimmed,visits,initial_links,check,budget=budget)
        candidate_links=initial_links;candidate_visits=visits;decision=None
        if not objective[0] and can_relocate:
            relocations+=1
            prior=prepared['report']['statistics']
            limits=dict(detours=prior['connected_detours'],connector_length_m=prior['connector_length_m'],
                        revisit_connector_m=prior['revisit_connector_length_m'])
            _,proposed,_,record=verified_relocation_research.propose(safe,trimmed,visits,initial_links,
                params['row_spacing_mm']/1000,motion_limits=limits,attempts=RELOCATION_ATTEMPTS,rounds=2,check=check,budget=budget)
            if record['adopted']:
                candidate_links=proposed;candidate_visits=record['visits'];decision=record
                objective=scan_visit_order.anchored_objective(trimmed,candidate_visits,candidate_links,check,budget=budget)
        if objective[0] or objective[1]>maximum+1e-12:
            trials.append(dict(trim=deepcopy(subset),reason='motion_prefilter',connector_length_m=objective[1]));continue
        trajectory,stats,geometry,density=_candidate(prepared,region,subset,candidate_visits,candidate_links,check,budget)
        accepted,flags=_acceptable(prepared,stats,density,require_baseline_density)
        trials.append(dict(trim=deepcopy(subset),reason='viable' if accepted else 'constraints',**flags,
                           statistics=stats,density=density,geometry=geometry))
        if accepted:
            if decision is not None:
                supplement_route_validation.reconstruct_relocated_and_validate(prepared['base'],trimmed,visits,initial_links,
                    prepared['mesh'],region,check=check,budget=budget)
                verified_relocation_research.validate_record(decision,trimmed,visits,initial_links,candidate_links,
                    params['row_spacing_mm']/1000,motion_limits=limits,attempts=RELOCATION_ATTEMPTS,rounds=2,check=check,budget=budget)
            rank=(density['proximity'][0]['near_fraction'],density['proximity'][1]['near_fraction'],
                  stats['connected_detours'],stats['revisit_connector_length_m'],stats['connector_length_m'],len(trials))
            viable.append((rank,subset,candidate_visits,candidate_links,initial_links,decision,trajectory,stats,geometry,density))
    result=dict(schema=VERSION,adopted=False,reason='no_qualifying_subset',trim=prepared['decision'],trials=trials,
                prior_statistics=prepared['report']['statistics'],prior_density=prepared['prior_density'],
                baseline_density=prepared['old_density'],require_baseline_density=require_baseline_density,relocations_tested=relocations,
                installed_route=False,installed_certificate=False,full_target_coverage_claim=False,
                execution_ready=False,physical_accuracy_verified=False)
    proof=None;domain=state['domain']
    for _,subset,candidate_visits,candidate_links,initial_links,decision,trajectory,stats,geometry,density in sorted(viable,key=lambda x:x[0])[:max_proofs]:
        check();requests,positions,transfer=fraction_transfer.transfer(prepared['prior'],trajectory,
            domain.ledger.arrays['region_faces'],previous_requests,check=check)
        seeds=scan_endpoints(trajectory,domain.ledger.arrays['region_faces'])+positions
        route=trim_subset_research._route(domain,coverage_workflow.route_binding(prepared['base']),geometry)
        footprint=fraction_footprint.nominal(state['current_model']['effective_width_mm'])
        algorithm=(coverage.RESEARCH_PAGED_FRACTION_REPROOF_ALGORITHM if coverage.checked_page_binding(domain.binding)
                   else coverage.RESEARCH_FRACTION_REPROOF_ALGORITHM)
        proof=coverage.evaluate(domain,seeds,footprint,route_binding=route,guide=old_certificate,
                                algorithm=algorithm,budget=proof_budget,check=check)
        coverage.validate(proof,domain,seeds,footprint,route_binding=route,guide=old_certificate,check=check)
        baseline=coverage_partition.compare_union(old_certificate,[proof],domain.areas,check)
        previous=coverage_partition.compare(previous_certificate,proof,domain.areas,check)
        if baseline['retained_subset'] and previous['retained_subset']:
            result.update(adopted=True,reason='research_candidate_validated',candidate_statistics=stats,candidate_density=density,
                geometry=geometry,baseline_comparison=baseline,previous_comparison=previous,coverage_summary=proof.summary,
                transfer=transfer,recipe=dict(trim=deepcopy(subset),visits=[list(v) for v in candidate_visits],
                    links=deepcopy(candidate_links),baseline_links=deepcopy(initial_links),relocation=deepcopy(decision),
                    requests=[list(v) for v in requests]))
            break
        proof=None;result['reason']='proof_retention_not_confirmed'
    _end(model,region,old_plan,base_plan,previous_recipe,previous_requests,old_certificate,previous_certificate,
         proof,state,snapshot,check)
    return result,proof


def replay(model,region,old_plan,old_certificate,base_plan,previous_recipe,previous_requests,previous_certificate,
           recipe,proof,*,require_baseline_density=True,check=lambda:None,budget=200_000):
    """Rebuild selected scans, geometry, transferred seeds and both proof sets."""
    _limits(1,1,require_baseline_density,budget)
    if not isinstance(recipe,dict) or set(recipe)!={'trim','visits','links','baseline_links','relocation','requests'}:
        raise ValueError('补线端部子集配方无效')
    check();snapshot=_snapshot(old_plan,base_plan,previous_recipe,previous_requests)
    recipe_snapshot=planning_revision.digest(recipe)
    prepared=_prepare(model,region,old_plan,old_certificate,base_plan,previous_recipe,previous_requests,previous_certificate,check,budget)
    trimmed=trim_subset_research._apply(prepared['segments'],prepared['decision']['selected'],recipe['trim'])
    if recipe['relocation'] is None:
        if recipe['visits']!=[list(v) for v in prepared['visits']] or recipe['links']!=recipe['baseline_links']:
            raise ValueError('补线端部子集无换序记录却改变访问顺序或连接')
    else:
        prior=prepared['report']['statistics'];limits=dict(detours=prior['connected_detours'],
            connector_length_m=prior['connector_length_m'],revisit_connector_m=prior['revisit_connector_length_m'])
        params=prepared['base'].planner['parameters']
        supplement_route_validation.reconstruct_relocated_and_validate(prepared['base'],trimmed,prepared['visits'],
            recipe['baseline_links'],prepared['mesh'],region,check=check,budget=budget)
        verified_relocation_research.validate_record(recipe['relocation'],trimmed,prepared['visits'],recipe['baseline_links'],
            recipe['links'],params['row_spacing_mm']/1000,
            motion_limits=limits,attempts=RELOCATION_ATTEMPTS,rounds=2,check=check,budget=budget)
        if recipe['relocation']['adopted'] is not True or recipe['visits']!=recipe['relocation']['visits']:
            raise ValueError('补线端部子集换序记录不一致')
    trajectory,stats,geometry,density=_candidate(prepared,region,recipe['trim'],recipe['visits'],recipe['links'],check,budget)
    accepted,flags=_acceptable(prepared,stats,density,require_baseline_density)
    if not accepted:raise ValueError('补线端部子集不满足密度或运动限制')
    state=prepared['state'];domain=state['domain']
    requests,positions,transfer=fraction_transfer.transfer(prepared['prior'],trajectory,
        domain.ledger.arrays['region_faces'],previous_requests,check=check)
    if recipe['requests']!=[list(v) for v in requests]:raise ValueError('补线端部子集的边内种子不一致')
    seeds=scan_endpoints(trajectory,domain.ledger.arrays['region_faces'])+positions
    route=trim_subset_research._route(domain,coverage_workflow.route_binding(prepared['base']),geometry)
    footprint=fraction_footprint.nominal(state['current_model']['effective_width_mm'])
    coverage.validate(proof,domain,seeds,footprint,route_binding=route,guide=old_certificate,check=check)
    baseline=coverage_partition.compare_union(old_certificate,[proof],domain.areas,check)
    previous=coverage_partition.compare(previous_certificate,proof,domain.areas,check)
    if not baseline['retained_subset'] or not previous['retained_subset']:
        raise ValueError('补线端部子集未重新确认已有集合')
    if planning_revision.digest(recipe)!=recipe_snapshot:raise ValueError('补线端部子集配方在重放期间发生变化')
    _end(model,region,old_plan,base_plan,previous_recipe,previous_requests,old_certificate,previous_certificate,
         proof,state,snapshot,check)
    return dict(schema=REPLAY_VERSION,geometry=geometry,statistics=stats,density=density,flags=flags,transfer=transfer,
                baseline_comparison=baseline,previous_comparison=previous,coverage_summary=proof.summary,
                installed_route=False,installed_certificate=False,full_target_coverage_claim=False,
                execution_ready=False,physical_accuracy_verified=False)
