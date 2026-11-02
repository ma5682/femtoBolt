"""Propose and reprove an optional full-segment trim of a checked research route."""
from __future__ import annotations

from dataclasses import replace
import math
from femto.planning import (auto_supplement_route_research,balanced_visit_improvement,continuous_quality,coverage_certificate as coverage,
    coverage_partition,coverage_workflow,effective_domain,fraction_footprint,fraction_transfer,
    planning_revision,row_proximity_research,supplement_coverage_research,supplement_insertion_research,
    supplement_route_validation,supplement_two_opt_research,surface_workspace,target_surface,terminal_overlap_research)
from femto.planning.coverage_domain import scan_endpoints

VERSION='research-auto-protected-trim-proposal-v1'


def _assess(before_density,after_density,previous_stats,statistics,tolerance):
    before=before_density['proximity'];after=after_density['proximity']
    if (not before or len(before)!=len(after)
            or any(a['radius_m']!=b['radius_m'] for a,b in zip(before,after))):
        raise ValueError('删短前后邻近采样配置不一致')
    old_values=[value['near_fraction'] for value in before]
    new_values=[value['near_fraction'] for value in after]
    density=(all(a is not None and b is not None and b<=a+1e-12 for a,b in zip(old_values,new_values))
             and any(b<a-1e-12 for a,b in zip(old_values,new_values)))
    motion=(statistics['connected_detours']<=previous_stats['connected_detours']
        and statistics['connector_length_m']<=previous_stats['connector_length_m']+tolerance+1e-12
        and statistics['revisit_connector_length_m']<=previous_stats['revisit_connector_length_m']+1e-12)
    return density,motion


def propose(model,region,old_plan,old_certificate,base_plan,previous_recipe,previous_requests,previous_certificate,*,
            proof_budget=None,max_connector_increase_m=0.,check=lambda:None,budget=200_000):
    """Return an admitted research candidate only after geometry and proof replay.

    Sampled 3-D proximity is an optimization heuristic, never physical coverage.
    A rejected proposal does not replace or mutate the previous route or proof.
    """
    if (not isinstance(previous_recipe,dict) or set(previous_recipe)!={'edges','visits','links'}
            or isinstance(max_connector_increase_m,bool) or not isinstance(max_connector_increase_m,(float,int))
            or not math.isfinite(max_connector_increase_m) or not 0<=max_connector_increase_m<=.1):
        raise ValueError('删短研究参数无效')
    def snapshot():
        return planning_revision.digest(dict(old=old_plan,base=base_plan,recipe=previous_recipe,requests=previous_requests))
    check();before=snapshot()
    prior,state=supplement_coverage_research._replay_with_state(model,region,old_plan,old_certificate,base_plan,
        previous_recipe['edges'],previous_recipe['visits'],previous_recipe['links'],previous_requests,previous_certificate,
        check=check,budget=budget)
    base=state['base'];domain=state['domain']
    previous=replace(base,ordered_items=state['items'])
    before_density=row_proximity_research.measure(previous,check=check)
    segments=sorted(state['original']+state['extra'],key=lambda item:item.block_index)
    trimmed,decision=terminal_overlap_research.propose(previous,segments,domain.ledger.arrays['region_faces'],
        previous_requests,previous_certificate,check=check)
    result=dict(schema=VERSION,adopted=False,reason='no_candidates',trim=decision,previous_density=before_density,
        installed_route=False,installed_certificate=False,full_target_coverage_claim=False,
        execution_ready=False,physical_accuracy_verified=False)
    final_proof=None
    if decision['selected']:
        params=base.planner['parameters'];record=params['quality_evidence'];width=params['effective_width_mm']
        spacing=params['row_spacing_mm']/1000
        mesh=surface_workspace.canonical_region_mesh(model.store.mesh,region)
        allowed=effective_domain.validate(params['effective_domain'],record,width,mesh,region,check)
        class Graph(continuous_quality.LazyVerifiedSurfaceGraph):quarter_portals=True
        graph=Graph(mesh,region,record,width,allowed,replace(base,ordered_items=trimmed),check)
        original_ids={id(item) for item in state['original']}
        trimmed_base=[new for old,new in zip(segments,trimmed) if id(old) in original_ids]
        trimmed_extra=[new for old,new in zip(segments,trimmed) if id(old) not in original_ids]
        visits=auto_supplement_route_research._source_visits(base,state['original'])
        chains,links,_,insertion=supplement_insertion_research.propose(graph,trimmed,trimmed_base,trimmed_extra,
            visits,spacing,check=check,budget=budget)
        measured=balanced_visit_improvement._metrics(chains,spacing,check)
        _,links,_,improvement=supplement_two_opt_research.propose(graph,trimmed,insertion['visits'],links,spacing,
            max_detours=measured['detours'],max_revisit_m=measured['revisit_connector_m'],check=check,budget=budget)
        items,statistics,blocks,geometry=supplement_route_validation.reconstruct_and_validate(base,trimmed,
            improvement['visits'],links,mesh,region,check=check,budget=budget)
        final=replace(base,ordered_items=items,statistics=statistics,blocks=blocks)
        after_density=row_proximity_research.measure(final,check=check)
        previous_stats=prior['statistics']
        density_improved,motion_improved=_assess(before_density,after_density,previous_stats,statistics,max_connector_increase_m)
        result.update(reason='geometry_or_proximity_not_improved',candidate_density=after_density,
            candidate_geometry=geometry,candidate_statistics=statistics,density_improved=density_improved,
            motion_constraints_passed=motion_improved,max_connector_increase_m=float(max_connector_increase_m))
        if density_improved and motion_improved:
            requests,positions,transfer=fraction_transfer.transfer(previous,final,domain.ledger.arrays['region_faces'],
                previous_requests,check=check)
            seeds=scan_endpoints(final,domain.ledger.arrays['region_faces'])+positions
            route=dict(schema='research-checked-supplement-route-v1',computation_sha256=domain.revision['sha256'],
                source_route=coverage_workflow.route_binding(base),items_sha256=geometry['items_sha256'],execution_ready=False)
            footprint=fraction_footprint.nominal(state['current_model']['effective_width_mm'])
            algorithm=(coverage.RESEARCH_PAGED_FRACTION_REPROOF_ALGORITHM if coverage.checked_page_binding(domain.binding)
                       else coverage.RESEARCH_FRACTION_REPROOF_ALGORITHM)
            final_proof=coverage.evaluate(domain,seeds,footprint,route_binding=route,guide=old_certificate,
                algorithm=algorithm,budget=proof_budget,check=check)
            coverage.validate(final_proof,domain,seeds,footprint,route_binding=route,guide=old_certificate,check=check)
            retained=coverage_partition.compare(previous_certificate,final_proof,domain.areas,check)
            baseline=coverage_partition.compare_union(old_certificate,[final_proof],domain.areas,check)
            result.update(reason='proof_retention_not_confirmed',previous_comparison=retained,old_comparison=baseline,
                coverage_summary=final_proof.summary,transfer=transfer)
            if retained['retained_subset'] and baseline['retained_subset']:
                result.update(adopted=True,reason='research_candidate_validated',route=route,
                    recipe=dict(edges=previous_recipe['edges'],visits=improvement['visits'],links=links,trim_decision=decision),
                    requests=[list(value) for value in requests])
            else:final_proof=None
    if snapshot()!=before:raise ValueError('删短研究期间输入发生变化')
    for plan,checked_domain in ((old_plan,state['old_domain']),(base_plan,domain)):
        planning_revision.require_current(model,region,plan['treatment_head'],plan['planning_context']['revision'])
        params=plan['surface_trajectory']['planner']['parameters']
        target_surface.validate(checked_domain.ledger,plan['planning_context']['revision'],
            evidence=params['quality_evidence'],domain=params['effective_domain'],check=check,reproduce=False)
    for proof in (old_certificate,previous_certificate):coverage._checked_assets(proof,check)
    if final_proof is not None:coverage._checked_assets(final_proof,check)
    check();return result,final_proof
