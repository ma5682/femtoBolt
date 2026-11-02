"""Replay proof-protected trims across base and supplement scan segments."""
from __future__ import annotations

from dataclasses import replace
from femto.planning import (coverage_certificate as coverage,coverage_partition,coverage_workflow,
    fraction_footprint,planning_revision,scan_fraction_seeds,supplement_coverage_research,
    supplement_route_validation,surface_workspace,target_surface,terminal_overlap_research)
from femto.planning.coverage_domain import scan_endpoints

VERSION='research-full-segment-trimmed-supplement-replay-v1'


def replay(model,region,old_plan,old_certificate,base_plan,previous_recipe,previous_requests,previous_certificate,
           trimmed_recipe,trimmed_requests,trimmed_certificate,*,check=lambda:None,budget=200_000):
    """Recheck the prior proof, recompute trims and replay the final proof."""
    if (not isinstance(previous_recipe,dict) or set(previous_recipe)!={'edges','visits','links'}
            or not isinstance(trimmed_recipe,dict) or set(trimmed_recipe)!={'edges','visits','links','trim_decision'}
            or previous_recipe['edges']!=trimmed_recipe['edges']
            or type(budget) is not int or not 1<=budget<=supplement_route_validation.MAX_POINTS
            or not isinstance(trimmed_requests,list) or len(trimmed_requests)>scan_fraction_seeds.MAX_REQUESTS
            or not isinstance(trimmed_recipe['visits'],list) or len(trimmed_recipe['visits'])>300
            or not isinstance(trimmed_recipe['links'],list) or len(trimmed_recipe['links'])>300):
        raise ValueError('完整扫描段删短配方无效或超过预算')
    count=0
    for link in trimmed_recipe['links']:
        check()
        if not isinstance(link,dict) or not isinstance(link.get('samples'),list):raise ValueError('删短连接记录无效')
        count+=len(link['samples'])
        if count>budget:raise ValueError('删短连接点超过预算')
    def snapshot():
        return planning_revision.digest(dict(old=old_plan,base=base_plan,previous=previous_recipe,
            previous_requests=previous_requests,trimmed=trimmed_recipe,trimmed_requests=trimmed_requests))
    check();before=snapshot()
    prior,state=supplement_coverage_research._replay_with_state(model,region,old_plan,old_certificate,base_plan,
        previous_recipe['edges'],previous_recipe['visits'],previous_recipe['links'],previous_requests,previous_certificate,
        check=check,budget=budget)
    domain=state['domain'];base=state['base'];previous_items=state['items']
    if planning_revision.digest([x.to_dict() for x in previous_items])!=prior['geometry']['items_sha256']:
        raise ValueError('删短前的路线与验证记录不一致')
    previous=replace(base,ordered_items=previous_items)
    segments=sorted(state['original']+state['extra'],key=lambda segment:segment.block_index)
    trimmed,decision=terminal_overlap_research.propose(previous,segments,domain.ledger.arrays['region_faces'],
        previous_requests,previous_certificate,check=check)
    if decision!=trimmed_recipe['trim_decision']:raise ValueError('删短决定无法重现')
    mesh=surface_workspace.canonical_region_mesh(model.store.mesh,region)
    items,statistics,blocks,geometry=supplement_route_validation.reconstruct_and_validate(base,trimmed,
        trimmed_recipe['visits'],trimmed_recipe['links'],mesh,region,check=check,budget=budget)
    final=replace(base,ordered_items=items,statistics=statistics,blocks=blocks)
    positions,position_report=scan_fraction_seeds.propose(final,domain.ledger.arrays['region_faces'],trimmed_requests,check=check)
    seeds=scan_endpoints(final,domain.ledger.arrays['region_faces'])+positions
    route=dict(schema='research-checked-supplement-route-v1',computation_sha256=domain.revision['sha256'],
        source_route=coverage_workflow.route_binding(base),items_sha256=planning_revision.digest([x.to_dict() for x in items]),
        execution_ready=False)
    footprint=fraction_footprint.nominal(state['current_model']['effective_width_mm'])
    coverage.validate(trimmed_certificate,domain,seeds,footprint,route_binding=route,guide=old_certificate,check=check)
    baseline=coverage_partition.compare_union(old_certificate,[trimmed_certificate],domain.areas,check)
    previous_comparison=coverage_partition.compare(previous_certificate,trimmed_certificate,domain.areas,check)
    if snapshot()!=before:raise ValueError('完整扫描段删短重放期间输入发生变化')
    for plan,checked_domain in ((old_plan,state['old_domain']),(base_plan,domain)):
        planning_revision.require_current(model,region,plan['treatment_head'],plan['planning_context']['revision'])
        params=plan['surface_trajectory']['planner']['parameters']
        target_surface.validate(checked_domain.ledger,plan['planning_context']['revision'],
            evidence=params['quality_evidence'],domain=params['effective_domain'],check=check,reproduce=False)
    for proof in (old_certificate,previous_certificate,trimmed_certificate):coverage._checked_assets(proof,check)
    check()
    return dict(schema=VERSION,prior=prior,trim=decision,geometry=geometry,statistics=statistics,positions=position_report,
        old_comparison=baseline,previous_comparison=previous_comparison,coverage_summary=trimmed_certificate.summary,
        route=route,installed_route=False,installed_certificate=False,full_target_coverage_claim=False,
        execution_ready=False,physical_accuracy_verified=False)
