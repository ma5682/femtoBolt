"""Independently replay a proof-protected trimmed supplement research route.

All prior evidence is validated first; trim proposals, route geometry and the
new single proof are then rebuilt. Results are research-only, not an installed
plan, clinical coverage claim, or robot command.
"""
from __future__ import annotations

from dataclasses import replace
from femto.planning import (coverage_certificate as coverage,coverage_partition,coverage_workflow,
    fraction_footprint,planning_revision,scan_fraction_seeds,
    supplement_coverage_research,supplement_route_validation,supplement_scan_segments,
    surface_workspace,target_surface,terminal_overlap_research)
from femto.planning.coverage_domain import scan_endpoints

VERSION='research-proof-protected-trimmed-supplement-replay-v1'


def replay(model,region,old_plan,old_certificate,base_plan,previous_recipe,previous_requests,previous_certificate,
           trimmed_recipe,trimmed_requests,trimmed_certificate,*,check=lambda:None,budget=200_000):
    """Revalidate the old proof and recipe before proposing and checking trim."""
    if (not isinstance(previous_recipe,dict) or not isinstance(trimmed_recipe,dict)
            or set(previous_recipe)!={'edges','visits','links'} or set(trimmed_recipe)!={'edges','visits','links'}):
        raise ValueError('删短前后的补线来源不一致')
    if (type(budget) is not int or not 1<=budget<=supplement_route_validation.MAX_POINTS
            or not isinstance(previous_requests,list) or len(previous_requests)>scan_fraction_seeds.MAX_REQUESTS
            or not isinstance(trimmed_requests,list) or len(trimmed_requests)>scan_fraction_seeds.MAX_REQUESTS):
        raise ValueError('删短路线重放参数无效或超过预算')
    for recipe in (previous_recipe,trimmed_recipe):
        if (not isinstance(recipe['edges'],list) or len(recipe['edges'])>supplement_scan_segments.MAX_EDGES
                or not isinstance(recipe['visits'],list) or len(recipe['visits'])>300
                or not isinstance(recipe['links'],list) or len(recipe['links'])>300):
            raise ValueError('删短路线配方无效或超过预算')
        points=0
        for link in recipe['links']:
            check()
            if not isinstance(link,dict) or not isinstance(link.get('samples'),list):raise ValueError('删短路线连接记录无效')
            points+=len(link['samples'])
            if points>budget:raise ValueError('删短路线连接点超过预算')
    if previous_recipe['edges']!=trimmed_recipe['edges']:raise ValueError('删短前后的补线来源不一致')
    def snapshot():
        return planning_revision.digest(dict(old_plan=old_plan,base_plan=base_plan,previous_recipe=previous_recipe,
            previous_requests=previous_requests,trimmed_recipe=trimmed_recipe,trimmed_requests=trimmed_requests))
    check();before=snapshot()
    prior,state=supplement_coverage_research._replay_with_state(model,region,old_plan,old_certificate,base_plan,
        previous_recipe['edges'],previous_recipe['visits'],previous_recipe['links'],previous_requests,previous_certificate,
        check=check,budget=budget)
    domain=state['domain'];model_record=state['current_model'];base=state['base']
    original=state['original'];extra=state['extra'];previous_items=state['items']
    if planning_revision.digest([item.to_dict() for item in previous_items])!=prior['geometry']['items_sha256']:
        raise ValueError('删短前的路线与已验证记录不一致')
    previous=replace(base,ordered_items=previous_items)
    trimmed_base,trim_report=terminal_overlap_research.propose(previous,original,domain.ledger.arrays['region_faces'],
        previous_requests,previous_certificate,check=check)
    trimmed_segments=sorted(trimmed_base+extra,key=lambda s:s.block_index)
    mesh=surface_workspace.canonical_region_mesh(model.store.mesh,region)
    items,statistics,blocks,geometry=supplement_route_validation.reconstruct_and_validate(base,trimmed_segments,
        trimmed_recipe['visits'],trimmed_recipe['links'],mesh,region,check=check,budget=budget)
    final=replace(base,ordered_items=items,statistics=statistics,blocks=blocks)
    positions,position_report=scan_fraction_seeds.propose(final,domain.ledger.arrays['region_faces'],trimmed_requests,check=check)
    seeds=scan_endpoints(final,domain.ledger.arrays['region_faces'])+positions
    route=dict(schema='research-checked-supplement-route-v1',computation_sha256=domain.revision['sha256'],
        source_route=coverage_workflow.route_binding(base),items_sha256=planning_revision.digest([item.to_dict() for item in items]),
        execution_ready=False)
    model_value=fraction_footprint.nominal(model_record['effective_width_mm'])
    coverage.validate(trimmed_certificate,domain,seeds,model_value,route_binding=route,guide=old_certificate,check=check)
    original_comparison=coverage_partition.compare_union(old_certificate,[trimmed_certificate],domain.areas,check)
    previous_comparison=coverage_partition.compare(previous_certificate,trimmed_certificate,domain.areas,check)
    if snapshot()!=before:raise ValueError('删短路线重放期间输入发生变化')
    planning_revision.require_current(model,region,old_plan['treatment_head'],old_plan['planning_context']['revision'])
    planning_revision.require_current(model,region,base_plan['treatment_head'],base_plan['planning_context']['revision'])
    params=base_plan['surface_trajectory']['planner']['parameters']
    target_surface.validate(domain.ledger,base_plan['planning_context']['revision'],evidence=params['quality_evidence'],
                            domain=params['effective_domain'],check=check,reproduce=False)
    for proof in (old_certificate,previous_certificate,trimmed_certificate):coverage._checked_assets(proof,check)
    check()
    return dict(schema=VERSION,prior=prior,trim=trim_report,geometry=geometry,statistics=statistics,positions=position_report,
        old_comparison=original_comparison,previous_comparison=previous_comparison,coverage_summary=trimmed_certificate.summary,
        route=route,installed_route=False,installed_certificate=False,full_target_coverage_claim=False,
        execution_ready=False,physical_accuracy_verified=False)
