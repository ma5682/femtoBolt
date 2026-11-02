"""Replay a source-bound supplement route and its single research proof.

The recipe contains original scan-edge indices, visits, graph link samples and
dyadic requests. Every coordinate is rebuilt from validated source routes;
the final geometry and its research certificate are independently checked.
This remains a research result and is not an installable product certificate.
"""
from __future__ import annotations

from dataclasses import replace
from femto.planning import (coverage_certificate as coverage,coverage_partition,coverage_workflow,
    fraction_footprint,planning_revision,scan_fraction_seeds,supplement_route_validation,
    supplement_scan_segments,surface_workspace,target_surface)
from femto.planning.coverage_domain import from_plan,scan_endpoints
from femto.planning.coverage_retention_union import _same_target
from femto.planning.surface_trajectory import ScanSegment,SurfaceTrajectory

VERSION='research-supplement-single-proof-replay-v1'


def replay(model,region,old_plan,old_certificate,base_plan,edges,visits,links,requests,certificate,*,check=lambda:None,budget=200_000):
    """Validate source inputs, rebuild the final route, then replay its proof."""
    report,_=_replay_with_state(model,region,old_plan,old_certificate,base_plan,edges,visits,links,requests,certificate,
                                check=check,budget=budget)
    return report


def _replay_with_state(model,region,old_plan,old_certificate,base_plan,edges,visits,links,requests,certificate,*,check=lambda:None,budget=200_000):
    """Return validated local intermediates only to this synchronous caller.

    These are not a cross-operation cache. The caller must retain its own input
    snapshots and recheck target assets and revisions before returning.
    """
    if (type(budget) is not int or not 1<=budget<=supplement_route_validation.MAX_POINTS
            or not isinstance(edges,list) or len(edges)>supplement_scan_segments.MAX_EDGES
            or not isinstance(visits,list) or len(visits)>300 or not isinstance(links,list) or len(links)>300
            or not isinstance(requests,list) or len(requests)>scan_fraction_seeds.MAX_REQUESTS):
        raise ValueError('补线路线重放参数无效或超过预算')
    count=0
    for link in links:
        check()
        if not isinstance(link,dict) or not isinstance(link.get('samples'),list):raise ValueError('补线路线连接记录无效')
        count+=len(link['samples'])
        if count>budget:raise ValueError('补线路线连接点超过预算')
    check();old_snapshot=planning_revision.digest(old_plan);base_snapshot=planning_revision.digest(base_plan)
    recipe_snapshot=planning_revision.digest(dict(edges=edges,visits=visits,links=links,requests=requests))
    old_domain,old_seeds,old_model,_=from_plan(model,region,old_plan,check)
    old=SurfaceTrajectory.from_dict(old_plan['surface_trajectory'],check=check)
    coverage.validate(old_certificate,old_domain,old_seeds,old_model,
                      route_binding=coverage_workflow.route_binding(old),check=check)
    domain,_,current_model,_=from_plan(model,region,base_plan,check)
    _same_target(old_domain,domain,check)
    if old_model!=current_model or old_model.get('schema')!='femto-nominal-fixed-ball-v1':
        raise ValueError('补线路线证明需要相同固定名义作用宽度')
    base=SurfaceTrajectory.from_dict(base_plan['surface_trajectory'],check=check)
    reconnection=base.gap_report.get('reconnection') or {}
    original=[ScanSegment.from_dict(item) for item in reconnection.get('base_items',[]) if item['kind']=='scan_segment']
    if not original:raise ValueError('补线路线缺少可重建的原扫描段')
    offset=max(point['row_index'] for segment in original for point in segment.points)+1
    extra,selection=supplement_scan_segments.materialize(old,edges,row_offset=offset,check=check)
    segments=sorted(original+extra,key=lambda s:s.block_index)
    mesh=surface_workspace.canonical_region_mesh(model.store.mesh,region)
    items,statistics,blocks,geometry=supplement_route_validation.reconstruct_and_validate(
        base,segments,visits,links,mesh,region,check=check,budget=budget)
    final=replace(base,ordered_items=items,statistics=statistics,blocks=blocks)
    positions,position_report=scan_fraction_seeds.propose(final,domain.ledger.arrays['region_faces'],requests,check=check)
    seeds=scan_endpoints(final,domain.ledger.arrays['region_faces'])+positions
    route=dict(schema='research-checked-supplement-route-v1',computation_sha256=domain.revision['sha256'],
        source_route=coverage_workflow.route_binding(base),items_sha256=planning_revision.digest([item.to_dict() for item in items]),
        execution_ready=False)
    footprint=fraction_footprint.nominal(current_model['effective_width_mm'])
    coverage.validate(certificate,domain,seeds,footprint,route_binding=route,guide=old_certificate,check=check)
    comparison=coverage_partition.compare_union(old_certificate,[certificate],domain.areas,check,include_pending_leaves=True)
    if (planning_revision.digest(old_plan)!=old_snapshot or planning_revision.digest(base_plan)!=base_snapshot
            or planning_revision.digest(dict(edges=edges,visits=visits,links=links,requests=requests))!=recipe_snapshot):
        raise ValueError('补线路线重放期间输入路线发生变化')
    planning_revision.require_current(model,region,old_plan['treatment_head'],old_plan['planning_context']['revision'])
    planning_revision.require_current(model,region,base_plan['treatment_head'],base_plan['planning_context']['revision'])
    for plan,checked_domain in ((old_plan,old_domain),(base_plan,domain)):
        params=plan['surface_trajectory']['planner']['parameters']
        target_surface.validate(checked_domain.ledger,plan['planning_context']['revision'],
            evidence=params['quality_evidence'],domain=params['effective_domain'],check=check,reproduce=False)
    coverage._checked_assets(old_certificate,check);coverage._checked_assets(certificate,check)
    check()
    report=dict(schema=VERSION,selection=selection,geometry=geometry,statistics=statistics,
        positions=position_report,comparison=comparison,coverage_summary=certificate.summary,
        route=route,installed_route=False,installed_certificate=False,full_target_coverage_claim=False,
        execution_ready=False,physical_accuracy_verified=False)
    return report,dict(domain=domain,old_domain=old_domain,old_trajectory=old,current_model=current_model,
                       base=base,original=original,extra=extra,items=items)
