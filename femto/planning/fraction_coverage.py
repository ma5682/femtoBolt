"""Route-bound, replayable dyadic-seed coverage research; not a product archive.

Only requests, not caller-provided fractional coordinates or error bounds,
are accepted. Each call validates the original route and target and regenerates
the records before computing or replaying the independent nominal proof.
"""
from __future__ import annotations

from femto.planning import coverage_certificate as coverage, coverage_workflow, footprint_model, fraction_footprint, fraction_priority, scan_fraction_seeds
from femto.planning.coverage_domain import from_plan
from femto.planning.coverage_retention_union import _same_target
from femto.planning.surface_trajectory import SurfaceTrajectory


def _inputs(model, region, plan, requests, check):
    domain, endpoints, original_model, _ = from_plan(model, region, plan, check)
    if original_model.get('schema') != 'femto-nominal-fixed-ball-v1':
        raise ValueError('研究分数位置只支持固定名义作用宽度')
    trajectory = SurfaceTrajectory.from_dict(plan['surface_trajectory'], check=check)
    faces = domain.ledger.arrays['region_faces']
    positions, proposal = scan_fraction_seeds.propose(trajectory, faces, requests, check=check)
    declared = fraction_footprint.nominal(original_model['effective_width_mm'])
    return domain, endpoints + positions, declared, coverage_workflow.route_binding(trajectory), proposal


def calculate(model, region, plan, requests, *, budget=None, check=lambda:None):
    """Compute a new research proof with regenerated, route-bound positions."""
    domain, seeds, footprint, route, proposal = _inputs(model, region, plan, requests, check)
    certificate = coverage.evaluate(domain, seeds, footprint, route_binding=route,
                                    budget=budget,
                                    algorithm=(coverage.PAGED_ALGORITHM if coverage.checked_page_binding(domain.binding)
                                               else coverage.CURRENT_ALGORITHM), check=check)
    return certificate, proposal


def replay(model, region, plan, requests, certificate, *, check=lambda:None):
    """Revalidate the route, rebuild seeds and independently replay the proof."""
    domain, seeds, footprint, route, proposal = _inputs(model, region, plan, requests, check)
    coverage.validate(certificate, domain, seeds, footprint, route_binding=route, check=check)
    return proposal


def prioritize(model, region, previous_plan, previous_certificate, plan, pending_leaves, *,
               nearest_edges_per_leaf=2, radius_m=.003, denominator=8, check=lambda:None):
    """Validate both plans and the old proof before ranking current-edge work.

    Pending indices are work hints only; their presence or absence does not
    assert coverage. The returned requests must still be independently proved.
    """
    old_domain, old_seeds, old_footprint, _ = from_plan(model, region, previous_plan, check)
    old_trajectory = SurfaceTrajectory.from_dict(previous_plan['surface_trajectory'], check=check)
    coverage.validate(previous_certificate, old_domain, old_seeds, old_footprint,
                      route_binding=coverage_workflow.route_binding(old_trajectory), check=check)
    domain, _, current_footprint, _ = from_plan(model, region, plan, check)
    _same_target(old_domain, domain, check)
    if old_footprint != current_footprint or old_footprint.get('schema') != footprint_model.SCHEMA:
        raise ValueError('分数位置优先级必须使用相同固定名义作用模型')
    trajectory = SurfaceTrajectory.from_dict(plan['surface_trajectory'], check=check)
    return fraction_priority.propose(trajectory, old_domain.ledger, previous_certificate, pending_leaves,
        nearest_edges_per_leaf=nearest_edges_per_leaf, radius_m=radius_m, denominator=denominator, check=check)


def guided_calculate(model, region, previous_plan, previous_certificate, plan, requests, *, budget=None, check=lambda:None):
    """Recheck an endpoint-only guide, then reprove current fractional seeds."""
    old_domain, old_seeds, old_footprint, _ = from_plan(model, region, previous_plan, check)
    old_trajectory = SurfaceTrajectory.from_dict(previous_plan['surface_trajectory'], check=check)
    old_route = coverage_workflow.route_binding(old_trajectory)
    coverage.validate(previous_certificate, old_domain, old_seeds, old_footprint, route_binding=old_route, check=check)
    domain, seeds, footprint, route, proposal = _inputs(model, region, plan, requests, check)
    _same_target(old_domain, domain, check)
    if old_footprint != footprint_model.nominal(footprint['effective_width_mm']):
        raise ValueError('研究分数位置复算不能改变作用宽度')
    certificate = coverage.evaluate(domain, seeds, footprint, route_binding=route, budget=budget,
        algorithm=(coverage.RESEARCH_PAGED_FRACTION_REPROOF_ALGORITHM
                   if coverage.checked_page_binding(domain.binding)
                   else coverage.RESEARCH_FRACTION_REPROOF_ALGORITHM), guide=previous_certificate, check=check)
    return certificate, proposal


def guided_replay(model, region, previous_plan, previous_certificate, plan, requests, certificate, *, check=lambda:None):
    """Revalidate the old proof and current route before guided proof replay."""
    old_domain, old_seeds, old_footprint, _ = from_plan(model, region, previous_plan, check)
    old_trajectory = SurfaceTrajectory.from_dict(previous_plan['surface_trajectory'], check=check)
    coverage.validate(previous_certificate, old_domain, old_seeds, old_footprint,
                      route_binding=coverage_workflow.route_binding(old_trajectory), check=check)
    domain, seeds, footprint, route, proposal = _inputs(model, region, plan, requests, check)
    _same_target(old_domain, domain, check)
    coverage.validate(certificate, domain, seeds, footprint, route_binding=route, guide=previous_certificate, check=check)
    return proposal


def retention_supplement(model, region, previous_plan, previous_certificate, plan, normal_certificate,
                         fraction_certificate, requests, *, gap_m=.008, check=lambda:None):
    """Propose old scan edges after replaying all route-bound research proofs.

    The partition union is a work-selection aid, not a new certificate. Old
    confirmed leaves with any unconfirmed portion are conservatively selected.
    The caller must still build, check and independently reprove the final route.
    """
    from femto.planning.coverage_partition import compare_union
    from femto.planning.coverage_retention_union import _matched_witnesses
    from femto.planning import planning_revision
    from femto.planning.planning_revision import digest
    from femto.planning.witness_scan_supplement import propose as propose_edges
    check();old_snapshot=digest(previous_plan);new_snapshot=digest(plan)
    old_domain,old_seeds,old_model,_=from_plan(model,region,previous_plan,check)
    old_trajectory=SurfaceTrajectory.from_dict(previous_plan['surface_trajectory'],check=check)
    old_route=coverage_workflow.route_binding(old_trajectory)
    coverage.validate(previous_certificate,old_domain,old_seeds,old_model,route_binding=old_route,check=check)
    domain,endpoints,current_model,_=from_plan(model,region,plan,check)
    trajectory=SurfaceTrajectory.from_dict(plan['surface_trajectory'],check=check)
    route=coverage_workflow.route_binding(trajectory)
    _same_target(old_domain,domain,check)
    if old_model!=current_model or old_model.get('schema')!=footprint_model.SCHEMA:
        raise ValueError('补线证据必须使用相同固定名义作用宽度')
    coverage.validate(normal_certificate,domain,endpoints,current_model,route_binding=route,check=check)
    positions,proposal=scan_fraction_seeds.propose(trajectory,domain.ledger.arrays['region_faces'],requests,check=check)
    fractional_model=fraction_footprint.nominal(current_model['effective_width_mm'])
    fractional_seeds=endpoints+positions
    coverage.validate(fraction_certificate,domain,fractional_seeds,fractional_model,
                      route_binding=route,guide=previous_certificate,check=check)
    exact=[p for p in fractional_seeds if p.get('midpoint_error_bound_m',0.)==0.]
    leaves,_,_=_matched_witnesses(previous_certificate,domain,exact,check)
    statuses=previous_certificate.arrays['status'].copy();statuses[statuses==1]=7;statuses[leaves]=1
    view=coverage.Certificate({},dict(previous_certificate.arrays,status=statuses))
    comparison=compare_union(previous_certificate,[normal_certificate,fraction_certificate,view],domain.areas,
                             check,include_pending_leaves=True)
    pending=comparison['pending_old_leaf_indices']
    edges,selection=propose_edges(old_trajectory,old_domain.ledger.arrays['region_faces'],previous_certificate,
        old_domain.areas,pending,representative_faces=old_domain.representative_faces,gap_m=gap_m,check=check)
    if digest(previous_plan)!=old_snapshot or digest(plan)!=new_snapshot:
        raise ValueError('补线候选计算期间路线发生变化')
    planning_revision.require_current(model,region,previous_plan['treatment_head'],previous_plan['planning_context']['revision'])
    planning_revision.require_current(model,region,plan['treatment_head'],plan['planning_context']['revision'])
    for certificate in (previous_certificate,normal_certificate,fraction_certificate):coverage._checked_assets(certificate,check)
    check()
    return dict(schema='research-replayed-fraction-retention-supplement-v1',
        previous_route=old_route,current_route=route,target_id=domain.revision['target_id'],
        comparison=comparison,matched_old_leaves=len(leaves),fraction_proposal=proposal,
        selected_edges=[list(edge) for edge in edges],selection=selection,
        installed_route=False,installed_certificate=False,full_target_coverage_claim=False,
        execution_ready=False,physical_accuracy_verified=False)
