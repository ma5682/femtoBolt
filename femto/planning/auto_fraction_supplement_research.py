"""Automatically propose supplemental old scan edges from replayed evidence.

The resulting edge list is a research proposal only. It requires reconnection,
whole-edge validation and a single independently replayed final-route proof.
"""
from __future__ import annotations

from femto.planning import (coverage_certificate as coverage, coverage_partition, coverage_workflow,
    footprint_model, fraction_footprint, fraction_priority, planning_revision, target_surface,
    witness_scan_supplement)
from femto.planning.coverage_domain import from_plan
from femto.planning.coverage_retention_union import _matched_witnesses, _same_target
from femto.planning.surface_trajectory import SurfaceTrajectory

VERSION='research-auto-fraction-supplement-proposal-v1'


def propose(model,region,old_plan,old_certificate,base_plan,normal_certificate,*,
            budget=None,gap_m=.008,check=lambda:None,_old_proof_session=None):
    """Recheck inputs, derive work from the pending set, then reprove and rank edges."""
    result,certificate,_=_propose_with_state(model,region,old_plan,old_certificate,base_plan,normal_certificate,
                                             budget=budget,gap_m=gap_m,check=check,_old_proof_session=_old_proof_session)
    return result,certificate


def _propose_with_state(model,region,old_plan,old_certificate,base_plan,normal_certificate,*,
                        budget=None,gap_m=.008,check=lambda:None,_old_proof_session=None):
    """Return freshly verified intermediates only for the same synchronous operation."""
    check()
    snapshots=(planning_revision.digest(old_plan),planning_revision.digest(base_plan))
    old_domain,old_seeds,old_model,_=from_plan(model,region,old_plan,check)
    old_trajectory=SurfaceTrajectory.from_dict(old_plan['surface_trajectory'],check=check)
    old_route=coverage_workflow.route_binding(old_trajectory)
    if _old_proof_session is None:
        coverage.validate(old_certificate,old_domain,old_seeds,old_model,route_binding=old_route,check=check)
    else:
        from femto.planning.old_coverage_session_research import _OldCoverageSession
        if type(_old_proof_session) is not _OldCoverageSession:raise ValueError('原证据复用上下文无效')
        _old_proof_session.require(model,region,old_plan,old_certificate,old_domain,old_seeds,old_model,old_route,check)
    domain,endpoints,current_model,_=from_plan(model,region,base_plan,check)
    trajectory=SurfaceTrajectory.from_dict(base_plan['surface_trajectory'],check=check)
    route=coverage_workflow.route_binding(trajectory)
    _same_target(old_domain,domain,check)
    if old_model!=current_model or old_model.get('schema')!=footprint_model.SCHEMA:
        raise ValueError('自动补线研究需要相同固定名义作用宽度')
    algorithm=normal_certificate.metadata.get('algorithm')
    guide=(old_certificate if algorithm in coverage.GUIDED_ALGORITHMS
           and algorithm not in coverage.RECALCULATION_ALGORITHMS else None)
    coverage.validate(normal_certificate,domain,endpoints,current_model,route_binding=route,guide=guide,check=check)
    initial=coverage_partition.compare_union(old_certificate,[normal_certificate],domain.areas,check,include_pending_leaves=True)
    requests,positions,ranking=fraction_priority.propose_batched(trajectory,old_domain.ledger,old_certificate,
        initial['pending_old_leaf_indices'],check=check)
    seeds=endpoints+positions;fraction_model=fraction_footprint.nominal(current_model['effective_width_mm'])
    fraction_algorithm=(coverage.RESEARCH_PAGED_FRACTION_REPROOF_ALGORITHM
        if coverage.checked_page_binding(domain.binding) else coverage.RESEARCH_FRACTION_REPROOF_ALGORITHM)
    certificate=coverage.evaluate(domain,seeds,fraction_model,route_binding=route,guide=old_certificate,
        algorithm=fraction_algorithm,budget=budget,check=check)
    coverage.validate(certificate,domain,seeds,fraction_model,route_binding=route,guide=old_certificate,check=check)
    exact=[point for point in seeds if point.get('midpoint_error_bound_m',0.)==0.]
    matched,_,_=_matched_witnesses(old_certificate,domain,exact,check)
    statuses=old_certificate.arrays['status'].copy();statuses[statuses==1]=7;statuses[matched]=1
    view=coverage.Certificate({},dict(old_certificate.arrays,status=statuses))
    comparison=coverage_partition.compare_union(old_certificate,[normal_certificate,certificate,view],domain.areas,
        check,include_pending_leaves=True)
    edges,selection=witness_scan_supplement.propose(old_trajectory,old_domain.ledger.arrays['region_faces'],
        old_certificate,old_domain.areas,comparison['pending_old_leaf_indices'],
        representative_faces=old_domain.representative_faces,gap_m=gap_m,check=check)
    if snapshots!=(planning_revision.digest(old_plan),planning_revision.digest(base_plan)):
        raise ValueError('自动补线研究期间输入路线发生变化')
    for plan,checked_domain in ((old_plan,old_domain),(base_plan,domain)):
        planning_revision.require_current(model,region,plan['treatment_head'],plan['planning_context']['revision'])
        params=plan['surface_trajectory']['planner']['parameters']
        target_surface.validate(checked_domain.ledger,plan['planning_context']['revision'],
            evidence=params['quality_evidence'],domain=params['effective_domain'],check=check,reproduce=False)
    for proof in (old_certificate,normal_certificate,certificate):coverage._checked_assets(proof,check)
    check()
    result=dict(schema=VERSION,old_route=old_route,current_route=route,target_id=domain.revision['target_id'],
        requests=[list(r) for r in requests],selected_edges=[list(edge) for edge in edges],
        initial_comparison=initial,comparison=comparison,ranking=ranking,selection=selection,
        matched_old_leaves=len(matched),installed_route=False,installed_certificate=False,
        full_target_coverage_claim=False,execution_ready=False,physical_accuracy_verified=False)
    return result,certificate,dict(old_domain=old_domain,domain=domain,old=old_trajectory,base=trajectory,
                                   current_model=current_model)
