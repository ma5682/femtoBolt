"""Prioritize candidate scan positions directly from old proof witnesses.

The old proof and candidate route are checked before proposing work. Returned
proofs remain research artifacts and require independent final-route replay.
"""
from femto.planning import (coverage_certificate as coverage,coverage_partition,coverage_workflow,
    footprint_model,fraction_footprint,fraction_priority,planning_revision,target_surface)
from femto.planning.coverage_domain import from_plan
from femto.planning.coverage_retention_union import _matched_witnesses,_same_target
from femto.planning.surface_trajectory import SurfaceTrajectory

VERSION='research-direct-fraction-witness-v1'


def propose(model,region,old_plan,old_certificate,base_plan,*,budget=None,check=lambda:None,_old_proof_session=None):
    """Generate and replay a fractional candidate proof from unmatched old witnesses."""
    check();snapshot=planning_revision.digest(dict(old=old_plan,base=base_plan,region=region,proof=old_certificate.metadata))
    old_domain,old_seeds,old_width,_=from_plan(model,region,old_plan,check)
    old=SurfaceTrajectory.from_dict(old_plan['surface_trajectory'],check=check);old_route=coverage_workflow.route_binding(old)
    if _old_proof_session is None:
        coverage.validate(old_certificate,old_domain,old_seeds,old_width,route_binding=old_route,check=check)
    else:
        from femto.planning.old_coverage_session_research import _OldCoverageSession
        if type(_old_proof_session) is not _OldCoverageSession:raise ValueError('原证据复用上下文无效')
        _old_proof_session.require(model,region,old_plan,old_certificate,old_domain,old_seeds,old_width,old_route,check)
    domain,endpoints,width,_=from_plan(model,region,base_plan,check)
    base=SurfaceTrajectory.from_dict(base_plan['surface_trajectory'],check=check);route=coverage_workflow.route_binding(base)
    _same_target(old_domain,domain,check)
    if old_width!=width or width.get('schema')!=footprint_model.SCHEMA:
        raise ValueError('直接见证研究需要相同固定名义作用宽度')
    _,_,missing=_matched_witnesses(old_certificate,domain,endpoints,check)
    requests,positions,ranking=fraction_priority.propose_batched(base,old_domain.ledger,old_certificate,missing,check=check)
    nominal=fraction_footprint.nominal(width['effective_width_mm'])
    algorithm=(coverage.RESEARCH_PAGED_FRACTION_REPROOF_ALGORITHM if coverage.checked_page_binding(domain.binding)
               else coverage.RESEARCH_FRACTION_REPROOF_ALGORITHM)
    certificate=coverage.evaluate(domain,endpoints+positions,nominal,route_binding=route,guide=old_certificate,
                                  algorithm=algorithm,budget=budget,check=check)
    coverage.validate(certificate,domain,endpoints+positions,nominal,route_binding=route,guide=old_certificate,check=check)
    comparison=coverage_partition.compare_union(old_certificate,[certificate],domain.areas,check,include_pending_leaves=True)
    for plan,checked in ((old_plan,old_domain),(base_plan,domain)):
        planning_revision.require_current(model,region,plan['treatment_head'],plan['planning_context']['revision'])
        params=plan['surface_trajectory']['planner']['parameters']
        target_surface.validate(checked.ledger,plan['planning_context']['revision'],
            evidence=params['quality_evidence'],domain=params['effective_domain'],check=check,reproduce=False)
    for proof in (old_certificate,certificate):coverage._checked_assets(proof,check)
    check()
    if snapshot!=planning_revision.digest(dict(old=old_plan,base=base_plan,region=region,proof=old_certificate.metadata)):
        raise ValueError('直接见证研究期间输入发生变化')
    return dict(schema=VERSION,requests=[list(r) for r in requests],unmatched_old_leaves=len(missing),
        ranking=ranking,comparison=comparison,installed_route=False,installed_certificate=False,
        full_target_coverage_claim=False,execution_ready=False,physical_accuracy_verified=False,
        semantics='scan_position_proposal_and_independent_fraction_reproof'),certificate
