"""Product operations for new P3 versions and reversible return to the v3 base."""
from __future__ import annotations

from copy import deepcopy

from femto.planning import coverage_workflow as coverage, coverage_refinement as refinement
from femto.planning import surface_workspace, planning_revision, planning_state
from femto.planning.surface_trajectory import SurfaceTrajectory, to_player_plan
from femto.operation_diagnostics import diagnosed


def _plan(original, trajectory):
    result = deepcopy(original)
    result.pop(coverage.KEY, None)
    result.update(to_player_plan(trajectory), schema=surface_workspace.SCHEMA, surface_trajectory=trajectory.to_dict())
    return result


@diagnosed('refinement.operation', operation=True)
def calculate(model, region, plan, check=lambda: None, progress=lambda text: None, *, budget=None,
              candidate_policy='current', coverage_policy='current'):
    if plan['surface_trajectory']['planner']['algorithm_version'] == 'nominal-variable-width-surface-v1':
        from femto.planning.variable_refinement_workflow import calculate as calculate_variable
        if candidate_policy != 'current' or coverage_policy != 'current':raise ValueError('变宽补充使用独立固定版本策略')
        return calculate_variable(model,region,plan,check,progress,budget=budget)
    if candidate_policy == 'current':
        from femto.planning.refinement_candidates_v2 import POLICY
        candidate_policy = deepcopy(POLICY)
    if coverage_policy == 'current':
        from femto.planning.coverage_interleaved import POLICY
        coverage_policy = deepcopy(POLICY)
    planning_revision.require_current(model, region, plan['treatment_head'], plan['planning_context']['revision'])
    planning_state.validate(plan, model.store.mesh, region, check)
    base = SurfaceTrajectory.from_dict(plan['surface_trajectory'])
    if base.planner['algorithm_version'] not in (planning_revision.ALGORITHMS['route'], 'constructed-surface-route-v1'):
        raise ValueError('本补充版本已完成；请先恢复补充前路线')
    certificate = coverage.cached(model, plan)
    if certificate is None:raise ValueError('请先核验模型覆盖，再进行补充规划')
    progress('核对原目标、原路线和覆盖证据')
    ledger = surface_workspace.target_for(model, plan, check, reproduce=True)
    domain, seeds, footprint, route = coverage.bundle(base, ledger)
    refinement.coverage.validate(certificate, domain, seeds, footprint, route_binding=route, check=check)
    trajectory, target, proof = refinement.refine(base, ledger, certificate,
        surface_workspace.canonical_region_mesh(model.store.mesh, region), region,
        budget=budget, check=check, progress=progress, candidate_policy=candidate_policy, coverage_policy=coverage_policy)
    planning_revision.require_current(model, region, plan['treatment_head'], trajectory.planning_context['revision'])
    result = _plan(plan, trajectory)
    if base.planner['algorithm_version'] == 'constructed-surface-route-v1':
        from femto.planning.surface_trajectory import validate_trajectory
        validate_trajectory(trajectory,mesh=surface_workspace.canonical_region_mesh(model.store.mesh,region),region=region,check=check)
    # All work above is local; failed/cancelled preparation preserves old caches.
    check()
    return coverage.attach(model, region, result, proof, target, check=check)


def restore(model, region, plan, check=lambda: None):
    if plan['surface_trajectory']['planner']['algorithm_version'] == 'nominal-variable-width-refined-surface-v1':
        from femto.planning.variable_refinement_workflow import restore as restore_variable
        return restore_variable(model,region,plan,check)
    planning_revision.require_current(model, region, plan['treatment_head'], plan['planning_context']['revision'])
    planning_state.validate(plan, model.store.mesh, region, check)
    certificate = coverage.cached(model, plan)
    if not isinstance(certificate, refinement.RefinedCertificate):raise ValueError('没有可恢复的补充前证据')
    base = SurfaceTrajectory.from_dict(plan['surface_trajectory']['gap_report']['refinement']['base_trajectory'])
    target = refinement.rebind_target(model.target_ledger, base.planning_context['revision'])
    if target.metadata != base.planning_context['target']:raise ValueError('补充前目标资产无法恢复')
    return coverage.attach(model, region, _plan(plan, base), certificate.history[0], target, check=check)
