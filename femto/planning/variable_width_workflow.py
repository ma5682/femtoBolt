"""P5 opt-in, immutable failed attempts, and restoration of the P4 input."""
from __future__ import annotations

from pathlib import Path
import uuid

from femto.planning import variable_footprint, variable_width_route as route
from femto.planning import planning_revision, planning_state, surface_workspace, coverage_refinement, strict_workflow
from femto.planning.surface_trajectory import SurfaceTrajectory, canonical_json_bytes


def calculate(model, region: dict, plan: dict, profile: dict, check=lambda: None, progress=lambda text: None) -> dict:
    variable_footprint.require_current(profile)
    planning_revision.require_current(model, region, plan['treatment_head'], plan['planning_context']['revision'])
    planning_state.validate(plan, model.store.mesh, region, check)
    original = SurfaceTrajectory.from_dict(plan['surface_trajectory'])
    base = (SurfaceTrajectory.from_dict(original.gap_report['variable_width']['base_trajectory'])
            if original.planner['algorithm_version'] == route.ALGORITHM else original)
    progress('核对原目标与变宽模型，保留严格路线和连接搜索包络')
    ledger = surface_workspace.target_for(model, plan, check, reproduce=True)
    local = surface_workspace.canonical_region_mesh(model.store.mesh, region)
    try:
        progress('检查各段上下界、参考质量与严格表面支持')
        trajectory = route.apply(base, profile, local, region, check)
    except route.quality.PlanningConstraintBlocked as error:
        from femto.planning.planning_io import durable
        check()
        folder = Path(model.reconstruction)/'variable_width_attempts';folder.mkdir(parents=True, exist_ok=True)
        path = folder/('attempt_'+uuid.uuid4().hex+'.json')
        with path.open('xb') as stream:stream.write(canonical_json_bytes(error.report))
        durable(path)
        raise ValueError(str(error)+' 检查记录：'+str(path)) from error
    target = coverage_refinement.rebind_target(ledger, trajectory.planning_context['revision'])
    if target.metadata != trajectory.planning_context['target']:raise ValueError('变宽模型改变了原目标台账')
    planning_revision.require_current(model, region, plan['treatment_head'], trajectory.planning_context['revision'])
    variable_footprint.require_current(profile)
    return strict_workflow._install(model, region, plan, trajectory, target, check)


def restore(model, region: dict, plan: dict, check=lambda: None) -> dict:
    planning_revision.require_current(model, region, plan['treatment_head'], plan['planning_context']['revision'])
    planning_state.validate(plan, model.store.mesh, region, check)
    current = SurfaceTrajectory.from_dict(plan['surface_trajectory'])
    if current.planner['algorithm_version'] != route.ALGORITHM:raise ValueError('当前不是变宽模型版本')
    base = SurfaceTrajectory.from_dict(current.gap_report['variable_width']['base_trajectory'])
    ledger = surface_workspace.target_for(model, plan, check, reproduce=True)
    target = coverage_refinement.rebind_target(ledger, base.planning_context['revision'])
    if target.metadata != base.planning_context['target']:raise ValueError('应用模型前的原目标无法恢复')
    return strict_workflow._install(model, region, plan, base, target, check)
