"""Frozen-input P5 refinement and exact restoration of the P5 proof."""
from __future__ import annotations

from pathlib import Path
import uuid

from femto.planning import variable_width_refinement as refinement, variable_footprint
from femto.planning import coverage_workflow as coverage, planning_revision, planning_state, surface_workspace
from femto.planning.refinement_workflow import _plan
from femto.planning.surface_trajectory import SurfaceTrajectory, canonical_json_bytes, validate_trajectory
from femto.operation_diagnostics import diagnosed


def failed_attempt(model, report, check):
    from femto.planning.planning_io import durable
    check();folder=Path(model.reconstruction)/'variable_refinement_attempts';folder.mkdir(parents=True,exist_ok=True)
    path=folder/('attempt_'+uuid.uuid4().hex+'.json')
    with path.open('xb') as stream:stream.write(canonical_json_bytes(report))
    durable(path);return path


@diagnosed('variable_refinement.operation',operation=True)
def calculate(model,region,plan,check=lambda:None,progress=lambda text:None,*,budget=None):
    profile=plan['surface_trajectory']['planner']['parameters']['variable_footprint']
    variable_footprint.require_current(profile)
    planning_revision.require_current(model,region,plan['treatment_head'],plan['planning_context']['revision'])
    planning_state.validate(plan,model.store.mesh,region,check)
    base=SurfaceTrajectory.from_dict(plan['surface_trajectory'])
    if base.planner['algorithm_version']!=refinement.variable.ALGORITHM:raise ValueError('请先恢复补充前变宽路线')
    certificate=coverage.cached(model,plan)
    if certificate is None:raise ValueError('请先核验变宽模型覆盖')
    progress('核对原目标、变宽模型及补充前完整覆盖证据')
    ledger=surface_workspace.target_for(model,plan,check,reproduce=True)
    domain,seeds,footprint,route=coverage.bundle(base,ledger)
    refinement.coverage.validate(certificate,domain,seeds,footprint,route_binding=route,check=check)
    mesh=surface_workspace.canonical_region_mesh(model.store.mesh,region)
    try:
        trajectory,target,proof=refinement.refine(base,ledger,certificate,mesh,region,budget=budget,check=check,progress=progress)
    except refinement.fixed._Stop as error:
        report=dict(schema=refinement.SCHEMA,status=str(error),base_computation_sha256=base.planning_context['revision']['sha256'],
                    physical_accuracy_verified=False,execution_ready=False)
        path=failed_attempt(model,report,check)
        raise ValueError('变宽补充达到累计检查预算；原方案保留。记录：'+str(path)) from error
    except refinement.quality.PlanningConstraintBlocked as error:
        path=failed_attempt(model,error.report,check)
        raise ValueError(str(error)+' 检查记录：'+str(path)) from error
    if not any(r['accepted'] for r in trajectory.gap_report['refinement']['rounds']):
        path=failed_attempt(model,trajectory.gap_report['refinement'],check)
        raise ValueError('变宽补充未获得可接纳的新增扫描见证；原方案及覆盖保留。记录：'+str(path))
    validate_trajectory(trajectory,mesh=mesh,region=region,check=check)
    planning_revision.require_current(model,region,plan['treatment_head'],trajectory.planning_context['revision'])
    variable_footprint.require_current(profile);check()
    return coverage.attach(model,region,_plan(plan,trajectory),proof,target,check=check)


def restore(model,region,plan,check=lambda:None):
    planning_revision.require_current(model,region,plan['treatment_head'],plan['planning_context']['revision'])
    planning_state.validate(plan,model.store.mesh,region,check)
    current=SurfaceTrajectory.from_dict(plan['surface_trajectory'])
    if current.planner['algorithm_version']!=refinement.ALGORITHM:raise ValueError('当前不是变宽补充版本')
    certificate=coverage.cached(model,plan)
    if not isinstance(certificate,refinement.fixed.RefinedCertificate) or not certificate.history:
        raise ValueError('变宽补充前完整证据缺失')
    base=SurfaceTrajectory.from_dict(current.gap_report['refinement']['base_trajectory'])
    target=refinement.fixed.rebind_target(model.target_ledger,base.planning_context['revision'])
    if target.metadata!=base.planning_context['target']:raise ValueError('变宽补充前原目标无法恢复')
    return coverage.attach(model,region,_plan(plan,base),certificate.history[0],target,check=check)
