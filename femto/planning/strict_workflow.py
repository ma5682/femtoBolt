"""Explicit P4 opt-in and return to the unchanged ordinary route."""
from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import uuid

from femto.planning import strict_path, planning_revision, planning_state, surface_workspace
from femto.planning import coverage_workflow, coverage_refinement
from femto.planning.surface_trajectory import SurfaceTrajectory, to_player_plan


def _install(model,region,plan,trajectory,target,check):
    result=deepcopy(plan)
    result.pop(coverage_workflow.KEY,None)
    result.update(to_player_plan(trajectory),schema=surface_workspace.SCHEMA,surface_trajectory=trajectory.to_dict())
    result['plan_binding']=planning_state.binding(model.store.mesh,region,result)
    planning_state.validate(result,model.store.mesh,region,check)
    check()
    model.target_ledger=target;model.coverage_certificate=None;model.coverage_display=None
    return result


def calculate(model,region,plan,check=lambda: None,progress=lambda text: None,*,budget=None):
    planning_revision.require_current(model,region,plan['treatment_head'],plan['planning_context']['revision'])
    planning_state.validate(plan,model.store.mesh,region,check)
    base=SurfaceTrajectory.from_dict(plan['surface_trajectory'])
    if base.planner['algorithm_version']!=planning_revision.ALGORITHMS['route']:
        raise ValueError('请先恢复常规路线，再进行新的严格构造')
    ledger=surface_workspace.target_for(model,plan,check,reproduce=True)
    local=surface_workspace.canonical_region_mesh(model.store.mesh,region)
    try:trajectory=strict_path.reconstruct(base,local,region,budget=budget,check=check,progress=progress)
    except strict_path.quality.PlanningConstraintBlocked as error:
        # Retain a failed attempt as a separate derived artifact, without
        # touching the active route, old coverage, source mesh or raw RGB-D.
        from femto.planning.surface_trajectory import canonical_json_bytes
        from femto.planning.planning_io import durable
        check()
        folder=Path(model.reconstruction)/'strict_attempts';folder.mkdir(parents=True,exist_ok=True)
        path=folder/('attempt_'+uuid.uuid4().hex+'.json')
        pending=path.with_suffix('.pending')
        try:
            payload=canonical_json_bytes(error.report)
            with pending.open('xb') as stream:stream.write(payload)
            durable(pending);check()
            if path.exists():raise FileExistsError(path)
            os.replace(pending,path)
        finally:pending.unlink(missing_ok=True)
        raise ValueError(str(error)+' 检查记录：'+str(path)) from error
    target=coverage_refinement.rebind_target(ledger,trajectory.planning_context['revision'])
    if target.metadata!=trajectory.planning_context['target']:raise ValueError('严格重建目标台账无法重绑定')
    planning_revision.require_current(model,region,plan['treatment_head'],trajectory.planning_context['revision'])
    return _install(model,region,plan,trajectory,target,check)


def restore(model,region,plan,check=lambda: None):
    planning_revision.require_current(model,region,plan['treatment_head'],plan['planning_context']['revision'])
    planning_state.validate(plan,model.store.mesh,region,check)
    current=SurfaceTrajectory.from_dict(plan['surface_trajectory'])
    if current.planner['algorithm_version']!=strict_path.ALGORITHM:
        raise ValueError('请先恢复补充前路线，再恢复常规路线')
    base=SurfaceTrajectory.from_dict(current.gap_report['surface_support']['base_trajectory'])
    ledger=surface_workspace.target_for(model,plan,check,reproduce=True)
    target=coverage_refinement.rebind_target(ledger,base.planning_context['revision'])
    if target.metadata!=base.planning_context['target']:raise ValueError('常规路线目标证据无法恢复')
    return _install(model,region,plan,base,target,check)
