"""Prepare a separate saveable geometry revision; preserve all input plans."""
from copy import deepcopy
from dataclasses import replace
import json
from femto.planning import source_anchored_adjustment as adjustment
from femto.planning import planning_revision,planning_state,surface_workspace,target_surface
from femto.planning.surface_trajectory import SurfaceTrajectory,to_player_plan


def _plan(base_plan,trajectory,mesh,region):
    result=deepcopy(base_plan);result.pop('coverage_evidence',None)
    result.update(to_player_plan(trajectory),schema=surface_workspace.SCHEMA,surface_trajectory=trajectory.to_dict())
    result['plan_binding']=planning_state.binding(mesh,region,result)
    return result


def prepare(model,region,base_plan,source_plan,anchors,*,check=lambda:None,progress=lambda text:None):
    """Return a new geometry-only plan; never install old coverage in it.

    The caller keeps its selected/old plan until explicitly installing this
    returned result. A fresh derived snapshot holds its target assets.
    """
    before=planning_revision.digest([base_plan,source_plan,anchors,region])
    for plan in (base_plan,source_plan):
        planning_revision.require_current(model,region,plan['treatment_head'],plan['planning_context']['revision'])
        planning_state.validate_binding(plan,model.store.mesh,region,check)
    progress('从两份原路线重建来源修正并检查全部边')
    base=SurfaceTrajectory.from_dict(base_plan['surface_trajectory'],check=check)
    source=SurfaceTrajectory.from_dict(source_plan['surface_trajectory'],check=check)
    trajectory=adjustment.build(base,source,anchors,surface_workspace.canonical_region_mesh(model.store.mesh,region),region,check)
    from femto.planning.row_proximity_research import measure
    from femto.planning.auto_row_shift_screen_research import _acceptable
    if not _acceptable(source,trajectory,measure(source,check=check),measure(trajectory,check=check)):
        raise ValueError('来源修正没有同时满足路线长度和行间邻近改进条件')
    ledger=surface_workspace.target_for(model,base_plan,check,reproduce=True)
    from femto.planning.coverage_refinement import rebind_target
    target=rebind_target(ledger,trajectory.planning_context['revision'])
    if target.metadata!=trajectory.planning_context['target']:raise ValueError('来源修正改变了原目标资产')
    for plan in (base_plan,source_plan):
        planning_revision.require_current(model,region,plan['treatment_head'],plan['planning_context']['revision'])
    if before!=planning_revision.digest([base_plan,source_plan,anchors,region]):raise ValueError('来源修正期间输入发生变化')
    check();progress('保存新修订的独立输入与原目标，覆盖状态保持待核验')
    manifest_path=surface_workspace.snapshot(model,region,check)
    target_surface.write(manifest_path.parent,target,check)
    from femto.planning.mesh_slice_planner import _source_record
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    trajectory=replace(trajectory,source=dict(trajectory.source,**_source_record(manifest_path,manifest)),data_layers=manifest.get('data_layers'))
    result=_plan(base_plan,trajectory,model.store.mesh,region)
    planning_state.validate(result,model.store.mesh,region,check)
    check()
    if before!=planning_revision.digest([base_plan,source_plan,anchors,region]):raise ValueError('来源修正期间输入发生变化')
    planning_revision.require_current(model,region,result['treatment_head'],result['planning_context']['revision'])
    return result
