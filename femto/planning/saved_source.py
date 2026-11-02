"""Shared saved-source verification used by Qt and headless simulation.

The complete original plan, provenance and acquisition gates remain in place.
"""
from __future__ import annotations

import json
from pathlib import Path


def _load_source(path,cancel,progress, *, workspace=None):
    from femto.annotation.fixed_rgb_geometry import FixedRGBModel, GeometryBudget
    from femto.planning.region_project import load_source as load_standard
    from femto.planning.planning_state import binding, region_signature, validate
    path=Path(path).resolve()
    check=GeometryBudget(cancel).check
    if path.is_dir() and (path/'planning.region.zip').is_file():path=path/'planning.region.zip'
    if path.name=='trajectory.json' and (path.parent/'planning.region.zip').is_file():path=path.parent/'planning.region.zip'
    if path.is_file() and path.suffix.lower()=='.zip':
        from femto.planning.region_project import open_project
        return open_project(path,workspace=workspace,cancel=cancel,progress=progress)
    if not path.is_file():return load_standard(path,cancel=cancel,progress=progress)
    if path.stat().st_size>256*1024**2:raise ValueError('历史文件超过读取预算')
    record=json.loads(path.read_text(encoding='utf-8-sig'))
    trajectory=None
    coverage_results=[]
    if record.get('schema')=='femto-surface-trajectory-v1':
        from femto.planning.trajectory_export import load_trajectory
        progress('核对历史三维轨迹、几何与参数')
        trajectory=load_trajectory(path,check=check,reproduce=False,coverage_sink=coverage_results.append)
        manifest=json.loads(Path(trajectory.source['region_manifest']).read_text(encoding='utf-8-sig'))
    elif record.get('schema')=='femto-region-export-v1':manifest=record
    else:raise ValueError('请选择重建目录、区域工程 ZIP、区域清单或 trajectory.json。')
    original=manifest['region']
    model=FixedRGBModel(Path(manifest['source_mesh']).parent,cancel=cancel)
    if model.store.digest!=manifest['geometry_sha256'] or original.get('view_reference')!=model.view_reference:
        raise ValueError('历史记录与原始模型或参考图像不一致')
    region=model.store.normalized_region(dict(original,source_mesh=model.store.source))
    model.store.regions=[region];model.store.mark_saved()
    model.prepare_display(cancel=cancel)
    if trajectory is not None:
        from femto.planning.surface_trajectory import to_player_plan, ORDERED_SELECTION_ALGORITHM
        from femto.planning.treatment_head import load_record,head_parameters,_inputs
        from femto.planning.surface_workspace import SCHEMA
        if trajectory.planner['algorithm_version'] not in (ORDERED_SELECTION_ALGORITHM, 'quality-constrained-surface-v1', 'quality-constrained-surface-v2', 'quality-constrained-surface-v3', 'coverage-refined-surface-v1', 'constructed-surface-route-v1', 'constructed-coverage-refined-surface-v1', 'nominal-variable-width-surface-v1', 'nominal-variable-width-refined-surface-v1'):
            model.plan_warning='此旧路线使用历史算法，已恢复区域；请重新计算当前三维路线。'
        else:
            recorded=load_record(path.parent,trajectory,check=check)
            if recorded is None:
                model.plan_warning='历史路线没有完整治疗头参数，已恢复区域；请设置参数后重新计算。'
            else:
                head=head_parameters(recorded['mode'],**_inputs(recorded))
                plan=to_player_plan(trajectory)
                plan.update(schema=SCHEMA,region_id=region['annotation_id'],region_signature=region_signature(region),
                    coordinate_frame='world_rdf',unit='metre',treatment_head=head,surface_trajectory=trajectory.to_dict())
                plan['plan_binding']=binding(model.mesh,region,plan)
                validate(plan,model.mesh,region)
                if plan.get('planning_context') is not None:
                    import femto.planning.planning_revision as planning_revision
                    import femto.planning.target_surface as target_surface
                    context=plan['planning_context'];params=trajectory.planner['parameters']
                    from femto.dataset.task_context import read_task
                    model.task_context=read_task(Path(model.reconstruction).parent)
                    model.target_ledger=target_surface.read(path.parent,context['target'],context['revision'],
                        evidence=params['quality_evidence'],domain=params['effective_domain'],check=check)
                    planning_revision.require_current(model,region,head,context['revision'])
                else:
                    model.plan_warning='旧记录未建立原目标台账与完整计算修订；原路线语义保留。'
                if coverage_results:
                    from femto.planning.coverage_workflow import attach
                    plan=attach(model,region,plan,coverage_results[0],model.target_ledger,check=check)
                model.saved_plan=plan
    check()
    return model


def verified_capture_source(model, cancel=None):
    """Resolve a capture only when its complete reference binding matches."""
    from femto.annotation.fixed_rgb_geometry import FixedRGBModel, GeometryBudget
    from femto.planning.planning_drafts import source_binding
    check=GeometryBudget(cancel).check
    lineage=getattr(model,'source_lineage',{})
    candidates=[model.store.source]
    if isinstance(lineage,dict) and lineage.get('source_mesh'):candidates.append(lineage['source_mesh'])
    for value in candidates:
        check()
        try:
            root=Path(value).resolve().parent
            if root.name.casefold()!='reconstruction' or not (root/'reconstruction_report.json').is_file():continue
            original=model if Path(model.reconstruction).resolve()==root else FixedRGBModel(root,cancel=cancel)
            if source_binding(original)==source_binding(model):return root
        except (OSError,ValueError,KeyError,TypeError,RuntimeError):
            check()
    return None


def load_source(path,cancel=None,progress=lambda stage:None, *, require_confirmed=True, workspace=None):
    model=_load_source(path,cancel,progress,workspace=workspace)
    model.capture_source=verified_capture_source(model,cancel)
    if model.capture_source:
        from femto.dataset.acquisition_completion import read_completion
        from femto.annotation.fixed_rgb_geometry import GeometryBudget
        completion=read_completion(model.capture_source,check=GeometryBudget(cancel).check,require_confirmed=require_confirmed)
        if completion is not None:model.acquisition_completion=completion
    if not getattr(model,'task_context',None) and model.capture_source:
        from femto.dataset.task_context import read_task
        try:model.task_context=read_task(model.capture_source.parent)
        except (OSError,ValueError,KeyError,TypeError) as error:model.task_warning=str(error)
    from femto.planning.reference_display import recover as recover_display
    from femto.annotation.fixed_rgb_geometry import GeometryBudget
    recover_display(model, GeometryBudget(cancel).check)
    return model
