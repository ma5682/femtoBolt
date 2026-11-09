"""Shared saved-source verification used by Qt and headless simulation.

The complete original plan, provenance and acquisition gates remain in place.
"""
from __future__ import annotations

import json
from pathlib import Path
from femto.operation_diagnostics import diagnosed, record_operation


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
    model=load_standard(Path(manifest['source_mesh']).parent,cancel=cancel,progress=progress)
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
                    # Derived planning directories carry the original task in
                    # their bound record, not in a sibling task.json file.
                    model.task_context=getattr(model,'task_context',None) or read_task(Path(model.reconstruction).parent)
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
    """Resolve a matching capture and report external-reference availability.

    This runtime status is deliberately outside the saved plan/revision: a
    portable bundle remains replayable when its original capture moves. A
    match verifies the reconstruction and fixed reference binding, not a new
    observation of the patient or every raw frame in the original session.
    """
    from femto.annotation.fixed_rgb_geometry import FixedRGBModel, GeometryBudget
    from femto.planning.planning_drafts import source_binding
    from femto.dataset.reconstruction_inputs import verify,InputSourceChanged
    check=GeometryBudget(cancel).check
    lineage=getattr(model,'source_lineage',{})
    candidates=[model.store.source]
    if isinstance(lineage,dict) and lineage.get('source_mesh'):candidates.append(lineage['source_mesh'])
    origin=getattr(model.store,'origin_reference',{}) or {}
    context=getattr(model.store,'local_context',{}) or {}
    expected=bool(origin.get('source_dataset') or context.get('session_id'))
    issues=[];visited=set()
    for value in candidates:
        check()
        try:
            root=Path(value).resolve().parent
            if root.name.casefold()!='reconstruction' or root in visited:continue
            visited.add(root)
            if not (root/'reconstruction_report.json').is_file():
                if expected:issues.append(dict(status='missing',source=str(root),reason='关联重建记录不存在'))
                continue
            expected=True
            dataset=origin.get('source_dataset')
            if dataset and not Path(dataset).is_dir():
                issues.append(dict(status='missing',source=str(dataset),reason='原采集数据目录不存在'))
                continue
            original=model if Path(model.reconstruction).resolve()==root else FixedRGBModel(root,cancel=cancel)
            from femto.planning.patch_reference import derivation as patch_derivation, checked as check_patch
            patch = patch_derivation(model)
            expected_binding = check_patch(patch)['source_binding'] if patch else source_binding(model)
            if source_binding(original)==expected_binding:
                proof=(context.get('reconstruction_scope') or {}).get('input_provenance')
                if proof is not None:verify(proof,check)
                model.capture_source_verification=dict(status='verified',source=str(root),
                    message=('关联重建、固定参考及逐帧输入指纹已核验一致。' if proof is not None
                             else '关联重建与固定参考数据已核验一致；旧记录没有逐帧输入指纹。'),
                    input_files_verified=proof is not None,issues=[])
                return root
            issues.append(dict(status='mismatch',source=str(root),reason='几何、参考 RGB-D、标定或来源绑定与方案不一致'))
        except (OSError,ValueError,KeyError,TypeError,RuntimeError,IndexError,StopIteration) as error:
            check()
            status=('mismatch' if isinstance(error,InputSourceChanged) else
                    'missing' if isinstance(error,FileNotFoundError) else 'unavailable')
            issues.append(dict(status=status,source=str(getattr(error,'path',None) or getattr(error,'filename',None) or value),
                               reason=str(error) or type(error).__name__))
    if not expected:
        status='not_recorded';message='此记录未提供可核验的关联采集来源。'
    else:
        status='mismatch' if any(item['status']=='mismatch' for item in issues) else (
            'missing' if issues and all(item['status']=='missing' for item in issues) else 'unavailable')
        message={'mismatch':'原采集来源与方案不一致。','missing':'原采集来源缺失。',
                 'unavailable':'原采集来源无法核验。'}[status]
        message+='当前记录仍可回看；需恢复匹配来源才能查看关联采集或重新重建。'
        if not issues:issues.append(dict(status=status,source=str(origin.get('source_dataset','')),
                                        reason='未记录可定位的关联重建目录'))
    model.capture_source_verification=dict(status=status,source=None,message=message,issues=issues)
    return None


@diagnosed('load.saved_source')
def load_source(path,cancel=None,progress=lambda stage:None, *, require_confirmed=True, workspace=None, prepared=None):
    from femto.annotation.fixed_rgb_geometry import GeometryBudget
    from femto.dataset.prepared_review import PreparedCapture
    model = prepared.take_model(path, GeometryBudget(cancel).check) if isinstance(prepared, PreparedCapture) else None
    if model is None:
        with record_operation('load.source_geometry'):
            model=_load_source(path,cancel,progress,workspace=workspace)
    with record_operation('load.verify_capture_source'):
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
