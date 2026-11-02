"""Transactional continuation of the original target, retaining the previous plan."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import json

from femto.planning import target_surface, target_paging, surface_workspace, planning_revision, planning_state
from femto.planning.surface_trajectory import SurfaceTrajectory, canonical_json_bytes


def available(plan):
    context=(plan or {}).get('planning_context')
    if context is None:return False
    algorithm=plan['surface_trajectory']['planner']['algorithm_version']
    # Derived routes carry their own frozen base trajectory and target headers.
    # Continue the ordinary base first, then reconstruct any strict/width model.
    if algorithm != 'quality-constrained-surface-v3':
        return False
    return context['target']['metrics']['stop_reason'] in ('unit_budget','pair_budget','time_budget')


def calculate(model,region,plan,output,check=lambda:None,progress=lambda text:None,*,budget=None):
    """Keep the route, archive old evidence, and invalidate coverage of the shorter inventory."""
    if not available(plan):raise ValueError('当前目标无需续算，或需先恢复补充前路线')
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('续算结果必须保存到新目录')
    context=plan['planning_context'];revision=context['revision']
    planning_revision.require_current(model,region,plan['treatment_head'],revision)
    planning_state.validate(plan,model.store.mesh,region,check)
    params=plan['surface_trajectory']['planner']['parameters']
    quality,domain=params['quality_evidence'],params['effective_domain']
    progress('复核已有目标分块，保留原目标及旧方案证据')
    ledger=surface_workspace.target_for(model,plan,check,reproduce=False)
    if ledger.metadata['schema']==target_surface.PAGED_SCHEMA:
        bundle=target_paging.reproduce_materialized(ledger,revision,evidence=quality,domain=domain,check=check)
    else:bundle=target_paging.begin(ledger,revision,evidence=quality,domain=domain,check=check)
    progress('从上次停止的像素组续算下一块；每块保持原单元和工作预算')
    continued=target_paging.resume(bundle,revision,evidence=quality,domain=domain,budget=budget,check=check)
    result_ledger=target_paging.materialize(continued,revision)
    return store_result(model,region,plan,ledger,result_ledger,revision,output,check,progress,
                        operation='continuation')


def store_result(model,region,plan,ledger,result_ledger,revision,output,check,progress,*,operation):
    """Publish a new calculation with the old complete plan and evidence alongside."""
    from femto.planning.planning_io import staging_directory, publish
    from femto.planning import coverage_workflow, coverage_archive
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('目标计算结果必须保存到新目录')
    result=deepcopy(plan)
    result.pop(coverage_workflow.KEY,None)
    result['planning_context']=dict(plan['planning_context'],revision=revision,target=result_ledger.metadata)
    manifest_path=surface_workspace.snapshot(model,region,check)
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    target_surface.write(manifest_path.parent,result_ledger,check)
    from femto.planning.mesh_slice_planner import _source_record
    original=SurfaceTrajectory.from_dict(plan['surface_trajectory'])
    trajectory=replace(original,planning_context=result['planning_context'],
        source=dict(original.source,**_source_record(manifest_path,manifest)),data_layers=manifest.get('data_layers'))
    result['surface_trajectory']=trajectory.to_dict()
    result['plan_binding']=planning_state.binding(model.store.mesh,region,result)
    surface_workspace.validate(result,model.store.mesh,region,check)
    previous=coverage_workflow.cached(model,plan)
    progress('保存旧方案和新分块目标；新目标需要重新核验覆盖')
    output.parent.mkdir(parents=True,exist_ok=True)
    with staging_directory(output.parent,'.target-continuation-') as stage:
        old=stage/'previous';old.mkdir()
        target_surface.write(old,ledger,check)
        (old/'plan.json').write_bytes(canonical_json_bytes(plan))
        if previous is not None:coverage_archive.write_assets(old,previous,check)
        target_surface.write(stage,result_ledger,check)
        (stage/'plan.json').write_bytes(canonical_json_bytes(result))
        (stage/'continuation.json').write_bytes(canonical_json_bytes(dict(
            schema='femto-original-target-continuation-v1',operation=operation,target_id=revision['target_id'],
            computation_sha256=revision['sha256'],previous_plan_sha256=planning_revision.digest(plan),
            previous_target_id=ledger.metadata['target_id'],
            previous_target_sha256=ledger.metadata['sha256'],target_sha256=result_ledger.metadata['sha256'],
            route_result_sha256=trajectory.to_dict()['result_sha256'],
            previous_coverage_sha256=None if previous is None else previous.metadata['sha256'],
            coverage_status='requires_recalculation_for_extended_original_target',
            execution_ready=False,physical_accuracy_verified=False)))
        check()
        publish(lambda:stage.rename(output),check,None)
    return result,result_ledger
