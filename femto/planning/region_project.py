"""Portable region/planning workspace. ZIP assets are verified before extraction."""
from __future__ import annotations
from typing import Any, Callable

from femto.paths import OUTPUTS_ROOT
import json
import shutil
import uuid
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image
from femto.io_utils import JsonSizeLimitError, write_json, write_json_strict, sha256_file
from femto.operation_diagnostics import diagnosed
from femto.checksum_batch import sha256_files

SCHEMA='femto-region-project-v1'
MAX_PROJECT_JSON_BYTES = 100 * 1024**2
ASSETS={'scene_tsdf_mesh_m.ply','rgb.png','display_rgb.png','depth.npy','image_valid.npy',
        'mesh_triangle_uv.npy','texture_multiview.png','project.json',
        'target_surface.json','target_surface.npz','coverage_certificate.json','coverage_certificate.npz'}
from femto.planning.refinement_archive import ASSETS as REFINEMENT_ASSETS
from femto.planning.coverage_archive import HINT_ASSETS
ASSETS |= REFINEMENT_ASSETS | HINT_ASSETS


def digest(path: str | Path, check: Callable[[], None] | None = None) -> str:
    return sha256_file(path, check)


def _project_asset_hashes(root: Path, check: Callable[[], None]) -> dict[str, str]:
    paths = tuple(root.iterdir())
    batch = sha256_files(paths, check, hasher=digest)
    return {path.name: batch.hashes[path] for path in paths}


@diagnosed('portable.manifest')
def _write_manifest(path, payload, check, *, allow_nan=False):
    try:
        write_json(path, payload, indent=None, ensure_ascii=False, allow_nan=allow_nan,
                   check=check, max_bytes=MAX_PROJECT_JSON_BYTES)
    except JsonSizeLimitError as error:
        raise ValueError('完整工程清单超过 100 MiB 可回读预算；原方案未覆盖，当前内容仍保留') from error


@diagnosed('portable.archive')
def _pack_assets(root, check):
    from femto.planning.planning_io import copy_stream, durable
    archive=root/'bundle.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for file in root.iterdir():
            if file.name!='bundle.zip':
                with file.open('rb') as source,z.open(file.name,'w') as target:copy_stream(source,target,check)
    durable(archive)
    return archive


@diagnosed('portable.validation_manifest')
def _write_validation_plan(path, plan, check):
    write_json_strict(path, plan, indent=None, ensure_ascii=False, check=check)


@diagnosed('portable.save', operation=True)
def save_project(model: Any,region: Any,path: str | Path,plan: Any=None,check: Callable[..., Any]=lambda:None,*,planning_parameters: Any=None,progress: Callable[..., Any]=lambda stage:None,commit: Any=None,expected_digest: str | None=None,catalog_root: str | Path | None=None,allow_replace: bool=True,verified_assets=None,target_session=None) -> Path:
    options = dict(planning_parameters=planning_parameters, progress=progress, commit=commit,
                   expected_digest=expected_digest, catalog_root=catalog_root,
                   allow_replace=allow_replace, verified_assets=verified_assets)
    if target_session is None:
        return _save_project(model, region, path, plan, check, **options)
    from femto.planning.save_validation import SaveValidationSession
    from femto.planning.surface_workspace import target_for
    if type(target_session) is not SaveValidationSession or not (plan or {}).get('planning_context'):
        raise ValueError('当前方案不能使用原目标保存验证会话')
    # Rehash the actual cached/loaded target before consuming the one-use receipt.
    try:
        progress('复核本次保存的原目标验证结果')
        ledger = target_for(model, plan, check, reproduce=False)
        with target_session.consume(model, region, plan, ledger, check):
            return _save_project(model, region, path, plan, check, target_session=target_session, **options)
    finally:
        target_session.close()


def _save_project(model, region, path, plan, check, *, planning_parameters=None,
                  progress=lambda stage:None, commit=None, expected_digest=None,
                  catalog_root=None, allow_replace=True, verified_assets=None, target_session=None):
    """Keep full source geometry as occlusion context; select only one editable ROI."""
    from femto.annotation.scene_annotations import mesh_digest
    from femto.planning.region_path import signature
    from femto.planning.planning_io import publish,copy_stream,windows_file_retry,staging_directory
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    progress('核验保存目标与方案来源')
    baseline=digest(path,check) if path.exists() else None
    if baseline is not None and not allow_replace:raise FileExistsError('保存新版本需要新的文件名；原版本不会覆盖')
    if expected_digest is not None and baseline!=expected_digest:raise ValueError('方案文件已被其他窗口或工具修改，请另存为或重新打开')
    if path.exists():
        with zipfile.ZipFile(path) as existing:
            previous=json.loads(existing.read('project.json'))
        if previous.get('schema')!=SCHEMA or previous.get('region',{}).get('annotation_id')!=region['annotation_id']:
            raise ValueError('目标文件属于另一个区域，禁止覆盖；请选择新的另存为路径')
    from femto.planning.planning_state import parameters
    if planning_parameters is None and plan is not None:planning_parameters=parameters(plan)
    if plan is not None and planning_parameters!=parameters(plan):raise ValueError('Current parameters differ from path')
    region=model.store.normalized_region(region)
    if region.get('view_reference')!=model.view_reference:raise ValueError('区域与参考图像来源不匹配，未保存方案')
    from femto.dataset.acquisition_completion import (
        read_completion,portable_completion,validate_portable_completion)
    acquisition=getattr(model,'portable_acquisition',None)
    if acquisition is not None:
        validate_portable_completion(acquisition,model.store.digest,model.view_reference)
    else:
        completion=read_completion(getattr(model,'reconstruction',Path(model.store.source).parent),
                                   check=check,require_confirmed=True)
        if completion is not None:
            acquisition=portable_completion(completion,model.store.digest,model.view_reference)
    ledger = None
    if plan is not None and plan.get('planning_context') is not None:
        from femto.planning import planning_revision, target_surface
        from femto.planning.surface_workspace import target_for
        planning_revision.require_current(model, region, plan['treatment_head'], plan['planning_context']['revision'])
        if target_session is None and verified_assets is None:progress('重建并核验原目标')
        ledger = target_for(model, plan, check, reproduce=verified_assets is None and target_session is None)
    if plan is not None:
        if plan.get('surface_trajectory', {}).get('planner', {}).get('algorithm_version') in ('nominal-variable-width-surface-v1', 'nominal-variable-width-refined-surface-v1'):
            from femto.planning.variable_footprint import require_current
            require_current(plan['surface_trajectory']['planner']['parameters']['variable_footprint'])
        if target_session is not None:
            # Native route validation was bracketed by the same full-input
            # digest, and native publication succeeded. Recheck the normalized
            # region before consuming that one-use, worker-local result.
            target_session.recheck(model, region, plan, ledger, check)
        else:
            from femto.planning.planning_state import validate
            validate(plan, model.store.mesh, region, check=check)
    from femto.planning import coverage_workflow
    certificate = coverage_workflow.cached(model, plan)
    if verified_assets is not None:
        from femto.planning.refinement_archive import require_verified_assets
        from femto.planning.surface_trajectory import SurfaceTrajectory
        require_verified_assets(verified_assets, SurfaceTrajectory.from_dict(plan['surface_trajectory']), ledger, certificate, check)
    with staging_directory(path.parent) as root:
        check()
        # Write current bound geometry, never depend on a relocated source filename.
        progress('复制参考网格与图像资产')
        with open(model.store.source,'rb') as source,(root/'scene_tsdf_mesh_m.ply').open('wb') as target:copy_stream(source,target,check)
        from femto.geometry.fusion import read_geometry_safe
        if mesh_digest(read_geometry_safe(root/'scene_tsdf_mesh_m.ply',mesh=True))!=mesh_digest(model.store.mesh):
            raise ValueError('参考网格文件已改变，未保存方案；请重新加载正确来源')
        check()
        Image.fromarray(model.rgb).save(root/'rgb.png')
        from femto.planning.reference_display import save as save_display
        save_display(model, root, check)
        np.save(root/'depth.npy',model.projection.depth,allow_pickle=False)
        np.save(root/'image_valid.npy',getattr(model.projection,'image_valid',np.ones(model.rgb.shape[:2],bool)),allow_pickle=False)
        for name in ('mesh_triangle_uv.npy','texture_multiview.png'):
            source=Path(model.store.source).parent/name
            if source.is_file():shutil.copyfile(source,root/name)
        if (root/'mesh_triangle_uv.npy').exists()!=(root/'texture_multiview.png').exists():raise ValueError('Incomplete texture assets')
        if ledger is not None:
            target_surface.write(root, ledger, check)
            context = plan['planning_context']; params = plan['surface_trajectory']['planner']['parameters']
            target_surface.read(root, context['target'], context['revision'], evidence=params['quality_evidence'],
                                domain=params['effective_domain'], check=check, reproduce=False)
        if certificate is not None:
            from femto.planning.surface_trajectory import SurfaceTrajectory
            progress('保存模型覆盖证据并完整复算')
            from femto.planning import coverage_certificate
            if (target_session is not None and verified_assets is None
                    and type(certificate) is coverage_certificate.Certificate
                    and plan['surface_trajectory']['planner']['algorithm_version']=='quality-constrained-surface-v3'
                    and certificate.metadata['algorithm'] not in coverage_certificate.RECALCULATION_ALGORITHMS):
                target_session.write_verified_coverage(root,model,region,plan,ledger,certificate,check)
            else:
                coverage_workflow.write(root, certificate, SurfaceTrajectory.from_dict(plan['surface_trajectory']), ledger, check,
                    **({'verified_assets':verified_assets} if verified_assets is not None else {}))
        payload=dict(schema=SCHEMA,project_id=getattr(model,'project_id',None) or uuid.uuid4().hex,revision_id=uuid.uuid4().hex,
            coordinate_frame='world_rdf',unit='metre',geometry_sha256=mesh_digest(model.store.mesh),
            origin_reference=model.store.origin_reference,local_context=model.store.local_context,
            source_lineage=getattr(model,'source_lineage',dict(source_mesh=model.store.source,geometry_sha256=model.store.digest)),
            view_reference=model.view_reference,intrinsic=model.projection.k.tolist(),
            depth_min_m=model.projection.minimum,depth_max_m=model.projection.maximum,max_radius_mm=model.max_radius_mm,
            region=region,plan=plan,planning_state=dict(schema='femto-planning-state-v1',parameters=planning_parameters),
            physical_accuracy_verified=False,execution_ready=False,
            scope='single_editable_region_with_full_reference_geometry_and_image_context',
            files=_project_asset_hashes(root,check))
        from femto.dataset.task_context import checked_task
        if acquisition is not None:
            payload['acquisition_completion_required']=True
            payload['portable_acquisition']=acquisition
        payload['task_context']=checked_task(getattr(model,'task_context',None))
        payload['parent_revision_id']=getattr(model,'revision_id',None)
        _write_manifest(root/'project.json', payload, check)
        if (root/'project.json').stat().st_size > MAX_PROJECT_JSON_BYTES:
            raise ValueError('完整工程清单超过 100 MiB 可回读预算；原方案未覆盖，当前内容仍保留')
        progress('写入方案清单与压缩包')
        archive=_pack_assets(root, check)
        def replace():
            if (digest(path,check) if path.exists() else None)!=baseline:raise ValueError('保存期间目标方案已改变，未覆盖')
            if plan is not None and plan.get('surface_trajectory', {}).get('planner', {}).get('algorithm_version') in ('nominal-variable-width-surface-v1', 'nominal-variable-width-refined-surface-v1'):
                from femto.planning.variable_footprint import require_current
                require_current(plan['surface_trajectory']['planner']['parameters']['variable_footprint'])
            if ledger is not None:
                planning_revision.require_current(model, region, plan['treatment_head'], plan['planning_context']['revision'])
            if target_session is not None:
                target_session.recheck(model, region, plan, ledger, check)
            windows_file_retry(lambda:archive.replace(path))
        publish(replace,check,commit)
    model.project_id=payload['project_id']
    model.revision_id=payload['revision_id']
    from femto.planning.catalog import register_project
    model.last_save_warning=None
    try:register_project(getattr(model,'reconstruction',Path(model.store.source).parent),path)
    except OSError as error:model.last_save_warning='方案已保存，但历史索引更新失败：'+str(error)
    if catalog_root is not None:
        try:register_project(catalog_root,path)
        except OSError as error:model.last_save_warning='方案已保存，但全部方案索引更新失败：'+str(error)
    return path


def _open_project(path,workspace=None,cancel=None,progress=lambda stage:None):
    from femto.annotation.fixed_rgb_geometry import FixedRGBModel,FixedProjection,GeometryBudget,_mesh_budget_before_load,MAX_IMAGE_PIXELS
    from femto.annotation.scene_annotations import AnnotationStore,mesh_digest
    from femto.geometry.fusion import read_geometry_safe
    budget=GeometryBudget(cancel);budget.check()
    progress('检查工程文件和必要资产')
    path=Path(path)
    source_digest=digest(path,budget.check)
    with zipfile.ZipFile(path) as z:
        entries=z.infolist();names=[i.filename for i in entries]
        if len(names)!=len(set(names)) or not set(names)<=ASSETS or 'project.json' not in names:raise ValueError('Invalid project archive entries')
        if sum(i.file_size for i in entries)>1024**3 or any(i.file_size>512*1024**2 for i in entries):raise ValueError('Project exceeds 1 GiB budget')
        if z.getinfo('project.json').file_size>MAX_PROJECT_JSON_BYTES:raise ValueError('Project metadata too large')
        data=json.loads(z.read('project.json'))
        if data.get('schema')!=SCHEMA or data.get('unit')!='metre' or data.get('coordinate_frame')!='world_rdf':raise ValueError('Unsupported project coordinate contract')
        fields={'files','geometry_sha256','origin_reference','local_context','view_reference','intrinsic','depth_min_m','depth_max_m','max_radius_mm','region','project_id','source_lineage'}
        missing_fields=fields-set(data)
        if missing_fields:raise ValueError('工程缺少来源或必要字段：'+', '.join(sorted(missing_fields)))
        required={'scene_tsdf_mesh_m.ply','rgb.png','depth.npy','image_valid.npy'}
        missing=required-set(names) | (set(data.get('files',{}))-set(names))
        if missing:raise ValueError('工程缺少必要资产：'+', '.join(sorted(missing)))
        if not required<=set(data['files']) or set(data['files'])!=set(names)-{'project.json'}:raise ValueError('工程资产清单不完整或存在未登记文件')
        if ('mesh_triangle_uv.npy' in names)!=('texture_multiview.png' in names):raise ValueError('工程纹理资产不完整：需要UV与纹理成对存在')
        has_target = bool((data.get('plan') or {}).get('planning_context'))
        if has_target != ('target_surface.json' in names) or has_target != ('target_surface.npz' in names):
            raise ValueError('工程原目标台账资产或计算修订不完整')
        has_coverage = (data.get('plan') or {}).get('coverage_evidence') is not None
        if (data.get('plan') or {}).get('surface_trajectory'):
            from femto.planning.refinement_archive import check_inventory
            from femto.planning.surface_trajectory import SurfaceTrajectory
            check_inventory(set(names), SurfaceTrajectory.from_dict(data['plan']['surface_trajectory']),
                            data['plan'].get('coverage_evidence'))
        elif set(names) & (REFINEMENT_ASSETS | HINT_ASSETS):
            raise ValueError('工程包含孤立的补充规划证据')
        if (has_coverage and not has_target or has_coverage != ('coverage_certificate.json' in names)
                or has_coverage != ('coverage_certificate.npz' in names)):
            raise ValueError('工程覆盖证据与原目标资产不完整')
        if has_coverage and (z.getinfo('coverage_certificate.json').file_size > 1024**2
                             or z.getinfo('coverage_certificate.npz').file_size > 256*1024**2):
            raise ValueError('工程覆盖证据超出读取预算')
        if data.get('region',{}).get('view_reference')!=data.get('view_reference'):raise ValueError('区域与参考图像来源不匹配')
        parent=Path(workspace) if workspace else OUTPUTS_ROOT/'region_workspaces'
        root=parent
        root.mkdir(parents=True,exist_ok=True)
        progress('解包并核对资产指纹')
        for name,expected in data['files'].items():
            budget.check()
            with z.open(name) as source,(root/name).open('wb') as target:
                from femto.planning.planning_io import copy_stream
                copy_stream(source,target,budget.check)
            if digest(root/name,budget.check)!=expected:raise ValueError('Project asset checksum mismatch: '+name)
        _write_manifest(root/'project.json', data, budget.check, allow_nan=True)
    progress('加载参考几何与图像')
    mesh_path=root/'scene_tsdf_mesh_m.ply';_mesh_budget_before_load(mesh_path)
    mesh=read_geometry_safe(mesh_path,mesh=True)
    if mesh_digest(mesh)!=data['geometry_sha256']:raise ValueError('Project geometry fingerprint mismatch')
    with Image.open(root/'rgb.png') as im:
        if im.width*im.height>MAX_IMAGE_PIXELS:raise ValueError('Project image too large')
        rgb=np.asarray(im.convert('RGB')).copy()
    depth=np.load(root/'depth.npy',allow_pickle=False);valid=np.load(root/'image_valid.npy',allow_pickle=False)
    k=np.asarray(data['intrinsic'],float)
    if depth.shape!=rgb.shape[:2] or depth.dtype!=np.uint16 or valid.shape!=depth.shape or valid.dtype!=bool:raise ValueError('Project image/depth shape or type mismatch')
    if model_view:=data.get('view_reference'):
        if 'undistorted_intrinsic' in model_view and model_view['undistorted_intrinsic']!=k.tolist():raise ValueError('参考图像内参与工程内参不一致')
        if 'image_size' in model_view and model_view['image_size']!=[rgb.shape[1],rgb.shape[0]]:raise ValueError('参考图像尺寸与来源不一致')
    if k.shape!=(3,3) or not np.isfinite(k).all() or min(k[0,0],k[1,1])<=0:raise ValueError('Invalid project intrinsic')
    model=PortableRegionModel.__new__(PortableRegionModel)
    model.reconstruction=root;model.mesh=mesh;model.rgb=rgb
    model.store=AnnotationStore(mesh,mesh_path)
    model.store.origin_reference=data['origin_reference'];model.store.local_context=data['local_context']
    model.view_reference=data['view_reference'];model.max_radius_mm=float(data['max_radius_mm'])
    from femto.planning.reference_display import load as load_display
    load_display(model, root, budget.check)
    from femto.dataset.acquisition_completion import validate_portable_completion
    acquisition=data.get('portable_acquisition')
    if data.get('acquisition_completion_required') is True and acquisition is None:
        raise ValueError('便携方案缺少原采集质量与确认记录。')
    model.portable_acquisition=acquisition
    if acquisition is not None:
        model.acquisition_completion=validate_portable_completion(
            acquisition,data['geometry_sha256'],model.view_reference)
    model.project_id=data['project_id'];model.source_lineage=data['source_lineage'];model._overlay_cache={}
    from femto.dataset.task_context import checked_task
    model.task_context=checked_task(data.get('task_context'))
    model.revision_id=data.get('revision_id')
    model.parent_revision_id=data.get('parent_revision_id')
    progress('重建投影与遮挡支持')
    model.projection=FixedProjection(mesh,k,depth,data['depth_min_m'],data['depth_max_m'],check=budget.check)
    p=model.projection;p.image_valid=valid;p.valid[~valid]=False;p.new_support[~valid]=False;p.ids[~valid]=-1;p.z[~valid]=np.nan;p.reasons[~valid]=4
    region=dict(data['region'],source_mesh=str(mesh_path.resolve()))
    model.store.regions=[model.store.normalized_region(region)];model.store.mark_saved()
    model.prepare_display(cancel=cancel)
    model.saved_plan=None
    model.target_ledger=None
    model.coverage_certificate=None
    model.plan_warning=None
    settings=data.get('planning_state',{})
    if settings and settings.get('schema')!='femto-planning-state-v1':raise ValueError('Unknown planning state version')
    model.planning_parameters=settings.get('parameters')
    if not data.get('plan') and isinstance(model.planning_parameters,dict) and model.planning_parameters.get('kind')=='mesh_slice_selection_blocks_ordered':
        from femto.planning.surface_workspace import checked_head
        from femto.planning.fixed_width_layout import OPTIMIZED_PATCH_VERSION,FOLDBACK_PATCH_VERSION,ADAPTIVE_FLOOR_PATCH_VERSION,COORDINATED_FLOOR_PATCH_VERSION
        from femto.planning.continuous_quality import IMPROVED_ROUTING,SUPPORT_ROUTING,BALANCED_ROUTING,SEAM_ROUTING,SEAM_BALANCED_ROUTING,PRESERVED_DETOUR_ROUTING,SHORT_DETOUR_ROUTING,JOINT_REORDER_ROUTING,BRIDGE_REORDER_ROUTING,RELOCATION_ROUTING,DEEP_RELOCATION_ROUTING,FLOOR_DETOUR_ROUTING,UNIFIED_FLOOR_ROUTING,GLOBAL_FLOOR_ROUTING,REFINED_CONNECTOR_ROUTING
        values=model.planning_parameters.get('values')
        if not isinstance(values,dict) or not set(values)<= {'treatment_head','layout_algorithm'} or 'treatment_head' not in values:
            raise ValueError('保存的规划参数无效')
        head=checked_head(values['treatment_head']);layout=values.get('layout_algorithm')
        if layout not in (None,OPTIMIZED_PATCH_VERSION,FOLDBACK_PATCH_VERSION,ADAPTIVE_FLOOR_PATCH_VERSION,COORDINATED_FLOOR_PATCH_VERSION,IMPROVED_ROUTING,SUPPORT_ROUTING,BALANCED_ROUTING,SEAM_ROUTING,SEAM_BALANCED_ROUTING,PRESERVED_DETOUR_ROUTING,SHORT_DETOUR_ROUTING,JOINT_REORDER_ROUTING,BRIDGE_REORDER_ROUTING,RELOCATION_ROUTING,DEEP_RELOCATION_ROUTING,FLOOR_DETOUR_ROUTING,UNIFIED_FLOOR_ROUTING,GLOBAL_FLOOR_ROUTING,REFINED_CONNECTOR_ROUTING) or (layout is not None and head['mode']!='fixed_width_auto'):
            raise ValueError('保存的局部分区排布版本无效')
    if data.get('plan'):
        progress('重算核验路径和报告')
        # Plans store coordinates, never trust imported samples without regeneration.
        from femto.planning.region_path import signature,load_path
        from femto.planning.planning_state import validate_binding
        original=data['plan']
        if original.get('planning_context') is not None:
            from femto.planning import planning_revision
            context = original['planning_context']
            planning_revision.require_current(model, model.store.regions[0], original['treatment_head'], context['revision'])
        else:
            model.plan_warning='旧记录未建立原目标台账与完整计算修订；保留原路线语义，可生成新版本完成核验。'
        if 'plan_binding' in original:
            # Verify the untouched binding before relocating source fields.
            # load_path still performs full geometry/target/coverage replay.
            validate_binding(original, mesh, data['region'], check=budget.check)
        elif original.get('region_signature') != signature(data['region']):
            raise ValueError('Legacy project region association cannot be verified')
        else:model.plan_warning='旧方案缺少算法版本关联；已按原始来源和参数重算验证，请另存新版。'
        plan=dict(data['plan'],source_mesh=model.store.source,geometry_sha256=model.store.digest,
            local_context=model.store.local_context,origin_reference=model.store.origin_reference,
            region_signature=original['region_signature'] if 'plan_binding' in original else signature(model.store.regions[0]))
        from femto.annotation.measurement_audit import layer_contract
        plan['data_layers']=layer_contract(model.store)
        candidate=root/'candidate_path.json';_write_validation_plan(candidate, plan, budget.check)
        model.saved_plan=load_path(candidate,model.store,p,model.store.regions[0],budget.check,
            coverage_sink=lambda certificate:setattr(model, 'coverage_certificate', certificate),
            target_sink=lambda ledger:setattr(model, 'target_ledger', ledger))
        from femto.planning.planning_state import parameters
        if model.planning_parameters is not None and model.planning_parameters!=parameters(model.saved_plan):
            raise ValueError('Project parameters do not match saved path')
        model.planning_parameters=parameters(model.saved_plan)
        if model.coverage_certificate is not None:
            from femto.planning.coverage_display import build
            progress('准备已核验的覆盖显示')
            model.coverage_display=build(model.target_ledger,model.coverage_certificate,budget.check)
    budget.check()
    if digest(path,budget.check)!=source_digest:raise ValueError('加载期间工程文件已改变，请重新打开')
    model.project_source_path=path.resolve();model.project_file_digest=source_digest
    return model


@diagnosed('portable.read', operation=True)
def open_project(path: Any,workspace: str | Path | None=None,cancel: Any=None,progress: Callable[..., Any]=lambda stage:None) -> Any:
    parent=Path(workspace) if workspace else OUTPUTS_ROOT/'region_workspaces'
    parent.mkdir(parents=True,exist_ok=True)
    root=parent/uuid.uuid4().hex
    root.mkdir()
    (root/'.incomplete').touch()
    try:
        model=_open_project(path,root,cancel,progress)
        (root/'.incomplete').unlink()
        return model
    except BaseException:
        # Only remove this call's freshly allocated, checked workspace.
        if root.resolve().parent==parent.resolve():shutil.rmtree(root)
        raise



from femto.annotation.fixed_rgb_geometry import FixedRGBModel

class PortableRegionModel(FixedRGBModel):
    def load(self,path: str | Path,*,cancel: Any=None) -> Any:
        # Existing annotations exported inside this workspace retain its source binding.
        old_state=self.store.__dict__.copy();old_cache=self._overlay_cache.copy()
        try:
            data=json.loads(Path(path).read_text(encoding='utf-8-sig'))
            for item in data.get('points',[])+data.get('regions',[]):
                if item.get('view_reference')!=self.view_reference:raise ValueError('Reference RGB evidence mismatch')
            result=self.store.load(path);self.prepare_display(cancel=cancel);return result
        except Exception:
            self.store.__dict__.clear();self.store.__dict__.update(old_state);self._overlay_cache=old_cache
            raise


def load_source(path: str | Path,cancel: Any=None,progress: Callable[..., Any]=lambda stage:None) -> Any:
    progress('加载参考来源')
    return open_project(path,cancel=cancel,progress=progress) if Path(path).is_file() else FixedRGBModel(path,cancel=cancel)


def project_from_annotations(source: str | Path,index: int,cancel: Any=None,progress: Callable[..., Any]=lambda stage:None) -> Any:
    source=Path(source).resolve()
    if source.name!='annotations.json' or source.parent.parent.name!='annotations':
        raise ValueError('请选择当前模型的历史区域标记记录。')
    model=FixedRGBModel(source.parent.parent.parent,cancel=cancel)
    model.load(source,cancel=cancel)
    if type(index) is not int or not 0<=index<len(model.store.regions):raise ValueError('区域记录不存在')
    destination=model.reconstruction/'region_projects'/('region_'+uuid.uuid4().hex+'.region.zip')
    from femto.annotation.fixed_rgb_geometry import GeometryBudget
    return save_project(model,model.store.regions[index],destination,check=GeometryBudget(cancel).check,progress=progress,
        commit=getattr(cancel,"publish",None))


if __name__ == '__main__':
    # Keep this module useful as a headless archive converter. Interactive
    # opening belongs to the Qt workbench and is intentionally not duplicated.
    import argparse
    parser = argparse.ArgumentParser(description='Create a portable region project')
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--region-index', type=int, default=0)
    args = parser.parse_args()
    try:
        print(project_from_annotations(args.annotations, args.region_index))
    except Exception as error:
        parser.error(str(error))
