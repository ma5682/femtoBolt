"""Evidence-preserving planning surfaces, including shared single/multi TSDF.

The original RGB-D pixel grid, painted intent and full measured surface remain
available. A separate derived mesh reduces planning work. Missing source pixels
and differences above the stated engineering tolerance cannot become support.
"""
from __future__ import annotations
from femto.performance_detail import observed

from copy import copy, deepcopy
from pathlib import Path
import json
import numpy as np
from PIL import Image

SCHEMA = 'surface-patch-planning-reference-v1'
RECORD = 'surface_patch_reference.json'
SOURCE_ASSET = 'measured_source_surface.ply'
PROFILE = 'connected-3mm-cells-full-reference-adaptive-sampling-v1'
TOLERANCE_MM = 2.
SURFACE_METRIC = 'same_component_measured_triangle_distance_upper_bound_v1'


def derivation(model):
    value = (getattr(model, 'view_reference', None) or {}).get('derivation') or {}
    return value if value.get('schema') == SCHEMA else None


def checked(value):
    from femto.planning.route_sampling import POLICY
    from femto.geometry.surface_patches import VERSION, CELL_MM, MAX_DISPLACEMENT_MM
    from femto.local.planning_tsdf import PROFILE as TSDF_PROFILE, LEGACY_PROFILE, checked as check_tsdf
    if (not isinstance(value, dict) or value.get('schema') != SCHEMA or value.get('profile') not in (PROFILE, TSDF_PROFILE, LEGACY_PROFILE)
            or value.get('sampling_policy') != POLICY or value.get('route_sample_step_mm') != 5.
            or value.get('source_surface_tolerance_mm') != TOLERANCE_MM
            or value.get('selection_policy') != 'unchanged_full_resolution_intent'
            or value.get('source_asset') != SOURCE_ASSET):
        raise ValueError('粗分区规划参考参数或版本不一致')
    partition = value.get('partition') or {}
    if value['profile'] in (TSDF_PROFILE, LEGACY_PROFILE):
        if value['profile'] == TSDF_PROFILE and value.get('source_surface_metric') != SURFACE_METRIC:
            raise ValueError('实测表面距离核验版本不一致')
        scope = value.get('source_binding', {}).get('local_context', {}).get('reconstruction_scope', {})
        check_tsdf(partition, scope)
    elif (partition.get('schema') != VERSION or partition.get('cell_mm') != CELL_MM
            or partition.get('max_displacement_limit_mm') != MAX_DISPLACEMENT_MM):
        raise ValueError('粗分区表面算法或尺寸记录不一致')
    return value


@observed('patch.source_guard')
def guard(model, source_mesh, check=lambda: None, *, source_projection=None):
    """Recheck against the full source at every original reference pixel.

    This is discrete image/mesh support, not a continuous distance certificate.
    The existing whole-edge width, boundary and connection checks still run.
    """
    from femto.annotation.fixed_rgb_geometry import FixedProjection
    from femto.local.planning_tsdf import PROFILE as TSDF_PROFILE, LEGACY_PROFILE
    d = checked(derivation(model));p = model.projection
    source = source_projection or FixedProjection(source_mesh, p.k, p.depth, p.minimum, p.maximum, check=check)
    check()
    original_supported = source.valid & ~source.new_support & np.asarray(getattr(source, 'image_valid', p.image_valid))
    original_supported &= (p.depth>=p.minimum*1000)&(p.depth<=p.maximum*1000)
    source_z=np.asarray(source.z,float).copy()
    # Float32 ray casting can miss a ray landing exactly on dense pixel-mesh
    # vertices. Retry original triangles with exact orientation signs, never
    # substitute a nearby point or heal a positive gap.
    ids=np.flatnonzero(p.valid & ~original_supported & p.image_valid &
        (p.depth>=p.minimum*1000)&(p.depth<=p.maximum*1000))
    recovered_count=0
    if len(ids):
        from femto.geometry.reference_intersections import retry_missed_rays
        y,x=np.unravel_index(ids,p.depth.shape)
        rays=np.c_[(x-p.k[0,2])/p.k[0,0],(y-p.k[1,2])/p.k[1,1],np.ones(len(ids))]
        # The new TSDF profile proposes retries from the original depth, not
        # the derived surface. Preserve the historical profile's replay rule.
        hint = (p.depth[y,x].astype(float)/1000 if d['profile'] in (TSDF_PROFILE, LEGACY_PROFILE) else p.z[y,x])
        hits=retry_missed_rays(source_mesh,source.scene,rays,rays*hint[:,None],check)
        supported=np.isfinite(hits)&(hits>=p.minimum)&(hits<=p.maximum)&(np.abs(hits*1000-p.depth[y,x])<=10.)
        original_supported.ravel()[ids[supported]]=True;source_z.ravel()[ids[supported]]=hits[supported]
        recovered_count=int(supported.sum())
    with np.errstate(invalid='ignore'):
        error = np.abs(p.z.astype(float) - source_z) * 1000
    rejected = ~original_supported | ~np.isfinite(error) | (error > d['source_surface_tolerance_mm'])
    distance_audit = None
    if d['profile'] == TSDF_PROFILE:
        # A depth difference along an oblique ray is not distance to a measured
        # surface. Keep source-pixel support mandatory, then obtain an actual
        # triangle witness when the direct source intersection is too distant.
        from femto.geometry.measured_surface_distance import same_surface_support
        compared = p.valid & original_supported & np.isfinite(error)
        y,x = np.nonzero(compared)
        rays = np.c_[(x-p.k[0,2])/p.k[0,0], (y-p.k[1,2])/p.k[1,1], np.ones(len(x))]
        direct = error[y,x]*np.linalg.norm(rays,axis=1)
        confirmed = direct+0.001 <= TOLERANCE_MM
        reconsider = ~confirmed
        diagnostics = dict(component_mismatches=0, unresolved_reference_faces=0)
        measured_distance = np.empty(0)
        if reconsider.any():
            ry,rx = y[reconsider],x[reconsider];rr = rays[reconsider]
            accepted, measured_distance, diagnostics = same_surface_support(source_mesh, source.scene,
                rr*p.z[ry,rx,None], rr*source_z[ry,rx,None], source.ids[ry,rx], TOLERANCE_MM, check)
            confirmed[reconsider] = accepted
        rejected = ~compared
        rejected[y,x] = ~confirmed
        distance_audit = dict(distance_metric=SURFACE_METRIC,
            axial_exceeds_tolerance_pixels=int((compared & (error>TOLERANCE_MM)).sum()),
            surface_distance_rechecked_pixels=int(reconsider.sum()),
            surface_distance_confirmed_pixels=int(confirmed[reconsider].sum()),
            surface_distance_rejected_pixels=int((~confirmed).sum()),
            measured_witness_max_distance_mm=(float(measured_distance[confirmed[reconsider]].max())
                if confirmed[reconsider].any() else None), **diagnostics)
    newly_rejected = p.valid & rejected
    if d['profile'] in (TSDF_PROFILE, LEGACY_PROFILE):
        compared = p.valid & original_supported & np.isfinite(error)
        differences = error[compared]
        audit = dict(compared_supported_pixels=int(compared.sum()),
            exceeds_tolerance_pixels=int((compared & (error > TOLERANCE_MM)).sum()),
            source_missing_pixels=int((p.valid & ~original_supported).sum()),
            depth_difference_p95_mm=float(np.quantile(differences,.95)) if len(differences) else None,
            depth_difference_max_mm=float(differences.max()) if len(differences) else None,
            statistics_scope='candidate_supported_reference_pixels_before_source_guard')
    p.valid[rejected] = False;p.new_support[rejected] = False
    p.ids[rejected] = -1;p.z[rejected] = np.nan
    p.reasons[rejected & (p.reasons == 0)] = 1
    check()
    model.source_surface_check = dict(method='full_reference_pixel_source_surface_check_v1',
        tolerance_mm=TOLERANCE_MM, rejected_supported_pixels=int(newly_rejected.sum()),
        original_supported_pixels=int(original_supported.sum()),
        verified_float32_ray_misses=recovered_count,
        continuous_surface_error_certified=False, physical_accuracy_verified=False)
    if d['profile'] in (TSDF_PROFILE, LEGACY_PROFILE):
        model.source_surface_check.update(audit, method='full_reference_pixel_source_surface_check_v2')
    if distance_audit is not None:
        model.source_surface_check.update(distance_audit, method='full_reference_pixel_source_surface_check_v3')


def create(source, destination, check=lambda: None, progress=lambda text: None):
    from femto.annotation.fixed_rgb_geometry import FixedProjection
    from femto.annotation.scene_annotations import AnnotationStore, mesh_digest
    from femto.geometry.surface_patches import simplify
    from femto.geometry.fusion import export_mesh_formats, read_geometry_safe
    from femto.io_utils import sha256_file, write_json_strict
    from femto.planning.planning_io import copy_stream
    from femto.planning.planning_drafts import source_binding
    from femto.planning.route_sampling import POLICY
    from femto.planning.planning_revision import array_record
    from femto.dataset.acquisition_completion import read_completion, portable_completion
    check();destination = Path(destination).resolve()
    if destination.exists():raise FileExistsError('粗分区规划需要新的派生目录')
    if getattr(source, 'saved_plan', None) is not None:raise ValueError('已保存路线需要作为新版本重新生成')
    if source.view_reference.get('derivation'):raise ValueError('请从完整原始表面创建粗分区规划')
    source_path = Path(source.store.source).resolve()
    source_valid = np.asarray(getattr(source.projection, 'image_valid', np.ones(source.rgb.shape[:2], bool)))
    source_hash = sha256_file(source_path, check);source_digest = mesh_digest(source.mesh)
    if mesh_digest(read_geometry_safe(source_path, mesh=True)) != source_digest:
        raise ValueError('原始表面已变化，请重新加载')
    scope = (getattr(source.store, 'local_context', None) or {}).get('reconstruction_scope') or {}
    profile = PROFILE
    if scope.get('input_provenance'):
        from femto.local.planning_tsdf import build, PROFILE as TSDF_PROFILE
        progress('从保留的原始观测生成统一 TSDF 规划表面')
        mesh, report = build(Path(source.reconstruction)/'region_request.json', scope,
            source.projection.minimum, source.projection.maximum, check, progress)
        profile = TSDF_PROFILE
    else:
        # Historical/imported models without the complete RGB-D binding retain
        # the existing profile; never reinterpret an already saved derivation.
        progress('按局部表面分区选择代表点，保留边界与分离表面')
        mesh, report = simplify(source.mesh, check)
    destination.mkdir(parents=True, exist_ok=False)
    with source_path.open('rb') as reader, (destination / SOURCE_ASSET).open('xb') as writer:
        copy_stream(reader, writer, check)
    export_mesh_formats(mesh, destination, stem='scene_tsdf_mesh_m', write_obj_glb=False)
    # Plan and replay use the same canonical saved PLY coordinates.
    mesh = read_geometry_safe(destination / 'scene_tsdf_mesh_m.ply', mesh=True)
    d = dict(schema=SCHEMA, profile=profile, sampling_policy=POLICY, route_sample_step_mm=5.,
        source_surface_tolerance_mm=TOLERANCE_MM, source_asset=SOURCE_ASSET,
        selection_policy='unchanged_full_resolution_intent',
        source_geometry_sha256=source_digest, source_mesh_file_sha256=source_hash,
        planning_geometry_sha256=mesh_digest(mesh), source_view_reference=deepcopy(source.view_reference),
        source_binding=source_binding(source), partition=report,
        reference_rgb=array_record(source.rgb), reference_depth=array_record(source.projection.depth),
        reference_valid=array_record(source_valid),
        physical_accuracy_verified=False, execution_ready=False)
    if profile != PROFILE:d['source_surface_metric'] = SURFACE_METRIC
    model = copy(source);model.mesh = mesh;model.reconstruction = destination
    model.store = AnnotationStore(mesh, destination / 'scene_tsdf_mesh_m.ply')
    model.store.origin_reference = deepcopy(source.store.origin_reference)
    model.store.local_context = deepcopy(source.store.local_context)
    model.view_reference = dict(deepcopy(source.view_reference), derivation=d)
    model._overlay_cache = {};model.saved_plan = None;model.target_ledger = None;model.coverage_certificate = None
    model.source_lineage = dict(schema=SCHEMA, source_mesh=str(source_path), derivation=d)
    model.projection = FixedProjection(mesh, source.projection.k, source.projection.depth,
        source.projection.minimum, source.projection.maximum, check=check)
    p = model.projection
    p.image_valid = np.array(source_valid, copy=True)
    progress('用完整原表面核对规划表面，保留原选区和缺失范围')
    guard(model, source.mesh, check, source_projection=source.projection)
    acquisition = getattr(source, 'portable_acquisition', None)
    completion = acquisition['completion'] if acquisition else read_completion(source.reconstruction, check=check, require_confirmed=True)
    if completion is not None:model.portable_acquisition = portable_completion(completion, model.store.digest, model.view_reference)
    Image.fromarray(model.rgb).save(destination / 'reference_rgb.png')
    np.save(destination / 'reference_depth.npy', p.depth, allow_pickle=False)
    np.save(destination / 'reference_valid.npy', p.image_valid, allow_pickle=False)
    for name in ('region_request.json', 'requested_mask.png'):
        path = Path(source.reconstruction) / name
        if path.is_file():
            with path.open('rb') as reader, (destination / name).open('xb') as writer:copy_stream(reader, writer, check)
    if sha256_file(source_path, check) != source_hash or mesh_digest(source.mesh) != source_digest:
        raise ValueError('创建分区期间原始表面发生变化，未接纳结果')
    if profile != PROFILE:
        from femto.dataset.reconstruction_inputs import verify
        from femto.geometry.local_region import model_selection
        verify(scope['input_provenance'], check)
        model_selection(model)
    # Publish the loadable sidecar only after every source has been rechecked.
    write_json_strict(destination / RECORD, dict(schema=SCHEMA, view_reference=model.view_reference,
        source_lineage=model.source_lineage, origin_reference=model.store.origin_reference,
        local_context=model.store.local_context, depth_min_m=p.minimum, depth_max_m=p.maximum,
        max_radius_mm=model.max_radius_mm, task_context=getattr(model, 'task_context', None),
        source_surface_check=model.source_surface_check,
        portable_acquisition=getattr(model, 'portable_acquisition', None)), check=check)
    check();return model


def install_assets(model, root, check=lambda: None):
    """Required on derived-directory and portable reloads, before ROI validation."""
    from femto.io_utils import sha256_file
    from femto.geometry.fusion import read_geometry_safe
    from femto.annotation.scene_annotations import mesh_digest
    from femto.planning.planning_revision import array_record
    from femto.annotation.fixed_rgb_geometry import _mesh_budget_before_load
    d = derivation(model)
    if d is None:return
    checked(d);check()
    if (model.store.digest != d['planning_geometry_sha256']
            or {k:v for k,v in model.view_reference.items() if k != 'derivation'} != d['source_view_reference']
            or array_record(model.rgb) != d['reference_rgb']
            or array_record(model.projection.depth) != d['reference_depth']
            or array_record(model.projection.image_valid) != d['reference_valid']
            or model.store.local_context != d['source_binding']['local_context']
            or model.store.origin_reference != d['source_binding']['origin_reference']):
        raise ValueError('粗分区参考的几何或原始图像绑定不一致')
    path = Path(root) / SOURCE_ASSET
    _mesh_budget_before_load(path)
    if sha256_file(path, check) != d['source_mesh_file_sha256']:
        raise ValueError('粗分区规划所需的完整原始表面缺失或已改变')
    source = read_geometry_safe(path, mesh=True)
    if mesh_digest(source) != d['source_geometry_sha256']:raise ValueError('原始表面几何指纹不一致')
    guard(model, source, check)


def verify_asset(model, check=lambda: None):
    from femto.io_utils import sha256_file
    d = derivation(model)
    if d is None:return None
    checked(d)
    path = Path(model.reconstruction) / SOURCE_ASSET
    if sha256_file(path, check) != d['source_mesh_file_sha256']:raise ValueError('完整原表面已改变，未保存方案')
    return path


def save_assets(model, root, check=lambda: None):
    from femto.planning.planning_io import copy_stream
    path = verify_asset(model, check)
    if path is None:return
    with path.open('rb') as reader, (Path(root) / SOURCE_ASSET).open('xb') as writer:copy_stream(reader, writer, check)
    verify_asset(model, check)


def load_into(model, root, check=lambda: None):
    from femto.annotation.fixed_rgb_geometry import FixedProjection, _mesh_budget_before_load, MAX_IMAGE_PIXELS
    from femto.annotation.scene_annotations import AnnotationStore
    from femto.geometry.fusion import read_geometry_safe
    root = Path(root).resolve();metadata = root / RECORD
    if metadata.stat().st_size > 16*1024**2:raise ValueError('粗分区参考记录超过读取预算')
    data = json.loads(metadata.read_text(encoding='utf-8'))
    if data['schema'] != SCHEMA:raise ValueError('未知的表面分区参考版本')
    _mesh_budget_before_load(root / 'scene_tsdf_mesh_m.ply')
    model.reconstruction = root;model.mesh = read_geometry_safe(root / 'scene_tsdf_mesh_m.ply', mesh=True)
    model.store = AnnotationStore(model.mesh, root / 'scene_tsdf_mesh_m.ply')
    model.store.origin_reference = data['origin_reference'];model.store.local_context = data['local_context']
    model.view_reference = data['view_reference'];model.source_lineage = data['source_lineage']
    with Image.open(root / 'reference_rgb.png') as im:
        if im.width*im.height > MAX_IMAGE_PIXELS:raise ValueError('粗分区参考图像超过读取预算')
        model.rgb = np.asarray(im.convert('RGB')).copy()
    depth = np.load(root / 'reference_depth.npy', allow_pickle=False, mmap_mode='r')
    valid = np.load(root / 'reference_valid.npy', allow_pickle=False, mmap_mode='r')
    if depth.shape != model.rgb.shape[:2] or valid.shape != depth.shape or valid.dtype != bool or depth.dtype != np.uint16:
        raise ValueError('粗分区参考图像与深度不一致')
    depth,valid = np.array(depth),np.array(valid)
    model.projection = FixedProjection(model.mesh, np.asarray(model.view_reference['undistorted_intrinsic']),
        depth, data['depth_min_m'], data['depth_max_m'], check=check)
    model.projection.image_valid = valid
    model.max_radius_mm = data['max_radius_mm'];model.task_context = data['task_context']
    model.portable_acquisition = data['portable_acquisition'];model._overlay_cache = {}
    install_assets(model, root, check)


def load(root, cancel=None):
    from femto.annotation.fixed_rgb_geometry import FixedRGBModel, GeometryBudget
    model = FixedRGBModel.__new__(FixedRGBModel)
    load_into(model, root, GeometryBudget(cancel).check)
    return model
