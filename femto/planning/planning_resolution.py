"""Explicit, derived planning references on unchanged measured W0 geometry.

Integer-lattice RGB/depth samples retain their original camera rays. This is a
new reference for a new selection, never an automatic change to an existing ROI.
The original acquisition assessment remains source history, not confirmation of
the derived reference or physical execution accuracy.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import math

import cv2
import numpy as np
from PIL import Image

SCHEMA = 'reduced-planning-reference-v1'
STRIDES = (4, 8, 16)
SAMPLE_STEPS = (1., 2., 3., 5.)


def profile(stride: int, sample_step_mm: float, *, sampling_policy: str | None = None) -> str:
    if type(stride) is not int or stride not in STRIDES:
        raise ValueError('简化参考分辨率档位无效')
    if (type(sample_step_mm) not in (int, float) or not math.isfinite(sample_step_mm)
            or sample_step_mm not in SAMPLE_STEPS):
        raise ValueError('简化路线取点间隔无效')
    if sampling_policy is not None:
        from femto.planning.route_sampling import POLICY, CURVATURE_POLICY
        if sampling_policy not in (POLICY, CURVATURE_POLICY) or sample_step_mm != 5.:
            raise ValueError('自动取点策略或最大采样间隔无效')
        return f'stride{stride}-original-mesh-{sampling_policy}'
    return f'stride{stride}-original-mesh-step{sample_step_mm:g}mm-v1'


def route_sample_step_mm(view_reference: dict, *, decision: dict | None = None, mesh=None) -> float:
    derivation = view_reference.get('derivation') if isinstance(view_reference, dict) else None
    from femto.planning.patch_reference import SCHEMA as PATCH_SCHEMA, checked
    if isinstance(derivation, dict) and derivation.get('schema') == PATCH_SCHEMA:
        checked(derivation)
        from femto.planning.route_sampling import validate
        if not isinstance(decision, dict) or decision.get('policy') != derivation['sampling_policy']:
            raise ValueError('粗分区沿线取点策略不一致')
        return float(validate(decision, mesh)['sample_step_mm'])
    if not isinstance(derivation, dict) or derivation.get('schema') != SCHEMA:
        return 1.
    step, stride = derivation.get('route_sample_step_mm'), derivation.get('stride')
    policy = derivation.get('sampling_policy')
    expected = profile(stride, step, sampling_policy=policy)
    if derivation.get('profile') != expected:
        raise ValueError('简化参考配置与路线取点间隔不一致')
    if policy is not None:
        from femto.planning.route_sampling import validate
        if not isinstance(decision,dict) or decision.get('policy')!=policy:
            raise ValueError('沿线取点决策与参考策略版本不一致')
        return float(validate(decision, mesh)['sample_step_mm'])
    if decision is not None:
        raise ValueError('固定采样参考不能附带自动取点决策')
    return float(step)


def create_reduced_model(source, destination, check=lambda: None, progress=lambda text: None,
                         *, stride=8, route_sample_step_mm=3., sampling_policy=None):
    """Create a separate, empty model; caller installs it only after success."""
    from femto.annotation.fixed_rgb_geometry import FixedRGBModel, FixedProjection
    from femto.annotation.scene_annotations import AnnotationStore, mesh_digest
    from femto.dataset.acquisition_completion import portable_completion, validate_portable_completion, read_completion
    from femto.geometry.fusion import read_geometry_safe
    from femto.io_utils import sha256_file, write_json_strict
    from femto.planning.planning_io import copy_stream
    from femto.planning.planning_revision import array_record

    chosen_profile = profile(stride, route_sample_step_mm, sampling_policy=sampling_policy)
    check()
    if (source.store.points or source.store.regions or getattr(source.store, '_history', [])
            or getattr(source, 'saved_plan', None) is not None):
        raise ValueError('请在选择区域和生成路线之前创建简化规划参考；已有标记或方案不会自动转换')
    if (source.view_reference.get('derivation') or {}).get('schema') == SCHEMA:
        raise ValueError('当前已经是简化规划参考，请从原始模型重新创建')
    destination = Path(destination).resolve()
    if destination.exists():raise FileExistsError('简化规划参考需要新的独立目录')
    source_path = Path(source.store.source).resolve()
    if destination == source_path.parent or source_path.is_relative_to(destination):
        raise ValueError('派生目录不能包含或替换原始模型')
    progress('核对原始模型与参考图像')
    source_hash = sha256_file(source_path, check)
    original_digest = mesh_digest(source.mesh)
    if (source.store.mesh is not source.mesh and mesh_digest(source.store.mesh) != original_digest
            or mesh_digest(read_geometry_safe(source_path, mesh=True)) != original_digest):
        raise ValueError('原始网格文件与已加载模型不一致，未创建简化参考')
    rgb, depth = np.asarray(source.rgb), np.asarray(source.projection.depth)
    k = np.asarray(source.projection.k, float)
    original_valid = np.asarray(getattr(source.projection, 'image_valid', np.ones(depth.shape, bool)))
    if (rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3
            or depth.dtype != np.uint16 or depth.shape != rgb.shape[:2]
            or original_valid.dtype != bool or original_valid.shape != depth.shape
            or k.shape != (3, 3) or not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0
            or source.view_reference.get('image_size') != [rgb.shape[1], rgb.shape[0]]
            or source.view_reference.get('undistorted_intrinsic') != k.tolist()):
        raise ValueError('原始参考图像、深度、内参或来源尺寸不一致')
    acquisition = deepcopy(getattr(source, 'portable_acquisition', None))
    if acquisition is None:
        completion = read_completion(source.reconstruction, check=check, require_confirmed=True)
        if completion is not None:
            acquisition = portable_completion(completion, original_digest, source.view_reference)
    if acquisition is not None:validate_portable_completion(acquisition, original_digest, source.view_reference)
    check()
    offset = stride // 2
    grid = (slice(offset, None, stride), slice(offset, None, stride))
    reduced_rgb, reduced_depth = rgb[grid].copy(), depth[grid].copy()
    if min(reduced_depth.shape) < 2:raise ValueError('参考图像过小，无法使用此简化档位')
    reduced_k = k.copy()
    reduced_k[:2] /= stride
    reduced_k[:2, 2] -= offset/stride
    reduced_valid = cv2.erode(original_valid.astype(np.uint8),
        np.ones((stride+1, stride+1), np.uint8), borderType=cv2.BORDER_CONSTANT,
        borderValue=0)[grid].astype(bool)
    check()
    destination.mkdir(parents=True, exist_ok=False)

    def copy_asset(start, end):
        check()
        with Path(start).open('rb') as reader, Path(end).open('xb') as writer:
            copy_stream(reader, writer, check)
        check()

    progress('建立独立简化参考，保留完整三维表面')
    mesh_path = destination/'scene_tsdf_mesh_m.ply'
    copy_asset(source_path, mesh_path)
    if sha256_file(mesh_path, check) != source_hash:raise ValueError('派生网格复制指纹不一致')
    for name in ('mesh_triangle_uv.npy', 'texture_multiview.png'):
        asset = source_path.parent/name
        if asset.is_file():copy_asset(asset, destination/name)
    Image.fromarray(reduced_rgb).save(destination/'reference_rgb.png');check()
    Image.fromarray(reduced_depth).save(destination/'reference_depth.png');check()
    derivation = dict(schema=SCHEMA, profile=chosen_profile, stride=stride,
        route_sample_step_mm=float(route_sample_step_mm),
        source_pixel_offset_xy=[offset, offset],
        source_pixel_mapping='u_source=stride*u_derived+offset; same for v',
        depth_sampling='exact_source_measurement_no_interpolation',
        support_scope='sampled_measured_rays_not_full_resolution_evidence',
        source_view_reference=deepcopy(source.view_reference),
        original_rgb=array_record(rgb), original_depth=array_record(depth),
        original_image_valid=array_record(original_valid), source_mesh_file_sha256=source_hash,
        source_geometry_sha256=original_digest, geometry='unchanged_original_mesh',
        source_mesh_deviation_mm=0., original_image_size=[rgb.shape[1], rgb.shape[0]],
        derived_image_size=[reduced_rgb.shape[1], reduced_rgb.shape[0]],
        selection_policy='new_selection_required_no_automatic_roi_transfer',
        source_assessment_only=acquisition is not None, derived_geometry_physically_confirmed=False,
        physical_accuracy_verified=False, execution_ready=False)
    if sampling_policy is not None:derivation['sampling_policy'] = sampling_policy
    view = dict(source.view_reference,
        rgb_sha256=sha256_file(destination/'reference_rgb.png', check),
        depth_sha256=sha256_file(destination/'reference_depth.png', check),
        undistorted_intrinsic=reduced_k.tolist(), image_size=derivation['derived_image_size'],
        view='derived_initial_camera_integer_subsample',
        mapping='exact sampled original RGB/depth ray -> unchanged W0 mesh', derivation=derivation)
    model = FixedRGBModel.__new__(FixedRGBModel)
    model.reconstruction = destination
    # Copy geometry ownership so later source/UI edits cannot mutate this model.
    model.mesh = deepcopy(source.mesh)
    model.rgb = reduced_rgb
    model.store = AnnotationStore(model.mesh, mesh_path)
    model.store.origin_reference = deepcopy(source.store.origin_reference)
    model.store.local_context = deepcopy(source.store.local_context)
    model.view_reference = view
    from femto.planning.reference_display import install as install_display
    install_display(model, rgb)
    model.max_radius_mm = source.max_radius_mm
    model.task_context = deepcopy(getattr(source, 'task_context', None))
    if hasattr(source, 'planning_quality_config'):
        model.planning_quality_config = deepcopy(source.planning_quality_config)
    model._overlay_cache = {}
    progress('重建简化参考的深度与遮挡检查')
    model.projection = FixedProjection(model.mesh, reduced_k, reduced_depth,
        source.projection.minimum, source.projection.maximum, check=check)
    projection = model.projection
    projection.image_valid = reduced_valid
    projection.valid[~reduced_valid] = False;projection.new_support[~reduced_valid] = False
    projection.ids[~reduced_valid] = -1;projection.z[~reduced_valid] = np.nan;projection.reasons[~reduced_valid] = 4
    model.saved_plan = None;model.target_ledger = None;model.coverage_certificate = None
    model.source_lineage = dict(schema='reduced-planning-lineage-v1',
        source_mesh=str(source_path), geometry_sha256=original_digest,
        original_view_reference=deepcopy(source.view_reference),
        original_source_lineage=deepcopy(getattr(source, 'source_lineage', None)),
        original_acquisition_wrapper=acquisition, derivation=deepcopy(derivation))
    if acquisition is not None:
        model.portable_acquisition = portable_completion(acquisition['completion'], model.store.digest, view)
    check()
    if sha256_file(source_path, check) != source_hash or mesh_digest(source.mesh) != original_digest:
        raise ValueError('创建期间原始模型发生变化，未接纳派生参考')
    write_json_strict(destination/'planning_reference.json',
        dict(schema=SCHEMA, view_reference=view, source_lineage=model.source_lineage,
             origin_reference=model.store.origin_reference, local_context=model.store.local_context,
             physical_accuracy_verified=False, execution_ready=False), check=check)
    model.store.mark_saved();check()
    return model
