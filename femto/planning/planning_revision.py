"""Content-addressed calculation inputs, independent of save UUIDs and UI requests.

Only equal *complete* input records may reuse a result. No partial cross-revision
cache is supported. Navigation, names, notes and portable file locations are not
calculation inputs; acquisition/reference identities and body state are.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from typing import Any

import numpy as np

from femto.planning.surface_trajectory import canonical_json_bytes
from femto.planning.algorithm_registry import UNIFIED_FLOOR_ROUTING, GLOBAL_FLOOR_ROUTING, REFINED_CONNECTOR_ROUTING
from femto.planning.fixed_width_layout import ADAPTIVE_FLOOR_PATCH_VERSION,COORDINATED_FLOOR_PATCH_VERSION

SCHEMA = 'femto-planning-computation-v1'
CONTEXT_SCHEMA = 'femto-planning-context-v1'
LEGACY_ALGORITHMS = dict(route='quality-constrained-surface-v3', target='reference-pixel-clipping-v1')
ALGORITHMS = dict(LEGACY_ALGORITHMS, routing='verified_shared_face_graph_lazy_v2')
AUTO_LAYOUT_ALGORITHMS = dict(ALGORITHMS, routing='verified_shared_face_graph_quarter_portals_v1',
                              layout='fixed-width-auto-layout-v1')
RESIDUAL_LAYOUT_ALGORITHMS = dict(AUTO_LAYOUT_ALGORITHMS,layout='fixed-width-auto-layout-residual-v2')
ORDERED_LAYOUT_ALGORITHMS = dict(RESIDUAL_LAYOUT_ALGORITHMS,routing='verified_shared_face_graph_visit_order_v1')
PATCH_LAYOUT_ALGORITHMS = dict(ORDERED_LAYOUT_ALGORITHMS,layout='fixed-width-local-direction-patches-v1')
STABLE_PATCH_LAYOUT_ALGORITHMS = dict(ORDERED_LAYOUT_ALGORITHMS,layout='fixed-width-local-direction-stable-v2')
REPAIR_PATCH_LAYOUT_ALGORITHMS = dict(ORDERED_LAYOUT_ALGORITHMS,layout='fixed-width-local-direction-repair-v3')
OPTIMIZED_PATCH_LAYOUT_ALGORITHMS = dict(ORDERED_LAYOUT_ALGORITHMS,layout='fixed-width-local-direction-budgeted-v4')
IMPROVED_PATCH_LAYOUT_ALGORITHMS = dict(OPTIMIZED_PATCH_LAYOUT_ALGORITHMS,routing='verified_shared_face_graph_visit_order_two_opt_v2')
SUPPORT_PATCH_LAYOUT_ALGORITHMS = dict(OPTIMIZED_PATCH_LAYOUT_ALGORITHMS,routing='verified_shared_face_graph_visit_order_support_v3')
BALANCED_PATCH_LAYOUT_ALGORITHMS = dict(OPTIMIZED_PATCH_LAYOUT_ALGORITHMS,routing='verified_shared_face_graph_visit_order_balanced_v4')
SEAM_PATCH_LAYOUT_ALGORITHMS = dict(OPTIMIZED_PATCH_LAYOUT_ALGORITHMS,routing='verified_shared_face_graph_visit_order_seam_v5')
SEAM_BALANCED_PATCH_LAYOUT_ALGORITHMS = dict(OPTIMIZED_PATCH_LAYOUT_ALGORITHMS,routing='verified_shared_face_graph_visit_order_seam_balanced_v6')
PRESERVED_DETOUR_PATCH_LAYOUT_ALGORITHMS = dict(OPTIMIZED_PATCH_LAYOUT_ALGORITHMS,routing='verified_shared_face_graph_visit_order_preserved_detour_v7')
SHORT_DETOUR_PATCH_LAYOUT_ALGORITHMS = dict(OPTIMIZED_PATCH_LAYOUT_ALGORITHMS,routing='verified_shared_face_graph_visit_order_short_detour_v8')
JOINT_REORDER_PATCH_LAYOUT_ALGORITHMS = dict(OPTIMIZED_PATCH_LAYOUT_ALGORITHMS,routing='verified_shared_face_graph_visit_order_joint_reorder_v9')
BRIDGE_REORDER_PATCH_LAYOUT_ALGORITHMS = dict(OPTIMIZED_PATCH_LAYOUT_ALGORITHMS,routing='verified_shared_face_graph_visit_order_bridge_reorder_v10')
RELOCATION_PATCH_LAYOUT_ALGORITHMS = dict(OPTIMIZED_PATCH_LAYOUT_ALGORITHMS,routing='verified_shared_face_graph_visit_order_relocation_v11')
DEEP_RELOCATION_PATCH_LAYOUT_ALGORITHMS = dict(OPTIMIZED_PATCH_LAYOUT_ALGORITHMS,routing='verified_shared_face_graph_visit_order_relocation_v12')
RESEARCH_ADAPTIVE_PATCH_ALGORITHMS = dict(BRIDGE_REORDER_PATCH_LAYOUT_ALGORITHMS,layout='research-fixed-width-local-direction-adaptive-half-v6')
RESEARCH_ADAPTIVE_RELOCATION_ALGORITHMS = dict(RELOCATION_PATCH_LAYOUT_ALGORITHMS,layout='research-fixed-width-local-direction-adaptive-half-v6')
RESEARCH_MINIMAL_RELOCATION_ALGORITHMS = dict(RELOCATION_PATCH_LAYOUT_ALGORITHMS,layout='research-fixed-width-local-direction-minimal-v7')
RESEARCH_SHIFT_RELOCATION_ALGORITHMS = dict(RELOCATION_PATCH_LAYOUT_ALGORITHMS,layout='research-fixed-width-local-direction-bounded-shift-v8')
RESEARCH_SHIFT_DEEP_RELOCATION_ALGORITHMS = dict(DEEP_RELOCATION_PATCH_LAYOUT_ALGORITHMS,layout='research-fixed-width-local-direction-bounded-shift-v8')
FOLDBACK_PATCH_LAYOUT_ALGORITHMS = dict(SEAM_BALANCED_PATCH_LAYOUT_ALGORITHMS,layout='fixed-width-local-direction-foldback-v5')
FLOOR_DEEP_RELOCATION_ALGORITHMS = dict(DEEP_RELOCATION_PATCH_LAYOUT_ALGORITHMS,layout='fixed-width-local-direction-spacing-floor-v9')
BOUNDED_FLOOR_ALGORITHMS = dict(FLOOR_DEEP_RELOCATION_ALGORITHMS,routing='verified_shared_face_graph_floor_detour_v13',minimum_metric_fraction=.65)
UNIFIED_FLOOR_ALGORITHMS = dict(BOUNDED_FLOOR_ALGORITHMS,routing=UNIFIED_FLOOR_ROUTING)
ADAPTIVE_UNIFIED_FLOOR_ALGORITHMS = dict((k,v) for k,v in UNIFIED_FLOOR_ALGORITHMS.items() if k!='minimum_metric_fraction')
ADAPTIVE_UNIFIED_FLOOR_ALGORITHMS['layout'] = ADAPTIVE_FLOOR_PATCH_VERSION
COORDINATED_UNIFIED_FLOOR_ALGORITHMS = dict(ADAPTIVE_UNIFIED_FLOOR_ALGORITHMS,layout=COORDINATED_FLOOR_PATCH_VERSION)
GLOBAL_FLOOR_ALGORITHMS = dict(UNIFIED_FLOOR_ALGORITHMS,routing=GLOBAL_FLOOR_ROUTING)
GLOBAL_ADAPTIVE_FLOOR_ALGORITHMS = dict(ADAPTIVE_UNIFIED_FLOOR_ALGORITHMS,routing=GLOBAL_FLOOR_ALGORITHMS['routing'])
GLOBAL_COORDINATED_FLOOR_ALGORITHMS = dict(COORDINATED_UNIFIED_FLOOR_ALGORITHMS,routing=GLOBAL_FLOOR_ALGORITHMS['routing'])
REFINED_COORDINATED_FLOOR_ALGORITHMS = dict(GLOBAL_COORDINATED_FLOOR_ALGORITHMS,routing=REFINED_CONNECTOR_ROUTING)
EXACT_TARGET = 'reference-pixel-exact-partition-v2'
TARGET_ALGORITHMS = (LEGACY_ALGORITHMS['target'], EXACT_TARGET)
NUMERICS = dict(mesh='source_vertices_float32_then_float64_without_reordering',
                clipping='float64_halfspaces_filtered_predicates_v1',
                pixel_cells='[u-0.5,u+0.5) x [v-0.5,v+0.5)',
                area='nominal_mesh_area_not_a_certified_error_bound')
QUALITY = dict(schema='reference-quality-policy-v1', depth_gate_mm=10.0,
               whole_face_rule='projected_bbox_all_pixels_supported',
               registration_uncertainty_mm=None, calibration_uncertainty_mm=None)
DEPENDENCIES = ('image', 'depth', 'calibration', 'geometry', 'reference', 'selection',
                'quality', 'effect_model', 'algorithms', 'numerics')
_METADATA = {'name', 'note', 'created_at', 'updated_at', 'source_mesh', 'boundary_world_m'}


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def array_record(value: Any, dtype: str | None = None) -> dict[str, Any]:
    array = np.ascontiguousarray(value, dtype=dtype)
    if array.dtype.hasobject:
        raise ValueError('计算修订不接受对象数组')
    return dict(shape=list(array.shape), dtype=array.dtype.str,
                sha256=hashlib.sha256(memoryview(array.reshape(-1)).cast('B')).hexdigest())


def canonical_arrays(mesh: Any) -> tuple[np.ndarray, np.ndarray]:
    """Same coordinates and original face order as the region PLY exporter."""
    vertices = np.asarray(mesh.vertices).astype('<f4').astype('<f8')
    faces = np.asarray(mesh.triangles if hasattr(mesh, 'triangles') else mesh.faces, dtype='<i8')
    return vertices, faces


def geometry_record(mesh: Any) -> dict[str, Any]:
    faces = np.asarray(mesh.triangles if hasattr(mesh, 'triangles') else mesh.faces, dtype='<i8')
    native = hashlib.sha256()
    native.update(np.asarray(mesh.vertices, dtype='<f8').tobytes())
    native.update(faces.tobytes())
    vertices, faces = canonical_arrays(mesh)
    return dict(source_sha256=native.hexdigest(), vertices=array_record(vertices), faces=array_record(faces))


def semantic_region(region: dict | None) -> dict | None:
    return None if region is None else {key: value for key, value in region.items() if key not in _METADATA}


def _target_inputs(inputs: dict) -> dict:
    # Region annotation IDs, quality/support classifications and eligibility are
    # deliberately absent. They cannot shrink or rename the requested target.
    selection = inputs['selection'] or {}
    intent = selection.get('selection_intent') or {}
    return {key: inputs[key] for key in ('image', 'depth', 'calibration', 'geometry', 'reference')} | dict(
        intent={key: intent.get(key) for key in ('coordinate_frame', 'image_size', 'selected_mask_sha256',
                                                  'painted_forbidden_sha256', 'selection_method')},
        target_algorithm=inputs['algorithms']['target'], numerics=inputs['numerics'])


def create(inputs: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(inputs, dict) or set(inputs) != set(DEPENDENCIES):
        raise ValueError('计算修订缺少完整输入依赖')
    record = dict(schema=SCHEMA, inputs=deepcopy(inputs), target_id=digest(_target_inputs(inputs)),
                  execution_ready=False, physical_accuracy_verified=False)
    record['sha256'] = digest(record)
    return record


def validate(record: dict) -> dict:
    if (not isinstance(record, dict) or record.get('schema') != SCHEMA
            or record != create(record.get('inputs', {}))):
        raise ValueError('计算修订内容或摘要不一致')
    return record


def capture(model: Any, region: dict | None, head: dict, *, quality: dict | None = None,
            effect_model: dict | None = None, algorithms: dict | None = None,
            numerics: dict | None = None) -> dict:
    """Hash actual loaded evidence, not cached store.digest or asset filenames."""
    projection = model.projection
    task = getattr(model, 'task_context', None)
    task = None if task is None else {key: value for key, value in task.items() if key not in _METADATA}
    inputs = dict(
        image=dict(rgb=array_record(model.rgb), image_valid=array_record(
            getattr(projection, 'image_valid', np.ones(model.rgb.shape[:2], bool)))),
        depth=dict(values=array_record(projection.depth), minimum_m=projection.minimum, maximum_m=projection.maximum),
        calibration=dict(intrinsic=np.asarray(projection.k, float).tolist(), view_reference=deepcopy(model.view_reference)),
        geometry=geometry_record(model.store.mesh),
        reference=dict(origin=deepcopy(model.store.origin_reference), local_context=deepcopy(model.store.local_context),
                       task=deepcopy(task), coordinate_frame='world_rdf', unit='metre'),
        selection=deepcopy(semantic_region(region)),
        quality=deepcopy(quality if quality is not None else getattr(model, 'planning_quality_config', QUALITY)),
        effect_model=deepcopy(effect_model if effect_model is not None else dict(
            schema='existing-v3-width-clearance-model-v1', treatment_head=head, coverage_certificate=None)),
        algorithms=deepcopy((ORDERED_LAYOUT_ALGORITHMS if head.get('mode')=='fixed_width_auto' else ALGORITHMS)
                            if algorithms is None else algorithms),
        numerics=deepcopy(NUMERICS if numerics is None else numerics))
    return create(inputs)


def changed(saved: dict, current: dict) -> list[str]:
    validate(saved); validate(current)
    return [key for key in DEPENDENCIES if saved['inputs'][key] != current['inputs'][key]]


def require_current(model: Any, region: dict, head: dict, revision: dict) -> None:
    differences = changed(revision, capture_for(model, region, head, revision))
    if differences:
        raise ValueError('计算输入已变化，请重新生成路线：' + ', '.join(differences))


def supported_target_algorithms(algorithms: dict) -> bool:
    if not isinstance(algorithms, dict) or algorithms.get('target') not in TARGET_ALGORITHMS:return False
    if algorithms.get('route') == 'source-anchored-surface-v1':
        from femto.planning.source_anchored_adjustment import checked_algorithms as anchored_algorithms
        return anchored_algorithms(algorithms)
    algorithms = dict(algorithms, target=LEGACY_ALGORITHMS['target'])
    if algorithms in (BOUNDED_FLOOR_ALGORITHMS,UNIFIED_FLOOR_ALGORITHMS,ADAPTIVE_UNIFIED_FLOOR_ALGORITHMS,COORDINATED_UNIFIED_FLOOR_ALGORITHMS,GLOBAL_FLOOR_ALGORITHMS,GLOBAL_ADAPTIVE_FLOOR_ALGORITHMS,GLOBAL_COORDINATED_FLOOR_ALGORITHMS,REFINED_COORDINATED_FLOOR_ALGORITHMS):return True
    if 'minimum_metric_fraction' in algorithms:
        fraction=algorithms['minimum_metric_fraction']
        return (type(fraction) in (int,float) and np.isfinite(fraction) and .5<=fraction<=1.
                and {k:v for k,v in algorithms.items() if k!='minimum_metric_fraction'}==FLOOR_DEEP_RELOCATION_ALGORITHMS)
    shift_hash=algorithms.get('row_shifts_sha256')
    if (isinstance(shift_hash,str) and len(shift_hash)==64
            and all(ch in '0123456789abcdef' for ch in shift_hash)
            and {k:v for k,v in algorithms.items() if k!='row_shifts_sha256'} in
                (RESEARCH_SHIFT_RELOCATION_ALGORITHMS,RESEARCH_SHIFT_DEEP_RELOCATION_ALGORITHMS)):
        return True
    if algorithms in (ALGORITHMS,LEGACY_ALGORITHMS,AUTO_LAYOUT_ALGORITHMS,RESIDUAL_LAYOUT_ALGORITHMS,ORDERED_LAYOUT_ALGORITHMS,PATCH_LAYOUT_ALGORITHMS,STABLE_PATCH_LAYOUT_ALGORITHMS,REPAIR_PATCH_LAYOUT_ALGORITHMS,OPTIMIZED_PATCH_LAYOUT_ALGORITHMS,IMPROVED_PATCH_LAYOUT_ALGORITHMS,SUPPORT_PATCH_LAYOUT_ALGORITHMS,BALANCED_PATCH_LAYOUT_ALGORITHMS,SEAM_PATCH_LAYOUT_ALGORITHMS,SEAM_BALANCED_PATCH_LAYOUT_ALGORITHMS,PRESERVED_DETOUR_PATCH_LAYOUT_ALGORITHMS,SHORT_DETOUR_PATCH_LAYOUT_ALGORITHMS,JOINT_REORDER_PATCH_LAYOUT_ALGORITHMS,BRIDGE_REORDER_PATCH_LAYOUT_ALGORITHMS,RELOCATION_PATCH_LAYOUT_ALGORITHMS,DEEP_RELOCATION_PATCH_LAYOUT_ALGORITHMS,FOLDBACK_PATCH_LAYOUT_ALGORITHMS,RESEARCH_ADAPTIVE_PATCH_ALGORITHMS,RESEARCH_ADAPTIVE_RELOCATION_ALGORITHMS,RESEARCH_MINIMAL_RELOCATION_ALGORITHMS,FLOOR_DEEP_RELOCATION_ALGORITHMS):return True
    from femto.planning.coverage_refinement import checked_algorithms
    from femto.planning.strict_path import checked_algorithms as strict_algorithms
    from femto.planning.variable_width_route import checked_algorithms as variable_algorithms
    from femto.planning.variable_width_refinement import checked_algorithms as variable_refinement_algorithms
    return checked_algorithms(algorithms) or strict_algorithms(algorithms) or variable_algorithms(algorithms) or variable_refinement_algorithms(algorithms)


def capture_for(model: Any, region: dict, head: dict, saved: dict) -> dict:
    """Replay a registered supplementary policy while rehashing all live inputs."""
    algorithms = saved['inputs']['algorithms']
    if algorithms.get('route') in ('nominal-variable-width-surface-v1', 'nominal-variable-width-refined-surface-v1'):
        from femto.planning.variable_footprint import validate as validate_profile
        effect = saved['inputs']['effect_model']
        profile = validate_profile(effect['profile'])
        return capture(model, region, head, algorithms=algorithms,
            effect_model=dict(schema='variable-footprint-width-clearance-model-v1', treatment_head=head,
                              profile=profile, coverage_certificate=None))
    return capture(model, region, head, algorithms=algorithms if supported_target_algorithms(algorithms) else None)


def install_allowed(*, request: int, current_request: int, cancelled: bool,
                    revision: Any = None, current_revision: Any = None) -> bool:
    """Request generation and complete calculation binding are separate gates."""
    return not cancelled and request == current_request and revision == current_revision


def validate_context(context: dict, *, region: dict | None = None, mesh: Any = None,
                     algorithm: str | None = None) -> dict:
    if not isinstance(context, dict) or set(context) != {'schema', 'revision', 'target'} or context['schema'] != CONTEXT_SCHEMA:
        raise ValueError('目标台账与计算修订上下文无效')
    revision = validate(context['revision'])
    target = context['target']
    if (target.get('computation_sha256') != revision['sha256']
            or target.get('target_id') != revision['target_id']
            or target.get('target_completion') != 'unverified'
            or target.get('execution_ready') is not False or target.get('physical_accuracy_verified') is not False):
        raise ValueError('目标台账混用了不同计算修订或完成状态')
    if algorithm is not None and revision['inputs']['algorithms']['route'] != algorithm:
        raise ValueError('计算修订与路线算法不一致')
    if region is not None and revision['inputs']['selection'] != semantic_region(region):
        raise ValueError('计算修订与原选区不一致')
    if mesh is not None and revision['inputs']['geometry'] != geometry_record(mesh):
        raise ValueError('计算修订与源网格不一致')
    return revision
