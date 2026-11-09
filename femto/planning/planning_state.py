"""Versioned planning provenance and a single UI state decision, independent of rendering."""
from __future__ import annotations
from femto.performance_detail import observed
from typing import Any, Callable

from functools import wraps
from copy import deepcopy
def signature(value: Any) -> str:
    from femto.planning.region_path import signature as digest
    return digest(value)

BINDING_VERSION = 'femto-plan-binding-v1'
ALGORITHMS = {'raster': 'raster-v1', 'metric': 'mesh-contours-v1',
              'coverage': 'mesh-coverage-v1', 'serpentine': 'serpentine-v1', 'serpentine_v2':'serpentine-v2', 'serpentine_v3':'serpentine-v3', 'serpentine_v4':'serpentine-v4', 'surface_spacing':'mesh-surface-spacing-v1',
              'mesh_slice':'mesh-slice-serpentine-v1', 'mesh_slice_connected':'mesh-slice-serpentine-v2',
              'mesh_slice_blocks':'mesh-slice-serpentine-v3.1', 'mesh_slice_selection_blocks':'mesh-slice-serpentine-v4',
              'mesh_slice_selection_blocks_optimized':'mesh-slice-serpentine-v4.1',
              'mesh_slice_selection_blocks_face_portals':'mesh-slice-serpentine-v4.2',
              'mesh_slice_selection_blocks_bidirectional':'mesh-slice-serpentine-v4.3',
              'mesh_slice_selection_blocks_ordered':'mesh-slice-serpentine-v4.4'}


def trajectory_binding(trajectory: Any) -> dict[str, Any]:
    """Small immutable binding view for export/history/UI state."""
    from femto.planning.surface_trajectory import SurfaceTrajectory, validate_trajectory
    model = SurfaceTrajectory.from_dict(trajectory) if isinstance(trajectory, dict) else trajectory
    validate_trajectory(model)
    payload = model.to_dict()
    return dict(schema='femto-trajectory-binding-v1',region_id=model.source.get('region_id'),
                region_geometry_sha256=model.source.get('region_signature'),
                geometry_sha256=model.source.get('geometry_sha256'),
                parameters=deepcopy(model.planner.get('parameters')),
                algorithm_version=model.planner.get('algorithm_version'),
                result_sha256=payload['result_sha256'],source=deepcopy(model.source),
                execution_ready=False,physical_accuracy_verified=False)


def region_signature(region: Any) -> str:
    # Metadata and portable asset locations are not planning inputs.
    return signature({k: v for k, v in region.items() if k not in
        ('name', 'note', 'created_at', 'updated_at', 'source_mesh', 'boundary_world_m')})


def parameters(plan: dict[str, Any]) -> dict[str, Any]:
    if plan.get('schema') == 'femto-workspace-surface-plan-v1':
        return dict(kind='mesh_slice_selection_blocks_ordered',
                    values={'treatment_head': deepcopy(plan['treatment_head'])})
    if 'coverage_parameters' in plan:
        return dict(kind='surface_spacing' if plan['coverage_parameters'].get('pattern')=='surface_spacing_v1' else 'serpentine_v4' if plan['coverage_parameters'].get('pattern')=='serpentine_v4' else 'serpentine_v3' if plan['coverage_parameters'].get('pattern')=='serpentine_v3' else 'serpentine_v2' if plan['coverage_parameters'].get('pattern')=='serpentine_v2' else 'serpentine' if plan['coverage_parameters'].get('pattern') == 'serpentine_v1' else 'coverage',
                    values=deepcopy(plan['coverage_parameters']))
    if plan['schema'] == 'femto-region-metric-path-v1':
        return dict(kind='metric', values=dict(spacing_mm=plan['spacing_mm'], sample_step_mm=plan['sample_step_mm']))
    return dict(kind='raster', values=dict(row_step_px=plan['row_step_px'], max_step_mm=plan['max_step_mm']))


def binding(mesh: Any, region: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    from femto.annotation.scene_annotations import mesh_digest
    params = parameters(plan)
    if params['kind'] == 'mesh_slice_selection_blocks_ordered':
        keys = ('segments', 'chain_breaks', 'route_order', 'blocks', 'surface_mode',
                'spacing_report', 'spacing_mm', 'spacing_semantics', 'selection_semantics',
                'motion_demo', 'treatment_head', 'surface_trajectory',
                'execution_ready', 'physical_accuracy_verified')
        if 'planning_context' in plan:
            keys += ('planning_context',)
        if 'coverage_evidence' in plan:
            keys += ('coverage_evidence',)
        return dict(schema=BINDING_VERSION, region_id=region['annotation_id'],
                    region_geometry_sha256=region_signature(region), geometry_sha256=mesh_digest(mesh),
                    parameters=params, algorithm_version=plan['surface_trajectory']['planner']['algorithm_version'],
                    result_sha256=signature({key: plan.get(key) for key in keys}))
    return dict(schema=BINDING_VERSION, region_id=region['annotation_id'],
                region_geometry_sha256=region_signature(region), geometry_sha256=mesh_digest(mesh),
                parameters=params, algorithm_version=ALGORITHMS[params['kind']],
                result_sha256=signature({k: plan.get(k) for k in
                    (('segments', 'coverage', 'boundary_check', 'image_support', 'exclusions','chain_breaks','start_selection','route_order','motion_demo',*(['coverage_inspection'] if params['kind'] in ('serpentine_v3','serpentine_v4','surface_spacing') else []),*(['route_quality'] if params['kind']=='serpentine_v4' else []),*(['surface_mode','spacing_report'] if params['kind']=='surface_spacing' else [])) if params['kind'] in ('serpentine_v2','serpentine_v3','serpentine_v4','surface_spacing') else ('segments', 'coverage', 'boundary_check', 'image_support', 'exclusions'))}))


def versioned(fn: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(fn)
    def generate(projection, mesh, region, *args, **kwargs):
        plan = fn(projection, mesh, region, *args, **kwargs)
        plan['region_signature'] = region_signature(region)
        plan['plan_binding'] = binding(mesh, region, plan)
        return plan
    return generate


def validate_binding(plan: Any, mesh: Any, region: Any, check=lambda: None) -> None:
    """Check original fingerprints before relocation; not a geometry proof."""
    check()
    if plan.get('plan_binding') != binding(mesh, region, plan):
        raise ValueError('Path samples/source/report mismatch：方案来源、几何、参数或算法版本不匹配；请重新生成路径')
    if plan.get('region_signature') != region_signature(region):
        raise ValueError('Region changed; regenerate path')
    check()


def _validate(plan: Any, mesh: Any, region: Any, check=lambda: None):
    """Run all existing checks once and retain the validated surface object."""
    validate_binding(plan, mesh, region, check=check)
    if plan.get('schema') == 'femto-workspace-surface-plan-v1':
        from femto.planning.surface_workspace import validate as validate_surface
        trajectory = validate_surface(plan, mesh, region, check=check)
        from femto.planning.coverage_workflow import check_binding
        check_binding(plan)
        return trajectory
    return None


def validate(plan: Any, mesh: Any, region: Any, check=lambda: None) -> None:
    # Preserve the public validation-only return contract.
    _validate(plan, mesh, region, check)


@observed('save.validate_current_geometry')
def validated_surface(plan: Any, mesh: Any, region: Any, check=lambda: None):
    """Return this invocation's checked route, never an earlier acceptance."""
    original_binding = deepcopy(plan.get('plan_binding'))
    trajectory = _validate(plan, mesh, region, check)
    if trajectory is None:
        raise ValueError('模型覆盖核验需要当前工作台三维路线')
    # Detect intervening mutations before reusing the checked object. This is
    # a fingerprint/identity check, not a cached geometric acceptance.
    validate_binding(plan, mesh, region, check=check)
    if plan.get('plan_binding') != original_binding:
        raise ValueError('三维路线在覆盖核验期间发生变化，请重新核验')
    return trajectory


def state(region: Any, plan: Any, *, dirty: bool=False, pending: bool=False, busy: bool=False, error: Any=None, request: Any=None, geometry_digest: str | None=None, draft: bool=False, computation_revision: dict | None=None) -> dict[str, Any]:
    model_error = None
    valid = bool(region and plan and plan.get('plan_binding') and
                 plan['plan_binding'].get('schema') == BINDING_VERSION and
                 plan['plan_binding'].get('algorithm_version') == (plan['surface_trajectory']['planner']['algorithm_version']
                    if plan.get('schema') == 'femto-workspace-surface-plan-v1' else ALGORITHMS.get(parameters(plan)['kind'])) and
                 plan['plan_binding'].get('parameters') == parameters(plan) and
                 (geometry_digest is None or plan['plan_binding'].get('geometry_sha256') == geometry_digest) and
                 plan['plan_binding']['region_id'] == region['annotation_id'] and
                 plan['plan_binding']['region_geometry_sha256'] == region_signature(region) and
                 (request is None or parameters(plan) == request))
    if valid and plan.get('planning_context') is not None:
        from femto.planning.planning_revision import validate_context
        try:
            recorded = validate_context(plan['planning_context'], region=region)
            valid = computation_revision is None or recorded == computation_revision
        except (ValueError, KeyError, TypeError):
            valid = False
    if valid and plan.get('surface_trajectory', {}).get('planner', {}).get('algorithm_version') in ('nominal-variable-width-surface-v1', 'nominal-variable-width-refined-surface-v1'):
        from femto.planning.variable_footprint import require_current
        try:require_current(plan['surface_trajectory']['planner']['parameters']['variable_footprint'])
        except (ValueError, KeyError, TypeError) as failure:
            valid = False;model_error = str(failure)
    if not region: phase, reason = 'no_region', '无区域：请单选已完成的主区域'
    elif busy: phase, reason = 'generating', '生成中：请等待，或取消任务'
    elif error: phase, reason = 'failed', '生成失败：' + error
    elif valid:
        phase, reason = 'valid', '路径有效'
        if plan.get('spacing_report',{}).get('status')=='not_fully_verified':reason+='；表面间距未完全满足或无法证实，详见间距报告'
    elif plan: phase, reason = 'invalid', model_error or '路径失效：区域、禁区或参数已改变，请重新生成'
    else: phase, reason = 'not_generated', '未生成路径：请先生成路线'
    usable = valid and not busy and not draft
    return dict(phase=phase, reason=reason, valid=usable,
                save_state='pending' if pending else ('dirty' if dirty else 'saved'),
                can_generate=bool(region) and not busy and not draft, can_use=usable,
                can_save=bool(region) and not busy and not draft)
