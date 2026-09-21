"""Versioned planning provenance and a single UI state decision, independent of rendering."""
from __future__ import annotations
from typing import Any, Callable

from functools import wraps
from copy import deepcopy
def signature(value: Any) -> str:
    from femto.planning.region_path import signature as digest
    return digest(value)

BINDING_VERSION = 'femto-plan-binding-v1'
ALGORITHMS = {'raster': 'raster-v1', 'metric': 'mesh-contours-v1',
              'coverage': 'mesh-coverage-v1', 'serpentine': 'serpentine-v1', 'serpentine_v2':'serpentine-v2', 'serpentine_v3':'serpentine-v3', 'serpentine_v4':'serpentine-v4', 'surface_spacing':'mesh-surface-spacing-v1',
              'mesh_slice':'mesh-slice-serpentine-v1'}


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
    if 'coverage_parameters' in plan:
        return dict(kind='surface_spacing' if plan['coverage_parameters'].get('pattern')=='surface_spacing_v1' else 'serpentine_v4' if plan['coverage_parameters'].get('pattern')=='serpentine_v4' else 'serpentine_v3' if plan['coverage_parameters'].get('pattern')=='serpentine_v3' else 'serpentine_v2' if plan['coverage_parameters'].get('pattern')=='serpentine_v2' else 'serpentine' if plan['coverage_parameters'].get('pattern') == 'serpentine_v1' else 'coverage',
                    values=deepcopy(plan['coverage_parameters']))
    if plan['schema'] == 'femto-region-metric-path-v1':
        return dict(kind='metric', values=dict(spacing_mm=plan['spacing_mm'], sample_step_mm=plan['sample_step_mm']))
    return dict(kind='raster', values=dict(row_step_px=plan['row_step_px'], max_step_mm=plan['max_step_mm']))


def binding(mesh: Any, region: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    from femto.annotation.scene_annotations import mesh_digest
    params = parameters(plan)
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


def validate(plan: Any, mesh: Any, region: Any) -> None:
    if plan.get('plan_binding') != binding(mesh, region, plan):
        raise ValueError('Path samples/source/report mismatch：方案来源、几何、参数或算法版本不匹配；请重新生成路径')
    if plan.get('region_signature') != region_signature(region):
        raise ValueError('Region changed; regenerate path')


def state(region: Any, plan: Any, *, dirty: bool=False, pending: bool=False, busy: bool=False, error: Any=None, request: Any=None, geometry_digest: str | None=None, draft: bool=False) -> dict[str, Any]:
    valid = bool(region and plan and plan.get('plan_binding') and
                 plan['plan_binding'].get('schema') == BINDING_VERSION and
                 plan['plan_binding'].get('algorithm_version') == ALGORITHMS.get(parameters(plan)['kind']) and
                 plan['plan_binding'].get('parameters') == parameters(plan) and
                 (geometry_digest is None or plan['plan_binding'].get('geometry_sha256') == geometry_digest) and
                 plan['plan_binding']['region_id'] == region['annotation_id'] and
                 plan['plan_binding']['region_geometry_sha256'] == region_signature(region) and
                 (request is None or parameters(plan) == request))
    if not region: phase, reason = 'no_region', '无区域：请单选已完成的主区域'
    elif busy: phase, reason = 'generating', '生成中：请等待，或取消任务'
    elif error: phase, reason = 'failed', '生成失败：' + error
    elif valid:
        phase, reason = 'valid', '路径有效'
        if plan.get('spacing_report',{}).get('status')=='not_fully_verified':reason+='；表面间距未完全满足或无法证实，详见间距报告'
    elif plan: phase, reason = 'invalid', '路径失效：区域、禁区或参数已改变，请重新生成'
    else: phase, reason = 'not_generated', '未生成路径：请先生成路线'
    usable = valid and not busy and not draft
    return dict(phase=phase, reason=reason, valid=usable,
                save_state='pending' if pending else ('dirty' if dirty else 'saved'),
                can_generate=bool(region) and not busy and not draft, can_use=usable,
                can_save=bool(region) and not busy and not draft)
