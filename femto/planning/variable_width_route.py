"""Apply a nominal variable footprint to an unchanged, constructed P4 route.

The generating/search envelope is retained as a conservative upper bound.
Every existing scan and connector edge is rechecked with its segment bound;
this operation never silently modifies the path or bridges an existing gap.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import math

from femto.planning import planning_revision as revisions, variable_footprint as footprint, strict_path
from femto.planning import quality_constrained as quality
from femto.planning.surface_trajectory import SurfaceTrajectory, ScanSegment, canonical_json_bytes, validate_trajectory

ALGORITHM = 'nominal-variable-width-surface-v1'
SCHEMA = 'femto-variable-width-route-v1'


def algorithms_for(base: SurfaceTrajectory, model: dict) -> dict:
    from femto.planning.coverage_workflow import route_binding
    return dict(route=ALGORITHM, target=base.planning_context['revision']['inputs']['algorithms']['target'], footprint=dict(
        schema=SCHEMA, model_sha256=model['sha256'], base_route=route_binding(base)))


def checked_algorithms(algorithms: dict) -> bool:
    if not isinstance(algorithms, dict) or set(algorithms) != {'route', 'target', 'footprint'}:return False
    binding = algorithms.get('footprint')
    return (algorithms['route'] == ALGORITHM and algorithms['target'] in revisions.TARGET_ALGORITHMS
            and isinstance(binding, dict) and set(binding) == {'schema', 'model_sha256', 'base_route'}
            and binding['schema'] == SCHEMA and isinstance(binding['model_sha256'], str)
            and len(binding['model_sha256']) == 64 and isinstance(binding['base_route'], dict)
            and binding['base_route'].get('algorithm') == strict_path.ALGORITHM)


def context_for(base: SurfaceTrajectory, model: dict) -> dict:
    inputs = deepcopy(base.planning_context['revision']['inputs'])
    inputs['algorithms'] = algorithms_for(base, model)
    inputs['effect_model'] = dict(schema='variable-footprint-width-clearance-model-v1',
        treatment_head=inputs['effect_model']['treatment_head'], profile=deepcopy(model), coverage_certificate=None)
    revision = revisions.create(inputs)
    target = deepcopy(base.planning_context['target']);target['computation_sha256'] = revision['sha256']
    if target.get('construction_pages') is not None:
        from femto.planning.target_paging import rebind_header_records
        target['construction_pages']=rebind_header_records(target,revision)
    target['sha256'] = revisions.digest({k:v for k,v in target.items() if k != 'sha256'})
    return dict(schema=revisions.CONTEXT_SCHEMA, revision=revision, target=target)


def _diameter(upper: float, generation_width: float) -> float:
    # The existing outer predicate has a 0.5 mm minimum. Using that larger
    # envelope for narrower models is conservative; inner radii are not clamped.
    value = max(.5, upper*2000)
    if value/2000 < upper:value = math.nextafter(value, math.inf)
    if value > generation_width and generation_width/2000 >= upper:value = generation_width
    return value


def _bounds(base: SurfaceTrajectory, model: dict, check=lambda: None) -> list[dict]:
    if base.planner['algorithm_version'] != strict_path.ALGORITHM or base.planning_context is None:
        raise ValueError('请先恢复补充前路线并完成严格重建，再应用变宽模型')
    rows = footprint.edge_bounds(base, model, check)
    width = base.planner['parameters']['effective_width_mm']
    if any(row['status'] != 'bounded_nominal' for row in rows):
        raise quality.PlanningConstraintBlocked('仅有离散宽度样本，整段上下界未核验；请提供明确插值规则。',
            dict(schema=SCHEMA, status='unconfirmed_interpolation', model=model,
                 execution_ready=False, physical_accuracy_verified=False))
    if any(row['upper_half_width_m'] > width/2000 for row in rows):
        raise quality.PlanningConstraintBlocked('模型段内上界超过当前路线的生成宽度；请调大有效作用宽度并重新生成、严格重建。',
            dict(schema=SCHEMA, status='outer_bound_exceeds_generation_envelope', model=model,
                 generation_width_mm=width, execution_ready=False, physical_accuracy_verified=False))
    for row in rows:row['outer_check_width_mm'] = _diameter(row['upper_half_width_m'], width)
    return rows


def make(base: SurfaceTrajectory, model: dict, report: dict) -> SurfaceTrajectory:
    planner = dict(base.planner, kind='variable_width_surface', algorithm_version=ALGORITHM,
                   parameters=dict(base.planner['parameters'], variable_footprint=deepcopy(model)))
    return replace(base, planner=planner, planning_context=context_for(base, model),
                   gap_report=dict(base.gap_report, variable_width=report))


def apply(base: SurfaceTrajectory, model: dict, mesh, region, check=lambda: None) -> SurfaceTrajectory:
    from femto.planning.effective_domain import validate as validate_domain
    rows = _bounds(base, model, check)
    params = base.planner['parameters']
    allowed = validate_domain(params['effective_domain'], params['quality_evidence'], params['effective_width_mm'], mesh, region, check)
    budget = strict_path.SupportBudget(**params['surface_support']['budget']).checked()
    engine = strict_path.engine_for(mesh, allowed, budget, check, region=region)
    checker = strict_path.Checks(engine)
    diagnostics = []
    flags, reasons, margins = quality.edge_checks([s for s in base.ordered_items if isinstance(s, ScanSegment)],
        mesh, params['quality_evidence'], params['effective_width_mm'], check, allowed_faces=allowed,
        support_checker=checker, diagnostics=diagnostics, edge_widths_mm=[row['outer_check_width_mm'] for row in rows])
    for row, flag, reason, margin, diagnostic, support in zip(rows, flags, reasons, margins, diagnostics, checker.decisions):
        row.update(accepted=bool(flag), reason=reason, residual_margin_mm=float(margin),
                   quality_diagnostic=diagnostic, support_check=support)
    report = dict(schema=SCHEMA, base_trajectory=base.to_dict(), records=rows,
        checked_edges=len(rows), accepted_edges=int(sum(flags)), connector_edges=sum(row['role']=='connector' for row in rows),
        zero_inner_edges=sum(row['lower_half_width_m']==0 and row['role']!='connector' for row in rows),
        generation_envelope_width_mm=params['effective_width_mm'], model_sha256=model['sha256'],
        preserved_points_and_anchors=True, preserved_connection_search_envelope=True,
        scope='nominal_model_surface_geometry_only', offset_bound_applied=False,
        physical_accuracy_verified=False, execution_ready=False)
    if not flags.all():
        raise quality.PlanningConstraintBlocked('变宽模型仍有边未通过完整检查；原路线与覆盖保留，详见检查记录。', report)
    result = make(base, model, report)
    if len(canonical_json_bytes(result.to_dict())) > 64*1024*1024:
        raise ValueError('变宽路线超过原 64 MiB 记录预算')
    return result


def validate(trajectory: SurfaceTrajectory, mesh=None, region=None, check=lambda: None) -> None:
    from femto.planning.refinement_validation import semantic_payload
    if len(canonical_json_bytes(trajectory.to_dict())) > 64*1024*1024:
        raise ValueError('变宽路线超过原 64 MiB 记录预算')
    report = trajectory.gap_report.get('variable_width')
    if not isinstance(report, dict) or report.get('schema') != SCHEMA:raise ValueError('变宽路线证据缺失')
    base = SurfaceTrajectory.from_dict(report['base_trajectory'])
    model = footprint.validate(trajectory.planner['parameters']['variable_footprint'])
    expected_bounds = _bounds(base, model, check)
    rows = report['records']
    if not isinstance(rows, list) or len(rows) != len(expected_bounds):raise ValueError('变宽逐边记录缺失')
    for row, bound in zip(rows, expected_bounds):
        if (set(row) != set(bound) | {'accepted', 'reason', 'residual_margin_mm', 'quality_diagnostic', 'support_check'}
                or canonical_json_bytes({k:row.get(k) for k in bound}) != canonical_json_bytes(bound) or row.get('accepted') is not True
                or row.get('reason') != 'accepted' or row.get('support_check', {}).get('status') != 'passed'
                or row.get('quality_diagnostic', {}).get('accepted') is not True):
            raise ValueError('变宽界、原边身份或准入证据不一致')
    expected_report = dict(schema=SCHEMA, base_trajectory=base.to_dict(), records=rows,
        checked_edges=len(rows), accepted_edges=len(rows), connector_edges=sum(row['role']=='connector' for row in rows),
        zero_inner_edges=sum(row['lower_half_width_m']==0 and row['role']!='connector' for row in rows),
        generation_envelope_width_mm=base.planner['parameters']['effective_width_mm'], model_sha256=model['sha256'],
        preserved_points_and_anchors=True, preserved_connection_search_envelope=True,
        scope='nominal_model_surface_geometry_only', offset_bound_applied=False,
        physical_accuracy_verified=False, execution_ready=False)
    if canonical_json_bytes(report) != canonical_json_bytes(expected_report):raise ValueError('变宽路线摘要与逐边记录不一致')
    if semantic_payload(make(base, model, report)) != semantic_payload(trajectory):
        raise ValueError('变宽模型改变了原严格路线、原目标或修订绑定')
    if mesh is not None or region is not None:validate_trajectory(base, mesh=mesh, region=region, check=check)
    if mesh is not None:
        if region is None:region = base.planning_context['revision']['inputs']['selection']
        if semantic_payload(apply(base, model, mesh, region, check)) != semantic_payload(trajectory):
            raise ValueError('变宽模型上下界、逐边准入或支持证据无法重现')


def scan_endpoints(trajectory: SurfaceTrajectory, original_faces) -> list[dict]:
    from femto.planning.coverage_domain import scan_endpoints as fixed_endpoints
    base = SurfaceTrajectory.from_dict(trajectory.gap_report['variable_width']['base_trajectory'])
    rows = {(r['segment_index'], r['edge_index']):r for r in trajectory.gap_report['variable_width']['records']}
    seeds = fixed_endpoints(base, original_faces)
    for seed in seeds:
        row = rows[seed['segment_index'], seed['edge_index']]
        seed.update(width_interval=deepcopy(row['interval']), lower_half_width_m=row['lower_half_width_m'])
    return seeds


def summary_text(trajectory: dict | SurfaceTrajectory) -> str:
    values = trajectory if isinstance(trajectory, dict) else trajectory.to_dict()
    report = values['gap_report']['variable_width']
    return (footprint.summary_text(values['planner']['parameters']['variable_footprint'])
            + f" 扫描与连接共 {report['checked_edges']} 条边已按段内上界核验；"
            + f"{report['zero_inner_edges']} 条扫描边的内包半径为零，不计覆盖。")
