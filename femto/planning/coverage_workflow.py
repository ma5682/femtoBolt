"""Product bindings for optional nominal coverage, separate from P1/v3 formats.

The route and target keep their original versions. Relocatable file paths and
display-only trajectory fields are excluded from the v2 route binding; all
numerical route contents, reports, inputs and W0 reference remain bound.
"""
from copy import deepcopy
import numpy as np

from femto.planning import coverage_archive, coverage_certificate as coverage, footprint_model
from femto.planning.coverage_domain import CoverageDomain, from_plan, scan_endpoints
from femto.planning.planning_revision import digest
from femto.planning.surface_trajectory import SurfaceTrajectory
from femto.operation_diagnostics import diagnosed
from femto.geometry.region_reuse import operation as geometry_operation
from femto.planning.evidence_preparation import operation as evidence_operation

KEY = 'coverage_evidence'
ROUTE_SCHEMA = 'femto-accepted-v3-coverage-source-v2'


class RecalculationRejected(ValueError):
    """A new proof cannot replace a same-input proof with a larger known set."""
    def __init__(self, comparison):
        self.comparison = deepcopy(comparison)
        area = comparison['lost_nominal_area_m2']*1e6
        super().__init__(f'本次预算内未能重新确认原覆盖中的 {area:.3f} mm²；原覆盖已保留，可重新核验。')


@diagnosed('coverage.recalculation_admission')
def require_recalculation_retention(previous, current, ledger, check=lambda:None):
    """Compare already computed/replayed proofs; never merge or inherit states."""
    coverage.validate_metadata(previous.metadata); coverage.validate_metadata(current.metadata)
    old, new = previous.metadata['inputs'], current.metadata['inputs']
    keys = ('domain', 'seeds_sha256', 'footprint', 'route')
    if any(old[key] != new[key] or digest(old[key]) != digest(new[key]) for key in keys):
        raise ValueError('覆盖复算必须绑定同一原目标、路线、扫描种子和作用模型')
    if (new['domain'].get('target_sha256') != ledger.metadata['sha256']
            or new['domain'].get('scope') != 'p1_original_reference_selection'):
        raise ValueError('覆盖复算比较与当前原目标台账不一致')
    from femto.planning.coverage_partition import compare
    comparison = compare(previous, current, ledger.arrays['area_m2'], check)
    if comparison['lost_weight_units']:
        raise RecalculationRejected(comparison)
    return comparison


def route_binding(trajectory):
    payload = trajectory.to_dict()
    algorithm = trajectory.planner['algorithm_version']
    if algorithm not in ('quality-constrained-surface-v3', 'source-anchored-surface-v1', 'coverage-refined-surface-v1', 'constructed-surface-route-v1', 'constructed-coverage-refined-surface-v1', 'nominal-variable-width-surface-v1', 'nominal-variable-width-refined-surface-v1') or trajectory.planning_context is None:
        raise ValueError('模型覆盖核验需要当前 v3 路线与原目标台账，请生成新版本')
    numerical = {key:value for key,value in payload.items()
                 if key not in ('source', 'route_diagram_status', 'data_layers')}
    numerical['source'] = {key:trajectory.source[key]
                           for key in ('geometry_sha256', 'region_id', 'region_signature')}
    if algorithm in ('coverage-refined-surface-v1', 'constructed-coverage-refined-surface-v1', 'nominal-variable-width-refined-surface-v1'):
        # Round coverage refers to this route. Its own proof metadata cannot
        # also be an input to itself. The complete refinement report is replayed
        # separately; its source, policy and base proof are in the revision.
        numerical['gap_report'] = {k:v for k,v in numerical['gap_report'].items() if k != 'refinement'}
    if algorithm == 'source-anchored-surface-v1':
        from femto.planning.source_anchored_adjustment import semantic
        numerical = semantic(payload)
    schema = ('femto-source-anchored-coverage-source-v1' if algorithm == 'source-anchored-surface-v1' else
              'femto-accepted-variable-width-refinement-coverage-source-v1' if algorithm == 'nominal-variable-width-refined-surface-v1' else
              'femto-accepted-variable-width-coverage-source-v1' if algorithm == 'nominal-variable-width-surface-v1' else
              ROUTE_SCHEMA if algorithm == 'quality-constrained-surface-v3' else
              'femto-accepted-refinement-coverage-source-v1' if algorithm == 'coverage-refined-surface-v1' else
              'femto-accepted-constructed-coverage-source-v1')
    return dict(schema=schema, route_semantics_sha256=digest(numerical),
                result_sha256=payload['result_sha256'],
                computation_sha256=trajectory.planning_context['revision']['sha256'],
                algorithm=trajectory.planner['algorithm_version'], accepted_scan_edges_only=True)


def bundle(trajectory, ledger):
    """Use only after the caller has validated the route and target assets."""
    context = trajectory.planning_context
    if ledger.metadata != context['target']:raise ValueError('覆盖与原目标台账不属于同一修订')
    domain = CoverageDomain._checked_target(ledger, context['revision'])
    seeds = scan_endpoints(trajectory, ledger.arrays['region_faces'])
    model = footprint_model.for_trajectory(trajectory)
    return domain, seeds, model, route_binding(trajectory)


def check_binding(plan):
    """Cheap association/structure check, never a substitute for proof replay."""
    metadata = plan.get(KEY)
    if metadata is None:return
    coverage.validate_metadata(metadata)
    trajectory = SurfaceTrajectory.from_dict(plan['surface_trajectory'])
    context = plan.get('planning_context')
    if context is None or context != trajectory.planning_context:raise ValueError('覆盖缺少相同的原目标计算修订')
    inputs = metadata['inputs']; domain = inputs['domain']
    algorithms = coverage.P2_ALGORITHMS
    if (trajectory.planner['algorithm_version'] in ('coverage-refined-surface-v1', 'constructed-coverage-refined-surface-v1', 'nominal-variable-width-refined-surface-v1')
            and trajectory.gap_report.get('refinement', {}).get('coverage_policy') is not None):
        algorithms += (coverage.reproof_algorithm_for_policy(trajectory.gap_report['refinement']['coverage_policy']),)
    generation_width = footprint_model.nominal(plan['treatment_head']['effective_treatment_width_mm'])['effective_width_mm']
    seeds = scan_endpoints(trajectory, np.asarray(context['revision']['inputs']['selection']['triangle_indices']))
    if (metadata['algorithm'] not in algorithms or inputs['route'] != route_binding(trajectory)
            or generation_width != trajectory.planner['parameters']['effective_width_mm']
            or inputs['seeds_sha256'] != digest(seeds)
            or inputs['footprint'] != footprint_model.for_trajectory(trajectory)
            or domain.get('scope') != 'p1_original_reference_selection'
            or domain.get('target_sha256') != context['target']['sha256']
            or domain.get('computation_sha256') != context['revision']['sha256']):
        raise ValueError('覆盖证据与当前路线、原目标或作用宽度不一致')


@diagnosed('coverage.operation', operation=True)
@geometry_operation()
@evidence_operation()
def calculate(model, region, plan, check=lambda:None, progress=lambda text:None, *, budget=None):
    from femto.planning.target_surface import is_deferred
    if is_deferred((plan.get('planning_context') or {}).get('target')):
        raise ValueError('目标面积尚未计算，请先计算目标面积再核验覆盖')
    if plan['surface_trajectory']['planner']['algorithm_version'] in ('nominal-variable-width-surface-v1', 'nominal-variable-width-refined-surface-v1'):
        from femto.planning.variable_footprint import require_current
        require_current(plan['surface_trajectory']['planner']['parameters']['variable_footprint'])
    if plan['surface_trajectory']['planner']['algorithm_version'] in ('coverage-refined-surface-v1', 'constructed-coverage-refined-surface-v1', 'nominal-variable-width-refined-surface-v1'):
        raise ValueError('补充版本已保存逐轮核验证据；请恢复补充前路线后重新核验')
    progress('核对原目标、作用宽度与同一条扫描路线')
    domain, seeds, footprint, route = from_plan(model, region, plan, check,
                                               binding_factory=route_binding)
    previous = cached(model, plan)
    guided = previous is not None and footprint['schema'] == footprint_model.SCHEMA
    if guided:
        expected = dict(domain=domain.binding, route=route, seeds_sha256=digest(seeds), footprint=footprint)
        if any(previous.metadata['inputs'][key] != value for key, value in expected.items()):
            raise ValueError('原集合优先复算必须绑定同一原目标、路线、扫描种子和作用模型')
    progress('核验模型覆盖：先重新确认原覆盖，再检查剩余范围；可取消' if guided else
             '核验模型覆盖：先处理原目标，再细分边界；可取消')
    paged=coverage.checked_page_binding(domain.binding)
    certificate = coverage.evaluate(domain, seeds, footprint, route_binding=route,
        algorithm=((coverage.GROUPED_RECALCULATION_ALGORITHM if guided else coverage.PAGED_ALGORITHM) if paged
                   else (coverage.RECALCULATION_ALGORITHM if guided else coverage.CURRENT_ALGORITHM)),
        budget=budget, check=check, **({'guide':previous} if guided else {}))
    if footprint['schema'] == 'femto-nominal-piecewise-ball-v1':
        from femto.planning.variable_footprint import require_current
        require_current(footprint)
    check()
    return certificate, domain.ledger


@geometry_operation()
@evidence_operation()
def attach(model, region, plan, certificate, ledger, *, check=lambda:None):
    """Install a locally computed or fully replayed certificate, not imported metadata."""
    from femto.planning.planning_state import binding
    result = deepcopy(plan)
    result[KEY] = deepcopy(certificate.metadata)
    check_binding(result)
    previous = cached(model, plan)
    if previous is not None:
        require_recalculation_retention(previous, certificate, ledger, check)
    result['plan_binding'] = binding(model.store.mesh, region, result)
    from femto.planning.coverage_display import build
    display = build(ledger, certificate, check)
    # Preparing the display may fail or be cancelled. Install all caches only
    # after it succeeds, so an earlier saved certificate remains usable.
    model.coverage_certificate = certificate
    model.target_ledger = ledger
    model.coverage_display = display
    return result


def cached(model, plan):
    metadata = (plan or {}).get(KEY)
    if metadata is None:return None
    certificate = getattr(model, 'coverage_certificate', None)
    if certificate is None or certificate.metadata != metadata:
        raise ValueError('模型覆盖证据资产缺失；请重新核验后保存')
    return certificate


def write(folder, certificate, trajectory, ledger, check=lambda:None, *, verified_assets=None):
    if trajectory.planner['algorithm_version'] == 'nominal-variable-width-surface-v1' and verified_assets is not None:
        from femto.planning.refinement_archive import require_verified_assets
        from femto.planning.planning_io import durable
        from pathlib import Path
        require_verified_assets(verified_assets, trajectory, ledger, certificate, check)
        coverage_archive.write_assets(folder, certificate, check)
        readback = coverage_archive.load_assets(folder, check)
        if readback.metadata != certificate.metadata:raise ValueError('本次变宽覆盖资产回读与已验证结果不一致')
        durable(Path(folder)/coverage_archive.ASSET);durable(Path(folder)/coverage_archive.MANIFEST)
        return
    if trajectory.planner['algorithm_version'] in ('coverage-refined-surface-v1', 'constructed-coverage-refined-surface-v1', 'nominal-variable-width-refined-surface-v1'):
        from femto.planning.refinement_archive import write as write_refinement
        return write_refinement(folder, certificate, trajectory, ledger, check, verified_assets=verified_assets)
    if verified_assets is not None:raise ValueError('当前路线不支持补充规划验证凭据')
    domain, seeds, model, route = bundle(trajectory, ledger)
    coverage_archive.write_checked(folder, certificate, domain, seeds, model, route_binding=route, check=check)


def read(folder, metadata, trajectory, ledger, check=lambda:None):
    if metadata is None:raise ValueError('覆盖资产没有绑定清单')
    from femto.planning.refinement_archive import check_inventory
    from pathlib import Path
    check_inventory({p.name for p in Path(folder).iterdir() if p.is_file()}, trajectory, metadata)
    if trajectory.planner['algorithm_version'] in ('coverage-refined-surface-v1', 'constructed-coverage-refined-surface-v1', 'nominal-variable-width-refined-surface-v1'):
        from femto.planning.refinement_archive import read as read_refinement
        return read_refinement(folder, metadata, trajectory, ledger, check)
    domain, seeds, model, route = bundle(trajectory, ledger)
    result = coverage_archive.read(folder, domain, seeds, model, route_binding=route, check=check)
    if result.metadata != metadata:raise ValueError('覆盖文件与方案中的证据清单不一致')
    return result


def summary_text(metadata):
    if metadata is None:return '模型覆盖：未进行此项核验。'
    values = metadata['summary']
    complete = values['original_target_coverage_complete']
    state = '名义模型内已确认' if complete else '尚有未确认范围'
    if values['status'] == 'calculation_incomplete':state = '达到计算限额，仍未完成'
    result = (f"模型覆盖：{state}。已确认 {values['confirmed_nominal_area_m2']*1e6:.1f} / "
              f"已定位 {values['located_nominal_area_m2']*1e6:.1f} mm²。")
    if metadata['inputs']['footprint']['schema'] == 'femto-nominal-piecewise-ball-v1':
        result = '名义变宽'+result
    if values['unlocated_pixel_count']:
        result += f" 另有 {values['unlocated_pixel_count']} 个像素含未定位范围，面积未知。"
    return result


def summaries(plan):
    plan = plan or {}
    stats = plan.get('surface_trajectory', {}).get('statistics', {})
    path = (f"路径：{stats.get('segment_count', '未知')} 条连续路线，"
            f"{stats.get('gap_count', '未知')} 处断点。" if plan else '路径：尚未生成。')
    target = (plan.get('planning_context') or {}).get('target')
    from femto.planning.target_surface import is_deferred, summary_text as target_summary
    if is_deferred(target):
        geometry = target_summary(target)
    elif target:
        values = target['summary']
        geometry = (f"原目标几何：已定位 {values['located_area_m2']*1e6:.1f} mm²；"
                    f"{values['unlocated_pixel_count']} 个像素含未定位范围。")
    else:geometry = '原目标几何：旧记录未建立台账。' if plan else '原目标几何：待建立。'
    return path, geometry, summary_text(plan.get(KEY))
