"""P3: bounded, append-only route refinement with independently replayed coverage.

The v3 acceptance predicate and shared-face search are reused unchanged. A
trial is installed only when common-partition coverage increases without loss.
All unsuccessful attempts remain evidence; they are not proof of infeasibility.
"""
from __future__ import annotations

from copy import deepcopy
from collections import Counter
from dataclasses import asdict, dataclass, replace
import math
from typing import Callable

import numpy as np

from femto.planning import coverage_certificate as coverage, planning_revision as revision_api
from femto.planning import quality_constrained as quality
from femto.planning.continuous_quality import VerifiedSurfaceGraph
from femto.planning.coverage_domain import CoverageDomain, scan_endpoints
from femto.planning.surface_trajectory import ScanSegment, SurfaceTrajectory
from femto.operation_diagnostics import diagnosed

ALGORITHM = 'coverage-refined-surface-v1'
ALGORITHMS = (ALGORITHM, 'constructed-coverage-refined-surface-v1')
SCHEMA = 'femto-coverage-refinement-v1'
OPTIMIZED_SCHEMA = 'femto-coverage-refinement-v2'
REPROOF_SCHEMA = 'femto-coverage-refinement-v3'
RESIDUAL_SCHEMA = 'femto-coverage-refinement-v4'
INTERLEAVED_SCHEMA = 'femto-coverage-refinement-v5'


@dataclass(frozen=True)
class RefinementBudget:
    max_rounds: int = 2
    candidates_per_round: int = 4
    max_edge_checks: int = 200_000
    max_output_points: int = 200_000
    max_added_length_m: float = .1
    min_gain_mm2: float = .1
    low_gain_rounds: int = 2
    coverage_seconds: float = 4.
    max_coverage_comparisons: int = 1_000_000
    max_coverage_adjacencies: int = 1_000_000
    max_coverage_proofs: int = 1_000_000

    def checked(self) -> RefinementBudget:
        caps = asdict(RefinementBudget())
        for key, cap in caps.items():
            value = getattr(self, key)
            if key in ('max_added_length_m', 'min_gain_mm2', 'coverage_seconds'):
                if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= cap:
                    raise ValueError('补充规划预算无效：'+key)
            elif type(value) is not int or not 0 <= value <= cap:
                raise ValueError('补充规划预算无效：'+key)
        if self.low_gain_rounds == 0:raise ValueError('低增益停止轮数必须大于零')
        return self


@dataclass(frozen=True)
class RefinedCertificate(coverage.Certificate):
    history: tuple[coverage.Certificate, ...] = ()

    @property
    def guide(self):
        # A refinement with no admitted additions can keep the recalculated
        # base. Its one hint is already part of history[0], never a new chain.
        if self.metadata['algorithm'] in coverage.RECALCULATION_ALGORITHMS and self.history:
            return getattr(self.history[0], 'guide', None)
        return None


class _Stop(Exception):pass


class _Meter:
    def __init__(self, budget):
        self.budget = budget
        self.edge_checks = 0
        self.counts = dict(comparisons=0, adjacencies=0, proofs=0)

    def reserve(self, count):
        if self.edge_checks+count > self.budget.max_edge_checks:raise _Stop('edge_budget')
        self.edge_checks += count

    def coverage_budget(self):
        return coverage.CoverageBudget(seconds=self.budget.coverage_seconds,
            **{'max_'+key:getattr(self.budget, 'max_coverage_'+key)-count for key, count in self.counts.items()})


class _Graph(VerifiedSurfaceGraph):
    def _checks(self, edges):
        self.meter.reserve(len(edges))
        return super()._checks(edges)


def algorithms_for(base, certificate, budget, candidate_policy=None, coverage_policy=None) -> dict:
    from femto.planning.coverage_workflow import route_binding
    algorithm = ALGORITHMS[1] if base.planner['algorithm_version'] == 'constructed-surface-route-v1' else ALGORITHM
    settings = dict(
        schema=SCHEMA, base_computation_sha256=base.planning_context['revision']['sha256'],
        base_route_sha256=revision_api.digest(route_binding(base)),
        base_coverage_sha256=certificate.metadata['sha256'], budget=asdict(budget))
    if candidate_policy is not None:
        from femto.planning.refinement_candidates_v2 import checked_policy
        settings.update(schema=OPTIMIZED_SCHEMA, candidate_policy=deepcopy(checked_policy(candidate_policy)))
    if coverage_policy is not None:
        if candidate_policy is None:raise ValueError('新覆盖复算需要当前补充候选策略')
        reproof_algorithm = coverage.reproof_algorithm_for_policy(coverage_policy)
        schemas = {coverage.REPROOF_ALGORITHM: REPROOF_SCHEMA, coverage.RESIDUAL_ALGORITHM: RESIDUAL_SCHEMA,
                   coverage.INTERLEAVED_ALGORITHM: INTERLEAVED_SCHEMA}
        settings.update(schema=schemas[reproof_algorithm],
                        coverage_policy=deepcopy(coverage_policy))
    return dict(route=algorithm, target=base.planning_context['revision']['inputs']['algorithms']['target'], refinement=settings)


def checked_algorithms(algorithms: dict) -> bool:
    if not isinstance(algorithms, dict) or set(algorithms) != {'route', 'target', 'refinement'}:return False
    if algorithms['route'] not in ALGORITHMS or algorithms['target'] not in revision_api.TARGET_ALGORITHMS:return False
    values = algorithms['refinement']
    if not isinstance(values, dict):return False
    fields = {'schema', 'base_computation_sha256', 'base_route_sha256', 'base_coverage_sha256', 'budget'}
    if values.get('schema') in (OPTIMIZED_SCHEMA, REPROOF_SCHEMA, RESIDUAL_SCHEMA, INTERLEAVED_SCHEMA):
        from femto.planning.refinement_candidates_v2 import checked_policy
        fields.add('candidate_policy')
        try:checked_policy(values.get('candidate_policy'))
        except ValueError:return False
    elif values.get('schema') != SCHEMA:return False
    if values.get('schema') in (REPROOF_SCHEMA, RESIDUAL_SCHEMA, INTERLEAVED_SCHEMA):
        fields.add('coverage_policy')
        expected = {REPROOF_SCHEMA: coverage.REPROOF_ALGORITHM, RESIDUAL_SCHEMA: coverage.RESIDUAL_ALGORITHM,
                    INTERLEAVED_SCHEMA: coverage.INTERLEAVED_ALGORITHM}[values['schema']]
        try:
            if coverage.reproof_algorithm_for_policy(values.get('coverage_policy')) != expected:return False
        except ValueError:return False
    if set(values) != fields:
        return False
    for key in ('base_computation_sha256', 'base_route_sha256', 'base_coverage_sha256'):
        value = values[key]
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):return False
    try:RefinementBudget(**values['budget']).checked()
    except (TypeError, ValueError):return False
    return True


def rebind_target(ledger, revision):
    """Copy the header for a new route revision; the original target arrays stay identical."""
    from femto.planning import target_surface as target
    from femto.planning.target_paging import rebind_headers
    return target._seal(dict(ledger.arrays), revision, target.TargetBudget(**ledger.metadata['budget']),
        deepcopy(ledger.metadata['metrics']), ledger.metadata['quality_evidence_sha256'], ledger.metadata['effective_domain_sha256'],
        construction_pages=rebind_headers(ledger,revision))


def rebind_coverage(certificate, trajectory, ledger):
    from femto.planning.coverage_workflow import bundle
    domain, seeds, model, route = bundle(trajectory, ledger)
    metadata = deepcopy(certificate.metadata)
    if metadata['inputs']['seeds_sha256'] != revision_api.digest(seeds) or metadata['inputs']['footprint'] != model:
        raise ValueError('补充覆盖重绑定不能改变扫描种子或作用模型')
    metadata['inputs'].update(domain=domain.binding, route=route)
    metadata['coverage_computation_sha256'] = revision_api.digest(metadata['inputs'])
    metadata['sha256'] = revision_api.digest({k:v for k,v in metadata.items() if k != 'sha256'})
    if metadata['algorithm'] in coverage.RECALCULATION_ALGORITHMS:
        from femto.planning.coverage_recalculation import RecalculatedCertificate, require_guide
        return RecalculatedCertificate(metadata, dict(certificate.arrays), require_guide(certificate))
    return coverage.Certificate(metadata, dict(certificate.arrays))


def _route(base, chains, context, report):
    items = quality._items(chains)
    if [i.to_dict() for i in items] == [i.to_dict() for i in base.ordered_items]:
        stats, blocks = deepcopy(base.statistics), deepcopy(base.blocks)
    else:
        classes, _ = quality.checked_evidence(base.planner['parameters']['quality_evidence'])
        stats, blocks = quality._summaries(items, base.blocks['items'], classes)
    algorithm = ALGORITHMS[1] if base.planner['algorithm_version'] == 'constructed-surface-route-v1' else ALGORITHM
    planner = dict(base.planner, kind='coverage_refined_surface', algorithm_version=algorithm,
        parameters=dict(base.planner['parameters'], coverage_refinement=context['revision']['inputs']['algorithms']['refinement']))
    return replace(base, planner=planner, planning_context=context, ordered_items=items,
        statistics=stats, blocks=blocks, gap_report=dict(base.gap_report, counts=stats['gap_counts'], refinement=report))


def _chains(trajectory):
    return [(s.block_index, deepcopy(s.points)) for s in trajectory.ordered_items if isinstance(s, ScanSegment)]


def _new_witnesses(current, fresh, seeds, target):
    old = {(r['source_face'], *r['xyz_m']) for r in scan_endpoints(current, target.arrays['region_faces'])}
    added = np.array([(r['source_face'], *r['xyz_m']) not in old for r in seeds], bool)
    arrays = fresh.arrays;selected = arrays['status'] == 1
    result = np.zeros(len(selected), bool)
    proof = arrays['witness'][selected]
    result[selected] = added[arrays['seed_input_index'][arrays['proof_seed'][proof]]]
    return result


def _connect(chains, addition, trajectory, mesh, region, allowed, meter, check):
    from femto.planning.selection_blocks import selection_face_assignment
    from femto.planning.refinement_candidates_v2 import connection_limit, connection_fits
    block, points = addition
    positions = [i for i, (b, _) in enumerate(chains) if b == block]
    samples = None
    limit = connection_limit(chains, points, trajectory, meter)
    limited = False
    if positions:
        position = positions[-1]
        # Only endpoint neighbourhoods are needed for this new connection.
        # Retained v3 geometry nodes and edge predicates remain unchanged;
        # omitted scan-interior detours may leave an explicit unresolved gap.
        previous = ScanSegment(0, chains[position][1][-2]['row_index'], chains[position][1][-2:], block)
        following = ScanSegment(1, points[0]['row_index'], points[:2], block)
        _, _, _, owners = selection_face_assignment(region, np.asarray(mesh.vertices), np.asarray(mesh.faces))
        component = trajectory.blocks['items'][block]['component_id']
        params = trajectory.planner['parameters']
        graph = _Graph.__new__(_Graph);graph.meter = meter
        limited = not connection_fits(limit, [chains[position][1][-1], points[0]], 'endpoint_lower_bound')
        if not limited:
            try:
                graph.__init__(mesh, region, params['quality_evidence'], params['effective_width_mm'],
                    allowed & (owners == component), replace(trajectory, ordered_items=[previous, following]), check)
                samples = graph.route(0, 1)
            except ValueError as error:
                if '预算' not in str(error):raise
                raise _Stop('graph_budget') from error
            if samples and not connection_fits(limit, [chains[position][1][-1]]+samples, 'graph'):
                samples = None;limited = True
    else:position = next((i-1 for i, (b, _) in enumerate(chains) if b > block), len(chains)-1)
    decisions = []
    if samples:
        meter.reserve(len(samples))
        edge = ScanSegment(0, points[0]['row_index'], [chains[position][1][-1]]+samples, block)
        flags, reasons, margins = quality.edge_checks([edge], mesh, params['quality_evidence'],
            params['effective_width_mm'], check, allowed_faces=allowed)
        if not flags.all():raise ValueError('共享面搜索输出未通过补充连接逐边复核')
        decisions = [dict(accepted=bool(a), reason=b, residual_margin_mm=float(c)) for a,b,c in zip(flags, reasons, margins)]
        chains[position][1].extend(deepcopy(samples));chains[position][1].extend(deepcopy(points[1:]))
    else:chains.insert(position+1, (block, deepcopy(points)))
    result = dict(block_index=block, connected=bool(samples), samples=samples or [], edge_checks=decisions,
        reason='verified_surface_path' if samples else 'connection_length_budget' if limited else
               'no_verified_graph_path' if positions else 'no_prior_chain_in_block')
    if limit is not None:result['connection_budget'] = limit
    return result


def candidate_recipes(current, domain, certificate, mesh, region, allowed, attempted, meter, check):
    if getattr(meter, 'candidate_policy', None) is not None:
        from femto.planning.refinement_candidates_v2 import select
        return select(current, domain, certificate, mesh, region, allowed, attempted, meter, check)
    from femto.planning.refinement_candidates import ranked_faces
    return ranked_faces(domain, certificate, mesh, region, current.blocks, allowed, attempted, check)


def construct_candidate(recipe, domain, mesh, row, step):
    if 'candidate_algorithm' in recipe:
        from femto.planning.refinement_candidates_v2 import candidate
        return candidate(recipe, domain, mesh, row, step)
    from femto.planning.refinement_candidates import slice_candidate
    return slice_candidate(recipe, domain, mesh, row, step)


@diagnosed('refinement.candidates')
def _propose(current, domain, certificate, mesh, region, allowed, attempted, meter, check):
    if hasattr(meter, 'support'):
        from femto.planning.strict_refinement import propose
        return propose(current, domain, certificate, mesh, region, allowed, attempted, meter, check)
    from femto.planning.refinement_candidates import retained_chains
    recipes = candidate_recipes(current, domain, certificate, mesh, region, allowed, attempted, meter, check)
    chains = _chains(current);records = [];params = current.planner['parameters']
    row = max(p['row_index'] for _, points in chains for p in points)+1
    status = 'no_candidates'
    for recipe in recipes[:meter.budget.candidates_per_round]:
        check();attempted.add(recipe['local_face'])
        record = dict(recipe=recipe, status='too_short_or_unconstructed', edge_checks=[], connections=[])
        records.append(record)
        candidate = construct_candidate(recipe, domain, mesh, row, params['sample_step_mm']/1000);row += 1
        if candidate is None:continue
        try:
            meter.reserve(len(candidate.points)-1)
            diagnostics = [] if getattr(meter, 'candidate_policy', None) is not None else None
            flags, reasons, margins = quality.edge_checks([candidate], mesh, params['quality_evidence'],
                params['effective_width_mm'], check, allowed_faces=allowed, diagnostics=diagnostics)
            record.update(status='checked', points=candidate.points,
                edge_checks=[dict(accepted=bool(ok), reason=reason, residual_margin_mm=float(margin))
                             for ok, reason, margin in zip(flags, reasons, margins)])
            if diagnostics is not None:record['quality_diagnostics'] = diagnostics
            for addition in retained_chains(candidate, flags):
                record['connections'].append(_connect(chains, addition, current, mesh, region, allowed, meter, check))
            if flags.any():status = 'proposed'
        except _Stop as error:
            record['status'] = str(error)
            return _chains(current), records, str(error)
    return chains, records, status


@diagnosed('refinement.rounds')
def refine(base: SurfaceTrajectory, ledger, certificate: coverage.Certificate, mesh, region: dict, *,
           budget: RefinementBudget | None = None, check: Callable = lambda: None,
           progress: Callable = lambda text: None, frontiers: list | None = None, candidate_policy=None,
           coverage_policy=None):
    """Caller validates the immutable base route, target and starting certificate first."""
    from femto.planning.coverage_workflow import route_binding
    from femto.planning.coverage_partition import compare
    from femto.planning.effective_domain import validate as validate_domain
    budget = (budget or RefinementBudget()).checked()
    if base.planner['algorithm_version'] not in (revision_api.ALGORITHMS['route'], 'constructed-surface-route-v1') or base.planning_context is None:
        raise ValueError('补充规划需要带原目标及覆盖证据的 v3 路线；请先恢复补充前路线')
    if certificate.metadata['algorithm'] not in coverage.P2_ALGORITHMS:raise ValueError('补充规划需要当前覆盖核验版本')
    check();params = base.planner['parameters']
    allowed = validate_domain(params['effective_domain'], params['quality_evidence'], params['effective_width_mm'], mesh, region, check)
    inputs = deepcopy(base.planning_context['revision']['inputs']);inputs['algorithms'] = algorithms_for(base, certificate, budget, candidate_policy, coverage_policy)
    revision = revision_api.create(inputs);target = rebind_target(ledger, revision)
    context = dict(schema=revision_api.CONTEXT_SCHEMA, revision=revision, target=target.metadata)
    domain = CoverageDomain._checked_target(target, revision)
    report = dict(schema=inputs['algorithms']['refinement']['schema'], base_trajectory=base.to_dict(), base_coverage=deepcopy(certificate.metadata),
        budget=asdict(budget), rounds=[], stop_reason='round_limit', edge_checks=0, coverage_work={},
        max_total_length_m=base.statistics['total_length_m']+budget.max_added_length_m,
        reason_routing=dict(confirmed='retain', outside_inner_seed_balls='candidate_on_eligible_surface',
            quality_excluded_or_pending='rescan_or_review', unlocated_pixels='rescan_or_review',
            ball_connectivity_unconfirmed='verify_connectivity', subdivision_limit='continue_coverage',
            target_construction_unconfirmed='review_target_construction', budget_unprocessed='continue_coverage'),
        failure_is_proof_of_infeasibility=False, registration_uncertainty_mm=None,
        physical_accuracy_verified=False, execution_ready=False)
    if candidate_policy is not None:report['candidate_policy'] = deepcopy(candidate_policy)
    if coverage_policy is not None:report['coverage_policy'] = deepcopy(coverage_policy)
    current = _route(base, _chains(base), context, report);previous = certificate;history = [certificate]
    attempted = set();meter = _Meter(budget);low = 0
    meter.candidate_policy = candidate_policy
    if base.planner['algorithm_version'] == 'constructed-surface-route-v1':
        from femto.planning.strict_path import engine_for, SupportBudget
        meter.support_budget = SupportBudget(**params['surface_support']['budget']).checked()
        meter.support = engine_for(mesh, allowed, meter.support_budget, check, region=region)
    for index in range(budget.max_rounds):
        check();progress(f'补充规划第 {index+1} 轮：按缺口面积选择可用表面并检查每条新增边')
        try:
            chains, candidates, status = _propose(current, domain, previous, mesh, region, allowed, attempted, meter, check)
        except _Stop as error:
            chains, candidates, status = _chains(current), [], str(error)
        trial = _route(base, chains, context, report)
        record = dict(index=index, candidates=candidates, status=status, accepted=False, certificate_index=None, comparison=None)
        if candidate_policy is not None:record['candidate_search'] = deepcopy(getattr(meter, 'search_info', {}))
        report['rounds'].append(record)
        if status in ('edge_budget', 'graph_budget') or hasattr(meter, 'support') and status.endswith('_budget'):
            report['stop_reason'] = status;break
        if status != 'proposed':
            record['status'] = 'no_admitted_candidates';low += 1
            if not candidates:report['stop_reason'] = 'no_eligible_candidates';break
        elif trial.statistics['point_count'] > budget.max_output_points:
            record['status'] = report['stop_reason'] = 'point_budget';break
        elif trial.statistics['total_length_m'] > report['max_total_length_m']:
            record['status'] = report['stop_reason'] = 'length_budget';break
        else:
            progress(f'补充规划第 {index+1} 轮：重新核验覆盖并比较共同分区')
            seeds = scan_endpoints(trial, target.arrays['region_faces'])
            if frontiers is not None and len(history)-1 >= len(frontiers):
                raise ValueError('补充覆盖复算缺少对应的停止边界')
            frontier = None if frontiers is None else frontiers[len(history)-1]
            fresh = coverage.evaluate(domain, seeds, certificate.metadata['inputs']['footprint'],
                route_binding=route_binding(trial),
                algorithm=coverage.CURRENT_ALGORITHM if coverage_policy is None else coverage.reproof_algorithm_for_policy(coverage_policy),
                budget=meter.coverage_budget(), check=check, _frontier=frontier,
                **({} if coverage_policy is None else {'guide':previous}))
            for key in meter.counts:meter.counts[key] += fresh.metadata['metrics'][key]
            record['certificate_index'] = len(history);record['coverage'] = deepcopy(fresh.metadata);history.append(fresh)
            comparison = compare(previous, fresh, domain.areas, check,
                new_witnesses=_new_witnesses(current, fresh, seeds, target));record['comparison'] = comparison
            gain = (comparison['retained_subset'] and comparison['new_scan_witness_weight_units'] > 0
                    and comparison['new_scan_witness_gain_m2']*1e6 >= budget.min_gain_mm2)
            record.update(accepted=bool(gain), status='accepted' if gain else 'coverage_regression' if not comparison['retained_subset'] else 'insufficient_gain')
            if gain:current = trial;previous = fresh;low = 0
            else:low += 1
        if low >= budget.low_gain_rounds:report['stop_reason'] = 'low_gain_limit';break
        if any(meter.counts[k] >= getattr(budget, 'max_coverage_'+k) for k in meter.counts):
            report['stop_reason'] = 'coverage_work_budget';break
    if frontiers is not None and len(frontiers) != len(history)-1:
        raise ValueError('补充覆盖复算多出未使用的停止边界')
    check();report.update(edge_checks=meter.edge_checks, coverage_work=meter.counts)
    if hasattr(meter, 'support'):
        report['strict_support'] = dict(face_checks=meter.support.face_checks,
            filter_comparisons=meter.support.filter_comparisons, construction_fast_paths=meter.support.fast_paths,
            support_queries=meter.support.queries,interval_queries=meter.support.interval_queries,
            aabb_rejected_faces=meter.support.aabb_rejected_faces,
            scope='exact_constructed_numeric_surface_only', preserved_base_anchors=True)
    current = _route(base, _chains(current), context, report)
    if hasattr(meter, 'support'):
        from femto.planning.surface_trajectory import canonical_json_bytes
        if len(canonical_json_bytes(current.to_dict())) > 64*1024*1024:
            raise ValueError('严格补充路线超过 64 MiB 保存预算；原路线保留')
    final = rebind_coverage(previous, current, target)
    certificate_result = RefinedCertificate(final.metadata, final.arrays, tuple(history))
    return current, target, certificate_result


def diagnostics_text(report) -> str:
    """Present already verified search evidence; never use a label as admission."""
    searches = [r['candidate_search'] for r in report['rounds'] if 'candidate_search' in r]
    if not searches:return ''
    names = dict(outside_selection='作用范围越出工作区域', reference_quality_unconfirmed='参考几何质量待确认',
        outside_selection_and_quality='区域边界和参考质量均未通过', footprint_outside_image='作用范围越出参考图',
        outer_projection_unavailable='无法投影参考范围', footprint_projection_unavailable='无法投影作用范围',
        short_symmetric_face_interval='面内可用长度不足', target_construction_unconfirmed='目标位置未确认',
        mesh_boundary_clearance='距离表面边界不足', chord_not_certified_on_mesh='整边表面支持未确认',
        outside_effective_domain='越出有效表面', quality_or_width_boundary='质量或作用范围未通过')
    rejected = Counter();edges = Counter();connections = 0
    for search in searches:rejected.update(search.get('rejection_counts', {}))
    for record in report['rounds']:
        for candidate in record['candidates']:
            for item in candidate.get('quality_diagnostics', candidate['edge_checks']):
                if not item['accepted']:edges[item['reason']] += 1
            connections += sum(link['reason'] == 'connection_length_budget' for link in candidate['connections'])
    def reasons(values):return '、'.join(f'{names.get(key, "表面支持未确认")} {count} 处' for key,count in sorted(values.items()))
    text = (f'候选筛查：检查 {sum(s.get("screened_faces", 0) for s in searches)} 个表面，'
            f'{sum(s.get("candidates_selected", 0) for s in searches)} 条候选进入逐边核验。')
    if rejected:text += ' 未通过位置：'+reasons(rejected)+'。'
    if edges:text += ' 逐边未通过：'+reasons(edges)+'。'
    if connections:text += f' {connections} 处连接超过分配的长度预算，保留断点。'
    regressed = sum(r['status'] == 'coverage_regression' for r in report['rounds'])
    if regressed:text += f' {regressed} 轮未能在预算内重新确认全部原覆盖，候选未采用。'
    return text


def summary_text(trajectory) -> str:
    if (isinstance(trajectory, dict) and (trajectory.get('planner',{}).get('algorithm_version') == ALGORITHMS[1]
            or trajectory.get('gap_report',{}).get('refinement',{}).get('schema') in (OPTIMIZED_SCHEMA,REPROOF_SCHEMA,RESIDUAL_SCHEMA,INTERLEAVED_SCHEMA))):
        # Installed history has already been verified by the workflow/reader.
        # A label must not trigger another recursive proof replay on Qt's thread.
        report=trajectory.get('gap_report',{}).get('refinement');statistics=trajectory['statistics']
    else:
        if isinstance(trajectory, dict):trajectory = SurfaceTrajectory.from_dict(trajectory)
        report = trajectory.gap_report.get('refinement');statistics=trajectory.statistics
    if report is None:return ''
    accepted = [r for r in report['rounds'] if r['accepted']]
    area = math.fsum(r['comparison']['gained_nominal_area_m2'] for r in accepted)*1e6
    length = statistics['total_length_m']-report['base_trajectory']['statistics']['total_length_m']
    reasons = dict(round_limit='达到轮数限额', low_gain_limit='连续增益不足', no_eligible_candidates='没有可用的补充候选',
        edge_budget='达到边检查限额', graph_budget='达到连接搜索限额', point_budget='达到点数限额',
        length_budget='达到长度限额', coverage_work_budget='达到覆盖计算限额')
    result = (f'补充规划：尝试 {len(report["rounds"])} 轮，接纳 {len(accepted)} 轮；'
            f'名义覆盖新增 {area:.1f} mm²，路线增加 {length*1000:.1f} mm。'
            +reasons.get(report['stop_reason'], '达到严格构造计算限额')+'；剩余未确认范围仍保留。')
    diagnostic = diagnostics_text(report)
    return result+('\n'+diagnostic if diagnostic else '')
