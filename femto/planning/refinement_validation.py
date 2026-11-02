"""Route-side validation for P3. Coverage claims additionally require asset replay."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from femto.planning import coverage_refinement as refinement, quality_constrained as quality
from femto.planning import planning_revision as revisions
from femto.planning.surface_trajectory import ScanSegment, SurfaceTrajectory, CONNECTED_ALGORITHM, validate_trajectory


def semantic_payload(trajectory) -> dict:
    payload = trajectory.to_dict()
    for key in ('route_diagram_status', 'data_layers'):payload.pop(key, None)
    payload['source'] = {k:trajectory.source[k] for k in ('geometry_sha256', 'region_id', 'region_signature')}
    return payload


def reproduce_base(trajectory, manifest_path, mesh, region, check=lambda: None):
    base = SurfaceTrajectory.from_dict(trajectory.gap_report['refinement']['base_trajectory'])
    if base.planner['algorithm_version'] == 'constructed-surface-route-v1':
        from femto.planning.strict_validation import reproduce_base as reproduce_strict
        return reproduce_strict(base, manifest_path, mesh, region, check)
    return reproduce_v3(base, manifest_path, mesh, region, check)


def reproduce_v3(base, manifest_path, mesh, region, check=lambda: None):
    from femto.planning.continuous_quality import candidate, replan
    params = dict(base.planner['parameters'])
    record = params.pop('quality_evidence');width = params.pop('effective_width_mm');domain = params.pop('effective_domain')
    routing=params.pop('continuous_routing');params.pop('preserve_face_crossings')
    generated = replan(candidate(manifest_path, check=check, **params), mesh, region, record, width, domain=domain, check=check, routing=routing)
    generated = replace(generated, planning_context=base.planning_context)
    if semantic_payload(generated) != semantic_payload(base):raise ValueError('补充前 v3 路线不能由绑定输入重现')


def _append_record(chains, candidate):
    from femto.planning.refinement_candidates import retained_chains
    points = candidate.get('points', [])
    flags = candidate['edge_checks']
    if len(points) < 2 or len(flags) != len(points)-1:raise ValueError('补充边检查记录缺失')
    if any(type(f['accepted']) is not bool or f['accepted'] != (f['reason'] == 'accepted') for f in flags):
        raise ValueError('补充边检查状态无效')
    block = candidate['recipe']['block_index']
    segment = ScanSegment(0, points[0]['row_index'], points, block)
    additions = retained_chains(segment, [f['accepted'] for f in flags])
    if len(additions) != len(candidate['connections']):raise ValueError('补充连接记录缺失')
    for (block, points), link in zip(additions, candidate['connections']):
        positions = [i for i, (b, _) in enumerate(chains) if b == block]
        position = positions[-1] if positions else next((i-1 for i, (b, _) in enumerate(chains) if b > block), len(chains)-1)
        samples = link['samples']
        decisions = link.get('edge_checks')
        if (not isinstance(decisions, list) or len(decisions) != len(samples)
                or any(d.get('accepted') is not True or d.get('reason') != 'accepted' for d in decisions)):
            raise ValueError('补充连接缺少逐边验收记录')
        if type(link['connected']) is not bool or link['connected'] != bool(samples) or link['block_index'] != block:
            raise ValueError('补充连接状态或工作块不一致')
        if samples:
            if (not positions or link['reason'] != 'verified_surface_path' or samples[-1]['xyz_m'] != points[0]['xyz_m']
                    or any(p.get('role') != 'connector' or p['row_index'] != points[0]['row_index'] for p in samples)):
                raise ValueError('补充连接不能跨工作块或改变端点')
            chains[position][1].extend(deepcopy(samples));chains[position][1].extend(deepcopy(points[1:]))
        else:
            expected_reason = 'no_verified_graph_path' if positions else 'no_prior_chain_in_block'
            bounded = (positions and candidate['recipe'].get('candidate_algorithm') ==
                       'quality-screened-symmetric-face-slices-v2' and link['reason'] == 'connection_length_budget'
                       and isinstance(link.get('connection_budget'), dict))
            if link['reason'] != expected_reason and not bounded:
                raise ValueError('补充连接未确认原因无效')
            chains.insert(position+1, (block, deepcopy(points)))


def _check_final(trajectory, mesh, region, check):
    params = trajectory.planner['parameters'];shadow = []
    for item in trajectory.ordered_items:
        if isinstance(item, ScanSegment):
            points = [dict({k:v for k,v in p.items() if k != 'revisit'}, role='scan') for p in item.points]
            shadow.append(replace(item, points=points, row_index=min(p['row_index'] for p in points), block_index=None))
        else:
            shadow.append(replace(item, reason='unsupported_geometry', detail=dict(inferred_cause='surface_gap',
                method='sweep_plane_outer_loop_containment', confidence='heuristic', decision_use='none')))
    validate_trajectory(replace(trajectory, planner=dict(trajectory.planner, algorithm_version=CONNECTED_ALGORITHM,
        parameters=dict(params, selection_blocks=False, block_planning=False, connect_rows=True)),
        ordered_items=shadow, blocks=None, planning_context=None), mesh=mesh, check=check)
    if mesh is None:return
    from femto.planning.effective_domain import validate as validate_domain
    allowed = validate_domain(params['effective_domain'], params['quality_evidence'], params['effective_width_mm'], mesh, region, check)
    final = [s for s in trajectory.ordered_items if isinstance(s, ScanSegment)]
    extra = {}
    if trajectory.planner['algorithm_version'] in ('constructed-surface-route-v1', 'constructed-coverage-refined-surface-v1'):
        from femto.planning.strict_path import Checks, engine_for, SupportBudget
        extra['support_checker'] = Checks(engine_for(mesh, allowed, SupportBudget(**params['surface_support']['budget']), check, region=region))
    okay, _, _ = quality.edge_checks(final, mesh, params['quality_evidence'], params['effective_width_mm'], check, allowed_faces=allowed, **extra)
    if not okay.all():raise ValueError('补充路线存在未通过 v3 整边检查的边')
    if region is not None:
        from femto.planning.selection_blocks import selection_face_assignment, check_projection_inside_selection
        _, _, labels, owners = selection_face_assignment(region, np.asarray(mesh.vertices), np.asarray(mesh.faces))
        if extra:
            for segment in final:
                component = trajectory.blocks['items'][segment.block_index]['component_id']
                if any(owners[p['triangle_index']] != component for p in segment.points):
                    raise ValueError('严格路径借用了其他工作块的支持面')
        check_projection_inside_selection(final, trajectory.blocks['items'], labels,
            np.asarray(region['view_reference']['undistorted_intrinsic']), np.asarray(region['view_reference']['image_size']))


def validate(trajectory, mesh=None, region=None, check=lambda: None):
    external_region=region is not None
    report = trajectory.gap_report.get('refinement') or {}
    if (report.get('schema') not in (refinement.SCHEMA, refinement.OPTIMIZED_SCHEMA, refinement.REPROOF_SCHEMA, refinement.RESIDUAL_SCHEMA, refinement.INTERLEAVED_SCHEMA) or report.get('failure_is_proof_of_infeasibility') is not False
            or report.get('execution_ready') is not False or report.get('physical_accuracy_verified') is not False
            or report.get('registration_uncertainty_mm', 0) is not None):
        raise ValueError('补充规划证据语义无效')
    budget = refinement.RefinementBudget(**report['budget']).checked()
    base = SurfaceTrajectory.from_dict(report['base_trajectory'])
    strict = trajectory.planner['algorithm_version'] == 'constructed-coverage-refined-surface-v1'
    expected_base = 'constructed-surface-route-v1' if strict else revisions.ALGORITHMS['route']
    if base.planner['algorithm_version'] != expected_base or base.planning_context is None:
        raise ValueError('补充规划原路线必须是带台账的 v3 版本')
    if strict and region is None:region=base.planning_context['revision']['inputs']['selection']
    if not strict or mesh is not None or external_region:
        validate_trajectory(base, mesh=mesh, region=region, check=check)
    refinement.coverage.validate_metadata(report['base_coverage'])
    candidate_policy = report.get('candidate_policy')
    algorithms = refinement.algorithms_for(base, SimpleNamespace(metadata=report['base_coverage']), budget,
        candidate_policy, report.get('coverage_policy'))
    if report['schema'] != algorithms['refinement']['schema']:
        raise ValueError('补充候选策略缺失或混用了不同版本')
    if trajectory.planner['parameters'].get('coverage_refinement') != algorithms['refinement']:
        raise ValueError('补充修订与原路线、核验或预算不一致')
    inputs = deepcopy(base.planning_context['revision']['inputs']);inputs['algorithms'] = algorithms
    revision = revisions.create(inputs)
    target = deepcopy(base.planning_context['target']);target['computation_sha256'] = revision['sha256']
    target['sha256'] = revisions.digest({k:v for k,v in target.items() if k != 'sha256'})
    context = dict(schema=revisions.CONTEXT_SCHEMA, revision=revision, target=target)
    if (strict or trajectory.planning_context is not None) and trajectory.planning_context != context:
        raise ValueError('补充规划改变了原目标或未绑定新计算修订')
    rounds = report['rounds'];chains = refinement._chains(base)
    if strict:
        from femto.planning.strict_path import SupportBudget
        from femto.planning.surface_trajectory import canonical_json_bytes
        support_budget=SupportBudget(**base.planner['parameters']['surface_support']['budget']).checked()
        metrics=report.get('strict_support') or {}
        if metrics.get('preserved_base_anchors') is not True or metrics.get('scope')!='exact_constructed_numeric_surface_only':
            raise ValueError('严格补充缺少原锚点保留记录')
        for key,limit in (('face_checks',support_budget.max_face_checks),('filter_comparisons',support_budget.max_filter_comparisons)):
            if type(metrics.get(key)) is not int or not 0<=metrics[key]<=limit:raise ValueError('严格补充超过支持检查预算')
        if len(canonical_json_bytes(trajectory.to_dict()))>64*1024*1024:raise ValueError('严格补充超过 64 MiB 记录预算')
    if not isinstance(rounds, list) or len(rounds) > budget.max_rounds:raise ValueError('补充轮数超过预算')
    for index, record in enumerate(rounds):
        check()
        if record['index'] != index or type(record['accepted']) is not bool:raise ValueError('补充轮次顺序无效')
        if len(record['candidates']) > budget.candidates_per_round:raise ValueError('补充候选数量超过预算')
        if strict:
            from femto.planning.strict_refinement import validate_candidate
            for candidate in record['candidates']:
                if candidate['status']=='checked':validate_candidate(candidate)
        if record['accepted']:
            gain = record['comparison']
            if (record['status'] != 'accepted' or gain.get('retained_subset') is not True
                    or gain['lost_weight_units'] != 0 or gain['gained_weight_units'] <= 0
                    or gain['new_scan_witness_weight_units'] <= 0 or gain['new_scan_witness_gain_m2']*1e6 < budget.min_gain_mm2):
                raise ValueError('补充路线缺少保留原覆盖的增益记录')
            for candidate in record['candidates']:
                if candidate['status'] == 'checked':_append_record(chains, candidate)
    expected = refinement._route(base, chains, context, report)
    expected = replace(expected, planning_context=trajectory.planning_context)
    if semantic_payload(expected) != semantic_payload(trajectory):raise ValueError('补充路线改变了原路线或新增边记录')
    if (type(report['edge_checks']) is not int or not 0 <= report['edge_checks'] <= budget.max_edge_checks
            or trajectory.statistics['point_count'] > budget.max_output_points
            or report['max_total_length_m'] != base.statistics['total_length_m']+budget.max_added_length_m
            or trajectory.statistics['total_length_m'] > report['max_total_length_m']):
        raise ValueError('补充路线超出累计预算')
    # A refused refinement emits byte-identical ordered items. The original
    # v3 validator above has already checked every one of those edges against
    # this mesh and selection; do not run the same predicates a second time.
    if trajectory.to_dict()['ordered_items'] != base.to_dict()['ordered_items']:
        _check_final(trajectory, mesh, region, check)
