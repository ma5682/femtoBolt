"""Build and independently replay bounded witness-preserving research routes.

This research workflow cannot install a product route, claim full target
coverage, or command equipment. Its old and candidate inputs, target, all
resulting edges and a single final proof are independently validated.
"""
from __future__ import annotations

from dataclasses import replace
from femto.planning import (auto_supplement_route_research, coverage_certificate as coverage,
    coverage_partition, coverage_workflow, fraction_footprint, fraction_transfer,
    planning_revision, row_proximity_research, scan_fraction_seeds, supplement_route_validation,
    surface_workspace, target_surface, trim_subset_research, witness_bend_research)
from femto.planning.coverage_domain import from_plan, scan_endpoints
from femto.planning.coverage_retention_union import _same_target
from femto.planning.surface_trajectory import ScanSegment, SurfaceTrajectory

VERSION = 'research-witness-bend-workflow-v1'
REPLAY_VERSION = 'research-witness-bend-replay-v1'
EDGE_VERSION = 'research-witness-edge-insertion-workflow-v1'
EDGE_REPLAY_VERSION = 'research-witness-edge-insertion-replay-v1'
EDGE_RECIPE_VERSION = 'research-witness-edge-insertion-recipe-v1'
MAX_PROOFS = 4
EPS = 1e-12


def _snapshot(old_plan, base_plan, requests, old_proof, prior_proof):
    return planning_revision.digest(dict(old=old_plan, base=base_plan, requests=requests,
        old_proof=old_proof.metadata, prior_proof=prior_proof.metadata))


def _prepare(model, region, old_plan, old_proof, base_plan, prior_requests, prior_proof, check, *, edge_insert=False, _old_proof_session=None):
    if not isinstance(prior_requests, list) or len(prior_requests) > scan_fraction_seeds.MAX_REQUESTS:
        raise ValueError('见证点研究位置请求无效或超过预算')
    old_domain, old_seeds, old_width, _ = from_plan(model, region, old_plan, check)
    old = SurfaceTrajectory.from_dict(old_plan['surface_trajectory'], check=check)
    old_route = coverage_workflow.route_binding(old)
    if _old_proof_session is None:
        coverage.validate(old_proof, old_domain, old_seeds, old_width, route_binding=old_route, check=check)
    else:
        from femto.planning.old_coverage_session_research import _OldCoverageSession
        if type(_old_proof_session) is not _OldCoverageSession: raise ValueError('原证据复用上下文无效')
        _old_proof_session.require(model, region, old_plan, old_proof, old_domain, old_seeds, old_width, old_route, check)
    domain, endpoints, width, _ = from_plan(model, region, base_plan, check)
    base = SurfaceTrajectory.from_dict(base_plan['surface_trajectory'], check=check)
    _same_target(old_domain, domain, check)
    if old_width != width or width.get('schema') != 'femto-nominal-fixed-ball-v1':
        raise ValueError('见证点研究需要相同固定名义作用宽度')
    positions, _ = scan_fraction_seeds.propose(base, domain.ledger.arrays['region_faces'], prior_requests, check=check)
    nominal = fraction_footprint.nominal(width['effective_width_mm'])
    coverage.validate(prior_proof, domain, endpoints+positions, nominal,
        route_binding=coverage_workflow.route_binding(base), guide=old_proof, check=check)
    previous = coverage_partition.compare_union(old_proof, [prior_proof], domain.areas,
                                                check, include_pending_leaves=True)
    reconnect = base.gap_report.get('reconnection') or {}
    segments = [ScanSegment.from_dict(item) for item in reconnect.get('base_items', [])
                if item['kind'] == 'scan_segment']
    if not segments or not isinstance(reconnect.get('connections'), list):
        raise ValueError('见证点研究缺少可重建的扫描及连接记录')
    visits = auto_supplement_route_research._source_visits(base, segments)
    proposals, proposal_report = witness_bend_research.propose(
        old, segments, old_domain.ledger.arrays['region_faces'], old_proof,
        previous['pending_old_leaf_indices'], representative_faces=old_domain.representative_faces,
        edge_insert=edge_insert, check=check)
    mesh = surface_workspace.canonical_region_mesh(model.store.mesh, region)
    return dict(old_domain=old_domain, domain=domain, old=old, base=base, segments=segments,
                visits=visits, links=reconnect['connections'], width=width, nominal=nominal,
                mesh=mesh, previous=previous, proposals=proposals, proposal_report=proposal_report,
                old_density=row_proximity_research.measure(old, check=check),
                base_density=row_proximity_research.measure(base, check=check))


def _point_checks(items, params, mesh, check):
    from femto.geometry.triangle_support import prepare_triangle_faces
    from femto.planning.surface_trajectory import _validate_scan_points, CONNECTED_ALGORITHM
    vertices, faces, prepared = prepare_triangle_faces(mesh.vertices, mesh.faces)
    expected = 0
    for item in items:
        check()
        if not isinstance(item, ScanSegment): continue
        points = [{key: value for key, value in point.items() if key != 'revisit'} for point in item.points]
        for point in points: point['role'] = 'scan'
        shadow = replace(item, points=points, row_index=min(p['row_index'] for p in points), block_index=None)
        expected = _validate_scan_points(shadow, CONNECTED_ALGORITHM, True, False,
            expected, params['sample_step_mm']/1000, vertices, faces, _prepared_faces=prepared)
    check()


def _candidate(state, region, proposal, check, budget):
    base = state['base']
    items, statistics, blocks, geometry = supplement_route_validation.reconstruct_relocated_and_validate(
        base, proposal['segments'], state['visits'], state['links'], state['mesh'], region, check=check, budget=budget)
    _point_checks(items, base.planner['parameters'], state['mesh'], check)
    final = replace(base, ordered_items=items, statistics=statistics, blocks=blocks)
    density = row_proximity_research.measure(final, check=check)
    return final, statistics, geometry, density


def _acceptable(state, statistics, density):
    old = state['old'].statistics; base = state['base'].statistics
    return (trim_subset_research._motion_ok(statistics, old)
            and statistics['total_length_m'] <= old['total_length_m'] + EPS
            and statistics['gap_count']-statistics['inter_block_gap_count'] <= old['gap_count']-old['inter_block_gap_count']
            and statistics['gap_count']-statistics['inter_block_gap_count'] <= base['gap_count']-base['inter_block_gap_count']
            and trim_subset_research._density_not_worse(density, state['old_density'])
            and trim_subset_research._density_not_worse(density, state['base_density'])
            and any(item['near_fraction'] < prior['near_fraction']-EPS
                    for item, prior in zip(density['proximity'], state['old_density']['proximity'])))


def _proof_inputs(state, final, geometry, prior_requests, check):
    domain = state['domain']
    requests, positions, transfer = fraction_transfer.transfer(
        state['base'], final, domain.ledger.arrays['region_faces'], prior_requests, check=check)
    seeds = scan_endpoints(final, domain.ledger.arrays['region_faces'])+positions
    route = dict(schema='research-checked-supplement-route-v1', computation_sha256=domain.revision['sha256'],
        source_route=coverage_workflow.route_binding(state['base']), items_sha256=geometry['items_sha256'],
        execution_ready=False)
    return requests, seeds, route, transfer


def _finish(model, region, old_plan, base_plan, old_proof, prior_proof, proof, state, snapshot, requests, check):
    if _snapshot(old_plan, base_plan, requests, old_proof, prior_proof) != snapshot:
        raise ValueError('见证点研究期间输入发生变化')
    for plan, domain in ((old_plan, state['old_domain']), (base_plan, state['domain'])):
        planning_revision.require_current(model, region, plan['treatment_head'], plan['planning_context']['revision'])
        params = plan['surface_trajectory']['planner']['parameters']
        target_surface.validate(domain.ledger, plan['planning_context']['revision'], evidence=params['quality_evidence'],
                                domain=params['effective_domain'], check=check, reproduce=False)
    for cert in (old_proof, prior_proof, *([proof] if proof is not None else [])):
        coverage._checked_assets(cert, check)
    check()
    if _snapshot(old_plan, base_plan, requests, old_proof, prior_proof) != snapshot:
        raise ValueError('见证点研究期间输入发生变化')


def build(model, region, old_plan, old_proof, base_plan, prior_requests, prior_proof, *,
          max_proofs=2, proof_budget=None, check=lambda: None, budget=200_000, placement='replace', _old_proof_session=None):
    """Return a qualifying research recipe and proof or a detached rejection."""
    if (type(max_proofs) is not int or not 1 <= max_proofs <= MAX_PROOFS
            or type(budget) is not int or not 1 <= budget <= 200_000
            or placement not in ('replace', 'edge_insertion')):
        raise ValueError('见证点研究预算无效')
    check(); snapshot = _snapshot(old_plan, base_plan, prior_requests, old_proof, prior_proof)
    state = _prepare(model, region, old_plan, old_proof, base_plan, prior_requests, prior_proof, check,
                     edge_insert=placement == 'edge_insertion', _old_proof_session=_old_proof_session)
    report = dict(schema=EDGE_VERSION if placement == 'edge_insertion' else VERSION, adopted=False, reason='no_qualifying_candidate',
        proposals=state['proposal_report'], prior_comparison=state['previous'], trials=[],
        installed_route=False, installed_certificate=False, full_target_coverage_claim=False,
        execution_ready=False, physical_accuracy_verified=False)
    accepted = None; evaluated = 0
    algorithm = (coverage.RESEARCH_PAGED_FRACTION_REPROOF_ALGORITHM
                 if coverage.checked_page_binding(state['domain'].binding) else coverage.RESEARCH_FRACTION_REPROOF_ALGORITHM)
    for proposal in state['proposals']:
        check()
        identity = {key: proposal[key] for key in ('seed', 'segment', 'point', 'distance_m', 'pending_leaves')}
        if placement == 'edge_insertion':
            identity.update(placement=proposal['placement'], edge=proposal['edge'], projection_fraction=proposal['projection_fraction'])
        try: final, statistics, geometry, density = _candidate(state, region, proposal, check, budget)
        except ValueError as error:
            check(); report['trials'].append(dict(proposal=identity, reason='geometry_rejected', detail=str(error))); continue
        if not _acceptable(state, statistics, density):
            report['trials'].append(dict(proposal=identity, reason='motion_or_density_rejected',
                                         statistics=statistics, density=density)); continue
        if evaluated >= max_proofs:
            report['reason'] = 'proof_budget_exhausted'; break
        evaluated += 1
        requests, seeds, route, transfer = _proof_inputs(state, final, geometry, prior_requests, check)
        proof = coverage.evaluate(state['domain'], seeds, state['nominal'], route_binding=route, guide=old_proof,
                                  algorithm=algorithm, budget=proof_budget, check=check)
        coverage.validate(proof, state['domain'], seeds, state['nominal'], route_binding=route, guide=old_proof, check=check)
        comparison = coverage_partition.compare_union(old_proof, [proof], state['domain'].areas,
                                                      check, include_pending_leaves=True)
        report['trials'].append(dict(proposal=identity, reason='accepted' if comparison['retained_subset'] else 'not_reconfirmed',
                                     statistics=statistics, density=density, comparison=comparison))
        if comparison['retained_subset']:
            accepted = proof
            report.update(adopted=True, reason='research_candidate_validated', geometry=geometry,
                          statistics=statistics, density=density, coverage_summary=proof.summary,
                          transfer=transfer, comparison=comparison,
                          recipe=dict(seed=proposal['seed'], segment=proposal['segment'], point=proposal['point'],
                                      requests=[list(value) for value in requests]))
            if placement == 'edge_insertion':
                report['recipe'].update(version=EDGE_RECIPE_VERSION, placement=placement, edge=proposal['edge'])
                from femto.planning import witness_edge_direct_replay_research as direct
                report['direct_recipe'] = direct.recipe(old_plan, old_proof, base_plan, identity,
                                                        report['recipe']['requests'])
            break
    _finish(model, region, old_plan, base_plan, old_proof, prior_proof, accepted, state, snapshot, prior_requests, check)
    return report, accepted


def replay(model, region, old_plan, old_proof, base_plan, prior_requests, prior_proof, recipe, proof, *,
           check=lambda: None, budget=200_000):
    """Re-derive candidate, geometric checks and single proof from source data."""
    old_keys = {'seed', 'segment', 'point', 'requests'}
    edge_keys = old_keys | {'version', 'placement', 'edge'}
    edge_insert = isinstance(recipe, dict) and set(recipe) == edge_keys
    if (not isinstance(recipe, dict) or set(recipe) not in (old_keys, edge_keys)
            or any(type(recipe[key]) is not int or recipe[key] < 0 for key in ('seed', 'segment', 'point'))
            or not isinstance(recipe['requests'], list) or len(recipe['requests']) > scan_fraction_seeds.MAX_REQUESTS
            or (edge_insert and (recipe['version'] != EDGE_RECIPE_VERSION or recipe['placement'] != 'edge_insertion'
                                 or type(recipe['edge']) is not int or recipe['edge'] < 0 or recipe['point'] != recipe['edge']+1))
            or type(budget) is not int or not 1 <= budget <= 200_000):
        raise ValueError('见证点研究配方无效或超过预算')
    check(); snapshot = _snapshot(old_plan, base_plan, prior_requests, old_proof, prior_proof); recipe_snapshot = planning_revision.digest(recipe)
    state = _prepare(model, region, old_plan, old_proof, base_plan, prior_requests, prior_proof, check, edge_insert=edge_insert)
    identity_keys = ('seed', 'segment', 'point', 'edge') if edge_insert else ('seed', 'segment', 'point')
    selected = [p for p in state['proposals'] if all(p[key] == recipe[key] for key in identity_keys)]
    if len(selected) != 1: raise ValueError('见证点配方不属于重新推导的候选')
    final, statistics, geometry, density = _candidate(state, region, selected[0], check, budget)
    if not _acceptable(state, statistics, density): raise ValueError('见证点配方未满足运动与密度约束')
    requests, seeds, route, transfer = _proof_inputs(state, final, geometry, prior_requests, check)
    if recipe['requests'] != [list(value) for value in requests]: raise ValueError('见证点配方的边内位置不一致')
    coverage.validate(proof, state['domain'], seeds, state['nominal'], route_binding=route, guide=old_proof, check=check)
    comparison = coverage_partition.compare_union(old_proof, [proof], state['domain'].areas, check, include_pending_leaves=True)
    if not comparison['retained_subset']: raise ValueError('见证点配方未重新确认原有名义集合')
    if planning_revision.digest(recipe) != recipe_snapshot: raise ValueError('见证点配方重放期间发生变化')
    _finish(model, region, old_plan, base_plan, old_proof, prior_proof, proof, state, snapshot, prior_requests, check)
    return dict(schema=EDGE_REPLAY_VERSION if edge_insert else REPLAY_VERSION, geometry=geometry, statistics=statistics, density=density,
                transfer=transfer, comparison=comparison, coverage_summary=proof.summary,
                installed_route=False, installed_certificate=False, full_target_coverage_claim=False,
                execution_ready=False, physical_accuracy_verified=False)
