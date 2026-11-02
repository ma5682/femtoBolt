"""Versioned, bounded reference screening before ordinary whole-edge admission."""
from bisect import insort
from collections import Counter
import math

import numpy as np

from femto.geometry import exact_surface as exact
from femto.planning import quality_constrained as quality
from femto.planning.refinement_candidates import ranked_faces
from femto.planning.surface_trajectory import ScanSegment

ALGORITHM = 'quality-screened-symmetric-face-slices-v2'
POLICY = dict(algorithm=ALGORITHM, max_faces_per_round=256, max_positions_per_face=3,
              max_rejection_examples=32, max_length_m=.004, min_length_m=.002,
              endpoint_inset=.02, length_bisections=6,
              screening='shared_reference_envelope_only_not_path_acceptance',
              screening_charged_to_edge_budget=True,
              order='shared_face_group_area_then_face_area_then_leaf_area',
              direction='largest_symmetric_interval_among_original_face_edges',
              connection_length_policy='per_candidate_share_and_remaining_total')


def checked_policy(value):
    if (not isinstance(value, dict) or set(value) != set(POLICY)
            or any(type(value[key]) is not type(expected) or value[key] != expected for key,expected in POLICY.items())):
        raise ValueError('补充候选策略版本或限额不受支持')
    return value


def connection_limit(chains, points, trajectory, meter):
    """Optional detours cannot consume the other bounded candidates' allowance."""
    if getattr(meter, 'candidate_policy', None) is None:return None
    def length(values):
        return float(np.linalg.norm(np.diff([p['xyz_m'] for p in values], axis=0), axis=1).sum())
    scan = length(points)
    share = meter.budget.max_added_length_m/max(1, meter.budget.max_rounds*meter.budget.candidates_per_round)
    remaining = (trajectory.gap_report['refinement']['max_total_length_m']
                 - math.fsum(length(values) for _, values in chains))
    return dict(policy=POLICY['connection_length_policy'], maximum_m=max(0., min(share, remaining)-scan),
                scan_length_m=scan, candidate_share_m=share, remaining_total_m=remaining)


def connection_fits(budget, points, stage):
    if budget is None:return True
    length = float(np.linalg.norm(np.diff([p['xyz_m'] for p in points], axis=0), axis=1).sum())
    budget[stage+'_length_m'] = length
    return length <= budget['maximum_m']


def _positions(domain, certificate, recipes, check):
    wanted = {row['source_face'] for row in recipes}
    result = {}
    arrays = certificate.arrays
    for count, index in enumerate(np.flatnonzero(arrays['status'] == 3)):
        if count % 256 == 0:check()
        root = int(arrays['root'][index]);face = int(domain.source_faces[root])
        if face not in wanted or domain.quality[root]:continue
        depth, code = int(arrays['depth'][index]), int(arrays['code'][index])
        key = (-float(domain.areas[root])/4**depth, root, depth, code)
        values = result.setdefault(face, [])
        insort(values, key)
        if len(values) > POLICY['max_positions_per_face']:values.pop()
    return result


def _geometry(recipe, domain, mesh):
    bary = domain.root(recipe['root'])
    if bary is None:return None
    bary = exact.descend(bary, recipe['depth'], recipe['code'])
    centre = np.array([float(sum(point[axis] for point in bary)/3) for axis in range(3)])
    triangle = np.asarray(mesh.vertices)[mesh.faces[recipe['local_face']]]
    lengths = np.linalg.norm(np.roll(triangle, -1, axis=0)-triangle, axis=1)
    spans = np.minimum(centre, np.roll(centre, -1))*lengths*(1-POLICY['endpoint_inset'])
    edge = int(np.argmax(spans))
    return centre, triangle, lengths, edge, min(float(spans[edge]), POLICY['max_length_m']/2)


def select(current, domain, certificate, mesh, region, allowed, attempted, meter, check):
    """Choose at most the existing per-round candidate cap; keep all omissions unknown."""
    recipes = ranked_faces(domain, certificate, mesh, region, current.blocks, allowed, attempted, check)
    pool = recipes[:POLICY['max_faces_per_round']]
    positions = _positions(domain, certificate, pool, check)
    reference = quality.ReferenceEnvelope(current.planner['parameters']['quality_evidence'],
                                          current.planner['parameters']['effective_width_mm'])
    info = dict(algorithm=ALGORITHM, ranked_faces=len(recipes), screened_faces=0,
        positions_considered=0, reference_checks=0, residual_pixels_examined=0,
        candidates_selected=0, deferred_faces=max(0, len(recipes)-len(pool)),
        rejection_counts={}, rejection_examples=[], screening_is_admission=False,
        unexamined_ranked_faces=len(recipes))
    meter.search_info = info
    reasons = Counter()
    selected = []

    def probe(mid, half):
        check();meter.reserve(1)
        info['reference_checks'] += 1
        value = reference.probe(mid, half, details=True)
        info['residual_pixels_examined'] = reference.sampled_pixels
        return value

    def reject(recipe, reason, detail=None):
        reasons[reason] += 1
        info['rejection_counts'] = dict(sorted(reasons.items()))
        if len(info['rejection_examples']) < POLICY['max_rejection_examples']:
            info['rejection_examples'].append(dict(local_face=recipe['local_face'],
                source_face=recipe['source_face'], root=recipe['root'], depth=recipe['depth'], code=recipe['code'],
                reason=reason, reference=detail))

    for template in pool:
        check()
        if len(selected) >= meter.budget.candidates_per_round:break
        info['screened_faces'] += 1
        info['unexamined_ranked_faces'] = len(recipes)-info['screened_faces']
        attempted.add(template['local_face'])
        for _, root, depth, code in positions.get(template['source_face'], ()):
            recipe = dict(template, root=root, depth=depth, code=code)
            info['positions_considered'] += 1
            geometry = _geometry(recipe, domain, mesh)
            if geometry is None:
                reject(recipe, 'target_construction_unconfirmed');continue
            centre, triangle, _, edge, maximum = geometry
            minimum = POLICY['min_length_m']/2
            if maximum < minimum:
                reject(recipe, 'short_symmetric_face_interval');continue
            mid = centre@triangle
            passed, _, detail = probe(mid, minimum)
            if not passed:
                reject(recipe, detail['reason'], detail);continue
            half = maximum
            if maximum > minimum:
                passed, _, _ = probe(mid, maximum)
                if not passed:
                    low, high = minimum, maximum
                    for _ in range(POLICY['length_bisections']):
                        middle = (low+high)/2
                        passed, _, _ = probe(mid, middle)
                        if passed:low = middle
                        else:high = middle
                    half = low
            selected.append(dict(recipe, candidate_algorithm=ALGORITHM,
                                 direction_edge=edge, half_length_m=float(half)))
            info['candidates_selected'] = len(selected)
            break
    info['unexamined_ranked_faces'] = len(recipes)-info['screened_faces']
    return selected


def candidate(recipe, domain, mesh, row, step):
    from femto.planning.mesh_slice_planner import _project
    if recipe.get('candidate_algorithm') != ALGORITHM:raise ValueError('补充候选构造版本不一致')
    geometry = _geometry(recipe, domain, mesh)
    if geometry is None:return None
    centre, triangle, lengths, edge, maximum = geometry
    half = recipe['half_length_m']
    if (type(half) not in (int, float) or not math.isfinite(half)
            or not POLICY['min_length_m']/2 <= half <= maximum or recipe['direction_edge'] != edge):
        raise ValueError('补充候选超出同一原面支持范围')
    derivative = np.zeros(3);derivative[edge] = -1;derivative[(edge+1)%3] = 1
    ends = np.array([centre-half/lengths[edge]*derivative, centre+half/lengths[edge]*derivative])@triangle
    length = float(np.linalg.norm(ends[1]-ends[0]))
    if length < POLICY['min_length_m']:return None
    count = max(1, math.ceil(length/step))
    if count > 200_000:raise ValueError('补充候选采样超过预算')
    k = np.asarray(domain.revision['inputs']['calibration']['intrinsic'])
    size = np.array(domain.ledger.arrays['mask'].shape[::-1])
    points = [dict(xyz_m=point.tolist(), triangle_index=recipe['local_face'],
        pixel=_project(point, k, size), geometry_support='measured_mesh', role='scan', row_index=row)
        for point in np.linspace(ends[0], ends[1], count+1)]
    return ScanSegment(0, row, points, recipe['block_index'])
