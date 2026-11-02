"""Bounded centre-face slices, ordered by uncovered shared-face components.

These are only candidates. The retained v3 whole-edge predicates decide which
subedges may be emitted; no new surface, distance welding or block transfer.
"""
from __future__ import annotations

from collections import defaultdict
import math
import numpy as np

from femto.geometry import exact_surface as exact
from femto.planning.surface_trajectory import ScanSegment


def ranked_faces(domain, certificate, local_mesh, region, blocks, allowed, attempted, check=lambda: None) -> list[dict]:
    from femto.planning.selection_blocks import selection_face_assignment
    _, _, _, owners = selection_face_assignment(region, np.asarray(local_mesh.vertices), np.asarray(local_mesh.faces))
    local_for = {int(face):i for i, face in enumerate(domain.ledger.arrays['region_faces'])}
    block_for = {item['component_id']:item['block_index'] for item in blocks['items']}
    weights, roots = defaultdict(list), {}
    a = certificate.arrays
    # Only a demonstrated ball-range deficit is eligible. Computation,
    # construction, connectivity and acquisition unknowns are not spacing gaps.
    for i in np.flatnonzero(a['status'] == 3):
        if int(i) % 256 == 0:check()
        root = int(a['root'][i]); face = int(domain.source_faces[root]); local = local_for.get(face)
        if domain.quality[root] or local is None or not allowed[local]:continue
        weight = float(domain.areas[root])/4**int(a['depth'][i])
        weights[local].append(weight)
        key = (-weight, root, int(a['depth'][i]), int(a['code'][i]))
        if local not in roots or key < roots[local]:roots[local] = key
    edges = defaultdict(list)
    for local in sorted(weights):
        check()
        face = local_mesh.faces[local]
        for x, y in zip(face, np.roll(face, -1)):edges[tuple(sorted((int(x), int(y))))].append(local)
    adjacent = defaultdict(set)
    for faces in edges.values():
        if len(faces) == 2 and owners[faces[0]] == owners[faces[1]]:
            a, b = faces;adjacent[a].add(b);adjacent[b].add(a)
    unseen = set(weights); groups = []
    while unseen:
        first = min(unseen);unseen.remove(first);stack = [first];members = []
        while stack:
            check();face = stack.pop();members.append(face)
            for other in sorted(adjacent[face] & unseen):unseen.remove(other);stack.append(other)
        area = math.fsum(v for face in members for v in weights[face])
        groups.append((-area, first, members))
    result = []
    for negative, first, members in sorted(groups):
        for local in sorted(members, key=lambda f:(-math.fsum(weights[f]), f)):
            if local in attempted:continue
            _, root, depth, code = roots[local]
            result.append(dict(local_face=local, source_face=int(domain.source_faces[root]),
                root=root, depth=depth, code=code, block_index=block_for[int(owners[local])],
                group_first_face=first, group_unconfirmed_nominal_area_m2=-negative))
    return result


def slice_candidate(recipe: dict, domain, mesh, row: int, step: float) -> ScanSegment | None:
    from femto.planning.mesh_slice_planner import _project
    bary = domain.root(recipe['root'])
    if bary is None:return None
    bary = exact.descend(bary, recipe['depth'], recipe['code'])
    centre = np.array([float(sum(p[i] for p in bary)/3) for i in range(3)])
    tri = np.asarray(mesh.vertices)[mesh.faces[recipe['local_face']]]
    # The longest face edge defines a reproducible slice direction, through
    # the selected uncovered leaf centre. Restrict it to the same real face.
    edges = np.roll(tri, -1, axis=0)-tri
    longest = int(np.argmax(np.linalg.norm(edges, axis=1)))
    derivative = np.zeros(3);derivative[longest] = -1;derivative[(longest+1)%3] = 1
    limits = [(0-centre[i])/v for i, v in enumerate(derivative) if v != 0]
    lo, hi = min(limits), max(limits)
    endpoints = np.array([centre+lo*.98*derivative, centre+hi*.98*derivative]) @ tri
    length = float(np.linalg.norm(endpoints[1]-endpoints[0]))
    if length < .002:return None
    count = max(1, math.ceil(length/step))
    if count > 200_000:raise ValueError('补充候选采样超过预算')
    k = np.asarray(domain.revision['inputs']['calibration']['intrinsic'])
    size = np.array(domain.ledger.arrays['mask'].shape[::-1])
    points = [dict(xyz_m=p.tolist(), triangle_index=recipe['local_face'],
        pixel=_project(p, k, size), geometry_support='measured_mesh', role='scan', row_index=row)
        for p in np.linspace(endpoints[0], endpoints[1], count+1)]
    return ScanSegment(0, row, points, recipe['block_index'])


def retained_chains(candidate, flags) -> list[tuple]:
    result, current = [], []
    for (left, right), accepted in zip(zip(candidate.points, candidate.points[1:]), flags):
        if accepted:
            if not current:current = [left]
            current.append(right)
        elif current:result.append((candidate.block_index, current));current = []
    if current:result.append((candidate.block_index, current))
    return result
