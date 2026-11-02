"""v4.1: shorten a proven mesh-edge connector inside its existing triangle corridor.

This is local length minimisation, not a global geodesic or a smooth/robot-ready path.
No surface is created, no scan line is moved, and successful boundary-band turns stay frozen.
"""
from __future__ import annotations

from collections import Counter
from typing import Callable
import math

import numpy as np

from femto.planning.surface_row_turns import SurfaceLinks
from femto.planning.surface_trajectory import POINT_BUDGET, _point_on_triangle


def _length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _supported(a: np.ndarray, b: np.ndarray, face: int, vertices: np.ndarray, faces: np.ndarray) -> bool:
    return (_point_on_triangle(a, vertices, faces[face])
            and _point_on_triangle(b, vertices, faces[face]))


def _compress_faces(points: np.ndarray, edge_faces: list[int],
                    allowed: Callable[[np.ndarray, np.ndarray], bool],
                    check: Callable[[], None]) -> tuple[np.ndarray, list[int]]:
    result, result_faces = [points[0].copy()], []
    start = 0
    while start < len(edge_faces):
        check()
        end = start + 1
        while end < len(edge_faces) and edge_faces[end] == edge_faces[start]:
            end += 1
        if end > start + 1 and allowed(points[start], points[end]):
            result.append(points[end].copy())
            result_faces.append(edge_faces[start])
        else:
            result.extend(p.copy() for p in points[start + 1:end + 1])
            result_faces.extend(edge_faces[start:end])
        start = end
    return np.asarray(result), result_faces


def optimize_surface_path(vertices: np.ndarray, faces: np.ndarray, points: np.ndarray,
                          edge_faces: list[int],
                          edge_allowed: Callable[[np.ndarray, np.ndarray], bool],
                          check: Callable[[], None] = lambda: None,
                          *, edge_counts: Counter | None = None, sweeps: int = 4) -> tuple[np.ndarray, list[int]]:
    """Return an independent, no-longer path with the same endpoints and supported corridor.

    The retained default is four forward/backward sweeps. A versioned caller
    can request fewer sweeps; this is a computation limit, not a quality gate.
    At each manifold portal the exact unconstrained local length minimum is tested; if a
    selection/exclusion constraint rejects it, the existing feasible vertex is retained.
    Vertex-only adjacency and non-manifold edges are never used as movable portals.
    """
    if type(sweeps) is not int or not 1<=sweeps<=4:
        raise ValueError('Surface path sweeps must be an integer from 1 to 4')
    check()
    original = np.asarray(points, dtype=float).copy()
    original_faces = [int(face) for face in edge_faces]
    if original.ndim != 2 or original.shape[1] != 3 or len(original) != len(original_faces) + 1:
        raise ValueError('Surface path needs one supporting face per edge')
    if not original_faces:
        return original, original_faces
    for index, face in enumerate(original_faces):
        check()
        if not _supported(original[index], original[index + 1], face, vertices, faces):
            return original, original_faces
    result, result_faces = _compress_faces(original, original_faces, edge_allowed, check)
    if edge_counts is None:
        edge_counts = Counter()
        for index, face in enumerate(faces):
            if index % 1024 == 0:
                check()
            edge_counts.update(tuple(sorted((int(a), int(b)))) for a, b in zip(face, np.roll(face, -1)))
    portals = {}
    for index in range(1, len(result) - 1):
        check()
        left_face, right_face = result_faces[index - 1:index + 1]
        if left_face == right_face:
            continue
        shared = sorted(set(faces[left_face]).intersection(faces[right_face]))
        if len(shared) == 2 and edge_counts[tuple(shared)] == 2:
            a, b = vertices[shared]
            vector = b - a
            squared = float(vector @ vector)
            if squared > 0:
                portals[index] = (a, vector, squared)
    for _ in range(sweeps):
        for indices in (range(1, len(result) - 1), range(len(result) - 2, 0, -1)):
            for index in indices:
                check()
                if index not in portals:
                    continue
                a, vector, squared = portals[index]
                before, current, after = result[index - 1:index + 2]
                t0, t1 = float((before - a) @ vector / squared), float((after - a) @ vector / squared)
                h0 = float(np.linalg.norm(before - a - t0 * vector))
                h1 = float(np.linalg.norm(after - a - t1 * vector))
                if h0 + h1 == 0:
                    continue
                t = float(np.clip((h1 * t0 + h0 * t1) / (h0 + h1), 0, 1))
                candidate = a + t * vector
                old = float(np.linalg.norm(before - current) + np.linalg.norm(after - current))
                new = float(np.linalg.norm(before - candidate) + np.linalg.norm(after - candidate))
                if (new < old and edge_allowed(before, candidate) and edge_allowed(candidate, after)
                        and _supported(before, candidate, result_faces[index - 1], vertices, faces)
                        and _supported(candidate, after, result_faces[index], vertices, faces)):
                    result[index] = candidate
    # A final whole-path check prevents numerical growth or a damaged face assignment.
    if _length(result) > _length(original):
        return original, original_faces
    for index, face in enumerate(result_faces):
        check()
        if (not edge_allowed(result[index], result[index + 1])
                or not _supported(result[index], result[index + 1], face, vertices, faces)):
            return original, original_faces
    return result, result_faces


class OptimizedSurfaceLinks(SurfaceLinks):
    """Keep v4 search/constraints; shorten only its mesh-edge fallback connectors."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_shortened = False
        self._portal_edge_counts = Counter()
        for index, face in enumerate(self.faces):
            if index % 1024 == 0:
                self.check()
            self._portal_edge_counts.update(tuple(sorted((int(a), int(b)))) for a, b in zip(face, np.roll(face, -1)))

    def _samples(self, path, left, right, step):
        from femto.planning.mesh_slice_planner import _project
        self._last_shortened = False
        endpoints = {-2: np.asarray(left['xyz_m'], float), -1: np.asarray(right['xyz_m'], float)}
        def point(index):
            return endpoints[index] if index < 0 else self.vertices[index]
        xyz = np.asarray([point(path[0][0])] + [point(end) for _, end, _ in path])
        ids = [face for _, _, face in path]
        optimized, face_ids = optimize_surface_path(self.vertices, self.faces, xyz, ids,
            lambda a, b: self._state(a, b) == 0, self.check, edge_counts=self._portal_edge_counts)
        if _length(optimized) >= _length(xyz):
            return super()._samples(path, left, right, step)
        samples = []
        for a, b, face in zip(optimized, optimized[1:], face_ids):
            self.check()
            length = float(np.linalg.norm(b - a))
            if length == 0:
                continue
            count = max(1, math.ceil(length / step))
            if len(samples) + count > POINT_BUDGET:
                raise ValueError(f'Surface link exceeds {POINT_BUDGET} samples')
            for t in np.linspace(0, 1, count + 1)[1:]:
                p = a * (1 - t) + b * t
                samples.append(dict(xyz_m=p.tolist(), pixel=_project(p, self.k, self.size),
                    triangle_index=int(face), geometry_support='measured_mesh', role='connector',
                    row_index=right['row_index']))
        self._last_shortened = bool(samples)
        return samples or None

    def route(self, left, right, step, e2, *, adjacent_rows):
        self._last_shortened = False
        samples, method = super().route(left, right, step, e2, adjacent_rows=adjacent_rows)
        if samples and method == 'surface_edges' and self._last_shortened:
            method = 'surface_corridor_shortened'
        return samples, method
