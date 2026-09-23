"""v4.2 fallback through measured, manifold shared-edge interior portals.

Successful v4.1 links stay unchanged. A finite set of projected pixel-interval
midpoints allows entering a face whose vertices lie outside the selected pixels.
This is neither a complete continuous-domain search nor a global geodesic method.
No mesh, selection mask, scan endpoint or geometric admission threshold is changed.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Iterator
import heapq
import math

import numpy as np

from femto.planning.surface_link_optimization import OptimizedSurfaceLinks
from femto.planning.surface_trajectory import POINT_BUDGET, _point_on_triangle


class _SearchBudget:
    """Independent hard computation caps; none is a geometric quality threshold."""

    def __init__(self, check: Callable[[], None]):
        self.check = check
        self.counts: dict[str, int] = defaultdict(int)

    def consume(self, kind: str, count: int = 1) -> None:
        self.check()
        self.counts[kind] += count
        if self.counts[kind] > POINT_BUDGET:
            raise ValueError(f'Surface face routing exceeds {POINT_BUDGET} {kind}')


def _portal_midpoints(a: np.ndarray, b: np.ndarray, intrinsic: np.ndarray,
                      image_size: np.ndarray, budget: _SearchBudget) -> Iterator[np.ndarray]:
    """Yield exact edge points at screen pixel-interval midpoints, not XYZ midpoints.

    Clipping cuts at the image borders keep an edge with outside-image vertices
    eligible where its interior is visible. Behind-camera edges are not admitted.
    The caller checks each candidate and every complete connecting segment.
    """
    budget.check()
    if not np.isfinite([a, b]).all() or a[2] <= 0 or b[2] <= 0 or np.array_equal(a, b):
        return
    uv = np.array([[intrinsic[0, 0] * p[0] / p[2] + intrinsic[0, 2],
                    intrinsic[1, 1] * p[1] / p[2] + intrinsic[1, 2]] for p in (a, b)])
    if not np.isfinite(uv).all():
        return
    cuts = {0., 1.}
    for axis in range(2):
        first, last = uv[:, axis]
        if first == last:
            continue
        low, high = sorted((float(first), float(last)))
        start = max(0, math.floor(low - .5) + 1)
        stop = min(int(image_size[axis]), math.ceil(high - .5))
        # Reserve before constructing a potentially large candidate sequence.
        budget.consume('pixel subdivisions', max(0, stop - start) + 2)
        for boundary in (0., float(image_size[axis])):
            s = float((boundary - first) / (last - first))
            if 0 < s < 1:
                cuts.add(s)
        for pixel in range(start, stop):
            budget.check()
            cuts.add(float((pixel + .5 - first) / (last - first)))
    ordered = sorted(cuts)
    for low, high in zip(ordered, ordered[1:]):
        budget.consume('portal candidates')
        s = (low + high) / 2
        # Perspective-correct inverse of projected linear interpolation.
        t = s * a[2] / ((1 - s) * b[2] + s * a[2])
        yield a * (1 - t) + b * t


class _PortalSearch:
    """Lazy A*: generate portals/visibility only on reached faces, with bounded work."""

    def __init__(self, router: FacePortalSurfaceLinks, left: dict[str, Any], right: dict[str, Any]):
        self.router = router
        self.budget = _SearchBudget(router.check)
        self.points = [np.asarray(left['xyz_m'], float), np.asarray(right['xyz_m'], float)]
        self.endpoint_faces = [int(left['triangle_index']), int(right['triangle_index'])]
        self.node_faces = [(self.endpoint_faces[0],), (self.endpoint_faces[1],)]
        self.edge_nodes: dict[tuple[int, int], list[int]] = {}
        self.face_nodes: dict[int, list[int]] = {}
        self.visibility: dict[tuple[int, int, int], bool] = {}
        self.budget.consume('graph nodes', 2)

    def nodes_on_face(self, face: int) -> list[int]:
        if face in self.face_nodes:
            return self.face_nodes[face]
        nodes = [index for index, owner in enumerate(self.endpoint_faces) if owner == face]
        for edge in self.router._face_portal_edges.get(face, []):
            if edge not in self.edge_nodes:
                new_nodes = []
                a, b = self.router.vertices[list(edge)]
                for point in _portal_midpoints(a, b, self.router.k, self.router.size, self.budget):
                    if self.router._state(point, point) != 0:
                        continue
                    self.budget.consume('graph nodes')
                    new_nodes.append(len(self.points))
                    self.points.append(point)
                    self.node_faces.append(self.router._portal_faces[edge])
                self.edge_nodes[edge] = new_nodes
            nodes.extend(self.edge_nodes[edge])
        self.face_nodes[face] = sorted(nodes)
        return self.face_nodes[face]

    def neighbors(self, node: int) -> Iterator[tuple[int, float, int]]:
        for face in self.node_faces[node]:
            for other in self.nodes_on_face(face):
                self.budget.check()
                if other == node:
                    continue
                key = (min(node, other), max(node, other), face)
                if key not in self.visibility:
                    self.budget.consume('visibility checks')
                    self.visibility[key] = self.router._state(self.points[node], self.points[other]) == 0
                if self.visibility[key]:
                    yield other, float(np.linalg.norm(self.points[other] - self.points[node])), face

    def run(self) -> tuple[np.ndarray, list[int]] | None:
        for point, face in zip(self.points, self.endpoint_faces):
            if (face not in self.router.member or self.router._state(point, point) != 0
                    or not _point_on_triangle(point, self.router.vertices, self.router.faces[face])):
                return None
        distances = {0: 0.}
        previous = {}
        queue = [(float(np.linalg.norm(self.points[0] - self.points[1])), 0., 0)]
        while queue:
            self.budget.check()
            _, distance, node = heapq.heappop(queue)
            if distance != distances.get(node):
                continue
            if node == 1:
                break
            for other, length, face in self.neighbors(node):
                cost = distance + length
                if cost < distances.get(other, math.inf):
                    distances[other] = cost
                    previous[other] = (node, face)
                    remaining = float(np.linalg.norm(self.points[other] - self.points[1]))
                    heapq.heappush(queue, (cost + remaining, cost, other))
        if 1 not in previous:
            return None
        nodes, faces = [1], []
        while nodes[-1] != 0:
            self.budget.check()
            node, face = previous[nodes[-1]]
            nodes.append(node)
            faces.append(face)
        return np.asarray([self.points[node] for node in reversed(nodes)]), list(reversed(faces))


class FacePortalSurfaceLinks(OptimizedSurfaceLinks):
    """Preserve v4.1 successes, then try a conservative interior-face fallback."""

    def _prepare_face_portals(self) -> None:
        if hasattr(self, '_face_portal_edges'):
            return
        edge_faces = defaultdict(list)
        for face in sorted(self.member):
            self.check()
            triangle = self.faces[face]
            for a, b in zip(triangle, np.roll(triangle, -1)):
                edge_faces[tuple(sorted((int(a), int(b))))].append(face)
        portal_faces = {}
        face_portal_edges = defaultdict(list)
        for edge, faces in sorted(edge_faces.items()):
            self.check()
            # Both faces must be in this fragment; non-manifold edges in the full
            # mesh remain forbidden even when a third face is outside this subset.
            if len(faces) != 2 or self._portal_edge_counts[edge] != 2:
                continue
            portal_faces[edge] = tuple(faces)
            for face in faces:
                face_portal_edges[face].append(edge)
        # Cancellation cannot leave a partially prepared cache reusable as complete.
        self._portal_faces = portal_faces
        self._face_portal_edges = face_portal_edges

    def _face_samples(self, points: np.ndarray, faces: list[int], right: dict[str, Any],
                      step: float) -> list[dict[str, Any]] | None:
        from femto.planning.mesh_slice_planner import _project
        samples = []
        for a, b, face in zip(points, points[1:], faces):
            self.check()
            if (self._state(a, b) != 0 or not _point_on_triangle(a, self.vertices, self.faces[face])
                    or not _point_on_triangle(b, self.vertices, self.faces[face])):
                return None
            length = float(np.linalg.norm(b - a))
            if length == 0:
                continue
            count = max(1, math.ceil(length / step))
            if len(samples) + count > POINT_BUDGET:
                raise ValueError(f'Surface face routing exceeds {POINT_BUDGET} samples')
            previous = a
            for t in np.linspace(0, 1, count + 1)[1:]:
                self.check()
                point = a * (1 - t) + b * t
                if self._state(previous, point) != 0:
                    return None
                samples.append(dict(xyz_m=point.tolist(), pixel=_project(point, self.k, self.size),
                    triangle_index=int(face), geometry_support='measured_mesh', role='connector',
                    row_index=right['row_index']))
                previous = point
        return samples or None

    def route(self, left: dict[str, Any], right: dict[str, Any], step: float, e2: Any,
              *, adjacent_rows: bool) -> tuple[list[dict[str, Any]] | None, str | None]:
        samples, method = super().route(left, right, step, e2, adjacent_rows=adjacent_rows)
        if samples:
            return samples, method
        self.check()
        self._prepare_face_portals()
        path = _PortalSearch(self, left, right).run()
        if path is None:
            return None, None
        samples = self._face_samples(*path, right, step)
        return (samples, 'surface_face_portals') if samples else (None, None)
