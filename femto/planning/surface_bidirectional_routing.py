"""v4.3: bounded directed, bidirectional A* over the same measured face portals.

Both frontiers share candidates and computation caps. Backward expansion checks
predecessor -> current in the final travel direction, never assumes visibility
is numerically symmetric, and never publishes an unverified connecting segment.
"""
from __future__ import annotations

from typing import Any, Iterator
import heapq
import math

import numpy as np

from femto.planning.surface_face_routing import FacePortalSurfaceLinks, _PortalSearch
from femto.planning.surface_link_optimization import OptimizedSurfaceLinks
from femto.planning.surface_trajectory import _point_on_triangle


class _BidirectionalPortalSearch(_PortalSearch):
    """Balanced Euclidean potentials; meeting alone is not a stopping condition."""

    def neighbors(self, node: int, distance: float, distances: dict[int, float],
                  *, reverse: bool = False) -> Iterator[tuple[int, float, int]]:
        for face in self.node_faces[node]:
            nodes = self.nodes_on_face(face)
            for other in nodes:
                self.budget.check()
                if other == node:
                    continue
                length = float(np.linalg.norm(self.points[other] - self.points[node]))
                if distance + length >= distances.get(other, math.inf):
                    continue
                if self._face_visibility_certified(face, nodes):
                    yield other, length, face
                    continue
                first, second = (other, node) if reverse else (node, other)
                key = (first, second, face)
                if key not in self.visibility:
                    self.budget.consume('visibility checks')
                    self.visibility[key] = self.router._state(self.points[first], self.points[second]) == 0
                if self.visibility[key]:
                    yield other, length, face

    def _path(self, meeting: int, previous: list[dict[int, tuple[int, int]]]) -> tuple[np.ndarray, list[int]]:
        nodes, faces = [meeting], []
        while nodes[-1] != 0:
            self.budget.check()
            node, face = previous[0][nodes[-1]]
            nodes.append(node)
            faces.append(face)
        nodes.reverse()
        faces.reverse()
        node = meeting
        while node != 1:
            self.budget.check()
            node, face = previous[1][node]
            nodes.append(node)
            faces.append(face)
        return np.asarray([self.points[node] for node in nodes]), faces

    def run(self) -> tuple[np.ndarray, list[int]] | None:
        for point, face in zip(self.points, self.endpoint_faces):
            if (face not in self.router.member or self.router._state(point, point) != 0
                    or not _point_on_triangle(point, self.router.vertices, self.router.faces[face])):
                return None
        self._endpoints_checked = True
        potentials: dict[int, float] = {}

        def potential(node: int) -> float:
            if node not in potentials:
                potentials[node] = (float(np.linalg.norm(self.points[node] - self.points[1]))
                                    - float(np.linalg.norm(self.points[node] - self.points[0]))) / 2
            return potentials[node]

        distances = [{0: 0.}, {1: 0.}]
        previous: list[dict[int, tuple[int, int]]] = [{}, {}]
        queues = [[(potential(0), 0., 0)], [(-potential(1), 0., 1)]]
        best, meeting = math.inf, None
        while queues[0] and queues[1]:
            self.budget.check()
            for side in (0, 1):
                while queues[side] and queues[side][0][1] != distances[side].get(queues[side][0][2]):
                    self.budget.check()
                    heapq.heappop(queues[side])
            # Potentials are +p and -p, so their offsets cancel. Each queue head
            # bounds its unfinished reduced-cost paths; the sum must reach the
            # best complete path before stopping. First contact is insufficient.
            if not queues[0] or not queues[1] or queues[0][0][0] + queues[1][0][0] >= best:
                break
            side = 0 if queues[0][0][0] <= queues[1][0][0] else 1
            _, distance, node = heapq.heappop(queues[side])
            for other, length, face in self.neighbors(node, distance, distances[side], reverse=side == 1):
                cost = distance + length
                if cost < distances[side].get(other, math.inf):
                    distances[side][other] = cost
                    previous[side][other] = (node, face)
                    priority = cost + (1 if side == 0 else -1) * potential(other)
                    heapq.heappush(queues[side], (priority, cost, other))
                    complete = cost + distances[1 - side].get(other, math.inf)
                    if complete < best:
                        best, meeting = complete, other
        return None if meeting is None else self._path(meeting, previous)


class BidirectionalFacePortalSurfaceLinks(FacePortalSurfaceLinks):
    """Keep successful v4.1 routes; replace only the supplemental portal search."""

    def route(self, left: dict[str, Any], right: dict[str, Any], step: float, e2: Any,
              *, adjacent_rows: bool) -> tuple[list[dict[str, Any]] | None, str | None]:
        samples, method = OptimizedSurfaceLinks.route(self, left, right, step, e2, adjacent_rows=adjacent_rows)
        if samples:
            return samples, method
        self.check()
        self._prepare_face_portals()
        path = _BidirectionalPortalSearch(self, left, right).run()
        if path is None:
            return None, None
        # Forward whole-segment, supporting-face and resampled-edge verification
        # is unchanged, including the separate original sample budget.
        samples = self._face_samples(*path, right, step)
        return (samples, 'surface_face_portals_bidirectional') if samples else (None, None)
