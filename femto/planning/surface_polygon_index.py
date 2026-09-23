"""Prepared screen-polygon predicates with the existing exact boundary rules.

Only immutable per-query-context data are indexed. Pixel selection, geometric
tolerances and parity/intersection semantics are unchanged; the arrays merely
avoid rebuilding every edge and testing obviously disjoint bounding boxes.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np


class _PreparedPolygon:
    def __init__(self, polygon: np.ndarray):
        self.a = np.asarray(polygon).copy()
        if not len(self.a):
            self.a = np.empty((0, 2), dtype=self.a.dtype)
        self.b = np.roll(self.a, -1, axis=0)
        self.minimum = np.minimum(self.a, self.b)
        self.maximum = np.maximum(self.a, self.b)

    def contains(self, point: np.ndarray) -> bool:
        if len(self.a) < 3:
            return False
        x, y = point
        active = (self.a[:, 1] > y) != (self.b[:, 1] > y)
        a, b = self.a[active], self.b[active]
        # Same strict ray crossing and operation order as _point_in_polygon.
        crossings = x < (b[:, 0] - a[:, 0]) * (y - a[:, 1]) / (b[:, 1] - a[:, 1]) + a[:, 0]
        return bool(np.count_nonzero(crossings) % 2)

    def intersects(self, a: np.ndarray, b: np.ndarray) -> bool:
        # Exactly the original bounding-box rejection, now evaluated in bulk.
        rejected = np.any(np.maximum(np.minimum(a, b), self.minimum) >
                          np.minimum(np.maximum(a, b), self.maximum) + 1e-9, axis=1)
        direction = b - a
        def cross(u: np.ndarray, v: np.ndarray) -> float:
            return float(u[0] * v[1] - u[1] * v[0])
        for index in np.flatnonzero(~rejected):
            c, d = self.a[index], self.b[index]
            # Keep scalar operations and frozen epsilon exactly as _intersects.
            if (cross(direction, c - a) * cross(direction, d - a) <= 1e-12 and
                    cross(d - c, a - c) * cross(d - c, b - c) <= 1e-12):
                return True
        return False


class PreparedPolygonIndex:
    """Snapshot already-validated polygons for repeated point/segment checks.

Rebuild this object if the source polygons change. No cache survives a planning
invocation, and no geometric acceptance result is reused across different edges.
"""

    def __init__(self, polygons: Iterable[np.ndarray]):
        self._polygons = tuple(_PreparedPolygon(polygon) for polygon in polygons)

    def contains(self, point: np.ndarray) -> bool:
        """Whether any polygon contains the point under the frozen parity rule."""
        point = np.asarray(point)
        return any(polygon.contains(point) for polygon in self._polygons)

    def blocks_segment(self, a: np.ndarray, b: np.ndarray) -> bool:
        """Whether an endpoint is inside or a segment touches/crosses a boundary."""
        a, b = np.asarray(a), np.asarray(b)
        return any(polygon.contains(a) or polygon.contains(b) or polygon.intersects(a, b)
                   for polygon in self._polygons)

    def boundary_outside_box(self, minimum: np.ndarray, maximum: np.ndarray) -> bool:
        """Prove every boundary edge is rejected by the frozen segment-box test.

        This does not prove that the box is outside a polygon: callers must have
        separately established that the endpoints being connected are not inside.
        Failure to prove separation says nothing about actual intersections.
        """
        minimum, maximum = np.asarray(minimum), np.asarray(maximum)
        if (not np.isfinite([minimum, maximum]).all()
                or np.any(minimum > maximum)):
            return False
        return all(np.all(np.any(np.maximum(minimum, polygon.minimum) >
                                 np.minimum(maximum, polygon.maximum) + 1e-9, axis=1))
                   for polygon in self._polygons)
