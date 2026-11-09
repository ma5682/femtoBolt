"""Per-construction preparation for the unchanged source-triangle predicates.

This is only for three original projected vertices. Clipped pixel polygons keep
the general predicates in pixel_clip. A bounded axis-aligned edge can provide
the same separating witness directly; all other cases keep the original axes,
reductions and exact orientations. Preparation stays local to one construction.
"""
from __future__ import annotations

import math
import numpy as np

from femto.geometry.pixel_clip import EPS, _orient_sign, interior_overlap

TINY = np.finfo(float).tiny


def _prepared_sign(start, end, point, original_start, original_end, original_point):
    """Same binary64 orientation filter; keep the original exact fallback.

    Python float operations perform the same separate subtract/multiply/add
    operations here, without allocating two tiny NumPy difference arrays.
    No fused expression, tolerance or determinant ordering is introduced.
    """
    if original_start.dtype != np.float64 or original_end.dtype != np.float64 or original_point.dtype != np.float64:
        return _orient_sign(original_start, original_end, original_point)
    x0, x1 = end[0]-start[0], end[1]-start[1]
    y0, y1 = point[0]-start[0], point[1]-start[1]
    left, right = x0*y1, x1*y0
    value = left-right
    bound = EPS*(abs(left)+abs(right))
    if abs(value) > bound:
        return 1 if value > 0 else -1
    return _orient_sign(original_start, original_end, original_point)


def _three_extrema(values):
    """Same three-value reductions, with overflow/NaN and zero ties retained."""
    a,b,c = float(values[0]),float(values[1]),float(values[2])
    if (math.isfinite(a) and math.isfinite(b) and math.isfinite(c)
            and a != 0. and b != 0. and c != 0.):
        return max(a,b,c),min(a,b,c),max(abs(a),abs(b),abs(c))
    return values.max(),values.min(),float(np.abs(values).max())


def _finite_signs(start: np.ndarray, end: np.ndarray, points: np.ndarray) -> set[int]:
    """Exact endpoint identities need no rational determinant for finite inputs.

    This shortcut is used only after both prepared polygons passed the finite
    check. Every other point keeps the original floating filter/exact fallback.
    """
    return {0 if ((point[0] == start[0] and point[1] == start[1])
                  or (point[0] == end[0] and point[1] == end[1]))
            else _orient_sign(start, end, point) for point in points}


class PreparedTriangle:
    """Lazy, bounded-by-owner preparation of immutable source coordinates."""

    __slots__ = ('points', 'vertex_ids', 'finite', '_edges', '_orientations', '_coordinates', '_axis_data')

    def __init__(self, points: np.ndarray, vertex_ids: np.ndarray):
        points.setflags(write=False)
        self.points = points
        self.vertex_ids = tuple(int(value) for value in vertex_ids)
        self.finite = bool(np.isfinite(points).all())
        self._coordinates = tuple(tuple(float(x) for x in point) for point in points)
        self._edges = {}
        self._orientations = {}
        self._axis_data = None
        # An exactly horizontal/vertical source edge is already one of the
        # original SAT axes. Coordinate ordering can establish the same
        # separating sign sets, including exact contact, without constructing
        # projections for every neighbouring face. Limit coordinate magnitude
        # so the original differences, products and gap cannot overflow.
        if self.finite and points.dtype == np.float64 and all(
                abs(x) <= 1e150 for point in self._coordinates for x in point):
            bounds = tuple((min(p[d] for p in self._coordinates),
                            max(p[d] for p in self._coordinates)) for d in range(2))
            boundaries = []
            for index, start in enumerate(self._coordinates):
                end = self._coordinates[(index+1) % 3]
                for dimension in range(2):
                    if start[dimension] == end[dimension] and start[1-dimension] != end[1-dimension]:
                        value = start[dimension]
                        low, high = bounds[dimension]
                        boundaries.append((dimension, value, low >= value, high <= value))
            self._axis_data = bounds, tuple(boundaries)

    def orientation(self, start: int, end: int, point: int) -> int:
        if self.finite and point in (start, end):
            return 0
        key = start, end, point
        if key not in self._orientations:
            self._orientations[key] = _prepared_sign(
                self._coordinates[start], self._coordinates[end], self._coordinates[point],
                self.points[start], self.points[end], self.points[point])
        return self._orientations[key]

    def edge(self, index: int):
        if index not in self._edges:
            start, end = self.points[index], self.points[(index+1) % 3]
            delta = end-start
            if not delta.any():
                self._edges[index] = None
            else:
                axis = np.array([-delta[1], delta[0]])
                own = (self.points-start) @ axis
                self._edges[index] = (start, end, axis, own.max(), own.min(),
                                      float(np.abs(own).max()))
        return self._edges[index]

    def signs(self, index: int) -> set[int]:
        return {self.orientation(index, (index+1) % 3, point) for point in range(3)}

    def other_signs(self, index: int, other) -> set[int]:
        start, end = self._coordinates[index], self._coordinates[(index+1) % 3]
        return {0 if point == start or point == end else _prepared_sign(
            start, end, point, self.points[index], self.points[(index+1) % 3], other.points[i])
            for i, point in enumerate(other._coordinates)}


def shared_edge_separates(a: PreparedTriangle, b: PreparedTriangle) -> bool:
    """Coordinate identity IDs replace exactly the original nine array_equal calls."""
    shared = [(i, j) for i in range(3) for j in range(3)
              if a.vertex_ids[i] == b.vertex_ids[j]]
    if len(shared) != 2:
        return False
    first, second = shared[0][0], shared[1][0]
    left = next(i for i in range(3) if i not in [s[0] for s in shared])
    right = next(j for j in range(3) if j not in [s[1] for s in shared])
    sa = a.orientation(first, second, left)
    sb = _prepared_sign(a._coordinates[first], a._coordinates[second], b._coordinates[right],
                        a.points[first], a.points[second], b.points[right])
    return sa*sb < 0


def prepared_overlap(a: PreparedTriangle, b: PreparedTriangle) -> bool | None:
    """Same overlap decision, with a cheap witness on an existing source axis."""
    if not a.finite or not b.finite:
        return interior_overlap(a.points, b.points)
    if a._axis_data is not None and b._axis_data is not None:
        for owner, other in ((a, b), (b, a)):
            bounds, _ = other._axis_data
            for dimension, value, positive, negative in owner._axis_data[1]:
                low, high = bounds[dimension]
                if (positive and high <= value) or (negative and low >= value):
                    return False
    uncertain = False
    for owner, other in ((a, b), (b, a)):
        for index in range(3):
            edge = owner.edge(index)
            if edge is None:
                continue
            start, end, axis, own_max, own_min, own_abs = edge
            opposite = (other.points-start) @ axis
            other_max, other_min, other_abs = _three_extrema(opposite)
            # Preserve pa/pb ordering even when finite inputs overflow. Python's
            # min/max ordering with NaNs must remain the original ordering.
            if owner is a:
                amax, amin, aabs = own_max, own_min, own_abs
                bmax, bmin, babs = other_max, other_min, other_abs
            else:
                amax, amin, aabs = other_max, other_min, other_abs
                bmax, bmin, babs = own_max, own_min, own_abs
            gap = min(amax, bmax) - max(amin, bmin)
            scale = max(aabs, babs, TINY)
            if gap < -EPS*scale:
                return False
            if abs(gap) <= EPS*scale:
                if owner is a:
                    signs_a = owner.signs(index)
                    signs_b = owner.other_signs(index, b)
                else:
                    signs_a = owner.other_signs(index, a)
                    signs_b = owner.signs(index)
                if (signs_a <= {0, 1} and signs_b <= {0, -1}) or (signs_a <= {0, -1} and signs_b <= {0, 1}):
                    return False
                uncertain = True
    return None if uncertain else True
