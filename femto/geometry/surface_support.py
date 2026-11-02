"""Constructed points and exact segment support in a binary-input triangle mesh.

Display XYZ is a rounded derivative of a face/edge construction. It is never
promoted back to a coplanarity proof. Positive gaps are not closed by epsilon.
This module proves numeric support only, not acquisition or physical accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
import math
from typing import Callable, Iterable

import numpy as np

from femto.geometry.exact_surface import rational, xyz, dot, subtract, bounds

SCHEMA = 'femto-constructed-surface-point-v1'
METHOD = 'exact-constructed-chord-intervals-v1'
MAX_BITS = 4096


class NumericalLimit(ValueError):
    """The bounded representation cannot establish a result."""


def fraction(value) -> Q:
    if (not isinstance(value, list) or len(value) != 2
            or any(type(v) is not int for v in value) or value[1] <= 0
            or max(abs(v).bit_length() for v in value) > MAX_BITS):
        raise NumericalLimit('invalid_or_unbounded_rational')
    result = Q(*value)
    if [result.numerator, result.denominator] != value:
        raise NumericalLimit('noncanonical_rational')
    return result


def encoded(value: Q) -> list[int]:
    if max(abs(value.numerator).bit_length(), value.denominator.bit_length()) > MAX_BITS:
        raise NumericalLimit('rational_bit_budget')
    return [value.numerator, value.denominator]


def cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def barycentric(point, triangle):
    a, b, c = triangle
    u, v, p = subtract(b, a), subtract(c, a), subtract(point, a)
    aa, ab, bb = dot(u, u), dot(u, v), dot(v, v)
    determinant = aa*bb-ab*ab
    if determinant == 0:return None
    pu, pv = dot(p, u), dot(p, v)
    s, t = (pu*bb-pv*ab)/determinant, (pv*aa-pu*ab)/determinant
    return (1-s-t, s, t)


@dataclass(frozen=True)
class Decision:
    status: str
    reason: str
    intervals: tuple = ()
    gaps: tuple = ()
    candidates: int = 0
    exact_faces: int = 0
    isolated_contacts: int = 0
    degenerate_faces: int = 0
    construction_fast_path: bool = False

    def to_dict(self):
        return dict(status=self.status, reason=self.reason,
            intervals=[[encoded(a), encoded(b), int(face)] for a, b, face in self.intervals],
            gaps=[[encoded(a), encoded(b)] for a, b in self.gaps],
            candidates=self.candidates, exact_faces=self.exact_faces,
            isolated_contacts=self.isolated_contacts, degenerate_faces=self.degenerate_faces,
            construction_fast_path=self.construction_fast_path)


class MeshSupport:
    """Lazy exact triangles with conservative, outward-bounded AABB rejection."""
    def __init__(self, vertices, faces, *, allowed=None, groups=None, max_face_checks=200_000,
                 max_filter_comparisons=20_000_000, check: Callable=lambda: None):
        self.vertices = np.asarray(vertices, float)
        self.faces = np.asarray(faces)
        if (self.vertices.ndim != 2 or self.vertices.shape[1] != 3 or not np.isfinite(self.vertices).all()
                or self.faces.ndim != 2 or self.faces.shape[1] != 3 or self.faces.dtype.kind not in 'iu'
                or len(self.faces) > 4_000_000
                or self.faces.size and (self.faces.min() < 0 or self.faces.max() >= len(self.vertices))):
            raise ValueError('invalid_support_mesh')
        if type(max_face_checks) is not int or not 0 <= max_face_checks <= 200_000:
            raise ValueError('invalid_support_budget')
        if type(max_filter_comparisons) is not int or not 0 <= max_filter_comparisons <= 20_000_000:
            raise ValueError('invalid_support_filter_budget')
        self.allowed = np.ones(len(self.faces), bool) if allowed is None else np.asarray(allowed, bool)
        if self.allowed.shape != (len(self.faces),):raise ValueError('invalid_support_domain')
        self.groups = None if groups is None else np.asarray(groups)
        if self.groups is not None and (self.groups.shape != (len(self.faces),) or self.groups.dtype.kind not in 'iu'):
            raise ValueError('invalid_support_groups')
        self.max_face_checks, self.check = max_face_checks, check
        self.max_filter_comparisons = max_filter_comparisons
        self.face_checks = self.filter_comparisons = self.fast_paths = 0
        self.queries = self.interval_queries = self.aabb_rejected_faces = 0
        self._triangles = {}
        triangles = self.vertices[self.faces]
        self.low, self.high = triangles.min(axis=1), triangles.max(axis=1)

    def triangle(self, face: int):
        if type(face) is not int or not 0 <= face < len(self.faces):raise ValueError('invalid_support_face')
        if face not in self._triangles:
            self.check()
            self._triangles[face] = tuple(rational(p) for p in self.vertices[self.faces[face]])
        return self._triangles[face]

    def point(self, face: int, weights: Iterable[Q]) -> dict:
        triangle = self.triangle(face)
        weights = tuple(weights)
        if len(weights) != 3 or sum(weights) != 1 or min(weights) < 0:
            raise ValueError('point_outside_support_face')
        if cross(subtract(triangle[1], triangle[0]), subtract(triangle[2], triangle[0])) == (0, 0, 0):
            raise ValueError('degenerate_support_face')
        record = dict(schema=SCHEMA, face=face, barycentric=[encoded(v) for v in weights])
        zeros = [i for i, value in enumerate(weights) if value == 0]
        if len(zeros) >= 1:
            nonzero = [i for i, value in enumerate(weights) if value != 0]
            if len(nonzero) == 1:
                record['vertex'] = int(self.faces[face, nonzero[0]])
            else:
                ids = sorted(int(self.faces[face, i]) for i in nonzero)
                record['edge'] = ids
                record['edge_parameter'] = encoded(weights[list(self.faces[face]).index(ids[1])])
        return record

    def resolve(self, record: dict):
        if not isinstance(record, dict) or record.get('schema') != SCHEMA:
            raise ValueError('missing_constructed_point')
        face = record.get('face')
        if type(face) is not int:raise ValueError('invalid_support_face')
        if 'vertex' in record and type(record['vertex']) is not int:raise ValueError('invalid_support_vertex')
        if 'edge' in record and (not isinstance(record['edge'],list) or len(record['edge'])!=2
                                or any(type(index) is not int for index in record['edge'])):
            raise ValueError('invalid_support_edge')
        if 'edge_parameter' in record:fraction(record['edge_parameter'])
        values = record.get('barycentric')
        if not isinstance(values, list) or len(values) != 3:raise ValueError('invalid_barycentric_record')
        weights = tuple(fraction(v) for v in values)
        canonical = self.point(face, weights)
        if record != canonical:raise ValueError('inconsistent_point_construction')
        return xyz(weights, self.triangle(face))

    def closest(self, point, face: int) -> dict:
        """Create a NEW closest-point candidate, with no support claim for input XYZ."""
        point, tri = rational(point), self.triangle(face)
        bary = barycentric(point, tri)
        if bary is None:raise ValueError('degenerate_support_face')
        if min(bary) >= 0:return self.point(face, bary)
        choices = []
        for i, j in ((0, 1), (1, 2), (2, 0)):
            direction = subtract(tri[j], tri[i]);length2 = dot(direction, direction)
            if not length2:continue
            t = min(Q(1), max(Q(0), dot(subtract(point, tri[i]), direction)/length2))
            weights = [Q(0)]*3;weights[i], weights[j] = 1-t, t
            delta = subtract(xyz(weights, tri), point)
            choices.append((dot(delta, delta), i, weights))
        return self.point(face, min(choices, key=lambda row: (row[0], row[1]))[2])

    def on_face(self, point, face: int):
        tri = self.triangle(face)
        normal = cross(subtract(tri[1], tri[0]), subtract(tri[2], tri[0]))
        if normal == (0, 0, 0) or dot(subtract(point, tri[0]), normal):return None
        weights = barycentric(point, tri)
        return self.point(face, weights) if min(weights) >= 0 else None

    def interpolate(self, left, right, t: Q, face: int):
        a, b = self.resolve(left), self.resolve(right)
        record = self.on_face(tuple(x+t*(y-x) for x,y in zip(a,b)), face)
        if record is None:raise ValueError('interpolation_has_no_constructed_face')
        return record

    def display(self, record):
        point = self.resolve(record)
        near = [float(v) for v in point]
        if not all(math.isfinite(v) for v in near):raise NumericalLimit('unrepresentable_display_point')
        # L1 is a rational upper bound on the 3D displacement of rounded XYZ.
        error = sum(abs(v-Q(n)) for v, n in zip(point, near))
        upper = float(error)
        if Q(upper) < error:upper = math.nextafter(upper, math.inf)
        return near, upper

    def _reserve(self):
        self.check()
        if self.face_checks >= self.max_face_checks:raise NumericalLimit('support_face_budget')
        self.face_checks += 1

    def _interval(self, a, b, face):
        self._reserve()
        tri = self.triangle(face)
        normal = cross(subtract(tri[1], tri[0]), subtract(tri[2], tri[0]))
        if normal == (0, 0, 0):return 'degenerate', None
        da, db = dot(subtract(a, tri[0]), normal), dot(subtract(b, tri[0]), normal)
        if da or db:
            if da == db:return 'outside', None
            t = da/(da-db)
            if not 0 <= t <= 1:return 'outside', None
            point = tuple(x+t*(y-x) for x, y in zip(a, b))
            weights = barycentric(point, tri)
            if min(weights) < 0:return 'outside', None
            return 'isolated', (t, t)
        wa, wb = barycentric(a, tri), barycentric(b, tri)
        lo, hi = Q(0), Q(1)
        for start, end in zip(wa, wb):
            delta = end-start
            if delta > 0:lo = max(lo, -start/delta)
            elif delta < 0:hi = min(hi, -start/delta)
            elif start < 0:return 'outside', None
        if lo > hi:return 'outside', None
        encoded(lo);encoded(hi)
        return 'coplanar', (lo, hi)

    def chord(self, left, right, *, candidate_faces=None) -> Decision:
        """Prove a complete parameter interval or retain a bounded unknown result.

        Explicit candidate_faces defines the support domain of the query. By
        default every allowed AABB-intersecting face is considered, not a k-NN
        guess. Isolated noncoplanar contacts cannot fill a positive interval.
        """
        try:
            self.check()
            self.queries += 1
            a, b = self.resolve(left), self.resolve(right)
            allowed = self.allowed
            if self.groups is not None:
                group = self.groups[left['face']]
                if group < 0 or group != self.groups[right['face']]:
                    return Decision('rejected', 'different_work_blocks')
            if candidate_faces is None and left['face'] == right['face'] and allowed[left['face']]:
                self._reserve();self.fast_paths += 1
                return Decision('passed', 'constructed_convex_face', ((Q(0), Q(1), left['face']),),
                                candidates=1, construction_fast_path=True)
            if candidate_faces is None:
                # An exact shared-edge/vertex construction can be represented
                # in either incident face. This is a proof-preserving change of
                # barycentric coordinates, not a second projection or snapping.
                for face in sorted({left['face'], right['face']}):
                    if not allowed[face]:continue
                    self._reserve()
                    if self.on_face(a, face) is not None and self.on_face(b, face) is not None:
                        self.fast_paths += 1
                        return Decision('passed', 'constructed_incident_face', ((Q(0), Q(1), face),),
                                        candidates=1, exact_faces=1, construction_fast_path=True)
            low, high = bounds((a, b))
            lower, upper = low.min(axis=0), high.max(axis=0)
            self.interval_queries += 1
            if self.groups is not None:allowed = allowed & (self.groups == group)
            if self.filter_comparisons+len(self.faces) > self.max_filter_comparisons:
                raise NumericalLimit('support_filter_budget')
            self.filter_comparisons += len(self.faces)
            overlap = allowed & np.all(self.high >= lower, axis=1) & np.all(self.low <= upper, axis=1)
            if candidate_faces is not None:
                ids = list(candidate_faces)
                if any(type(i) is not int or not 0 <= i < len(self.faces) for i in ids):
                    raise ValueError('invalid_candidate_face')
                selected = np.zeros(len(self.faces), bool);selected[ids] = True;overlap &= selected
            candidates = np.flatnonzero(overlap)
            self.aabb_rejected_faces += int(np.count_nonzero(allowed))-len(candidates)
            intervals, contacts, degenerates = [], 0, 0
            for face in candidates:
                kind, interval = self._interval(a, b, int(face))
                if kind == 'degenerate':degenerates += 1
                if kind == 'isolated':contacts += 1
                if interval is not None and kind == 'coplanar':
                    # A zero-length geometric segment is supported by a point
                    # in a nondegenerate triangle, represented by all t in [0,1].
                    intervals.append((*interval, int(face)))
            intervals.sort()
            reach, gaps = Q(0), []
            for lo, hi, _ in intervals:
                if lo > reach:gaps.append((reach, lo))
                reach = max(reach, hi)
            if reach < 1:gaps.append((reach, Q(1)))
            if not gaps:
                return Decision('passed', 'exact_interval_union', tuple(intervals), candidates=len(candidates),
                    exact_faces=len(candidates), isolated_contacts=contacts, degenerate_faces=degenerates)
            # Degenerate surfaces have no 2D support semantics in this version.
            # Preserve an unknown when such a candidate could affect the gap.
            return Decision('unconfirmed' if degenerates else 'rejected',
                'degenerate_candidate_face' if degenerates else 'positive_support_gap',
                tuple(intervals), tuple(gaps), len(candidates), len(candidates), contacts, degenerates)
        except NumericalLimit as error:
            return Decision('unconfirmed', str(error))
