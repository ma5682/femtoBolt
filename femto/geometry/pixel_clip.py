"""Clip canonical triangles by reference-pixel ray halfspaces.

Coordinates/areas are nominal float64 mesh quantities, not certified bounds.
Near-zero clipping signs fall back to rational arithmetic over the binary input
values. Unresolved overlap is explicit; sampling is never a visibility proof.
"""
from __future__ import annotations

from fractions import Fraction
import math

import numpy as np

EPS = np.finfo(float).eps * 64


def pixel_owner(uv: np.ndarray) -> np.ndarray:
    return np.floor(np.asarray(uv, float) + .5).astype(np.int64)


def project(points: np.ndarray, k: np.ndarray) -> np.ndarray:
    return points[..., :2] / points[..., 2:3] * [k[0, 0], k[1, 1]] + k[:2, 2]


def area2(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.
    shifted = poly - poly[0]
    return abs(float((shifted[:-1, 0]*shifted[1:, 1] - shifted[:-1, 1]*shifted[1:, 0]).sum())) / 2


def _rational_clip(triangle: np.ndarray, planes: np.ndarray) -> np.ndarray:
    points = [[Fraction(int(i == j)) for j in range(3)] for i in range(3)]
    tri = [[Fraction(float(x)) for x in row] for row in triangle]
    for plane in planes:
        normal = [Fraction(float(x)) for x in plane]
        distances = [sum(x*n for x, n in zip(vertex, normal)) for vertex in tri]
        output = []
        for a, b in zip(points, points[1:] + points[:1]):
            da, db = (sum(w*d for w, d in zip(point, distances)) for point in (a, b))
            if da >= 0:
                output.append(a)
            if (da < 0) != (db < 0):
                t = da / (da-db)
                output.append([x + t*(y-x) for x, y in zip(a, b)])
        points = []
        for point in output:
            if not points or point != points[-1]:
                points.append(point)
        if len(points) > 1 and points[0] == points[-1]:
            points.pop()
        if len(points) < 3:
            return np.empty((0, 3), float)
    return np.asarray(points, float)


def pixel_planes(k: np.ndarray, u: int, v: int) -> np.ndarray:
    """The four original ray halfspaces, reusable for one reference pixel."""
    return np.array([[k[0, 0], 0, k[0, 2]-(u-.5)],
                     [-k[0, 0], 0, u+.5-k[0, 2]],
                     [0, k[1, 1], k[1, 2]-(v-.5)],
                     [0, -k[1, 1], v+.5-k[1, 2]]], float)


def clip_triangle(triangle: np.ndarray, k: np.ndarray, u: int, v: int) -> tuple[np.ndarray, bool]:
    """Barycentric polygon on the source face; half-open ownership is by pixel_owner.

    Closed polygon boundaries are stored to keep mesh edges representable. Their
    measure is zero; queries on a shared pixel edge belong to the higher pixel.
    """
    return clip_triangle_planes(triangle, pixel_planes(k, u, v))


def clip_triangle_planes(triangle: np.ndarray, planes: np.ndarray) -> tuple[np.ndarray, bool]:
    """Clip with prepared planes; retain rounding checks and rational fallback."""
    # Until the first cut the polygon is exactly the three source vertices.
    # Multiplying by the identity repeats the same work for every pixel/face.
    points = None
    for plane in planes:
        distances = triangle @ plane
        # These are ordinary ndarrays; call their identical reductions directly
        # to avoid NumPy's generic array-dispatch wrapper for each tiny polygon.
        errors = EPS * np.abs(triangle * plane).sum(axis=1)
        if points is None:
            signs = distances
            bounds = errors + EPS * np.abs(distances)
            count = 3
            # Finite inputs can still overflow the dot products/error sums.
            # The identity matmul then propagates non-finite values differently;
            # retain the original path rather than changing its unknown result.
            if not np.isfinite(bounds).all():
                points = np.eye(3)
                signs = points @ distances
                bounds = points @ errors + EPS * (np.abs(points) @ np.abs(distances))
        else:
            signs = points @ distances
            bounds = points @ errors + EPS * (np.abs(points) @ np.abs(distances))
            count = len(points)
        if (np.abs(signs) <= bounds).any():
            return _rational_clip(triangle, planes), True
        positive = np.count_nonzero(signs > 0)
        if positive == count:
            continue
        if positive == 0:
            return np.empty((0, 3), float), False
        if points is None:
            points = np.eye(3)
        out = []
        for index, a in enumerate(points):
            b = points[(index+1) % len(points)]
            da, db = signs[index], signs[(index+1) % len(points)]
            if da > 0:
                out.append(a)
            if (da < 0) != (db < 0):
                out.append(a + (da/(da-db))*(b-a))
        if len(out) < 3:
            return np.empty((0, 3), float), False
        points = np.asarray(out)
    return np.eye(3) if points is None else points, False


def _orient_sign(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> int:
    x, y = b-a, c-a
    value = x[0]*y[1]-x[1]*y[0]
    bound = EPS*(abs(x[0]*y[1])+abs(x[1]*y[0]))
    if abs(value) > bound:
        return 1 if value > 0 else -1
    aa, bb, cc = ([Fraction(float(v)) for v in point] for point in (a, b, c))
    exact = (bb[0]-aa[0])*(cc[1]-aa[1]) - (bb[1]-aa[1])*(cc[0]-aa[0])
    return (exact > 0) - (exact < 0)


def shared_edge_separates(a: np.ndarray, b: np.ndarray, k: np.ndarray) -> bool:
    shared = [(i, j) for i in range(3) for j in range(3) if np.array_equal(a[i], b[j])]
    if len(shared) != 2:
        return False
    edge = project(a[[shared[0][0], shared[1][0]]], k)
    left = next(i for i in range(3) if i not in [s[0] for s in shared])
    right = next(j for j in range(3) if j not in [s[1] for s in shared])
    sa = _orient_sign(*edge, project(a[[left]], k)[0])
    sb = _orient_sign(*edge, project(b[[right]], k)[0])
    return sa*sb < 0


def interior_overlap(a: np.ndarray, b: np.ndarray) -> bool | None:
    """Separating-axis filter; near-touching numerical uncertainty is not overlap-free."""
    uncertain = False
    for poly in (a, b):
        for start, end in zip(poly, np.roll(poly, -1, axis=0)):
            edge = end-start
            if not np.any(edge):
                continue
            axis = np.array([-edge[1], edge[0]])
            pa, pb = (a-start) @ axis, (b-start) @ axis
            gap = min(pa.max(), pb.max()) - max(pa.min(), pb.min())
            scale = max(float(np.max(np.abs(pa))), float(np.max(np.abs(pb))), np.finfo(float).tiny)
            if gap < -EPS*scale:
                return False
            if abs(gap) <= EPS*scale:
                # Exact predicates can establish a shared boundary without
                # widening it. Never apply a positive overlap tolerance.
                signs_a = {_orient_sign(start, end, p) for p in a}
                signs_b = {_orient_sign(start, end, p) for p in b}
                if (signs_a <= {0, 1} and signs_b <= {0, -1}) or (signs_a <= {0, -1} and signs_b <= {0, 1}):
                    return False
                uncertain = True
    return None if uncertain else True


def covers_in_front(front_uv: np.ndarray, front_tri: np.ndarray,
                    back_uv: np.ndarray, back_xyz: np.ndarray, k: np.ndarray) -> bool:
    """Only discard a back polygon if one nearer polygon covers it entirely."""
    for a, b in zip(front_uv, np.roll(front_uv, -1, axis=0)):
        reference = {_orient_sign(a, b, p) for p in front_uv} - {0}
        if len(reference) != 1:
            return False
        inside = next(iter(reference))
        if any(_orient_sign(a, b, p) not in (0, inside) for p in back_uv):
            return False
    normal = np.cross(front_tri[1]-front_tri[0], front_tri[2]-front_tri[0])
    rays = np.c_[(back_uv-k[:2, 2]) / [k[0, 0], k[1, 1]], np.ones(len(back_uv))]
    denominator = rays @ normal
    if np.any(np.abs(denominator) <= EPS*np.sum(np.abs(rays*normal), axis=1)):
        return False
    depth = math.fsum(float(x*y) for x, y in zip(normal, front_tri[0])) / denominator
    delta = back_xyz[:, 2] - depth
    return bool(np.all(depth > 0) and np.all(delta > EPS*(np.abs(back_xyz[:, 2])+np.abs(depth))))
