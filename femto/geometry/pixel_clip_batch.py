"""Bounded batches of the existing ordered triangle/halfspace clipping.

Every row follows the scalar clipper's four halfspaces and polygon edge order.
Near-zero signs and non-finite intermediates use that original clipper. This is
an execution strategy only: no vertices, thresholds or exact cases are omitted.
"""
from __future__ import annotations

import numpy as np

from femto.geometry.pixel_clip import EPS, clip_triangle_planes, project


MAX_BATCH = 64


def project_clipped_polygons(clipped, triangles, intrinsic):
    """Project one bounded clip batch, grouped by the original polygon size.

    Every row keeps its matrix-product and reduction lengths. This prepares
    geometry only; callers still visit/check/charge faces in their old order.
    """
    if len(clipped) > MAX_BATCH or len(triangles) != len(clipped):
        raise ValueError('Polygon projection batch exceeds its bounded workspace')
    result = [None] * len(clipped)
    groups = {}
    for index, (barycentric, _) in enumerate(clipped):
        if len(barycentric) >= 3:
            groups.setdefault(len(barycentric), []).append(index)
    for size, ids in groups.items():
        barycentric = np.stack([clipped[index][0] for index in ids])
        xyz = np.matmul(barycentric, triangles[ids])
        projected = project(xyz, intrinsic)
        shifted = projected-projected[:, :1]
        area = np.abs((shifted[:, :-1, 0]*shifted[:, 1:, 1]
                       -shifted[:, :-1, 1]*shifted[:, 1:, 0]).sum(axis=1))/2
        edges = np.empty_like(projected)
        edges[:, :-1] = projected[:, 1:]-projected[:, :-1]
        edges[:, -1] = projected[:, 0]-projected[:, -1]
        error = EPS*np.maximum(1., np.abs(projected).max(axis=(1, 2)))*np.abs(edges).sum(axis=(1, 2))
        finite = np.isfinite(xyz).all(axis=(1, 2)) & np.isfinite(projected).all(axis=(1, 2))
        for row, index in enumerate(ids):
            result[index] = xyz[row], projected[row], float(area[row]), float(error[row]), bool(finite[row])
    return result


def clip_triangles_planes(triangles: np.ndarray, planes: np.ndarray, *,
                          scalar_clip=clip_triangle_planes) -> list[tuple[np.ndarray, bool]]:
    """Clip at most 64 triangles against shared or per-row pixel halfspaces.

    Matrix-vector products keep their 3-element dot order. Grouping polygons by
    their actual size also retains each original bound reduction, rather than
    padding the sum. Suspicious rows restart in the original scalar function.
    """
    triangles = np.asarray(triangles)
    count = len(triangles)
    if count > MAX_BATCH:
        raise ValueError('Pixel clipping batch exceeds its bounded workspace')
    if not count:
        return []
    planes = np.asarray(planes)
    if triangles.shape != (count, 3, 3) or planes.shape not in ((4, 3), (count, 4, 3)):
        raise ValueError('Expected triangle rows and four pixel halfspaces')
    all_planes = np.broadcast_to(planes, (count, 4, 3))
    if count < 8 or triangles.dtype != np.float64 or not np.isfinite(triangles).all():
        return [scalar_clip(triangle, row_planes) for triangle, row_planes in zip(triangles, all_planes)]
    # A triangle clipped by four halfspaces has at most seven vertices.
    points = np.zeros((count, 7, 3), dtype=np.float64)
    points[:, :3] = np.eye(3)
    sizes = np.full(count, 3, dtype=np.int8)
    cut = np.zeros(count, dtype=bool)
    pending = np.ones(count, dtype=bool)
    scalar = np.zeros(count, dtype=bool)
    for halfspace in range(4):
        ids = np.flatnonzero(pending)
        if not len(ids):
            break
        source = triangles[ids]
        plane = all_planes[ids, halfspace]
        distances = np.matmul(source, plane[:, :, None])[:, :, 0]
        errors = EPS * np.abs(source * plane[:, None, :]).sum(axis=2)
        signs = np.zeros((len(ids), 7), dtype=np.float64)
        bounds = np.zeros_like(signs)
        initial = ~cut[ids]
        signs[initial, :3] = distances[initial]
        bounds[initial, :3] = errors[initial] + EPS * np.abs(distances[initial])
        for size in range(3, 8):
            selected = np.flatnonzero(~initial & (sizes[ids] == size))
            if not len(selected):
                continue
            polygon = points[ids[selected], :size]
            values = distances[selected, :, None]
            signs[selected, :size] = np.matmul(polygon, values)[:, :, 0]
            bounds[selected, :size] = (np.matmul(polygon, errors[selected, :, None])[:, :, 0]
                + EPS * np.matmul(np.abs(polygon), np.abs(values))[:, :, 0])
        present = np.arange(7)[None, :] < sizes[ids, None]
        deferred = ((~np.isfinite(signs) | ~np.isfinite(bounds)
                     | (np.abs(signs) <= bounds)) & present).any(axis=1)
        scalar[ids[deferred]] = True
        pending[ids[deferred]] = False
        positive = ((signs > 0) & present).sum(axis=1)
        empty = ~deferred & (positive == 0)
        sizes[ids[empty]] = 0
        pending[ids[empty]] = False
        selected = np.flatnonzero(~deferred & (positive > 0) & (positive < sizes[ids]))
        if not len(selected):
            continue
        rows = ids[selected]
        polygon = points[rows]
        sign = signs[selected]
        valid = present[selected]
        next_index = (np.arange(7)[None, :] + 1) % sizes[rows, None]
        following = np.take_along_axis(polygon, next_index[:, :, None], axis=1)
        next_sign = np.take_along_axis(sign, next_index, axis=1)
        retained = (sign > 0) & valid
        crossing = ((sign < 0) != (next_sign < 0)) & valid
        positions = np.cumsum(retained.astype(np.int8) + crossing, axis=1)
        # Convex clipping has at most seven vertices mathematically. If finite
        # arithmetic ever violates that bound, retain the scalar behavior before
        # indexing the fixed workspace; numerical degeneracy is not a new error.
        overflow = positions[:, -1] > points.shape[1]
        if np.any(overflow):
            scalar[rows[overflow]] = True
            pending[rows[overflow]] = False
            keep = ~overflow
            rows, polygon, following = rows[keep], polygon[keep], following[keep]
            sign, next_sign = sign[keep], next_sign[keep]
            retained, crossing, positions = retained[keep], crossing[keep], positions[keep]
            if not len(rows):
                continue
        offsets = positions - retained - crossing
        clipped = np.zeros_like(polygon)
        row, edge = np.nonzero(retained)
        clipped[row, offsets[row, edge]] = polygon[row, edge]
        row, edge = np.nonzero(crossing)
        start, end = polygon[row, edge], following[row, edge]
        da, db = sign[row, edge], next_sign[row, edge]
        clipped[row, offsets[row, edge] + retained[row, edge]] = (
            start + (da / (da - db))[:, None] * (end - start))
        points[rows] = clipped
        sizes[rows] = positions[:, -1]
        cut[rows] = True
        short = rows[sizes[rows] < 3]
        sizes[short] = 0
        pending[short] = False
    return [(scalar_clip(triangles[i], all_planes[i]) if scalar[i]
             else (points[i, :sizes[i]].copy(), False)) for i in range(count)]
