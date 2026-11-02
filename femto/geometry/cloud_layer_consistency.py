"""Bounded batch queries for the existing point-cloud layer diagnostic.

This is an internal geometry-consistency statistic, not physical accuracy.
The legacy predicate remains authoritative at radius/height boundaries and
whenever a bounded candidate list cannot prove that it contains every neighbor.
"""
from __future__ import annotations

import numpy as np
import open3d as o3d


_RADIUS = .01
_HEIGHT = .004
_BATCH_SIZE = 512
_MAX_NEIGHBORS = 64
# Three squared components/two additions (distance), or three products/two
# additions (normal projection), need only a small multiple of double epsilon.
# Use 64 epsilon of the sum of absolute terms to cover both native and NumPy
# evaluation orders, cancellation, and the comparison arithmetic. Borderline
# results are not decided by this error allowance: they use the old predicate.
_ROUNDING_FACTOR = 64.0 * np.finfo(np.float64).eps
_RADIUS_SQUARED = _RADIUS * _RADIUS
_RADIUS_GUARD = _ROUNDING_FACTOR * _RADIUS_SQUARED
# Widen the candidate query before any classification. Merely examining the
# returned neighbors at the old radius could miss a boundary neighbor omitted
# by a different native distance implementation.
_CANDIDATE_RADIUS = max(
    np.nextafter(np.sqrt(_RADIUS_SQUARED + 2.0 * _RADIUS_GUARD), np.inf),
    # Also tolerate native bindings that round the scalar radius to Float32;
    # point coordinates and the acceptance calculation remain Float64. This
    # expands only the candidate list, never the original 10 mm neighborhood.
    float(np.nextafter(np.float32(_RADIUS), np.float32(np.inf))))


def count_second_sheet_points(cloud) -> int:
    """Count points with an original-radius neighbor over 4 mm along its normal.

    Float64 native queries are bounded by 512 x 64 neighbors. Full candidate
    rows fall back to the unbounded *single-point* legacy query, so dense clouds
    cannot silently lose neighbors or allocate an all-pairs neighbor table.
    Only candidate discovery and non-boundary predicates are batched; the final
    count, strict thresholds, and all input points/normals remain unchanged.
    """
    points = np.asarray(cloud.points)
    normals = np.asarray(cloud.normals)
    tree = None

    def legacy(indices):
        nonlocal tree
        if tree is None:
            tree = o3d.geometry.KDTreeFlann(cloud)
        count_second = 0
        for index in indices:
            point = points[index]
            count, ids, _ = tree.search_radius_vector_3d(point, _RADIUS)
            if count > 1 and np.any(
                    np.abs((points[np.asarray(ids)] - point) @ normals[index]) > _HEIGHT):
                count_second += 1
        return count_second

    # Preserve the original error/NaN behavior on malformed data. The ordinary
    # path is explicitly finite and keeps native distance arithmetic in range.
    # The broad 2**20-m coordinate bound avoids extreme-coordinate backend
    # behavior; outside it the unchanged legacy path still processes the data.
    if (len(points) < 2 or normals.shape != points.shape
            or not np.isfinite(points).all() or not np.isfinite(normals).all()
            or np.abs(points).max(initial=0.0) > 2.0 ** 20):
        return legacy(range(len(points)))

    try:
        tensor = o3d.core.Tensor(points, dtype=o3d.core.Dtype.Float64)
        search = o3d.core.nns.NearestNeighborSearch(
            tensor, index_dtype=o3d.core.Dtype.Int64)
        if not search.hybrid_index():
            return legacy(range(len(points)))
    except (RuntimeError, TypeError, AttributeError):
        # Some Open3D builds lack a compatible Float64 CPU index.
        return legacy(range(len(points)))

    second = 0
    for start in range(0, len(points), _BATCH_SIZE):
        end = min(start + _BATCH_SIZE, len(points))
        try:
            indices, _, counts = search.hybrid_search(
                tensor[start:end], radius=_CANDIDATE_RADIUS,
                max_knn=_MAX_NEIGHBORS)
        except (RuntimeError, TypeError):
            return second + legacy(range(start, len(points)))
        ids = indices.numpy()
        counts = counts.numpy()
        valid = np.arange(_MAX_NEIGHBORS)[None, :] < counts[:, None]
        delta = points[np.maximum(ids, 0)] - points[start:end, None, :]
        squared = ((delta[..., 0] * delta[..., 0]
                    + delta[..., 1] * delta[..., 1])
                   + delta[..., 2] * delta[..., 2])
        radius_boundary = valid & (np.abs(squared - _RADIUS_SQUARED) <= _RADIUS_GUARD)
        in_radius = valid & (squared < _RADIUS_SQUARED)
        with np.errstate(over='ignore', invalid='ignore'):
            products = delta * normals[start:end, None, :]
            heights = np.abs((products[..., 0] + products[..., 1]) + products[..., 2])
            height_guard = _ROUNDING_FACTOR * np.maximum(
                np.abs(products).sum(axis=2), _HEIGHT)
        height_boundary = in_radius & (
            (np.abs(heights - _HEIGHT) <= height_guard)
            | ~np.isfinite(heights) | ~np.isfinite(height_guard))
        fallback = ((counts >= _MAX_NEIGHBORS)
                    | radius_boundary.any(axis=1) | height_boundary.any(axis=1))
        accepted = ((in_radius.sum(axis=1) > 1)
                    & (in_radius & (heights > _HEIGHT)).any(axis=1))
        second += int(np.count_nonzero(accepted & ~fallback))
        if fallback.any():
            second += legacy(start + np.flatnonzero(fallback))
    return second
