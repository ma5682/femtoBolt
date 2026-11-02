"""Exact boundary-clearance predicates with conservative AABB rejection."""
from __future__ import annotations

import numpy as np

from femto.planning.region_coverage import point_segment_distance


MAX_DISTANCE_CHECKS = 100_000_000


def _boundary_tree(boundary, check):
    low, high = boundary.min(axis=1), boundary.max(axis=1)
    centers = (low + high) / 2

    def build(ids):
        check()
        bounds = low[ids].min(axis=0), high[ids].max(axis=0)
        if len(ids) <= 16:
            return *bounds, ids, None, None
        axis = int(np.argmax(np.ptp(centers[ids], axis=0)))
        order = ids[np.argsort(centers[ids, axis], kind='stable')]
        middle = len(order) // 2
        return *bounds, None, build(order[:middle]), build(order[middle:])

    return low, high, centers, build(np.arange(len(boundary)))


def boundary_clearance_mask(mids, half, boundary, radius, check=lambda: None):
    """Use the original distance-minus-half predicate on all nearby segments.

    Far segments are excluded only by a lower bound from their enclosing AABB.
    No segment is sampled or shortened, and the budget counts exact distance
    evaluations rather than the unvisited Cartesian product.
    """
    mids = np.asarray(mids, dtype=float)
    half = np.asarray(half, dtype=float)
    boundary = np.asarray(boundary, dtype=float)
    if not (np.isfinite(mids).all() and np.isfinite(half).all()
            and np.isfinite(boundary).all() and np.isfinite(radius)):
        raise ValueError('边界检查包含非有限坐标，无法验证路线。')
    accepted = np.ones(len(mids), dtype=bool)
    if not len(mids):
        return accepted
    bounds_low, bounds_high, _, root = _boundary_tree(boundary, check)
    # Outward rounding makes pruning conservative at floating-point equality;
    # acceptance itself uses the original unmodified arithmetic below.
    scale = max(1., float(np.abs(mids).max()), float(np.abs(boundary).max()))
    reach = np.nextafter(radius + half + 1e-9, np.inf) + 32 * np.finfo(float).eps * scale
    checked = 0
    stack = [(root, np.arange(len(mids)))]
    while stack:
        check()
        node, indexes = stack.pop()
        indexes = indexes[accepted[indexes]]
        low, high, segments, left, right = node
        delta = np.maximum(np.maximum(low - mids[indexes], mids[indexes] - high), 0.)
        indexes = indexes[np.einsum('ij,ij->i', delta, delta) <= reach[indexes] ** 2]
        if not len(indexes):
            continue
        if segments is None:
            stack.extend(((right, indexes), (left, indexes)))
            continue
        for segment in segments:
            indexes = indexes[accepted[indexes]]
            if not len(indexes):
                break
            # Individual segment boxes avoid unnecessary work inside a leaf.
            delta = np.maximum(np.maximum(bounds_low[segment] - mids[indexes],
                                          mids[indexes] - bounds_high[segment]), 0.)
            candidates = indexes[np.einsum('ij,ij->i', delta, delta) <= reach[indexes] ** 2]
            checked += len(candidates)
            if checked > MAX_DISTANCE_CHECKS:
                raise ValueError('Boundary check budget exceeded; reduce region or increase path spacing')
            a, b = boundary[segment]
            distance = point_segment_distance(mids[candidates], a, b)
            accepted[candidates] &= distance - half[candidates] >= radius + 1e-9
    return accepted


def prepare_nearest_boundary_distances(boundary,check=lambda:None):
    """Bind an immutable boundary index; every query still recomputes exact distances."""
    import open3d as o3d
    boundary=np.array(boundary,dtype=float,copy=True)
    if boundary.ndim!=3 or boundary.shape[1:]!=(2,3) or not len(boundary) or not np.isfinite(boundary).all():
        raise ValueError('边界索引输入无效')
    check();prepared=_boundary_tree(boundary,check)
    seed_tree=o3d.core.nns.NearestNeighborSearch(o3d.core.Tensor(prepared[2]))
    seed_tree.knn_index();boundary.setflags(write=False)
    boundary_scale=float(np.abs(boundary).max())
    def query(points):
        return nearest_boundary_distances(points,boundary,check,_prepared=(prepared,seed_tree,boundary_scale))
    return query


def nearest_boundary_distances(points, boundary, check=lambda: None, *, _prepared=None):
    """Return the exhaustive minimum, using a finite seed only for pruning."""
    import open3d as o3d
    points = np.asarray(points, dtype=float)
    boundary = np.asarray(boundary, dtype=float)
    if not (np.isfinite(points).all() and (_prepared is not None or np.isfinite(boundary).all())):
        raise ValueError('边界检查包含非有限坐标，无法验证路线。')
    nearest = np.full(len(points), np.inf)
    if not len(points) or not len(boundary):
        return nearest
    if _prepared is None:
        lows, highs, centers, root = _boundary_tree(boundary, check)
        seed_tree = o3d.core.nns.NearestNeighborSearch(o3d.core.Tensor(centers))
        seed_tree.knn_index()
    else:
        (lows,highs,centers,root),seed_tree,boundary_scale=_prepared
    if len(points) == 1:
        return _nearest_single(points, boundary, lows, highs, root, seed_tree, check,
                               float(np.abs(boundary).max()) if _prepared is None else boundary_scale)
    checked = 0

    def distances(indexes, segment):
        nonlocal checked
        checked += len(indexes)
        if checked > MAX_DISTANCE_CHECKS:
            raise ValueError('覆盖边界采样检查超过预算，请缩小区域')
        a, b = boundary[segment]
        nearest[indexes] = np.minimum(nearest[indexes], point_segment_distance(points[indexes], a, b))

    # The center lookup is not the distance result: evaluate that entire segment
    # to get an upper bound, then search every box able to improve that bound.
    for first in range(0, len(points), 32768):
        check()
        seeds, _ = seed_tree.knn_search(o3d.core.Tensor(points[first:first + 32768]), 1)
        seeds = seeds.numpy().ravel()
        if len(seeds)==1:
            distances(np.array([first]),int(seeds[0]))
        else:
            order = np.argsort(seeds, kind='stable')
            splits = np.flatnonzero(np.diff(seeds[order])) + 1
            for group in np.split(order, splits):
                distances(first + group, int(seeds[group[0]]))
    scale = max(1., float(np.abs(points).max()),
                float(np.abs(boundary).max()) if _prepared is None else boundary_scale)
    slack = 32 * np.finfo(float).eps * scale

    def candidates(indexes, low, high):
        delta = np.maximum(np.maximum(low - points[indexes], points[indexes] - high), 0.)
        reach = np.nextafter(nearest[indexes], np.inf) + slack
        return indexes[np.einsum('ij,ij->i', delta, delta) <= reach ** 2]

    stack = [(root, np.arange(len(points)))]
    while stack:
        check()
        node, indexes = stack.pop()
        low, high, segments, left, right = node
        indexes = candidates(indexes, low, high)
        if not len(indexes):
            continue
        if segments is None:
            stack.extend(((right, indexes), (left, indexes)))
            continue
        for segment in segments:
            group = candidates(indexes, lows[segment], highs[segment])
            if len(group):distances(group, segment)
    return nearest


def _nearest_single(points, boundary, lows, highs, root, seed_tree, check, boundary_scale):
    """Original one-row traversal without repeated index-array construction.

    Keep the same seed, tree order, box arithmetic and exact segment distances.
    The seed and every visited segment still spend the original query budget.
    """
    import open3d as o3d
    nearest = np.full(1, np.inf)
    checked = 0
    scale = max(1., float(np.abs(points).max()), boundary_scale)
    slack = 32 * np.finfo(float).eps * scale
    reach_squared = None

    def distance(segment):
        nonlocal checked,reach_squared
        checked += 1
        if checked > MAX_DISTANCE_CHECKS:
            raise ValueError('覆盖边界采样检查超过预算，请缩小区域')
        a, b = boundary[segment]
        nearest[:] = np.minimum(nearest, point_segment_distance(points, a, b))
        # Recompute lazily at the next box, after its cancellation check. No
        # distance or acceptance result is cached across separate queries.
        reach_squared = None

    def candidate(low, high):
        nonlocal reach_squared
        delta = np.maximum(np.maximum(low-points, points-high), 0.)
        if reach_squared is None:
            reach_squared = (np.nextafter(nearest, np.inf) + slack) ** 2
        return bool((np.einsum('ij,ij->i', delta, delta) <= reach_squared)[0])

    check()
    seeds, _ = seed_tree.knn_search(o3d.core.Tensor(points), 1)
    distance(int(seeds.numpy().ravel()[0]))
    stack = [root]
    while stack:
        check()
        low, high, segments, left, right = stack.pop()
        if not candidate(low, high):
            continue
        if segments is None:
            stack.extend((right, left))
            continue
        for segment in segments:
            if candidate(lows[segment], highs[segment]):
                distance(segment)
    return nearest
