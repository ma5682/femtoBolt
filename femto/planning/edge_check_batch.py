"""Batched edge predicates with the original decisions, budgets and errors.

Every rectangle, residual maximum and integral-image count is the same exact
binary64/integer computation as ``ReferenceEnvelope.probe``, evaluated for a
whole batch. Boundary clearance uses an exhaustive nearest-segment search and
admits a decision only when it is far from the original threshold. Chord
support still calls the original routine edge by edge. Anything unusual, near
a threshold or close to a work budget returns ``None`` so the caller runs the
unchanged scalar loop, which then also raises the original error.
"""
from __future__ import annotations

import numpy as np

_DEFAULT_ERRORS = dict(divide='warn', over='warn', under='ignore', invalid='warn')
_RESIDUAL_LIMIT = 100000000
# The original exhaustive nearest search spends at most one seed plus one
# evaluation per boundary segment for each point; below this it cannot stop.
_CLEARANCE_GUARD_M = 1e-12
_TABLE_BYTES = 64 * 1024 * 1024
_INDEX_PIECES = 262144
_QUERY_PAIRS = 262144
_ROW_CHUNK = 1 << 20
_TILE = 16


def _slice_bounds(start, stop, length):
    """Vectorised ``slice(max(0,start), min(length,stop)).indices(length)``."""
    start = np.minimum(np.maximum(start, 0), length)
    stop = np.minimum(stop, length)
    stop = np.where(stop < 0, np.maximum(stop + length, 0), stop)
    return start, stop


def _rectangles(mid, z, radius, radial, uv, k_max):
    """``_rectangle`` for rows with z > radius; identical elementwise binary64."""
    scale = k_max*radius/(z-radius)
    bound = scale*(1+radial/z)
    x, y = uv[:, 0], uv[:, 1]
    if not (np.all(np.abs(x)+np.abs(bound) < 2**30) and np.all(np.abs(y)+np.abs(bound) < 2**30)):
        return None
    return (np.floor(x-bound-.5).astype(np.int64), np.floor(y-bound-.5).astype(np.int64),
            np.ceil(x+bound+.5).astype(np.int64), np.ceil(y+bound+.5).astype(np.int64))


def _range_maxima(values, ya, yb, xa, xb, check):
    """Exact maxima of nonempty integer rectangles through a row sparse table."""
    if len(ya) <= 32:
        # A few rectangles: direct exact reductions avoid building tables.
        return np.array([values[top:bottom, left:right].max() for top, bottom, left, right
                         in zip(ya.tolist(), yb.tolist(), xa.tolist(), xb.tolist())], dtype=values.dtype)
    y_lo, y_hi = int(ya.min()), int(yb.max())
    x_lo, x_hi = int(xa.min()), int(xb.max())
    crop = values[y_lo:y_hi, x_lo:x_hi]
    widths = xb-xa
    levels = np.frexp(widths.astype(np.float64))[1].astype(np.int64)-1
    top = int(levels.max())
    if crop.nbytes*(top+1) > _TABLE_BYTES:
        return None
    tables = [crop]
    for level in range(1, top+1):
        check()
        previous, step = tables[-1], 1 << (level-1)
        tables.append(np.maximum(previous[:, :-step], previous[:, step:]))
    result = np.empty(len(ya), dtype=values.dtype)
    heights = yb-ya
    for level in np.unique(levels).tolist():
        table = tables[level]
        members = np.flatnonzero(levels == level)
        first = 0
        while first < len(members):
            check()
            # Bound the gathered row count of one chunk.
            sizes = heights[members[first:]]
            take = int(np.searchsorted(np.cumsum(sizes), _ROW_CHUNK, side='right'))
            take = max(1, min(take, len(sizes)))
            chosen = members[first:first+take]
            counts = heights[chosen]
            starts = np.r_[0, np.cumsum(counts)[:-1]]
            rows = (np.repeat(ya[chosen]-y_lo-starts, counts) + np.arange(int(counts.sum())))
            left = np.repeat(xa[chosen]-x_lo, counts)
            right = np.repeat(xb[chosen]-x_lo-(1 << level), counts)
            row_max = np.maximum(table[rows, left], table[rows, right])
            result[chosen] = np.maximum.reduceat(row_max, starts)
            first += take
    return result


def _residual_charges(ya, yb, xa, xb, shape):
    """Logical work charged by ``ReferenceEnvelope._residual_max`` per query."""
    height, width = shape
    top = ((ya+_TILE-1)//_TILE)*_TILE
    bottom = (yb//_TILE)*_TILE
    left = ((xa+_TILE-1)//_TILE)*_TILE
    right = (xb//_TILE)*_TILE
    tiled = (top < bottom) & (left < right)
    area = (yb-ya)*(xb-xa)
    interior = np.where(tiled, (bottom-top)*(right-left), 0)
    tiles = np.where(tiled, ((bottom-top)//_TILE)*((right-left)//_TILE), 0)
    return area-interior+tiles, tiled, height*width+((height+_TILE-1)//_TILE)*width


_PIECE_HALF_M = .00025


def _segment_index(boundary, check=lambda: None):
    """KNN index over short sub-segment midpoints of every boundary segment.

    Each boundary segment is split into pieces of half length at most 0.25 mm;
    a point is never closer to a segment than to its nearest piece midpoint
    minus that half length. Only the search order uses this index.
    """
    import open3d as o3d
    check()
    starts, ends = boundary[:, 0], boundary[:, 1]
    lengths = np.sqrt(np.einsum('ij,ij->i', ends-starts, ends-starts))
    counts = np.maximum(1., np.ceil(lengths/(2*_PIECE_HALF_M)))
    if not np.isfinite(counts).all() or counts.sum() > _INDEX_PIECES:
        return None
    pieces = counts.astype(np.int64)
    check()
    owner = np.repeat(np.arange(len(boundary)), pieces)
    first = np.repeat(np.r_[0, np.cumsum(pieces)[:-1]], pieces)
    fraction = ((np.arange(len(owner))-first)+.5)/pieces[owner]
    samples = starts[owner]+fraction[:, None]*(ends[owner]-starts[owner])
    tree = o3d.core.nns.NearestNeighborSearch(o3d.core.Tensor(samples))
    tree.knn_index()
    check()
    coordinate_guard = 128*np.finfo(float).eps*max(1., float(np.abs(boundary).max()))
    reach = float(np.nextafter((lengths/(2*pieces)).max(), np.inf))*(1+1e-9)+coordinate_guard+1e-12
    return starts, ends, owner, tree, reach


def _clearance(points, limits, index, check):
    """Decide nearest-boundary distance >= limit far from equality, else None.

    The minimum is exhaustive: a segment without any of the k nearest piece
    midpoints is at least the k-th distance minus the longest half piece away.
    """
    import open3d as o3d
    starts, ends, owner, tree, reach = index
    count = len(owner)
    guard = _CLEARANCE_GUARD_M + 128*np.finfo(float).eps*max(
        1., float(np.abs(starts).max()), float(np.abs(ends).max()), float(np.abs(points).max(initial=0)))
    clear = np.zeros(len(points), bool)
    pending = np.arange(len(points))
    k = min(count, 8)
    while len(pending):
        check()
        if len(pending)*k > _QUERY_PAIRS:
            return None
        query = points[pending]
        limit = limits[pending]
        ids, squared = tree.knn_search(o3d.core.Tensor(query), k)
        ids, squared = ids.numpy(), squared.numpy()
        # Every segment is covered by the sampled midpoints and 'reach'. Its
        # distance is at least the nearest midpoint distance minus that reach.
        # Far interiors therefore need no point-to-segment projections.
        if k == min(count, 8):
            clear_now = np.sqrt(squared[:, 0])*(1-1e-12)-reach > limit+guard
            if clear_now.any():
                clear[pending[clear_now]] = True
                if clear_now.all():
                    break
                keep = ~clear_now
                pending, query, limit = pending[keep], query[keep], limit[keep]
                ids, squared = ids[keep], squared[keep]
        ids = owner[ids]
        a, b = starts[ids], ends[ids]
        delta = b-a
        den = np.maximum(np.einsum('nkj,nkj->nk', delta, delta), 1e-30)
        offset = query[:, None, :]-a
        t = np.clip(np.einsum('nkj,nkj->nk', offset, delta)/den, 0, 1)
        closest = query[:, None, :]-(a+t[..., None]*delta)
        found = np.sqrt(np.einsum('nkj,nkj->nk', closest, closest)).min(axis=1)
        if np.any(np.abs(found-limit) <= guard):
            return None
        blocked = found < limit
        if k >= count:
            clear[pending[~blocked]] = True
            break
        farthest = np.sqrt(squared[:, -1])*(1-1e-12)-reach
        proven = ~blocked & (farthest > limit+guard)
        clear[pending[proven]] = True
        pending = pending[~blocked & ~proven]
        k = min(count, k*4)
    return clear


def _reference_probes(probed, mid, half, reference, check, small):
    # Lazy graph search frequently asks about just one edge. Use the existing
    # scalar probe there: large vector batches build many one-element arrays.
    # Work on a private counter until all checks succeed, just like the batch.
    residual, integral, k = reference.residual, reference.integral, reference.k
    if (small and len(probed) <= 4 and
            reference.sampled_pixels + (len(probed)+2)*residual.size <= _RESIDUAL_LIMIT):
        from copy import copy
        local = copy(reference)
        passed = np.zeros(len(probed), bool)
        margins = np.zeros(len(probed), float)
        for pos, row in enumerate(probed):
            check()
            passed[pos], margins[pos], _ = local.probe(mid[row], half[row])
        return margins, passed, local.sampled_pixels, local._residual_tiles
    probe_width = reference.width_mm
    global_residual = reference.global_residual
    k_max = max(k[0, 0], k[1, 1])
    focal = np.array([k[0, 0], k[1, 1]])
    centre = k[:2, 2]
    height, width = reference.height, reference.width
    spent = None
    new_tiles = None
    margin = np.zeros(len(probed), float)
    rect_ok = np.zeros(len(probed), bool)
    rect = [np.zeros(len(probed), np.int64) for _ in range(4)]
    if len(probed):
        h = half[probed]
        z = mid[probed, 2]
        outer_radius = h+probe_width/2000+global_residual
        outer = np.flatnonzero(z > outer_radius)
        if len(outer):
            rows = probed[outer]
            zo = z[outer]
            uv = mid[rows, :2]/zo[:, None]*focal+centre
            from femto.small_vectors import row_norms
            radial = row_norms(mid[rows, :2])
            check()
            outer_rect = _rectangles(mid[rows], zo, outer_radius[outer], radial, uv, k_max)
            if outer_rect is None:
                return None
            x0, y0, x1, y1 = outer_rect
            ya, yb = _slice_bounds(y0, y1+1, height)
            xa, xb = _slice_bounds(x0, x1+1, width)
            nonempty = (ya < yb) & (xa < xb)
            maxima = np.full(len(outer), global_residual)
            used = np.flatnonzero(nonempty)
            if len(used):
                charges, tiled, tile_start = _residual_charges(ya[used], yb[used], xa[used], xb[used], residual.shape)
                # Charges are nonnegative: staying within the limit at the end
                # means no intermediate original check could have raised.
                spent = (reference.sampled_pixels+int(charges.sum())
                         + (tile_start if tiled.any() and reference._residual_tiles is None else 0))
                if spent > _RESIDUAL_LIMIT:
                    return None
                values = _range_maxima(residual, ya[used], yb[used], xa[used], xb[used], check)
                if values is None:
                    return None
                maxima[used] = values.astype(np.float64)/1e6
                if tiled.any() and reference._residual_tiles is None:
                    # Publish tile preparation only after all checks succeed;
                    # subsequent calls must not charge the first tile twice.
                    if reference._shared_residual_tiles is not None:
                        new_tiles = reference._shared_residual_tiles.get(residual)
                    if new_tiles is None:
                        check()
                        tile_rows = np.maximum.reduceat(residual, np.arange(0, height, _TILE), axis=0)
                        new_tiles = np.maximum.reduceat(tile_rows, np.arange(0, width, _TILE), axis=1)
                        new_tiles.setflags(write=False)
            margin[outer] = maxima
            same = maxima == global_residual
            final = [np.where(same, part, 0) for part in outer_rect]
            valid = same.copy()
            inner = np.flatnonzero(~same)
            if len(inner):
                inner_radius = h[outer[inner]]+probe_width/2000+maxima[inner]
                zi = zo[inner]
                ready = np.flatnonzero(zi > inner_radius)
                if len(ready):
                    picked = inner[ready]
                    inner_rect = _rectangles(None, zi[ready], inner_radius[ready], radial[picked], uv[picked], k_max)
                    if inner_rect is None:
                        return None
                    for index, part in enumerate(inner_rect):
                        final[index][picked] = part
                    valid[picked] = True
            for index in range(4):
                rect[index][outer] = final[index]
            rect_ok[outer] = valid
    passed_probe = np.zeros(len(probed), bool)
    if rect_ok.any():
        chosen = np.flatnonzero(rect_ok)
        x0, y0, x1, y1 = (part[chosen] for part in rect)
        inside = (0 <= x0) & (x0 <= x1) & (x1 < width) & (0 <= y0) & (y0 <= y1) & (y1 < height)
        within = chosen[inside]
        if len(within):
            x0, y0, x1, y1 = x0[inside], y0[inside], x1[inside], y1[inside]
            bad = integral[y1+1, x1+1]-integral[y0, x1+1]-integral[y1+1, x0]+integral[y0, x0]
            clean = bad == 0
            passed_probe[within[clean]] = True
    return margin, passed_probe, spent, new_tiles


def evaluate(pairs, a, b, mid, half, faces, vertices, reference, width_mm, check, allowed_faces,
             boundary, boundary_query, prepared_faces, vertex_order, vertex_offsets):
    """Return (accepted, reasons, margins) or None for the original loop."""
    from femto.planning import quality_constrained as quality, boundary_clearance
    count = len(pairs)
    if (count < 1 or type(reference) is not quality.ReferenceEnvelope or np.geterr() != _DEFAULT_ERRORS
            or mid.dtype != np.float64 or half.dtype != np.float64 or not np.isfinite(mid).all()
            or not np.isfinite(half).all() or np.abs(mid).max() > 1e6):
        return None
    k = reference.k
    residual, integral = reference.residual, reference.integral
    if (type(k) is not np.ndarray or k.dtype != np.float64 or k.shape != (3, 3) or not np.isfinite(k).all()
            or np.abs(k).max() > 1e6 or type(residual) is not np.ndarray or residual.ndim != 2
            or residual.dtype.kind not in 'iu' or residual.shape != reference.classes.shape
            or type(boundary) is not np.ndarray or boundary.dtype != np.float64 or boundary.ndim != 3
            or boundary.shape[1:] != (2, 3) or not len(boundary) or not np.isfinite(boundary).all()
            or count*(len(boundary)+1) > boundary_clearance.MAX_DISTANCE_CHECKS):
        return None
    face_total = len(faces)
    try:
        left = [pair[0]['triangle_index'] for pair in pairs]
        right = [pair[1]['triangle_index'] for pair in pairs]
    except (KeyError, TypeError):
        return None
    if any(type(value) is not int or not 0 <= value < face_total for value in left+right):
        return None
    left, right = np.asarray(left, np.int64), np.asarray(right, np.int64)
    if allowed_faces is not None:
        if (type(allowed_faces) is not np.ndarray or allowed_faces.dtype != np.bool_
                or allowed_faces.shape != (face_total,)):
            return None
        domain = allowed_faces[left] & allowed_faces[right]
    else:
        domain = np.ones(count, bool)
    check()
    accepted = np.zeros(count, bool)
    margins = np.zeros(count, float)
    reasons = np.full(count, 'outside_effective_domain', dtype=object)
    probed = np.flatnonzero(domain)
    # A failed reference probe keeps the loop's generic reason; its detailed
    # cause is reported only through diagnostics, which use the loop.
    reasons[probed] = 'quality_or_width_boundary'
    probes = _reference_probes(probed, mid, half, reference, check, boundary_query is not None)
    if probes is None:
        return None
    margin, passed_probe, spent, new_tiles = probes
    margins[probed] = margin*1000
    candidates = probed[passed_probe]
    if len(candidates):
        # A graph binds one immutable boundary; reuse only its search index.
        holder = getattr(boundary_query, 'exact_nearest', None) if boundary_query is not None else None
        index = None
        if isinstance(holder, dict) and holder.get('boundary') is boundary and not boundary.flags.writeable:
            index = holder.get('index')
        if index is None:
            index = _segment_index(boundary, check)
            if index is None:
                return None
            if isinstance(holder, dict) and not boundary.flags.writeable:
                holder.update(boundary=boundary, index=index)
        # distance-half >= rhs, decided only when far from equality.
        limits = width_mm/2000+margin[passed_probe]+1e-9+half[candidates]
        if not np.isfinite(limits).all():
            return None
        clear = _clearance(mid[candidates], limits, index, check)
        if clear is None:
            return None
        reasons[candidates[~clear]] = 'mesh_boundary_clearance'
        supported_ids = candidates[clear]
        # Same-face groups use the exact batch decomposition of the original.
        interior = np.zeros(count, bool)
        groups = {}
        for i, (start, end) in enumerate(pairs):
            index = start.get('triangle_index')
            if (type(index) is int and 0 <= index < face_total
                    and type(end.get('triangle_index')) is int and end['triangle_index'] == index):
                groups.setdefault(index, []).append(i)
        for index, ids in groups.items():
            if len(ids) >= 8:
                check()
                face = prepared_faces(index)
                interior[ids] = face.solver.chord_inside(a[ids], b[ids], face)
        # Single-face chords far inside their face are surely supported by the
        # original routine; every other chord is still evaluated by it.
        single = supported_ids[(left[supported_ids] == right[supported_ids]) & ~interior[supported_ids]]
        if len(single):
            from femto.planning.triangle_certificates import chords_on_triangles
            certified = chords_on_triangles(a[single], b[single], vertices, faces, left[single],
                                            _prepared_faces=prepared_faces, check=check)
            interior[single[certified]] = True
        # At a face handoff the incoming edge often lies entirely on the end
        # face. A certified full [0,1] interval is already a proof for the
        # union of the two named faces. A failed certificate says nothing:
        # keep the original interval union and incident-face fallback below.
        handoffs = supported_ids[left[supported_ids] != right[supported_ids]]
        if (len(handoffs) >= 32 and quality._supported_chord is quality._ORIGINAL_SUPPORTED_CHORD
                and np.isfinite(vertices).all() and np.abs(vertices).max(initial=0) < 1e30):
            from femto.planning.triangle_certificates import chords_on_triangles
            certified = chords_on_triangles(a[handoffs], b[handoffs], vertices, faces, right[handoffs],
                                            _prepared_faces=prepared_faces, check=check)
            interior[handoffs[certified]] = True
        ready = supported_ids[interior[supported_ids]]
        accepted[ready] = True
        reasons[ready] = 'accepted'
        for position, i in enumerate(supported_ids[~interior[supported_ids]].tolist()):
            if position % 128 == 0:check()
            start, end = pairs[i]
            ids = [start['triangle_index'], end['triangle_index']]
            supported = quality._supported_chord(
                a[i], b[i], faces, vertices, ids, _prepared_faces=prepared_faces)
            if not supported:
                incident = np.unique(np.concatenate([vertex_order[vertex_offsets[v]:vertex_offsets[v+1]]//3
                    for v in np.unique(faces[ids])]))
                if allowed_faces is not None:incident = incident[allowed_faces[incident]]
                supported = quality._supported_chord(a[i], b[i], faces, vertices, incident.tolist(),
                                                     _prepared_faces=prepared_faces)
            if supported:
                accepted[i] = True
                reasons[i] = 'accepted'
            else:
                reasons[i] = 'chord_not_certified_on_mesh'
    check()
    if spent is not None:
        reference.sampled_pixels = spent
    if new_tiles is not None:
        reference._residual_tiles = new_tiles
    return accepted, reasons.tolist(), margins
