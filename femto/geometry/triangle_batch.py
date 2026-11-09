"""Cross-face positive certificates using the existing triangle error bounds.

No predicate result is retained. The scalar solver remains authoritative for
every row not certified here, including poorly conditioned and boundary rows.
"""
from __future__ import annotations

import numpy as np

from femto.geometry.triangle_solve import TriangleSolve, _EPS, _SQRT3

_POINT_INSIDE = TriangleSolve.point_inside
_CHORD_INSIDE = TriangleSolve.chord_inside
_COORDINATES = TriangleSolve.coordinates
_PREPARE = TriangleSolve._prepare
_BLOCK = 8192


class AllocationUnavailable(MemoryError):
    """Only optional batch storage failed; caller may use the scalar path."""


def _optional(function, *args):
    try:
        return function(*args)
    except MemoryError as error:
        raise AllocationUnavailable('cross-face temporary allocation unavailable') from error


def _table(groups, size, check):
    """Gather immutable per-face maps; never recompute an unguarded inverse."""
    mapping = _optional(np.full, size, -1, np.int64)
    prepared = []
    for index, (rows, face) in enumerate(groups):
        if index % 128 == 0:
            check()
        solver = face.solver
        if type(solver) is not TriangleSolve:
            continue
        solver._prepare()
        if solver._inverse is None:
            continue
        try:
            mapping[rows] = len(prepared)
            prepared.append(face)
        except MemoryError as error:
            raise AllocationUnavailable('cross-face table allocation unavailable') from error
    if not prepared:
        return mapping, None
    # Each table row is one original face. Input rows only gather this data;
    # neither source arrays nor solver objects are modified by the batch.
    return mapping, _optional(_columns, prepared)


def _columns(prepared):
    return dict(
        origin=np.asarray([p.origin for p in prepared]),
        ab=np.asarray([p.ab for p in prepared]),
        ac=np.asarray([p.ac for p in prepared]),
        normal=np.asarray([p.normal for p in prepared]),
        length=np.asarray([p.normal_length for p in prepared]),
        minimum=np.asarray([p.minimum_edge_length for p in prepared]),
        inverse=np.asarray([p.solver._inverse for p in prepared]),
        guard=np.asarray([p.solver._guard for p in prepared]),
        inverse_norm=np.asarray([p.solver._inverse_norm for p in prepared]),
        matrix_norm=np.asarray([p.solver._matrix_norm for p in prepared]))


def _rows(groups, size, check):
    mapping, table = _table(groups, size, check)
    if table is None:
        return
    usable = _optional(lambda: np.flatnonzero(mapping >= 0))
    for first in range(0, len(usable), _BLOCK):
        check()
        rows = usable[first:first+_BLOCK]
        yield rows, _optional(_gather, table, mapping, rows)


def _gather(table, mapping, rows):
    ids = mapping[rows]
    return {name: value[ids] for name, value in table.items()}


def points(points, groups, tolerance, check):
    result = _optional(np.zeros, len(points), bool)
    if (TriangleSolve.point_inside is not _POINT_INSIDE
            or TriangleSolve.coordinates is not _COORDINATES or TriangleSolve._prepare is not _PREPARE
            or not np.isscalar(tolerance) or not np.isfinite(tolerance) or tolerance <= 0):
        return result
    for rows, face in _rows(groups, len(points), check):
        result[rows] = _optional(_point_block, points, rows, face, tolerance)
    return result


def _point_block(points, rows, face, tolerance):
    xyz = points[rows]
    offset = xyz-face['origin']
    coordinates = (face['inverse'] @ offset[..., None])[..., 0]
    # Twice the established uncertainty absorbs the different batch dot
    # reduction as well as the original QR/SVD and residual uncertainty.
    # This only narrows the positive certificate. Every other row runs the
    # unchanged per-face predicate in the caller.
    uncertainty = 2*face['guard']*(1+np.max(np.abs(offset), axis=1)*face['inverse_norm'])
    allowed = tolerance/np.maximum(face['minimum'], tolerance)
    closest = (face['origin']+coordinates[:, 0, None]*face['ab']
               +coordinates[:, 1, None]*face['ac'])
    rounding = 128*_EPS*(np.max(np.abs(xyz), axis=1)+np.max(np.abs(face['origin']), axis=1)
                         +face['matrix_norm']*np.max(np.abs(coordinates), axis=1))
    residual_guard = _SQRT3*(2*face['matrix_norm']*uncertainty+rounding)
    normal_dot = np.abs(np.sum(offset*face['normal'], axis=1))
    normal_guard = 64*_EPS*np.sum(np.abs(offset)*np.abs(face['normal']), axis=1)
    return ((face['length'] > 0) & np.isfinite(face['length'])
                    & (normal_dot+normal_guard < tolerance*face['length'])
                    & (np.min(coordinates, axis=1) > -allowed+uncertainty)
                    & (coordinates.sum(axis=1) < 1+allowed-2*uncertainty)
                    & (np.linalg.norm(closest-xyz, axis=1)+residual_guard < tolerance))


def chords(starts, ends, groups, check):
    result = _optional(np.zeros, len(starts), bool)
    if (TriangleSolve.chord_inside is not _CHORD_INSIDE
            or TriangleSolve.coordinates is not _COORDINATES or TriangleSolve._prepare is not _PREPARE):
        return result
    for rows, face in _rows(groups, len(starts), check):
        result[rows] = _optional(_chord_block, starts, ends, rows, face)
    return result


def _chord_block(starts, ends, rows, face):
    offsets = np.stack((starts[rows]-face['origin'], ends[rows]-face['origin']), axis=1)
    coordinates = offsets @ face['inverse'].transpose(0, 2, 1)
    uncertainty = 2*face['guard'][:, None]*(1+np.max(np.abs(offsets), axis=2)*face['inverse_norm'][:, None])
    normal_dot = np.abs(np.sum(offsets*face['normal'][:, None], axis=2))
    normal_guard = 64*_EPS*np.sum(np.abs(offsets)*np.abs(face['normal'][:, None]), axis=2)
    inside = ((np.min(coordinates, axis=2) > -1e-8+uncertainty)
              & (coordinates.sum(axis=2) < 1+1e-8-2*uncertainty)
              & (normal_dot+normal_guard < 1e-6*face['length'][:, None]))
    return ((face['length'] > 1e-15) & np.isfinite(face['length']) & inside.all(axis=1))
