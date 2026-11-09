"""Batch adapters for the established, error-bounded triangle solver.

Cross-face batches share the existing condition/residual bounds. Uncertain
rows retain the original per-face certificate and scalar predicate fallback.
"""
from __future__ import annotations

import numpy as np

_DEFAULT_ERRORS = dict(divide='warn', over='warn', under='ignore', invalid='warn')
_FACE_BLOCK = 4096
_MAX_QUERY_ROWS = 262144


def _usable(points, vertices, faces, ids):
    return (np.geterr() == _DEFAULT_ERRORS and type(points) is np.ndarray and points.dtype == np.float64
            and points.ndim == 2 and points.shape[1:] == (3,) and np.isfinite(points).all()
            and type(vertices) is np.ndarray and vertices.dtype == np.float64
            and vertices.ndim == 2 and vertices.shape[1:] == (3,)
            and type(faces) is np.ndarray and faces.dtype.kind in 'iu'
            and faces.ndim == 2 and faces.shape[1:] == (3,)
            and type(ids) is np.ndarray and ids.dtype.kind in 'iu' and ids.shape == (len(points),)
            and (not len(ids) or (ids.min() >= 0 and ids.max() < len(faces))))


def _groups(ids, vertices, faces, prepared_faces):
    if prepared_faces is None:
        from femto.geometry.triangle_support import prepare_triangle_faces
        _, _, prepared_faces = prepare_triangle_faces(vertices, faces)
    groups = {}
    for i, face in enumerate(ids.tolist()):
        groups.setdefault(face, []).append(i)
    for face, rows in groups.items():
        yield rows, prepared_faces(face)


def _prepared_batches(groups, check):
    block = []
    for rows, face in groups:
        check()
        block.append((rows, face))
        if len(block) >= _FACE_BLOCK:
            yield block
            block = []
    if block:
        yield block


def points_on_triangles(points, vertices, faces, ids, tolerance_m=1e-6, *,
                        _prepared_faces=None, check=lambda: None):
    points, ids = np.asarray(points), np.asarray(ids)
    result = np.zeros(len(points), bool)
    if not len(points) or not _usable(points, vertices, faces, ids):
        return result
    if len(points) < 32 or len(points) > _MAX_QUERY_ROWS:
        for rows, prepared in _groups(ids, vertices, faces, _prepared_faces):
            check()
            result[rows] = prepared.solver.point_inside(points[rows], prepared, tolerance_m)
        return result
    for groups in _prepared_batches(_groups(ids, vertices, faces, _prepared_faces), check):
        if len(groups) >= 8:
            from femto.geometry.triangle_batch import points as certify, AllocationUnavailable
            try:
                result |= certify(points, groups, tolerance_m, check)
                if result.all():
                    return result
            except AllocationUnavailable:
                # No partial block was published. Keep original predicates;
                # cancellation and solver failures are never swallowed here.
                pass
        for rows, prepared in groups:
            pending = np.asarray(rows)[~result[rows]]
            if len(pending):
                check()
                result[pending] = prepared.solver.point_inside(points[pending], prepared, tolerance_m)
    return result


def chords_on_triangles(starts, ends, vertices, faces, ids, *,
                        _prepared_faces=None, check=lambda: None):
    starts, ends, ids = np.asarray(starts), np.asarray(ends), np.asarray(ids)
    result = np.zeros(len(starts), bool)
    if (not len(starts) or not _usable(starts, vertices, faces, ids)
            or not _usable(ends, vertices, faces, ids)):
        return result
    if len(starts) < 32 or len(starts) > _MAX_QUERY_ROWS:
        for rows, prepared in _groups(ids, vertices, faces, _prepared_faces):
            check()
            result[rows] = prepared.solver.chord_inside(starts[rows], ends[rows], prepared)
        return result
    for groups in _prepared_batches(_groups(ids, vertices, faces, _prepared_faces), check):
        if len(groups) >= 8:
            from femto.geometry.triangle_batch import chords as certify, AllocationUnavailable
            try:
                result |= certify(starts, ends, groups, check)
                if result.all():
                    return result
            except AllocationUnavailable:
                pass
        for rows, prepared in groups:
            pending = np.asarray(rows)[~result[rows]]
            if len(pending):
                check()
                result[pending] = prepared.solver.chord_inside(starts[pending], ends[pending], prepared)
    return result
