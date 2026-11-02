"""Bounded, operation-local preparation for unchanged triangle predicates.

Face geometry and a guarded matrix decomposition are retained. Every point and
chord is checked again; no query coordinates or predicate results are cached.
"""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from threading import get_ident
import hashlib

import numpy as np

from femto.geometry.triangle_solve import TriangleSolve

_prepared_operation_cache = ContextVar('triangle_preparation_operation', default=None)
_MAX_SNAPSHOT_BYTES = 16 * 1024 * 1024
_MAX_FACES = 4096


@contextmanager
def _prepared_operation():
    """Keep only immutable derived geometry inside one synchronous operation."""
    current=_prepared_operation_cache.get()
    if current is not None and current['active'] and current['thread']==get_ident():
        yield
        return
    state={'thread':get_ident(),'active':True,'entries':{}}
    token=_prepared_operation_cache.set(state)
    try:yield
    finally:
        state['active']=False
        state['entries'].clear()
        _prepared_operation_cache.reset(token)


@dataclass(frozen=True, slots=True)
class PreparedTriangleSupport:
    origin: np.ndarray
    ab: np.ndarray
    ac: np.ndarray
    normal: np.ndarray
    normal_length: float
    matrix: np.ndarray
    minimum_edge_length: float
    solver: TriangleSolve


def _prepare_one(vertices,faces,index):
    triangle = vertices[faces[index]]
    a, b, c = triangle
    ab, ac = b-a, c-a
    normal = np.cross(ab, ac)
    length = np.linalg.norm(normal)
    matrix = np.column_stack((ab, ac))
    minimum_edge_length = min(float(np.linalg.norm(ab)), float(np.linalg.norm(ac)))
    for value in (triangle, a, ab, ac, normal, matrix):
        value.setflags(write=False)
    return PreparedTriangleSupport(a, ab, ac, normal, length, matrix, minimum_edge_length,
                                   TriangleSolve(matrix))


def _fingerprint(vertices,faces):
    digest=hashlib.sha256()
    for array in (vertices,faces):
        value=np.ascontiguousarray(array)
        digest.update(str((value.shape,value.dtype.str)).encode('ascii'))
        digest.update(memoryview(value.reshape(-1)).cast('B'))
    return digest.digest()


def _immutable(value):
    return value.dtype.str,value.shape,value.tobytes()


def _view(value):
    return np.frombuffer(value[2],dtype=np.dtype(value[0])).reshape(value[1])


def prepare_triangle_faces(vertices, faces):
    """Return a frozen mesh snapshot and its bounded per-face preparation.

    The returned closure owns this snapshot and dies with the operation/graph.
    An independent call always takes a fresh snapshot, even for the same mesh.
    """
    vertices = np.array(vertices, copy=True)
    faces = np.array(faces, copy=True)
    vertices.setflags(write=False)
    faces.setflags(write=False)

    state=_prepared_operation_cache.get()
    if (state is not None and state['active'] and state['thread']==get_ident()
            and vertices.ndim==2 and vertices.shape[1:]==(3,) and vertices.dtype.kind in 'fiu'
            and faces.ndim==2 and faces.shape[1:]==(3,) and faces.dtype.kind in 'iu'
            and vertices.nbytes+faces.nbytes<=_MAX_SNAPSHOT_BYTES):
        # Back the public snapshot by immutable bytes as well. Otherwise a
        # caller could re-enable writes and desynchronise it from the shared
        # derivation used by this invocation.
        vertices=_view(_immutable(vertices));faces=_view(_immutable(faces))
        key=_fingerprint(vertices,faces);shared=state['entries'].get(key)
        if shared is None:
            # Private bytes-backed source snapshots cannot be made writable by
            # a caller holding this invocation's returned arrays or solver.
            source_vertices=vertices;source_faces=faces
            @lru_cache(maxsize=_MAX_FACES)
            def geometry(index):
                point=_prepare_one(source_vertices,source_faces,index)
                return (_immutable(point.origin),_immutable(point.ab),_immutable(point.ac),_immutable(point.normal),
                        point.normal_length,_immutable(point.matrix),point.minimum_edge_length)
            state['entries'].clear();state['entries'][key]=geometry;shared=geometry

        @lru_cache(maxsize=_MAX_FACES)
        def prepared(index):
            a,ab,ac,normal,length,matrix,minimum=shared(index)
            a,ab,ac,normal,matrix=map(_view,(a,ab,ac,normal,matrix))
            return PreparedTriangleSupport(a,ab,ac,normal,length,matrix,minimum,TriangleSolve(matrix))
        return vertices,faces,prepared

    @lru_cache(maxsize=4096)
    def prepared(index):
        return _prepare_one(vertices,faces,index)

    return vertices, faces, prepared
