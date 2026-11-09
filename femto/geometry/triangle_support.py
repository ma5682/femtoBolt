"""Bounded, operation-local preparation for unchanged triangle predicates.

Face geometry and a guarded matrix decomposition are retained. Every point and
chord is checked again; no query coordinates or predicate results are cached.
"""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from functools import _make_key
from collections import OrderedDict, namedtuple
from threading import get_ident
import hashlib
import math
import struct
import sys
import weakref

import numpy as np

from femto.geometry.triangle_solve import TriangleSolve
from femto.small_vectors import plain_small_dot

_PACK_18 = struct.Struct('=18d')
_DEFAULT_ERRORS = dict(divide='warn', over='warn', under='ignore', invalid='warn')

_prepared_operation_cache = ContextVar('triangle_preparation_operation', default=None)
_MAX_SNAPSHOT_BYTES = 16 * 1024 * 1024
_MAX_FACES = 4096
_MAX_SHARED_BYTES = 16 * 1024 * 1024
_MAX_SHARED_FACES = 32768
_LEGACY_SHARED_BYTES = 4 * 1024 * 1024
_CacheInfo = namedtuple('CacheInfo', 'hits misses maxsize currsize')


def _record_bytes(value):
    """Conservative per-entry Python size accounting, including child records."""
    size = sys.getsizeof(value)
    if isinstance(value, (tuple, list)):
        size += sum(_record_bytes(part) for part in value)
    return size


class _SharedGeometry:
    """Bound retained shared records; private prepared caches stay at 4096.

    This accounts retained keys/values plus a conservative container allowance.
    It is not a process RSS or an all-operation-memory bound: snapshots, returned
    views and each existing private prepared cache have separate owners.
    """
    def __init__(self, build, max_entries, byte_budget, *, pool=None):
        self.build = build
        self.max_entries, self.byte_budget = max_entries, byte_budget
        self.values = OrderedDict()
        self.retained_bytes = self.peak_bytes = 0
        self.hits = self.misses = 0
        self.pool = {'budget': byte_budget, 'owners': [], 'peak_bytes': 0} if pool is None else pool
        self.pool['owners'].append(weakref.ref(self))

    def __call__(self, index):
        key = _make_key((index,), {}, False)
        try:
            value, charge = self.values[key]
        except KeyError:
            pass
        else:
            self.values.move_to_end(key)
            self.hits += 1
            return value
        self.misses += 1
        value = self.build(index)
        self._retain(key, value)
        return value

    def _retain(self, key, value):
        charge = _record_bytes(key) + _record_bytes(value) + 512
        if self.max_entries and charge <= self.byte_budget:
            previous = self.values.pop(key, None)
            if previous is not None:
                self.retained_bytes -= previous[1]
            while self.values and len(self.values) >= self.max_entries:
                _, (_, previous_charge) = self.values.popitem(last=False)
                self.retained_bytes -= previous_charge
            owners = [ref() for ref in self.pool['owners'] if ref() is not None]
            self.pool['owners'] = [weakref.ref(owner) for owner in owners]
            total = sum(owner.retained_bytes for owner in owners)
            for owner in owners:
                while owner.values and total + charge > self.pool['budget']:
                    _, (_, previous_charge) = owner.values.popitem(last=False)
                    owner.retained_bytes -= previous_charge
                    total -= previous_charge
            self.values[key] = value, charge
            self.retained_bytes += charge
            self.peak_bytes = max(self.peak_bytes, self.retained_bytes)
            self.pool['peak_bytes'] = max(self.pool['peak_bytes'], total + charge)

    def preparation(self, index):
        key = _make_key((index,), {}, False)
        record = self(index)
        if len(record) in (6, 8):
            return record[-1]
        *_, matrix, minimum = _unpack_geometry(record)
        prepared = _solver_record(matrix)
        # Replace one immutable row and charge its complete new size. This
        # avoids a second per-face dictionary and remains in the original pool.
        self._retain(key, (*record, prepared))
        return prepared

    def cache_info(self):
        return _CacheInfo(self.hits, self.misses, self.max_entries, len(self.values))


def _shared_policy(state, face_count):
    from femto.resource_policy import preparation_resources
    resources = state.get('resources')
    if resources is None:
        resources = preparation_resources(face_count)
        state['resources'] = resources
    total, available = resources.memory.total_bytes, resources.memory.available_bytes
    known = type(total) is int and type(available) is int and total > 0 and 0 <= available <= total
    # This policy authorizes growth beyond the existing cache, not removal of
    # legacy reuse. Unknown/low headroom retains at most 4096 compact records.
    expand = known and resources.cache_budget_bytes > _LEGACY_SHARED_BYTES
    budget = min(_MAX_SHARED_BYTES, resources.cache_budget_bytes) if expand else _LEGACY_SHARED_BYTES
    entries = min(face_count, _MAX_SHARED_FACES if expand else _MAX_FACES)
    return entries, budget


def _pack_geometry(point):
    arrays = point.origin, point.ab, point.ac, point.normal, point.matrix
    # Mixed endian/dtype records retain their original separate representation.
    if (all(value.dtype.str == arrays[0].dtype.str for value in arrays)
            and all(value.shape == (3,) for value in arrays[:4]) and point.matrix.shape == (3, 2)):
        payload = b''.join(value.tobytes(order='C') for value in arrays)
        return True, arrays[0].dtype.str, payload, point.normal_length, point.minimum_edge_length
    return (*map(_immutable, arrays[:4]), point.normal_length, _immutable(point.matrix), point.minimum_edge_length)


def _unpack_geometry(record):
    if len(record) in (6, 8):
        record = record[:-1]
    if len(record) == 5:
        _, dtype, payload, length, minimum = record
        flat = np.frombuffer(payload, dtype=np.dtype(dtype))
        # Keep the original C-order (3,2) matrix, not a transposed (2,3) block.
        return flat[:3], flat[3:6], flat[6:9], flat[9:12], length, flat[12:18].reshape(3, 2), minimum
    a, ab, ac, normal, length, matrix, minimum = record
    a, ab, ac, normal, matrix = map(_view, (a, ab, ac, normal, matrix))
    return a, ab, ac, normal, length, matrix, minimum


def _solver_record(matrix):
    """Only immutable decomposition data, never point/chord decisions."""
    solver = TriangleSolve(matrix)
    solver._prepare()
    return (None if solver._inverse is None else solver._inverse.tobytes(order='C'),
            solver._guard, solver._inverse_norm, solver._matrix_norm)


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


def _prepare_one(vertices,faces,index, *, packed=False):
    triangle = vertices[faces[index]]
    if (packed and triangle.dtype == np.float64 and triangle.shape == (3, 3)
            and np.geterr() == _DEFAULT_ERRORS and plain_small_dot()):
        # Directly materialize the immutable record used by this cache. Avoid
        # constructing five tiny arrays and a disposable solver to pack them.
        # The process probe checks the actual 3-vector BLAS norm reduction;
        # bounded finite operands also exclude overflow in cross products.
        a, b, c = triangle.tolist()
        if all(-1e30 <= v <= 1e30 for row in (a, b, c) for v in row):
            x, y, z = a
            u, v, w = b[0]-x, b[1]-y, b[2]-z
            i, j, k = c[0]-x, c[1]-y, c[2]-z
            nx, ny, nz = v*k-w*j, w*i-u*k, u*j-v*i
            payload = _PACK_18.pack(x,y,z, u,v,w, i,j,k, nx,ny,nz, u,i,v,j,w,k)
            length = np.float64(math.sqrt(((0. + nx*nx) + ny*ny) + nz*nz))
            minimum = min(math.sqrt(((0. + u*u) + v*v) + w*w),
                          math.sqrt(((0. + i*i) + j*j) + k*k))
            return True, triangle.dtype.str, payload, length, minimum
    a, b, c = triangle
    ab, ac = b-a, c-a
    normal = np.cross(ab, ac)
    length = np.linalg.norm(normal)
    matrix = np.column_stack((ab, ac))
    minimum_edge_length = min(float(np.linalg.norm(ab)), float(np.linalg.norm(ac)))
    for value in (triangle, a, ab, ac, normal, matrix):
        value.setflags(write=False)
    point = PreparedTriangleSupport(a, ab, ac, normal, length, matrix, minimum_edge_length,
                                    TriangleSolve(matrix))
    return _pack_geometry(point) if packed else point


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
            def build_geometry(index):
                return _prepare_one(source_vertices,source_faces,index,packed=True)
            capacity,budget=_shared_policy(state,len(faces))
            pool=state.setdefault('shared_pool',{'budget':budget,'owners':[],'peak_bytes':0})
            geometry=_SharedGeometry(build_geometry,capacity,budget,pool=pool)
            geometry.preparation_method = TriangleSolve._prepare
            geometry.error_policy = np.geterr()
            state['entries'].clear();state['entries'][key]=geometry;shared=geometry

        @lru_cache(maxsize=_MAX_FACES)
        def prepared(index):
            a,ab,ac,normal,length,matrix,minimum=_unpack_geometry(shared(index))
            def obtain():
                # Escaped closures/contexts and replaced solver methods retain
                # local preparation. The geometry key already binds every
                # source byte, and the bound matrix is backed by immutable bytes.
                if (not state['active'] or state['thread'] != get_ident()
                        or _prepared_operation_cache.get() is not state
                        or TriangleSolve._prepare is not shared.preparation_method
                        or np.geterr() != shared.error_policy):
                    return None
                return shared.preparation(index)
            solver = TriangleSolve(matrix, _shared_preparation=(matrix, obtain))
            return PreparedTriangleSupport(a,ab,ac,normal,length,matrix,minimum,solver)
        return vertices,faces,prepared

    @lru_cache(maxsize=4096)
    def prepared(index):
        return _prepare_one(vertices,faces,index)

    return vertices, faces, prepared
