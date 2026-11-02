"""Bounded geometry reuse within one synchronous planning operation.

Only derived geometry is retained, never an acceptance verdict. File readers
still check their complete manifest, bytes hash and topology on every call.
The context is reset on success, cancellation and failure, including nesting.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
from pathlib import Path

import numpy as np
import trimesh


MAX_BYTES = 64 * 1024 * 1024
_active = ContextVar('region_geometry_operation', default=None)


@contextmanager
def operation():
    if _active.get() is not None:
        yield
        return
    token = _active.set({})
    try:
        from femto.geometry.triangle_support import _prepared_operation
        with _prepared_operation():
            yield
    finally:
        _active.reset(token)


def _arrays(mesh):
    if isinstance(mesh, trimesh.Trimesh):
        return (np.asarray(mesh.vertices), np.asarray(mesh.faces))
    arrays = tuple(np.asarray(getattr(mesh, name)) for name in
                   ('vertices', 'triangles', 'vertex_colors', 'vertex_normals',
                    'triangle_normals', 'triangle_material_ids', 'triangle_uvs'))
    return arrays + tuple(np.asarray(image) for image in mesh.textures)


def _fingerprint(arrays):
    digest = hashlib.sha256()
    for value in arrays:
        digest.update(str((value.shape, value.dtype.str)).encode('ascii'))
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.digest()


def _reuse(kind, key, build):
    cache = _active.get()
    if cache is None:
        return build()
    old = cache.get(kind)
    reusable = old is not None and not (
        not isinstance(old[1], trimesh.Trimesh) and old[1].adjacency_list)
    if reusable and old[0] == key and _fingerprint(_arrays(old[1])) == old[2]:
        return old[1]
    value = build()
    arrays = _arrays(value)
    # At most one extraction, one canonical mesh and one just-written PLY.
    cache.pop(kind, None)
    size = sum(array.nbytes for array in arrays)
    if size + sum(entry[3] for entry in cache.values()) <= MAX_BYTES:
        cache[kind] = (key, value, _fingerprint(arrays), size)
    return value


def extracted(mesh, indices, build):
    if (_active.get() is None or mesh.textures or mesh.has_triangle_uvs()
            or mesh.adjacency_list):
        return build()
    if type(indices) not in (list, np.ndarray):
        return build()
    selection = np.asarray(indices)
    if selection.ndim != 1 or selection.dtype.kind not in 'iu':
        return build()
    key = _fingerprint((*_arrays(mesh), selection))
    return _reuse('region', key, build)


def canonical(local, build):
    if _active.get() is None:
        return build()
    key = _fingerprint((np.asarray(local.vertices), np.asarray(local.triangles)))
    return _reuse('canonical', key, build)


def remember_ply(path, encoded, mesh):
    """Bind the exporter-created mesh to exactly the bytes that were written."""
    if _active.get() is None:
        return
    key = (Path(path).resolve(), hashlib.sha256(encoded).hexdigest())
    def build():
        return trimesh.Trimesh(
            vertices=np.asarray(mesh.vertices).astype(np.float32).astype(np.float64),
            faces=np.array(mesh.faces, copy=True), process=False)
    _reuse('ply', key, build)


def loaded_ply(path, verified_sha256):
    """Return geometry only after the caller has verified the actual file hash."""
    cache = _active.get()
    old = cache.get('ply') if cache is not None else None
    if (old is not None and old[0] == (Path(path).resolve(), verified_sha256)
            and _fingerprint(_arrays(old[1])) == old[2]):
        return old[1]
    return None
