"""Optional strict-binary64 cloud predicates; the NumPy path stays available.

The extension is built explicitly by tools/build_cloud_kernel.py. No runtime
compilation or download occurs. Buffers belong to the current call; no copied
points, normals, or geometry decisions survive it. Unsupported layouts, custom
NumPy error policies and extension allocation failures retain the old path.
"""
import os
import hashlib
from pathlib import Path
import numpy as np

try:
    if os.environ.get('FEMTO_DISABLE_CLOUD_KERNEL') == '1':
        _cloud_predicate = None
    else:
        from femto.geometry import _cloud_predicate
except (ImportError, OSError):
    _cloud_predicate = None

# A source checkout must rebuild after editing the C file. Packaged extensions
# may omit build-only sources; their ABI and recorded build still identify them.
if _cloud_predicate is not None:
    _source = Path(_cloud_predicate.__file__).resolve().parents[2] / 'native/cloud_predicate.c'
    try:
        if _source.is_file() and hashlib.sha256(_source.read_bytes()).hexdigest() != getattr(_cloud_predicate, 'SOURCE_SHA256', None):
            _cloud_predicate = None
    except OSError:
        _cloud_predicate = None


def available():
    return getattr(_cloud_predicate, 'ABI_VERSION', None) == 2


def _buffers_supported(arrays, integer_indices):
    if not available() or np.geterr() != dict(divide='warn', over='warn', under='ignore', invalid='warn'):
        return False
    return all(type(array) is np.ndarray and
               array.dtype == (np.int64 if index in integer_indices else np.float64) and
               array.flags.c_contiguous and array.flags.aligned
               for index, array in enumerate(arrays))


def classify(points, normals, queries, ids, splits):
    arrays = (points, normals, queries, ids, splits)
    if not _buffers_supported(arrays, (2, 3, 4)):
        return None
    from femto.geometry.cloud_layer_consistency import (
        _RADIUS_SQUARED, _RADIUS_GUARD, _HEIGHT, _ROUNDING_FACTOR)
    try:
        result = _cloud_predicate.classify(points, normals, queries, ids, splits,
            _RADIUS_SQUARED, _RADIUS_GUARD, _HEIGHT, _ROUNDING_FACTOR)
    except (MemoryError, BufferError, ValueError):
        return None
    status = np.frombuffer(result, np.uint8)
    return status == 1, status == 2


def negative_boxes(points, normals, keys, unique, offsets, low, high, sub_low, sub_high):
    arrays = (points, normals, keys, unique, offsets, low, high, sub_low, sub_high)
    if not _buffers_supported(arrays, (2, 3, 4)):
        return None
    from femto.geometry.cloud_layer_consistency import _CANDIDATE_RADIUS, _HEIGHT, _ROUNDING_FACTOR
    try:
        result = _cloud_predicate.negative_boxes(*arrays, _CANDIDATE_RADIUS, _HEIGHT, _ROUNDING_FACTOR)
    except (MemoryError, BufferError, ValueError):
        return None
    return np.frombuffer(result, np.uint8).astype(bool)
