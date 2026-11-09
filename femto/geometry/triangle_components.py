"""Bounded integer edge connectivity; no geometric thresholds or repairs."""
import numpy as np
from femto.performance_detail import observed, count


MAX_WORK_BYTES = 384 * 1024**2
MAX_ROUNDS = 64


@observed('mesh.integer_components')
def triangle_components(mesh):
    """Return face labels/counts, or None for the original native fallback.

    Faces connect through an identical undirected edge, including nonmanifold
    edges. A shared vertex alone does not connect them. Minimum face IDs define
    the component order. Component areas are deliberately not computed here.
    Optional array workspace is estimated at 256 bytes per input face;
    oversized/unsupported inputs or optional allocation failure use Open3D.
    """
    faces = np.asarray(mesh.triangles)
    if (faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in 'iu'
            or len(faces) > MAX_WORK_BYTES//256
            or (faces.size and (faces.min() < 0 or faces.max() >= 2**31))):
        return None
    if not len(faces):
        return np.empty(0, np.int64), np.empty(0, np.int64)
    try:
        return _labels(faces)
    except MemoryError:
        return None


def _labels(faces):
    # Row order is (face 0 edge 0..2, face 1 edge 0..2, ...).
    keys = np.empty((len(faces), 3), np.int64)
    for column, (a, b) in enumerate(((0, 1), (1, 2), (2, 0))):
        low = np.minimum(faces[:, a], faces[:, b]).astype(np.int64)
        high = np.maximum(faces[:, a], faces[:, b]).astype(np.int64)
        keys[:, column] = (low << 32) | high
    del low, high
    keys = keys.ravel()
    order = np.argsort(keys, kind='stable')
    keys = keys[order]
    same = keys[:-1] == keys[1:]
    order //= 3
    a, b = order[:-1][same], order[1:][same]
    del keys, order, same
    parent = np.arange(len(faces), dtype=np.int64)
    for iteration in range(MAX_ROUNDS):
        left, right = parent[a], parent[b]
        active = left != right
        if not active.any():
            break
        # Contract the edge graph as well as its parent forest; already joined
        # pairs need no future work. Every hook strictly decreases its parent.
        a, b = left[active], right[active]
        del left, right, active
        high, low = np.maximum(a, b), np.minimum(a, b)
        np.minimum.at(parent, high, low)
        del high, low
        while True:
            shortened = parent[parent]
            if np.array_equal(parent, shortened):
                break
            parent = shortened
    else:
        return None
    del a, b, left, right, active
    _, labels, counts = np.unique(parent, return_inverse=True, return_counts=True)
    count('mesh.integer_component_work', faces=len(faces), rounds=iteration)
    return labels, counts
