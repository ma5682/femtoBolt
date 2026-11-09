"""Integer edge-row grouping with the original lexicographic result contract."""
import numpy as np


def unique_edge_rows(edges):
    """Return sorted unique endpoint pairs and multiplicities, preserving dtype.

    Callers choose directed or undirected input rows. Unsupported ranges/types
    retain NumPy's original row grouping, including its exceptional behaviour.
    """
    if (edges.ndim != 2 or edges.shape[1] != 2 or edges.dtype.kind not in 'iu'
            or (edges.size and (edges.min() < 0 or edges.max() >= 2**31))):
        return np.unique(edges, axis=0, return_counts=True)
    keys = (edges[:, 0].astype(np.int64) << 32) | edges[:, 1].astype(np.int64)
    values, counts = np.unique(keys, return_counts=True)
    unique = np.column_stack((values >> 32, values & ((1 << 32)-1))).astype(edges.dtype, copy=False)
    return unique, counts
