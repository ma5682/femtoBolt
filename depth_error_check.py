"""Depth integrity and local repair consistency, in millimetres (not ground truth)."""
from __future__ import annotations

import numpy as np


def _error_statistics(errors: np.ndarray) -> dict:
    errors = np.asarray(errors, dtype=np.float64)
    bins = [0, 1, 5, 10, 20, 30, 50, 51, 100, 1000, 65536]
    counts, _ = np.histogram(errors, bins=bins)
    return dict(sample_count=int(errors.size),
                mean_absolute_error_mm=float(errors.mean()) if errors.size else None,
                max_error_mm=float(errors.max()) if errors.size else None,
                depth_difference_histogram=dict(bin_edges_mm=bins, counts=counts.tolist(),
                                                convention='absolute difference; left inclusive, right exclusive except last'))


def depth_error_check(original_depth: np.ndarray, filled_depth: np.ndarray,
                      min_range_mm: float, max_range_mm: float, *, max_error_mm: float = 50.0):
    """Return rejected repair mask and report; zeros/out-of-range are never truth.

    Original valid pixels are compared at identical coordinates. At missing
    pixels, report a separate proxy against original valid 8-neighbors; reject
    if any neighbor differs by >50 mm or fewer than five neighbors exist.
    """
    from depth_quality import invalid_depth_mask
    original, filled = np.asarray(original_depth), np.asarray(filled_depth)
    if original.dtype != np.uint16 or filled.dtype != np.uint16 or original.shape != filled.shape:
        raise ValueError('Error check requires matching uint16 millimetre images')
    if not 0 < max_error_mm <= 50:
        raise ValueError('Error limit must be positive and at most 50 mm')
    valid = ~invalid_depth_mask(original, min_range_mm, max_range_mm)
    filled_valid = ~invalid_depth_mask(filled, min_range_mm, max_range_mm)
    differences = np.abs(filled[valid].astype(np.float64) - original[valid].astype(np.float64))
    candidates = ~valid & filled_valid
    rejected = np.zeros(original.shape, dtype=bool)
    deviations = []
    unsupported = 0
    for y, x in zip(*np.nonzero(candidates)):
        y0, y1 = max(0,y-1), min(original.shape[0],y+2)
        x0, x1 = max(0,x-1), min(original.shape[1],x+2)
        neighbors = original[y0:y1,x0:x1][valid[y0:y1,x0:x1]].astype(np.float64)
        if neighbors.size:
            delta = float(np.max(np.abs(neighbors - float(filled[y,x]))))
            deviations.append(delta)
        else:
            delta = float('inf')
        is_border = y == 0 or x == 0 or y == original.shape[0]-1 or x == original.shape[1]-1
        if neighbors.size < 5 or is_border:
            unsupported += 1
        rejected[y,x] = is_border or neighbors.size < 5 or delta > max_error_mm
    changed = int(np.count_nonzero(differences))
    report = dict(
        schema='femto-depth-error-check-v1', threshold_mm=max_error_mm,
        original_valid_comparison={**_error_statistics(differences), 'changed_pixels':changed},
        repair_neighbor_consistency={**_error_statistics(np.asarray(deviations)),
                                    'reference':'maximum absolute difference to original valid 8-neighbors per repaired pixel',
                                    'candidate_pixels':int(candidates.sum()), 'unsupported_pixels':unsupported,
                                    'rejected_pixels':int(rejected.sum())},
        original_integrity_passed=changed == 0,
        passed=changed == 0 and not rejected.any(),
        repair_ground_truth_error_mm=None,
        limitation='Missing pixels have no measured ground truth. Neighbor consistency is a proxy, not reconstruction accuracy.')
    return rejected, report
