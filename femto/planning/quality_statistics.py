"""Preparation for the per-block quality and length summaries."""
from itertools import islice
import numpy as np

from femto.annotation import selection_intent as selection
from femto.planning import task_preparation
from femto.small_vectors import row_norms


def component_support(classes, component_ids):
    """Count current evidence in raster order, returning only immutable counts."""
    key = None
    known_dtype = type(classes) is np.ndarray and classes.dtype in (np.dtype('int32'), np.dtype('uint8'))
    if (known_dtype and len(component_ids) <= selection.MAX_COMPONENTS
            and all(type(i) is int and 0 <= i < selection.MAX_COMPONENTS for i in component_ids)):
        fingerprint = task_preparation.array_key(classes)
        if fingerprint is not None:
            key = (fingerprint, tuple(component_ids), selection.label_components,
                   selection._as_mask, selection.cv2.connectedComponents, selection.MAX_PIXELS)
            found = task_preparation.lookup('quality_component_counts', key)
            if found is not None:
                return found

    labels, _ = selection.label_components(classes > 0)
    pending = int(((classes > 0) & (classes != 1)).sum())
    rows = []
    for component_id in component_ids:
        selected = labels == component_id + 1
        if known_dtype:
            values = classes[selected]
            # The histogram only reports classes 1 through 7. Bound its size
            # even if a caller presents an unexpected integer class.
            values = np.clip(values, 0, 8, out=values)
            histogram = np.bincount(values, minlength=8)
            rows.append((len(values) - int(histogram[1]), tuple(map(int, histogram[1:8]))))
        else:
            rows.append((int((selected & (classes != 1)).sum()),
                         tuple(int((selected & (classes == code)).sum()) for code in range(1, 8))))
    result = pending, tuple(rows)
    task_preparation.remember('quality_component_counts', key, result, 1024 + 512*len(rows))
    return result


def edge_lengths(points):
    """Keep scalar dtype semantics for legacy inputs and the original addition order."""
    # Limit the temporary XYZ matrix even for a single very long route.
    for start in range(0, len(points) - 1, 1024):
        group = points[start:start + 1025]
        coordinates = [point['xyz_m'] for point in group]
        if (len(group) >= 16 and all(type(xyz) is list and len(xyz) == 3
                and all(type(value) is float for value in xyz) for xyz in coordinates)):
            xyz = np.asarray(coordinates)
            yield from row_norms(xyz[1:] - xyz[:-1]).tolist()
        else:
            for a, b in zip(coordinates, islice(coordinates, 1, None)):
                yield float(np.linalg.norm(np.asarray(a) - b))
