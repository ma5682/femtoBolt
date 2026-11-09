"""Bounded parallel image arithmetic; report decisions remain on the caller."""
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from contextlib import contextmanager
import os

import cv2
import numpy as np

MIN_PARALLEL_PIXELS = 262144
MAX_WORKERS = 4


def _component(classes, code):
    total, labels, stats, _ = cv2.connectedComponentsWithStats(
        (classes == code).astype(np.uint8), connectivity=4)
    present, first = np.unique(labels[labels > 0], return_index=True)
    return code, total-1, stats, present[np.argsort(first, kind='stable')]


@contextmanager
def component_rows(classes, check):
    """Yield six class results in original order, using a private pixel snapshot.

    Workers perform only OpenCV/NumPy arithmetic. Cancellation callbacks, the
    aggregate fragmentation budget, errors and record numbering stay ordered
    on the calling thread. No pool or result survives this call.
    """
    def serial():
        for code in range(2, 8):
            check()
            yield _component(classes, code)
    workers = min(MAX_WORKERS, os.cpu_count() or 1)
    if (workers < 2 or type(classes) is not np.ndarray or classes.ndim != 2
            or classes.dtype.kind not in 'uib' or classes.size < MIN_PARALLEL_PIXELS):
        yield serial()
        return
    check()
    try:
        snapshot = np.frombuffer(classes.tobytes(order='C'), dtype=classes.dtype).reshape(classes.shape)
    except MemoryError:
        yield serial()
        return
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='quality-components') as pool:
        futures = [pool.submit(_component, snapshot, code) for code in range(2, 8)]
        def rows():
            for future in futures:
                while True:
                    check()
                    try:
                        value = future.result(timeout=.02)
                        break
                    except FutureTimeout:
                        # A TimeoutError raised by the worker must propagate.
                        if future.done():
                            value = future.result()
                            break
                yield value
        try:
            yield rows()
        finally:
            for future in futures:
                future.cancel()
