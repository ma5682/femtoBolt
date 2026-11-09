"""One generation's bounded immutable RLE decompression, never route verdicts."""
from contextlib import contextmanager
from contextvars import ContextVar
from operator import is_
from threading import get_ident
import numpy as np
from femto.performance_detail import count
from femto.plain_copy import plain_integers as _plain_ints

MAX_BYTES = 16 * 1024 * 1024
_active = ContextVar('route_evidence_decompression', default=None)


@contextmanager
def operation():
    state = _state()
    if state is not None:
        yield
        return
    state = dict(active=True, thread=get_ident(), entry=None, prepared={}, digest=None)
    token = _active.set(state)
    try:
        yield
    finally:
        state['active'] = False
        state['entry'] = None
        state['digest'] = None
        state['prepared'].clear()
        _active.reset(token)


def _state():
    state = _active.get()
    return state if state is not None and state['active'] and state['thread'] == get_ident() else None


def _ordinary(record):
    # JSON hashing equates list/tuple and integer subclasses in some cases;
    # retain the decoder's strict Python integer/container contract on hits.
    return (type(record) is dict and type(record.get('sha256')) is str
            and (type(record.get('image_size')) is list or type(record.get('image_size')) is tuple)
            and len(record['image_size']) == 2 and all(type(v) is int for v in record['image_size'])
            and all(type(record.get(key)) is list and _plain_ints(record[key])
                    for key in ('classes_runs', 'residual_um_runs')))


def _restore(entry):
    return tuple(np.frombuffer(payload, dtype=np.int32).reshape(shape).copy()
                 for shape, payload in entry[1])


def lookup(record):
    """Caller must have recomputed and matched the complete record SHA first."""
    state = _state()
    if state is None or state['entry'] is None:
        return None
    entry = state['entry']
    from femto.annotation import selection_intent
    policy = (selection_intent.MAX_PIXELS, selection_intent.MAX_RUNS)
    if (type(record) is not dict or type(record.get('sha256')) is not str
            or entry[0] != record['sha256'] or entry[2] != policy
            or sum(len(payload) for _, payload in entry[1]) > MAX_BYTES):
        state['entry'] = None
        return None
    if not _ordinary(record):
        return None
    try:
        arrays = _restore(entry)
    except MemoryError:
        state['entry'] = None
        return None
    count('route.evidence_preparation', hits=1)
    return arrays


_RUN_KEYS = ('classes_runs', 'residual_um_runs')


def record_digest(record, compute, encode):
    """compute(record) unless its canonical bytes provably equal the last hashed record.

    Small fields are compared as their own canonical bytes. Each run list is
    compared element-by-element by identity against a private copy taken at
    hashing time, so every element is the same exact int object (no True/1.0
    aliasing) and the full canonical JSON, hence the SHA, is byte-identical.
    Any difference, including in-place mutation, recomputes from content.
    """
    state = _state()
    if (state is None or type(record) is not dict
            or any(type(record.get(key)) is not list for key in _RUN_KEYS)):
        return compute(record)
    runs = tuple(record[key] for key in _RUN_KEYS)
    if sum(len(values) for values in runs) * 8 > MAX_BYTES:
        return compute(record)
    small = encode({k: v for k, v in record.items() if k != 'sha256' and k not in _RUN_KEYS})
    found = state['digest']
    if (found is not None and found[0] == small
            and all(len(values) == len(copy) and all(map(is_, values, copy))
                    for values, copy in zip(runs, found[1]))):
        count('route.evidence_digest', hits=1)
        return found[2]
    state['digest'] = None
    try:
        copies = tuple(list(values) for values in runs)
    except MemoryError:
        return compute(record)
    result = compute(record)
    state['digest'] = (small, copies, result)
    return result


def _snapshot(arrays):
    return tuple((value.shape, value.tobytes(order='C')) for value in arrays)


def remember(record, arrays):
    state = _state()
    if state is None or state['entry'] is not None or not _ordinary(record):
        return
    if (any(type(value) is not np.ndarray or value.dtype != np.dtype('int32') for value in arrays)
            or sum(value.nbytes for value in arrays) > MAX_BYTES):
        return
    try:
        from femto.annotation import selection_intent
        entry = (record['sha256'], _snapshot(arrays),
                 (selection_intent.MAX_PIXELS, selection_intent.MAX_RUNS))
    except MemoryError:
        return
    state['entry'] = entry
    count('route.evidence_preparation', snapshots=1, bytes=sum(value.nbytes for value in arrays))
