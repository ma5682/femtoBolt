"""Bounded immutable preparation owned by one synchronous planning operation.

Every lookup fingerprints current input contents and retains type contracts.
Only deterministic preparation is shared; route decisions and work counters
remain local to their consumers. Independent reads have no inherited state.
"""
import hashlib
import json
import numpy as np
from femto.performance_detail import count
from femto.planning.evidence_preparation import _state

MAX_BYTES = 64 * 1024 * 1024
MAX_RECORD_BYTES = 8 * 1024 * 1024
_LEAF_TYPE_IDS = frozenset(map(id, (str, int, float, bool, type(None))))
_CONTAINER_TYPE_IDS = frozenset(map(id, (dict, list)))


def array_key(value):
    if (_state() is None or type(value) is not np.ndarray
            or not value.flags.c_contiguous or value.dtype.hasobject
            or value.nbytes > MAX_BYTES):
        return None
    try:
        return value.shape, value.dtype.str, hashlib.sha256(memoryview(value).cast('B')).digest()
    except MemoryError:
        return None


def _plain_json(value):
    # Plain JSON leaves distinguish bool/int and int/float in the serialized
    # key. Subclasses and tuples take the original validation path instead.
    stack = [value]
    visited = 0
    containers = set()
    while stack:
        current = stack.pop(); kind = type(current); visited += 1
        if visited > MAX_RECORD_BYTES:
            return False
        if id(kind) in _CONTAINER_TYPE_IDS:
            if id(current) in containers:return False
            containers.add(id(current))
        if kind is dict:
            if any(type(key) is not str for key in current):return False
            stack.extend(current.values())
        elif kind is list:
            for start in range(0, len(current), 32768):
                part = current[start:start+32768]
                # Never deduplicate type objects by equality: custom metaclass
                # equality/hash could otherwise hide a strict-type violation.
                kinds = set(map(id, map(type, part)))
                if not kinds <= _CONTAINER_TYPE_IDS | _LEAF_TYPE_IDS:return False
                if kinds & _CONTAINER_TYPE_IDS:
                    stack.extend(item for item in part if id(type(item)) in _CONTAINER_TYPE_IDS)
        elif id(kind) not in _LEAF_TYPE_IDS:
            return False
    return True


def record_key(value):
    if _state() is None:return None
    try:
        if not _plain_json(value):return None
        payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                             separators=(',', ':'), allow_nan=False).encode('utf-8')
    except (TypeError, ValueError, RecursionError, MemoryError):
        return None
    if len(payload) > MAX_RECORD_BYTES:return None
    try:
        return hashlib.sha256(payload).digest()
    except MemoryError:
        return None


def lookup(kind, key):
    state = _state()
    if state is None or key is None:return None
    entry = state['prepared'].get(kind)
    if entry is None:return None
    if entry[0] != key or entry[2] > MAX_BYTES:
        state['prepared'].pop(kind, None)
        return None
    count('route.task_preparation', **{kind+'_hits': 1})
    return entry[1]


def remember(kind, key, value, size):
    state = _state()
    if state is None or key is None:return
    entries = state['prepared']
    entries.pop(kind, None)
    if size <= MAX_BYTES and size+sum(entry[2] for entry in entries.values()) <= MAX_BYTES:
        entries[kind] = (key, value, size)
        count('route.task_preparation', **{kind+'_builds': 1}, bytes=size)


def frozen(value):
    return np.frombuffer(value.tobytes(order='C'), dtype=value.dtype).reshape(value.shape)


def integral_image(classes):
    key = array_key(classes)
    found = lookup('reference_integral', key)
    if found is not None:return found
    result = np.pad((classes != 1).astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    if key is not None:
        try:prepared = frozen(result)
        except MemoryError:return result
        remember('reference_integral', key, prepared, prepared.nbytes)
        return prepared
    return result


def selection_key(region, vertices, faces, functions):
    from femto.annotation import selection_intent as rules
    record = record_key({key:region.get(key) for key in
                         ('view_reference', 'selection_intent', 'selection_support')})
    geometry = (array_key(vertices), array_key(faces))
    if record is None or any(part is None for part in geometry):return None
    names=('MAX_PIXELS','MAX_RUNS','MAX_COMPONENTS','MAX_HOLES','INTENT_SCHEMA','SUPPORT_SCHEMA',
           'IMAGE_FRAME','MISSING_GEOMETRY_POLICY','SUPPORT_SOURCE','SUPPORT_EVIDENCE',
           'SUPPORT_INTERPRETATION','DERIVED_SURFACE_STATUS','DERIVED_SURFACE_REASON')
    contract={name:getattr(rules,name) for name in names}
    contract.update(SELECTION_METHODS=list(rules.SELECTION_METHODS),
                    SUPPORT_CLASSES=[list(item) for item in rules.SUPPORT_CLASSES.items()],
                    UNCONFIRMED_CLASSES=list(rules.UNCONFIRMED_CLASSES))
    contract_key=record_key(contract)
    if contract_key is None:return None
    helpers=(rules.decode_runs,rules._image_size,rules.describe_selection,rules.label_components,
             rules.intent_mask,rules.digest,rules._depth_range,rules._as_mask,rules._bbox,
             rules.pixel_indices)
    policy = (contract_key, tuple(sorted(np.geterr().items())),
              tuple(id(function) for function in (*functions,*helpers)))
    return record, geometry, policy


def selection_lookup(key):
    found = lookup('selection_assignment', key)
    if found is None:return None
    try:
        intent, support = json.loads(found[0])
        return intent, support, found[1].copy(), found[2].copy()
    except MemoryError:
        return None


def selection_remember(key, result):
    if key is None:return
    intent, support, labels, owners = result
    try:
        record = json.dumps([intent, support], ensure_ascii=False,
                            separators=(',', ':'), allow_nan=False).encode('utf-8')
        values = (record, frozen(labels), frozen(owners))
    except (TypeError, ValueError, MemoryError):
        return
    remember('selection_assignment', key, values, len(record)+labels.nbytes+owners.nbytes)
