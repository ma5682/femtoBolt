"""Bounded GC-tracked object/root evidence for known CPython arena addresses.

Diagnostic use only: never dereference arbitrary native addresses, force GC,
retain sampled objects between calls, or infer object identity from reused IDs.
Untracked strings, tuples/dicts, NumPy buffers and native objects are incomplete.
"""
from bisect import bisect_right
from collections import Counter
import gc
import sys
import time
import types


def type_name(value):
    kind = type(value)
    module = type.__getattribute__(kind, '__module__')
    name = type.__getattribute__(kind, '__qualname__')
    return (module if type(module) is str else '<non_string_module>') + '.' + name


def shallow_size(value):
    # Do not invoke application-defined __sizeof__ while inspecting all types.
    if type(value) in (str, bytes, bytearray, tuple, list, dict, set, frozenset):
        return sys.getsizeof(value)
    try:
        return object.__sizeof__(value)
    except TypeError:
        return 0


def arena_index(address, ranges, bases):
    index = bisect_right(bases, address)-1
    return index if index >= 0 and address < ranges[index][0]+ranges[index][1] else None


def census(arena_ranges, *, sample_limit=12, max_depth=3, max_referrers=48, focus_bases=None):
    """Return scalar descriptions; root paths show reachability, not ownership.

    gc.get_objects temporarily retains all tracked objects. That list and this
    observer's traversal containers are excluded from reported referrer paths.
    Sampling can alter natural GC timing; compare a separate no-census control.
    """
    if not gc.isenabled():
        raise RuntimeError('Object census requires natural enabled GC')
    if not 0 <= sample_limit <= 32 or not 1 <= max_depth <= 4 or not 1 <= max_referrers <= 128:
        raise ValueError('Diagnostic bounds exceeded')
    started = time.perf_counter()
    before = gc.get_stats()
    ranges = sorted((int(base), int(size)) for base, size in arena_ranges)
    if any(base <= 0 or size <= 0 for base, size in ranges):
        raise ValueError('Invalid arena range')
    if any(a+size > b for (a, size), (b, _) in zip(ranges, ranges[1:])):
        raise ValueError('Overlapping arena ranges')
    bases = [base for base, _ in ranges]
    focus = set(bases) if focus_bases is None else set(focus_bases)
    objects = gc.get_objects()
    counts, sizes, arena_counts = Counter(), Counter(), Counter()
    samples, sampled_types = [], set()
    for value in objects:
        index = arena_index(id(value), ranges, bases)
        if index is None:
            continue
        label = type_name(value)
        counts[label] += 1
        size = shallow_size(value)
        sizes[label] += size
        arena_counts[index] += 1
        if (ranges[index][0] in focus and len(samples) < sample_limit
                and label not in sampled_types and size <= 512):
            samples.append(value)
            sampled_types.add(label)
    # Do not call getattr/repr on arbitrary application objects or expose values.
    module_roots = {}
    for module_name, module in tuple(sys.modules.items()):
        if type(module) is not types.ModuleType:
            continue
        namespace = vars(module)
        module_roots.setdefault(id(namespace), []).append(module_name + '.__dict__')
    excluded = {id(objects), id(samples), id(module_roots), id(counts), id(sizes), id(arena_counts)}
    paths = []
    for value in samples:
        pending = [(value, [type_name(value)], 0)]
        seen = set()
        excluded.add(id(pending))
        found, visited, truncated = [], 0, False
        while pending and visited < 128:
            target, path, depth = pending.pop()
            identity = id(target)
            if identity in seen:
                continue
            seen.add(identity)
            visited += 1
            if identity in module_roots:
                found.extend(path+[label] for label in module_roots[identity][:3])
                continue
            if depth >= max_depth:
                continue
            references = gc.get_referrers(target)
            excluded.add(id(references))
            if len(references) > max_referrers:
                truncated = True
            for parent in references[:max_referrers]:
                if id(parent) in excluded or type(parent) is types.FrameType:
                    continue
                # Traversal queue tuples are observer references, not roots.
                if type(parent) is tuple and len(parent) == 3 and parent[0] is target and parent[1] is path:
                    continue
                entry = (parent, path+[type_name(parent)], depth+1)
                excluded.add(id(entry))
                pending.append(entry)
            excluded.remove(id(references))
            del references
        paths.append(dict(type=type_name(value), address=hex(id(value)),
                          arena_base=hex(ranges[arena_index(id(value), ranges, bases)][0]),
                          module_paths=found[:8], visited=visited,
                          bounded_incomplete=truncated or bool(pending)))
        excluded.remove(id(pending))
        pending.clear()
    return dict(schema='femto-arena-tracked-roots-v1',
                gc_tracked_objects=len(objects), known_arenas=len(ranges),
                root_sampling_focus_arenas=len(focus.intersection(bases)),
                arenas_with_tracked_objects=len(arena_counts),
                tracked_objects_in_known_arenas=sum(counts.values()),
                tracked_shallow_bytes=sum(sizes.values()),
                types=[dict(type=label, count=count, shallow_bytes=sizes[label])
                       for label, count in counts.most_common(40)],
                samples=paths, seconds=time.perf_counter()-started,
                natural_collections_during_census=[b['collections']-a['collections']
                                                 for a, b in zip(before, gc.get_stats())],
                limitations=['GC-tracked objects only; untracked blocks and native buffers not enumerated.',
                             'Shallow sizes use safe builtin/base headers; extension payload bytes are incomplete.',
                             'Root paths are bounded reachability samples, not exclusive allocation ownership.',
                             'Addresses can be reused; cross-cycle matching does not prove survival.',
                             'get_objects temporarily retains objects and can affect natural GC; use control run.'])
