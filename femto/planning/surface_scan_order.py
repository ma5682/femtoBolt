"""Deterministic scan-strip visits; geometry and the accepted scan samples stay fixed.

Adjacent physical slices continue a strip only through a mutually unique positive
overlap of their e1 intervals. Splits, merges and missing slices end strips. This
is an ordering heuristic, not a connectivity or collision certificate: the caller
must still route every selected transition on the measured surface.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Any, Callable

import numpy as np


def _strips(records: list[dict[str, Any]], intervals: list[tuple[float, float]],
            check: Callable[[], None]) -> list[list[int]]:
    rows: dict[int, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        check()
        rows[int(record['slice_index'])].append(index)
    following: dict[int, int] = {}
    previous: dict[int, int] = {}
    for row in sorted(rows):
        check()
        if row + 1 not in rows:
            continue
        events = []
        for side, indices in enumerate((rows[row], rows[row + 1])):
            for index in indices:
                check()
                low, high = intervals[index]
                if low < high:
                    # Ends precede starts at equal coordinates: touching alone
                    # is not a positive overlap and cannot establish a strip.
                    events.extend(((low, 1, side, index), (high, 0, side, index)))
        degree: dict[int, int] = defaultdict(int)
        neighbor: dict[int, int] = {}
        active: list[set[int]] = [set(), set()]
        for _, starting, side, index in sorted(events):
            check()
            if not starting:
                active[side].discard(index)
                continue
            for other in sorted(active[1 - side]):
                check()
                degree[index] += 1
                degree[other] += 1
                neighbor[index] = other
                neighbor[other] = index
            active[side].add(index)
        for index in rows[row]:
            check()
            if degree[index] == 1 and degree[neighbor[index]] == 1:
                other = neighbor[index]
                following[index] = other
                previous[other] = index
    strips = []
    for index in range(len(records)):
        check()
        if index in previous:
            continue
        strip = [index]
        while strip[-1] in following:
            check()
            strip.append(following[strip[-1]])
        strips.append(strip)
    return strips


def order_scan_records(records: list[dict[str, Any]], frame: dict[str, Any],
                       check: Callable[[], None]) -> list[dict[str, Any]]:
    """Visit each original scan once, keeping the original first scan and direction.

    Within a strip, alternate e1 direction on successive physical rows. Subsequent
    strips offer both traversal ends and both initial scan directions; choose the
    nearest entry in XYZ, then stable input indices. No route search, resampling,
    geometric threshold, mask reinterpretation or input mutation is performed.
    Returned records, point dictionaries and their nested values are independent.
    """
    check()
    if not records:
        return []
    origin = np.asarray(frame['origin'], dtype=float)
    e1 = np.asarray(frame['e1'], dtype=float)
    intervals = []
    increasing = []
    endpoints = []
    for record in records:
        check()
        points = np.asarray([point['xyz_m'] for point in record['points']], dtype=float)
        coordinates = (points - origin) @ e1
        intervals.append((float(coordinates.min()), float(coordinates.max())))
        increasing.append(bool(coordinates[0] <= coordinates[-1]))
        endpoints.append((points[0], points[-1]))
    strips = _strips(records, intervals, check)
    first_strip = next(index for index, strip in enumerate(strips) if 0 in strip)
    initial = strips.pop(first_strip)
    position = initial.index(0)
    if position == len(initial) - 1:
        initial = list(reversed(initial))
    elif position:
        # Canonical records start at a physical-row extreme. Keeping this case
        # well-defined also protects the first-scan contract for external callers.
        strips.append(initial[:position])
        initial = initial[position:]
    result: list[dict[str, Any]] = []

    def emit(indices: list[int], first_increasing: bool) -> np.ndarray:
        for offset, index in enumerate(indices):
            check()
            record = deepcopy(records[index])
            direction = first_increasing ^ bool(offset % 2)
            if increasing[index] != direction:
                record['points'].reverse()
            result.append(record)
        return np.asarray(result[-1]['points'][-1]['xyz_m'], dtype=float)

    current = emit(initial, increasing[0])
    # Precompute just the four entry choices, not four copies of every strip.
    variants = []
    for strip_id, strip in enumerate(strips):
        check()
        for reverse_rows in (False, True):
            first = strip[-1] if reverse_rows else strip[0]
            for first_increasing in (True, False):
                entry = endpoints[first][0 if increasing[first] == first_increasing else 1]
                variants.append((strip_id, min(strip), reverse_rows, first_increasing, entry))
    remaining = set(range(len(strips)))
    while remaining:
        check()
        candidates = []
        for variant in variants:
            check()
            if variant[0] in remaining:
                candidates.append(variant)
        distances = np.linalg.norm(np.asarray([variant[4] for variant in candidates]) - current, axis=1)
        chosen = min(range(len(candidates)), key=lambda index: (
            float(distances[index]), candidates[index][1], candidates[index][2],
            not candidates[index][3]))
        strip_id, _, reverse_rows, first_increasing, _ = candidates[chosen]
        indices = list(reversed(strips[strip_id])) if reverse_rows else strips[strip_id]
        current = emit(indices, first_increasing)
        remaining.remove(strip_id)
    return result
