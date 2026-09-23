"""v4.4 visit-order contracts and display-only exact spacing-point lookup."""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Any

import numpy as np


def validate_visit_rows(blocks: list[dict[str, Any]], segments: list[Any], gaps: list[Any],
                        *, faces: np.ndarray | None = None) -> None:
    """Rows retain their original identities even when the visit order goes back.

    With a bound mesh every point must use a row owned by its actual connected
    mesh fragment. Without it, face-index bounds provide only a conservative
    check; normal export/reload validation also supplies the bound mesh.
    """
    from femto.planning.surface_trajectory import face_components
    components = None if faces is None else face_components(faces)
    rows_by_block = {}
    for block in blocks:
        if block['status'] == 'unplannable':
            continue
        first, count = block['row_index_first'], block['row_count']
        cursor = first
        fragments = {}
        for fragment in block.get('fragments', []):
            if fragment.get('status') != 'planned':
                continue
            start, size = fragment.get('row_index_first'), fragment.get('row_count')
            component = fragment.get('mesh_component_id')
            if (type(start) is not int or start != cursor or type(size) is not int or size <= 0
                    or type(component) is not int or component < 0 or component in fragments):
                raise ValueError('Invalid ordered-visit fragment row identity/range')
            fragments[component] = (start, start + size, fragment)
            cursor += size
        if not fragments or count <= 0 or cursor != first + count:
            raise ValueError('Ordered-visit fragment rows do not cover the work block')
        rows_by_block[block['block_index']] = (first, first + count, fragments)
    for segment in segments:
        if not segment.points or segment.row_index != segment.points[0].get('row_index'):
            raise ValueError('Ordered-visit chain row header must identify its first point')
        if segment.block_index not in rows_by_block:
            raise ValueError('Ordered-visit chain belongs to a block without planned rows')
        low, high, fragments = rows_by_block[segment.block_index]
        chain_components = set()
        for point in segment.points:
            row, face = point.get('row_index'), point['triangle_index']
            if type(row) is not int or not low <= row < high:
                raise ValueError('Ordered-visit point row is outside its work block')
            if components is not None:
                component = int(components[face])
                owner = fragments.get(component)
                if owner is None or not owner[0] <= row < owner[1]:
                    raise ValueError('Ordered-visit point row is outside its measured mesh fragment')
                chain_components.add(component)
            else:
                if not any(start <= row < end and record['face_index_min'] <= face <= record['face_index_max']
                           for start, end, record in fragments.values()):
                    raise ValueError('Ordered-visit point row/face has no compatible mesh fragment')
        if len(chain_components) > 1:
            raise ValueError('Ordered-visit chain cannot bridge disconnected mesh fragments')
    for index, gap in enumerate(gaps):
        before, after = segments[index].points[-1], segments[index + 1].points[0]
        if (type(gap.row_index_from) is not int or type(gap.row_index_to) is not int
                or gap.row_index_from != before['row_index'] or gap.row_index_to != after['row_index']):
            raise ValueError('Ordered-visit gap rows must identify its actual endpoint rows')


def player_spacing_report(report: dict[str, Any], segments: list[list[dict[str, Any]]]) -> dict[str, Any]:
    """Copy canonical measurements and map them to unique exact playback points.

    Row identity and exact XYZ, not Euclidean proximity, supply the evidence.
    Prefer the retained scan sample over coincident connector samples. Missing
    or ambiguous evidence produces a visible explanation and no seek target.
    This mapping is never serialized back into the trajectory/spacing report.
    """
    locations = defaultdict(list)
    for chain_index, points in enumerate(segments):
        for sample_index, point in enumerate(points):
            key = (point['row_index'], tuple(point['xyz_m']))
            locations[key].append((chain_index, sample_index, point.get('role') != 'connector'))
    result = deepcopy(report)
    source_rows = report.get('scan_segment_rows', [])
    for measurement in result.get('measurements', []):
        source = measurement.get('chain_index')
        candidates = []
        if type(source) is int and 0 <= source < len(source_rows):
            candidates = locations.get((source_rows[source], tuple(measurement.get('xyz_m', []))), [])
        scans = [candidate for candidate in candidates if candidate[2]]
        candidates = scans or candidates
        if len(candidates) == 1:
            chain_index, sample_index, _ = candidates[0]
            measurement['player_location'] = dict(status='unique', chain_index=chain_index, sample_index=sample_index)
        else:
            measurement['player_location'] = dict(
                status='ambiguous' if candidates else 'unavailable',
                explanation='该测量点无法唯一对应当前播放路线，未定位；原行距测量保留，未使用最近邻代替。')
    return result
