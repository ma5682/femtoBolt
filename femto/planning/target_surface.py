"""Original requested surface inventory, built before route quality filtering.

Pixel cells clip the full canonical source mesh, including boundary faces whose
centroids were not selected by the legacy route planner. Every located child
keeps its original face, barycentric vertices, pixel parent and local route-face
mapping. Unknown pixels have no invented area. This is not a coverage certificate.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import lru_cache
import json
from pathlib import Path
import time
from typing import Any, Callable
import zipfile

import numpy as np

from femto.annotation.selection_intent import intent_mask
from femto.geometry.pixel_clip import (EPS, area2, clip_triangle_planes, covers_in_front,
                                       interior_overlap, pixel_planes, project, shared_edge_separates)
from femto.geometry.pixel_clip_batch import MAX_BATCH, clip_triangles_planes, project_clipped_polygons
from femto.geometry.triangle_separation import (PreparedTriangle, prepared_overlap,
                                                shared_edge_separates as prepared_shared_edge_separates)
from femto.planning import planning_revision as revision_api
from femto.operation_diagnostics import diagnosed

SCHEMA = 'femto-original-target-surface-v1'
PAGED_SCHEMA = 'femto-original-target-surface-paged-v1'
MANIFEST = 'target_surface.json'
ASSET = 'target_surface.npz'
MAX_ASSET_BYTES = 512 * 1024**2
PIXEL_STATES = {0: 'unselected', 1: 'located', 2: 'missing_reference_depth',
                3: 'depth_outside_range', 4: 'no_source_pixel', 5: 'no_mesh_surface',
                6: 'depth_conflict', 7: 'occlusion_or_multiple_surface_ambiguity',
                8: 'numerically_unconfirmed', 9: 'budget_unprocessed',
                10: 'partly_located_pixel_cell'}
QUALITY_STATES = {0: 'eligible_in_current_route_domain', 1: 'excluded_by_quality_domain',
                  2: 'outside_legacy_centroid_region', 3: 'quality_pending'}
_DTYPES = dict(vertices='<f8', faces='<i8', region_faces='<i8', depth='<u2', image_valid='|b1',
               mask='|b1', quality_classes='|u1', face_quality='|u1', pixel_status='|u1',
               source_face='<i8', local_face='<i8', pixel_index='<i8', parent_id='<i8',
               child_index='|u1', barycentric='<f8', area_m2='<f8', quality_status='|u1',
               duplicate_faces='<i8')
_LEAVES = ('source_face', 'local_face', 'pixel_index', 'parent_id', 'child_index',
           'barycentric', 'area_m2', 'quality_status')


@dataclass(frozen=True)
class TargetBudget:
    max_candidates: int = 1_000_000
    max_units: int = 200_000
    max_pair_checks: int = 500_000
    seconds: float = 45.

    def checked(self) -> TargetBudget:
        limits = TargetBudget()
        for name in ('max_candidates', 'max_units', 'max_pair_checks'):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value <= getattr(limits, name):
                raise ValueError('目标台账预算无效：' + name)
        if type(self.seconds) not in (int, float) or not np.isfinite(self.seconds) or not 0 <= self.seconds <= limits.seconds:
            raise ValueError('目标台账时间预算无效')
        return self


@dataclass(frozen=True)
class TargetLedger:
    metadata: dict
    arrays: dict[str, np.ndarray]

    @property
    def summary(self) -> dict:
        return self.metadata['summary']


def _summary(arrays: dict) -> dict:
    status = arrays['pixel_status']
    selected = arrays['mask']
    unknown = selected & (status != 1)
    area = arrays['area_m2']
    quality = arrays['quality_status']
    return dict(selected_pixel_count=int(selected.sum()), located_pixel_count=int((status == 1).sum()),
                unlocated_pixel_count=int(unknown.sum()),
                pixels_with_located_parts=int(len(np.unique(arrays['pixel_index']))),
                leaf_count=len(area), located_area_m2=float(area.sum()),
                quality_eligible_area_m2=float(area[quality == 0].sum()),
                quality_excluded_area_m2=float(area[quality != 0].sum()),
                outside_route_region_area_m2=float(area[quality == 2].sum()),
                unlocated_area_m2=None, total_target_area_m2=None if unknown.any() else float(area.sum()),
                unknown_by_reason={reason: int((status == code).sum()) for code, reason in PIXEL_STATES.items() if code > 1},
                coverage_status='not_evaluated', physical_accuracy_verified=False, execution_ready=False)


def _seal(arrays: dict, revision: dict, budget: TargetBudget, metrics: dict,
          evidence_sha: str, domain_sha: str | None, *, construction_pages=None) -> TargetLedger:
    arrays = {name: np.ascontiguousarray(value, dtype=_DTYPES[name]) for name, value in arrays.items()}
    if sum(a.nbytes for a in arrays.values()) > MAX_ASSET_BYTES:
        raise ValueError('目标台账数组超过 512 MiB 预算')
    for array in arrays.values():
        array.setflags(write=False)
    metadata = dict(schema=SCHEMA, asset=ASSET, coordinate_frame='world_rdf', unit='metre',
                    target_id=revision['target_id'], computation_sha256=revision['sha256'],
                    quality_evidence_sha256=evidence_sha, effective_domain_sha256=domain_sha,
                    source_face_space='original_full_source_mesh', local_face_space='canonical_original_region_mesh',
                    local_face_missing=-1, parent_rule='pixel_linear_index * source_face_count + original_face_id',
                    child_rule='deterministic_clipped_polygon_fan_index',
                    pixel_cells=revision_api.NUMERICS['pixel_cells'], boundary_owner='floor(uv + 0.5)',
                    area_semantics=revision_api.NUMERICS['area'],
                    unknown_area_policy='unknown_pixels_have_no_invented_3d_area',
                    pixel_states={str(k): v for k, v in PIXEL_STATES.items()},
                    quality_states={str(k): v for k, v in QUALITY_STATES.items()},
                    target_completion='unverified', coverage_certificate=None,
                    budget=asdict(budget), metrics=metrics, summary=_summary(arrays),
                    arrays={name: revision_api.array_record(value) for name, value in sorted(arrays.items())},
                    execution_ready=False, physical_accuracy_verified=False)
    if construction_pages is not None:
        metadata['schema']=PAGED_SCHEMA
        metadata['construction_pages']=deepcopy(construction_pages)
    metadata['sha256'] = revision_api.digest(metadata)
    return TargetLedger(metadata, arrays)


def _empty_leaves() -> dict:
    return {name: np.empty((0, 3, 3) if name == 'barycentric' else (0,), dtype=_DTYPES[name]) for name in _LEAVES}


def _quality_values(arrays: dict) -> np.ndarray:
    local = arrays['local_face']
    result = np.full(len(local), 2, np.uint8)
    inside = local >= 0
    result[inside] = np.where(arrays['face_quality'][local[inside]] == 0, 0, 1)
    result[arrays['quality_classes'].ravel()[arrays['pixel_index']] != 1] = 3
    if np.any(inside):
        result[inside & (arrays['face_quality'][np.maximum(local, 0)] == 255)] = 3
    return result


def _source_separation(triangles: np.ndarray, intrinsic: np.ndarray,
                       source_vertex_ids: np.ndarray | None = None, *,
                       source_projection: tuple[np.ndarray, np.ndarray] | None = None) -> Callable:
    """Reuse only source-face separation within this one immutable construction.

    Pixel-polygon overlap, visibility and depth still need their original checks.
    Every pixel pair is charged even when its source separation was seen before.
    source_projection, when supplied, is this construction's immutable pair of
    sorted original face IDs and projected vertices, never a cross-build cache.
    """
    # IDs identify equal coordinate values, not source mesh vertex indices:
    # separately indexed but coincident vertices remain shared in the predicate.
    if source_vertex_ids is None and np.isfinite(triangles).all():
        _, source_vertex_ids = np.unique(triangles.reshape(-1, 3), axis=0, return_inverse=True)
        source_vertex_ids = source_vertex_ids.reshape(-1, 3)

    @lru_cache(maxsize=4096)
    def prepared(face: int) -> PreparedTriangle:
        uv = None
        if source_projection is not None:
            source_ids, source_uv = source_projection
            row = np.searchsorted(source_ids, face)
            # The index excludes duplicate/invalid faces, so its row number is
            # not an original face ID. Unindexed faces retain the old path.
            if row < len(source_ids) and source_ids[row] == face:
                uv = source_uv[row]
        if uv is None:
            uv = project(triangles[face], intrinsic)
        ids = source_vertex_ids[face] if source_vertex_ids is not None else (-1, -2, -3)
        return PreparedTriangle(uv, ids)

    @lru_cache(maxsize=4096)
    def separated(left: int, right: int) -> bool:
        if source_vertex_ids is None:
            # Non-finite standalone inputs retain the original equality and
            # exception semantics; production validates finite source geometry.
            return (shared_edge_separates(triangles[left], triangles[right], intrinsic)
                    or interior_overlap(prepared(left).points, prepared(right).points) is False)
        a, b = prepared(left), prepared(right)
        return prepared_shared_edge_separates(a, b) or prepared_overlap(a, b) is False
    return separated


def _candidate_groups(candidate_pixels: list, candidate_faces: list) -> tuple:
    """Group row-major pixels, preserving the original face order within each.

    The stable sort already orders pixels; detecting adjacent changes avoids
    np.unique's second sort and its index/permutation buffers.
    """
    pixel_ids = np.concatenate(candidate_pixels)
    face_ids = np.concatenate(candidate_faces)
    order = np.argsort(pixel_ids, kind='stable')
    pixel_ids = pixel_ids[order]
    starts = np.r_[0, np.flatnonzero(pixel_ids[1:] != pixel_ids[:-1]) + 1]
    return pixel_ids[starts], face_ids[order], starts, np.r_[starts[1:], len(pixel_ids)]


def _triangle_area(origin: np.ndarray, first: np.ndarray, second: np.ndarray) -> float:
    """The original float64 cross/dot norm, without generic-axis dispatch."""
    a, b = first-origin, second-origin
    cross = np.array((a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]))
    return float(np.sqrt(cross.dot(cross)) / 2)


class _PixelClipBatches:
    """Keep only the next bounded group of candidate clips, never decisions.

    The caller still visits/checks/charges every pixel and face in its original
    order. At most 63 future candidate clips may be prepared speculatively; an
    explicit replay frontier bounds even that work. No cross-build state lives
    here and no future clip is counted until the original loop consumes it.
    """
    def __init__(self, triangles, k, width, groups, face_ids, ends, end_group):
        self.triangles, self.k, self.width = triangles, k, width
        self.groups, self.face_ids, self.ends = groups, face_ids, ends
        self.limit = int(ends[end_group-1]) if end_group else 0
        self.first, self.cached = 0, []
        self.projected = None

    def get(self, offset):
        if not self.first <= offset < self.first + len(self.cached):
            stop = min(offset + MAX_BATCH, self.limit)
            if offset >= stop:
                raise ValueError('Pixel clipping request exceeds construction frontier')
            pixels = self.groups[np.searchsorted(self.ends, np.arange(offset, stop), side='right')]
            y, x = pixels // self.width, pixels % self.width
            planes = np.zeros((stop-offset, 4, 3), dtype=np.float64)
            planes[:, 0, 0] = self.k[0, 0]; planes[:, 1, 0] = -self.k[0, 0]
            planes[:, 2, 1] = self.k[1, 1]; planes[:, 3, 1] = -self.k[1, 1]
            planes[:, 0, 2] = self.k[0, 2] - (x-.5)
            planes[:, 1, 2] = x+.5 - self.k[0, 2]
            planes[:, 2, 2] = self.k[1, 2] - (y-.5)
            planes[:, 3, 2] = y+.5 - self.k[1, 2]
            self.cached = clip_triangles_planes(self.triangles[self.face_ids[offset:stop]], planes,
                                                 scalar_clip=clip_triangle_planes)
            self.first = offset
            self.projected = None
        return self.cached[offset-self.first]

    def polygon(self, offset):
        self.get(offset)
        if self.projected is None:
            self.projected = project_clipped_polygons(self.cached,
                self.triangles[self.face_ids[self.first:self.first+len(self.cached)]], self.k)
        return self.projected[offset-self.first]


def _clip_pixel_polygons(
    face_ids: np.ndarray, starts: np.ndarray, ends: np.ndarray, index: int,
    triangles: np.ndarray, k: np.ndarray, x: int, y: int, check: Callable, *, batches=None,
) -> tuple[list, int, int, int, int, int]:
    polygons = []
    planes = pixel_planes(k, x, y) if batches is None else None
    reason, clip_count, exact_count, pair_count, occluded = 5, 0, 0, 0, 0
    pixel_faces = face_ids[starts[index]:ends[index]]
    for ordinal, face in enumerate(pixel_faces):
        if clip_count % 64 == 0:
            check()
        tri = triangles[face]
        bary, exact = (clip_triangle_planes(tri, planes) if batches is None
                       else batches.get(int(starts[index])+ordinal))
        clip_count += 1; exact_count += int(exact)
        if len(bary) < 3:
            continue
        if batches is None:
            xyz = bary @ tri
            projected = project(xyz, k)
            size = area2(projected)
        else:
            xyz, projected, size, error_scale, finite = batches.polygon(int(starts[index])+ordinal)
        if size == 0:
            continue
        # Small polygons are not automatically uncertain. Assess rounding
        # relative to their local edges; a fixed pixel-area epsilon would
        # discard valid tiny pieces on otherwise ordinary grid boundaries.
        if batches is None:
            edges = np.empty_like(projected)
            edges[:-1] = projected[1:]-projected[:-1]
            edges[-1] = projected[0]-projected[-1]
            error_scale = EPS * max(1., float(np.abs(projected).max())) * float(np.abs(edges).sum())
            finite = np.isfinite(xyz).all() and np.isfinite(projected).all()
        if not finite or size < error_scale:
            reason = 8
            break
        polygons.append((int(face), bary, xyz, projected))
    return polygons, reason, clip_count, exact_count, pair_count, occluded


def _resolve_pixel_occlusion(
    polygons: list, reason: int, pair_count: int, occluded: int, metrics: dict,
    budget: TargetBudget, triangles: np.ndarray, k: np.ndarray,
    source_separated: Callable, check: Callable,
) -> tuple[set, bool, int, int, int]:
    discarded = set()
    exhausted = False
    if reason != 8:
        for a in range(len(polygons)):
            if a in discarded:
                continue
            for b in range(a+1, len(polygons)):
                if b in discarded:
                    continue
                pair_count += 1
                if pair_count % 64 == 0:
                    check()
                if metrics['pair_checks']+pair_count > budget.max_pair_checks:
                    exhausted = True
                    break
                fa, _, xa, ua = polygons[a]; fb, _, xb, ub = polygons[b]
                # Source-triangle separation also handles faces touching
                # at just one original vertex, before any clipping rounding.
                if source_separated(fa, fb):
                    continue
                overlaps = interior_overlap(ua, ub)
                if overlaps is False:
                    continue
                if overlaps and covers_in_front(ua, triangles[fa], ub, xb, k):
                    discarded.add(b); occluded += 1
                elif overlaps and covers_in_front(ub, triangles[fb], ua, xa, k):
                    discarded.add(a); occluded += 1
                    break
                else:
                    reason = 8 if overlaps is None else 7
                    break
            if exhausted or reason in (7, 8):
                break
    return discarded, exhausted, reason, pair_count, occluded


def _pixel_target_children(
    polygons: list, discarded: set, reason: int, limits: dict, depth: np.ndarray,
    y: int, x: int, gate_m: float, local_lookup: np.ndarray, pixel: int,
    faces: np.ndarray, partition=None,
) -> tuple[list, int]:
    visible = [p for i, p in enumerate(polygons) if i not in discarded]
    if reason not in (7, 8) and visible:
        if any(((p[2][:, 2] < limits['minimum_m']) | (p[2][:, 2] > limits['maximum_m'])).any() for p in visible):
            reason = 3
        elif any((np.abs(p[2][:, 2]-depth[y, x]) > gate_m).any() for p in visible):
            reason = 6
        else:
            if partition is not None:
                reason = partition.classify([p[0] for p in visible], x, y)
            else:
                projected_area = sum(area2(p[3]) for p in visible)
                reason = 1 if projected_area == 1. else 10
                if projected_area > 1. + EPS * max(1, len(visible)):
                    reason = 8
    children = []
    if reason in (1, 10):
        for face, bary, xyz, _ in visible:
            for child in range(1, len(bary)-1):
                area = _triangle_area(xyz[0], xyz[child], xyz[child+1])
                if area == 0:
                    continue
                if not np.isfinite(area) or area < np.finfo(float).tiny:
                    reason = 8; children = []
                    break
                children.append((face, int(local_lookup[face]), int(pixel), int(pixel)*len(faces)+face,
                                 child-1, bary[[0, child, child+1]], area))
            if reason == 8:
                break
    return children, reason


@diagnosed('target.construct')
def _construct(inputs: dict, revision: dict, budget: TargetBudget, check: Callable,
               *, frontier: dict | None = None, start_group: int = 0) -> tuple[dict, dict]:
    """Bounded, deterministic pixel grouping; a budget stop never drops intent."""
    check(); budget.checked()
    started = time.monotonic()
    vertices, faces, mask = inputs['vertices'], inputs['faces'], inputs['mask']
    h, w = mask.shape
    k = np.asarray(revision['inputs']['calibration']['intrinsic'], float)
    depth = inputs['depth'].astype(float) / 1000
    limits = revision['inputs']['depth']
    measured = np.isfinite(depth) & (depth >= limits['minimum_m']) & (depth <= limits['maximum_m'])
    active = mask & inputs['image_valid'] & measured
    status = np.zeros(mask.shape, np.uint8)
    status[mask] = 5
    status[mask & ~measured] = 3
    status[mask & (depth == 0)] = 2
    status[mask & ~inputs['image_valid']] = 4
    output = dict(inputs, **_empty_leaves(), pixel_status=status)
    # Deduplicate identical geometric triangles, including reordered/duplicated vertices.
    _, vertex_ids = np.unique(vertices, axis=0, return_inverse=True)
    keys = np.sort(vertex_ids[faces], axis=1)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    representative = first[inverse]
    duplicate_ids = np.flatnonzero(representative != np.arange(len(faces)))
    output['duplicate_faces'] = np.c_[duplicate_ids, representative[duplicate_ids]].astype('<i8')
    metrics = dict(candidate_count=0, indexed_faces=0, candidate_pixel_groups=0, processed_pixel_groups=0,
                   clip_count=0, exact_clip_fallbacks=0, pair_checks=0, occluded_polygons=0,
                   duplicate_face_count=len(duplicate_ids), maximum_subdivision_depth=0,
                   coverage_seed_comparisons=0, stop_reason='complete')
    if frontier is not None:start_group=frontier.get('page_start_group',0)
    if type(start_group) is not int or start_group < 0:
        raise ValueError('原目标续算前沿无效')
    if start_group:metrics['page_start_group']=start_group
    triangles = vertices[faces]
    partition = None
    if revision['inputs']['algorithms']['target'] == revision_api.EXACT_TARGET:
        from femto.geometry.exact_pixel_partition import PixelPartition
        partition = PixelPartition(triangles, k, check)
    finite = np.isfinite(triangles).all(axis=(1, 2))
    if not finite.all():
        raise ValueError('原目标源网格包含非有限坐标')
    check()
    crossing = (triangles[:, :, 2].min(axis=1) <= 0) & (triangles[:, :, 2].max(axis=1) > 0)
    if crossing.any():
        status[active] = 8
        metrics['stop_reason'] = 'source_crosses_reference_camera_plane'
        return output, metrics
    good = (triangles[:, :, 2] > 0).all(axis=1)
    good &= np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]), axis=1) > 0
    source_ids = np.sort(first[good[first]])
    uv = project(triangles[source_ids], k)
    # project uses only elementwise divide/multiply/add: each batch row is
    # bit-identical to projecting that source face alone. Clipped pixel
    # polygons still project their own barycentric geometry below.
    source_ids.setflags(write=False)
    uv.setflags(write=False)
    source_separated = _source_separation(
        triangles, k, vertex_ids[faces], source_projection=(source_ids, uv))
    lo = np.maximum(np.floor(np.clip(uv.min(axis=1)+.5, -1, max(w, h)+1)).astype(np.int64), [0, 0])
    hi = np.minimum(np.floor(np.clip(uv.max(axis=1)+.5, -1, max(w, h)+1)).astype(np.int64), [w-1, h-1])
    candidate_pixels, candidate_faces = [], []
    # Exact integer rectangle counts avoid reducing the same image windows
    # once per source face. Face order, checks and budget charging stay below.
    integral = np.pad(active.astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    rectangle_counts = np.zeros(len(source_ids), dtype=np.int64)
    nonempty = (lo[:, 0] <= hi[:, 0]) & (lo[:, 1] <= hi[:, 1])
    x0s, y0s = lo[nonempty].T
    x1s, y1s = (hi[nonempty]+1).T
    rectangle_counts[nonempty] = (integral[y1s, x1s]-integral[y0s, x1s]
                                 -integral[y1s, x0s]+integral[y0s, x0s])
    # Bounding rectangles are an index only; actual geometry is always clipped.
    for i, face in enumerate(source_ids):
        if i % 128 == 0:
            check()
        if ((frontier is None and budget.seconds > 0 and time.monotonic()-started >= budget.seconds)
                or (frontier is not None and frontier['stop_reason'] == 'index_time_budget'
                    and i == frontier['indexed_faces'])):
            status[active] = 9
            metrics['stop_reason'] = 'index_time_budget'
            return output, metrics
        metrics['indexed_faces'] += 1
        x0, y0 = lo[i]; x1, y1 = hi[i]
        if x0 > x1 or y0 > y1:
            continue
        count = int(rectangle_counts[i])
        if metrics['candidate_count'] + count > budget.max_candidates:
            status[active] = 9
            metrics['stop_reason'] = 'candidate_budget'
            return output, metrics
        if count:
            window = active[y0:y1+1, x0:x1+1]
            yy, xx = np.nonzero(window)
            candidate_pixels.append((yy+y0)*w + xx+x0)
            candidate_faces.append(np.full(count, face, np.int64))
            metrics['candidate_count'] += count
    check()
    if not candidate_pixels:
        return output, metrics
    groups, face_ids, starts, ends = _candidate_groups(candidate_pixels, candidate_faces)
    metrics['candidate_pixel_groups'] = len(groups)
    if start_group>len(groups):raise ValueError('原目标续算前沿超出候选范围')
    metrics['processed_pixel_groups']=start_group
    status.ravel()[groups] = 9
    local_lookup = np.full(len(faces), -1, np.int64)
    local_lookup[inputs['region_faces']] = np.arange(len(inputs['region_faces']))
    rows = {name: [] for name in _LEAVES if name != 'quality_status'}
    end_group = len(groups) if frontier is None else frontier['processed_pixel_groups']
    clip_batches = _PixelClipBatches(triangles, k, w, groups, face_ids, ends, end_group)
    gate_m = .01  # Existing reference localisation gate, not a calibrated error bound.
    for index, pixel in enumerate(groups):
        if index<start_group:continue
        if index % 32 == 0:
            check()
        if frontier is not None and index >= frontier['processed_pixel_groups']:
            metrics['stop_reason'] = frontier['stop_reason']
            break
        if frontier is None and time.monotonic()-started >= budget.seconds:
            metrics['stop_reason'] = 'time_budget'
            break
        y, x = divmod(int(pixel), w)
        polygons, reason, clip_count, exact_count, pair_count, occluded = _clip_pixel_polygons(face_ids, starts, ends, index, triangles, k, x, y, check, batches=clip_batches)
        discarded, exhausted, reason, pair_count, occluded = _resolve_pixel_occlusion(polygons, reason, pair_count, occluded, metrics, budget, triangles, k, source_separated, check)
        if exhausted:
            metrics['stop_reason'] = 'pair_budget'
            break
        children, reason = _pixel_target_children(polygons, discarded, reason, limits, depth, y, x, gate_m, local_lookup, pixel, faces, partition)
        if len(rows['area_m2']) + len(children) > budget.max_units:
            metrics['stop_reason'] = 'unit_budget'
            break
        for child in children:
            for name, value in zip(rows, child):
                rows[name].append(value)
        status[y, x] = reason
        metrics['processed_pixel_groups'] += 1
        metrics['clip_count'] += clip_count; metrics['exact_clip_fallbacks'] += exact_count
        metrics['pair_checks'] += pair_count; metrics['occluded_polygons'] += occluded
    if frontier is not None and metrics['processed_pixel_groups'] == len(groups):
        metrics['stop_reason'] = frontier['stop_reason']
    for name, values in rows.items():
        output[name] = np.asarray(values, dtype=_DTYPES[name]).reshape((-1, 3, 3) if name == 'barycentric' else (-1,))
    output['quality_status'] = _quality_values(output)
    check()
    return output, metrics


def _face_quality(domain: dict | None, region: dict, evidence: dict, vertices: np.ndarray,
                  faces: np.ndarray, check: Callable) -> np.ndarray:
    if domain is None:
        return np.full(len(region['triangle_indices']), 255, np.uint8)
    from femto.annotation.selection_intent import decode_runs
    from femto.planning.effective_domain import validate
    import trimesh
    selected_faces = faces[region['triangle_indices']]
    used, inverse = np.unique(selected_faces, return_inverse=True)
    mesh = trimesh.Trimesh(vertices=vertices[used], faces=inverse.reshape(-1, 3), process=False)
    validate(domain, evidence, domain['effective_width_mm'], mesh, region, check)
    return decode_runs(domain['face_class_runs'], [len(selected_faces), 1], values=range(4)).ravel().astype(np.uint8)


def build(model: Any, region: dict, revision: dict, *, evidence: dict | None = None,
          domain: dict | None = None, budget: TargetBudget | None = None,
          check: Callable = lambda: None) -> TargetLedger:
    from femto.planning.quality_constrained import evidence as get_evidence, checked_evidence
    revision_api.validate(revision)
    budget = (budget or TargetBudget()).checked()
    if revision['inputs']['numerics'] != revision_api.NUMERICS or not revision_api.supported_target_algorithms(revision['inputs']['algorithms']):
        raise ValueError('原目标台账不支持该算法或数值策略')
    if revision['inputs']['quality'] != revision_api.QUALITY:
        raise ValueError('原目标台账不支持该质量配置')
    if revision['inputs']['selection'] != revision_api.semantic_region(region):
        raise ValueError('原目标台账选区与计算修订不一致')
    k = np.asarray(model.projection.k, float)
    if (k.shape != (3, 3) or not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0
            or k[0, 1] != 0 or k[1, 0] != 0 or not np.array_equal(k[2], [0, 0, 1])):
        raise ValueError('原目标需要与参考投影一致的无偏斜内参')
    if model.store.origin_reference.get('kind') != 'initial_color_camera':
        raise ValueError('原目标需要可追溯的 W0 参考图变换')
    mask = intent_mask(region['selection_intent']).copy()
    vertices, faces = revision_api.canonical_arrays(model.store.mesh)
    if len(faces) > 4_000_000 or mask.size > 8_388_608:
        raise ValueError('原目标超过源网格或图像预算')
    record = get_evidence(model, region, check, allow_empty=True) if evidence is None else evidence
    classes, _ = checked_evidence(record, region)
    inputs = dict(vertices=vertices, faces=faces, region_faces=np.asarray(region['triangle_indices'], dtype='<i8'),
                  depth=np.array(model.projection.depth, copy=True), image_valid=np.array(
                      getattr(model.projection, 'image_valid', np.ones(mask.shape, bool)), copy=True),
                  mask=mask, quality_classes=classes.astype(np.uint8),
                  face_quality=_face_quality(domain, region, record, vertices, faces, check))
    _check_inputs(inputs, revision)
    arrays, metrics = _construct(inputs, revision, budget, check)
    return _seal(arrays, revision, budget, metrics, record['sha256'], None if domain is None else domain['sha256'])


def with_quality(ledger: TargetLedger, revision: dict, region: dict, evidence: dict, domain: dict,
                 check: Callable = lambda: None) -> TargetLedger:
    """Attach eligibility after construction; no target geometry may be removed."""
    if ledger.metadata['schema']==PAGED_SCHEMA:
        raise ValueError('分块目标的质量输入不能原地更换，请按新输入重建并核验各块')
    from femto.planning.quality_constrained import checked_evidence
    arrays = dict(ledger.arrays)
    classes, _ = checked_evidence(evidence, region)
    arrays['quality_classes'] = classes.astype(np.uint8)
    arrays['face_quality'] = _face_quality(domain, region, evidence, arrays['vertices'], arrays['faces'], check)
    arrays['quality_status'] = _quality_values(arrays)
    return _seal(arrays, revision, TargetBudget(**ledger.metadata['budget']), ledger.metadata['metrics'],
                 evidence['sha256'], domain['sha256'])


def _check_inputs(arrays: dict, revision: dict) -> None:
    inputs = revision['inputs']
    expected = {name: inputs['geometry'][name] for name in ('vertices', 'faces')}
    expected.update(depth=inputs['depth']['values'], image_valid=inputs['image']['image_valid'])
    for name, record in expected.items():
        if revision_api.array_record(arrays[name]) != record:
            raise ValueError('原目标输入与计算修订不一致：' + name)
    region = inputs['selection']
    if (not np.array_equal(arrays['mask'], intent_mask(region['selection_intent']))
            or not np.array_equal(arrays['region_faces'], region['triangle_indices'])):
        raise ValueError('原目标选区或源面映射不一致')
    faces, vertices = arrays['faces'], arrays['vertices']
    if (faces.ndim != 2 or faces.shape[1:] != (3,) or vertices.ndim != 2 or vertices.shape[1:] != (3,)
            or not len(faces) or not len(vertices) or np.any(faces < 0) or np.any(faces >= len(vertices))
            or not np.isfinite(vertices).all() or not np.array_equal(vertices, vertices.astype('<f4').astype('<f8'))):
        raise ValueError('原目标规范网格或索引无效')
    ids = arrays['region_faces']
    if (ids.ndim != 1 or not len(ids) or np.any(ids < 0) or np.any(ids >= len(faces))
            or np.any(np.diff(ids) <= 0) or arrays['face_quality'].shape != ids.shape):
        raise ValueError('原目标局部面映射无效')
    mask = arrays['mask']
    if any(arrays[name].shape != mask.shape for name in ('depth', 'image_valid', 'quality_classes')):
        raise ValueError('原目标参考数组尺寸不一致')
    if not np.array_equal(arrays['quality_classes'] > 0, mask):
        raise ValueError('原目标质量分类改变了原选择意图')
    for name, value in arrays.items():
        if name in _DTYPES and value.dtype.str != _DTYPES[name]:
            raise ValueError('原目标输入数组类型无效：' + name)


def validate_metadata(metadata: dict, revision: dict) -> None:
    revision_api.validate(revision)
    if (not revision_api.supported_target_algorithms(revision['inputs']['algorithms'])
            or revision['inputs']['numerics'] != revision_api.NUMERICS
            or revision['inputs']['quality'] != revision_api.QUALITY):
        raise ValueError('原目标算法或数值策略不受支持')
    if (not isinstance(metadata, dict) or metadata.get('schema') not in (SCHEMA,PAGED_SCHEMA) or metadata.get('asset') != ASSET
            or metadata.get('sha256') != revision_api.digest({k: v for k, v in metadata.items() if k != 'sha256'})
            or metadata.get('target_id') != revision['target_id'] or metadata.get('computation_sha256') != revision['sha256']
            or metadata.get('target_completion') != 'unverified' or metadata.get('coverage_certificate') is not None
            or metadata.get('execution_ready') is not False or metadata.get('physical_accuracy_verified') is not False):
        raise ValueError('原目标台账摘要、来源或语义不一致')
    TargetBudget(**metadata['budget']).checked()
    records = metadata.get('arrays')
    if not isinstance(records, dict) or set(records) != set(_DTYPES):
        raise ValueError('原目标资产数组清单不完整')
    for name, record in records.items():
        shape = record.get('shape')
        if (record.get('dtype') != _DTYPES[name] or not isinstance(shape, list) or not 1 <= len(shape) <= 3
                or any(type(v) is not int or v < 0 or v > 12_000_000 for v in shape)
                or np.prod(shape, dtype=object)*np.dtype(_DTYPES[name]).itemsize > MAX_ASSET_BYTES):
            raise ValueError('原目标数组尺寸或类型无效：' + name)
    n = records['area_m2']['shape'][0]
    if metadata['schema']==PAGED_SCHEMA:
        from femto.planning.target_paging import validate_headers
        validate_headers(metadata,revision)
    elif 'construction_pages' in metadata or n > metadata['budget']['max_units']:
        raise ValueError('原目标叶单元超过预算')
    for name in _LEAVES:
        if records[name]['shape'] != ([n, 3, 3] if name == 'barycentric' else [n]):
            raise ValueError('原目标叶单元数组不一致')
    if records['faces']['shape'][0] > 4_000_000 or np.prod(records['mask']['shape']) > 8_388_608:
        raise ValueError('原目标源资产超过预算')


@diagnosed('target.validate')
def validate(ledger: TargetLedger, revision: dict, *, evidence: dict, domain: dict | None,
             check: Callable = lambda: None, reproduce: bool = True) -> TargetLedger:
    from femto.planning.quality_constrained import checked_evidence
    metadata, arrays = ledger.metadata, ledger.arrays
    validate_metadata(metadata, revision)
    if set(arrays) != set(_DTYPES):
        raise ValueError('原目标资产缺少数组')
    for name, value in arrays.items():
        check()
        if revision_api.array_record(value) != metadata['arrays'][name]:
            raise ValueError('原目标资产损坏：' + name)
    _check_inputs(arrays, revision)
    classes, _ = checked_evidence(evidence, revision['inputs']['selection'])
    if (metadata['quality_evidence_sha256'] != evidence['sha256']
            or metadata['effective_domain_sha256'] != (None if domain is None else domain['sha256'])
            or not np.array_equal(arrays['quality_classes'], classes)):
        raise ValueError('原目标质量证据混用了不同修订')
    expected_quality = _face_quality(domain, revision['inputs']['selection'], evidence,
                                     arrays['vertices'], arrays['faces'], check)
    if not np.array_equal(arrays['face_quality'], expected_quality):
        raise ValueError('原目标质量资格与有效域不一致')
    if reproduce:
        if metadata['schema']==PAGED_SCHEMA:
            from femto.planning.target_paging import reproduce_materialized
            reproduce_materialized(ledger,revision,evidence=evidence,domain=domain,check=check)
            return ledger
        source_names = ('vertices', 'faces', 'region_faces', 'depth', 'image_valid', 'mask', 'quality_classes', 'face_quality')
        metrics = metadata['metrics']
        rebuilt, counts = _construct({name: arrays[name] for name in source_names}, revision,
                                      TargetBudget(**metadata['budget']), check,
                                      frontier=metrics)
        rebuilt_ledger = _seal(rebuilt, revision, TargetBudget(**metadata['budget']), counts,
                               evidence['sha256'], None if domain is None else domain['sha256'])
        if rebuilt_ledger.metadata != metadata:
            raise ValueError('原目标单元、归属或面积不能由原输入重现')
    elif _summary(arrays) != metadata['summary']:
        raise ValueError('原目标面积或未知范围摘要不一致')
    return ledger


def write(folder: Path, ledger: TargetLedger, check: Callable = lambda: None) -> None:
    """Only write inside a caller-owned derived staging/snapshot directory."""
    from femto.planning.surface_trajectory import canonical_json_bytes
    check()
    with (Path(folder)/ASSET).open('wb') as stream:
        np.savez_compressed(stream, **{name: ledger.arrays[name] for name in sorted(ledger.arrays)})
    (Path(folder)/MANIFEST).write_bytes(canonical_json_bytes(ledger.metadata))
    check()


def read(folder: Path, metadata: dict, revision: dict, *, evidence: dict, domain: dict | None,
         check: Callable = lambda: None, reproduce: bool = True) -> TargetLedger:
    """Reject corrupt, oversized, mixed-revision or non-reproducible assets."""
    validate_metadata(metadata, revision)
    folder = Path(folder)
    path = folder / MANIFEST
    if not path.is_file() or path.stat().st_size > 1024**2:
        raise ValueError('原目标台账清单缺失或过大')
    if json.loads(path.read_text(encoding='utf-8-sig')) != metadata:
        raise ValueError('原目标清单与路线绑定不一致')
    asset = folder / ASSET
    if not asset.is_file() or asset.stat().st_size > MAX_ASSET_BYTES:
        raise ValueError('原目标数组资产缺失或过大')
    with zipfile.ZipFile(asset) as archive:
        items = archive.infolist()
        names = [item.filename for item in items]
        if set(names) != {name+'.npy' for name in _DTYPES} or len(names) != len(_DTYPES):
            raise ValueError('原目标数组包包含重复或未知条目')
        if sum(item.file_size for item in items) > MAX_ASSET_BYTES:
            raise ValueError('原目标解压大小超过预算')
        for name in _DTYPES:
            check()
            with archive.open(name+'.npy') as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError('原目标数组版本不支持')
                record = metadata['arrays'][name]
                if list(shape) != record['shape'] or dtype.str != record['dtype'] or fortran or dtype.hasobject:
                    raise ValueError('原目标数组头与已核验的尺寸/类型不一致：' + name)
                if archive.getinfo(name+'.npy').file_size != stream.tell()+int(np.prod(shape, dtype=object))*dtype.itemsize:
                    raise ValueError('原目标数组字节长度不一致')
    with np.load(asset, allow_pickle=False) as source:
        arrays = {name: source[name] for name in _DTYPES}
    for array in arrays.values():
        array.setflags(write=False)
    return validate(TargetLedger(deepcopy(metadata), arrays), revision, evidence=evidence, domain=domain,
                    check=check, reproduce=reproduce)


def summary_text(metadata: dict | None) -> str:
    if metadata is None:
        return '原目标台账：旧记录未进行此项核验。'
    summary = metadata['summary']
    return (f"原目标：已定位面积 {summary['located_area_m2']*1e6:.1f} mm²，"
            f"其中 {summary['quality_excluded_area_m2']*1e6:.1f} mm² 未进入当前有效域；"
            f"另有 {summary['unlocated_pixel_count']} 个像素含未定位范围，面积未知。"
            '模型表面覆盖尚未核验。')
