"""Measured-reference quality and swept-width constraints for research surface paths.

The residual is an observed mesh/raw-depth disagreement, NOT a calibrated error
bound. No missing depth, pose or registration uncertainty is invented here.
Original regions and candidate-planner versions are never rewritten.
"""
from __future__ import annotations

from collections import Counter
from copy import copy, deepcopy
from dataclasses import replace
import hashlib
import math
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
from femto.operation_diagnostics import diagnosed

from femto.annotation.selection_intent import (
    decode_runs, encode_runs, intent_mask as selection_mask, digest,
)
from femto.planning.surface_trajectory import (
    Gap, ScanSegment, SurfaceTrajectory, ORDERED_SELECTION_ALGORITHM,
    canonical_json_bytes,
)

ALGORITHM = 'quality-constrained-surface-v1'
DOMAIN_ALGORITHM = 'quality-constrained-surface-v2'
EVIDENCE_SCHEMA = 'femto-reference-quality-constraint-v1'
METHOD = 'whole_chord_mesh_support_and_boundary_lower_bound_and_projected_enclosing_ball'


class PlanningConstraintBlocked(ValueError):
    """A blocked route still supplies a diagnostic record, never an empty valid plan."""
    def __init__(self, message: str, report: dict):
        super().__init__(message)
        self.report=report


def evidence(model: Any, region: dict, check: Callable = lambda: None, *, allow_empty: bool = False) -> dict:
    """Snapshot real reference diagnostics; selection intent remains untouched."""
    from femto.annotation.reference_quality import assess_selection, require_geometry
    mask = selection_mask(region['selection_intent'])
    check()
    result = assess_selection(model, mask, np.zeros_like(mask), check=check)
    if not allow_empty or result['status'] != 'checked':require_geometry(result)
    if region.get('view_reference') != model.view_reference:
        raise ValueError('选区与当前参考影像绑定不一致')
    projection = model.projection
    good = result['classes'] == 1
    if np.any(np.abs(projection.z[good].astype(float)*1000-projection.depth[good]) > 10.000001):
        raise ValueError('参考深度与投影质量状态不一致，请重新加载模型')
    yy, xx = np.indices(mask.shape)
    k = np.asarray(projection.k, float)
    # Ray-length disagreement, rounded UP to micrometres. It does not estimate
    # lateral calibration/pose error and is never labelled physical accuracy.
    ray = np.sqrt(1 + ((xx-k[0, 2])/k[0, 0])**2 + ((yy-k[1, 2])/k[1, 1])**2)
    residual = np.zeros(mask.shape, np.int32)
    residual[good] = np.ceil(np.abs(projection.z[good].astype(float)*1000
                                    - projection.depth[good])*ray[good]*1000).astype(np.int32)
    check()
    record = dict(schema=EVIDENCE_SCHEMA, image_size=[mask.shape[1], mask.shape[0]],
        view_reference=deepcopy(model.view_reference), intent_sha256=digest(mask),
        classes_runs=encode_runs(result['classes']), residual_um_runs=encode_runs(residual),
        residual_semantics='ceil_abs_mesh_minus_raw_reference_depth_times_ray_norm_um',
        depth_gate_mm=10.0, registration_uncertainty_mm=None, calibration_uncertainty_mm=None,
        physical_accuracy_verified=False, execution_ready=False)
    record['sha256'] = _hash(record)
    return record


def _hash(record: dict) -> str:
    return hashlib.sha256(canonical_json_bytes({k: v for k, v in record.items() if k != 'sha256'})).hexdigest()


def checked_evidence(record: dict, region: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    if (record.get('schema') != EVIDENCE_SCHEMA or record.get('sha256') != _hash(record)
            or record.get('registration_uncertainty_mm') is not None
            or record.get('calibration_uncertainty_mm') is not None
            or record.get('physical_accuracy_verified') is not False
            or record.get('execution_ready') is not False):
        raise ValueError('质量约束证据无效或已改变')
    classes = decode_runs(record['classes_runs'], record['image_size'], values=range(8))
    k=np.asarray(record['view_reference'].get('undistorted_intrinsic'),float)
    if (k.shape!=(3,3) or not np.isfinite(k).all() or min(k[0,0],k[1,1])<=0
            or record['view_reference'].get('image_size')!=record['image_size']):
        raise ValueError('质量证据参考标定或图像尺寸无效')
    runs = record['residual_um_runs']
    if any(type(v) is not int or v < 0 or v > 100000000 for v in runs[::2]):
        raise ValueError('参考残差数据无效')
    residual = decode_runs(runs, record['image_size'], values=set(runs[::2]))
    if np.any(residual[classes != 1]):
        raise ValueError('缺失像素不能包含伪造残差')
    if (record.get('intent_sha256') != digest(classes>0)
            or record.get('residual_semantics') != 'ceil_abs_mesh_minus_raw_reference_depth_times_ray_norm_um'
            or record.get('depth_gate_mm') != 10.0):
        raise ValueError('质量证据的选择意图或残差语义不一致')
    if region is not None:
        mask = selection_mask(region['selection_intent'])
        if (record['view_reference'] != region['view_reference'] or record['intent_sha256'] != digest(mask)
                or not np.array_equal(classes > 0, mask)):
            raise ValueError('质量约束与原选区或参考影像不一致')
    return classes, residual


def _rectangle(mid: np.ndarray, radius: float, k: np.ndarray) -> tuple[int, int, int, int] | None:
    # Contains the projection of the entire 3D ball, including off-axis rays.
    z = mid[2]
    if z <= radius:
        return None
    uv = mid[:2]/z * [k[0, 0], k[1, 1]] + k[:2, 2]
    bound = max(k[0, 0], k[1, 1])*radius/(z-radius)*(1+np.linalg.norm(mid[:2])/z)
    low = np.floor(uv-bound-.5).astype(int)
    high = np.ceil(uv+bound+.5).astype(int)
    return int(low[0]), int(low[1]), int(high[0]), int(high[1])


def _supported_chord(a: np.ndarray, b: np.ndarray, faces: np.ndarray,
                     vertices: np.ndarray, face_ids: list[int], *, _prepared_faces=None) -> bool:
    """Certify the chord is covered by the supplied triangles, not just samples.

    Folded-surface chords or chords spanning unavailable faces are rejected if this
    conservative union cannot certify them; no interpolation is emitted.
    """
    intervals = []
    for index in sorted(set(face_ids)):
        if _prepared_faces is None:
            tri = vertices[faces[index]]
            origin = tri[0]
            u, v = tri[1]-origin, tri[2]-origin
            normal = np.cross(u, v)
            length = np.linalg.norm(normal)
        else:
            prepared = _prepared_faces(index)
            origin, u, v = prepared.origin, prepared.ab, prepared.ac
            normal, length = prepared.normal, prepared.normal_length
        if length <= 1e-15 or max(abs((a-origin)@normal), abs((b-origin)@normal))/length > 1e-6:
            continue
        matrix = np.column_stack((u, v)) if _prepared_faces is None else prepared.matrix
        offsets = np.column_stack((a-origin, b-origin))
        if (_prepared_faces is not None and math.isfinite(length)
                and prepared.solver.chord_inside_one(offsets)):
            intervals.append((0., 1.))
            continue
        coords = np.linalg.lstsq(matrix, offsets, rcond=None)[0]
        weights = np.vstack((1-coords.sum(axis=0), coords))
        lo, hi = 0., 1.
        for start, end in weights:
            delta = end-start
            if abs(delta) <= 1e-15:
                if start < -1e-8:hi = -1.
            elif delta > 0:lo = max(lo, (-1e-8-start)/delta)
            else:hi = min(hi, (-1e-8-start)/delta)
        if lo <= hi:intervals.append((lo, hi))
    reach = 0.
    for lo, hi in sorted(intervals):
        if lo > reach+1e-10:return False
        reach = max(reach, hi)
    return reach >= 1.-1e-10


class ReferenceEnvelope:
    """The existing reference-image gate, reusable for candidate screening.

    A successful probe alone never admits a path: mesh boundary and whole-edge
    support still require edge_checks. The residual-neighborhood budget covers
    every probe made with this instance.
    """
    def __init__(self, record: dict, width_mm: float):
        if isinstance(width_mm, bool) or not np.isfinite(width_mm) or not .5 <= width_mm <= 100:
            raise ValueError('有效作用宽度必须在 0.5–100 mm 范围内')
        self.classes, self.residual = checked_evidence(record)
        self.k = np.asarray(record['view_reference']['undistorted_intrinsic'], float)
        self.width_mm = width_mm
        self.integral = np.pad((self.classes != 1).astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        self.height, self.width = self.classes.shape
        self.global_residual = float(self.residual.max())/1e6
        self.sampled_pixels = 0
        self._residual_tiles = None

    def _residual_max(self, outer):
        """Exact rectangle maximum: whole 16px tiles plus unrounded borders.

        Count values actually reduced, including tile construction. This keeps
        the existing work limit without charging overlapping rectangles for
        rescanning every interior pixel. No residual or footprint is rounded.
        """
        x0, y0, x1, y1 = outer
        # Match the original NumPy slice semantics, including off-image bounds.
        ya, yb, _ = slice(max(0, y0), min(self.height, y1+1)).indices(self.height)
        xa, xb, _ = slice(max(0, x0), min(self.width, x1+1)).indices(self.width)
        if ya >= yb or xa >= xb:
            return self.global_residual
        tile = 16
        top, bottom = ((ya+tile-1)//tile)*tile, (yb//tile)*tile
        left, right = ((xa+tile-1)//tile)*tile, (xb//tile)*tile
        if top >= bottom or left >= right:
            pieces = [self.residual[ya:yb, xa:xb]]
        else:
            if self._residual_tiles is None:
                self.sampled_pixels += self.residual.size + ((self.height+tile-1)//tile)*self.width
                if self.sampled_pixels > 100000000:
                    raise ValueError('局部残差邻域检查超过预算，请缩小选区')
                rows = np.maximum.reduceat(self.residual, np.arange(0, self.height, tile), axis=0)
                self._residual_tiles = np.maximum.reduceat(rows, np.arange(0, self.width, tile), axis=1)
                self._residual_tiles.setflags(write=False)
            pieces = [self.residual[ya:top, xa:xb], self.residual[bottom:yb, xa:xb],
                      self.residual[top:bottom, xa:left], self.residual[top:bottom, right:xb],
                      self._residual_tiles[top//tile:bottom//tile, left//tile:right//tile]]
        self.sampled_pixels += sum(piece.size for piece in pieces)
        if self.sampled_pixels > 100000000:
            raise ValueError('局部残差邻域检查超过预算，请缩小选区')
        return float(max(piece.max() for piece in pieces if piece.size))/1e6

    def probe(self, mid: np.ndarray, half: float, *, details=False, width_mm=None):
        width = self.width_mm if width_mm is None else width_mm
        if width_mm is not None and (isinstance(width, bool) or not np.isfinite(width) or not .5 <= width <= 100):
            raise ValueError('逐边外包宽度必须在 0.5–100 mm 范围内')
        margin = 0.
        reason = 'outer_projection_unavailable'
        rect = None
        outer = _rectangle(mid, half+width/2000+self.global_residual, self.k)
        if outer is not None:
            margin = self._residual_max(outer)
            # Identical radii produce the identical rectangle; the residual
            # query above still runs and consumes the unchanged work budget.
            rect = (outer if margin == self.global_residual else
                    _rectangle(mid, half+width/2000+margin, self.k))
            reason = 'footprint_projection_unavailable'
            if rect is not None:
                x0, y0, x1, y1 = rect
                inside = 0 <= x0 <= x1 < self.width and 0 <= y0 <= y1 < self.height
                reason = 'footprint_outside_image'
                if inside:
                    bad = self.integral[y1+1, x1+1]-self.integral[y0, x1+1]-self.integral[y1+1, x0]+self.integral[y0, x0]
                    reason = 'accepted' if bad == 0 else 'reference_footprint_unconfirmed'
        diagnostic = None
        if details:
            outside = pending = None
            if rect is not None:
                x0, y0, x1, y1 = rect
                values = self.classes[max(0, y0):min(self.height, y1+1), max(0, x0):min(self.width, x1+1)]
                outside = int(np.count_nonzero(values == 0))
                pending = int(np.count_nonzero((values != 0) & (values != 1)))
                if reason == 'reference_footprint_unconfirmed':
                    reason = ('outside_selection_and_quality' if outside and pending else
                              'outside_selection' if outside else 'reference_quality_unconfirmed')
            diagnostic = dict(reason=reason, pixel_bounds=None if rect is None else list(rect),
                outside_selection_pixels=outside, quality_unconfirmed_pixels=pending,
                residual_margin_mm=float(margin*1000))
        return reason == 'accepted', margin, diagnostic


@diagnosed('route.edge_checks')
def edge_checks(segments: list[ScanSegment], mesh: Any, record: dict, width_mm: float,
                check: Callable = lambda: None, *, allowed_faces: np.ndarray | None = None,
                support_checker: Any = None, diagnostics: list | None = None,
                edge_widths_mm: list[float] | None = None) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Reject whole edges; accepted coverage balls cannot touch unknown pixels."""
    if isinstance(width_mm, bool) or not np.isfinite(width_mm) or not .5 <= width_mm <= 100:
        raise ValueError('有效作用宽度必须在 0.5–100 mm 范围内')
    reference = ReferenceEnvelope(record, width_mm)
    return _edge_checks(segments, mesh, reference, width_mm, check, allowed_faces=allowed_faces,
                        support_checker=support_checker, diagnostics=diagnostics, edge_widths_mm=edge_widths_mm)


@diagnosed('route.edge_prepare')
def _prepare_edge_checks(mesh, record, width_mm, check, allowed_faces):
    """Bind one graph's inputs; only immutable preparation is reused.

    The graph never accepts a caller-supplied validation flag or cached edge
    decision. Separate graphs and ordinary edge_checks still validate their
    own evidence. Snapshot arrays prevent a later caller mutation from mixing
    new geometry or quality with the bound inputs.
    """
    check()
    reference = ReferenceEnvelope(record, width_mm)
    from femto.geometry.triangle_support import prepare_triangle_faces
    vertices, faces, prepared_faces = prepare_triangle_faces(mesh.vertices, mesh.faces)
    frozen_mesh = SimpleNamespace(vertices=vertices, faces=faces)
    allowed = None if allowed_faces is None else np.array(allowed_faces, dtype=bool, copy=True)
    for value in (reference.classes, reference.residual, reference.k, reference.integral,
                  frozen_mesh.vertices, frozen_mesh.faces, allowed):
        if value is not None:value.setflags(write=False)
    context=None;boundary_distances=None
    check()

    @diagnosed('route.edge_checks')
    def evaluate(segments):
        nonlocal context,boundary_distances
        check()
        if context is None:
            context=_mesh_edge_context(frozen_mesh.vertices,frozen_mesh.faces,allowed)
            for value in context:value.setflags(write=False)
            from femto.planning.boundary_clearance import prepare_nearest_boundary_distances
            boundary_distances=prepare_nearest_boundary_distances(context[2],check)
        # The existing residual-neighbourhood limit is per edge-check batch.
        # Keep a separate counter for every call, including recursive splits.
        batch = copy(reference)
        batch.sampled_pixels = 0
        return _edge_checks(segments, frozen_mesh, batch, width_mm, check, allowed_faces=allowed,
                            _mesh_context=context,_boundary_distances=boundary_distances,
                            _prepared_faces=prepared_faces)

    return evaluate


def _mesh_edge_context(vertices,faces,allowed_faces):
    vertex_order = np.argsort(faces.ravel(), kind='stable')
    vertex_offsets = np.r_[0, np.cumsum(np.bincount(faces.ravel(), minlength=len(vertices)))]
    if allowed_faces is not None:
        allowed_faces=np.asarray(allowed_faces,bool)
        if allowed_faces.shape!=(len(faces),) or not allowed_faces.any():
            raise ValueError('有效区域没有可规划网格面，请检查缺失范围或补扫')
    boundary_faces=faces if allowed_faces is None else faces[allowed_faces]
    edges = np.sort(boundary_faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = vertices[unique[counts == 1]]
    if np.any(counts > 2) or not len(boundary):
        raise ValueError('缺少可核验的流形网格边界，不能检查覆盖带')
    return vertex_order,vertex_offsets,boundary


def _edge_checks(segments, mesh, reference, width_mm, check, *, allowed_faces=None,
                 support_checker=None, diagnostics=None, edge_widths_mm=None,
                 _mesh_context=None,_boundary_distances=None,_prepared_faces=None):
    """The same complete edge predicates for fresh and graph-bound inputs."""
    pairs = [(a, b) for s in segments for a, b in zip(s.points, s.points[1:])]
    if len(pairs) > 200000:raise ValueError('质量约束路线超过检查预算')
    if edge_widths_mm is not None and (len(edge_widths_mm) != len(pairs) or any(
            isinstance(w, bool) or not np.isfinite(w) or not .5 <= w <= width_mm for w in edge_widths_mm)):
        raise ValueError('逐边外包宽度缺失或超过原生成包络')
    if _prepared_faces is None:
        from femto.geometry.triangle_support import prepare_triangle_faces
        vertices, faces, _prepared_faces = prepare_triangle_faces(mesh.vertices, mesh.faces)
    else:
        vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    check()
    vertex_order,vertex_offsets,boundary=(_mesh_edge_context(vertices,faces,allowed_faces)
        if _mesh_context is None else _mesh_context)
    a = np.asarray([p[0]['xyz_m'] for p in pairs]).reshape(-1, 3)
    b = np.asarray([p[1]['xyz_m'] for p in pairs]).reshape(-1, 3)
    mid, half = (a+b)/2, np.linalg.norm(a-b, axis=1)/2
    if support_checker is not None:
        # P4's exact anchors are authoritative; rounded display XYZ is only a
        # proxy for existing width/reference checks. Enclose its rounding error
        # before applying those same predicates. Legacy arithmetic is unchanged.
        half = np.nextafter(half + support_checker.begin(pairs), np.inf)
    from femto.planning.boundary_clearance import nearest_boundary_distances
    distance = (nearest_boundary_distances(mid, boundary, check) if _boundary_distances is None
                else _boundary_distances(mid))
    accepted = np.zeros(len(pairs), bool)
    margins = np.zeros(len(pairs), float)
    reasons = []
    # A same-face interior chord can share a single matrix decomposition with
    # its peers. This certifies the complete affine interval [0,1]; cross-face
    # unions and any near-threshold case still use _supported_chord unchanged.
    interior_chords = np.zeros(len(pairs), bool)
    if support_checker is None and len(pairs) >= 8:
        groups = {}
        for i, (left, right) in enumerate(pairs):
            index = left.get('triangle_index')
            if (type(index) is int and 0 <= index < len(faces)
                    and type(right.get('triangle_index')) is int and right['triangle_index'] == index):
                groups.setdefault(index, []).append(i)
        for index, ids in groups.items():
            if len(ids) >= 8:
                check()
                prepared = _prepared_faces(index)
                interior_chords[ids] = prepared.solver.chord_inside(a[ids], b[ids], prepared)
    for i, (left, right) in enumerate(pairs):
        if i % 128 == 0:check()
        if allowed_faces is not None and not all(allowed_faces[p['triangle_index']] for p in (left,right)):
            if diagnostics is not None:diagnostics.append(dict(accepted=False, reason='outside_effective_domain'))
            reasons.append('outside_effective_domain');continue
        reason = 'quality_or_width_boundary'
        current_width = width_mm if edge_widths_mm is None else edge_widths_mm[i]
        passed, margin, detail = reference.probe(mid[i], half[i], details=diagnostics is not None,
                                               width_mm=None if edge_widths_mm is None else current_width)
        margins[i] = margin*1000
        if passed:
            if distance[i]-half[i] >= current_width/2000+margin+1e-9:
                ids = [left['triangle_index'], right['triangle_index']]
                supported = (True if interior_chords[i] else support_checker.supported(i, left, right) if support_checker is not None
                             else _supported_chord(a[i], b[i], faces, vertices, ids,
                                                   _prepared_faces=_prepared_faces))
                if not supported and support_checker is None:
                    incident = np.unique(np.concatenate([vertex_order[vertex_offsets[v]:vertex_offsets[v+1]]//3
                        for v in np.unique(faces[ids])]))
                    if allowed_faces is not None:incident=incident[allowed_faces[incident]]
                    supported = _supported_chord(a[i], b[i], faces, vertices, incident.tolist(),
                                                 _prepared_faces=_prepared_faces)
                if supported:
                    accepted[i] = True
                    reason = 'accepted'
                else:reason = ('chord_not_certified_on_mesh' if support_checker is None else support_checker.reason(i))
            else:reason = 'mesh_boundary_clearance'
            if detail is not None:detail['reason'] = reason
        if diagnostics is not None:diagnostics.append(dict(detail, accepted=bool(accepted[i])))
        reasons.append(reason)
    check()
    return accepted, reasons, margins


def _items(chains: list[tuple[int, list[dict]]]) -> list:
    items, route = [], 0
    previous = None
    for index, (block, raw) in enumerate(chains):
        points = deepcopy(raw)
        for point in points:
            point['route_index'] = route
            route += 1
        # A new segment starts at rest; incoming motion is not an emitted edge.
        points[0]['role'] = 'scan'
        points[0].pop('revisit', None)
        segment = ScanSegment(index, points[0]['row_index'], points, block)
        if previous is not None:
            left, right = previous.points[-1], points[0]
            inter = previous.block_index != block
            direction = np.asarray(points[1]['xyz_m'])-right['xyz_m']
            norm = np.linalg.norm(direction)
            direction = direction/norm if norm > 0 else np.array([1.,0.,0.])
            detail = dict(inferred_cause='selection_break' if inter else 'quality_or_width_constraint',
                method=METHOD, confidence='conservative_research_constraint', decision_use='research_route_only')
            if not inter:detail['link_failure_reason'] = 'quality_or_width_constraint'
            items.append(Gap(index-1, 'selection_break' if inter else 'unsupported_geometry', detail,
                dict(route_index=left['route_index'], xyz_m=left['xyz_m']),
                dict(route_index=right['route_index'], xyz_m=right['xyz_m']), direction.tolist(),
                left['row_index'], right['row_index'], float(np.linalg.norm(np.asarray(left['xyz_m'])-right['xyz_m']))))
        items.append(segment)
        previous = segment
    return items


def _summaries(items: list, templates: list[dict], classes: np.ndarray) -> tuple[dict, dict]:
    from femto.planning.selection_blocks import _walk_connectors
    from femto.annotation.selection_intent import label_components
    labels, _ = label_components(classes > 0)
    pending = int(((classes > 0) & (classes != 1)).sum())
    blocks = []
    segments = [s for s in items if isinstance(s, ScanSegment)]
    gaps = [g for g in items if isinstance(g, Gap)]
    for source in templates:
        block = {k: deepcopy(source[k]) for k in ('block_index', 'component_id', 'face_count', 'area_m2',
                 'pixel_bounds', 'selection', 'face_index_min', 'face_index_max', 'face_indices_sha256') if k in source}
        owned = [s for s in segments if s.block_index == block['block_index']]
        points = [p for s in owned for p in s.points]
        component_pixels = labels == block['component_id']+1
        component_pending = int((component_pixels & (classes != 1)).sum())
        block['selection']['support'] = dict(geometry_unconfirmed_pixels=component_pending,
            quality_source=EVIDENCE_SCHEMA, by_class={str(code):int((component_pixels & (classes==code)).sum()) for code in range(1,8)})
        scan = connector = 0.
        turns, detours, revisits, revisit_length = _walk_connectors(owned)
        for s in owned:
            for a,b in zip(s.points,s.points[1:]):
                length = float(np.linalg.norm(np.asarray(a['xyz_m'])-b['xyz_m']))
                if b.get('role') == 'connector':connector += length
                else:scan += length
        block.update(status='incomplete' if owned else 'unplannable', reason='quality_width_review_required',
            message='覆盖带按质量与宽度约束截断；目标范围未宣称完成，请核对或补扫。',
            segment_count=len(owned), gap_count=max(0,len(owned)-1), point_count=len(points),
            row_count=len({p['row_index'] for p in points}), scan_length_m=scan, connector_length_m=connector,
            segment_index_first=owned[0].segment_index if owned else None,
            row_index_first=min((p['row_index'] for p in points), default=None),
            connected_row_turns=turns, connected_detours=detours,
            continuity=dict(chain_count=len(owned), internal_gap_count=max(0,len(owned)-1),
                status='single_chain' if len(owned)==1 else 'broken' if owned else 'none'),
            coverage=dict(status='unverified', uncovered_face_count=None, uncovered_area_m2=None),
            link_failure_counts={'quality_or_width_constraint':max(0,len(owned)-1)},
            revisit=dict(connector_sample_count=revisits, connector_length_m=revisit_length))
        blocks.append(block)
    points = [p for s in segments for p in s.points]
    z = [p['xyz_m'][2] for p in points]
    stats = dict(block_count=len(blocks), plannable_block_count=sum(b['segment_count']>0 for b in blocks),
        planned_block_count=0, incomplete_block_count=sum(b['segment_count']>0 for b in blocks),
        unplannable_block_count=sum(b['segment_count']==0 for b in blocks),
        segment_count=len(segments), point_count=len(points), gap_count=len(gaps),
        gap_counts=dict(sorted(Counter(g.reason for g in gaps).items())), row_count=len({p['row_index'] for p in points}),
        scan_length_m=sum(b['scan_length_m'] for b in blocks), connector_length_m=sum(b['connector_length_m'] for b in blocks),
        connected_row_turns=sum(b['connected_row_turns'] for b in blocks),
        connected_detours=sum(b['connected_detours'] for b in blocks), inter_block_gap_count=sum(g.reason=='selection_break' for g in gaps),
        continuity_status='partial', coverage_status='partial', geometry_unconfirmed_pixel_count=pending,
        unplannable_face_count=sum(b['face_count'] for b in blocks if not b['segment_count']),
        unplannable_area_m2=sum(b['area_m2'] for b in blocks if not b['segment_count']),
        uncovered_face_count=None, uncovered_fragment_count=None,
        revisit_connector_sample_count=sum(b['revisit']['connector_sample_count'] for b in blocks),
        revisit_connector_length_m=sum(b['revisit']['connector_length_m'] for b in blocks),
        treatment_coverage_claim='none_research_footprint_only',
        depth_min_m=float(min(z)), depth_max_m=float(max(z)), depth_span_m=float(max(z)-min(z)))
    stats['total_length_m'] = stats['scan_length_m']+stats['connector_length_m']
    return stats, dict(schema='femto-surface-trajectory-blocks-v2', items=blocks,
        block_definition='original physician selection components, retained even when constrained out',
        order_rule='retained candidate visit order; no new transitions')


def constrain(candidate: SurfaceTrajectory, mesh: Any, region: dict, record: dict, width_mm: float,
              check: Callable = lambda: None, *, domain: dict | None = None) -> SurfaceTrajectory:
    classes, _ = checked_evidence(record, region)
    segments = [s for s in candidate.ordered_items if isinstance(s, ScanSegment)]
    allowed_faces=None
    if domain is not None:
        from femto.planning.effective_domain import validate as validate_domain
        allowed_faces=validate_domain(domain,record,width_mm,mesh,region,check)
    accepted, reasons, margins = edge_checks(segments, mesh, record, width_mm, check,allowed_faces=allowed_faces)
    chains, rejected, offset = [], [], 0
    for segment in segments:
        current = []
        for edge_index,(left,right) in enumerate(zip(segment.points,segment.points[1:])):
            if accepted[offset]:
                if not current:current = [left]
                current.append(right)
            elif current:
                chains.append((segment.block_index,current));current=[]
            if not accepted[offset]:
                reason = reasons[offset]
                if (rejected and rejected[-1]['candidate_segment_index']==segment.segment_index
                        and rejected[-1]['edge_end']==edge_index-1 and rejected[-1]['reason']==reason):
                    rejected[-1].update(edge_end=edge_index, end_xyz_m=right['xyz_m'], end_pixel=right.get('pixel'))
                else:
                    rejected.append(dict(candidate_segment_index=segment.segment_index, block_index=segment.block_index,
                        edge_start=edge_index, edge_end=edge_index, reason=reason,
                        start_xyz_m=left['xyz_m'], end_xyz_m=right['xyz_m'],
                        start_pixel=left.get('pixel'), end_pixel=right.get('pixel')))
            offset += 1
        if current:chains.append((segment.block_index,current))
    if not chains:
        raise PlanningConstraintBlocked(
            '几何质量与有效作用宽度联合约束后无可保留路线；请检查缺失范围、补扫或核对作用宽度。原选区保留。',
            dict(status='blocked',candidate_edges=len(accepted),retained_edges=0,rejected_edges=len(accepted),
                 rejection_counts=dict(Counter(reasons)),rejected_ranges=rejected,
                 target_completion='incomplete',execution_ready=False,physical_accuracy_verified=False))
    items = _items(chains)
    pending = int(((classes>0)&(classes!=1)).sum())
    stats, blocks = _summaries(items, candidate.blocks['items'], classes)
    report = dict(schema='femto-quality-width-report-v1', method=METHOD,
        candidate_edges=len(accepted), retained_edges=int(accepted.sum()), rejected_edges=int((~accepted).sum()),
        rejection_counts=dict(sorted(Counter(r for r in reasons if r!='accepted').items())),
        edge_decision_runs=encode_runs(accepted), local_residual_margin_max_mm=float(margins.max(initial=0)),
        effective_width_mm=float(width_mm), selected_pixels=int((classes>0).sum()), pending_pixels=pending,
        pending_classes_runs=encode_runs(np.where(classes==1,0,classes)),
        rejected_ranges=rejected, continuous_visibility_verified=False,
        target_completion='incomplete_or_unverified', registration_uncertainty_mm=None,
        rescan_required=pending>0, execution_ready=False, physical_accuracy_verified=False)
    planner = dict(kind='quality_constrained_surface', algorithm_version=ALGORITHM,
        parameters=dict(candidate.planner['parameters'], effective_width_mm=float(width_mm), quality_evidence=deepcopy(record)),
        candidate_algorithm=ORDERED_SELECTION_ALGORITHM, sweep_frame=candidate.planner['sweep_frame'])
    gap_report=dict(counts=stats['gap_counts'], quality_constraint=report)
    if domain is not None:
        planner['algorithm_version']=DOMAIN_ALGORITHM
        planner['parameters']['effective_domain']=deepcopy(domain)
        report['candidate_segment_edges']=[len(s.points)-1 for s in segments]
        report['effective_domain_sha256']=domain['sha256']
        gap_report['candidate_gaps']=[g.to_dict() for g in candidate.ordered_items if isinstance(g,Gap)]
        gap_report['missing_regions']=deepcopy(domain['missing_regions'])
    return replace(candidate, planner=planner, ordered_items=items, statistics=stats, blocks=blocks,
        gap_report=gap_report,
        spacing_report=dict(status='not_fully_verified', reason='candidate_route_trimmed_by_quality_and_width',
            target_spacing_mm=candidate.planner['parameters']['row_spacing_mm'], distribution=None), row_report=[])


def validate(trajectory: SurfaceTrajectory, mesh: Any = None, region: dict | None = None,
             check: Callable = lambda: None) -> None:
    """Validate offline records and, with geometry, recheck every retained edge."""
    from femto.planning.surface_trajectory import validate_trajectory, CONNECTED_ALGORITHM
    params = trajectory.planner['parameters']
    classes, _ = checked_evidence(params['quality_evidence'], region)
    check()
    if (params['quality_evidence']['intent_sha256'] != trajectory.source.get('selection_intent_sha256')
            or trajectory.planner.get('candidate_algorithm') != ORDERED_SELECTION_ALGORITHM):
        raise ValueError('质量约束来源或候选算法绑定不一致')
    segments = [s for s in trajectory.ordered_items if isinstance(s, ScanSegment)]
    # Reuse the stable basic trajectory validator for points, step limits,
    # mesh membership, gap endpoints and orientation prohibitions.
    shadow = []
    for item in trajectory.ordered_items:
        check()
        if isinstance(item, ScanSegment):
            points = [{k:v for k,v in p.items() if k!='revisit'} for p in item.points]
            # Chord support is independently checked below for both motion roles.
            for point in points:point['role']='scan'
            shadow.append(replace(item, points=points, row_index=min(p['row_index'] for p in points), block_index=None))
        else:
            shadow.append(replace(item, reason='unsupported_geometry', detail=dict(inferred_cause='surface_gap',
                method='sweep_plane_outer_loop_containment', confidence='heuristic', decision_use='none')))
    validate_trajectory(replace(trajectory, planner=dict(trajectory.planner, algorithm_version=CONNECTED_ALGORITHM,
        parameters=dict(params, selection_blocks=False, block_planning=False, connect_rows=True)),
        ordered_items=shadow, blocks=None), mesh=mesh)
    canonical = _items([(s.block_index,s.points) for s in segments])
    if [i.to_dict() for i in canonical] != [i.to_dict() for i in trajectory.ordered_items]:
        raise ValueError('质量约束路线索引或断点记录不一致')
    pending=int(((classes>0)&(classes!=1)).sum())
    stats, blocks = _summaries(trajectory.ordered_items, trajectory.blocks['items'], classes)
    if stats!=trajectory.statistics or blocks!=trajectory.blocks:
        raise ValueError('质量约束路线统计不一致')
    report=trajectory.gap_report['quality_constraint']
    if (report.get('schema')!='femto-quality-width-report-v1' or report.get('method')!=METHOD
            or report.get('continuous_visibility_verified') is not False
            or not np.isfinite(report.get('local_residual_margin_max_mm',np.nan))
            or report['local_residual_margin_max_mm']<0):
        raise ValueError('质量约束报告语义无效')
    decisions = decode_runs(report['edge_decision_runs'], [report['candidate_edges'],1], values=(0,1))
    if (report['retained_edges']!=sum(len(s.points)-1 for s in segments)
            or report['candidate_edges']!=report['retained_edges']+report['rejected_edges']
            or report['effective_width_mm']!=params['effective_width_mm']
            or report['pending_pixels']!=pending or report['target_completion']!='incomplete_or_unverified'
            or report['physical_accuracy_verified'] is not False or report['execution_ready'] is not False
            or int(decisions.sum()) != report['retained_edges']
            or sum(report['rejection_counts'].values()) != report['rejected_edges']
            or report['selected_pixels'] != int((classes>0).sum())
            or report['pending_classes_runs'] != encode_runs(np.where(classes==1,0,classes))
            or report['rescan_required'] != (pending>0)
            or report['registration_uncertainty_mm'] is not None):
        raise ValueError('质量约束报告与路线不一致')
    domain=params.get('effective_domain')
    if (trajectory.planner['algorithm_version']==DOMAIN_ALGORITHM)!=(domain is not None):
        raise ValueError('有效区域记录与路线版本不一致')
    allowed_faces=None
    if domain is not None:
        from femto.planning.effective_domain import validate as validate_domain
        allowed_faces=validate_domain(domain,params['quality_evidence'],params['effective_width_mm'],mesh,region,check)
        if (report.get('effective_domain_sha256')!=domain['sha256']
                or trajectory.gap_report.get('missing_regions')!=domain['missing_regions']):
            raise ValueError('路线与有效区域、缺失处理记录不一致')
        sizes=report['candidate_segment_edges']
        if any(type(n) is not int or n<1 for n in sizes) or sum(sizes)!=report['candidate_edges']:
            raise ValueError('候选路线边数不一致')
        original_gaps=trajectory.gap_report.get('candidate_gaps',[])
        if len(original_gaps)!=len(sizes)-1:raise ValueError('候选路线原断点记录不完整')
        from femto.planning.surface_trajectory import GAP_REASONS
        point_offsets=np.cumsum(np.asarray(sizes)+1)
        for index,gap in enumerate(original_gaps):
            if (gap.get('gap_index')!=index or gap.get('reason') not in GAP_REASONS
                    or gap.get('previous_point',{}).get('route_index')!=int(point_offsets[index]-1)
                    or gap.get('next_point',{}).get('route_index')!=int(point_offsets[index])):
                raise ValueError('候选路线原断点来源不一致')
        offsets=np.r_[0,np.cumsum(sizes)]
        excluded=np.zeros(report['candidate_edges'],bool)
        counts=Counter()
        for item in report['rejected_ranges']:
            segment,start,end=(item[k] for k in ('candidate_segment_index','edge_start','edge_end'))
            if (any(type(n) is not int for n in (segment,start,end)) or not 0<=segment<len(sizes)
                    or not 0<=start<=end<sizes[segment]):raise ValueError('排除范围索引无效')
            target=slice(offsets[segment]+start,offsets[segment]+end+1)
            if excluded[target].any():raise ValueError('排除范围重复')
            excluded[target]=True;counts[item['reason']]+=end-start+1
            for key in ('start_xyz_m','end_xyz_m'):
                if np.asarray(item[key]).shape!=(3,) or not np.isfinite(item[key]).all():raise ValueError('排除范围坐标无效')
        if not np.array_equal(excluded,~decisions.ravel().astype(bool)) or dict(counts)!=report['rejection_counts']:
            raise ValueError('排除范围与逐边决定不一致')
    if mesh is not None:
        if region is not None:
            from femto.planning.selection_blocks import selection_face_assignment, check_projection_inside_selection, _area
            from femto.planning.surface_trajectory import face_list_sha256
            intent, support, labels, owners = selection_face_assignment(region, np.asarray(mesh.vertices), np.asarray(mesh.faces))
            blocks=trajectory.blocks['items']
            if sorted(b['component_id'] for b in blocks)!=list(range(len(intent['components']))):
                raise ValueError('约束后工作块必须保留全部原选区组件')
            if any(s.block_index not in range(len(blocks)) for s in segments):
                raise ValueError('路线工作块索引无效')
            for index,block in enumerate(blocks):
                component=int(block['component_id'])
                ids=np.flatnonzero(owners==component)
                selected=intent['components'][component]
                if (block['block_index']!=index or block['face_count']!=len(ids)
                        or block['face_indices_sha256']!=face_list_sha256(ids)
                        or block['pixel_bounds']!=selected['bbox']
                        or block['selection']['selected_pixels']!=selected['pixel_count']
                        or abs(block['area_m2']-_area(np.asarray(mesh.vertices),np.asarray(mesh.faces),ids))>1e-12):
                    raise ValueError('约束后的工作块与原区域面或选择意图不一致')
            check_projection_inside_selection(segments, blocks, labels,
                np.asarray(region['view_reference']['undistorted_intrinsic']), np.asarray(region['view_reference']['image_size']))
        okay, _, _ = edge_checks(segments, mesh, params['quality_evidence'], params['effective_width_mm'],check,allowed_faces=allowed_faces)
        if not okay.all():raise ValueError('已保存路线违反几何质量或覆盖带边界约束')


def report_text(trajectory: SurfaceTrajectory) -> str:
    report=trajectory.gap_report['quality_constraint']
    routing=trajectory.gap_report.get('reconnection')
    continuity=[] if routing is None else [
        f"受约束贴面重新连接：{routing['before_chains']} 段合并为 {routing['after_chains']} 条连续路线，新增 {routing['added_connector_edges']} 条经整边核验的连接边。",
        '扫描折线保留真实三角面交点；换行在有效表面内寻路，搜索失败不证明物理上一定无法连通。',
    ]
    domain=trajectory.planner['parameters'].get('effective_domain')
    domain_lines=[] if domain is None else [
        f"有效表面域：{domain['eligible_face_count']} 面，排除 {domain['excluded_face_count']} 面；缺失范围 {len(domain['missing_regions'])} 处。",
        '有效域是质量合格面集合与宽度/残差边界谓词的交集，不是已认证的完整覆盖区域。',
        '候选原始断点分类（另存原值，不与本轮约束拒绝原因混淆）：'+str(dict(Counter(g['reason'] for g in trajectory.gap_report['candidate_gaps']))),
        *[f"缺失 {item['id']+1}：{item['reason']}，{item['pixel_count']} 像素，范围 {item['pixel_bounds']}；需检查或补扫。" for item in domain['missing_regions']],
    ]
    return '\n'.join([
        '几何质量与覆盖宽度联合约束路线（研发预览）',
        *continuity,
        *domain_lines,
        f"有效作用宽度 {report['effective_width_mm']:g} mm；局部参考残差余量最大 {report['local_residual_margin_max_mm']:g} mm。",
        f"预检查候选边 {report['candidate_edges']}，保留 {report['retained_edges']}，拒绝 {report['rejected_edges']}；未经验证的断点不连接。",
        f"原选区 {report['selected_pixels']} 像素；质量待检查/需补扫 {report['pending_pixels']} 像素；选区意图完整保留。",
        '拒绝原因：'+str(report['rejection_counts']),
        '参考残差是深度与网格的一致性观测，不是经标定的误差上界；配准和标定不确定性未知。',
        '覆盖带采用球包络与投影矩形保守检查，不能证明真实接触、剂量或治疗覆盖；原目标覆盖以同修订的独立覆盖证据为准。',
        '补扫须建立独立记录并重新核验、重新生成；旧路线不会自动升级或迁移到新几何。',
        'execution_ready=false；physical_accuracy_verified=false。',
    ])+'\n'
