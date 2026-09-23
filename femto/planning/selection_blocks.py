"""mesh-slice-serpentine-v4: work blocks are the physician's selection components.

A block is one 4-connected component of the authoritative selection mask (see
`femto.annotation.selection_intent`). Region faces are assigned to blocks by their centroid
pixel, exactly the rule that selected them. Inside a block the measured mesh may fall into
several fragments; that is a geometry diagnosis, never a block boundary: every fragment is
sliced and linked on its own surface, fragments are visited one after another and the seam
between them stays a gap. A block is `planned` only when it is exactly one continuous chain,
`incomplete` when chains exist but links failed, `unplannable` when no chain exists at all.
Blocks are separated by `selection_break` gaps. Nothing is ever bridged across missing
geometry, explicit unselected pixels, painted forbidden polygons or other blocks.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
import math

import numpy as np
import trimesh

from femto.annotation.selection_intent import (
    SUPPORT_CLASSES, UNCONFIRMED_CLASSES, component_labels, component_support_counts, components_of_pixels,
    intent_mask, support_classes, validate_selection_intent, validate_selection_support,
)
from femto.planning.surface_trajectory import (
    BIDIRECTIONAL_SELECTION_ALGORITHM, FACE_PORTAL_SELECTION_ALGORITHM, Gap, LINK_FAILURE_REASONS, ROW_TURN_FAILURE_REASONS,
    SELECTION_ALGORITHM, SELECTION_BLOCKS_SCHEMA,
    SELECTION_BLOCK_REASONS, ScanSegment, SurfaceTrajectory, _close, _length_split, face_components,
    face_list_sha256, validate_trajectory,
)


INTENT_MISSING_MESSAGE = ("区域导出缺少选择意图记录（旧导出）：无法区分明确未选与采集缺失，"
                          "请在固定 RGB 页面重新选择并导出该区域后再生成；旧轨迹仍可预览和播放")
REVISIT_CRITERION = "connector_sample_within_half_nominal_row_spacing_of_scan_sample_of_row_not_joined_by_that_connector"


def _project_uv(points: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    z = points[:, 2]
    ok = z > 0
    safe = np.where(ok, z, 1.0)
    uv = np.c_[intrinsic[0, 0] * points[:, 0] / safe + intrinsic[0, 2],
               intrinsic[1, 1] * points[:, 1] / safe + intrinsic[1, 2]]
    uv[~ok] = np.nan
    return uv


def selection_face_assignment(region: dict[str, Any], vertices: np.ndarray, faces: np.ndarray
                              ) -> tuple[dict[str, Any], dict[str, Any] | None, np.ndarray, np.ndarray]:
    """(intent, support, component label image, face -> selection component id)."""
    from femto.planning.mesh_slice_planner import Unplannable
    if not isinstance(region.get("selection_intent"), dict):
        raise Unplannable("selection_intent_missing", INTENT_MISSING_MESSAGE)
    view = region["view_reference"]
    intent = validate_selection_intent(region["selection_intent"], view["image_size"])
    support = None
    if region.get("selection_support") is not None:
        support = validate_selection_support(region["selection_support"], intent_mask(intent))
    labels = component_labels(intent)
    intrinsic = np.asarray(view["undistorted_intrinsic"], dtype=np.float64)
    owners = components_of_pixels(labels, _project_uv(vertices[faces].mean(axis=1), intrinsic))
    if np.any(owners < 0):
        raise ValueError("Region PLY contains faces whose centroid pixel is outside the selection mask; "
                         "the export is inconsistent with its selection intent")
    return intent, support, labels, owners


def _reading_key(bbox: list[float], component_id: int, corner: str) -> tuple[float, float, int]:
    x0, y0, x1, y1 = bbox
    return (y0 if "top" in corner else -y1, x0 if "left" in corner else -x1, component_id)


def _area(vertices: np.ndarray, faces: np.ndarray, ids: np.ndarray) -> float:
    if not len(ids):
        return 0.0
    triangles = vertices[faces[ids]]
    return float(np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                         triangles[:, 2] - triangles[:, 0]), axis=1).sum() / 2)


def _mask_centre(labels: np.ndarray, component_id: int) -> list[float]:
    ys, xs = np.nonzero(labels == component_id + 1)
    return [float(xs.mean()), float(ys.mean())]


def _fragment_descriptors(vertices: np.ndarray, faces: np.ndarray, ids: np.ndarray, components: np.ndarray,
                          intrinsic: np.ndarray, image_size: np.ndarray, corner: str) -> list[dict[str, Any]]:
    """Mesh fragments of one block in reading order; unprojectable fragments last."""
    from femto.planning.mesh_slice_planner import _pixel_extent
    descriptors = []
    for component in sorted(set(int(v) for v in components[ids])):
        fragment_ids = ids[components[ids] == component]
        bounds, center = _pixel_extent(vertices[np.unique(faces[fragment_ids])], intrinsic, image_size)
        key = ((1, 0.0, 0.0, component) if bounds is None else
               (0, bounds[1] if "top" in corner else -bounds[3], bounds[0] if "left" in corner else -bounds[2], component))
        descriptors.append({"mesh_component_id": component, "face_ids": fragment_ids,
                            "area_m2": _area(vertices, faces, fragment_ids),
                            "pixel_bounds": bounds, "center_pixel": center, "key": key})
    descriptors.sort(key=lambda item: item["key"])
    return descriptors


def _chord_classes(a: Any, b: Any, labels: np.ndarray, classes: np.ndarray | None, intrinsic: np.ndarray,
                   image_size: np.ndarray, step_m: float) -> tuple[bool, bool]:
    """(crosses explicitly unselected pixels, crosses geometry-unconfirmed selected pixels)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    count = max(2, int(math.ceil(float(np.linalg.norm(b - a)) / step_m)) + 1)
    uv = _project_uv(a[None] * (1 - np.linspace(0, 1, count))[:, None] + b[None] * np.linspace(0, 1, count)[:, None],
                     intrinsic)
    height, width = labels.shape
    finite = np.isfinite(uv).all(axis=1)
    pix = np.floor(uv[finite] + .5).astype(np.int64)
    inside = (pix[:, 0] >= 0) & (pix[:, 0] < width) & (pix[:, 1] >= 0) & (pix[:, 1] < height)
    pix = pix[inside]
    if not len(pix):
        return False, False
    unselected = bool(np.any(labels[pix[:, 1], pix[:, 0]] == 0))
    unconfirmed = False
    if classes is not None:
        unconfirmed = bool(np.any(np.isin(classes[pix[:, 1], pix[:, 0]], UNCONFIRMED_CLASSES)))
    return unselected, unconfirmed


def _refine_gap_causes(gaps: list[Gap], segments: list[ScanSegment], labels: np.ndarray,
                       classes: np.ndarray | None, intrinsic: np.ndarray, image_size: np.ndarray,
                       step_m: float) -> list[Gap]:
    """Tell explicit unselected holes from missing geometry using the intent; still `decision_use=none`."""
    refined = []
    for gap in gaps:
        if gap.reason != "surface_gap":
            refined.append(gap)
            continue
        a = segments[gap.gap_index].points[-1]["xyz_m"]
        b = segments[gap.gap_index + 1].points[0]["xyz_m"]
        unselected, unconfirmed = _chord_classes(a, b, labels, classes, intrinsic, image_size, step_m)
        cause = ("explicit_unselected_hole_likely" if unselected else
                 "geometry_unconfirmed_likely" if unconfirmed else gap.detail["inferred_cause"])
        refined.append(replace(gap, detail=dict(gap.detail, inferred_cause=cause)))
    return refined


def _plan_fragment(vertices: np.ndarray, faces: np.ndarray, ids: np.ndarray, mesh_component: int,
                   components: np.ndarray, params: dict[str, Any], region: dict[str, Any], intrinsic: np.ndarray,
                   image_size: np.ndarray, check: Callable[[], None], budget: dict[str, int], row_offset: int,
                   block_index: int, labels: np.ndarray, classes: np.ndarray | None,
                   allowed: np.ndarray, algorithm_version: str = SELECTION_ALGORITHM) -> dict[str, Any]:
    from femto.planning.mesh_slice_planner import _eligible_turns, _prepare_block
    from femto.planning.surface_row_turns import SurfaceLinks, connect_links
    prepared = _prepare_block(vertices, faces, ids, mesh_component, components, params, region, intrinsic,
                              image_size, check, budget, row_offset, block_index, allowed)
    step_m = params["sample_step_mm"] / 1000
    gaps = _refine_gap_causes(prepared["gaps"], prepared["segments"], labels, classes, intrinsic, image_size, step_m)
    from femto.planning.surface_trajectory import OPTIMIZED_SELECTION_ALGORITHM
    from femto.planning.surface_link_optimization import OptimizedSurfaceLinks
    router_class = OptimizedSurfaceLinks if algorithm_version == OPTIMIZED_SELECTION_ALGORITHM else SurfaceLinks
    if algorithm_version == FACE_PORTAL_SELECTION_ALGORITHM:
        from femto.planning.surface_face_routing import FacePortalSurfaceLinks
        router_class = FacePortalSurfaceLinks
    elif algorithm_version == BIDIRECTIONAL_SELECTION_ALGORITHM:
        from femto.planning.surface_bidirectional_routing import BidirectionalFacePortalSurfaceLinks
        router_class = BidirectionalFacePortalSurfaceLinks
    router = router_class(SimpleNamespace(vertices=vertices, faces=faces), region, prepared["frame"], check, ids,
                          allowed=allowed)
    linked = connect_links(prepared["segments"], gaps, _eligible_turns(prepared["ordered_records"]), router,
                           step_m, check, e2=prepared["e2"], used=budget["points"])
    budget["points"] += linked["added"]
    return dict(prepared, gaps=gaps, **linked)


def _mark_revisits(chains: list[list[dict[str, Any]]], spacing_m: float,
                   check: Callable[[], None]) -> dict[str, Any]:
    """Flag connector samples that pass over rows they do not join; report only, never a decision."""
    import open3d as o3d
    scan = [(point["xyz_m"], point["row_index"]) for chain in chains for point in chain
            if point.get("role") != "connector"]
    result = {"connector_sample_count": 0, "connector_length_m": 0.0, "rows": [],
              "criterion": REVISIT_CRITERION, "radius_m": spacing_m / 2,
              "semantics": "nominal_row_spacing_not_treatment_head_width", "decision_use": "none"}
    if not scan:
        return result
    xyz = np.asarray([item[0] for item in scan], dtype=np.float64)
    rows = np.asarray([item[1] for item in scan], dtype=np.int64)
    tree = o3d.geometry.KDTreeFlann(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz)))
    revisited: set[int] = set()
    count = 0
    length = 0.0
    queries = 0
    for chain in chains:
        index = 0
        last_row = None
        while index < len(chain):
            point = chain[index]
            if point.get("role") != "connector":
                last_row = point["row_index"]
                index += 1
                continue
            start = index
            while index < len(chain) and chain[index].get("role") == "connector":
                index += 1
            joined = {last_row, chain[index]["row_index"] if index < len(chain) else chain[start]["row_index"]}
            for offset in range(start, index):
                queries += 1
                if queries % 256 == 0:
                    check()
                sample = chain[offset]
                _, found, _ = tree.search_radius_vector_3d(np.asarray(sample["xyz_m"], dtype=np.float64), spacing_m / 2)
                others = sorted({int(rows[i]) for i in found if int(rows[i]) not in joined})
                if others:
                    sample["revisit"] = True
                    count += 1
                    revisited.update(others)
                    length += float(np.linalg.norm(np.asarray(sample["xyz_m"]) - np.asarray(chain[offset - 1]["xyz_m"])))
    result.update(connector_sample_count=count, connector_length_m=length, rows=sorted(revisited))
    return result


def coverage_status(unplannable_blocks: int, uncovered_fragments: int, unconfirmed: int | None) -> str:
    """Coverage is checked apart from continuity: selected pixels without measured geometry keep
    their requirement, so they make coverage partial instead of silently vanishing."""
    if unplannable_blocks or uncovered_fragments or (unconfirmed or 0) > 0:
        return "partial"
    return "complete" if unconfirmed == 0 else "complete_fragments_geometry_support_unknown"


def _seam_gap(previous: list[dict[str, Any]], following: list[dict[str, Any]]) -> Gap:
    from femto.planning.mesh_slice_planner import _direction, _gap_detail
    before, after = previous[-1], following[0]
    return Gap(0, "disconnected_surface",
               dict(_gap_detail("not_applicable"), link_failure_reason="disconnected_measured_surface"),
               {"route_index": before["route_index"], "xyz_m": before["xyz_m"]},
               {"route_index": after["route_index"], "xyz_m": after["xyz_m"]},
               _direction(previous), previous[0]["row_index"], after["row_index"],
               float(np.linalg.norm(np.asarray(after["xyz_m"]) - np.asarray(before["xyz_m"]))))


def _walk_connectors(segments: list[ScanSegment]) -> tuple[int, int, int, float]:
    """(connected row turns, connected detours, revisit samples, revisit length) from chains alone."""
    turns = detours = revisits = 0
    revisit_length = 0.0
    for segment in segments:
        points = segment.points
        index = 0
        last_row = None
        while index < len(points):
            point = points[index]
            if point.get("role") != "connector":
                last_row = point["row_index"]
                index += 1
                continue
            start = index
            while index < len(points) and points[index].get("role") == "connector":
                if points[index].get("revisit") is True:
                    revisits += 1
                    revisit_length += float(np.linalg.norm(np.asarray(points[index]["xyz_m"])
                                                           - np.asarray(points[index - 1]["xyz_m"])))
                index += 1
            next_row = points[index]["row_index"] if index < len(points) else points[start]["row_index"]
            if next_row != last_row:
                turns += 1
            else:
                detours += 1
    return turns, detours, revisits, revisit_length


def _depth(points: np.ndarray) -> dict[str, float]:
    return {"depth_min_m": float(points.min()), "depth_max_m": float(points.max()),
            "depth_span_m": float(points.max() - points.min())}


def plan_selection_blocks(manifest_path: Path, manifest: dict[str, Any], mesh: trimesh.Trimesh,
                          params: dict[str, Any], check: Callable[[], None], *,
                          algorithm_version: str = SELECTION_ALGORITHM) -> SurfaceTrajectory:
    from femto.planning.mesh_slice_planner import (
        Unplannable, _depth_statistics, _direction, _gap_detail, _level_segments, _merge_spacing, _pca_frame,
        _row_report, _source_record)
    from femto.planning.surface_row_turns import chain_items
    from femto.planning.surface_spacing_report import spacing_report as measure_spacing
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    region = manifest["region"]
    intrinsic = np.asarray(region["view_reference"]["undistorted_intrinsic"], dtype=np.float64)
    image_size = np.asarray(region["view_reference"]["image_size"], dtype=np.int64)
    intent, support, labels, owners = selection_face_assignment(region, vertices, faces)
    classes = None if support is None else support_classes(support)
    components = face_components(faces)
    corner = params["start_corner"]
    spacing_m = params["row_spacing_mm"] / 1000
    ordered = sorted(intent["components"], key=lambda item: _reading_key(item["bbox"], item["component_id"], corner))
    budget = {"points": 0}
    spacing_budget = {"expansions": 0, "distance_pairs": 0}
    offsets = {"row": 0, "route": 0, "segment": 0, "gap": 0, "pre_segment": 0}
    items: list[ScanSegment | Gap] = []
    blocks: list[dict[str, Any]] = []
    dropped_fragments: list[dict[str, Any]] = []
    row_report: list[dict[str, Any]] = []
    spacing_reports: list[dict[str, Any]] = []
    scan_rows: list[int] = []
    final_gaps: list[Gap] = []
    spacing_mesh = SimpleNamespace(vertices=vertices, triangles=faces)
    for block_index, component in enumerate(ordered):
        check()
        cid = component["component_id"]
        ids = np.flatnonzero(owners == cid)
        pixel_counts = component_support_counts(support, labels, cid)
        record: dict[str, Any] = {
            "block_index": block_index, "component_id": cid, "face_count": int(len(ids)),
            "face_index_min": int(ids.min()) if len(ids) else 0, "face_index_max": int(ids.max()) if len(ids) else 0,
            "face_indices_sha256": face_list_sha256(ids), "area_m2": _area(vertices, faces, ids),
            "pixel_bounds": [float(v) for v in component["bbox"]], "center_pixel": _mask_centre(labels, cid),
            "selection": {"selected_pixels": component["pixel_count"], "hole_count": len(component["hole_ids"]),
                          "support": pixel_counts},
        }
        fragments = (_fragment_descriptors(vertices, faces, ids, components, intrinsic, image_size, corner)
                     if len(ids) else [])
        fragment_records: list[dict[str, Any]] = []
        allowed = labels == cid + 1  # every sample and every edge must project onto this block's own pixels
        block_chains: list[list[dict[str, Any]]] = []
        block_kept: list[Gap] = []
        totals = Counter()
        lengths_scan = 0.0
        connector_length = 0.0
        turn_failures: Counter = Counter()
        link_failures: Counter = Counter()
        depth_values: list[np.ndarray] = []
        block_row_first = offsets["row"]
        block_dropped = 0
        block_segments: list[ScanSegment] = []
        block_reports: list[dict[str, Any]] = []
        for fragment_index, descriptor in enumerate(fragments):
            check()
            fragment_ids = descriptor["face_ids"]
            fragment = {"fragment_index": fragment_index, "mesh_component_id": descriptor["mesh_component_id"],
                        "face_count": int(len(fragment_ids)), "face_index_min": int(fragment_ids.min()),
                        "face_index_max": int(fragment_ids.max()), "face_indices_sha256": face_list_sha256(fragment_ids),
                        "area_m2": descriptor["area_m2"], "pixel_bounds": descriptor["pixel_bounds"],
                        "center_pixel": descriptor["center_pixel"]}
            try:
                planned = _plan_fragment(vertices, faces, fragment_ids, descriptor["mesh_component_id"], components,
                                         params, region, intrinsic, image_size, check, budget, offsets["row"],
                                         block_index, labels, classes, allowed, algorithm_version)
            except Unplannable as error:
                fragment.update(status="unplannable", reason=error.reason, message=str(error))
                if error.details:
                    extra = [dict(item, fragment_index=fragment_index) for item in error.details.get("dropped_fragments", [])]
                    dropped_fragments.extend(extra)
                    block_dropped += len(extra)
                    fragment["dropped_fragment_count"] = len(extra)
                fragment_records.append(fragment)
                continue
            if block_chains:
                block_kept.append(_seam_gap(block_chains[-1], planned["chains"][0]))
                link_failures["disconnected_measured_surface"] += 1
            block_chains.extend(planned["chains"])
            block_kept.extend(planned["kept"])
            row_count = len({segment.row_index for segment in planned["segments"]})
            scan_length = float(sum(planned["lengths"]))
            lengths_scan += scan_length
            connector_length += planned["length"]
            totals.update(planned["connected"])
            for gap in planned["kept"]:
                link_failures[gap.detail["link_failure_reason"]] += 1
                if gap.reason == "row_turn":
                    turn_failures[gap.detail["row_turn_failure_reason"]] += 1
            depth_values.append(np.asarray([point["xyz_m"][2] for chain in planned["chains"] for point in chain]))
            row_report.extend(_row_report(planned["segments"], planned["lengths"], [], block_index))
            check()
            report = measure_spacing(spacing_mesh, {"triangle_indices": [int(v) for v in fragment_ids]},
                                     {"segments": _level_segments(planned["ordered_records"], planned["segments"])},
                                     params["row_spacing_mm"], params["row_spacing_mm"] * params["spacing_tolerance"],
                                     check, budget=spacing_budget)
            report["measurements"] = [dict(row, chain_index=row["chain_index"] + offsets["pre_segment"],
                                           block_index=block_index, fragment_index=fragment_index)
                                      for row in report["measurements"]]
            report.update(block_index=block_index, fragment_index=fragment_index)
            block_reports.append(report)
            scan_rows.extend(segment.row_index for segment in planned["segments"])
            extra = [dict(item, fragment_index=fragment_index) for item in planned["dropped_fragments"]]
            dropped_fragments.extend(extra)
            block_dropped += len(extra)
            block_segments.extend(planned["segments"])
            fragment.update(status="planned", sweep_frame=planned["frame"], row_index_first=offsets["row"],
                            row_count=row_count, chain_count=len(planned["chains"]),
                            internal_gap_count=len(planned["kept"]), scan_length_m=scan_length,
                            connector_length_m=planned["length"],
                            connected_row_turns=int(planned["connected"]["row_turn"]),
                            connected_detours=int(planned["connected"]["detour"]),
                            link_methods=dict(sorted(planned["methods"].items())),
                            dropped_fragment_count=len(extra))
            fragment_records.append(fragment)
            offsets["row"] += row_count
            offsets["pre_segment"] += len(planned["segments"])
        uncovered = [fragment for fragment in fragment_records if fragment["status"] != "planned"]
        record["fragments"] = fragment_records
        record["mesh_fragment_count"] = len(fragment_records)
        record["planned_fragment_count"] = len(fragment_records) - len(uncovered)
        record["coverage"] = {
            "status": ("complete" if fragment_records and not uncovered and (pixel_counts or {}).get("geometry_unconfirmed_pixels", 0) == 0
                       else "partial" if (fragment_records and len(uncovered) < len(fragment_records)) else "none"),
            "fragment_status": "complete" if fragment_records and not uncovered else "partial" if len(uncovered) < len(fragment_records) else "none",
            "uncovered_fragment_count": len(uncovered),
            "uncovered_face_count": int(sum(fragment["face_count"] for fragment in uncovered)),
            "uncovered_area_m2": float(sum(fragment["area_m2"] for fragment in uncovered)),
            "geometry_unconfirmed_pixels": None if pixel_counts is None else pixel_counts["geometry_unconfirmed_pixels"],
            "dropped_fragment_count": block_dropped,
            "treatment_coverage_claim": "none_without_effective_head_width",
        }
        if not block_chains:
            reason = "no_supported_geometry" if not fragment_records else "all_fragments_unplannable"
            message = ("选区内没有任何有 Mesh 支持的三角面（几何待确认 / 需要补采）" if not fragment_records else
                       "块内每个网格碎片都无法规划：" + "; ".join(
                           f"碎片 {f['fragment_index'] + 1}（{f['face_count']} 面）{f['reason']}" for f in fragment_records))
            record.update(status="unplannable", reason=reason, message=message,
                          continuity={"status": "none", "chain_count": 0, "internal_gap_count": 0,
                                      "link_failure_counts": {}},
                          revisit=None, dropped_fragment_count=block_dropped)
            blocks.append(record)
            continue
        revisit = _mark_revisits(block_chains, spacing_m, check)
        if items:
            inter_gap = offsets["gap"]
            offsets["gap"] += 1
        else:
            inter_gap = None
        block_items, route_end = chain_items(block_chains, block_kept, route_start=offsets["route"],
                                             segment_start=offsets["segment"], gap_start=offsets["gap"],
                                             block_index=block_index)
        segments_final = [item for item in block_items if isinstance(item, ScanSegment)]
        kept = [item for item in block_items if isinstance(item, Gap)]
        if inter_gap is not None:
            previous = items[-1]
            assert isinstance(previous, ScanSegment)
            first = segments_final[0]
            before, after = previous.points[-1], first.points[0]
            gap = Gap(inter_gap, "selection_break", _gap_detail("not_applicable"),
                      {"route_index": before["route_index"], "xyz_m": before["xyz_m"]},
                      {"route_index": after["route_index"], "xyz_m": after["xyz_m"]},
                      _direction(previous.points), previous.row_index, first.row_index,
                      float(np.linalg.norm(np.asarray(after["xyz_m"]) - before["xyz_m"])))
            items.append(gap)
            final_gaps.append(gap)
        items.extend(block_items)
        final_gaps.extend(kept)
        spacing_reports.extend(block_reports)
        depth = np.concatenate(depth_values)
        row_count = offsets["row"] - block_row_first
        record.update(
            status="planned" if len(segments_final) == 1 else "incomplete",
            sweep_frame=dict(next(f["sweep_frame"] for f in fragment_records if f["status"] == "planned"),
                             role="first_planned_fragment_frame"),
            segment_index_first=segments_final[0].segment_index, segment_count=len(segments_final),
            gap_count=len(kept), gap_counts=dict(sorted(Counter(g.reason for g in kept).items())),
            point_count=route_end - offsets["route"], row_index_first=block_row_first, row_count=row_count,
            scan_length_m=lengths_scan, connector_length_m=connector_length,
            connected_row_turns=int(totals["row_turn"]), connected_detours=int(totals["detour"]),
            row_turn_failure_counts=dict(sorted(turn_failures.items())),
            link_failure_counts=dict(sorted(link_failures.items())),
            dropped_fragment_count=block_dropped,
            continuity={"status": "single_chain" if len(segments_final) == 1 else "broken",
                        "chain_count": len(segments_final), "internal_gap_count": len(kept),
                        "link_failure_counts": dict(sorted(link_failures.items()))},
            revisit=revisit, **_depth(depth))
        blocks.append(record)
        offsets.update(route=route_end, segment=segments_final[-1].segment_index + 1, gap=offsets["gap"] + len(kept))
    with_chain = [block for block in blocks if block["status"] != "unplannable"]
    if not with_chain:
        raise Unplannable("all_blocks_unplannable",
                          "No block can be planned: " + "; ".join(
                              f"block {b['block_index'] + 1} ({b['face_count']} faces): {b['reason']}" for b in blocks),
                          details={"blocks": blocks})
    for row in row_report:
        row["gap_count"] = sum(gap.row_index_from == row["row_index"] for gap in final_gaps)
    segments_all = [item for item in items if isinstance(item, ScanSegment)]
    gap_counts = Counter(gap.reason for gap in final_gaps)
    all_z = np.asarray([point["xyz_m"][2] for segment in segments_all for point in segment.points])
    planned_blocks = [block for block in blocks if block["status"] == "planned"]
    incomplete_blocks = [block for block in blocks if block["status"] == "incomplete"]
    unplannable_blocks = [block for block in blocks if block["status"] == "unplannable"]
    uncovered_faces = int(sum(block["coverage"]["uncovered_face_count"] for block in blocks))
    uncovered_area = float(sum(block["coverage"]["uncovered_area_m2"] for block in blocks))
    uncovered_fragments = int(sum(block["coverage"]["uncovered_fragment_count"] for block in blocks))
    unconfirmed = None if support is None else int(support["geometry_unconfirmed_pixel_count"])
    try:
        display_frame = dict(_pca_frame(mesh, dict(params, row_direction="auto"))[4],
                             role="display_only_whole_mesh_frame")
    except Unplannable:
        display_frame = dict(with_chain[0]["sweep_frame"], role="display_only_first_planned_block_frame")
    link_failure_counts = Counter()
    cause_counts = Counter()
    for gap in final_gaps:
        if "link_failure_reason" in gap.detail:
            link_failure_counts[gap.detail["link_failure_reason"]] += 1
        cause_counts[gap.reason + ":" + gap.detail["inferred_cause"]] += 1
    statistics = {
        "segment_count": len(segments_all), "gap_count": len(final_gaps),
        "gap_counts": dict(sorted(gap_counts.items())), "point_count": offsets["route"],
        "scan_length_m": float(sum(block["scan_length_m"] for block in with_chain)),
        "connector_length_m": float(sum(block["connector_length_m"] for block in with_chain)),
        "connected_row_turns": int(sum(block["connected_row_turns"] for block in with_chain)),
        "connected_detours": int(sum(block["connected_detours"] for block in with_chain)),
        "row_count": len(row_report), **_depth_statistics(all_z),
        "block_count": len(blocks), "plannable_block_count": len(with_chain),
        "planned_block_count": len(planned_blocks), "incomplete_block_count": len(incomplete_blocks),
        "unplannable_block_count": len(unplannable_blocks),
        "inter_block_gap_count": int(gap_counts.get("selection_break", 0)),
        "continuity_status": "complete" if len(planned_blocks) == len(blocks) else "partial",
        "coverage_status": coverage_status(len(unplannable_blocks), uncovered_fragments, unconfirmed),
        "geometry_status": ("unknown_without_support_record" if unconfirmed is None else
                            "all_selected_pixels_supported" if unconfirmed == 0 else "unconfirmed_pixels_present"),
        "unplannable_face_count": uncovered_faces, "unplannable_area_m2": uncovered_area,
        "uncovered_fragment_count": uncovered_fragments, "uncovered_face_count": uncovered_faces,
        "selection_component_count": len(blocks), "mesh_fragment_count": int(sum(block["mesh_fragment_count"] for block in blocks)),
        "geometry_unconfirmed_pixel_count": unconfirmed,
        "revisit_connector_sample_count": int(sum(block["revisit"]["connector_sample_count"] for block in with_chain)),
        "revisit_connector_length_m": float(sum(block["revisit"]["connector_length_m"] for block in with_chain)),
        "link_failure_counts": dict(sorted(link_failure_counts.items())),
        "treatment_coverage_claim": "none_without_effective_head_width",
    }
    statistics["total_length_m"] = statistics["scan_length_m"] + statistics["connector_length_m"]
    block_record = {
        "schema": SELECTION_BLOCKS_SCHEMA,
        "block_definition": "one block per 4-connected component of the physician selection mask; region faces "
                            "assigned by centroid pixel; mesh connectivity is only a fragment diagnosis",
        "order_rule": {"visit_order": "selection_mask_bbox_reading_order_from_start_corner",
                       "start_corner": corner,
                       "keys": "top: bbox y0 ascending / bottom: bbox y1 descending, then left: x0 ascending / right: x1 descending",
                       "tie_break": "selection component id ascending (raster order of first selected pixel)"},
        "fragment_rule": "mesh fragments of one block are visited in reference-image reading order; the seam "
                         "between two measured fragments is a disconnected_surface gap, never bridged",
        "link_rule": {"first": "frozen v2/v3.1 boundary-band turn for adjacent rows, accepted only if every edge stays on the block's selected pixels",
                      "then": "shortest path along the fragment's own mesh edges; every edge lies on one selected "
                              "triangle, stays inside the reference image, never touches a painted forbidden polygon and "
                              "projects only onto this block's selected pixels (sampled at least once per pixel)",
                      "never": "across mesh gaps, explicit unselected pixels, other fragments or other blocks"},
        "scan_rule": "scan samples projecting onto pixels outside the block's selection component are dropped; "
                     "no scan or connector edge enters explicitly unselected pixels",
        "selection_binding": {"intent_sha256": intent["selected_mask_sha256"],
                              "support_sha256": None if support is None else support["class_sha256"],
                              "component_count": len(intent["components"]),
                              "hole_count": len(intent["explicit_unselected_holes"]),
                              "missing_geometry_policy": intent["missing_geometry_policy"]},
        "items": blocks,
    }
    gap_report = {
        "counts": dict(sorted(gap_counts.items())), "dropped_fragments": dropped_fragments,
        "dropped_fragment_count": len(dropped_fragments),
        "unplannable_blocks": [{key: block[key] for key in ("block_index", "component_id", "reason", "message", "face_count", "area_m2")}
                               for block in unplannable_blocks],
        "incomplete_blocks": [{"block_index": block["block_index"], "component_id": block["component_id"],
                               "chain_count": block["continuity"]["chain_count"],
                               "internal_gap_count": block["continuity"]["internal_gap_count"],
                               "link_failure_counts": block["link_failure_counts"]} for block in incomplete_blocks],
        "row_turn_failure_counts": dict(sorted(Counter(
            gap.detail["row_turn_failure_reason"] for gap in final_gaps
            if gap.reason == "row_turn" and "row_turn_failure_reason" in gap.detail).items())),
        "link_failure_counts": dict(sorted(link_failure_counts.items())),
        "gap_cause_counts": dict(sorted(cause_counts.items())),
        "geometry_unconfirmed": None if support is None else {
            "pixel_count": unconfirmed, "by_reason": {name: support["counts"][name] for code, name in SUPPORT_CLASSES.items()
                                                       if code in UNCONFIRMED_CLASSES},
            "derived_surface": support["derived_surface"],
            "meaning": "selected pixels without measured mesh keep their coverage requirement; no XYZ was invented"},
        "missing_geometry_policy": intent["missing_geometry_policy"],
    }
    if algorithm_version != SELECTION_ALGORITHM:
        block_record["link_rule"]["then"] += ("; locally shorten inside that existing triangle corridor using same-face "
            "chords and manifold shared-edge crossings; preserve endpoints, check every edge against the selection "
            "and forbidden polygons; never increase connector length; not global geodesic or smoothing")
    if algorithm_version == FACE_PORTAL_SELECTION_ALGORITHM:
        block_record["link_rule"]["then"] += ("; when the retained connector search fails, search supported face-interior "
            "paths through manifold shared-edge portals inside the same measured fragment and selected work block; "
            "recheck every complete edge, never bridge missing geometry or explicitly unselected pixels")
    elif algorithm_version == BIDIRECTIONAL_SELECTION_ALGORITHM:
        block_record["link_rule"]["then"] += ("; when the retained connector search fails, use balanced bidirectional "
            "search through the finite supported manifold shared-edge portal graph inside the same measured fragment "
            "and selected work block; preserve complete-edge selection and forbidden checks, never bridge missing "
            "geometry or explicitly unselected pixels; retained v4.2 reproduces with its own search")
    planner = {"kind": "mesh_slice", "algorithm_version": algorithm_version,
               "parameters": dict(params, connect_rows=True, block_planning=True, selection_blocks=True),
               "sweep_frame": display_frame}
    source = dict(_source_record(manifest_path, manifest), selection_intent_sha256=intent["selected_mask_sha256"])
    trajectory = SurfaceTrajectory(
        manifest["origin_reference"], planner, source, items, statistics, gap_report,
        _merge_spacing(spacing_reports, params, spacing_budget, scan_rows), row_report,
        manifest.get("data_layers"), None, block_record)
    validate_trajectory(trajectory, mesh=mesh, region=region)
    return trajectory


def _edge_state(p: np.ndarray, q: np.ndarray, intrinsic: np.ndarray, image_size: np.ndarray,
                polygons: list[np.ndarray], *, polygon_index: Any = None) -> int:
    from femto.planning.mesh_slice_planner import _project
    from femto.planning.surface_polygon_index import PreparedPolygonIndex
    uv = [_project(v, intrinsic, image_size) for v in (p, q)]
    if any(v is None for v in uv):
        return 1
    u, v = map(np.asarray, uv)
    index = PreparedPolygonIndex(polygons) if polygon_index is None else polygon_index
    return 2 if index.blocks_segment(u, v) else 0


def check_selection_blocks(trajectory: SurfaceTrajectory, blocks: list[dict[str, Any]], segments: list[ScanSegment],
                           gaps: list[Gap], vertices: Any, faces: Any, region: Any) -> None:
    """v4 consistency: bookkeeping from ordered_items, then binding to mesh + selection intent when given."""
    stats = trajectory.statistics if isinstance(trajectory.statistics, dict) else {}
    with_chain = [block for block in blocks if block["status"] != "unplannable"]
    owned_by = {block["block_index"]: [segment for segment in segments if segment.block_index == block["block_index"]]
                for block in blocks}
    if any(segment.block_index not in {block["block_index"] for block in with_chain} for segment in segments):
        raise ValueError("Scan segment refers to a block without chains")
    for index, gap in enumerate(gaps):
        crossing = segments[index].block_index != segments[index + 1].block_index
        if crossing != (gap.reason == "selection_break"):
            raise ValueError("Exactly the inter-block gaps are selection_break in selection-block mode")
        if not crossing:
            if gap.detail.get("link_failure_reason") not in LINK_FAILURE_REASONS:
                raise ValueError("Every intra-block gap needs a link_failure_reason")
            if gap.reason == "row_turn" and gap.detail.get("row_turn_failure_reason") not in ROW_TURN_FAILURE_REASONS:
                raise ValueError("Every intra-block row_turn gap needs a row_turn_failure_reason")
    next_row = 0
    totals = Counter()
    lengths = {"scan": 0.0, "connector": 0.0, "revisit": 0.0}
    turn_failures: Counter = Counter()
    link_failures: Counter = Counter()
    for block in blocks:
        if type(block.get("mesh_fragment_count")) is not int or not isinstance(block.get("fragments"), list) \
                or len(block["fragments"]) != block["mesh_fragment_count"] or not isinstance(block.get("coverage"), dict) \
                or not isinstance(block.get("continuity"), dict) or not isinstance(block.get("selection"), dict):
            raise ValueError("Selection block record is missing fragment/coverage/continuity/selection fields")
        if type(block.get("area_m2")) is not float or not math.isfinite(block["area_m2"]) or block["area_m2"] < 0:
            raise ValueError("Invalid block area")
        if block["face_count"] == 0 and (block["face_index_min"] or block["face_index_max"]):
            raise ValueError("A block without faces has no face index range")
        planned_fragments = [fragment for fragment in block["fragments"] if fragment.get("status") == "planned"]
        uncovered = [fragment for fragment in block["fragments"] if fragment.get("status") != "planned"]
        coverage = block["coverage"]
        unconfirmed_pixels = ((block["selection"].get("support") or {}).get("geometry_unconfirmed_pixels") or 0)
        expected_status = ("complete" if block["fragments"] and not uncovered and unconfirmed_pixels == 0
                           else "partial" if (block["fragments"] and len(uncovered) < len(block["fragments"])) else "none")
        if (block.get("planned_fragment_count") != len(planned_fragments)
                or coverage.get("status") != expected_status
                or coverage.get("uncovered_fragment_count") != len(uncovered)
                or coverage.get("uncovered_face_count") != sum(fragment["face_count"] for fragment in uncovered)
                or not _close(coverage.get("uncovered_area_m2"), float(sum(fragment["area_m2"] for fragment in uncovered)))
                or sum(fragment["face_count"] for fragment in block["fragments"]) != block["face_count"]):
            raise ValueError("Block fragment coverage bookkeeping does not match its fragments")
        owned = owned_by[block["block_index"]]
        if block["status"] == "unplannable":
            if owned or planned_fragments or block["continuity"].get("chain_count") != 0:
                raise ValueError("An unplannable block must not own chains")
            continue
        points = [point for segment in owned for point in segment.points]
        if (len(owned) != block["segment_count"] or not owned or owned[0].segment_index != block["segment_index_first"]
                or [segment.segment_index for segment in owned] != list(range(owned[0].segment_index, owned[0].segment_index + len(owned)))
                or len(points) != block["point_count"]):
            raise ValueError("Block segment/point bookkeeping does not match ordered_items")
        rows = sorted({point["row_index"] for point in points})
        if block.get("row_index_first") != next_row or rows != list(range(next_row, next_row + block["row_count"])):
            raise ValueError("Block row bookkeeping does not match its scan points")
        next_row += block["row_count"]
        if sum(fragment["row_count"] for fragment in planned_fragments) != block["row_count"]:
            raise ValueError("Fragment row counts do not add up to the block row count")
        first, last = owned[0].segment_index, owned[-1].segment_index
        inside = [gap for index, gap in enumerate(gaps) if first <= index < last]
        if block["status"] == "planned" and (inside or len(owned) != 1):
            raise ValueError("A planned selection block must be one chain without internal gaps")
        counts = dict(sorted(Counter(gap.reason for gap in inside).items()))
        turns, detours, revisits, revisit_length = _walk_connectors(owned)
        split = [_length_split(segment.points) for segment in owned]
        scan, connector = float(sum(v[0] for v in split)), float(sum(v[1] for v in split))
        links = dict(sorted(Counter(gap.detail["link_failure_reason"] for gap in inside).items()))
        turn_reasons = dict(sorted(Counter(gap.detail["row_turn_failure_reason"] for gap in inside if gap.reason == "row_turn").items()))
        continuity = block["continuity"]
        revisit = block.get("revisit") if isinstance(block.get("revisit"), dict) else {}
        z = np.asarray([point["xyz_m"][2] for point in points], dtype=np.float64)
        checks = {
            "gap_count": block.get("gap_count") == len(inside),
            "gap_counts": block.get("gap_counts") == counts,
            "connected_row_turns": block.get("connected_row_turns") == turns,
            "connected_detours": block.get("connected_detours") == detours,
            "scan_length_m": _close(block.get("scan_length_m"), scan),
            "connector_length_m": _close(block.get("connector_length_m"), connector),
            "link_failure_counts": block.get("link_failure_counts") == links,
            "row_turn_failure_counts": block.get("row_turn_failure_counts") == turn_reasons,
            "continuity": (continuity.get("chain_count") == len(owned) and continuity.get("internal_gap_count") == len(inside)
                           and continuity.get("status") == ("single_chain" if len(owned) == 1 else "broken")
                           and continuity.get("link_failure_counts") == links),
            "revisit": (revisit.get("connector_sample_count") == revisits
                        and _close(revisit.get("connector_length_m"), revisit_length)),
            "depth": (_close(block.get("depth_min_m"), float(z.min())) and _close(block.get("depth_max_m"), float(z.max()))
                      and _close(block.get("depth_span_m"), float(z.max() - z.min()))),
        }
        for key, ok in checks.items():
            if not ok:
                raise ValueError(f"Selection block bookkeeping mismatch: {key}")
        totals.update({"turns": turns, "detours": detours, "revisits": revisits, "rows": block["row_count"]})
        lengths["scan"] += scan
        lengths["connector"] += connector
        lengths["revisit"] += revisit_length
        turn_failures.update(turn_reasons)
        link_failures.update(links)
    planned = [block for block in blocks if block["status"] == "planned"]
    incomplete = [block for block in blocks if block["status"] == "incomplete"]
    unplannable = [block for block in blocks if block["status"] == "unplannable"]
    gap_counts = dict(sorted(Counter(gap.reason for gap in gaps).items()))
    z = np.asarray([point["xyz_m"][2] for segment in segments for point in segment.points], dtype=np.float64)
    uncovered_faces = sum(block["coverage"]["uncovered_face_count"] for block in blocks)
    uncovered_fragments = sum(block["coverage"]["uncovered_fragment_count"] for block in blocks)
    expected = {
        "block_count": len(blocks), "plannable_block_count": len(with_chain), "planned_block_count": len(planned),
        "incomplete_block_count": len(incomplete), "unplannable_block_count": len(unplannable),
        "segment_count": len(segments), "point_count": sum(len(segment.points) for segment in segments),
        "gap_count": len(gaps), "gap_counts": gap_counts, "row_count": totals["rows"],
        "connected_row_turns": totals["turns"], "connected_detours": totals["detours"],
        "inter_block_gap_count": gap_counts.get("selection_break", 0),
        "continuity_status": "complete" if len(planned) == len(blocks) else "partial",
        "coverage_status": coverage_status(len(unplannable), uncovered_fragments, stats.get("geometry_unconfirmed_pixel_count")),
        "unplannable_face_count": uncovered_faces, "uncovered_face_count": uncovered_faces,
        "uncovered_fragment_count": uncovered_fragments,
        "selection_component_count": len(blocks),
        "mesh_fragment_count": sum(block["mesh_fragment_count"] for block in blocks),
        "revisit_connector_sample_count": totals["revisits"],
        "link_failure_counts": dict(sorted(link_failures.items())),
        "treatment_coverage_claim": "none_without_effective_head_width",
    }
    for key, value in expected.items():
        if stats.get(key) != value:
            raise ValueError(f"Selection block statistics mismatch: {key}")
    closeness = {
        "scan_length_m": (stats.get("scan_length_m"), lengths["scan"]),
        "connector_length_m": (stats.get("connector_length_m"), lengths["connector"]),
        "total_length_m": (stats.get("total_length_m"), lengths["scan"] + lengths["connector"]),
        "revisit_connector_length_m": (stats.get("revisit_connector_length_m"), lengths["revisit"]),
        "unplannable_area_m2": (stats.get("unplannable_area_m2"), float(sum(block["coverage"]["uncovered_area_m2"] for block in blocks))),
        "depth_min_m": (stats.get("depth_min_m"), float(z.min())),
        "depth_max_m": (stats.get("depth_max_m"), float(z.max())),
        "depth_span_m": (stats.get("depth_span_m"), float(z.max() - z.min())),
    }
    for key, (saved, computed) in closeness.items():
        if not _close(saved, computed):
            raise ValueError(f"Selection block statistics mismatch: {key}")
    report = trajectory.gap_report if isinstance(trajectory.gap_report, dict) else {}
    if (report.get("counts") != gap_counts or report.get("link_failure_counts") != dict(sorted(link_failures.items()))
            or report.get("row_turn_failure_counts") != dict(sorted(turn_failures.items()))):
        raise ValueError("Selection block gap report does not match ordered_items")
    binding = trajectory.blocks.get("selection_binding")
    if not isinstance(binding, dict) or binding.get("intent_sha256") != trajectory.source.get("selection_intent_sha256"):
        raise ValueError("Selection binding digest is missing or inconsistent")
    if faces is None:
        return
    for segment in segments:
        for point in segment.points:
            if point["triangle_index"] >= len(faces):
                raise ValueError("Trajectory point refers to a face outside the region PLY")
    if region is None:
        return
    _check_selection_binding(trajectory, blocks, segments, vertices, faces, region, binding)


def check_projection_inside_selection(segments: list[ScanSegment], blocks: list[dict[str, Any]], labels: np.ndarray,
                                      intrinsic: np.ndarray, image_size: np.ndarray) -> None:
    """Every scan/connector point and every edge between consecutive points must project onto the
    pixels of its own selection component: never onto unselected pixels or another block."""
    from femto.planning.surface_row_turns import edge_pixels_allowed
    from femto.planning.mesh_slice_planner import _project
    for segment in segments:
        allowed = labels == blocks[segment.block_index]["component_id"] + 1
        previous = None
        for point in segment.points:
            pixel = _project(np.asarray(point["xyz_m"], dtype=np.float64), intrinsic, image_size)
            if pixel is None:
                raise ValueError("Trajectory point projects outside the reference image")
            u, v = int(np.floor(pixel[0] + .5)), int(np.floor(pixel[1] + .5))
            if not (0 <= u < allowed.shape[1] and 0 <= v < allowed.shape[0] and allowed[v, u]):
                raise ValueError("Trajectory point projects onto an explicitly unselected pixel or another block")
            if previous is not None and not edge_pixels_allowed(previous, point["xyz_m"], intrinsic, image_size, allowed):
                raise ValueError("Trajectory edge crosses explicitly unselected pixels or another block")
            previous = point["xyz_m"]


def _check_selection_binding(trajectory: SurfaceTrajectory, blocks: list[dict[str, Any]], segments: list[ScanSegment],
                             vertices: np.ndarray, faces: np.ndarray, region: dict[str, Any], binding: dict[str, Any]) -> None:
    from femto.planning.mesh_slice_planner import Unplannable, _forbidden_polygons
    try:
        intent, support, labels, owners = selection_face_assignment(region, vertices, faces)
    except Unplannable as error:
        raise ValueError("Selection-block trajectory needs a region export with selection intent: " + str(error)) from error
    if (binding.get("intent_sha256") != intent["selected_mask_sha256"]
            or binding.get("support_sha256") != (None if support is None else support["class_sha256"])
            or binding.get("component_count") != len(intent["components"])
            or binding.get("hole_count") != len(intent["explicit_unselected_holes"])):
        raise ValueError("Trajectory selection binding does not match the region export")
    components = face_components(faces)
    view = region["view_reference"]
    intrinsic = np.asarray(view["undistorted_intrinsic"], dtype=np.float64)
    image_size = np.asarray(view["image_size"], dtype=np.int64)
    polygons = _forbidden_polygons(region)
    from femto.planning.surface_polygon_index import PreparedPolygonIndex
    polygon_index = PreparedPolygonIndex(polygons)
    by_component = {item["component_id"]: item for item in intent["components"]}
    if sorted(by_component) != sorted(block["component_id"] for block in blocks):
        raise ValueError("Block set does not match the selection components")
    for block in blocks:
        ids = np.flatnonzero(owners == block["component_id"])
        component = by_component[block["component_id"]]
        if (len(ids) != block["face_count"] or face_list_sha256(ids) != block["face_indices_sha256"]
                or (len(ids) and (int(ids.min()) != block["face_index_min"] or int(ids.max()) != block["face_index_max"]))
                or not _close(block["area_m2"], _area(vertices, faces, ids))
                or block["pixel_bounds"] != [float(v) for v in component["bbox"]]
                or block["selection"].get("selected_pixels") != component["pixel_count"]
                or block["selection"].get("hole_count") != len(component["hole_ids"])
                or block["selection"].get("support") != component_support_counts(support, labels, block["component_id"])):
            raise ValueError("Block face mapping or selection counts do not match the bound region export")
        fragments = block["fragments"]
        expected = sorted(set(int(v) for v in components[ids]))
        if sorted(fragment["mesh_component_id"] for fragment in fragments) != expected:
            raise ValueError("Block mesh fragments do not match mesh connectivity")
        for fragment in fragments:
            fragment_ids = ids[components[ids] == fragment["mesh_component_id"]]
            if (len(fragment_ids) != fragment["face_count"] or face_list_sha256(fragment_ids) != fragment["face_indices_sha256"]
                    or int(fragment_ids.min()) != fragment["face_index_min"] or int(fragment_ids.max()) != fragment["face_index_max"]
                    or not _close(fragment["area_m2"], _area(vertices, faces, fragment_ids))):
                raise ValueError("Fragment face mapping does not match the bound mesh")
    for segment in segments:
        expected_component = blocks[segment.block_index]["component_id"]
        previous = None
        for point in segment.points:
            if int(owners[point["triangle_index"]]) != expected_component:
                raise ValueError("Scan point lies on a face outside its selection block")
            if point.get("role") == "connector" and previous is not None:
                state = _edge_state(np.asarray(previous), np.asarray(point["xyz_m"]), intrinsic, image_size, polygons,
                                    polygon_index=polygon_index)
                if state == 2:
                    raise ValueError("Connector edge crosses a painted forbidden / excluded region")
                if state == 1:
                    raise ValueError("Connector edge leaves the reference image")
            previous = point["xyz_m"]
    check_projection_inside_selection(segments, blocks, labels, intrinsic, image_size)
