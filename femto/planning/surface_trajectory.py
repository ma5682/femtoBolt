"""Unified, research-only surface trajectory records in W0/RDF metres."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import hashlib
import json
import math

import numpy as np


SCHEMA = "femto-surface-trajectory-v1"
AXES = {"x": "right", "y": "down", "z": "forward"}
GAP_REASONS = {
    "forbidden_region", "disconnected_surface", "surface_gap",
    "unsupported_geometry", "row_turn",
}


def canonical_json_bytes(value: Any) -> bytes:
    """Stable UTF-8 JSON used for result and file integrity records."""
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def ordered_items_sha256(items: Iterable[Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(list(items))).hexdigest()


def _finite_xyz(value: Any, label: str = "xyz_m") -> list[float]:
    xyz = np.asarray(value, dtype=np.float64)
    if xyz.shape != (3,) or not np.isfinite(xyz).all():
        raise ValueError(f"Invalid {label}")
    return [float(v) for v in xyz]


@dataclass(frozen=True)
class ScanSegment:
    segment_index: int
    row_index: int
    points: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "scan_segment", "segment_index": self.segment_index,
                "row_index": self.row_index, "points": self.points}

    @classmethod
    def from_dict(cls, value: Any) -> "ScanSegment":
        if not isinstance(value, dict) or value.get("kind") != "scan_segment":
            raise ValueError("Invalid scan segment")
        points = value.get("points")
        if not isinstance(points, list) or len(points) < 2:
            raise ValueError("A scan segment needs at least two supported points")
        return cls(value.get("segment_index"), value.get("row_index"), points)


@dataclass(frozen=True)
class Gap:
    gap_index: int
    reason: str
    detail: dict[str, Any]
    previous_point: dict[str, Any]
    next_point: dict[str, Any]
    scan_direction_w0: list[float]
    row_index_from: int
    row_index_to: int
    chord_length_m: float

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "gap", "gap_index": self.gap_index, "reason": self.reason,
                "detail": self.detail, "previous_point": self.previous_point,
                "next_point": self.next_point, "scan_direction_w0": self.scan_direction_w0,
                "row_index_from": self.row_index_from, "row_index_to": self.row_index_to,
                "chord_length_m": self.chord_length_m}

    @classmethod
    def from_dict(cls, value: Any) -> "Gap":
        if not isinstance(value, dict) or value.get("kind") != "gap":
            raise ValueError("Invalid trajectory gap")
        return cls(value.get("gap_index"), value.get("reason"), value.get("detail"),
                   value.get("previous_point"), value.get("next_point"),
                   value.get("scan_direction_w0"), value.get("row_index_from"),
                   value.get("row_index_to"), value.get("chord_length_m"))


@dataclass(frozen=True)
class SurfaceTrajectory:
    origin_reference: dict[str, Any]
    planner: dict[str, Any]
    source: dict[str, Any]
    ordered_items: list[ScanSegment | Gap]
    statistics: dict[str, Any]
    gap_report: dict[str, Any]
    spacing_report: dict[str, Any]
    row_report: list[dict[str, Any]]
    data_layers: Any = None
    route_diagram_status: Any = None

    def to_dict(self) -> dict[str, Any]:
        items = [item.to_dict() for item in self.ordered_items]
        result = {
            "schema": SCHEMA, "coordinate_frame": "world_rdf", "unit": "metre",
            "axes": AXES, "origin_reference": self.origin_reference,
            "planner": self.planner, "source": self.source, "ordered_items": items,
            "statistics": self.statistics, "gap_report": self.gap_report,
            "spacing_report": self.spacing_report, "row_report": self.row_report,
            "orientation_fields": None, "normals_are_not_tool_pose": True,
            "execution_ready": False, "physical_accuracy_verified": False,
            "pixel_semantics": "display_and_exclusion_check_only",
            "result_sha256": ordered_items_sha256(items),
        }
        if self.data_layers is not None:
            result["data_layers"] = self.data_layers
        if self.route_diagram_status is not None:
            result["route_diagram_status"] = self.route_diagram_status
        return result

    @classmethod
    def from_dict(cls, value: Any) -> "SurfaceTrajectory":
        if not isinstance(value, dict):
            raise ValueError("Invalid surface trajectory document")
        if (value.get("schema") != SCHEMA or value.get("coordinate_frame") != "world_rdf" or
                value.get("unit") != "metre" or value.get("axes") != AXES or
                value.get("orientation_fields", object()) is not None or
                value.get("normals_are_not_tool_pose") is not True or
                value.get("execution_ready") is not False or
                value.get("physical_accuracy_verified") is not False or
                value.get("pixel_semantics") != "display_and_exclusion_check_only"):
            raise ValueError("Invalid surface trajectory coordinate/safety contract")
        items: list[ScanSegment | Gap] = []
        for item in value.get("ordered_items", []):
            items.append(ScanSegment.from_dict(item) if item.get("kind") == "scan_segment"
                         else Gap.from_dict(item))
        result = cls(value.get("origin_reference"), value.get("planner"), value.get("source"),
                     items, value.get("statistics"), value.get("gap_report"),
                     value.get("spacing_report"), value.get("row_report"),
                     value.get("data_layers"), value.get("route_diagram_status"))
        validate_trajectory(result, expected_result_sha256=value.get("result_sha256"))
        return result


def _forbidden_orientation_keys(value: Any) -> None:
    forbidden = {"pitch", "yaw", "roll", "rx", "ry", "rz", "quaternion", "normal"}
    if isinstance(value, dict):
        if forbidden.intersection(k.casefold() for k in value):
            raise ValueError("Surface trajectory contains forbidden orientation fields")
        for child in value.values():
            _forbidden_orientation_keys(child)
    elif isinstance(value, list):
        for child in value:
            _forbidden_orientation_keys(child)


def _point_on_triangle(point: Any, vertices: np.ndarray, triangle: np.ndarray,
                       tolerance_m: float = 1e-6) -> bool:
    xyz = np.asarray(point, dtype=np.float64)
    a, b, c = vertices[triangle]
    ab, ac, ap = b - a, c - a, xyz - a
    normal = np.cross(ab, ac)
    length = float(np.linalg.norm(normal))
    if not math.isfinite(length) or length <= 0:
        return False
    if abs(float(np.dot(ap, normal))) / length > tolerance_m:
        return False
    matrix = np.column_stack((ab, ac))
    uv, *_ = np.linalg.lstsq(matrix, ap, rcond=None)
    scale = max(min(float(np.linalg.norm(ab)), float(np.linalg.norm(ac))), tolerance_m)
    barycentric_tolerance = tolerance_m / scale
    closest = a + ab * uv[0] + ac * uv[1]
    return (float(np.linalg.norm(closest - xyz)) <= tolerance_m
            and float(min(uv)) >= -barycentric_tolerance
            and float(sum(uv)) <= 1 + barycentric_tolerance)


def validate_trajectory(trajectory: SurfaceTrajectory, *, mesh: Any = None,
                        expected_result_sha256: Any = None) -> None:
    if not isinstance(trajectory.origin_reference, dict) or trajectory.origin_reference.get("kind") != "initial_color_camera":
        raise ValueError("Surface trajectory requires an initial_color_camera origin")
    if not isinstance(trajectory.planner, dict) or not isinstance(trajectory.source, dict):
        raise ValueError("Invalid trajectory planner/source binding")
    if not trajectory.ordered_items or not isinstance(trajectory.ordered_items[0], ScanSegment) or not isinstance(trajectory.ordered_items[-1], ScanSegment):
        raise ValueError("Trajectory must start and end with a scan segment")
    vertices = faces = None
    if mesh is not None:
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
    expected_route = 0
    segments: list[ScanSegment] = []
    gaps: list[Gap] = []
    sample_step_m = float(trajectory.planner.get("parameters", {}).get("sample_step_mm", 1.0)) / 1000
    connected_rows = trajectory.planner.get('algorithm_version') == 'mesh-slice-serpentine-v2'
    if connected_rows and trajectory.planner.get('parameters', {}).get('connect_rows') is not True:
        raise ValueError('Connected row trajectory requires connect_rows=True')
    for index, item in enumerate(trajectory.ordered_items):
        if bool(index % 2) != isinstance(item, Gap):
            raise ValueError("ordered_items must strictly alternate segment and gap")
        if isinstance(item, ScanSegment):
            if type(item.segment_index) is not int or item.segment_index != len(segments):
                raise ValueError("Invalid segment index")
            if type(item.row_index) is not int or item.row_index < 0:
                raise ValueError("Invalid row index")
            previous = None
            for point in item.points:
                if not isinstance(point, dict) or point.get("geometry_support") != "measured_mesh":
                    raise ValueError("Every trajectory point needs measured_mesh support")
                xyz = _finite_xyz(point.get("xyz_m"))
                point_row = point.get('row_index')
                row_valid = (type(point_row) is int and point_row >= item.row_index) if connected_rows else point_row == item.row_index
                if point.get("route_index") != expected_route or not row_valid:
                    raise ValueError("Invalid global route index or row binding")
                triangle_index = point.get("triangle_index")
                if type(triangle_index) is not int or triangle_index < 0:
                    raise ValueError("Invalid local triangle index")
                pixel = point.get("pixel")
                if pixel is not None:
                    uv = np.asarray(pixel, dtype=np.float64)
                    if uv.shape != (2,) or not np.isfinite(uv).all():
                        raise ValueError("Invalid display pixel")
                role = point.get("role")
                if role is not None and role not in ("scan", "connector"):
                    raise ValueError("Invalid trajectory point role")
                if previous is not None and np.linalg.norm(np.asarray(xyz) - previous) > sample_step_m + 1e-9:
                    raise ValueError("Adjacent scan samples exceed sample_step_mm")
                if faces is not None:
                    if triangle_index >= len(faces) or not _point_on_triangle(xyz, vertices, faces[triangle_index]):
                        raise ValueError("Trajectory point is not supported by its local triangle")
                    if connected_rows and point.get('role') == 'connector' and previous is not None:
                        if not _point_on_triangle(previous, vertices, faces[triangle_index]):
                            raise ValueError('Connector edge is not supported by a single triangle')
                previous = np.asarray(xyz)
                expected_route += 1
            segments.append(item)
        else:
            if type(item.gap_index) is not int or item.gap_index != len(gaps) or item.reason not in GAP_REASONS:
                raise ValueError("Invalid gap index/reason")
            if item.detail != {"inferred_cause": item.detail.get("inferred_cause"),
                               "method": "sweep_plane_outer_loop_containment",
                               "confidence": "heuristic", "decision_use": "none"}:
                raise ValueError("Invalid informational gap detail")
            _finite_xyz(item.previous_point.get("xyz_m"), "gap previous xyz_m")
            _finite_xyz(item.next_point.get("xyz_m"), "gap next xyz_m")
            _finite_xyz(item.scan_direction_w0, "scan_direction_w0")
            if not math.isfinite(item.chord_length_m) or item.chord_length_m < 0:
                raise ValueError("Invalid gap chord length")
            gaps.append(item)
    if len(gaps) != len(segments) - 1:
        raise ValueError("Every adjacent scan segment needs exactly one gap")
    for index, gap in enumerate(gaps):
        before, after = segments[index].points[-1], segments[index + 1].points[0]
        if gap.previous_point != {"route_index": before["route_index"], "xyz_m": before["xyz_m"]}:
            raise ValueError("Gap previous endpoint mismatch")
        if gap.next_point != {"route_index": after["route_index"], "xyz_m": after["xyz_m"]}:
            raise ValueError("Gap next endpoint mismatch")
    payload = trajectory.to_dict()
    _forbidden_orientation_keys(payload)
    if expected_result_sha256 is not None and payload["result_sha256"] != expected_result_sha256:
        raise ValueError("Surface trajectory result hash mismatch")


def to_player_plan(trajectory: SurfaceTrajectory | dict[str, Any]) -> dict[str, Any]:
    model = SurfaceTrajectory.from_dict(trajectory) if isinstance(trajectory, dict) else trajectory
    validate_trajectory(model)
    segments = [[dict(point, motion_kind=point.get("role", "scan")) for point in item.points]
                for item in model.ordered_items if isinstance(item, ScanSegment)]
    gaps = [item for item in model.ordered_items if isinstance(item, Gap)]
    result = {
        "schema": "femto-surface-trajectory-player-view-v1",
        "segments": segments,
        "chain_breaks": [{
            "from_chain": index, "to_chain": index + 1,
            "from_pixel": segments[index][-1].get("pixel"),
            "to_pixel": segments[index + 1][0].get("pixel"),
            "from_xyz_m": gap.previous_point["xyz_m"],
            "to_xyz_m": gap.next_point["xyz_m"],
            "reason": gap.reason,
            "explanation": f"{gap.reason}：段间没有已规划运动；继续只定位下一段起点",
            "transition_planned": False, "transition_time_s": None,
        } for index, gap in enumerate(gaps)],
        "motion_demo": {"speed_mm_s": 20.0, "cross_chain_time_s": None if gaps else 0.0,
                        "gap_policy": "stop_require_explicit_continue_no_interpolated_transition",
                        "execution_ready": False},
        "route_order": {"mode": "surface_trajectory_ordered_items",
                        "continuous_chains": len(segments), "disconnected_breaks": len(gaps),
                        "connector_edges": sum(point.get("role") == "connector"
                                               for segment in segments for point in segment[1:])},
        "execution_ready": False, "physical_accuracy_verified": False,
    }
    if model.planner.get("parameters", {}).get("row_spacing_mm") is not None:
        result.update(spacing_mm=model.planner["parameters"]["row_spacing_mm"],
                      spacing_semantics="parallel_plane_spacing_not_uniform_surface_spacing",
                      surface_mode={"schema": "femto-surface-trajectory-player-view-v1",
                                    "algorithm": model.planner.get("algorithm_version")},
                      spacing_report=model.spacing_report)
    return result


def _projection_region_mesh(region: dict[str, Any], store: Any, region_manifest: Any):
    """Bind the adapter to an explicit export, or rebuild from the current source mesh."""
    global_ids = sorted(set(int(value) for value in region["triangle_indices"]))
    if not global_ids:
        raise ValueError("Projection plan region has no triangles")
    manifest_path = None
    if region_manifest is not None:
        if isinstance(region_manifest, dict):
            if not region_manifest.get("manifest_path"):
                raise ValueError("A manifest dictionary needs an explicit manifest_path")
            manifest_path = Path(region_manifest["manifest_path"]).resolve()
        else:
            manifest_path = Path(region_manifest).resolve()
        from femto.planning.mesh_slice_planner import _load_input, _sha256
        checked_path, manifest, mesh = _load_input(manifest_path)
        if manifest.get("region", {}).get("annotation_id") != region.get("annotation_id"):
            raise ValueError("Exported region identity does not match the projection plan")
        from femto.planning.planning_state import region_signature
        if region_signature(manifest["region"]) != region_signature(region):
            raise ValueError("Exported region content does not match the projection plan")
        mesh_sha256 = manifest["file_sha256"][manifest["files"]["mesh_file"]]
        source = {"region_export_dir": str(checked_path.parent),
                  "region_manifest": str(checked_path),
                  "region_manifest_sha256": _sha256(checked_path),
                  "mesh_file": manifest["files"]["mesh_file"],
                  "mesh_file_sha256": mesh_sha256,
                  "geometry_sha256": manifest.get("geometry_sha256"),
                  "bound_region_mesh": {"source": "exported_ply", "sha256": mesh_sha256}}
        origin = manifest["origin_reference"]
        data_layers = manifest.get("data_layers")
    else:
        from femto.annotation.scene_annotations import mesh_digest, region_mesh
        selected = region_mesh(store.mesh, global_ids)
        import trimesh
        mesh = trimesh.Trimesh(vertices=np.asarray(selected.vertices),
                               faces=np.asarray(selected.triangles), process=False)
        mesh_sha256 = mesh_digest(selected)
        source = {"mesh_file": None, "mesh_file_sha256": mesh_sha256,
                  "geometry_sha256": getattr(store, "digest", None),
                  "bound_region_mesh": {"source": "rebuilt_from_source_mesh", "sha256": mesh_sha256}}
        origin = store.origin_reference
        try:
            from femto.annotation.measurement_audit import layer_contract
            data_layers = layer_contract(store)
        except (AttributeError, KeyError, TypeError, ValueError):
            data_layers = getattr(store, "data_layers", None)
    if len(mesh.faces) != len(global_ids):
        raise ValueError("Bound region mesh face order/count mismatch")
    return mesh, {global_id: local for local, global_id in enumerate(global_ids)}, source, origin, data_layers


def _relocate_faces(points: list[np.ndarray], mesh: Any) -> list[int | None]:
    if not points:
        return []
    import open3d as o3d
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    legacy = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),
                                      o3d.utility.Vector3iVector(faces.astype(np.int32)))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(legacy))
    query = scene.compute_closest_points(o3d.core.Tensor(np.asarray(points, np.float32)))
    closest = query["points"].numpy().astype(float)
    seeds = query["primitive_ids"].numpy().astype(np.int64)
    result: list[int | None] = []
    face_min = vertices[faces].min(axis=1) - 1e-6
    face_max = vertices[faces].max(axis=1) + 1e-6
    for point, nearest, seed in zip(points, closest, seeds):
        if seed < 0 or float(np.linalg.norm(point - nearest)) > 1e-6:
            result.append(None)
            continue
        candidates = np.flatnonzero(((face_min <= point) & (point <= face_max)).all(axis=1))
        valid = [int(index) for index in candidates
                 if _point_on_triangle(point, vertices, faces[index], 1e-6)]
        result.append(min(valid) if valid else None)
    return result


def from_projection_plan(plan: dict[str, Any], region: dict[str, Any], store: Any,
                         region_manifest: Any = None) -> SurfaceTrajectory:
    """Read-only adapter from current fixed-RGB plan dictionaries."""
    if not isinstance(plan, dict) or not isinstance(region, dict):
        raise ValueError("Invalid projection plan/region")
    mesh, mapping, mesh_source, origin, data_layers = _projection_region_mesh(
        region, store, region_manifest)
    if not isinstance(origin, dict) or origin.get("kind") != "initial_color_camera":
        raise ValueError("Projection plan requires an initial_color_camera origin")
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    converted: list[list[dict[str, Any]]] = []
    unresolved: list[tuple[int, int, np.ndarray]] = []
    face_sources: dict[tuple[int, int], str] = {}
    for chain_index, chain in enumerate(plan.get("segments", [])):
        if not isinstance(chain, list) or len(chain) < 2:
            raise ValueError("Projection plan contains an unplayable chain")
        output = []
        for point_index, point in enumerate(chain):
            xyz = np.asarray(point.get("xyz_m"), dtype=np.float64)
            if xyz.shape != (3,) or not np.isfinite(xyz).all():
                raise ValueError("Projection plan contains invalid XYZ")
            pixel = np.asarray(point.get("pixel"), dtype=np.float64)
            if pixel.shape != (2,) or not np.isfinite(pixel).all():
                raise ValueError("Projection plan contains invalid display pixel")
            global_index = point.get("triangle_index")
            local = mapping.get(global_index) if type(global_index) is int else None
            if local is not None and _point_on_triangle(xyz, vertices, faces[local], 1e-6):
                face_sources[(chain_index, point_index)] = "mapped_from_source_mesh_index"
            else:
                unresolved.append((chain_index, point_index, xyz))
                local = None
            output.append({"xyz_m": xyz.tolist(), "pixel": pixel.tolist(),
                           "triangle_index": local, "geometry_support": "measured_mesh",
                           "role": point.get("motion_kind", "scan")})
        converted.append(output)
    relocated = _relocate_faces([record[2] for record in unresolved], mesh)
    failures = []
    for (chain_index, point_index, xyz), local in zip(unresolved, relocated):
        if local is None:
            failures.append({"chain_index": chain_index, "point_index": point_index,
                             "xyz_m": xyz.tolist()})
        else:
            converted[chain_index][point_index]["triangle_index"] = local
            face_sources[(chain_index, point_index)] = "relocated_geometrically"
    if failures:
        raise ValueError("Projection plan points cannot be bound to the region PLY: " + json.dumps(failures[:20]))
    route_index = 0
    segments = []
    max_edge = 0.0
    for chain_index, chain in enumerate(converted):
        row_index = chain_index
        for point_index, point in enumerate(chain):
            point.update(row_index=row_index, route_index=route_index,
                         face_index_source=face_sources[(chain_index, point_index)])
            route_index += 1
        edges = np.linalg.norm(np.diff(np.asarray([point["xyz_m"] for point in chain]), axis=0), axis=1)
        max_edge = max(max_edge, float(edges.max(initial=0.0)))
        segments.append(ScanSegment(chain_index, row_index, chain))
    if not segments:
        raise ValueError("Projection plan has no scan segment")
    breaks = {item.get("from_chain"): item for item in plan.get("chain_breaks", [])}
    items: list[ScanSegment | Gap] = []
    gap_counts: dict[str, int] = {}
    for index, segment in enumerate(segments):
        items.append(segment)
        if index == len(segments) - 1:
            continue
        record = breaks.get(index, {})
        explanation = str(record.get("reason", "")) + " " + str(record.get("explanation", ""))
        left_y = float(np.mean([point["pixel"][1] for point in segment.points if point.get("pixel") is not None]))
        right_y = float(np.mean([point["pixel"][1] for point in segments[index + 1].points if point.get("pixel") is not None]))
        reason = "row_turn" if "row" in explanation.casefold() or "turn" in explanation.casefold() or abs(left_y - right_y) > 1 else "unsupported_geometry"
        gap_counts[reason] = gap_counts.get(reason, 0) + 1
        before, after = segment.points[-1], segments[index + 1].points[0]
        delta = np.asarray(segment.points[-1]["xyz_m"]) - np.asarray(segment.points[-2]["xyz_m"])
        direction = (delta / max(float(np.linalg.norm(delta)), 1e-15)).tolist()
        items.append(Gap(index, reason, {
            "inferred_cause": "not_applicable", "method": "sweep_plane_outer_loop_containment",
            "confidence": "heuristic", "decision_use": "none",
        }, {"route_index": before["route_index"], "xyz_m": before["xyz_m"]},
           {"route_index": after["route_index"], "xyz_m": after["xyz_m"]}, direction,
           segment.row_index, segments[index + 1].row_index,
           float(np.linalg.norm(np.asarray(after["xyz_m"]) - before["xyz_m"]))))
    lengths = [float(np.linalg.norm(np.diff(np.asarray([point["xyz_m"] for point in segment.points]), axis=0), axis=1).sum())
               for segment in segments]
    z = np.asarray([point["xyz_m"][2] for segment in segments for point in segment.points])
    from femto.planning.planning_state import region_signature
    source = dict(mesh_source, region_id=region.get("annotation_id"), region_signature=region_signature(region),
                  projection_plan_result_sha256=plan.get("plan_binding", {}).get("result_sha256"))
    algorithm = plan.get("plan_binding", {}).get("algorithm_version") or plan.get("schema")
    planner = {"kind": "projection_fixed_rgb", "algorithm_version": "projection-adapter-v1",
               "parameters": {"source_algorithm_version": algorithm,
                              "sample_step_mm": max(1e-9, max_edge * 1000)}}
    row_report = [{"row_index": segment.row_index, "segment_count": 1,
                   "point_count": len(segment.points), "scan_length_m": lengths[segment.segment_index],
                   "depth_min_m": min(point["xyz_m"][2] for point in segment.points),
                   "depth_max_m": max(point["xyz_m"][2] for point in segment.points),
                   "depth_span_m": max(point["xyz_m"][2] for point in segment.points) - min(point["xyz_m"][2] for point in segment.points),
                   "gap_count": int(segment.segment_index < len(segments) - 1)} for segment in segments]
    trajectory = SurfaceTrajectory(origin, planner, source, items,
        {"segment_count": len(segments), "gap_count": len(segments) - 1,
         "gap_counts": dict(sorted(gap_counts.items())), "point_count": route_index,
         "scan_length_m": float(sum(lengths)), "row_count": len(segments),
         "depth_min_m": float(z.min()), "depth_max_m": float(z.max()),
         "depth_span_m": float(z.max() - z.min())},
        {"counts": dict(sorted(gap_counts.items())), "dropped_fragments": [],
         "dropped_fragment_count": 0, "relocated_point_count": len(unresolved)},
        plan.get("spacing_report", {}), row_report, data_layers)
    validate_trajectory(trajectory, mesh=mesh)
    return trajectory
