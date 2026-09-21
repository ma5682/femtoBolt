"""Unified, research-only surface trajectory records in W0/RDF metres."""
from __future__ import annotations

from dataclasses import dataclass
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
                if point.get("route_index") != expected_route or point.get("row_index") != item.row_index:
                    raise ValueError("Invalid global route index or row binding")
                triangle_index = point.get("triangle_index")
                if type(triangle_index) is not int or triangle_index < 0:
                    raise ValueError("Invalid local triangle index")
                pixel = point.get("pixel")
                if pixel is not None:
                    uv = np.asarray(pixel, dtype=np.float64)
                    if uv.shape != (2,) or not np.isfinite(uv).all():
                        raise ValueError("Invalid display pixel")
                if previous is not None and np.linalg.norm(np.asarray(xyz) - previous) > sample_step_m + 1e-9:
                    raise ValueError("Adjacent scan samples exceed sample_step_mm")
                if faces is not None:
                    if triangle_index >= len(faces) or not _point_on_triangle(xyz, vertices, faces[triangle_index]):
                        raise ValueError("Trajectory point is not supported by its local triangle")
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
    segments = [item.points for item in model.ordered_items if isinstance(item, ScanSegment)]
    gaps = [item for item in model.ordered_items if isinstance(item, Gap)]
    return {
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
