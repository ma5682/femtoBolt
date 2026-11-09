"""Unified, research-only surface trajectory records in W0/RDF metres."""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable
import hashlib
import json
import math
from threading import get_ident

import numpy as np
from femto.planning.fixed_width_layout import ADAPTIVE_FLOOR_PATCH_VERSION,COORDINATED_FLOOR_PATCH_VERSION


SCHEMA = "femto-surface-trajectory-v1"
AXES = {"x": "right", "y": "down", "z": "forward"}
GAP_REASONS = {
    "forbidden_region", "disconnected_surface", "surface_gap",
    "unsupported_geometry", "row_turn", "selection_break",
}
CONNECTED_ALGORITHM = "mesh-slice-serpentine-v2"
BLOCK_ALGORITHM = "mesh-slice-serpentine-v3.1"
BLOCK_ALGORITHMS = ("mesh-slice-serpentine-v3", BLOCK_ALGORITHM)
# v4: work blocks are the physician's selection components; every planned block is one chain.
SELECTION_ALGORITHM = "mesh-slice-serpentine-v4"
OPTIMIZED_SELECTION_ALGORITHM = "mesh-slice-serpentine-v4.1"
FACE_PORTAL_SELECTION_ALGORITHM = "mesh-slice-serpentine-v4.2"
BIDIRECTIONAL_SELECTION_ALGORITHM = "mesh-slice-serpentine-v4.3"
ORDERED_SELECTION_ALGORITHM = "mesh-slice-serpentine-v4.4"
SELECTION_ALGORITHMS = (SELECTION_ALGORITHM, OPTIMIZED_SELECTION_ALGORITHM,
                        FACE_PORTAL_SELECTION_ALGORITHM, BIDIRECTIONAL_SELECTION_ALGORITHM,
                        ORDERED_SELECTION_ALGORITHM)
ROW_TURN_FAILURE_REASONS = {
    "non_adjacent_slice", "endpoint_not_on_boundary", "forbidden_blocked",
    "ambiguous_boundary", "no_boundary_path_in_band",
    "connection_deferred",
}
LINK_FAILURE_REASONS = {
    "disconnected_measured_surface", "forbidden_blocked", "outside_reference_image",
    "explicit_unselected_blocked", "no_surface_path", "budget_exceeded",
    "connection_deferred",
}
BLOCKS_SCHEMA = "femto-surface-trajectory-blocks-v1"
SELECTION_BLOCKS_SCHEMA = "femto-surface-trajectory-blocks-v2"
POINT_BUDGET = 200_000
BLOCK_REASONS = {
    "insufficient_extent", "no_positive_area", "axis_perpendicular",
    "narrower_than_row_spacing", "no_supported_segment",
}
SELECTION_BLOCK_REASONS = BLOCK_REASONS | {"no_supported_geometry", "all_fragments_unplannable"}
BLOCK_STATISTIC_KEYS = ("segment_count", "gap_count", "point_count", "row_count",
                        "scan_length_m", "connector_length_m", "connected_row_turns")


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
    block_index: Any = None

    def to_dict(self) -> dict[str, Any]:
        result = {"kind": "scan_segment", "segment_index": self.segment_index,
                  "row_index": self.row_index, "points": self.points}
        if self.block_index is not None:
            result["block_index"] = self.block_index
        return result

    @classmethod
    def from_dict(cls, value: Any) -> "ScanSegment":
        if not isinstance(value, dict) or value.get("kind") != "scan_segment":
            raise ValueError("Invalid scan segment")
        points = value.get("points")
        if not isinstance(points, list) or len(points) < 2:
            raise ValueError("A scan segment needs at least two supported points")
        return cls(value.get("segment_index"), value.get("row_index"), points, value.get("block_index"))


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
    blocks: Any = None
    planning_context: Any = None

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
        if self.blocks is not None:
            result["blocks"] = self.blocks
        if self.planning_context is not None:
            result['planning_context'] = self.planning_context
        return result

    @classmethod
    def from_dict(cls, value: Any, *, mesh: Any = None, region: Any = None,
                  check=lambda: None) -> "SurfaceTrajectory":
        """Parse and validate once, including caller-supplied geometric support."""
        check()
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
            check()
            items.append(ScanSegment.from_dict(item) if item.get("kind") == "scan_segment"
                         else Gap.from_dict(item))
        result = cls(value.get("origin_reference"), value.get("planner"), value.get("source"),
                     items, value.get("statistics"), value.get("gap_report"),
                     value.get("spacing_report"), value.get("row_report"),
                     value.get("data_layers"), value.get("route_diagram_status"),
                     value.get("blocks"), value.get('planning_context'))
        validate_trajectory(result, mesh=mesh, region=region, check=check,
                            expected_result_sha256=value.get("result_sha256"))
        check()
        return result


_FORBIDDEN_ORIENTATION_KEYS = frozenset({"pitch", "yaw", "roll", "rx", "ry", "rz", "quaternion", "normal"})
_JSON_CONTAINERS = (dict, list, tuple)


def _forbidden_orientation_keys(value: Any) -> None:
    if isinstance(value, dict):
        if _FORBIDDEN_ORIENTATION_KEYS.intersection(k.casefold() for k in value):
            raise ValueError("Surface trajectory contains forbidden orientation fields")
        for child in value.values():
            if isinstance(child, _JSON_CONTAINERS):
                _forbidden_orientation_keys(child)
    elif isinstance(value, (list, tuple)):
        # JSON serializes tuples as lists; reject the same nested keys before
        # publication as on independent readback. Scalars have no field keys.
        if type(value) is list and len(value) >= 256:
            from femto.plain_copy import plain_integers
            if plain_integers(value):
                return
        for child in value:
            if isinstance(child, _JSON_CONTAINERS):
                _forbidden_orientation_keys(child)


def _point_on_triangle(point: Any, vertices: np.ndarray, triangle: np.ndarray,
                       tolerance_m: float = 1e-6, *, _prepared=None) -> bool:
    xyz = np.asarray(point, dtype=np.float64)
    if _prepared is None:
        a, b, c = vertices[triangle]
        ab, ac, ap = b - a, c - a, xyz - a
        normal = np.cross(ab, ac)
        length = float(np.linalg.norm(normal))
    else:
        a, ab, ac = _prepared.origin, _prepared.ab, _prepared.ac
        ap, normal = xyz - a, _prepared.normal
        length = float(_prepared.normal_length)
    if not math.isfinite(length) or length <= 0:
        return False
    if abs(float(np.dot(ap, normal))) / length > tolerance_m:
        return False
    if _prepared is not None and _prepared.solver.point_inside_one(xyz, ap, _prepared, tolerance_m):
        return True
    matrix = np.column_stack((ab, ac)) if _prepared is None else _prepared.matrix
    uv, *_ = np.linalg.lstsq(matrix, ap, rcond=None)
    edge_length = (min(float(np.linalg.norm(ab)), float(np.linalg.norm(ac)))
                   if _prepared is None else _prepared.minimum_edge_length)
    scale = max(edge_length, tolerance_m)
    barycentric_tolerance = tolerance_m / scale
    closest = a + ab * uv[0] + ac * uv[1]
    return (float(np.linalg.norm(closest - xyz)) <= tolerance_m
            and float(min(uv)) >= -barycentric_tolerance
            and float(sum(uv)) <= 1 + barycentric_tolerance)


_CERTIFICATE_POINT_PREDICATE = _point_on_triangle


def face_components(faces: np.ndarray) -> np.ndarray:
    """Deterministic face connectivity labels: faces sharing an edge share a label.

    Labels are 0-based and numbered by ascending smallest face index, so the
    same PLY always yields the same component ids. No vertex welding or
    tolerance merging: topology only."""
    faces = np.asarray(faces, dtype=np.int64)
    # The existing integer-edge contraction preserves minimum-face labels,
    # including duplicate/nonmanifold edges and vertex-only separation.
    from femto.geometry.triangle_components import triangle_components
    from types import SimpleNamespace
    prepared = triangle_components(SimpleNamespace(triangles=faces))
    if prepared is not None:
        return prepared[0]
    parent = np.arange(len(faces), dtype=np.int64)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            if a > b:
                a, b = b, a
            parent[b] = a

    incident: dict[tuple[int, int], int] = {}
    for face_index, face in enumerate(faces):
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge = tuple(sorted((int(a), int(b))))
            if edge in incident:
                union(face_index, incident[edge])
            else:
                incident[edge] = face_index
    roots = [find(i) for i in range(len(faces))]
    mapping = {root: index for index, root in enumerate(sorted(set(roots)))}
    return np.asarray([mapping[root] for root in roots], dtype=np.int64)


def face_list_sha256(face_indices: Any) -> str:
    return hashlib.sha256(canonical_json_bytes([int(v) for v in face_indices])).hexdigest()


def _validated_blocks(trajectory: SurfaceTrajectory) -> list[dict[str, Any]]:
    blocks = trajectory.blocks
    selection = trajectory.planner.get("algorithm_version") in SELECTION_ALGORITHMS
    expected_schema = SELECTION_BLOCKS_SCHEMA if selection else BLOCKS_SCHEMA
    if (not isinstance(blocks, dict) or blocks.get("schema") != expected_schema or
            not isinstance(blocks.get("items"), list) or not blocks["items"]):
        raise ValueError(f"Block trajectory requires a {expected_schema} record")
    for position, item in enumerate(blocks["items"]):
        if not isinstance(item, dict) or item.get("block_index") != position:
            raise ValueError("Block items must be numbered by visit order")
        for key in ("component_id", "face_count", "face_index_min", "face_index_max"):
            if type(item.get(key)) is not int or item[key] < 0:
                raise ValueError(f"Invalid block {key}")
        digest = item.get("face_indices_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("Invalid block face list digest")
        if item.get("status") == "planned" or (selection and item.get("status") == "incomplete"):
            for key in ("segment_index_first", "segment_count", "point_count", "row_index_first", "row_count"):
                if type(item.get(key)) is not int or item[key] < 0:
                    raise ValueError(f"Invalid planned block {key}")
            if item["segment_count"] < 1:
                raise ValueError("A planned block needs at least one scan segment")
            if selection and (item["status"] == "planned") != (item["segment_count"] == 1):
                raise ValueError("A planned selection block is exactly one continuous chain")
        elif item.get("status") == "unplannable":
            if item.get("reason") not in (SELECTION_BLOCK_REASONS if selection else BLOCK_REASONS):
                raise ValueError("Invalid unplannable block reason")
        else:
            raise ValueError("Invalid block status")
    return blocks["items"]


def _close(left: Any, right: Any) -> bool:
    return (type(left) is float and type(right) is float and math.isfinite(left)
            and math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9))


def _length_split(points: list[dict[str, Any]]) -> tuple[float, float]:
    """(scan, connector) polyline length of one chain; an edge is a connector edge when its end is."""
    xyz = np.asarray([point["xyz_m"] for point in points], dtype=np.float64)
    steps = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    connector = np.asarray([point.get("role") == "connector" for point in points[1:]])
    return float(steps[~connector].sum()), float(steps[connector].sum())


def _check_block_statistics(trajectory: SurfaceTrajectory, blocks: list[dict[str, Any]],
                            segments: list[ScanSegment], gaps: list[Gap],
                            vertices: Any, faces: Any) -> None:
    """Every per-block and global figure must be recomputable from ordered_items (and the mesh)."""
    strict = trajectory.planner.get("algorithm_version") == BLOCK_ALGORITHM
    planned = [item for item in blocks if item["status"] == "planned"]
    next_row = 0
    totals = {"scan": 0.0, "connector": 0.0, "turns": 0, "rows": 0}
    failures: Counter[str] = Counter()
    for item in planned:
        owned = [segment for segment in segments if segment.block_index == item["block_index"]]
        points = [point for segment in owned for point in segment.points]
        rows = sorted({point["row_index"] for point in points})
        if (item.get("row_index_first") != next_row or rows != list(range(next_row, next_row + item["row_count"]))):
            raise ValueError("Block row bookkeeping does not match its scan points")
        next_row += item["row_count"]
        lengths = [_length_split(segment.points) for segment in owned]
        scan, connector = float(sum(v[0] for v in lengths)), float(sum(v[1] for v in lengths))
        if not _close(item.get("scan_length_m"), scan) or not _close(item.get("connector_length_m"), connector):
            raise ValueError("Block scan/connector length does not match its scan points")
        first, last = owned[0].segment_index, owned[-1].segment_index
        inside = [gap for index, gap in enumerate(gaps) if first <= index < last]
        counts = dict(sorted(Counter(gap.reason for gap in inside).items()))
        turns = item["row_count"] - 1 - counts.get("row_turn", 0)
        if (item.get("gap_count") != len(inside) or item.get("gap_counts") != counts
                or type(item.get("connected_row_turns")) is not int or item["connected_row_turns"] != turns):
            raise ValueError("Block gap classification or connected row turns do not match ordered_items")
        reasons = Counter(gap.detail.get("row_turn_failure_reason") for gap in inside if gap.reason == "row_turn")
        if strict and (None in reasons or "row_turn_failure_counts" not in item):
            raise ValueError("Every block-mode row_turn gap needs a diagnostic reason")
        reasons.pop(None, None)
        if "row_turn_failure_counts" in item and item["row_turn_failure_counts"] != dict(sorted(reasons.items())):
            raise ValueError("Block row-turn failure counts do not match its gaps")
        failures.update(reasons)
        z = np.asarray([point["xyz_m"][2] for point in points], dtype=np.float64)
        if not (_close(item.get("depth_min_m"), float(z.min())) and _close(item.get("depth_max_m"), float(z.max()))
                and _close(item.get("depth_span_m"), float(z.max() - z.min()))):
            raise ValueError("Block depth statistics do not match its scan points")
        totals["scan"] += scan
        totals["connector"] += connector
        totals["turns"] += turns
        totals["rows"] += item["row_count"]
    _check_global_block_statistics(trajectory, blocks, segments, gaps, totals, failures, strict)


def _check_global_block_statistics(trajectory: SurfaceTrajectory, blocks: list[dict[str, Any]],
                                   segments: list[ScanSegment], gaps: list[Gap], totals: dict[str, Any],
                                   failures: Counter, strict: bool) -> None:
    stats = trajectory.statistics
    unplannable = [item for item in blocks if item["status"] != "planned"]
    gap_counts = dict(sorted(Counter(gap.reason for gap in gaps).items()))
    z = np.asarray([point["xyz_m"][2] for segment in segments for point in segment.points], dtype=np.float64)
    checks = {
        "row_count": stats.get("row_count") == totals["rows"],
        "gap_count": stats.get("gap_count") == len(gaps),
        "gap_counts": stats.get("gap_counts") == gap_counts,
        "connected_row_turns": stats.get("connected_row_turns") == totals["turns"],
        "scan_length_m": _close(stats.get("scan_length_m"), totals["scan"]),
        "connector_length_m": _close(stats.get("connector_length_m"), totals["connector"]),
        "total_length_m": _close(stats.get("total_length_m"), totals["scan"] + totals["connector"]),
        "unplannable_face_count": stats.get("unplannable_face_count") == sum(item["face_count"] for item in unplannable),
        "unplannable_area_m2": _close(stats.get("unplannable_area_m2"), float(sum(item["area_m2"] for item in unplannable))),
        "depth_min_m": _close(stats.get("depth_min_m"), float(z.min())),
        "depth_max_m": _close(stats.get("depth_max_m"), float(z.max())),
        "depth_span_m": _close(stats.get("depth_span_m"), float(z.max() - z.min())),
    }
    for key, ok in checks.items():
        if not ok:
            raise ValueError(f"Block statistics mismatch: {key}")
    report = trajectory.gap_report if isinstance(trajectory.gap_report, dict) else {}
    if report.get("counts") != gap_counts:
        raise ValueError("Block gap report counts do not match ordered_items")
    if strict and "row_turn_failure_counts" not in report:
        raise ValueError("Block-mode gap report needs row_turn_failure_counts")
    if "row_turn_failure_counts" in report and report["row_turn_failure_counts"] != dict(sorted(failures.items())):
        raise ValueError("Gap report row-turn failure counts do not match the gaps")


def _check_block_consistency(trajectory: SurfaceTrajectory, blocks: list[dict[str, Any]],
                             segments: list[ScanSegment], gaps: list[Gap],
                             vertices: Any, faces: Any) -> None:
    planned = [item for item in blocks if item["status"] == "planned"]
    for item in planned:
        owned = [segment for segment in segments if segment.block_index == item["block_index"]]
        if (len(owned) != item["segment_count"] or not owned or
                owned[0].segment_index != item["segment_index_first"] or
                [segment.segment_index for segment in owned] != list(range(owned[0].segment_index, owned[0].segment_index + len(owned))) or
                sum(len(segment.points) for segment in owned) != item["point_count"]):
            raise ValueError("Block segment/point bookkeeping does not match ordered_items")
    if any(segment.block_index not in {item["block_index"] for item in planned} for segment in segments):
        raise ValueError("Scan segment refers to a block that was not planned")
    for index, gap in enumerate(gaps):
        crossing = segments[index].block_index != segments[index + 1].block_index
        if crossing != (gap.reason == "disconnected_surface"):
            raise ValueError("Only inter-block gaps may be disconnected_surface in block mode")
    stats = trajectory.statistics if isinstance(trajectory.statistics, dict) else {}
    expected = {"block_count": len(blocks), "plannable_block_count": len(planned),
                "unplannable_block_count": len(blocks) - len(planned),
                "segment_count": len(segments), "point_count": sum(len(s.points) for s in segments),
                "inter_block_gap_count": sum(g.reason == "disconnected_surface" for g in gaps),
                "coverage_status": "complete" if len(planned) == len(blocks) else "partial"}
    for key, value in expected.items():
        if stats.get(key) != value:
            raise ValueError(f"Block statistics mismatch: {key}")
    for item in blocks:
        if type(item.get("area_m2")) is not float or not math.isfinite(item["area_m2"]) or item["area_m2"] < 0:
            raise ValueError("Invalid block area")
    _check_block_statistics(trajectory, blocks, segments, gaps, vertices, faces)
    if faces is None:
        return
    components = face_components(faces)
    if len(set(components.tolist())) != len(blocks):
        raise ValueError("Block count does not match mesh connectivity")
    for item in blocks:
        ids = np.flatnonzero(components == item["component_id"])
        if (len(ids) != item["face_count"] or int(ids.min()) != item["face_index_min"] or
                int(ids.max()) != item["face_index_max"] or face_list_sha256(ids) != item["face_indices_sha256"]):
            raise ValueError("Block face mapping does not match the bound mesh")
        triangles = vertices[faces[ids]]
        area = float(np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                             triangles[:, 2] - triangles[:, 0]), axis=1).sum() / 2)
        if not _close(item["area_m2"], area):
            raise ValueError("Block area does not match the bound mesh")
    for segment in segments:
        expected_component = blocks[segment.block_index]["component_id"]
        if any(int(components[point["triangle_index"]]) != expected_component for point in segment.points):
            raise ValueError("Scan point lies on a face outside its block")


def _validate_scan_points(
    item: ScanSegment, version: str, connected_rows: bool, selection_mode: bool,
    expected_route: int, sample_step_m: float, vertices: np.ndarray | None,
    faces: np.ndarray | None, *, _prepared_faces=None,
) -> int:
    # Group only syntactically ordinary candidates. All original structural
    # checks below still run in order, and uncertain geometry follows the old
    # scalar predicate. These local flags never cross validation requests.
    plain, steps, on_face, previous_on_face = _scan_point_certificates(
        item.points, vertices, faces, sample_step_m, _prepared_faces=_prepared_faces,
        connected_rows=connected_rows)
    previous = None
    for point_index, point in enumerate(item.points):
        if not isinstance(point, dict) or point.get("geometry_support") != "measured_mesh":
            raise ValueError("Every trajectory point needs measured_mesh support")
        xyz = list(point["xyz_m"]) if plain[point_index] else _finite_xyz(point.get("xyz_m"))
        point_row = point.get('row_index')
        row_valid = ((type(point_row) is int and point_row >= 0) if version == ORDERED_SELECTION_ALGORITHM
                     else (type(point_row) is int and point_row >= item.row_index) if connected_rows
                     else point_row == item.row_index)
        if point.get("route_index") != expected_route or not row_valid:
            raise ValueError("Invalid global route index or row binding")
        triangle_index = point.get("triangle_index")
        if type(triangle_index) is not int or triangle_index < 0:
            raise ValueError("Invalid local triangle index")
        pixel = point.get("pixel")
        if pixel is not None and not (type(pixel) is list and len(pixel) == 2
                                      and all(type(v) is float and math.isfinite(v) for v in pixel)):
            uv = np.asarray(pixel, dtype=np.float64)
            if uv.shape != (2,) or not np.isfinite(uv).all():
                raise ValueError("Invalid display pixel")
        role = point.get("role")
        if role is not None and role not in ("scan", "connector"):
            raise ValueError("Invalid trajectory point role")
        if "revisit" in point and (not selection_mode or role != "connector" or point["revisit"] is not True):
            raise ValueError("revisit is only a v4 flag on connector samples")
        if (previous is not None and steps[point_index] != 1
                and (steps[point_index] == 2 or np.linalg.norm(np.asarray(xyz) - previous) > sample_step_m + 1e-9)):
            raise ValueError("Adjacent scan samples exceed sample_step_mm")
        if faces is not None:
            if triangle_index >= len(faces):
                raise ValueError("Trajectory point is not supported by its local triangle")
            surely_on = on_face[point_index]
            surely_previous = previous_on_face[point_index]
            needs_previous = connected_rows and point.get('role') == 'connector' and previous is not None
            prepared = (None if _prepared_faces is None or (surely_on and (surely_previous or not needs_previous))
                        else _prepared_faces(triangle_index))
            if not surely_on and not _point_on_triangle(xyz, vertices, faces[triangle_index], _prepared=prepared):
                raise ValueError("Trajectory point is not supported by its local triangle")
            if needs_previous and not surely_previous:
                if not _point_on_triangle(previous, vertices, faces[triangle_index], _prepared=prepared):
                    raise ValueError('Connector edge is not supported by a single triangle')
        previous = np.asarray(xyz)
        expected_route += 1
    return expected_route


def _scan_point_certificates(points, vertices, faces, sample_step_m, *, _prepared_faces=None,
                             connected_rows=None):
    """Per-point shortcuts that can only confirm what the scalar checks accept.

    ``plain`` marks a dict point whose xyz is a finite three-float list, so the
    original conversion would return the same values. ``steps`` is 1 for a
    step clearly within the limit, 2 for one clearly beyond it and 0 when the
    scalar norm must decide. Triangle flags are surely-True certificates only.
    """
    count = len(points)
    plain = np.zeros(count, bool)
    steps = np.zeros(count, np.int8)
    on_face = np.zeros(count, bool)
    previous_on_face = np.zeros(count, bool)
    rows = []
    for index, point in enumerate(points):
        xyz = point.get('xyz_m') if type(point) is dict else None
        if (type(xyz) is list and len(xyz) == 3 and type(xyz[0]) is float and type(xyz[1]) is float
                and type(xyz[2]) is float and math.isfinite(xyz[0]) and math.isfinite(xyz[1])
                and math.isfinite(xyz[2])):
            plain[index] = True
            rows.append(xyz)
        else:
            rows.append((math.nan, math.nan, math.nan))
    if count < 2 or not plain.any() or np.geterr() != dict(divide='warn', over='warn', under='ignore', invalid='warn'):
        return plain, steps, on_face, previous_on_face
    xyz = np.asarray(rows, dtype=np.float64)
    both = plain[1:] & plain[:-1]
    delta = xyz[1:]-xyz[:-1]
    length = np.sqrt(np.einsum('ij,ij->i', np.where(both[:, None], delta, 0.), np.where(both[:, None], delta, 0.)))
    limit = sample_step_m+1e-9
    # Keep a wide relative guard: the scalar norm decides anything nearby.
    guard = 1e-12+1e-9*limit
    steps[1:][both & (length < limit-guard)] = 1
    steps[1:][both & (length > limit+guard)] = 2
    if (faces is not None and vertices is not None
            and _point_on_triangle is _CERTIFICATE_POINT_PREDICATE):
        from femto.planning.triangle_certificates import points_on_triangles
        indexes = np.asarray([point.get('triangle_index') if type(point) is dict and type(point.get('triangle_index')) is int
                              and 0 <= point.get('triangle_index') < len(faces) else -1 for point in points], dtype=np.int64)
        usable = plain & (indexes >= 0)
        if usable.any():
            chosen = np.flatnonzero(usable)
            later = chosen[chosen > 0]
            later = later[plain[later-1]]
            if connected_rows is False:
                later = later[:0]
            elif connected_rows is True:
                # Only connectors consume this additional proof below. Keep
                # unusual role types on the original path without invoking
                # custom equality during this optional preflight.
                later = later[[type(role := points[i].get('role')) is not str or role == 'connector'
                               for i in later]]
            same = indexes[later-1] == indexes[later]
            different = later[~same]
            # Current and preceding points use a single private XYZ/face
            # snapshot. Submit the identical (previous XYZ, same face) only
            # once; every handoff still needs the previous XYZ on its NEW face.
            # Each invocation of the validator performs these checks anew.
            if len(different) and len(chosen)+len(different) <= 16384:
                result = points_on_triangles(
                    xyz[np.r_[chosen, different-1]], vertices, faces, indexes[np.r_[chosen, different]],
                    _prepared_faces=_prepared_faces)
                on_face[chosen], previous_on_face[different] = result[:len(chosen)], result[len(chosen):]
            else:
                on_face[chosen] = points_on_triangles(xyz[chosen], vertices, faces, indexes[chosen],
                                                      _prepared_faces=_prepared_faces)
                if len(different):
                    previous_on_face[different] = points_on_triangles(
                        xyz[different-1], vertices, faces, indexes[different], _prepared_faces=_prepared_faces)
            previous_on_face[later[same]] = on_face[later[same]-1]
    return plain, steps, on_face, previous_on_face


def _validate_gap_record(item: Gap, gaps: list[Gap], selection_mode: bool, block_mode: bool) -> None:
    if type(item.gap_index) is not int or item.gap_index != len(gaps) or item.reason not in GAP_REASONS:
        raise ValueError("Invalid gap index/reason")
    if item.reason == "selection_break" and not selection_mode:
        raise ValueError("selection_break gaps only exist in selection-block trajectories")
    expected_detail = {"inferred_cause": item.detail.get("inferred_cause"),
                       "method": "sweep_plane_outer_loop_containment",
                       "confidence": "heuristic", "decision_use": "none"}
    failure = item.detail.get("row_turn_failure_reason") if isinstance(item.detail, dict) else None
    if failure is not None:
        if not block_mode or item.reason != "row_turn" or failure not in ROW_TURN_FAILURE_REASONS:
            raise ValueError("row_turn_failure_reason is only a diagnostic on block-mode row_turn gaps")
        expected_detail["row_turn_failure_reason"] = failure
    link_failure = item.detail.get("link_failure_reason") if isinstance(item.detail, dict) else None
    if link_failure is not None:
        if not selection_mode or item.reason == "selection_break" or link_failure not in LINK_FAILURE_REASONS:
            raise ValueError("link_failure_reason is only a diagnostic on v4 intra-block gaps")
        expected_detail["link_failure_reason"] = link_failure
    if item.detail != expected_detail:
        raise ValueError("Invalid informational gap detail")
    _finite_xyz(item.previous_point.get("xyz_m"), "gap previous xyz_m")
    _finite_xyz(item.next_point.get("xyz_m"), "gap next xyz_m")
    _finite_xyz(item.scan_direction_w0, "scan_direction_w0")
    if not math.isfinite(item.chord_length_m) or item.chord_length_m < 0:
        raise ValueError("Invalid gap chord length")


def validate_trajectory(trajectory: SurfaceTrajectory, *, mesh: Any = None,
                        expected_result_sha256: Any = None, region: Any = None, check=lambda: None) -> None:
    if trajectory.planning_context is not None:
        from femto.planning.planning_revision import validate_context
        from femto.planning.target_surface import validate_metadata
        context = trajectory.planning_context
        revision = validate_context(context, region=region, algorithm=trajectory.planner.get('algorithm_version'))
        validate_metadata(context['target'], revision)
        params = trajectory.planner['parameters']
        from femto.planning.fixed_width_layout import SHIFT_PATCH_VERSION,FLOOR_PATCH_VERSION
        from femto.planning.planning_revision import digest
        layout_record=params.get('fixed_width_layout') or {}
        layout_version=(layout_record.get('request') or {}).get('version')
        algorithms=revision['inputs']['algorithms']
        if layout_version==FLOOR_PATCH_VERSION or algorithms.get('layout')==FLOOR_PATCH_VERSION or 'minimum_metric_fraction' in algorithms:
            from femto.planning.planning_revision import supported_target_algorithms
            if (layout_version!=FLOOR_PATCH_VERSION or algorithms.get('layout')!=FLOOR_PATCH_VERSION
                    or layout_record['request'].get('minimum_metric_fraction')!=algorithms.get('minimum_metric_fraction')
                    or params.get('continuous_routing')!=algorithms.get('routing')
                    or not supported_target_algorithms(algorithms)):
                raise ValueError('尺度行距下限与计算修订不一致')
        if layout_version in (ADAPTIVE_FLOOR_PATCH_VERSION,COORDINATED_FLOOR_PATCH_VERSION) or algorithms.get('layout') in (ADAPTIVE_FLOOR_PATCH_VERSION,COORDINATED_FLOOR_PATCH_VERSION):
            from femto.planning.planning_revision import supported_target_algorithms
            if (layout_version not in (ADAPTIVE_FLOOR_PATCH_VERSION,COORDINATED_FLOOR_PATCH_VERSION) or algorithms.get('layout')!=layout_version
                    or 'minimum_metric_fraction' in layout_record['request']
                    or params.get('continuous_routing')!=algorithms.get('routing') or not supported_target_algorithms(algorithms)):
                raise ValueError('自适应尺度行距与计算修订不一致')
        if layout_version==SHIFT_PATCH_VERSION or algorithms.get('layout')==SHIFT_PATCH_VERSION:
            shifts=layout_record.get('row_shifts')
            if (layout_version!=SHIFT_PATCH_VERSION or algorithms.get('layout')!=SHIFT_PATCH_VERSION
                    or not isinstance(shifts,list) or not 1<=len(shifts)<=4
                    or algorithms.get('row_shifts_sha256')!=digest(shifts)):
                raise ValueError('研究行偏移与计算修订不一致')
        if (trajectory.planner.get('algorithm_version')=='quality-constrained-surface-v3'
                and (params.get('fixed_width_layout') or {}).get('request',{}).get('version') in
                    ('fixed-width-local-direction-repair-v3','fixed-width-local-direction-budgeted-v4','fixed-width-local-direction-foldback-v5','research-fixed-width-local-direction-adaptive-half-v6','research-fixed-width-local-direction-minimal-v7','fixed-width-local-direction-spacing-floor-v9',ADAPTIVE_FLOOR_PATCH_VERSION,COORDINATED_FLOOR_PATCH_VERSION)):
            from femto.planning.repair_workflow import validate as validate_repair
            validate_repair(trajectory,mesh,region,check)
        reference = revision['inputs']['calibration']['view_reference']
        if (reference.get('derivation') or {}).get('schema') in ('reduced-planning-reference-v1', 'surface-patch-planning-reference-v1'):
            from femto.planning.planning_resolution import route_sample_step_mm
            if params.get('sample_step_mm') != route_sample_step_mm(
                    reference, decision=params.get('sampling_decision'), mesh=mesh):
                raise ValueError('路线采样间隔与简化规划参考的计算修订不一致')
        if (revision['inputs']['geometry']['source_sha256'] != trajectory.source.get('geometry_sha256')
                or revision['inputs']['reference']['origin'] != trajectory.origin_reference
                or context['target']['quality_evidence_sha256'] != params.get('quality_evidence', {}).get('sha256')
                or context['target']['effective_domain_sha256'] != params.get('effective_domain', {}).get('sha256')):
            raise ValueError('路线与原目标来源、质量证据或计算修订不一致')
        # Keep all retained v1/v2/v3 validators unchanged. Their internal
        # candidate projections do not become new calculation results.
        if trajectory.planner.get('algorithm_version') not in ('source-anchored-surface-v1', 'constructed-surface-route-v1', 'constructed-coverage-refined-surface-v1', 'nominal-variable-width-surface-v1', 'nominal-variable-width-refined-surface-v1'):
            validate_trajectory(replace(trajectory, planning_context=None), mesh=mesh,
                                expected_result_sha256=expected_result_sha256, region=region, check=check)
            return
    if not isinstance(trajectory.origin_reference, dict) or trajectory.origin_reference.get("kind") != "initial_color_camera":
        raise ValueError("Surface trajectory requires an initial_color_camera origin")
    if not isinstance(trajectory.planner, dict) or not isinstance(trajectory.source, dict):
        raise ValueError("Invalid trajectory planner/source binding")
    if trajectory.planner.get('algorithm_version') == 'source-anchored-surface-v1':
        from femto.planning.source_anchored_adjustment import validate as validate_anchored
        validate_anchored(trajectory,mesh=mesh,region=region,check=check)
        if expected_result_sha256 is not None and trajectory.to_dict()['result_sha256'] != expected_result_sha256:
            raise ValueError('Surface trajectory result hash mismatch')
        return
    if trajectory.planner.get('algorithm_version') == 'nominal-variable-width-refined-surface-v1':
        from femto.planning.variable_width_refinement import validate as validate_variable_refinement
        validate_variable_refinement(trajectory,mesh=mesh,region=region,check=check)
        if expected_result_sha256 is not None and trajectory.to_dict()['result_sha256'] != expected_result_sha256:
            raise ValueError('Surface trajectory result hash mismatch')
        return
    if trajectory.planner.get('algorithm_version') == 'nominal-variable-width-surface-v1':
        from femto.planning.variable_width_route import validate as validate_variable
        validate_variable(trajectory, mesh=mesh, region=region, check=check)
        if expected_result_sha256 is not None and trajectory.to_dict()['result_sha256'] != expected_result_sha256:
            raise ValueError('Surface trajectory result hash mismatch')
        return
    if trajectory.planner.get('algorithm_version') == 'constructed-surface-route-v1':
        from femto.planning.strict_validation import validate as validate_strict
        validate_strict(trajectory, mesh=mesh, region=region, check=check)
        if expected_result_sha256 is not None and trajectory.to_dict()['result_sha256'] != expected_result_sha256:
            raise ValueError('Surface trajectory result hash mismatch')
        return
    if trajectory.planner.get('algorithm_version') in ('coverage-refined-surface-v1', 'constructed-coverage-refined-surface-v1'):
        from femto.planning.refinement_validation import validate as validate_refinement
        validate_refinement(trajectory, mesh=mesh, region=region, check=check)
        if expected_result_sha256 is not None and trajectory.to_dict()['result_sha256'] != expected_result_sha256:
            raise ValueError('Surface trajectory result hash mismatch')
        return
    if trajectory.planner.get('algorithm_version') == 'quality-constrained-surface-v3':
        from femto.planning.continuous_quality import validate as validate_continuous
        validate_continuous(trajectory,mesh=mesh,region=region,check=check)
        if expected_result_sha256 is not None and trajectory.to_dict()['result_sha256'] != expected_result_sha256:
            raise ValueError('Surface trajectory result hash mismatch')
        return
    if trajectory.planner.get('algorithm_version') in ('quality-constrained-surface-v1', 'quality-constrained-surface-v2'):
        from femto.planning.quality_constrained import validate as validate_quality
        validate_quality(trajectory, mesh=mesh, region=region, check=check)
        if expected_result_sha256 is not None and trajectory.to_dict()['result_sha256'] != expected_result_sha256:
            raise ValueError('Surface trajectory result hash mismatch')
        return
    if not trajectory.ordered_items or not isinstance(trajectory.ordered_items[0], ScanSegment) or not isinstance(trajectory.ordered_items[-1], ScanSegment):
        raise ValueError("Trajectory must start and end with a scan segment")
    vertices = faces = prepared_faces = None
    if mesh is not None:
        from femto.geometry.triangle_support import prepare_triangle_faces
        vertices, faces, prepared_faces = prepare_triangle_faces(
            np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64))
    expected_route = 0
    segments: list[ScanSegment] = []
    gaps: list[Gap] = []
    parameters = trajectory.planner.get("parameters", {})
    sample_step_m = float(parameters.get("sample_step_mm", 1.0)) / 1000
    version = trajectory.planner.get("algorithm_version")
    selection_mode = version in SELECTION_ALGORITHMS
    block_mode = version in BLOCK_ALGORITHMS or selection_mode
    connected_rows = version == CONNECTED_ALGORITHM or block_mode
    if connected_rows and parameters.get("connect_rows") is not True:
        raise ValueError("Connected row trajectory requires connect_rows=True")
    if block_mode and parameters.get("block_planning") is not True:
        raise ValueError("Block trajectory requires block_planning=True")
    if selection_mode != (parameters.get("selection_blocks") is True):
        raise ValueError("selection_blocks=True requires a mesh-slice-serpentine-v4/v4.1/v4.2/v4.3/v4.4 contract")
    if not block_mode and trajectory.blocks is not None:
        raise ValueError("Block records are only valid for block-mode algorithm versions")
    blocks = _validated_blocks(trajectory) if block_mode else None
    for index, item in enumerate(trajectory.ordered_items):
        if bool(index % 2) != isinstance(item, Gap):
            raise ValueError("ordered_items must strictly alternate segment and gap")
        if isinstance(item, ScanSegment):
            if type(item.segment_index) is not int or item.segment_index != len(segments):
                raise ValueError("Invalid segment index")
            if type(item.row_index) is not int or item.row_index < 0:
                raise ValueError("Invalid row index")
            if block_mode:
                if type(item.block_index) is not int or not 0 <= item.block_index < len(blocks):
                    raise ValueError("Every block-mode scan segment needs a valid block_index")
                if segments and item.block_index < segments[-1].block_index:
                    raise ValueError("Blocks must be scanned one after another in visit order")
            elif item.block_index is not None:
                raise ValueError("block_index is only valid for block trajectories")
            expected_route = _validate_scan_points(item, version, connected_rows, selection_mode,
                expected_route, sample_step_m, vertices, faces, _prepared_faces=prepared_faces)
            segments.append(item)
        else:
            if ('connection_deferred' in (item.detail.get('link_failure_reason'),item.detail.get('row_turn_failure_reason'))
                    and parameters.get('fixed_width_layout',{}).get('request',{}).get('version') not in
                    ('fixed-width-local-direction-patches-v1','fixed-width-local-direction-stable-v2','fixed-width-local-direction-repair-v3','fixed-width-local-direction-budgeted-v4','fixed-width-local-direction-foldback-v5','research-fixed-width-local-direction-adaptive-half-v6','research-fixed-width-local-direction-minimal-v7','research-fixed-width-local-direction-bounded-shift-v8','fixed-width-local-direction-spacing-floor-v9',ADAPTIVE_FLOOR_PATCH_VERSION,COORDINATED_FLOOR_PATCH_VERSION)):
                raise ValueError('Deferred connections require the local-patch candidate version')
            _validate_gap_record(item, gaps, selection_mode, block_mode)
            gaps.append(item)
    if len(gaps) != len(segments) - 1:
        raise ValueError("Every adjacent scan segment needs exactly one gap")
    for index, gap in enumerate(gaps):
        before, after = segments[index].points[-1], segments[index + 1].points[0]
        if gap.previous_point != {"route_index": before["route_index"], "xyz_m": before["xyz_m"]}:
            raise ValueError("Gap previous endpoint mismatch")
        if gap.next_point != {"route_index": after["route_index"], "xyz_m": after["xyz_m"]}:
            raise ValueError("Gap next endpoint mismatch")
    if selection_mode:
        if version == ORDERED_SELECTION_ALGORITHM:
            from femto.planning.surface_visit_validation import validate_visit_rows
            validate_visit_rows(blocks, segments, gaps, faces=faces)
        from femto.planning.selection_blocks import check_selection_blocks
        check_selection_blocks(trajectory, blocks, segments, gaps, vertices, faces, region)
    elif block_mode:
        _check_block_consistency(trajectory, blocks, segments, gaps, vertices, faces)
    payload = trajectory.to_dict()
    _forbidden_orientation_keys(payload)
    if expected_result_sha256 is not None and payload["result_sha256"] != expected_result_sha256:
        raise ValueError("Surface trajectory result hash mismatch")


def non_overlapping_labels(boxes: list[tuple[Any, tuple[float, float, float, float]]]
                           ) -> tuple[list[Any], list[Any]]:
    """Display only: keep labels in the given order, skip any whose box overlaps a kept one.

    Boxes are (x0, y0, x1, y1). Returns (kept keys, skipped keys); deterministic, no threshold
    on block size, so small blocks are still numbered whenever there is room."""
    kept: list[tuple[Any, tuple[float, float, float, float]]] = []
    skipped: list[Any] = []
    for key, box in boxes:
        clash = any(box[0] < other[2] and other[0] < box[2] and box[1] < other[3] and other[1] < box[3]
                    for _, other in kept)
        if clash:
            skipped.append(key)
        else:
            kept.append((key, box))
    return [key for key, _ in kept], skipped


def block_view(trajectory: SurfaceTrajectory) -> dict[str, Any] | None:
    """Display-only block summary; never changes ordered_items."""
    if trajectory.blocks is None:
        return None
    selection = trajectory.blocks.get("schema") == SELECTION_BLOCKS_SCHEMA
    chain_blocks = [item.block_index for item in trajectory.ordered_items if isinstance(item, ScanSegment)]
    items = []
    for block in trajectory.blocks["items"]:
        chains = [index for index, value in enumerate(chain_blocks) if value == block["block_index"]]
        keys = ("block_index", "component_id", "status", "reason", "message", "face_count",
                "area_m2", "segment_count", "gap_count", "point_count", "row_count",
                "scan_length_m", "connector_length_m", "connected_row_turns")
        if selection:
            keys += ("connected_detours", "mesh_fragment_count", "planned_fragment_count", "continuity",
                     "coverage", "selection", "revisit", "link_failure_counts")
        items.append({key: block.get(key) for key in keys}
                     | {"chain_indices": chains, "label": f"块 {block['block_index'] + 1}"})
    stats = trajectory.statistics
    summary_keys = ("block_count", "plannable_block_count", "unplannable_block_count", "coverage_status",
                    "inter_block_gap_count", "unplannable_face_count", "unplannable_area_m2")
    if selection:
        summary_keys += ("incomplete_block_count", "continuity_status", "uncovered_fragment_count",
                         "uncovered_face_count", "geometry_unconfirmed_pixel_count", "revisit_connector_length_m")
    summary = {key: stats.get(key) for key in summary_keys}
    return {"chain_blocks": chain_blocks, "items": items, "summary": summary,
            "order_rule": trajectory.blocks.get("order_rule"),
            "block_definition": trajectory.blocks.get("block_definition"),
            "selection_blocks": selection}


_exporting = ContextVar('validated_export_trajectory', default=None)


def _export_fingerprint(trajectory):
    return hashlib.sha256(canonical_json_bytes(trajectory.to_dict())).digest()


def require_current_export(trajectory):
    receipt = _exporting.get()
    if not (receipt is not None and receipt['active'] and receipt['thread'] == get_ident()
            and receipt['model'] is trajectory and receipt['digest'] == _export_fingerprint(trajectory)):
        raise ValueError('路线在导出核验后发生变化')


@contextmanager
def exporting(trajectory: SurfaceTrajectory, *, check=lambda: None):
    """Validate this export's content, including the mutable nested records."""
    check()
    before = _export_fingerprint(trajectory)
    validate_trajectory(trajectory, check=check)
    check()
    if before != _export_fingerprint(trajectory):
        raise ValueError('路线在导出核验期间发生变化')
    receipt = dict(model=trajectory, digest=before, active=True, thread=get_ident())
    token = _exporting.set(receipt)
    try:
        yield
    finally:
        receipt['active'] = False
        _exporting.reset(token)


def to_player_plan(trajectory: SurfaceTrajectory | dict[str, Any]) -> dict[str, Any]:
    if isinstance(trajectory, dict):
        model = SurfaceTrajectory.from_dict(trajectory)
    else:
        model = trajectory
        receipt = _exporting.get()
        if not (receipt is not None and receipt['active'] and receipt['thread'] == get_ident()
                and receipt['model'] is model and receipt['digest'] == _export_fingerprint(model)):
            validate_trajectory(model)
    return _player_plan_view(model)


def _player_plan_view(model: SurfaceTrajectory) -> dict[str, Any]:
    """Pure projection, used only immediately after validating this object.

    Public conversion still validates its input. This helper keeps no cache or
    acceptance flag and must not be used to resume an earlier validation.
    """
    segments = [[dict(point, motion_kind=point.get("role", "scan")) for point in item.points]
                for item in model.ordered_items if isinstance(item, ScanSegment)]
    gaps = [item for item in model.ordered_items if isinstance(item, Gap)]
    blocks = block_view(model)
    chain_blocks = blocks["chain_blocks"] if blocks else None

    def explanation(index: int, gap: Gap) -> str:
        if chain_blocks and chain_blocks[index] != chain_blocks[index + 1]:
            if gap.reason == "selection_break":
                return (f"{gap.reason}：块 {chain_blocks[index] + 1} 与块 {chain_blocks[index + 1] + 1} 是医师分别选择的区域（选择分离），"
                        "块间没有已规划运动；继续仅定位下一块起点")
            return (f"{gap.reason}：块 {chain_blocks[index] + 1} 与块 {chain_blocks[index + 1] + 1} 是拓扑分离的网格表面，"
                    "块间没有已规划运动；继续仅定位下一块起点")
        link = gap.detail.get("link_failure_reason") if isinstance(gap.detail, dict) else None
        if link is not None:
            return f"{gap.reason}：块 {chain_blocks[index] + 1} 内部未能沿可信表面连接（{link}），该块未完成；继续只定位下一段起点"
        return f"{gap.reason}：段间没有已规划运动；继续只定位下一段起点"

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
            "explanation": explanation(index, gap),
            "transition_planned": False, "transition_time_s": None,
            **({"block_from": chain_blocks[index], "block_to": chain_blocks[index + 1],
                "inter_block": chain_blocks[index] != chain_blocks[index + 1]} if chain_blocks else {}),
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
    if blocks is not None:
        result["blocks"] = blocks
        if blocks.get("selection_blocks"):
            result["selection_semantics"] = {
                "work_blocks": "one_block_per_physician_selection_component",
                "inter_block_gap": "selection_break_no_planned_motion",
                "intra_block_gap": "block_incomplete_link_failure_reason_recorded",
                "revisit": "connector_samples_within_half_row_spacing_of_other_scanned_rows_flagged_energy_not_assumed_on",
                "treatment_coverage_claim": "none_without_effective_head_width",
            }
    if model.planner.get("parameters", {}).get("row_spacing_mm") is not None:
        spacing = model.spacing_report
        if model.planner.get("algorithm_version") == ORDERED_SELECTION_ALGORITHM:
            from femto.planning.surface_visit_validation import player_spacing_report
            spacing = player_spacing_report(model.spacing_report, segments)
        result.update(spacing_mm=model.planner["parameters"]["row_spacing_mm"],
                      spacing_semantics="parallel_plane_spacing_not_uniform_surface_spacing",
                      surface_mode={"schema": "femto-surface-trajectory-player-view-v1",
                                    "algorithm": model.planner.get("algorithm_version")},
                      spacing_report=spacing)
    if model.planning_context is not None:
        result['planning_context'] = model.planning_context
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
