"""Treatment-head parameters for surface trajectories and a geometric coverage estimate.

Legacy mesh planners use only row spacing; the quality-constrained workbench also
uses effective width for its swept-footprint check. Treatment depth, housing diameter and the
safety margin are recorded for future focus placement / collision checks and are not used.
The coverage number is a geometric estimate on the measured region mesh, not treatment
coverage; connectors are never counted because energy is not assumed to be on while moving.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
import json
import math

import numpy as np

from femto.planning.surface_trajectory import ScanSegment, SurfaceTrajectory, canonical_json_bytes


SCHEMA = "femto-treatment-head-parameters-v1"
FILE_NAME = "treatment_head.json"
MODES = ("direct_row_spacing", "from_effective_width", "fixed_width_auto")
ROW_SPACING_RANGE_MM = (0.5, 100.0)  # the planner's existing accepted range; never clamped
PLACEHOLDER_DEFAULTS = {"row_spacing_mm": 5.0, "effective_treatment_width_mm": 5.0, "overlap_percent": 20.0,
                        "treatment_depth_mm": 20.0, "head_diameter_mm": 40.0, "safety_margin_mm": 5.0}
RANGES = {"row_spacing_mm": ROW_SPACING_RANGE_MM, "effective_treatment_width_mm": (0.5, 100.0),
          "overlap_percent": (0.0, 90.0), "treatment_depth_mm": (0.0, 200.0),
          "head_diameter_mm": (1.0, 300.0), "safety_margin_mm": (0.0, 100.0)}
COVERAGE_RULE = ("region_face_centroid_within_half_effective_width_euclidean_3d_of_nearest_scan_sample;"
                 "connector_samples_excluded;area_weighted")


def _number(name: str, value: Any) -> float:
    low, high = RANGES[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"Invalid {name}")
    value = float(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be within {low:g}–{high:g}")
    return value


def head_parameters(mode: str, *, row_spacing_mm: Any = None,
                    effective_treatment_width_mm: Any = PLACEHOLDER_DEFAULTS["effective_treatment_width_mm"],
                    overlap_percent: Any = PLACEHOLDER_DEFAULTS["overlap_percent"],
                    treatment_depth_mm: Any = PLACEHOLDER_DEFAULTS["treatment_depth_mm"],
                    head_diameter_mm: Any = PLACEHOLDER_DEFAULTS["head_diameter_mm"],
                    safety_margin_mm: Any = PLACEHOLDER_DEFAULTS["safety_margin_mm"],
                    values_status: str = "placeholder_not_device_parameters") -> dict[str, Any]:
    """Validated parameter record; the row spacing is computed, never clamped to the planner range."""
    if mode not in MODES:
        raise ValueError("Invalid treatment-head mode")
    if values_status not in ("placeholder_not_device_parameters", "user_entered_unverified"):
        raise ValueError("Invalid treatment-head values_status")
    width = _number("effective_treatment_width_mm", effective_treatment_width_mm)
    overlap = _number("overlap_percent", overlap_percent)
    if mode in ("from_effective_width", "fixed_width_auto"):
        spacing = round(width * (1.0 - overlap / 100.0), 6)
        low, high = ROW_SPACING_RANGE_MM
        if not low <= spacing <= high:
            raise ValueError(f"Computed row spacing {spacing:g} mm is outside the planner range {low:g}–{high:g} mm")
    else:
        if row_spacing_mm is None:
            raise ValueError("Direct mode requires row_spacing_mm")
        spacing = _number("row_spacing_mm", row_spacing_mm)
    return {
        "schema": SCHEMA, "mode": mode, "values_status": values_status,
        "row_spacing_mm": spacing,
        "row_spacing_rule": ("maximum_spacing_for_fixed_width_auto_layout" if mode == "fixed_width_auto" else
                             "effective_treatment_width_mm*(1-overlap_percent/100)" if mode == "from_effective_width"
                             else "entered_directly"),
        "effective_treatment_width_mm": width, "overlap_percent": overlap,
        "overlap_used_for_row_spacing": mode in ("from_effective_width", "fixed_width_auto"),
        "deferred": {
            "treatment_depth_mm": _number("treatment_depth_mm", treatment_depth_mm),
            "head_diameter_mm": _number("head_diameter_mm", head_diameter_mm),
            "safety_margin_mm": _number("safety_margin_mm", safety_margin_mm),
            "usage": "recorded_only_not_used_by_planner",
            "safety_margin_semantics": "housing_clearance_for_future_collision_check_not_region_inset",
            "treatment_depth_semantics": "future_focus_offset_along_surface_normal_not_applied",
        },
        "execution_ready": False, "physical_accuracy_verified": False,
    }


def _inputs(record: dict[str, Any]) -> dict[str, Any]:
    deferred = record.get("deferred") or {}
    return dict(row_spacing_mm=record.get("row_spacing_mm") if record.get("mode") == "direct_row_spacing" else None,
                effective_treatment_width_mm=record.get("effective_treatment_width_mm"),
                overlap_percent=record.get("overlap_percent"),
                treatment_depth_mm=deferred.get("treatment_depth_mm"),
                head_diameter_mm=deferred.get("head_diameter_mm"),
                safety_margin_mm=deferred.get("safety_margin_mm"),
                values_status=record.get("values_status"))


def _region_mesh(trajectory: SurfaceTrajectory) -> Any:
    import trimesh
    manifest_path = Path(trajectory.source["region_manifest"]).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    return trimesh.load(manifest_path.parent / manifest["files"]["mesh_file"], force="mesh", process=False)


def scan_samples(trajectory: SurfaceTrajectory) -> np.ndarray:
    points = [point["xyz_m"] for item in trajectory.ordered_items if isinstance(item, ScanSegment)
              for point in item.points if point.get("role", "scan") == "scan"]
    return np.asarray(points, dtype=float).reshape(-1, 3)


def coverage_estimate(trajectory: SurfaceTrajectory, width_mm: float, mesh: Any = None,
                      check: Callable[[], None] = lambda: None) -> dict[str, Any]:
    """Area of the region mesh whose face centroid lies within half the effective width of a scan sample."""
    mesh = _region_mesh(trajectory) if mesh is None else mesh
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    corners = vertices[faces]
    area = 0.5 * np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1)
    centroids = corners.mean(axis=1)
    samples = scan_samples(trajectory)
    check()
    half = width_mm / 2000.0
    if len(samples):
        import open3d as o3d
        search = o3d.core.nns.NearestNeighborSearch(o3d.core.Tensor(samples))
        search.knn_index()
        _, squared = search.knn_search(o3d.core.Tensor(centroids), 1)
        distance = np.sqrt(squared.numpy().reshape(-1).astype(float))
    else:
        distance = np.full(len(faces), np.inf)
    check()
    covered = distance <= half
    # Faces touching a mesh boundary edge (outer edge of the region or a hole rim).
    edges = np.sort(faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_vertices = np.zeros(len(vertices), dtype=bool)
    boundary_vertices[unique[counts == 1].ravel()] = True
    near_boundary = boundary_vertices[faces].any(axis=1)
    total = float(area.sum())
    uncovered = ~covered

    def summary(mask: np.ndarray) -> dict[str, Any]:
        return {"face_count": int(mask.sum()), "area_m2": round(float(area[mask].sum()), 9)}

    return {
        "rule": COVERAGE_RULE, "effective_treatment_width_mm": float(width_mm),
        "scan_sample_count": int(len(samples)), "region_face_count": int(len(faces)),
        "region_area_m2": round(total, 9), "covered": summary(covered),
        "covered_area_ratio": None if total <= 0 else round(float(area[covered].sum()) / total, 6),
        "uncovered_near_boundary_or_hole": summary(uncovered & near_boundary),
        "uncovered_interior": summary(uncovered & ~near_boundary),
        "caveats": ["nearest_scan_sample_not_polyline_underestimates_by_at_most_half_sample_step",
                    "euclidean_3d_not_geodesic", "face_centroid_decides_whole_face",
                    "placeholder_width_if_values_status_is_placeholder"],
        "claim": "geometric_estimate_on_measured_region_mesh_not_treatment_coverage",
        "decision_use": "none",
    }


def _payload(trajectory: SurfaceTrajectory, record: dict[str, Any], mesh: Any, check: Callable[[], None]) -> dict[str, Any]:
    planned = trajectory.planner.get("parameters", {}).get("row_spacing_mm")
    if planned is None or abs(float(planned) - record["row_spacing_mm"]) > 1e-9:
        raise ValueError("Treatment-head row spacing does not match the trajectory planner parameters")
    return dict(record, result_sha256=trajectory.to_dict()["result_sha256"],
                coverage_estimate=coverage_estimate(trajectory, record["effective_treatment_width_mm"], mesh, check))


def report_lines(payload: dict[str, Any]) -> list[str]:
    coverage = payload["coverage_estimate"]
    deferred = payload["deferred"]
    ratio = coverage["covered_area_ratio"]
    placeholder = payload["values_status"] == "placeholder_not_device_parameters"
    mode = ("按有效治疗宽度计算：{:g} mm ×（1 − {:g}%）".format(payload["effective_treatment_width_mm"], payload["overlap_percent"])
            if payload["mode"] == "from_effective_width" else "直接指定")
    return [
        "", "【治疗头参数】" + ("（占位值，非设备参数）" if placeholder else "（用户输入，未经设备核验）"),
        f"行距 {payload['row_spacing_mm']:g} mm，来源：{mode}",
        f"有效治疗宽度 {payload['effective_treatment_width_mm']:g} mm；重叠率 {payload['overlap_percent']:g}%"
        + ("" if payload["overlap_used_for_row_spacing"] else "（直接指定模式下未用于行距）"),
        f"仅记录、未参与规划：治疗深度 {deferred['treatment_depth_mm']:g} mm、治疗头直径 {deferred['head_diameter_mm']:g} mm、"
        f"安全边界 {deferred['safety_margin_mm']:g} mm（外壳避让余量，留给以后碰撞检测；不是选区内缩）",
        "几何覆盖估算（非治疗覆盖；连接段不计入；面心到最近扫描采样的三维距离 ≤ 半个有效宽度）：",
        "  区域面积 {:.1f} cm²，估算覆盖 {:.1f} cm²（{}）".format(
            coverage["region_area_m2"] * 1e4, coverage["covered"]["area_m2"] * 1e4,
            "无法计算" if ratio is None else f"{ratio * 100:.1f}%"),
        "  未覆盖：边缘或孔洞周边 {} 面 / {:.2f} cm²；内部 {} 面 / {:.2f} cm²".format(
            coverage["uncovered_near_boundary_or_hole"]["face_count"],
            coverage["uncovered_near_boundary_or_hole"]["area_m2"] * 1e4,
            coverage["uncovered_interior"]["face_count"], coverage["uncovered_interior"]["area_m2"] * 1e4),
    ]


def write_record(folder: Path, trajectory: SurfaceTrajectory, record: dict[str, Any],
                 check: Callable[[], None] = lambda: None) -> dict[str, Any]:
    """Write treatment_head.json into a staging folder and append its section to the report."""
    payload = _payload(trajectory, record, None, check)
    (Path(folder) / FILE_NAME).write_bytes(canonical_json_bytes(payload))
    report = Path(folder) / "trajectory_report.txt"
    if report.is_file():
        text = report.read_text(encoding="utf-8-sig").rstrip("\n")
        report.write_text(text + "\n" + "\n".join(report_lines(payload)) + "\n", encoding="utf-8-sig")
    return payload


def load_record(folder: str | Path, trajectory: SurfaceTrajectory, mesh: Any = None,
                check: Callable[[], None] = lambda: None) -> dict[str, Any] | None:
    """None for trajectories without a record; otherwise recompute everything and require equality."""
    path = Path(folder) / FILE_NAME
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ValueError("Treatment-head record schema mismatch")
    if payload.get("result_sha256") != trajectory.to_dict()["result_sha256"]:
        raise ValueError("Treatment-head record is bound to a different trajectory result")
    record = head_parameters(payload.get("mode"), **_inputs(payload))
    if json.loads(canonical_json_bytes(_payload(trajectory, record, mesh, check))) != payload:
        raise ValueError("Treatment-head record does not reproduce from its inputs and the trajectory")
    return payload
