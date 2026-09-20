"""Conservative single-observation cleanup and auditable geometry comparison."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import open3d as o3d


def clean_depth(depth: np.ndarray) -> tuple[np.ndarray, dict]:
    """Reject isolated depth returns; never average, fill or move retained samples."""
    valid = np.isfinite(depth) & (depth >= 250) & (depth <= 2210)
    values = np.where(valid, depth, 0).astype(np.int32)
    padded = np.pad(values, 1)
    support = np.zeros(depth.shape, np.uint8)
    height, width = depth.shape
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == dy == 0:
                continue
            neighbor = padded[1 + dy:1 + dy + height, 1 + dx:1 + dx + width]
            support += ((neighbor > 0) & (np.abs(neighbor - values) <= 30))
    keep = valid & (support >= 2)
    cleaned = np.where(keep, values, 0).astype(np.uint16)
    return cleaned, {
        "method": "single_frame_8_neighbor_depth_support",
        "depth_range_mm_inclusive": [250, 2210],
        "neighbor_depth_tolerance_mm": 30, "minimum_neighbors": 2,
        "input_pixels": int(depth.size), "in_range_pixels": int(valid.sum()),
        "isolated_pixels_removed": int((valid & ~keep).sum()),
        "retained_pixels": int(keep.sum()),
        "smoothing": False, "hole_filling": False, "temporal_merge": False,
        "limitation": "Conservative rejection only; coherent outlier patches can remain. Thin isolated features may be removed; compare raw evidence.",
    }


def evidence_manifest(dataset: Path, frame: dict, output: Path) -> dict:
    """Fingerprint preserved inputs; derived products never overwrite these files."""
    files = [dataset / name for name in ("manifest.json", "frames.jsonl", frame["rgb"], frame["depth"], frame["pose"])]
    files += sorted((dataset / "calibration").glob("*.json"))
    entries = []
    for path in files:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        entries.append({"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest.hexdigest()})
    report = {"schema": "femto-single-frame-evidence-v1", "source_dataset": str(dataset),
              "frame": frame, "files": entries,
              "depth_unit": "uint16 millimetres; zero unknown", "geometry_unit": "metre",
              "coordinate_system": "World RDF (+X right, +Y down, +Z forward); T_WC maps camera to world"}
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _sample(points: np.ndarray, limit: int = 100000) -> np.ndarray:
    if len(points) <= limit:
        return points
    return points[np.linspace(0, len(points) - 1, limit, dtype=np.int64)]


def _stats(distances: np.ndarray, total: int) -> dict:
    return {"total_points": int(total), "evaluated_points": len(distances),
            "median_mm": float(np.median(distances) * 1000),
            "p95_mm": float(np.quantile(distances, .95) * 1000),
            "max_mm": float(np.max(distances) * 1000),
            "fraction_over_10mm": float(np.mean(distances > .01))}


def compare_geometry(raw, cleaned, mesh, cleaning: dict, output: Path) -> dict:
    """Internal surface consistency, explicitly not a physical accuracy certificate."""
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    comparisons = {}
    for name, cloud in (("raw_to_mesh", raw), ("cleaned_to_mesh", cleaned)):
        points = np.asarray(cloud.points)
        distances = scene.compute_distance(o3d.core.Tensor(_sample(points), dtype=o3d.core.Dtype.Float32)).numpy()
        comparisons[name] = _stats(distances, len(points))
    vertices = np.asarray(mesh.vertices)
    vertex_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(_sample(vertices)))
    comparisons["mesh_vertices_to_cleaned_points"] = _stats(
        np.asarray(vertex_cloud.compute_point_cloud_distance(cleaned)), len(vertices))
    report = {"schema": "femto-single-frame-quality-v1", "frame_count": 1,
              "coordinate_frame": "world_rdf", "geometry_unit": "metre",
              "cleaning": cleaning, "surface_consistency": comparisons,
              "sampling": "Deterministic uniform indices; up to 100000 points per comparison",
              "diagnostic_threshold_mm": 10,
              "diagnostic_threshold_note": "Coverage diagnostic only, not an acceptance tolerance",
              "scale_validation": {"status": "pending_physical_measurement", "accepted": None},
              "range_endpoint_2_21m": "pending_physical_test", "accepted": None,
              "limitations": ["Distances compare products of the same observation, not physical accuracy.",
                              "Meshing can exclude unsupported or disconnected source points; zero same-source distance is not proof of accuracy.",
                              "Single-view occlusions and unseen surfaces remain unknown."]}
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
