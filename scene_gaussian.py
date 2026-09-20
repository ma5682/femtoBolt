"""Create a commercial-clean Gaussian initialization from an RGB-D point cloud.

This module only writes the documented 3D Gaussian PLY attribute layout.  It
does not contain Graphdeco source code and does not train the Gaussian model.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import open3d as o3d

from fusion import write_point_cloud_safe


SH_C0 = 0.28209479177387814


def _select_cloud(cloud: o3d.geometry.PointCloud, voxel_m: float, max_points: int) -> o3d.geometry.PointCloud:
    selected = cloud.voxel_down_sample(voxel_m) if voxel_m > 0 else o3d.geometry.PointCloud(cloud)
    if len(selected.points) > max_points:
        indexes = np.linspace(0, len(selected.points) - 1, max_points, dtype=np.int64)
        selected = selected.select_by_index(indexes.tolist())
    return selected


def _nearest_scales(cloud: o3d.geometry.PointCloud, fallback_m: float) -> np.ndarray:
    if len(cloud.points) < 2:
        return np.full(len(cloud.points), fallback_m, dtype=np.float32)
    distances = np.asarray(cloud.compute_nearest_neighbor_distance(), dtype=np.float64)
    valid = distances[np.isfinite(distances) & (distances > 0)]
    fallback = float(np.median(valid)) if len(valid) else fallback_m
    distances = np.where(np.isfinite(distances) & (distances > 0), distances, fallback)
    return np.clip(distances * 0.75, 0.002, 0.05).astype(np.float32)


def write_gaussian_ply(
    path: Path,
    points_m: np.ndarray,
    colors_rgb: np.ndarray,
    scales_m: np.ndarray,
    *,
    opacity: float = 0.85,
) -> None:
    """Write binary little-endian PLY with standard 3DGS scalar properties."""
    points = np.asarray(points_m, dtype=np.float32)
    colors = np.asarray(colors_rgb, dtype=np.float32)
    scales = np.asarray(scales_m, dtype=np.float32).reshape(-1)
    if points.ndim != 2 or points.shape[1] != 3 or colors.shape != points.shape:
        raise ValueError("points and colors must both have shape (N, 3)")
    if scales.shape != (len(points),):
        raise ValueError("scales must have shape (N,)")
    if not 0.0 < opacity < 1.0:
        raise ValueError("opacity must be strictly between zero and one")

    names = [
        "x", "y", "z", "nx", "ny", "nz",
        "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
        "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3",
    ]
    dtype = np.dtype([(name, "<f4") for name in names])
    records = np.zeros(len(points), dtype=dtype)
    for axis, name in enumerate(("x", "y", "z")):
        records[name] = points[:, axis]
    sh_dc = (np.clip(colors, 0.0, 1.0) - 0.5) / SH_C0
    for axis, name in enumerate(("f_dc_0", "f_dc_1", "f_dc_2")):
        records[name] = sh_dc[:, axis]
    records["opacity"] = math.log(opacity / (1.0 - opacity))
    log_scales = np.log(np.maximum(scales, 1e-6))
    for name in ("scale_0", "scale_1", "scale_2"):
        records[name] = log_scales
    records["rot_0"] = 1.0

    header = [
        "ply",
        "format binary_little_endian 1.0",
        "comment Femto scene Gaussian initialization; untrained",
        f"element vertex {len(records)}",
        *[f"property float {name}" for name in names],
        "end_header",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        output.write("\n".join(header).encode("ascii"))
        records.tofile(output)


def export_gaussian_initialization(
    cloud: o3d.geometry.PointCloud,
    output_dir: Path,
    *,
    voxel_m: float = 0.015,
    max_points: int = 100_000,
    source_dataset: str | None = None,
) -> dict[str, Any]:
    selected = _select_cloud(cloud, voxel_m, max_points)
    points = np.asarray(selected.points, dtype=np.float64)
    colors = np.asarray(selected.colors, dtype=np.float64)
    if not len(points):
        raise ValueError("Cannot initialize Gaussians from an empty point cloud")
    if colors.shape != points.shape:
        colors = np.full(points.shape, 0.65, dtype=np.float64)
        selected.colors = o3d.utility.Vector3dVector(colors)
    scales = _nearest_scales(selected, voxel_m)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "gaussian_init_untrained.ply"
    preview_path = output_dir / "gaussian_init_preview.ply"
    write_gaussian_ply(model_path, points, colors, scales)
    if not write_point_cloud_safe(preview_path, selected):
        raise RuntimeError("Open3D failed to write the Gaussian preview point cloud")
    manifest = {
        "schema": "femto-gaussian-initialization-v1",
        "created_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "status": "initialization_only_untrained",
        "source_dataset": source_dataset,
        "point_count": int(len(points)),
        "coordinate_system": "same world coordinates as source T_WC; metres",
        "attributes": [
            "x/y/z", "nx/ny/nz", "f_dc_0..2", "opacity",
            "scale_0..2", "rot_0..3",
        ],
        "initialization": {
            "voxel_m": voxel_m,
            "max_points": max_points,
            "opacity": 0.85,
            "scale": "0.75 * nearest-neighbour distance, clipped to 2-50 mm",
            "rotation_quaternion": "identity, scalar-first",
        },
        "warning": (
            "This PLY is a geometry-derived Gaussian initialization and point-splat preview. "
            "It has not been optimized by 3DGS training and must not be presented as a trained photoreal model."
        ),
        "commercial_provenance": "independent PLY writer; no Graphdeco source",
    }
    manifest_path = output_dir / "gaussian_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "model": str(model_path),
        "preview": str(preview_path),
        "manifest": str(manifest_path),
        "point_count": int(len(points)),
        "trained": False,
    }
