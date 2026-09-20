"""Canonical K6 (BROWN_CONRADY_K6) back-projection of an aligned depth image to a metric point cloud."""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import open3d as o3d


def depth_cloud(
    depth_mm: np.ndarray, color_bgr: np.ndarray, intrinsics: Any, distortion: dict,
    minimum_mm: float, maximum_mm: float, *, confidence: np.ndarray | None = None,
    temporal_confidence: bool = False,
) -> o3d.geometry.PointCloud:
    valid = np.isfinite(depth_mm) & (depth_mm > 0)
    valid &= (depth_mm >= minimum_mm) & (depth_mm <= maximum_mm)
    if confidence is not None:
        if confidence.shape != depth_mm.shape or not np.isfinite(confidence).all():
            raise ValueError("Depth confidence must match the pixel grid and be finite")
        if temporal_confidence and (np.any(confidence < 0) or np.any(confidence > 1)):
            raise ValueError("Temporal confidence must be in [0,1]")
        if not temporal_confidence and not np.isin(confidence, np.array([0,.3,1],np.float32)).all():
            raise ValueError("Expected confidence 0, 0.3 or 1")
        valid &= confidence > 0
    rows, columns = np.nonzero(valid)
    cloud = o3d.geometry.PointCloud()
    if not len(rows):
        return cloud if confidence is None else (cloud, np.empty(0,np.float32))
    if "BROWN_CONRADY_K6" not in str(distortion.get("model", "")):
        raise ValueError("Full-range fusion requires the RGB BROWN_CONRADY_K6 calibration")
    matrix = np.array([
        [intrinsics.fx, 0.0, intrinsics.cx],
        [0.0, intrinsics.fy, intrinsics.cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    coefficients = np.array([
        distortion[key] for key in ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")
    ], dtype=np.float64)
    pixels = np.column_stack((columns, rows)).astype(np.float64).reshape(-1, 1, 2)
    rays = cv2.undistortPoints(pixels, matrix, coefficients).reshape(-1, 2)
    depth_m = depth_mm[rows, columns].astype(np.float64) / 1000.0
    points = np.column_stack((rays * depth_m[:, None], depth_m))
    finite = np.isfinite(points).all(axis=1)
    cloud.points = o3d.utility.Vector3dVector(points[finite])
    cloud.colors = o3d.utility.Vector3dVector(
        color_bgr[rows, columns, ::-1][finite].astype(np.float64) / 255.0
    )
    return cloud if confidence is None else (cloud, confidence[rows,columns][finite].astype(np.float32))
