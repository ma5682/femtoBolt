"""Calibrated RGB-D frame primitives: K6 undistortion maps, Open3D RGBD preparation, pose loading."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import open3d as o3d

from femto.dataset.scene_dataset_validator import validate_pose
from femto.io_utils import load_json
from femto.operation_diagnostics import diagnosed


def distortion_maps(dataset: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    intrinsic = load_json(dataset / "calibration" / "color_intrinsic.json")
    distortion = load_json(dataset / "calibration" / "distortion.json")["rgb"]
    model = str(distortion.get("model", ""))
    if "BROWN_CONRADY_K6" not in model:
        raise ValueError(f"Unsupported RGB distortion model: {model or 'missing'}")
    width, height = int(intrinsic["width"]), int(intrinsic["height"])
    K = np.array(
        [[intrinsic["fx"], 0.0, intrinsic["cx"]], [0.0, intrinsic["fy"], intrinsic["cy"]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    coefficients = np.array(
        [
            distortion["k1"], distortion["k2"], distortion["p1"], distortion["p2"],
            distortion["k3"], distortion["k4"], distortion["k5"], distortion["k6"],
        ],
        dtype=np.float64,
    )
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, coefficients, (width, height), 0.0, (width, height))
    map_x, map_y = cv2.initUndistortRectifyMap(
        K, coefficients, np.eye(3), new_K, (width, height), cv2.CV_32FC1
    )
    return map_x, map_y, new_K, width, height

@diagnosed('reconstruction.rgbd_prepare')
def prepare_rgbd_arrays(
    color_bgr: np.ndarray,
    depth_u16: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
    K: np.ndarray,
    scale: float,
    depth_min_m: float,
    depth_max_m: float,
    *, depth_filter=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calibrate/filter RGB-D pixels once, without constructing an Open3D image."""
    color = cv2.remap(color_bgr, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    depth = prepare_depth(depth_u16, map_x, map_y, depth_min_m, depth_max_m)
    if depth_filter is not None:
        depth = depth_filter(depth)
    height, width = depth.shape
    if scale != 1.0:
        scaled_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
        color = cv2.resize(color, scaled_size, interpolation=cv2.INTER_AREA)
        depth = cv2.resize(depth, scaled_size, interpolation=cv2.INTER_NEAREST)
    scaled_K = K.copy()
    scaled_K[0, :] *= scale
    scaled_K[1, :] *= scale
    return color, depth, scaled_K


def prepare_depth(depth_u16, map_x, map_y, depth_min_m, depth_max_m):
    """Undistort measured depth and apply the inclusive millimetre range."""
    depth = cv2.remap(depth_u16, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    valid = (depth >= int(round(depth_min_m * 1000.0))) & (depth <= int(round(depth_max_m * 1000.0)))
    return np.where(valid, depth, 0).astype(np.uint16)


def prepare_rgbd(
    color_bgr: np.ndarray, depth_u16: np.ndarray, map_x: np.ndarray, map_y: np.ndarray,
    K: np.ndarray, scale: float, depth_min_m: float, depth_max_m: float, *, depth_filter=None,
) -> tuple[o3d.geometry.RGBDImage, o3d.camera.PinholeCameraIntrinsic, np.ndarray, np.ndarray]:
    color, depth, scaled_K = prepare_rgbd_arrays(color_bgr, depth_u16, map_x, map_y,
        K, scale, depth_min_m, depth_max_m, depth_filter=depth_filter)
    return rgbd_from_prepared(color, depth, scaled_K, depth_max_m)


def rgbd_from_prepared(color, depth, scaled_K, depth_max_m):
    """Build an RGBD image from already calibrated pixels without filtering twice."""
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        color.shape[1], color.shape[0], scaled_K[0, 0], scaled_K[1, 1], scaled_K[0, 2], scaled_K[1, 2]
    )
    rgb = np.ascontiguousarray(color[:, :, ::-1])
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(rgb),
        o3d.geometry.Image(np.ascontiguousarray(depth)),
        depth_scale=1000.0,
        # Open3D drops z >= truncation; retain the inclusive millimetre endpoint.
        depth_trunc=depth_max_m + 0.0005,
        convert_rgb_to_intensity=False,
    )
    return rgbd, intrinsic, color, depth

def load_poses(dataset: Path, frames: list[dict[str, Any]]) -> list[np.ndarray]:
    poses: list[np.ndarray] = []
    for frame in frames:
        record = load_json(dataset / frame["pose"])
        if not record.get("pose_valid", False):
            raise ValueError(f"Frame {frame['frame']} does not have a valid T_WC")
        error = validate_pose(record.get("T_WC"))
        if error:
            raise ValueError(f"Frame {frame['frame']} pose error: {error}")
        poses.append(np.asarray(record["T_WC"], dtype=np.float64))
    return poses
