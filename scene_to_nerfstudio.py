"""Convert a validated Femto RGB-D scene to a Nerfstudio/gsplat dataset.

The converter is an independent implementation based on the documented
Nerfstudio data format. It does not contain Graphdeco source code.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np

from scene_dataset_validator import load_json, load_jsonl, resolve_dataset, validate_pose
from scene_gsplat_split import make_split, file_sha256
from depth_quality import process_depth_quality


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASETS = ROOT / "datasets"
CV_CAMERA_TO_OPENGL_CAMERA = np.diag([1.0, -1.0, -1.0, 1.0])
DEFAULT_SEED_VOXEL_M = 0.01
DEFAULT_MAX_SEED_POINTS = 1_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Femto scene_* RGB-D + T_WC into Nerfstudio transforms.json"
    )
    parser.add_argument("dataset", nargs="?", type=Path)
    parser.add_argument("--datasets-root", type=Path, default=DEFAULT_DATASETS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-static", action="store_true", help="Allow identical poses for format testing only")
    parser.add_argument("--seed-voxel-mm", type=float, default=DEFAULT_SEED_VOXEL_M * 1000.0)
    parser.add_argument("--max-seed-points", type=int, default=DEFAULT_MAX_SEED_POINTS)
    return parser.parse_args()


def imread_unicode(path: Path, flags: int) -> np.ndarray:
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, flags)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def imwrite_unicode(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError(f"Cannot encode image: {path}")
    path.write_bytes(encoded.tobytes())


def camera_to_opengl(T_WC: np.ndarray) -> np.ndarray:
    """Convert camera-to-world from Orbbec/OpenCV RDF to OpenGL RUB camera axes."""
    return np.asarray(T_WC, dtype=np.float64) @ CV_CAMERA_TO_OPENGL_CAMERA


def pose_diversity(poses: list[np.ndarray]) -> tuple[float, float]:
    centers = np.stack([pose[:3, 3] for pose in poses])
    translation_span = float(np.linalg.norm(centers[:, None] - centers[None, :], axis=-1).max())
    reference = poses[0][:3, :3]
    angles = []
    for pose in poses:
        relative = reference.T @ pose[:3, :3]
        cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
        angles.append(np.degrees(np.arccos(cosine)))
    return translation_span, float(max(angles, default=0.0))


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "median": None, "max": None, "p95": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "max": float(np.max(array)),
        "p95": float(np.percentile(array, 95)),
    }


def _depth_range_m(manifest: dict[str, Any]) -> tuple[float, float]:
    camera = manifest.get("camera") or {}
    candidate = camera.get("depth_range_m")
    if isinstance(candidate, list) and len(candidate) == 2:
        minimum, maximum = float(candidate[0]), float(candidate[1])
        if math.isfinite(minimum) and math.isfinite(maximum) and 0.0 <= minimum < maximum:
            return minimum, maximum
    parameters = manifest.get("parameters") or {}
    minimum = float(parameters.get("depth_min_mm", 100.0)) / 1000.0
    maximum = float(parameters.get("depth_max_mm", 8000.0)) / 1000.0
    if math.isfinite(minimum) and math.isfinite(maximum) and 0.0 <= minimum < maximum:
        return minimum, maximum
    return 0.1, 8.0


def _metadata_number(frame: dict[str, Any], stream: str, field: str) -> float | None:
    value = ((frame.get("frame_metadata") or {}).get(stream) or {}).get(field)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _sample_rgbd_world_points(
    color_bgr: np.ndarray,
    depth_mm: np.ndarray,
    K: np.ndarray,
    T_WC: np.ndarray,
    depth_range_m: tuple[float, float],
    target_points: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Back-project one aligned RGB-D frame into the capture world coordinates."""
    height, width = depth_mm.shape
    stride = max(1, int(math.ceil(math.sqrt((height * width) / max(target_points, 1)))))
    rows = np.arange(0, height, stride, dtype=np.int32)
    cols = np.arange(0, width, stride, dtype=np.int32)
    sampled_depth_m = depth_mm[np.ix_(rows, cols)].astype(np.float64) * 0.001
    valid = (
        np.isfinite(sampled_depth_m)
        & (sampled_depth_m >= depth_range_m[0])
        & (sampled_depth_m <= depth_range_m[1])
    )
    full_valid = (
        np.isfinite(depth_mm)
        & (depth_mm >= depth_range_m[0] * 1000.0)
        & (depth_mm <= depth_range_m[1] * 1000.0)
    )
    valid_ratio = float(np.count_nonzero(full_valid) / full_valid.size)
    if not np.any(valid):
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8), valid_ratio

    grid_x, grid_y = np.meshgrid(cols, rows)
    z = sampled_depth_m[valid]
    x = (grid_x[valid] - float(K[0, 2])) * z / float(K[0, 0])
    y = (grid_y[valid] - float(K[1, 2])) * z / float(K[1, 1])
    camera_points = np.column_stack((x, y, z))
    world_points = camera_points @ T_WC[:3, :3].T + T_WC[:3, 3]
    colors_rgb = color_bgr[np.ix_(rows, cols)][valid][:, ::-1].copy()
    return world_points, colors_rgb, valid_ratio


def _voxel_downsample_first(
    points: np.ndarray,
    colors: np.ndarray,
    voxel_m: float,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    if points.size == 0:
        return points.reshape(0, 3), colors.reshape(0, 3)
    finite = np.isfinite(points).all(axis=1)
    points, colors = points[finite], colors[finite]
    if voxel_m > 0.0 and len(points):
        voxel_keys = np.floor(points / voxel_m).astype(np.int64)
        _, keep = np.unique(voxel_keys, axis=0, return_index=True)
        keep.sort()
        points, colors = points[keep], colors[keep]
    if len(points) > max_points:
        keep = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
        points, colors = points[keep], colors[keep]
    return points, colors


def write_rgb_point_cloud_ply(path: Path, points: np.ndarray, colors_rgb: np.ndarray) -> None:
    """Write the standard XYZRGB vertex fields consumed by Nerfstudio dataparser PLY loading."""
    if len(points) != len(colors_rgb) or not len(points):
        raise ValueError("RGB-D initialization point cloud is empty or inconsistent")
    vertices = np.empty(
        len(points),
        dtype=np.dtype(
            [
                ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                ("red", "u1"), ("green", "u1"), ("blue", "u1"),
            ]
        ),
    )
    vertices["x"], vertices["y"], vertices["z"] = points.T.astype(np.float32)
    vertices["red"], vertices["green"], vertices["blue"] = colors_rgb.T.astype(np.uint8)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    path.write_bytes(header + vertices.tobytes())


def convert(
    dataset: Path,
    output: Path,
    allow_static: bool = False,
    *,
    seed_voxel_m: float = DEFAULT_SEED_VOXEL_M,
    max_seed_points: int = DEFAULT_MAX_SEED_POINTS,
) -> dict[str, Any]:
    manifest = load_json(dataset / "manifest.json")
    frames = load_jsonl(dataset / "frames.jsonl")
    if manifest.get("schema") != "femto-full-scene-rgbd-v1" or not frames:
        raise ValueError("Input is not a non-empty femto-full-scene-rgbd-v1 dataset")
    if output.exists():
        raise FileExistsError(f"Output already exists; refusing to overwrite: {output}")
    if seed_voxel_m <= 0.0 or max_seed_points <= 0:
        raise ValueError("Seed voxel and maximum seed point count must be positive")

    poses_cv: list[np.ndarray] = []
    pose_records: list[dict[str, Any]] = []
    for frame in frames:
        pose_record = load_json(dataset / frame["pose"])
        if not pose_record.get("pose_valid", False):
            raise ValueError(f"Frame {frame['frame']} has no validated T_WC")
        pose_error = validate_pose(pose_record.get("T_WC"))
        if pose_error:
            raise ValueError(f"Frame {frame['frame']} pose error: {pose_error}")
        pose = np.asarray(pose_record["T_WC"], dtype=np.float64)
        poses_cv.append(pose)
        pose_records.append(pose_record)

    translation_span, rotation_span_deg = pose_diversity(poses_cv)
    if not allow_static and translation_span < 0.02 and rotation_span_deg < 2.0:
        raise ValueError(
            "Camera poses have no useful viewpoint diversity; capture a moving-camera sequence "
            "or pass --allow-static only for format testing"
        )

    intrinsic = load_json(dataset / "calibration" / "color_intrinsic.json")
    distortion = load_json(dataset / "calibration" / "distortion.json")["rgb"]
    distortion_model = str(distortion.get("model", ""))
    if "BROWN_CONRADY_K6" not in distortion_model:
        raise ValueError(f"Unsupported RGB distortion model: {distortion_model or 'missing'}")
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

    (output / "images").mkdir(parents=True, exist_ok=False)
    (output / "derived" / "depth").mkdir(parents=True, exist_ok=False)
    (output / "depth_original").mkdir(parents=True, exist_ok=False)
    (output / "derived" / "confidence").mkdir(parents=True, exist_ok=False)
    (output / "quality").mkdir(parents=True, exist_ok=False)
    output_frames: list[dict[str, Any]] = []
    seed_points: list[np.ndarray] = []
    seed_colors: list[np.ndarray] = []
    depth_valid_ratios: list[float] = []
    luminance_medians: list[float] = []
    sync_delta_us: list[float] = []
    color_exposures: list[float] = []
    color_gains: list[float] = []
    depth_quality_frames: list[dict[str, Any]] = []
    local_measured=bool(manifest.get('local_capture',{}).get('measured_only'))
    active_depth_range_m = _depth_range_m(manifest)
    split = make_split(len(frames))
    target_points_per_frame = max(2_000, max_seed_points // max(len(split['train']), 1))
    for ordinal, (frame, pose_cv, pose_record) in enumerate(zip(frames, poses_cv, pose_records)):
        color = imread_unicode(dataset / frame["rgb"], cv2.IMREAD_COLOR)
        depth = imread_unicode(dataset / frame["depth"], cv2.IMREAD_UNCHANGED)
        if color.shape[:2] != (height, width) or depth.shape != (height, width):
            raise ValueError(f"Frame {frame['frame']} dimensions do not match calibration")
        if depth.dtype != np.uint16:
            raise ValueError(f"Frame {frame['frame']} depth is not uint16 millimetres")
        from temporal_keyframe import prepared_depth
        filled_source, source_confidence, depth_quality, depth_quality_paths = prepared_depth(
            dataset,frame,depth,
            active_depth_range_m[0] * 1000.0,
            active_depth_range_m[1] * 1000.0,
            output / "derived" / "depth_quality" / f"frame_{ordinal:06d}",
            rgb_bgr=color,
            keep_artifacts=not local_measured,
        )
        color_rectified = cv2.remap(color, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        if 'original_depth' in frame:
            depth=imread_unicode(dataset/frame['original_depth'],cv2.IMREAD_UNCHANGED)
        depth_rectified = cv2.remap(depth, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        filled_rectified = cv2.remap(filled_source, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        confidence_rectified = cv2.remap(source_confidence, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        points, colors, valid_ratio = _sample_rgbd_world_points(
            color_rectified,
            depth_rectified,
            new_K,
            pose_cv,
            active_depth_range_m,
            target_points_per_frame,
        )
        if len(points) and ordinal in split['train']:
            seed_points.append(points)
            seed_colors.append(colors)
        depth_valid_ratios.append(valid_ratio)
        gray = cv2.cvtColor(color_rectified, cv2.COLOR_BGR2GRAY)
        luminance_medians.append(float(np.median(gray)))
        if frame.get("rgb_depth_timestamp_delta_us") is not None:
            try:
                sync_delta_us.append(abs(float(frame["rgb_depth_timestamp_delta_us"])))
            except (TypeError, ValueError):
                pass
        exposure = _metadata_number(frame, "color", "exposure")
        gain = _metadata_number(frame, "color", "gain")
        if exposure is not None:
            color_exposures.append(exposure)
        if gain is not None:
            color_gains.append(gain)
        stem = f"{ordinal:06d}"
        imwrite_unicode(output / "images" / f"{stem}.png", color_rectified)
        imwrite_unicode(output / "depth_original" / f"{stem}.png", depth_rectified)
        imwrite_unicode(output / "derived" / "depth" / f"{stem}.png", filled_rectified)
        np.save(output / "derived" / "confidence" / f"{stem}.npy", confidence_rectified.astype(np.float32), allow_pickle=False)
        depth_quality_frames.append({
            "source_frame": int(frame["frame"]),
            "original_invalid_rate": depth_quality["original_invalid_rate"],
            "filled_invalid_rate": depth_quality["filled_invalid_rate"],
            "repaired_pixels": depth_quality["repaired_pixels"],
            "policy": depth_quality["policy"],
            "artifacts": depth_quality_paths,
        })
        output_frames.append(
            {
                "file_path": f"images/{stem}.png",
                "depth_file_path": f"derived/depth/{stem}.png",
                "original_depth_file_path": f"depth_original/{stem}.png",
                "depth_confidence_file_path": f"derived/confidence/{stem}.npy",
                "transform_matrix": camera_to_opengl(pose_cv).tolist(),
                "source_frame": int(frame["frame"]),
                "source_camera_timestamp_us": int(frame.get("camera_timestamp_us", -1)),
                "pose_source": pose_record.get("pose_source"),
                "source_frame_metadata": frame.get("frame_metadata") or {},
            }
        )

    if not seed_points:
        raise ValueError("No valid RGB-D points remain in the active Femto depth range")
    points, colors = _voxel_downsample_first(
        np.concatenate(seed_points, axis=0),
        np.concatenate(seed_colors, axis=0),
        seed_voxel_m,
        max_seed_points,
    )
    seed_path = output / "rgbd_init_points.ply"
    write_rgb_point_cloud_ply(seed_path, points, colors)
    split['seed_sha256'] = file_sha256(seed_path)

    luminance_summary = _summary(luminance_medians)
    exposure_summary = _summary(color_exposures)
    gain_summary = _summary(color_gains)
    luma_min = float(luminance_summary["min"] or 0.0)
    luma_max = float(luminance_summary["max"] or 0.0)
    exposure_min = float(exposure_summary["min"] or 0.0)
    exposure_max = float(exposure_summary["max"] or 0.0)
    gain_min = float(gain_summary["min"] or 0.0)
    gain_max = float(gain_summary["max"] or 0.0)
    luminance_ratio = luma_max / max(luma_min, 1.0)
    exposure_ratio = exposure_max / max(exposure_min, 1.0) if color_exposures else 1.0
    gain_ratio = gain_max / max(gain_min, 1.0) if color_gains else 1.0
    photometric_unstable = (
        luminance_ratio > 1.4 or exposure_ratio > 1.5 or gain_ratio > 1.5
    )
    quality_warnings: list[str] = []
    if float(np.median(depth_valid_ratios)) < 0.15:
        quality_warnings.append("median valid depth coverage is below 15%")
    if photometric_unstable:
        quality_warnings.append("large brightness/exposure variation; exposure compensation is recommended")
    sync_summary = _summary(sync_delta_us)
    if float(sync_summary["p95"] or 0.0) > 5_000.0:
        quality_warnings.append("RGB-Depth timestamp delta p95 exceeds 5 ms")

    quality_report = {
        "schema": "femto-3dgs-input-quality-v1",
        "frame_count": len(output_frames),
        "depth_valid_ratio": _summary(depth_valid_ratios),
        "rgb_depth_abs_timestamp_delta_us": sync_summary,
        "color_luminance_median": luminance_summary,
        "color_exposure_metadata": exposure_summary,
        "color_gain_metadata": gain_summary,
        "photometric_unstable": photometric_unstable,
        "warnings": quality_warnings,
        "depth_hole_filling": {
            "frames": depth_quality_frames,
            "training_weight": "original=1.0, repaired=0.3, unknown=0.0",
            "evaluation_truth": "original observed depth only",
        },
    }
    quality_report_path = output / "quality" / "3dgs_input_report.json"
    quality_report_path.write_text(
        json.dumps(quality_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    transforms = {
        "camera_model": "OPENCV",
        "fl_x": float(new_K[0, 0]),
        "fl_y": float(new_K[1, 1]),
        "cx": float(new_K[0, 2]),
        "cy": float(new_K[1, 2]),
        "w": width,
        "h": height,
        "k1": 0.0,
        "k2": 0.0,
        "k3": 0.0,
        "k4": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "depth_unit_scale_factor": 0.001,
        "depth_confidence_values": ({"range": [0.0, 1.0], "repaired_max": 0.3, "unknown": 0.0,
                                     "policy": "preserve temporal confidence; temporal observations are not independent ground truth"}
                                    if manifest.get('temporal_fusion') else {"original": 1.0, "repaired": 0.3, "unknown": 0.0}),
        "ply_file_path": seed_path.name,
        "frames": output_frames,
        "dataset_split": split,
        "train_filenames": [output_frames[index]['file_path'] for index in split['train']],
        "val_filenames": [output_frames[index]['file_path'] for index in split['validation']],
        "test_filenames": [output_frames[index]['file_path'] for index in split['test']],
    }
    (output / "transforms.json").write_text(
        json.dumps(transforms, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    conversion_manifest = {
        "schema": "femto-to-nerfstudio-v2",
        "created_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "source_dataset": str(dataset),
        "source_schema": manifest["schema"],
        "dataset_split": split,
        "frame_count": len(output_frames),
        "translation_span_m": translation_span,
        "rotation_span_deg": rotation_span_deg,
        "source_camera_axes": "+X right, +Y down, +Z forward",
        "target_camera_axes": "+X right, +Y up, +Z back; -Z view direction",
        "camera_axis_conversion_right_multiply": CV_CAMERA_TO_OPENGL_CAMERA.tolist(),
        "image_processing": "OpenCV full rational distortion rectification; RGB PNG linear, depth PNG nearest",
        "depth_hole_filling": {
            "enabled": True,
            "seed_geometry": "original observed depth only",
            "training_depth": "derived filled depth with per-pixel confidence",
            "evaluation_depth": "original observed depth only",
            "frame_reports": depth_quality_frames,
        },
        "source_distortion_model": distortion_model,
        "depth_unit": "uint16 millimetres; zero is unknown",
        "active_depth_range_m": list(active_depth_range_m),
        "initialization": {
            "type": "aligned RGB-D world point cloud",
            "ply_file_path": seed_path.name,
            "point_count": int(len(points)),
            "voxel_m": float(seed_voxel_m),
            "max_points": int(max_seed_points),
            "coordinate_system": "source world coordinates shared by camera poses",
            "semantic_mask_applied": False,
            "seed_frame_indices": split['train'],
            "seed_sha256": split['seed_sha256'],
        },
        "photometric": {
            "unstable": photometric_unstable,
            "luminance_median_ratio": luminance_ratio,
            "exposure_ratio": exposure_ratio,
            "gain_ratio": gain_ratio,
            "exposure_metadata_available": bool(color_exposures),
        },
        "quality_report": str(quality_report_path.relative_to(output).as_posix()),
        "commercial_provenance": "independent adapter using documented Nerfstudio format; no Graphdeco source",
    }
    (output / "conversion_manifest.json").write_text(
        json.dumps(conversion_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return conversion_manifest


def main() -> int:
    args = parse_args()
    try:
        dataset = resolve_dataset(args.dataset, args.datasets_root)
        output = (args.output or (dataset / "derived" / "nerfstudio")).resolve()
        report = convert(
            dataset,
            output,
            allow_static=args.allow_static,
            seed_voxel_m=args.seed_voxel_mm / 1000.0,
            max_seed_points=args.max_seed_points,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "blocked", "error": str(error)}, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps({"status": "converted", "output": str(output), **report}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
