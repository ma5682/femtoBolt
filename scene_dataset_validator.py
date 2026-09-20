"""Validate a Femto full-scene RGB-D dataset before reconstruction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from io_utils import imread_unicode_or_none, load_json, load_jsonl


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASETS = ROOT / "datasets"
REQUIRED_CALIBRATION = (
    "color_intrinsic.json",
    "depth_intrinsic.json",
    "depth_to_color.json",
    "distortion.json",
    "factory_calibration.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate RGB-D files, timestamps, calibration and T_WC poses"
    )
    parser.add_argument("dataset", nargs="?", type=Path)
    parser.add_argument("--datasets-root", type=Path, default=DEFAULT_DATASETS)
    parser.add_argument("--require-pose", action="store_true")
    parser.add_argument("--max-sync-delta-ms", type=float, default=20.0)
    parser.add_argument("--min-valid-depth-ratio", type=float, default=0.01)
    parser.add_argument("--min-depth-mm", type=float, default=500.0)
    parser.add_argument("--max-depth-mm", type=float, default=3860.0)
    parser.add_argument("--no-write-report", action="store_true")
    return parser.parse_args()


def resolve_dataset(dataset: Path | None, datasets_root: Path) -> Path:
    if dataset is not None:
        return dataset.resolve()
    latest = datasets_root.resolve() / "latest_scene.txt"
    if not latest.exists():
        raise FileNotFoundError("No dataset argument and datasets/latest_scene.txt is absent")
    return Path(latest.read_text(encoding="utf-8").strip()).resolve()


def validate_pose(matrix: Any) -> str | None:
    try:
        pose = np.asarray(matrix, dtype=np.float64)
    except (TypeError, ValueError):
        return "T_WC is not numeric"
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        return "T_WC must be a finite 4x4 matrix"
    if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        return "T_WC bottom row is invalid"
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-2):
        return "T_WC rotation is not orthonormal"
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=2e-2):
        return "T_WC rotation determinant is not +1"
    return None


def add_issue(target: list[dict[str, Any]], code: str, message: str, **context: Any) -> None:
    target.append({"code": code, "message": message, **context})


def validate_dataset(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    dataset = resolve_dataset(args.dataset, args.datasets_root)
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {}

    if not dataset.is_dir():
        add_issue(errors, "dataset_missing", f"Dataset directory does not exist: {dataset}")
        return {"dataset": str(dataset), "errors": errors, "warnings": warnings}, 2

    manifest_path = dataset / "manifest.json"
    frames_path = dataset / "frames.jsonl"
    if not manifest_path.exists():
        add_issue(errors, "manifest_missing", "manifest.json is missing")
    if not frames_path.exists():
        add_issue(errors, "frames_missing", "frames.jsonl is missing")
    if errors:
        return {"dataset": str(dataset), "errors": errors, "warnings": warnings}, 2

    manifest = load_json(manifest_path)
    frames = load_jsonl(frames_path)
    metrics["manifest_frame_count"] = int(manifest.get("frame_count", -1))
    metrics["frames_log_count"] = len(frames)
    if manifest.get("schema") != "femto-full-scene-rgbd-v1":
        add_issue(errors, "schema_invalid", "Unexpected or missing dataset schema")
    if metrics["manifest_frame_count"] != len(frames):
        add_issue(errors, "frame_count_mismatch", "manifest frame_count differs from frames.jsonl")
    if not frames:
        add_issue(errors, "empty_dataset", "No captured frame records")

    calibration_dir = dataset / "calibration"
    for name in REQUIRED_CALIBRATION:
        if not (calibration_dir / name).exists():
            add_issue(errors, "calibration_missing", f"Missing calibration/{name}", file=name)
    color_intrinsic_path = calibration_dir / "color_intrinsic.json"
    if color_intrinsic_path.exists():
        intrinsic = load_json(color_intrinsic_path)
        if any(float(intrinsic.get(key, 0.0)) <= 0 for key in ("width", "height", "fx", "fy")):
            add_issue(errors, "intrinsic_invalid", "Aligned color intrinsic has non-positive dimensions/focal length")

    valid_depth_ratios: list[float] = []
    blur_scores: list[float] = []
    sync_deltas_us: list[int] = []
    camera_timestamps: list[int] = []
    valid_pose_count = 0
    expected_shape: tuple[int, int] | None = None

    for ordinal, record in enumerate(frames):
        frame_id = int(record.get("frame", ordinal))
        rgb_path = dataset / str(record.get("rgb", ""))
        depth_path = dataset / str(record.get("depth", ""))
        pose_path = dataset / str(record.get("pose", ""))
        for kind, path in (("rgb", rgb_path), ("depth", depth_path), ("pose", pose_path)):
            if not path.is_file():
                add_issue(errors, "frame_file_missing", f"Missing {kind} file", frame=frame_id, file=str(path))
        if not rgb_path.is_file() or not depth_path.is_file():
            continue

        rgb = imread_unicode_or_none(rgb_path, cv2.IMREAD_COLOR)
        depth = imread_unicode_or_none(depth_path, cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None:
            add_issue(errors, "image_decode_failed", "RGB or depth image cannot be decoded", frame=frame_id)
            continue
        if depth.dtype != np.uint16 or depth.ndim != 2:
            add_issue(errors, "depth_format_invalid", "Depth must be single-channel uint16 millimetres", frame=frame_id)
            continue
        if rgb.shape[:2] != depth.shape:
            add_issue(errors, "rgb_depth_shape_mismatch", "Aligned RGB and depth dimensions differ", frame=frame_id)
        if expected_shape is None:
            expected_shape = depth.shape
        elif depth.shape != expected_shape:
            add_issue(errors, "frame_shape_changed", "Frame dimensions changed within dataset", frame=frame_id)

        valid = (depth >= args.min_depth_mm) & (depth <= args.max_depth_mm)
        ratio = float(np.count_nonzero(valid) / valid.size)
        valid_depth_ratios.append(ratio)
        if ratio < args.min_valid_depth_ratio:
            add_issue(warnings, "low_depth_coverage", "Usable depth coverage is low", frame=frame_id, ratio=ratio)
        grey = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        blur_scores.append(float(cv2.Laplacian(grey, cv2.CV_64F).var()))

        timestamp = int(record.get("depth_timestamp_us", record.get("camera_timestamp_us", -1)))
        camera_timestamps.append(timestamp)
        delta = int(record.get("rgb_depth_timestamp_delta_us", 0))
        sync_deltas_us.append(delta)
        if abs(delta) > args.max_sync_delta_ms * 1000.0:
            add_issue(errors, "rgb_depth_not_synced", "RGB/Depth timestamp delta exceeds limit", frame=frame_id, delta_us=delta)

        if pose_path.is_file():
            pose = load_json(pose_path)
            pose_valid = bool(pose.get("pose_valid", False))
            if pose_valid:
                pose_error = validate_pose(pose.get("T_WC"))
                if pose_error:
                    add_issue(errors, "pose_invalid", pose_error, frame=frame_id)
                else:
                    valid_pose_count += 1

    if len(camera_timestamps) > 1 and any(
        current <= previous for previous, current in zip(camera_timestamps, camera_timestamps[1:])
    ):
        add_issue(errors, "timestamp_not_monotonic", "Depth camera timestamps are not strictly increasing")

    if valid_pose_count != len(frames):
        issue = errors if args.require_pose else warnings
        add_issue(
            issue,
            "pose_incomplete",
            "Not every frame has a validated T_WC; TSDF/3DGS must not run yet",
            valid_pose_count=valid_pose_count,
            frame_count=len(frames),
        )

    metrics.update(
        {
            "valid_pose_count": valid_pose_count,
            "image_height": expected_shape[0] if expected_shape else None,
            "image_width": expected_shape[1] if expected_shape else None,
            "valid_depth_ratio_min": min(valid_depth_ratios, default=None),
            "valid_depth_ratio_mean": float(np.mean(valid_depth_ratios)) if valid_depth_ratios else None,
            "blur_laplacian_min": min(blur_scores, default=None),
            "blur_laplacian_median": float(np.median(blur_scores)) if blur_scores else None,
            "rgb_depth_sync_abs_max_us": max((abs(value) for value in sync_deltas_us), default=None),
        }
    )
    report = {
        "schema": "femto-scene-validation-v1",
        "dataset": str(dataset),
        "ready_for_multiview_reconstruction": not errors and valid_pose_count == len(frames),
        "metrics": metrics,
        "errors": errors,
        "warnings": warnings,
    }
    if not args.no_write_report:
        report_path = dataset / "quality" / "validation_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        report["report_path"] = str(report_path)
    return report, 2 if errors else 0


def main() -> int:
    args = parse_args()
    try:
        report, exit_code = validate_dataset(args)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        report = {"ready_for_multiview_reconstruction": False, "errors": [{"code": "validator_failed", "message": str(error)}]}
        exit_code = 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
