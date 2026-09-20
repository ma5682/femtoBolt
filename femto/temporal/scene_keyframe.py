"""Select one coherent fixed-camera RGB-D keyframe without modifying raw capture."""
from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import time
from typing import Any

import cv2
import numpy as np

from femto.io_utils import imread_unicode, load_json, load_jsonl, write_json, write_jsonl


def _depth_range(manifest: dict[str, Any]) -> tuple[float, float]:
    parameters = manifest.get("parameters", {})
    camera = manifest.get("camera", {})
    range_m = camera.get("depth_range_m")
    minimum = float(parameters.get("depth_min_mm", 1000.0 * range_m[0] if range_m else 500.0))
    maximum = float(parameters.get("depth_max_mm", 1000.0 * range_m[1] if range_m else 3860.0))
    return minimum, maximum


def _measure_frames(source: Path, frames: list[dict[str, Any]], minimum_mm: float,
                    maximum_mm: float) -> list[dict[str, Any]]:
    raw_metrics: list[dict[str, Any]] = []
    expected_shape: tuple[int, int] | None = None
    for ordinal, frame in enumerate(frames):
        depth = imread_unicode(source / frame["depth"], cv2.IMREAD_UNCHANGED)
        color = imread_unicode(source / frame["rgb"], cv2.IMREAD_COLOR)
        if depth.dtype != np.uint16 or depth.ndim != 2 or color.shape[:2] != depth.shape:
            raise ValueError(f"Frame {frame['frame']} is not aligned uint16 RGB-D")
        if expected_shape is None:
            expected_shape = depth.shape
        elif depth.shape != expected_shape:
            raise ValueError("Frame dimensions changed within the sequence")
        positive = depth > 0
        valid = positive & (depth >= minimum_mm) & (depth <= maximum_mm)
        too_near = positive & (depth < minimum_mm)
        too_far = positive & (depth > maximum_mm)
        valid_values = depth[valid]
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        raw_metrics.append(
            {
                "frame_ordinal": ordinal,
                "source_frame": int(frame["frame"]),
                "valid_depth_pixels": int(valid_values.size),
                "valid_depth_ratio": float(valid.mean()),
                "too_near_depth_ratio": float(too_near.mean()),
                "too_far_depth_ratio": float(too_far.mean()),
                "no_depth_return_ratio": float((~positive).mean()),
                "median_depth_mm": float(np.median(valid_values)) if valid_values.size else math.nan,
                "rgb_blur_variance": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                "rgb_depth_delta_us": int(frame.get("rgb_depth_timestamp_delta_us", 0)),
            }
        )
    return raw_metrics


def build_fixed_keyframe_dataset(
    source: Path,
    output: Path,
) -> tuple[Path, dict[str, Any]]:
    """Create a one-frame dataset whose RGB and depth come from the same best frame."""
    source = source.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists; refusing to overwrite: {output}")
    manifest = load_json(source / "manifest.json")
    frames = load_jsonl(source / "frames.jsonl")
    if manifest.get("schema") != "femto-full-scene-rgbd-v1" or not frames:
        raise ValueError("Input is not a non-empty femto-full-scene-rgbd-v1 dataset")
    minimum_mm, maximum_mm = _depth_range(manifest)

    raw_metrics = _measure_frames(source, frames, minimum_mm, maximum_mm)

    useful = [item for item in raw_metrics if item["valid_depth_pixels"] >= 1000]
    if not useful:
        raise ValueError("No frame contains enough valid depth for keyframe reconstruction")
    coverage_reference = max(1e-9, float(np.quantile([x["valid_depth_ratio"] for x in useful], 0.75)))
    blur_reference = max(1e-9, float(np.quantile([x["rgb_blur_variance"] for x in useful], 0.75)))
    depth_reference = float(np.median([x["median_depth_mm"] for x in useful]))
    depth_scale = max(300.0, depth_reference * 0.20)
    for item in raw_metrics:
        coverage = min(item["valid_depth_ratio"] / coverage_reference, 1.10)
        sharpness = min(item["rgb_blur_variance"] / blur_reference, 1.25)
        depth_consistency = (
            math.exp(-abs(item["median_depth_mm"] - depth_reference) / depth_scale)
            if math.isfinite(item["median_depth_mm"])
            else 0.0
        )
        sync = math.exp(-abs(item["rgb_depth_delta_us"]) / 20_000.0)
        item["selection_score"] = float(
            0.55 * coverage + 0.20 * sharpness + 0.20 * depth_consistency + 0.05 * sync
        )
    selected_metric = max(raw_metrics, key=lambda item: item["selection_score"])
    selected = int(selected_metric["frame_ordinal"])
    selected_frame = frames[selected]

    for child in ("rgb", "depth", "pose"):
        (output / child).mkdir(parents=True, exist_ok=False)
    shutil.copytree(source / "calibration", output / "calibration")
    shutil.copy2(source / selected_frame["rgb"], output / "rgb" / "000000.jpg")
    shutil.copy2(source / selected_frame["depth"], output / "depth" / "000000.png")
    write_json(
        output / "pose" / "000000.json",
        {
            "schema": "femto-scene-frame-pose-v1",
            "frame": 0,
            "pose_valid": True,
            "T_WC": np.eye(4).tolist(),
            "pose_source": "single_fixed_camera_keyframe_identity",
            "coordinate_system": "+X right, +Y down, +Z forward; metres",
            "source_dataset": str(source),
            "source_frame": int(selected_frame["frame"]),
        },
    )
    derived_frame = {
        **selected_frame,
        "frame": 0,
        "source_frame": int(selected_frame["frame"]),
        "rgb": "rgb/000000.jpg",
        "depth": "depth/000000.png",
        "pose": "pose/000000.json",
        "pose_valid": True,
    }
    write_jsonl(output / "frames.jsonl", [derived_frame])
    derived_manifest = {
        **manifest,
        "created_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "status": "derived_fixed_camera_keyframe",
        "frame_count": 1,
        "source_dataset": str(source),
        "pose": {
            "valid": True,
            "source": "single_fixed_camera_keyframe_identity",
            "warning": "Single camera viewpoint; unobserved surfaces remain unknown.",
        },
        "parameters": {
            **dict(manifest.get("parameters", {})),
            "fixed_camera": True,
            "processing_mode": "keyframe",
            "source_frame_count": len(frames),
            "selected_source_frame": int(selected_frame["frame"]),
            "depth_min_mm": minimum_mm,
            "depth_max_mm": maximum_mm,
        },
        "files": {
            "rgb": "rgb/000000.jpg",
            "depth": "depth/000000.png",
            "pose": "pose/000000.json",
            "frames": "frames.jsonl",
            "calibration": "calibration/*.json",
        },
    }
    write_json(output / "manifest.json", derived_manifest)
    report = {
        "schema": "femto-fixed-camera-keyframe-report-v1",
        "created_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "source_dataset": str(source),
        "output_dataset": str(output),
        "source_frame_count": len(frames),
        "selected_source_frame": int(selected_frame["frame"]),
        "selected_valid_depth_ratio": selected_metric["valid_depth_ratio"],
        "selected_too_near_depth_ratio": selected_metric["too_near_depth_ratio"],
        "selected_too_far_depth_ratio": selected_metric["too_far_depth_ratio"],
        "selected_no_depth_return_ratio": selected_metric["no_depth_return_ratio"],
        "depth_range_mm": [minimum_mm, maximum_mm],
        "low_depth_coverage_warning": selected_metric["valid_depth_ratio"] < 0.35,
        "frame_selection_metrics": raw_metrics,
        "limitations": [
            "Only measured pixels from one coherent RGB-D frame are reconstructed.",
            "RGB pixels without valid depth cannot become geometry.",
            "Only the fixed camera viewpoint is available.",
        ],
    }
    write_json(output / "keyframe_report.json", report)
    marker = source / "derived" / "latest_keyframe_dataset.txt"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(str(output), encoding="utf-8")
    return output, report
