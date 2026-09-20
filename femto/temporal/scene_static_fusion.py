"""Build one robust fixed-camera RGB-D frame without modifying raw capture."""
from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import time
from typing import Any

import cv2
import numpy as np

from femto.geometry.fusion import temporal_median_depth
from femto.io_utils import imread_unicode, imwrite_unicode, load_json, load_jsonl, write_json, write_jsonl


def _select_reference_frame(
    depths: list[np.ndarray],
    masks: list[np.ndarray],
    fused_depth: np.ndarray,
    fused_mask: np.ndarray,
) -> tuple[int, list[dict[str, Any]]]:
    target = fused_mask.astype(bool)
    scores: list[dict[str, Any]] = []
    for index, (depth, mask) in enumerate(zip(depths, masks)):
        valid = mask.astype(bool)
        overlap = target & valid
        union = target | valid
        overlap_ratio = float(np.count_nonzero(overlap) / max(1, np.count_nonzero(union)))
        residual = (
            float(np.median(np.abs(depth[overlap].astype(np.float32) - fused_depth[overlap])))
            if np.count_nonzero(overlap) >= 100
            else 1000.0
        )
        # Coverage mismatch penalizes transient foreground and depth dropouts.
        score = residual + 300.0 * (1.0 - overlap_ratio)
        scores.append(
            {
                "frame_ordinal": index,
                "median_depth_residual_mm": residual,
                "overlap_ratio": overlap_ratio,
                "selection_score": score,
            }
        )
    selected = min(scores, key=lambda item: item["selection_score"])["frame_ordinal"]
    return int(selected), scores


def build_static_median_dataset(
    source: Path,
    output: Path,
    *,
    minimum_support_ratio: float = 0.30,
) -> tuple[Path, dict[str, Any]]:
    """Create a one-frame fixed-camera derived dataset using temporal depth median."""
    source = source.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists; refusing to overwrite: {output}")
    if not 0.1 <= minimum_support_ratio <= 1.0:
        raise ValueError("minimum_support_ratio must be in [0.1, 1.0]")
    manifest = load_json(source / "manifest.json")
    frames = load_jsonl(source / "frames.jsonl")
    if manifest.get("schema") != "femto-full-scene-rgbd-v1" or not frames:
        raise ValueError("Input is not a non-empty femto-full-scene-rgbd-v1 dataset")

    parameters = manifest.get("parameters", {})
    range_m = manifest.get("camera", {}).get("depth_range_m")
    minimum_depth_mm = float(
        parameters.get("depth_min_mm", 1000.0 * range_m[0] if range_m else 500.0)
    )
    maximum_depth_mm = float(
        parameters.get("depth_max_mm", 1000.0 * range_m[1] if range_m else 3860.0)
    )
    depths: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    expected_shape: tuple[int, int] | None = None
    for frame in frames:
        depth = imread_unicode(source / frame["depth"], cv2.IMREAD_UNCHANGED)
        if depth.dtype != np.uint16 or depth.ndim != 2:
            raise ValueError(f"Frame {frame['frame']} depth is not uint16 millimetres")
        if expected_shape is None:
            expected_shape = depth.shape
        elif depth.shape != expected_shape:
            raise ValueError("Depth frame dimensions changed within the sequence")
        valid = (depth >= minimum_depth_mm) & (depth <= maximum_depth_mm)
        depths.append(depth)
        masks.append(valid.astype(np.uint8))

    minimum_support = max(2, int(math.ceil(len(frames) * minimum_support_ratio)))
    fused_depth, fused_mask, support = temporal_median_depth(
        depths,
        masks,
        minimum_support=minimum_support,
        chunk_rows=48,
    )
    if np.count_nonzero(fused_mask) < 1000:
        raise ValueError(
            f"Fixed-camera temporal median retained too little depth; keep the scene within {minimum_depth_mm/1000:.2f}-{maximum_depth_mm/1000:.2f} m"
        )
    selected, scores = _select_reference_frame(depths, masks, fused_depth, fused_mask)
    selected_frame = frames[selected]
    color = imread_unicode(source / selected_frame["rgb"], cv2.IMREAD_COLOR)
    if color.shape[:2] != fused_depth.shape:
        raise ValueError("Selected RGB and fused depth dimensions differ")

    for child in ("rgb", "depth", "pose"):
        (output / child).mkdir(parents=True, exist_ok=False)
    shutil.copytree(source / "calibration", output / "calibration")
    imwrite_unicode(output / "rgb" / "000000.jpg", color, [cv2.IMWRITE_JPEG_QUALITY, 96])
    imwrite_unicode(
        output / "depth" / "000000.png",
        np.rint(np.clip(fused_depth, 0, 65535)).astype(np.uint16),
    )
    imwrite_unicode(output / "temporal_support.png", support)
    write_json(
        output / "pose" / "000000.json",
        {
            "schema": "femto-scene-frame-pose-v1",
            "frame": 0,
            "pose_valid": True,
            "T_WC": np.eye(4).tolist(),
            "pose_source": "fixed_camera_identity_after_temporal_median",
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
        "status": "derived_fixed_camera_temporal_median",
        "frame_count": 1,
        "source_dataset": str(source),
        "pose": {
            "valid": True,
            "source": "fixed_camera_identity_after_temporal_median",
            "warning": "Single camera viewpoint; unobserved sides remain unknown.",
        },
        "parameters": {
            **dict(manifest.get("parameters", {})),
            "fixed_camera": True,
            "source_frame_count": len(frames),
            "minimum_support_ratio": minimum_support_ratio,
            "minimum_support_frames": minimum_support,
            "selected_color_source_frame": int(selected_frame["frame"]),
            "depth_min_mm": minimum_depth_mm,
            "depth_max_mm": maximum_depth_mm,
        },
        "files": {
            "rgb": "rgb/000000.jpg",
            "depth": "depth/000000.png",
            "pose": "pose/000000.json",
            "frames": "frames.jsonl",
            "calibration": "calibration/*.json",
            "temporal_support": "temporal_support.png",
        },
    }
    write_json(output / "manifest.json", derived_manifest)
    report = {
        "schema": "femto-fixed-camera-median-report-v1",
        "created_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "source_dataset": str(source),
        "output_dataset": str(output),
        "source_frame_count": len(frames),
        "minimum_support_frames": minimum_support,
        "selected_color_source_frame": int(selected_frame["frame"]),
        "retained_depth_pixels": int(np.count_nonzero(fused_mask)),
        "retained_depth_ratio": float(np.count_nonzero(fused_mask) / fused_mask.size),
        "frame_selection_metrics": scores,
        "limitations": [
            "Only the fixed camera viewpoint is reconstructed.",
            "Moving foreground present in enough source frames can remain in the temporal median.",
            "Large unseen surfaces are not filled.",
        ],
    }
    write_json(output / "static_fusion_report.json", report)
    marker = source / "derived" / "latest_fixed_dataset.txt"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(str(output), encoding="utf-8")
    return output, report
