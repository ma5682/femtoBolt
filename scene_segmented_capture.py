"""Full-profile RGB-D capture and fusion for a stationary Femto Bolt."""
from __future__ import annotations

import math
from pathlib import Path
import time
from typing import Any, Callable

import cv2
import numpy as np
import open3d as o3d

from fusion import temporal_median_depth, write_point_cloud_safe
from orbbec_rgbd import AlignedFrame, DEPTH_MODE_SPECS, OrbbecAlignedCamera
from scene_capture import append_jsonl, imwrite_unicode, json_dump, write_calibration
from scene_pose_estimator import assess_rgb_motion_pair


SEGMENTED_DEPTH_MODES = ("wide", "near", "far")
SEGMENTED_FRAMES_PER_MODE = 5
MINIMUM_SUPPORT_RATIO = 0.60
OVERLAP_DISTANCE_M = 0.006
MERGE_PRIORITY = ("near", "wide", "far")
TUNED_DEPTH_MODES = ("wide_detail", "wide", "near", "far")
TUNED_FRAMES_PER_MODE = 10
TUNED_FPS = 15
TUNED_WARMUP_FRAMES = 15
HARDWARE_SPEC_URL = (
    "https://doc.orbbec.com/documentation/Orbbec%20Femto%20Bolt%20Documentation/"
    "Femto%20Bolt%20Hardware%20Specifications"
)


def common_rgb_view(frame: AlignedFrame, camera: OrbbecAlignedCamera) -> np.ndarray:
    intrinsic = camera.intrinsics
    matrix = np.array([
        [intrinsic.fx, 0, intrinsic.cx], [0, intrinsic.fy, intrinsic.cy], [0, 0, 1],
    ], dtype=np.float64)
    distortion = camera.calibration["rgb_distortion"]
    coefficients = np.array([
        distortion[key] for key in ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")
    ])
    horizontal, vertical = np.meshgrid(np.linspace(-.3, .3, 480), np.linspace(-.2, .2, 320))
    rays = np.column_stack((horizontal.ravel(), vertical.ravel(), np.ones(horizontal.size)))
    pixels, _ = cv2.projectPoints(rays, np.zeros(3), np.zeros(3), matrix, coefficients)
    pixels = pixels.reshape(320, 480, 2).astype(np.float32)
    if (pixels[:, :, 0].min() < 0 or pixels[:, :, 1].min() < 0
            or pixels[:, :, 0].max() >= frame.color_bgr.shape[1]
            or pixels[:, :, 1].max() >= frame.color_bgr.shape[0]):
        raise ValueError("Calibration does not cover the common RGB motion-check view")
    gray = cv2.cvtColor(frame.color_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.remap(gray, pixels[:, :, 0], pixels[:, :, 1], cv2.INTER_LINEAR)


def audit_static_views(views: list[np.ndarray], previous: np.ndarray | None = None) -> dict:
    pairs = [(index - 1, index) for index in range(1, len(views))]
    pairs.append((0, len(views) - 1))
    details = []
    if previous is not None:
        details.append({"pair": "previous_mode_to_current", **assess_rgb_motion_pair(previous, views[0], scale=.5)})
    for source, target in pairs:
        details.append({"pair": [source, target], **assess_rgb_motion_pair(views[source], views[target], scale=.5)})
    passed = all(item.get("reliable") and item.get("appears_fixed") for item in details)
    return {"passed": passed, "pairs": details, "scope": "common RGB background motion; not external pose or proof every object is static"}


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


def save_segment(
    output: Path, mode: str, frames: list[AlignedFrame], camera: OrbbecAlignedCamera,
) -> tuple[o3d.geometry.PointCloud, dict]:
    minimum_mm = camera.min_depth_mm
    maximum_mm = camera.max_depth_mm
    masks = [
        np.isfinite(frame.depth_mm)
        & (frame.depth_mm >= minimum_mm)
        & (frame.depth_mm <= maximum_mm)
        for frame in frames
    ]
    minimum_support = math.ceil(len(frames) * MINIMUM_SUPPORT_RATIO)
    median_depth, _, support = temporal_median_depth(
        [frame.depth_mm for frame in frames], masks, minimum_support=minimum_support,
    )
    color = np.zeros_like(frames[0].color_bgr)
    best_distance = np.full(median_depth.shape, np.inf, dtype=np.float32)
    for frame, mask in zip(frames, masks):
        distance = np.abs(frame.depth_mm - median_depth)
        closer = mask & (distance < best_distance)
        color[closer] = frame.color_bgr[closer]
        best_distance[closer] = distance[closer]
    derived = output / "segments" / mode / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    np.save(derived / "median_depth_mm.npy", median_depth)
    imwrite_unicode(derived / "support_count.png", support)
    imwrite_unicode(derived / "color.png", color)
    cloud = depth_cloud(
        median_depth, color, camera.intrinsics, camera.calibration["rgb_distortion"],
        minimum_mm, maximum_mm,
    )
    cloud_path = derived / "point_cloud_m.ply"
    if not cloud.has_points():
        raise RuntimeError(f"{mode}: no depth pixels supported by {minimum_support} frames")
    if not write_point_cloud_safe(cloud_path, cloud):
        raise RuntimeError(f"Failed to save {cloud_path}")
    report = {
        "mode": mode,
        "depth_profile": {**DEPTH_MODE_SPECS[mode], "actual_profile": camera.metadata.get("depth_profile")},
        "color_profile": camera.metadata.get("color_profile"),
        "depth_range_mm": [minimum_mm, maximum_mm],
        "frame_count": len(frames),
        "minimum_support_frames": minimum_support,
        "point_count": len(cloud.points),
        "points_per_raw_frame": [int(np.count_nonzero(mask)) for mask in masks],
        "raw_frames": f"segments/{mode}/raw/frames.jsonl",
        "calibration": f"segments/{mode}/calibration",
        "cloud": str(cloud_path.relative_to(output)).replace("\\", "/"),
        "bounds_m": [cloud.get_min_bound().tolist(), cloud.get_max_bound().tolist()],
        "coverage": {
            "aligned_pixels": int(support.size),
            "no_return_all_frames": int(np.count_nonzero(~np.any(
                [np.isfinite(frame.depth_mm) & (frame.depth_mm > 0) for frame in frames], axis=0,
            ))),
            "below_minimum_observations": int(sum(np.count_nonzero(
                np.isfinite(frame.depth_mm) & (frame.depth_mm > 0) & (frame.depth_mm < minimum_mm),
            ) for frame in frames)),
            "above_maximum_observations": int(sum(np.count_nonzero(
                np.isfinite(frame.depth_mm) & (frame.depth_mm > maximum_mm),
            ) for frame in frames)),
            "weak_support_pixels": int(np.count_nonzero((support > 0) & (support < minimum_support))),
            "stable_pixels": int(np.count_nonzero(support >= minimum_support)),
            "native_valid_pixels_per_frame": [
                int(np.count_nonzero(
                    np.isfinite(frame.native_depth_mm)
                    & (frame.native_depth_mm >= minimum_mm)
                    & (frame.native_depth_mm <= maximum_mm)
                )) if frame.native_depth_mm is not None else None for frame in frames
            ],
        },
    }
    json_dump(output / "segments" / mode / "manifest.json", report)
    return cloud, report


def merge_segments(
    output: Path, clouds: list[o3d.geometry.PointCloud], reports: list[dict],
) -> dict:
    modes = tuple(report["mode"] for report in reports)
    if modes not in (SEGMENTED_DEPTH_MODES, TUNED_DEPTH_MODES) or len(clouds) != len(reports):
        raise ValueError("Every requested depth mode must complete before fusion")
    priority = ("near", "wide_detail", "wide", "far") if "wide_detail" in modes else MERGE_PRIORITY
    combined = o3d.geometry.PointCloud()
    for cloud in clouds:
        combined += cloud
    if not combined.has_points():
        raise RuntimeError("Three-mode capture produced no usable points")
    full_path = output / "full_range_unmerged_point_cloud_m.ply"
    if not write_point_cloud_safe(full_path, combined):
        raise RuntimeError(f"Failed to save {full_path}")
    mode_clouds = {report["mode"]: cloud for report, cloud in zip(reports, clouds)}
    merged = o3d.geometry.PointCloud()
    selected_counts = {}
    for mode in priority:
        candidate = mode_clouds[mode]
        if merged.has_points():
            distances = np.asarray(candidate.compute_point_cloud_distance(merged))
            candidate = candidate.select_by_index(np.flatnonzero(distances > OVERLAP_DISTANCE_M).tolist())
        selected_counts[mode] = len(candidate.points)
        merged += candidate
    merged_path = output / "full_range_point_cloud_m.ply"
    if not write_point_cloud_safe(merged_path, merged):
        raise RuntimeError(f"Failed to save {merged_path}")
    report = {
        "schema": "femto-static-segmented-depth-cloud-v3" if "wide_detail" in modes else "femto-static-segmented-depth-cloud-v2",
        "capture_modes": list(modes),
        "status": "complete",
        "created_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "hardware_specification": HARDWARE_SPEC_URL,
        "scene_requirement": "camera and scene remain static across all mode switches",
        "coordinate_system": "color_camera: +X right, +Y down, +Z forward; metres",
        "depth_unit": "millimetres; raw depth preserved as float32 NPY",
        "depth_range_m": [0.25, 5.46],
        "segments": reports,
        "points_per_mode": {item["mode"]: item["point_count"] for item in reports},
        "points_before_merge": len(combined.points),
        "point_count": len(merged.points),
        "bounds_m": [merged.get_min_bound().tolist(), merged.get_max_bound().tolist()],
        "filters": {
            "range": "each mode's entire official operating range, inclusive",
            "temporal_minimum_support_ratio": MINIMUM_SUPPORT_RATIO,
            "voxel_m": None,
            "exclusive_distance_bands": False,
            "spatial_stride": 1,
        },
        "overlap_selection": {
            "priority": list(priority),
            "distance_m": OVERLAP_DISTANCE_M,
            "selected_points_per_mode": selected_counts,
            "policy": "Keep higher-resolution observations unchanged; other modes fill spatial gaps",
            "coordinate_or_color_averaging": False,
        },
        "projection": "per-mode RGB K6 undistortion and aligned-depth backprojection",
        "unmerged_file": full_path.name,
        "file": merged_path.name,
        "limitation": "Only observed RGB-aligned surfaces; no unseen surfaces or inferred geometry",
    }
    json_dump(output / "manifest.json", report)
    (output.parent / "latest_full_range_cloud.txt").write_text(str(output), encoding="utf-8")
    return report


def capture_three_modes(
    output: Path, progress: Callable[[int, str], None] | None = None,
    *, tuned: bool = False,
) -> dict:
    modes = TUNED_DEPTH_MODES if tuned else SEGMENTED_DEPTH_MODES
    frame_count = TUNED_FRAMES_PER_MODE if tuned else SEGMENTED_FRAMES_PER_MODE
    output.mkdir(parents=True, exist_ok=True)
    json_dump(output / "manifest.json", {"status": "capturing", "segments": []})
    clouds, reports = [], []
    skipped_modes = []
    previous_view = None
    try:
        for index, mode in enumerate(modes, start=1):
            if progress is not None:
                progress(index, mode)
            segment = output / "segments" / mode
            camera = None
            frames = []
            for attempt in range(1, 4):
                options = {"fps": TUNED_FPS, "retain_native_depth": True} if tuned else {}
                color_size = (1280, 960) if mode in ("near", "far") else ((1920, 1080) if tuned else None)
                camera = OrbbecAlignedCamera(mode, color_size=color_size, **options)
                frames = []
                try:
                    camera.start()
                    deadline = time.monotonic() + 20.0
                    seen = set()
                    warmed_frames = 0
                    while len(frames) < frame_count and time.monotonic() < deadline:
                        frame = camera.read(1000)
                        if frame is None:
                            time.sleep(0.02)
                            continue
                        identity = (frame.frame_index, frame.depth_timestamp_us)
                        if identity in seen:
                            continue
                        seen.add(identity)
                        if tuned and warmed_frames < TUNED_WARMUP_FRAMES:
                            warmed_frames += 1
                            continue
                        frames.append(frame)
                    if len(frames) != frame_count:
                        raise RuntimeError(f"{mode}: only {len(frames)}/{frame_count} aligned RGB-D frames")
                    break
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(0.5)
                finally:
                    camera.stop()
            write_calibration(segment, camera, camera.intrinsics)
            for frame_number, frame in enumerate(frames, start=1):
                stem = f"frame_{frame_number:03d}"
                raw = segment / "raw"
                raw.mkdir(parents=True, exist_ok=True)
                imwrite_unicode(raw / f"{stem}_color.png", frame.color_bgr)
                np.save(raw / f"{stem}_depth_mm.npy", frame.depth_mm)
                if frame.native_depth_mm is not None:
                    np.save(raw / f"{stem}_native_depth_mm.npy", frame.native_depth_mm)
                imwrite_unicode(
                    raw / f"{stem}_valid.png",
                    (np.isfinite(frame.depth_mm) & (frame.depth_mm > 0)).astype(np.uint8) * 255,
                )
                append_jsonl(raw / "frames.jsonl", {
                    "frame_index": frame.frame_index,
                    "color": f"{stem}_color.png",
                    "depth": f"{stem}_depth_mm.npy",
                    "native_depth": f"{stem}_native_depth_mm.npy" if frame.native_depth_mm is not None else None,
                    "valid_mask": f"{stem}_valid.png",
                    "color_timestamp_us": frame.color_timestamp_us,
                    "depth_timestamp_us": frame.depth_timestamp_us,
                    "color_system_timestamp_us": frame.color_system_timestamp_us,
                    "depth_system_timestamp_us": frame.depth_system_timestamp_us,
                    "frame_metadata": frame.frame_metadata,
                })
            if tuned:
                views = [common_rgb_view(frame, camera) for frame in frames]
                static_report = audit_static_views(views, previous_view)
                json_dump(segment / "static_check.json", static_report)
                if not static_report["passed"]:
                    raise RuntimeError(
                        f"{mode}: 场景/相机发生移动或无法确认静止，已保留原始帧，停止融合以避免重影。"
                        "请固定相机、保持场景静止后重试。"
                    )
                previous_view = views[-1]
                if mode == "wide_detail":
                    support = np.sum([
                        np.isfinite(frame.depth_mm)
                        & (frame.depth_mm >= camera.min_depth_mm)
                        & (frame.depth_mm <= camera.max_depth_mm) for frame in frames
                    ], axis=0)
                    if not np.any(support >= math.ceil(frame_count * MINIMUM_SUPPORT_RATIO)):
                        skipped = {
                            "mode": mode, "status": "skipped_no_stable_aligned_depth",
                            "reason": "Optional wide-detail enhancement has no stable aligned depth; original three modes remain required",
                            "raw_evidence": f"segments/{mode}/raw",
                            "static_check_passed": True,
                        }
                        json_dump(segment / "manifest.json", skipped)
                        skipped_modes.append(skipped)
                        continue
            cloud, report = save_segment(output, mode, frames, camera)
            report["depth_profile"]["fps"] = TUNED_FPS if tuned else DEPTH_MODE_SPECS[mode]["fps"]
            report["capture_settings"] = {
                "fps": TUNED_FPS if tuned else DEPTH_MODE_SPECS[mode]["fps"],
                "warmup_complete_frames": TUNED_WARMUP_FRAMES if tuned else 0,
                "frames_saved": frame_count,
                "native_depth_retained": tuned,
                "depth_exposure": "factory mode timing; no manual exposure or laser override",
            }
            if tuned:
                report["static_check"] = {"file": f"segments/{mode}/static_check.json", "passed": True}
            json_dump(segment / "manifest.json", report)
            clouds.append(cloud)
            reports.append(report)
            json_dump(output / "manifest.json", {"status": "capturing", "segments": reports})
        report = merge_segments(output, clouds, reports)
        report["capture_preset"] = "official_detail_coverage" if tuned else "legacy_three_modes"
        report["requested_modes"] = list(modes)
        report["skipped_modes"] = skipped_modes
        report["enhancement_applied"] = "wide_detail" in report["capture_modes"]
        report["stationary_background_checks_passed"] = True if tuned else None
        json_dump(output / "manifest.json", report)
        return report
    except Exception as error:
        json_dump(output / "manifest.json", {
            "status": "failed", "segments": reports, "error": str(error),
        })
        raise
