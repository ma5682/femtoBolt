"""Capture-session lifecycle shared by the Qt application and local worker.

Frame and photo writers, the ``ActiveScan`` record and the sticky camera-motion
hint state live here so that ``femto.capture`` never imports the application.
"""
from __future__ import annotations

from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
import importlib.metadata
from pathlib import Path
import time
from typing import Any, Callable

import cv2
import numpy as np

from femto.dataset.calibration_files import write_calibration
from femto.geometry.depth_cloud import depth_cloud
from femto.geometry.fusion import write_point_cloud_safe
from femto.hardware.orbbec_rgbd import AlignedFrame, OrbbecAlignedCamera
from femto.io_utils import append_jsonl, imwrite_unicode, write_json
from femto.capture.scene_capture import depth_preview
from femto.paths import PHOTOS_ROOT


MOTION_CHECK_INTERVAL_S = 1.0
MOTION_CHECK_SCALE = 0.25
MOTION_FINAL_WAIT_S = 1.0
ALIGNED_FRAME_STARTUP_TIMEOUT_S = 15.0

def unique_timestamp_dir(root: Path, prefix: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    base = root / f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}"
    candidate = base
    suffix = 1
    while candidate.exists():
        candidate = root / f"{base.name}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate

def copy_frame(frame: AlignedFrame, *, include_native_depth: bool = True) -> AlignedFrame:
    return AlignedFrame(
        color_bgr=frame.color_bgr.copy(),
        depth_mm=frame.depth_mm.copy(),
        timestamp_us=frame.timestamp_us,
        system_timestamp_us=frame.system_timestamp_us,
        frame_index=frame.frame_index,
        color_timestamp_us=frame.color_timestamp_us,
        color_system_timestamp_us=frame.color_system_timestamp_us,
        depth_timestamp_us=frame.depth_timestamp_us,
        depth_system_timestamp_us=frame.depth_system_timestamp_us,
        frame_metadata={stream: dict(values) for stream, values in frame.frame_metadata.items()},
        native_depth_mm=None if frame.native_depth_mm is None or not include_native_depth else frame.native_depth_mm.copy(),
    )

def save_rgbd_photo(
    frame: AlignedFrame,
    camera: OrbbecAlignedCamera,
    output_root: Path = PHOTOS_ROOT,
) -> Path:
    """Save one lossless depth snapshot plus color, calibration and PLY."""
    if camera.intrinsics is None:
        raise RuntimeError("Camera intrinsics are unavailable")
    output = unique_timestamp_dir(output_root, "femto_photo")
    color_path = output / "color.jpg"
    depth_path = output / "depth_mm.png"
    preview_path = output / "depth_preview.jpg"
    point_path = output / "point_cloud_m.ply"
    imwrite_unicode(color_path, frame.color_bgr, [cv2.IMWRITE_JPEG_QUALITY, 96])
    depth_u16 = np.rint(np.clip(frame.depth_mm, 0, 65535)).astype(np.uint16)
    imwrite_unicode(depth_path, depth_u16)
    imwrite_unicode(
        preview_path,
        depth_preview(
            np.where(frame.depth_mm > 0, frame.depth_mm, np.nan),
            camera.min_depth_mm,
            camera.max_depth_mm,
        ),
        [cv2.IMWRITE_JPEG_QUALITY, 94],
    )
    write_calibration(output, camera, camera.intrinsics)

    from femto.depth.depth_quality import process_depth_quality
    filled_depth, depth_confidence, quality_report, quality_paths = process_depth_quality(
        depth_u16, camera.min_depth_mm, camera.max_depth_mm,
        output / "derived" / "depth_quality", rgb_bgr=frame.color_bgr,
    )

    valid = (
        np.isfinite(frame.depth_mm)
        & (frame.depth_mm >= camera.min_depth_mm)
        & (frame.depth_mm <= camera.max_depth_mm)
    )
    cloud = depth_cloud(
        frame.depth_mm,
        frame.color_bgr,
        camera.intrinsics,
        camera.calibration["rgb_distortion"],
        camera.min_depth_mm,
        camera.max_depth_mm,
    )
    if not write_point_cloud_safe(point_path, cloud):
        raise RuntimeError("Open3D failed to save the single-frame point cloud")
    from femto.geometry.depth_confidence_cloud import build_confident_depth_cloud, write_confident_point_cloud
    derived_cloud, point_confidence = build_confident_depth_cloud(
        depth_u16, filled_depth, depth_confidence, frame.color_bgr, camera.intrinsics,
        camera.calibration["rgb_distortion"], camera.min_depth_mm, camera.max_depth_mm,
    )
    derived_point_path = output / "derived" / "point_cloud_hole_filled_m.ply"
    write_confident_point_cloud(derived_point_path, derived_cloud, point_confidence)
    manifest = {
        "schema": "femto-rgbd-photo-v1",
        "depth_quality": {
            "original_invalid_rate": quality_report["original_invalid_rate"],
            "filled_invalid_rate": quality_report["filled_invalid_rate"],
            "repaired_pixels": quality_report["repaired_pixels"],
            "policy": quality_report["policy"],
            "files": {key: str(Path(value).relative_to(output.resolve())) for key, value in quality_paths.items()},
            "derived_point_cloud": str(derived_point_path.relative_to(output)),
        },
        "created_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "device": dict(camera.metadata),
        "alignment": "depth_to_color",
        "coordinate_system": "+X right, +Y down, +Z forward",
        "depth_unit": "uint16 millimetres; zero is unknown",
        "pose_valid": False,
        "pose_warning": "A single photo has no external T_WC and is not a multi-view reconstruction.",
        "timestamps_us": {
            "color_camera": frame.color_timestamp_us,
            "depth_camera": frame.depth_timestamp_us,
            "color_system": frame.color_system_timestamp_us,
            "depth_system": frame.depth_system_timestamp_us,
            "rgb_depth_delta": frame.color_timestamp_us - frame.depth_timestamp_us,
        },
        "frame_metadata": frame.frame_metadata,
        "image": {
            "width": int(frame.color_bgr.shape[1]),
            "height": int(frame.color_bgr.shape[0]),
            "depth_range_m": [camera.min_depth_mm / 1000.0, camera.max_depth_mm / 1000.0],
            "valid_depth_ratio_in_active_range": float(np.count_nonzero(valid) / valid.size),
        },
        "files": {
            "color": color_path.name,
            "depth": depth_path.name,
            "depth_preview": preview_path.name,
            "point_cloud": point_path.name,
            "calibration": "calibration/*.json",
        },
    }
    write_json(output / "manifest.json", manifest)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "latest_photo.txt").write_text(str(output), encoding="utf-8")
    return output

@dataclass
class ActiveScan:
    session: Path
    frames_log: Path
    started: float
    next_sample: float
    duration_s: float
    fps: float
    max_frames: int
    fixed_camera: bool
    processing_mode: str
    depth_min_mm: float
    depth_max_mm: float
    frame_count: int = 0
    reconstruct_after: bool = True
    last_preview: np.ndarray | None = None
    motion_reference_bgr: np.ndarray | None = None
    next_motion_check: float = 0.0
    # Operator feedback starts at "not moved": this means that no reliable
    # movement evidence has been observed yet, not that a fixed-camera Pose
    # has been established.  Only a reliable moving assessment can promote it
    # to the sticky "moved" state.
    motion_state: str = "not_moved"
    motion_report: dict[str, Any] | None = None
    motion_future: Future[dict[str, Any]] | None = None
    capture_quality: dict[str, Any] | None = None
    quality_history: list[dict[str, Any]] = field(default_factory=list)

def update_capture_motion_state(previous: str, report: dict[str, Any]) -> str:
    """Fold one optical-flow hint into a sticky moving-capture state.

    The result is operator feedback only.  It is never written as T_WC and it
    cannot satisfy any reconstruction pose gate.
    """
    if previous == "moved":
        return previous
    if not report.get("reliable", False):
        return previous
    appears_fixed = report.get("appears_fixed")
    if appears_fixed is True:
        return "not_moved"
    if appears_fixed is False:
        return "moved"
    return previous

def consume_capture_motion_future(
    previous_state: str,
    previous_report: dict[str, Any] | None,
    future: Future[dict[str, Any]] | None,
    *,
    wait_timeout_s: float | None = None,
) -> tuple[str, dict[str, Any] | None, bool]:
    """Consume a completed background-motion task without blocking capture.

    A non-``None`` timeout is used only after capture has stopped, when a short
    bounded wait cannot reduce the saved-frame rate.  Worker failures remain an
    auditable unreliable hint and never stop RGB-D capture or become a pose.
    """
    if future is None:
        return previous_state, previous_report, False
    if wait_timeout_s is None and not future.done():
        return previous_state, previous_report, False
    try:
        report = future.result(timeout=wait_timeout_s)
    except FutureTimeoutError:
        return previous_state, previous_report, False
    except Exception as error:
        report = {
            "reliable": False,
            "appears_fixed": None,
            "error": f"{type(error).__name__}: {error}",
        }
    state = update_capture_motion_state(previous_state, report)
    # Once movement has reliable positive evidence, keep the report that
    # justifies the sticky state.  A later fixed/inconclusive pair must not
    # leave the manifest with state="moved" but an assessment saying fixed.
    previous_is_movement_evidence = bool(
        previous_state == "moved"
        and previous_report
        and previous_report.get("reliable", False)
        and previous_report.get("appears_fixed") is False
    )
    current_is_movement_evidence = bool(
        report.get("reliable", False) and report.get("appears_fixed") is False
    )
    if state == "moved" and previous_is_movement_evidence and not current_is_movement_evidence:
        report = previous_report
    previous_is_fixed_evidence = bool(
        previous_state == "not_moved"
        and previous_report
        and previous_report.get("reliable", False)
        and previous_report.get("appears_fixed") is True
    )
    current_is_fixed_evidence = bool(
        report.get("reliable", False) and report.get("appears_fixed") is True
    )
    if state == "not_moved" and previous_is_fixed_evidence and not current_is_fixed_evidence:
        report = previous_report
    return state, report, True

def harvest_capture_motion(scan: ActiveScan, *, wait_timeout_s: float | None = None) -> bool:
    """Apply one worker result on the camera thread, if it is available."""
    state, report, consumed = consume_capture_motion_future(
        scan.motion_state,
        scan.motion_report,
        scan.motion_future,
        wait_timeout_s=wait_timeout_s,
    )
    if consumed:
        try:
            scan.capture_quality = scan.motion_future.result().get("capture_quality", {"guidance": "uncertain"})
        except Exception:
            scan.capture_quality = {"guidance": "uncertain"}
        scan.quality_history.append(dict(saved_frame_count=scan.frame_count, **scan.capture_quality))
        scan.motion_state = state
        scan.motion_report = report
        scan.motion_future = None
    return consumed

def wait_for_aligned_frame(
    camera: OrbbecAlignedCamera,
    *,
    timeout_s: float = ALIGNED_FRAME_STARTUP_TIMEOUT_S,
    read_timeout_ms: int = 1000,
    clock: Callable[[], float] = time.monotonic,
) -> AlignedFrame | None:
    """Wait until a genuinely aligned RGB-D frame arrives or a deadline expires.

    Some SDK startup framesets are incomplete and return quickly, so a fixed
    attempt count can expire in well under its intended number of seconds.
    """
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    if read_timeout_ms <= 0:
        raise ValueError("read_timeout_ms must be positive")
    deadline = clock() + timeout_s
    while True:
        remaining_s = deadline - clock()
        if remaining_s <= 0:
            return None
        timeout_ms = max(1, min(read_timeout_ms, int(np.ceil(remaining_s * 1000.0))))
        frame = camera.read(timeout_ms)
        if frame is not None:
            return frame

def save_scan_frame(scan: ActiveScan, frame: AlignedFrame, *, lossless: bool = False) -> None:
    scan.lossless_rgb = lossless
    index = scan.frame_count
    stem = f"{index:06d}"
    rgb_path = scan.session / "rgb" / (f"{stem}.png" if lossless else f"{stem}.jpg")
    depth_path = scan.session / "depth" / f"{stem}.png"
    pose_path = scan.session / "pose" / f"{stem}.json"
    imwrite_unicode(rgb_path, frame.color_bgr, [] if lossless else [cv2.IMWRITE_JPEG_QUALITY, 95])
    imwrite_unicode(
        depth_path,
        np.rint(np.clip(frame.depth_mm, 0, 65535)).astype(np.uint16),
    )
    write_json(
        pose_path,
        {
            "schema": "femto-scene-frame-pose-v1",
            "frame": index,
            "pose_valid": scan.fixed_camera,
            "T_WC": np.eye(4).tolist() if scan.fixed_camera else None,
            "pose_source": (
                "fixed_camera_identity_user_selected"
                if scan.fixed_camera
                else "not_available; estimate before multiview reconstruction"
            ),
            "coordinate_system": "+X right, +Y down, +Z forward",
        },
    )
    append_jsonl(
        scan.frames_log,
        {
            "frame": index,
            "camera_frame_index": frame.frame_index,
            "camera_timestamp_us": frame.timestamp_us,
            "system_timestamp_us": frame.system_timestamp_us,
            "color_timestamp_us": frame.color_timestamp_us,
            "color_system_timestamp_us": frame.color_system_timestamp_us,
            "depth_timestamp_us": frame.depth_timestamp_us,
            "depth_system_timestamp_us": frame.depth_system_timestamp_us,
            "rgb_depth_timestamp_delta_us": frame.color_timestamp_us - frame.depth_timestamp_us,
            "frame_metadata": frame.frame_metadata,
            "rgb": rgb_path.relative_to(scan.session).as_posix(),
            "depth": depth_path.relative_to(scan.session).as_posix(),
            "pose": pose_path.relative_to(scan.session).as_posix(),
            "pose_valid": scan.fixed_camera,
        },
    )
    scan.frame_count += 1

def finalize_scan(scan: ActiveScan, camera: OrbbecAlignedCamera) -> Path:
    versions: dict[str, str] = {}
    for distribution in ("pyorbbecsdk2", "numpy", "opencv-contrib-python"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            pass
    manifest = {
        "schema": "femto-full-scene-rgbd-v1",
        "created_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "status": "captured" if scan.fixed_camera else "captured_pose_pending",
        "scene_scope": "complete camera field; no semantic mask applied",
        "frame_count": scan.frame_count,
        "camera": {
            **dict(camera.metadata),
            "intrinsics": camera.intrinsics.to_dict() if camera.intrinsics else None,
        },
        "pose": {
            "valid": scan.fixed_camera,
            "source": "fixed_camera_identity_user_selected" if scan.fixed_camera else "not_available",
            "warning": (
                "Single camera viewpoint; unobserved sides remain unknown."
                if scan.fixed_camera
                else "Estimate and validate every T_WC before TSDF or 3DGS training."
            ),
        },
        "parameters": {
            "capture_seconds": scan.duration_s,
            "capture_fps": scan.fps,
            "max_frames": scan.max_frames,
            "fixed_camera": scan.fixed_camera,
            "processing_mode": scan.processing_mode,
            "depth_min_mm": scan.depth_min_mm,
            "depth_max_mm": scan.depth_max_mm,
            "entry": "femto.app.qt_main",
        },
        "camera_motion_feedback": {
            "state": scan.motion_state if not scan.fixed_camera else "not_applicable",
            "assessment": scan.motion_report,
            "latest_capture_quality": scan.capture_quality,
            "purpose": (
                "operator capture feedback only; not_moved means no reliable movement evidence; "
                "not a pose, localization result, or fixed-camera classification"
            ),
        },
        "software": versions,
        "files": {
            "rgb": "rgb/*.png" if getattr(scan, 'lossless_rgb', False) else "rgb/*.jpg",
            "depth": "depth/*.png",
            "pose": "pose/*.json",
            "frames": "frames.jsonl",
            "calibration": "calibration/*.json",
        },
    }
    if not scan.fixed_camera:
        quality_path = scan.session / "derived" / "capture_quality.json"
        quality_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(quality_path, {"schema": "femto-moving-capture-quality-v1",
            "hint_only": True, "samples": scan.quality_history,
            "purpose": "Image/depth guidance only; not a live pose or overlap measurement"})
    write_json(scan.session / "manifest.json", manifest)
    if scan.last_preview is not None:
        imwrite_unicode(
            scan.session / "last_preview.jpg",
            scan.last_preview,
            [cv2.IMWRITE_JPEG_QUALITY, 94],
        )
    # Follow the actual session root (including isolated replay/test roots).
    # Writing to DATASETS_ROOT here let tests replace the user's latest scene.
    (scan.session.parent / "latest_scene.txt").write_text(str(scan.session), encoding="utf-8")
    return scan.session
