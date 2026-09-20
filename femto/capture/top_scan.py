"""Finite top-view patient RGB-D capture, 3-D joints, mask, surface and TSDF."""
from __future__ import annotations

import argparse
import ctypes
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import time
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
import open3d as o3d

from femto.geometry.fusion import (
    build_fixed_camera_tsdf,
    create_surface_cloud,
    export_mesh_formats,
    fill_small_mesh_holes,
    median_joints,
    temporal_median_depth,
    write_point_cloud_safe,
)
from femto.io_utils import append_jsonl, imwrite_unicode, write_json as json_dump
from femto.hardware.orbbec_rgbd import OrbbecAlignedCamera
from femto.geometry.pose_geometry import (
    LANDMARK_NAMES,
    POSE_CONNECTIONS,
    Intrinsics,
    Landmark2D,
    derive_patient_regions,
    landmarks_to_3d,
    masked_depth_residual_mm,
    serialise_joints,
    torso_motion_mm,
)


from femto.paths import REPO_ROOT as ROOT
DEFAULT_MODEL = ROOT / "models" / "pose_landmarker_full.task"
DEFAULT_OUTPUT = ROOT / "outputs"
MIN_DEPTH_MM = 500.0
MAX_DEPTH_MM = 3860.0


def enable_windows_dpi_awareness() -> None:
    """Keep OpenCV image pixels and mouse coordinates aligned on scaled displays."""
    if os.name != "nt":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Femto Bolt fixed top-camera patient surface + 3-D pose capture"
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--accepted-frames", type=int, default=30)
    parser.add_argument("--capture-seconds", type=float, default=12.0)
    parser.add_argument("--minimum-frames", type=int, default=8)
    parser.add_argument("--inference-width", type=int, default=640)
    parser.add_argument("--min-valid-joints", type=int, default=12)
    parser.add_argument("--min-mask-depth-pixels", type=int, default=5000)
    parser.add_argument("--max-pose-motion-mm", type=float, default=45.0)
    parser.add_argument("--max-depth-residual-mm", type=float, default=35.0)
    parser.add_argument("--mask-threshold", type=float, default=0.55)
    parser.add_argument(
        "--allow-missing-stability",
        action="store_true",
        help="Diagnostic only: do not reject when torso/depth stability cannot be computed",
    )
    parser.add_argument("--tsdf-voxel-mm", type=float, default=5.0)
    parser.add_argument("--tsdf-trunc-mm", type=float, default=20.0)
    parser.add_argument("--tsdf-image-scale", type=float, default=0.5)
    parser.add_argument("--minimum-component-triangles", type=int, default=50)
    parser.add_argument(
        "--fill-holes-mm",
        type=float,
        default=8.0,
        help="Fill only bounded TSDF holes up to this radius; 0 disables repair",
    )
    parser.add_argument("--skip-tsdf", action="store_true")
    parser.add_argument("--no-window", action="store_true")
    parser.add_argument(
        "--auto-start",
        action="store_true",
        help="Start capture immediately instead of waiting for the preview button",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def media_pipe_model_path(project_model: Path) -> Path:
    """Copy the task model to ASCII-only storage for MediaPipe on Windows."""
    cache = Path(os.environ["LOCALAPPDATA"]) / "FemtoPatientPose" / "models"
    cache.mkdir(parents=True, exist_ok=True)
    runtime_model = cache / project_model.name
    source_hash = sha256(project_model)
    if not runtime_model.exists() or sha256(runtime_model) != source_hash:
        shutil.copy2(project_model, runtime_model)
    return runtime_model


def mediapipe_landmarks(raw_landmarks: Any) -> list[Landmark2D]:
    return [
        Landmark2D(
            x=float(item.x),
            y=float(item.y),
            visibility=float(getattr(item, "visibility", 1.0) or 0.0),
            presence=float(getattr(item, "presence", 1.0) or 0.0),
        )
        for item in raw_landmarks
    ]


def clean_person_mask(mask_probability: np.ndarray, output_shape: tuple[int, int], threshold: float) -> np.ndarray:
    mask_probability = np.asarray(mask_probability)
    if mask_probability.ndim == 3:
        mask_probability = mask_probability[:, :, 0]
    height, width = output_shape
    mask_probability = cv2.resize(
        mask_probability.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR
    )
    mask = (mask_probability >= threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return mask


def draw_overlay(
    color_bgr: np.ndarray,
    mask: np.ndarray | None,
    pixels: np.ndarray | None,
    points_m: np.ndarray | None,
    accepted: int,
    target: int,
    status: str,
    fps: float,
) -> np.ndarray:
    display = color_bgr.copy()
    if mask is not None:
        region = mask.astype(bool)
        tint = np.zeros_like(display)
        tint[:, :, 1] = 190
        display[region] = cv2.addWeighted(display[region], 0.72, tint[region], 0.28, 0)
    if pixels is not None and points_m is not None:
        valid = np.isfinite(points_m).all(axis=1)
        for a, b in POSE_CONNECTIONS:
            if valid[a] and valid[b]:
                cv2.line(display, tuple(pixels[a]), tuple(pixels[b]), (30, 230, 255), 3, cv2.LINE_AA)
        for index, pixel in enumerate(pixels):
            if valid[index]:
                cv2.circle(display, tuple(pixel), 5, (40, 40, 255), -1, cv2.LINE_AA)
    cv2.rectangle(display, (0, 0), (display.shape[1], 88), (18, 22, 28), -1)
    cv2.putText(
        display,
        f"TOP PATIENT SCAN  accepted {accepted}/{target}  {fps:.1f} FPS",
        (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (235, 245, 255), 2, cv2.LINE_AA,
    )
    cv2.putText(
        display,
        status[:110],
        (18, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (120, 225, 180), 2, cv2.LINE_AA,
    )
    return display


def fit_for_display(image: np.ndarray, max_width: int = 1280, max_height: int = 820) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale == 1.0:
        return image
    return cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)


def draw_capture_control(
    image: np.ndarray,
    *,
    capture_active: bool,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Draw a fixed-size click target on the already fitted preview image."""
    display = image.copy()
    height, width = display.shape[:2]
    button_width = min(320, max(210, width // 4))
    # Keep the control inside the central 75% of the source image. Some Win32
    # OpenCV windows crop their right edge when the desktop is display-scaled.
    x1 = min(width - button_width - 16, max(16, int(round(width * 0.36))))
    y1 = 12
    x2 = width - 16
    y2 = min(height - 12, 76)
    if capture_active:
        fill = (48, 112, 188)
        border = (105, 188, 255)
        label = "CAPTURING..."
    else:
        fill = (40, 148, 74)
        border = (100, 235, 135)
        label = "START CAPTURE"
    cv2.rectangle(display, (x1, y1), (x2, y2), fill, -1, cv2.LINE_AA)
    cv2.rectangle(display, (x1, y1), (x2, y2), border, 2, cv2.LINE_AA)
    text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2)
    text_x = x1 + max(8, (button_width - text_size[0]) // 2)
    text_y = y1 + (y2 - y1 + text_size[1]) // 2
    cv2.putText(
        display,
        label,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (245, 250, 255),
        2,
        cv2.LINE_AA,
    )
    return display, (x1, y1, x2, y2)


def write_selected_frame(
    session: Path,
    index: int,
    color_bgr: np.ndarray,
    depth_mm: np.ndarray,
    mask: np.ndarray,
    pose_record: dict,
) -> None:
    stem = f"{index:04d}"
    imwrite_unicode(
        session / "selected" / "color" / f"{stem}.jpg",
        color_bgr,
        [cv2.IMWRITE_JPEG_QUALITY, 92],
    )
    imwrite_unicode(
        session / "selected" / "depth" / f"{stem}.png",
        np.rint(np.clip(depth_mm, 0, 65535)).astype(np.uint16),
    )
    imwrite_unicode(session / "selected" / "mask" / f"{stem}.png", mask)
    json_dump(session / "selected" / "pose" / f"{stem}.json", pose_record)


def create_session(output_root: Path) -> Path:
    session = output_root / time.strftime("patient_top_%Y%m%d_%H%M%S")
    for child in ("color", "depth", "mask", "pose"):
        (session / "selected" / child).mkdir(parents=True, exist_ok=True)
    return session


def main() -> int:
    enable_windows_dpi_awareness()
    args = parse_args()
    args.model = args.model.resolve()
    args.output_root = args.output_root.resolve()
    if not args.model.exists():
        raise FileNotFoundError(f"MediaPipe model not found: {args.model}")
    runtime_model = media_pipe_model_path(args.model)
    session = create_session(args.output_root)
    event_log = session / "frame_events.jsonl"
    window_title = "Femto Bolt - Top Patient Surface + 3D Joints"

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(runtime_model)),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=True,
    )

    accepted_colors: list[np.ndarray] = []
    accepted_depths: list[np.ndarray] = []
    accepted_masks: list[np.ndarray] = []
    accepted_joints: list[np.ndarray] = []
    accepted_pixels: list[np.ndarray] = []
    accepted_landmarks: list[list[Landmark2D]] = []
    accepted_meta: list[dict] = []
    rejected_counts: dict[str, int] = {}
    reference_joints = None
    reference_depth = None
    reference_mask = None
    last_timestamp_ms = -1
    last_display = None
    frames_seen = 0
    preview_frames_seen = 0
    preview_fps_started = time.monotonic()
    capture_active = bool(args.auto_start or args.no_window)
    capture_was_started = capture_active
    capture_started_monotonic = time.monotonic() if capture_active else None
    ui_state: dict[str, Any] = {
        "capture_requested": capture_active,
        "button_rect": None,
    }

    def on_mouse(event: int, x: int, y: int, flags: int, param: Any) -> None:
        del flags, param
        if event != cv2.EVENT_LBUTTONDOWN or capture_active:
            return
        rect = ui_state.get("button_rect")
        if rect is None:
            return
        x1, y1, x2, y2 = rect
        if x1 <= x <= x2 and y1 <= y <= y2:
            ui_state["capture_requested"] = True

    try:
        with OrbbecAlignedCamera() as camera, PoseLandmarker.create_from_options(options) as landmarker:
            assert camera.intrinsics is not None
            intrinsics = camera.intrinsics
            json_dump(session / "camera_intrinsic.json", intrinsics.to_dict())
            json_dump(session / "camera_metadata.json", camera.metadata)
            if not args.no_window:
                cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
                cv2.setMouseCallback(window_title, on_mouse)

            while len(accepted_depths) < args.accepted_frames:
                if ui_state["capture_requested"] and not capture_active:
                    capture_active = True
                    capture_was_started = True
                    capture_started_monotonic = time.monotonic()
                    frames_seen = 0
                    rejected_counts.clear()
                    reference_joints = None
                    reference_depth = None
                    reference_mask = None
                if capture_active:
                    assert capture_started_monotonic is not None
                    elapsed = time.monotonic() - capture_started_monotonic
                    if elapsed >= args.capture_seconds:
                        break
                frame = camera.read(1000)
                if frame is None:
                    continue
                if camera.intrinsics != intrinsics:
                    intrinsics = camera.intrinsics
                    assert intrinsics is not None
                    json_dump(session / "camera_intrinsic.json", intrinsics.to_dict())
                preview_frames_seen += 1
                if capture_active:
                    frames_seen += 1
                height, width = frame.color_bgr.shape[:2]
                infer_width = min(args.inference_width, width)
                infer_height = max(1, int(round(height * infer_width / width)))
                infer_bgr = cv2.resize(
                    frame.color_bgr, (infer_width, infer_height), interpolation=cv2.INTER_AREA
                )
                infer_rgb = cv2.cvtColor(infer_bgr, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=infer_rgb)
                timestamp_ms = max(last_timestamp_ms + 1, int(frame.timestamp_us // 1000))
                last_timestamp_ms = timestamp_ms
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                status = "Adjust camera/patient, then click START CAPTURE"
                accepted = False
                reject_reason = "preview"
                mask = None
                pixels = None
                points_m = None
                metrics: dict[str, Any] = {}
                pose_landmarks = result.pose_landmarks[0] if result.pose_landmarks else None
                segmentation = result.segmentation_masks[0] if result.segmentation_masks else None
                if pose_landmarks is not None and segmentation is not None:
                    landmarks = mediapipe_landmarks(pose_landmarks)
                    mask = clean_person_mask(
                        segmentation.numpy_view(), frame.depth_mm.shape, args.mask_threshold
                    )
                    valid_depth = (
                        np.isfinite(frame.depth_mm)
                        & (frame.depth_mm >= MIN_DEPTH_MM)
                        & (frame.depth_mm <= MAX_DEPTH_MM)
                    )
                    mask_depth_pixels = int((mask.astype(bool) & valid_depth).sum())
                    points_m, pixels, depth_samples = landmarks_to_3d(
                        landmarks,
                        frame.depth_mm,
                        intrinsics,
                        min_depth_mm=MIN_DEPTH_MM,
                        max_depth_mm=MAX_DEPTH_MM,
                        mask=mask,
                    )
                    valid_joints = int(np.isfinite(points_m).all(axis=1).sum())
                    pose_motion = (
                        None if reference_joints is None else torso_motion_mm(points_m, reference_joints)
                    )
                    depth_residual = (
                        None
                        if reference_depth is None
                        else masked_depth_residual_mm(
                            frame.depth_mm, reference_depth, mask, reference_mask
                        )
                    )
                    metrics = {
                        "valid_3d_joints": valid_joints,
                        "mask_depth_pixels": mask_depth_pixels,
                        "torso_motion_mm": pose_motion,
                        "depth_residual_mm": depth_residual,
                    }
                    if not capture_active:
                        status = (
                            f"READY | 3D joints {valid_joints}/33 | "
                            f"masked depth {mask_depth_pixels:,} px | click START CAPTURE"
                        )
                    elif valid_joints < args.min_valid_joints:
                        reject_reason = "too_few_valid_joints"
                    elif mask_depth_pixels < args.min_mask_depth_pixels:
                        reject_reason = "insufficient_masked_depth"
                    elif reference_joints is not None and pose_motion is None and not args.allow_missing_stability:
                        reject_reason = "pose_motion_unavailable"
                    elif pose_motion is not None and pose_motion > args.max_pose_motion_mm:
                        reject_reason = "pose_motion"
                    elif reference_depth is not None and depth_residual is None and not args.allow_missing_stability:
                        reject_reason = "depth_residual_unavailable"
                    elif depth_residual is not None and depth_residual > args.max_depth_residual_mm:
                        reject_reason = "depth_residual"
                    else:
                        accepted = True
                        reject_reason = ""
                        accepted_index = len(accepted_depths)
                        visibility = [item.visibility for item in landmarks]
                        presence = [item.presence for item in landmarks]
                        pose_record = {
                            "schema": "femto-patient-pose-v1",
                            "selected_index": accepted_index,
                            "camera_timestamp_us": frame.timestamp_us,
                            "system_timestamp_us": frame.system_timestamp_us,
                            "frame_index": frame.frame_index,
                            "coordinate_system": "Orbbec color camera: +X right, +Y down, +Z forward",
                            "coordinate_unit": "metres",
                            "T_WC": np.eye(4).tolist(),
                            "pose_source": "fixed_camera_identity; invalid if the camera moved",
                            "joints": serialise_joints(points_m, pixels, visibility, presence),
                            "quality": metrics,
                        }
                        write_selected_frame(
                            session, accepted_index, frame.color_bgr, frame.depth_mm, mask, pose_record
                        )
                        accepted_colors.append(frame.color_bgr.copy())
                        accepted_depths.append(frame.depth_mm.copy())
                        accepted_masks.append(mask.copy())
                        accepted_joints.append(points_m.copy())
                        accepted_pixels.append(pixels.copy())
                        accepted_landmarks.append(landmarks)
                        accepted_meta.append(pose_record)
                        if reference_joints is None:
                            reference_joints = points_m.copy()
                            reference_depth = frame.depth_mm.copy()
                            reference_mask = mask.copy()
                        status = f"Accepted {len(accepted_depths)}/{args.accepted_frames}"
                        json_dump(session / "latest_joints.json", pose_record)
                elif capture_active:
                    reject_reason = "no_pose"
                    status = "Rejected: no_pose"
                if capture_active and not accepted:
                    rejected_counts[reject_reason] = rejected_counts.get(reject_reason, 0) + 1
                    status = f"Rejected: {reject_reason}"
                if capture_active:
                    append_jsonl(
                        event_log,
                        {
                            "frame_seen": frames_seen,
                            "camera_timestamp_us": frame.timestamp_us,
                            "accepted": accepted,
                            "reason": reject_reason or "accepted",
                            "metrics": metrics,
                        },
                    )
                fps = preview_frames_seen / max(
                    time.monotonic() - preview_fps_started, 1e-6
                )
                last_display = draw_overlay(
                    frame.color_bgr,
                    mask,
                    pixels,
                    points_m,
                    len(accepted_depths),
                    args.accepted_frames,
                    status,
                    fps,
                )
                if not args.no_window:
                    last_display, source_button_rect = draw_capture_control(
                        last_display, capture_active=capture_active
                    )
                    fitted = fit_for_display(last_display)
                    scale_x = fitted.shape[1] / last_display.shape[1]
                    scale_y = fitted.shape[0] / last_display.shape[0]
                    x1, y1, x2, y2 = source_button_rect
                    ui_state["button_rect"] = (
                        int(round(x1 * scale_x)),
                        int(round(y1 * scale_y)),
                        int(round(x2 * scale_x)),
                        int(round(y2 * scale_y)),
                    )
                    cv2.resizeWindow(window_title, fitted.shape[1], fitted.shape[0])
                    cv2.imshow(window_title, fitted)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break
                    if key in (ord("s"), ord(" ")) and not capture_active:
                        ui_state["capture_requested"] = True
                    if cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1:
                        break
    finally:
        cv2.destroyAllWindows()

    if last_display is not None:
        imwrite_unicode(
            session / "last_overlay.jpg", last_display, [cv2.IMWRITE_JPEG_QUALITY, 94]
        )

    accepted_count = len(accepted_depths)
    if not capture_was_started:
        report = {
            "status": "closed_before_capture",
            "preview_frames_seen": preview_frames_seen,
            "message": "Preview was closed before START CAPTURE was pressed.",
        }
        json_dump(session / "quality_report.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return 0
    if accepted_count < args.minimum_frames:
        report = {
            "status": "insufficient_accepted_frames",
            "accepted_frames": accepted_count,
            "required_minimum": args.minimum_frames,
            "frames_seen": frames_seen,
            "rejected": rejected_counts,
        }
        json_dump(session / "quality_report.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return 2

    minimum_support = max(3, int(math.ceil(accepted_count * 0.30)))
    fused_depth, fused_mask, support = temporal_median_depth(
        accepted_depths, accepted_masks, minimum_support=minimum_support
    )
    fused_joints, joint_support = median_joints(accepted_joints, minimum_support)
    regions = derive_patient_regions(fused_joints)
    representative_color = accepted_colors[accepted_count // 2]
    imwrite_unicode(
        session / "fused_depth_mm.png",
        np.rint(np.clip(fused_depth, 0, 65535)).astype(np.uint16),
    )
    imwrite_unicode(session / "fused_human_mask.png", fused_mask)
    imwrite_unicode(
        session / "fused_color.jpg",
        representative_color,
        [cv2.IMWRITE_JPEG_QUALITY, 94],
    )
    imwrite_unicode(
        session / "temporal_support.png", np.clip(support, 0, 255).astype(np.uint8)
    )

    surface_cloud = create_surface_cloud(
        fused_depth, fused_mask, representative_color, intrinsics
    )
    surface_path = session / "patient_surface_points_m.ply"
    if surface_cloud.is_empty() or not write_point_cloud_safe(surface_path, surface_cloud):
        raise RuntimeError("Failed to export patient surface point cloud")

    joints_record = {
        "schema": "femto-patient-fused-pose-v1",
        "coordinate_system": "Orbbec color camera: +X right, +Y down, +Z forward",
        "coordinate_unit": "metres",
        "fixed_camera_assumption": True,
        "joints": serialise_joints(fused_joints),
        "joint_temporal_support": [int(value) for value in joint_support],
    }
    json_dump(session / "joints_3d_m.json", joints_record)
    json_dump(session / "patient_regions_3d_m.json", regions)

    with (session / "joints_3d_m.csv").open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.writer(output)
        writer.writerow(["index", "name", "x_m", "y_m", "z_m", "support_frames"])
        for index, name in enumerate(LANDMARK_NAMES):
            point = fused_joints[index]
            writer.writerow(
                [
                    index,
                    name,
                    "" if not np.isfinite(point[0]) else float(point[0]),
                    "" if not np.isfinite(point[1]) else float(point[1]),
                    "" if not np.isfinite(point[2]) else float(point[2]),
                    int(joint_support[index]),
                ]
            )

    mesh_result = None
    if not args.skip_tsdf:
        mesh = build_fixed_camera_tsdf(
            accepted_colors,
            accepted_depths,
            accepted_masks,
            intrinsics,
            voxel_length_m=args.tsdf_voxel_mm / 1000.0,
            sdf_trunc_m=args.tsdf_trunc_mm / 1000.0,
            image_scale=args.tsdf_image_scale,
            minimum_component_triangles=args.minimum_component_triangles,
        )
        if len(mesh.triangles):
            observed = export_mesh_formats(mesh, session)
            mesh_result = {"observed": observed, "filled": None}
            if args.fill_holes_mm > 0:
                filled = fill_small_mesh_holes(mesh, args.fill_holes_mm / 1000.0)
                repaired = export_mesh_formats(
                    filled,
                    session,
                    stem="patient_surface_tsdf_filled_m",
                    report_name="mesh_filled_report.json",
                    note=(
                        f"Only bounded holes up to {args.fill_holes_mm:g} mm plus "
                        "triangle/quad micro-holes were filled; silhouette and unobserved "
                        "surfaces remain open."
                    ),
                )
                repaired["triangles_added"] = int(len(filled.triangles) - len(mesh.triangles))
                repaired["maximum_hole_radius_mm"] = float(args.fill_holes_mm)
                mesh_result["filled"] = repaired

    versions = {}
    for distribution in ("pyorbbecsdk2", "mediapipe", "numpy", "open3d", "opencv-python", "trimesh"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            pass
    run_mode = "diagnostic" if args.allow_missing_stability else "standard"
    report = {
        "status": "complete_diagnostic" if run_mode == "diagnostic" else "complete",
        "run_mode": run_mode,
        "project_goal": "基于 Femto Bolt RGB-D 相机的仰卧人体三维表面重建与人体关键区域定位系统",
        "capture_mode": "finite burst, fixed top camera",
        "accepted_frames": accepted_count,
        "target_frames": args.accepted_frames,
        "frames_seen": frames_seen,
        "rejected": rejected_counts,
        "minimum_temporal_support": minimum_support,
        "valid_fused_joints": int(np.isfinite(fused_joints).all(axis=1).sum()),
        "surface_points": int(len(surface_cloud.points)),
        "tsdf": mesh_result,
        "limitations": [
            "This run assumes the camera did not move; every T_WC is identity.",
            "MediaPipe/Mask landmarks are navigation labels, not validated treatment targets.",
            "The model is the top visible patient surface and is expected to remain open.",
            "A moving eye-in-hand scan requires synchronized robot T_BF and calibrated T_FC.",
        ],
    }
    json_dump(session / "quality_report.json", report)
    manifest = {
        "schema": "femto-patient-top-scan-v1",
        "created_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "session": session.name,
        "model": {"path": str(args.model), "sha256": sha256(args.model)},
        "camera": {**camera.metadata, "intrinsics": intrinsics.to_dict()},
        "software": versions,
        "parameters": vars(args) | {"model": str(args.model), "output_root": str(args.output_root)},
        "outputs": {
            "surface_point_cloud": surface_path.name,
            "fused_depth": "fused_depth_mm.png",
            "fused_color": "fused_color.jpg",
            "fused_human_mask": "fused_human_mask.png",
            "temporal_support": "temporal_support.png",
            "joints": "joints_3d_m.json",
            "joints_csv": "joints_3d_m.csv",
            "regions": "patient_regions_3d_m.json",
            "quality_report": "quality_report.json",
            "selected_frames": "selected/",
            "frame_events": "frame_events.jsonl",
            "tsdf": None if mesh_result is None else {
                "observed": {key: Path(value).name for key, value in mesh_result["observed"].items() if key in {"ply", "obj", "glb"}},
                "filled": None if mesh_result["filled"] is None else {key: Path(value).name for key, value in mesh_result["filled"].items() if key in {"ply", "obj", "glb"}},
            },
        },
    }
    json_dump(session / "manifest.json", manifest)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "latest_session.txt").write_text(str(session), encoding="utf-8")
    print(f"[COMPLETE] {session}", flush=True)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
