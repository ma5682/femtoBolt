"""Capture an unmasked full-scene Femto Bolt RGB-D dataset."""
from __future__ import annotations

import argparse
import ctypes
import importlib.metadata
import json
import os
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np

from io_utils import append_jsonl, imwrite_unicode, write_json as json_dump
from orbbec_rgbd import OrbbecAlignedCamera


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASETS = ROOT / "datasets"
MIN_DEPTH_MM = 500.0
MAX_DEPTH_MM = 3860.0


def enable_windows_dpi_awareness() -> None:
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
        description="Capture complete aligned RGB-D frames without semantic masking"
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_DATASETS)
    parser.add_argument("--capture-seconds", type=float, default=15.0)
    parser.add_argument("--capture-fps", type=float, default=10.0)
    parser.add_argument("--max-frames", type=int, default=150)
    parser.add_argument(
        "--fixed-camera",
        action="store_true",
        help="Mark T_WC=identity as valid; use only when the camera cannot move",
    )
    parser.add_argument("--auto-start", action="store_true")
    parser.add_argument("--no-window", action="store_true")
    return parser.parse_args()


def write_calibration(session: Path, camera: OrbbecAlignedCamera, intrinsics: Any) -> None:
    """Save raw factory calibration and the aligned RGB-D pixel-grid intrinsic."""
    calibration = camera.calibration
    json_dump(session / "calibration" / "color_intrinsic.json", intrinsics.to_dict())
    json_dump(
        session / "calibration" / "depth_intrinsic.json",
        calibration["depth_intrinsic"],
    )
    json_dump(
        session / "calibration" / "depth_to_color.json",
        calibration["depth_to_color"],
    )
    json_dump(
        session / "calibration" / "distortion.json",
        {
            "rgb": calibration["rgb_distortion"],
            "depth": calibration["depth_distortion"],
        },
    )
    json_dump(session / "calibration" / "factory_calibration.json", calibration)
    json_dump(session / "calibration" / "camera_metadata.json", camera.metadata)


def create_session(output_root: Path) -> Path:
    session = output_root / time.strftime("scene_%Y%m%d_%H%M%S")
    for child in ("rgb", "depth", "pose"):
        (session / child).mkdir(parents=True, exist_ok=False)
    return session


def fit_for_display(image: np.ndarray, max_width: int = 1280, max_height: int = 820) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale == 1.0:
        return image
    return cv2.resize(
        image,
        (int(round(width * scale)), int(round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def depth_preview(
    depth_mm: np.ndarray,
    min_depth_mm: float = MIN_DEPTH_MM,
    max_depth_mm: float = MAX_DEPTH_MM,
) -> np.ndarray:
    valid = np.isfinite(depth_mm) & (depth_mm >= min_depth_mm) & (depth_mm <= max_depth_mm)
    normalized = np.zeros(depth_mm.shape, dtype=np.uint8)
    normalized[valid] = np.clip(
        (depth_mm[valid] - min_depth_mm) * 255.0 / (max_depth_mm - min_depth_mm),
        0,
        255,
    ).astype(np.uint8)
    coloured = cv2.applyColorMap(255 - normalized, cv2.COLORMAP_TURBO)
    coloured[~valid] = 0
    return coloured


def draw_preview(
    color_bgr: np.ndarray,
    depth_mm: np.ndarray,
    *,
    active: bool,
    saved_frames: int,
    max_frames: int,
    elapsed: float,
    capture_seconds: float,
    fps: float,
    pose_valid: bool,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    display = color_bgr.copy()
    height, width = display.shape[:2]
    cv2.rectangle(display, (0, 0), (width, 102), (18, 22, 28), -1)
    state = "CAPTURING" if active else "ADJUST SCENE"
    pose_text = "fixed identity" if pose_valid else "pending / invalid"
    cv2.putText(
        display,
        f"FULL SCENE RGB-D  {state}  frames {saved_frames}/{max_frames}  {fps:.1f} FPS",
        (22, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (240, 248, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        display,
        f"time {elapsed:.1f}/{capture_seconds:.1f}s | camera pose: {pose_text}",
        (22, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (130, 225, 180) if pose_valid else (80, 190, 255),
        2,
        cv2.LINE_AA,
    )

    button_width = min(390, max(250, width // 5))
    x1 = min(width - button_width - 20, max(20, int(round(width * 0.36))))
    y1, y2 = 12, 88
    if active:
        fill, border, label = (45, 90, 180), (100, 175, 255), "STOP AND SAVE"
    else:
        fill, border, label = (40, 148, 74), (100, 235, 135), "START CAPTURE"
    cv2.rectangle(display, (x1, y1), (x1 + button_width, y2), fill, -1, cv2.LINE_AA)
    cv2.rectangle(display, (x1, y1), (x1 + button_width, y2), border, 3, cv2.LINE_AA)
    text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2)
    cv2.putText(
        display,
        label,
        (x1 + (button_width - text_size[0]) // 2, 61),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (245, 250, 255),
        2,
        cv2.LINE_AA,
    )

    inset_width = max(240, width // 5)
    inset_height = int(round(inset_width * depth_mm.shape[0] / depth_mm.shape[1]))
    depth_small = cv2.resize(
        depth_preview(depth_mm), (inset_width, inset_height), interpolation=cv2.INTER_NEAREST
    )
    ix1 = width - inset_width - 18
    iy1 = height - inset_height - 18
    display[iy1 : iy1 + inset_height, ix1 : ix1 + inset_width] = depth_small
    cv2.rectangle(
        display,
        (ix1 - 2, iy1 - 2),
        (ix1 + inset_width + 2, iy1 + inset_height + 2),
        (235, 245, 255),
        2,
    )
    return display, (x1, y1, x1 + button_width, y2)


def main() -> int:
    enable_windows_dpi_awareness()
    args = parse_args()
    if args.capture_fps <= 0 or args.max_frames <= 0 or args.capture_seconds <= 0:
        raise ValueError("capture fps, frame count and duration must be positive")
    output_root = args.output_root.resolve()
    window_title = "Femto Bolt - Full Scene RGB-D Capture"
    active = bool(args.auto_start or args.no_window)
    ever_started = active
    stop_requested = False
    session: Path | None = None
    frames_log: Path | None = None
    # Camera warm-up can take several seconds. Start capture time only when
    # the camera is ready and the output session has been created.
    capture_started = None
    next_sample = 0.0
    saved_frames = 0
    preview_frames = 0
    preview_started = time.monotonic()
    last_display = None
    camera_metadata: dict[str, Any] = {}
    ui: dict[str, Any] = {"button_rect": None, "toggle_requested": False}

    def on_mouse(event: int, x: int, y: int, flags: int, param: Any) -> None:
        del flags, param
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        rect = ui.get("button_rect")
        if rect is None:
            return
        x1, y1, x2, y2 = rect
        if x1 <= x <= x2 and y1 <= y <= y2:
            ui["toggle_requested"] = True

    try:
        with OrbbecAlignedCamera() as camera:
            assert camera.intrinsics is not None
            intrinsics = camera.intrinsics
            if not args.no_window:
                cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
                cv2.setMouseCallback(window_title, on_mouse)

            while True:
                if ui["toggle_requested"]:
                    ui["toggle_requested"] = False
                    if active and ever_started:
                        stop_requested = True
                    elif not active:
                        active = True
                        ever_started = True
                        capture_started = time.monotonic()
                        next_sample = capture_started
                        session = create_session(output_root)
                        frames_log = session / "frames.jsonl"
                        write_calibration(session, camera, intrinsics)

                if active and session is None:
                    capture_started = time.monotonic()
                    next_sample = capture_started
                    session = create_session(output_root)
                    frames_log = session / "frames.jsonl"
                    write_calibration(session, camera, intrinsics)

                frame = camera.read(1000)
                if frame is None:
                    continue
                preview_frames += 1
                now = time.monotonic()
                elapsed = 0.0 if capture_started is None else now - capture_started

                if active and not stop_requested and now >= next_sample:
                    assert session is not None and frames_log is not None
                    index = saved_frames
                    stem = f"{index:06d}"
                    rgb_path = session / "rgb" / f"{stem}.jpg"
                    depth_path = session / "depth" / f"{stem}.png"
                    pose_path = session / "pose" / f"{stem}.json"
                    imwrite_unicode(rgb_path, frame.color_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    imwrite_unicode(
                        depth_path,
                        np.rint(np.clip(frame.depth_mm, 0, 65535)).astype(np.uint16),
                    )
                    pose_record = {
                        "schema": "femto-scene-frame-pose-v1",
                        "frame": index,
                        "pose_valid": bool(args.fixed_camera),
                        "T_WC": np.eye(4).tolist() if args.fixed_camera else None,
                        "pose_source": (
                            "fixed_camera_identity"
                            if args.fixed_camera
                            else "not_available; estimate before multiview reconstruction"
                        ),
                        "coordinate_system": "+X right, +Y down, +Z forward",
                    }
                    json_dump(pose_path, pose_record)
                    append_jsonl(
                        frames_log,
                        {
                            "frame": index,
                            "camera_frame_index": frame.frame_index,
                            "camera_timestamp_us": frame.timestamp_us,
                            "system_timestamp_us": frame.system_timestamp_us,
                            "color_timestamp_us": frame.color_timestamp_us,
                            "color_system_timestamp_us": frame.color_system_timestamp_us,
                            "depth_timestamp_us": frame.depth_timestamp_us,
                            "depth_system_timestamp_us": frame.depth_system_timestamp_us,
                            "rgb_depth_timestamp_delta_us": (
                                frame.color_timestamp_us - frame.depth_timestamp_us
                            ),
                            "frame_metadata": frame.frame_metadata,
                            "rgb": rgb_path.relative_to(session).as_posix(),
                            "depth": depth_path.relative_to(session).as_posix(),
                            "pose": pose_path.relative_to(session).as_posix(),
                            "pose_valid": bool(args.fixed_camera),
                        },
                    )
                    saved_frames += 1
                    next_sample += 1.0 / args.capture_fps

                if active and (
                    stop_requested
                    or elapsed >= args.capture_seconds
                    or saved_frames >= args.max_frames
                ):
                    break

                fps = preview_frames / max(now - preview_started, 1e-6)
                last_display, source_rect = draw_preview(
                    frame.color_bgr,
                    frame.depth_mm,
                    active=active,
                    saved_frames=saved_frames,
                    max_frames=args.max_frames,
                    elapsed=elapsed,
                    capture_seconds=args.capture_seconds,
                    fps=fps,
                    pose_valid=bool(args.fixed_camera),
                )
                if not args.no_window:
                    fitted = fit_for_display(last_display)
                    sx = fitted.shape[1] / last_display.shape[1]
                    sy = fitted.shape[0] / last_display.shape[0]
                    x1, y1, x2, y2 = source_rect
                    ui["button_rect"] = (
                        int(round(x1 * sx)),
                        int(round(y1 * sy)),
                        int(round(x2 * sx)),
                        int(round(y2 * sy)),
                    )
                    cv2.resizeWindow(window_title, fitted.shape[1], fitted.shape[0])
                    cv2.imshow(window_title, fitted)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        if active:
                            stop_requested = True
                            break
                        return 0
                    if key in (ord("s"), ord(" ")):
                        ui["toggle_requested"] = True
                    if cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1:
                        if active:
                            stop_requested = True
                            break
                        return 0
            camera_metadata = dict(camera.metadata)
    finally:
        cv2.destroyAllWindows()

    if not ever_started or session is None:
        return 0

    versions = {}
    for distribution in ("pyorbbecsdk2", "numpy", "opencv-contrib-python"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            pass
    manifest = {
        "schema": "femto-full-scene-rgbd-v1",
        "created_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "status": "captured" if args.fixed_camera else "captured_pose_pending",
        "scene_scope": "complete camera field; no semantic mask applied",
        "frame_count": saved_frames,
        "camera": {**camera_metadata, "intrinsics": intrinsics.to_dict()},
        "pose": {
            "valid": bool(args.fixed_camera),
            "source": "fixed_camera_identity" if args.fixed_camera else "not_available",
            "warning": (
                None
                if args.fixed_camera
                else "Estimate and validate every T_WC before TSDF or 3DGS training."
            ),
        },
        "parameters": {
            "capture_seconds": args.capture_seconds,
            "capture_fps": args.capture_fps,
            "max_frames": args.max_frames,
            "fixed_camera": args.fixed_camera,
        },
        "software": versions,
        "files": {
            "rgb": "rgb/*.jpg",
            "depth": "depth/*.png",
            "pose": "pose/*.json",
            "frames": "frames.jsonl",
            "calibration": "calibration/*.json",
        },
    }
    json_dump(session / "manifest.json", manifest)
    if last_display is not None:
        imwrite_unicode(session / "last_preview.jpg", last_display, [cv2.IMWRITE_JPEG_QUALITY, 94])
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "latest_scene.txt").write_text(str(session), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    print(f"[COMPLETE] {session}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
