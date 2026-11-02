"""Compatibility entry point for the Qt Femto workstation.

The former module hosted the complete legacy desktop UI.  It now re-exports
the capture-session API used by workers and forwards normal launches to the
PySide6 workbench in :mod:`femto.app.qt_main`.  Keeping this import path avoids
breaking scripts that only use the backend writers while removing any GUI
toolkit dependency from the application package.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

from femto.capture.scan_session import (
    ALIGNED_FRAME_STARTUP_TIMEOUT_S,
    MOTION_CHECK_INTERVAL_S,
    MOTION_CHECK_SCALE,
    MOTION_FINAL_WAIT_S,
    ActiveScan,
    consume_capture_motion_future,
    copy_frame,
    finalize_scan,
    harvest_capture_motion,
    save_rgbd_photo,
    save_scan_frame,
    unique_timestamp_dir,
    update_capture_motion_state,
    wait_for_aligned_frame,
)
from femto.capture.scene_capture import create_session, enable_windows_dpi_awareness
from femto.dataset.calibration_files import write_calibration
from femto.gsplat.gsplat_backend import (
    PROFILE_KEYS as THREE_DGS_PROFILE_KEYS,
    PROFILES,
    assess_training_input,
    latest_trained_model,
    open_viewer as open_gsplat_viewer,
    runtime_status as gsplat_runtime_status,
    train as train_direct_gsplat,
)
from femto.hardware.orbbec_rgbd import AlignedFrame, DEPTH_MODE_SPECS, OrbbecAlignedCamera
from femto.io_utils import write_json
from femto.paths import DATASETS_ROOT, LOG_ROOT, OUTPUTS_ROOT, PHOTOS_ROOT, REPO_ROOT as ROOT
from femto.pipeline.complete_demo import run_demo
from femto.temporal.scene_keyframe import build_fixed_keyframe_dataset
from femto.tracking.moving_quality import capture_feedback
from femto.tracking.scene_pose_estimator import assess_rgb_motion_pair


SCAN_MODE_KEYS = ("keyframe", "moving")
APP_DEPTH_MODE = "wide_detail"
APP_COLOR_SIZE = (1920, 1080)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Femto Bolt PySide6 workstation")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Connect, save one RGB-D photo and exit")
    parser.add_argument("--ui-preview", action="store_true",
                        help="Open the Qt workbench without connecting a camera")
    parser.add_argument("--project", type=Path,
                        help="Open an existing reconstruction or planning archive")
    return parser.parse_args(argv)


class FemtoControlApp:
    """Lazy compatibility factory for callers of the removed app class.

    A returned instance is the Qt :class:`WorkstationWindow`; importing this
    module does not instantiate Qt, making backend scripts safe in headless
    processes.
    """

    def __new__(cls, *args: Any, **kwargs: Any):
        from femto.app.qt_main import WorkstationWindow

        connect_camera = kwargs.pop("connect_camera", True)
        project = kwargs.pop("project", None)
        # Legacy callers passed a root widget as the first positional value.
        # It has no role in the Qt workbench and is intentionally ignored.
        del args
        return WorkstationWindow(connect_camera=connect_camera, project=project)


def smoke_test() -> int:
    """Connect once, write one RGB-D photo and print a JSON health record."""
    camera: OrbbecAlignedCamera | None = None
    started = time.monotonic()
    try:
        with OrbbecAlignedCamera(APP_DEPTH_MODE, color_size=APP_COLOR_SIZE) as active_camera:
            camera = active_camera
            frame = wait_for_aligned_frame(camera)
            if frame is None:
                raise RuntimeError("Camera produced no aligned RGB-D frame before the startup deadline")
            output = save_rgbd_photo(frame, camera)
    except Exception as error:
        payload: dict[str, Any] = {
            "status": "failed", "error": str(error),
            "elapsed_s": time.monotonic() - started,
        }
        if camera is not None:
            payload["connection_type"] = camera.metadata.get("connection_type")
            payload["read_diagnostics"] = camera.read_diagnostics_snapshot()
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps({
        "status": "passed", "photo": str(output),
        "color_shape": list(frame.color_bgr.shape),
        "depth_shape": list(frame.depth_mm.shape),
        "depth_mode": camera.metadata.get("depth_mode"),
        "depth_range_m": camera.metadata.get("depth_range_m"),
        "depth_profile": camera.metadata.get("depth_profile"),
        "color_profile": camera.metadata.get("color_profile"),
        "rgb_depth_delta_us": frame.color_timestamp_us - frame.depth_timestamp_us,
        "connection_type": camera.metadata.get("connection_type"),
        "elapsed_s": time.monotonic() - started,
        "read_diagnostics": camera.read_diagnostics_snapshot(),
    }, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    args = parse_args()
    if args.smoke_test:
        return smoke_test()
    # qt_main owns QApplication creation and parses its own UI options.  Keep
    # this module as a stable ``python -m`` entry without importing Qt eagerly.
    from femto.app.qt_main import main as qt_main

    return qt_main()


if __name__ == "__main__":
    raise SystemExit(main())
