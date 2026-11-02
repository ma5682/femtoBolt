"""Shared headless fixtures for camera-worker tests.

The Qt migration removed widget construction from these tests. The worker
contract only needs queues, locks and stop state, so this fixture builds a
lightweight object and exposes explicit worker adapters for regression tests.
"""
from __future__ import annotations

import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace


def _dependencies():
    import femto.app.femto_control_app as control
    from femto.app.camera_worker import CameraWorker
    from femto.capture.scene_capture import create_session
    from femto.capture.scan_session import (
        ActiveScan, copy_frame, finalize_scan, harvest_capture_motion,
        save_rgbd_photo, save_scan_frame, wait_for_aligned_frame,
    )
    from femto.dataset.calibration_files import write_calibration
    from femto.tracking.moving_quality import capture_feedback
    return control, CameraWorker, SimpleNamespace(
        APP_DEPTH_MODE=control.APP_DEPTH_MODE,
        APP_COLOR_SIZE=control.APP_COLOR_SIZE,
        SCAN_MODE_KEYS=control.SCAN_MODE_KEYS,
        # Resolve through the application module so tests that patch its
        # output roots exercise the same paths as the production worker.
        DATASETS_ROOT=control.DATASETS_ROOT,
        OUTPUTS_ROOT=control.OUTPUTS_ROOT,
        OrbbecAlignedCamera=control.OrbbecAlignedCamera,
        wait_for_aligned_frame=wait_for_aligned_frame,
        create_session=create_session,
        write_calibration=write_calibration,
        time=__import__('time'),
        ActiveScan=ActiveScan,
        MOTION_CHECK_INTERVAL_S=control.MOTION_CHECK_INTERVAL_S,
        MOTION_CHECK_SCALE=control.MOTION_CHECK_SCALE,
        MOTION_FINAL_WAIT_S=control.MOTION_FINAL_WAIT_S,
        copy_frame=copy_frame,
        save_rgbd_photo=save_rgbd_photo,
        save_scan_frame=save_scan_frame,
        finalize_scan=finalize_scan,
        harvest_capture_motion=harvest_capture_motion,
        capture_feedback=capture_feedback,
        ThreadPoolExecutor=ThreadPoolExecutor,
        threading=threading,
        traceback=__import__('traceback'),
        run_demo=control.run_demo,
        build_fixed_keyframe_dataset=control.build_fixed_keyframe_dataset,
    )


def bare_camera_app():
    """Return the minimum state consumed by CameraWorker dispatch tests."""
    app = SimpleNamespace()
    app.language = "zh"
    app.event_queue = queue.Queue()
    app.command_queue = queue.Queue()
    app.stop_event = threading.Event()
    app.frame_lock = threading.Lock()
    app.temporal_thread = None
    app.camera_thread = None
    app.photo_thread = None
    app.latest_frame = None
    app.camera_ready = False
    app.processing = False
    app.temporal_busy = False
    app.temporal_collecting = False
    app.latest_result = None
    app.latest_temporal = None
    app.preview_counter = 0
    def translate(key, **kwargs):
        messages = {
            "app.capture.single_only": {
                "zh": "当前采集模式仅支持单帧重建",
                "en": "paused capture supports single-frame reconstruction only",
            }
        }
        return messages.get(key, {}).get(app.language, key)
    app._t = translate
    app._dispatch_temporal = lambda burst, **kwargs: None
    app._photo_worker = lambda frame, camera: None
    def camera_worker():
        _control, Worker, deps = _dependencies()
        Worker(app, deps).run()
    def reconstruction_worker(dataset, processing_mode):
        _control, Worker, deps = _dependencies()
        Worker(app, deps).reconstruction_worker(Path(dataset), processing_mode)
    app._camera_worker = camera_worker
    app._reconstruction_worker = reconstruction_worker
    return app
