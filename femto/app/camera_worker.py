"""Camera-owner loop and capture command handlers for the control app."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import queue
from types import SimpleNamespace
from typing import Any


class CameraWorker:
    """Run the single SDK owner while forwarding UI state to the app."""

    def __init__(self, app: Any, dependencies: SimpleNamespace) -> None:
        self.app = app
        self.dependencies = dependencies

    def _cmd_shutdown(self, state: SimpleNamespace, _payload: Any) -> None:
        self.app.stop_event.set()

    def _cmd_local_start(self, state: SimpleNamespace, payload: Any) -> None:
        requested_session = payload.get("session_id") if isinstance(payload, dict) else None
        local_active = state.local is not None and state.local.thread.is_alive()
        if state.scan is not None or state.temporal_burst is not None or local_active:
            self.app.event_queue.put(
                (
                    "local_error",
                    {
                        "session_id": requested_session,
                        "source": None,
                        "error": "Another capture is active",
                    },
                )
            )
            return
        try:
            from femto.capture.local_capture import LocalCapture

            state.local = LocalCapture(
                state.camera,
                self.dependencies.DATASETS_ROOT,
                self.dependencies.OUTPUTS_ROOT,
                self.app.event_queue,
                session_id=requested_session,
                shutdown_event=self.app.stop_event,
                cancel_event=payload.get("cancel_event") if isinstance(payload, dict) else None,
                initial_frame=payload.get("initial_frame") if isinstance(payload, dict) else None,
                scan_roi=payload.get("scan_roi") if isinstance(payload, dict) else None,
                region_first=payload.get("region_first",False) if isinstance(payload,dict) else False,
            )
        except Exception as error:
            self.app.event_queue.put(
                (
                    "local_error",
                    {
                        "session_id": requested_session,
                        "source": None,
                        "error": str(error),
                    },
                )
            )

    @staticmethod
    def _local_payload(state: SimpleNamespace, payload: Any) -> tuple[bool, Any]:
        if not isinstance(payload, dict):
            return True, payload
        if state.local is None or payload.get("session_id") != state.local.session_id:
            return False, payload
        return True, payload.get("value")

    def _cmd_local_request(self, state: SimpleNamespace, payload: Any) -> None:
        valid, value = self._local_payload(state, payload)
        if valid and state.local is not None:
            state.local.request()

    def _cmd_local_pause(self, state: SimpleNamespace, payload: Any) -> None:
        valid, value = self._local_payload(state, payload)
        if valid and state.local is not None:
            state.local.pause(bool(value))

    def _cmd_local_finish(self, state: SimpleNamespace, payload: Any) -> None:
        valid, value = self._local_payload(state, payload)
        if valid and state.local is not None:
            state.local.finish(bool(value))

    def _cmd_local_check(self, state: SimpleNamespace, payload: Any) -> None:
        valid, _ = self._local_payload(state, payload)
        if valid and state.local is not None:
            state.local.check_current()

    def _cmd_local_check_cancel(self, state: SimpleNamespace, payload: Any) -> None:
        valid, _ = self._local_payload(state, payload)
        if valid and state.local is not None:
            state.local.cancel_check()

    def _cmd_temporal_start(self, state: SimpleNamespace, payload: dict[str, Any]) -> None:
        local_active = state.local is not None and state.local.thread.is_alive()
        temporal_writer_active = (
            getattr(self.app, "temporal_thread", None)
            and self.app.temporal_thread.is_alive()
        )
        if local_active:
            return
        if state.scan is not None or state.temporal_burst is not None or temporal_writer_active:
            self.app.event_queue.put(
                ("temporal_error", {"error": "Another capture is active", "source": None})
            )
            return
        from femto.app.app_temporal import TemporalBurst

        state.temporal_burst = TemporalBurst(
            state.camera,
            int(payload["count"]),
            str(payload["mode"]),
        )
        state.temporal_burst.build_model = bool(payload.get("build_model", False))
        self.app.event_queue.put(
            (
                "temporal_progress",
                {"frames": 0, "total": state.temporal_burst.count},
            )
        )

    def _cmd_temporal_stop(self, state: SimpleNamespace, _payload: Any) -> None:
        if state.temporal_burst is not None:
            self.app._dispatch_temporal(state.temporal_burst, fuse=False)
            state.temporal_burst = None

    def _cmd_photo(self, state: SimpleNamespace, _payload: Any) -> None:
        if state.temporal_burst is not None:
            return
        if state.local is not None and state.local.thread.is_alive():
            return
        with self.app.frame_lock:
            snapshot = (
                self.dependencies.copy_frame(self.app.latest_frame)
                if self.app.latest_frame is not None
                else None
            )
        if snapshot is None:
            self.app.event_queue.put(("error", "相机还没有可保存的 RGB-D 帧。"))
        elif self.app.photo_thread is None or not self.app.photo_thread.is_alive():
            self.app.photo_thread = self.dependencies.threading.Thread(
                target=self.app._photo_worker,
                args=(snapshot, state.camera),
                name="FemtoPhoto",
                daemon=True,
            )
            self.app.photo_thread.start()

    def _cmd_start_scan(self, state: SimpleNamespace, payload: dict[str, Any]) -> None:
        if state.scan is not None:
            return
        if state.local is not None and state.local.thread.is_alive():
            return
        temporal_writer_active = (
            getattr(self.app, "temporal_thread", None)
            and self.app.temporal_thread.is_alive()
        )
        if state.temporal_burst is not None or temporal_writer_active:
            return
        duration_s = float(payload["duration_s"])
        fps = float(payload["fps"])
        processing_mode = str(payload.get("processing_mode", "keyframe"))
        if processing_mode not in self.dependencies.SCAN_MODE_KEYS:
            self.app.event_queue.put(
                ("error", self.app._t("app.capture.single_only"))
            )
            return
        fixed_camera = processing_mode == "keyframe"
        max_frames = max(2, int(round(duration_s * fps)))
        session = self.dependencies.create_session(self.dependencies.DATASETS_ROOT)
        if state.camera.intrinsics is None:
            raise RuntimeError("Camera intrinsics are unavailable")
        self.dependencies.write_calibration(session, state.camera, state.camera.intrinsics)
        now = self.dependencies.time.monotonic()
        state.scan = self.dependencies.ActiveScan(
            session=session,
            frames_log=session / "frames.jsonl",
            started=now,
            next_sample=now,
            duration_s=duration_s,
            fps=fps,
            max_frames=max_frames,
            fixed_camera=fixed_camera,
            processing_mode=processing_mode,
            depth_min_mm=state.camera.min_depth_mm,
            depth_max_mm=state.camera.max_depth_mm,
            reconstruct_after=True,
            next_motion_check=now + self.dependencies.MOTION_CHECK_INTERVAL_S,
        )
        self.app.event_queue.put(
            (
                "scan_started",
                {
                    "session": str(session),
                    "max_frames": max_frames,
                    "fixed_camera": fixed_camera,
                    "processing_mode": processing_mode,
                },
            )
        )

    def _finish_scan(self, state: SimpleNamespace) -> None:
        finished = self.dependencies.finalize_scan(state.scan, state.camera)
        self.app.event_queue.put(
            (
                "scan_finished",
                {
                    "session": str(finished),
                    "frames": state.scan.frame_count,
                    "reconstruct": state.scan.reconstruct_after,
                    "fixed_camera": state.scan.fixed_camera,
                    "processing_mode": state.scan.processing_mode,
                    "capture_quality": state.scan.capture_quality,
                    "motion_state": state.scan.motion_state,
                    "motion_report": state.scan.motion_report,
                },
            )
        )
        state.scan = None

    def _cmd_stop_scan(self, state: SimpleNamespace, payload: dict[str, Any]) -> None:
        if state.scan is None:
            return
        state.scan.reconstruct_after = bool(payload.get("reconstruct", True))
        self.dependencies.harvest_capture_motion(
            state.scan,
            wait_timeout_s=self.dependencies.MOTION_FINAL_WAIT_S,
        )
        self._finish_scan(state)

    def drain_commands(self, state: SimpleNamespace) -> None:
        handlers = {
            "shutdown": self._cmd_shutdown,
            "local_start": self._cmd_local_start,
            "local_request": self._cmd_local_request,
            "local_pause": self._cmd_local_pause,
            "local_finish": self._cmd_local_finish,
            "local_check": self._cmd_local_check,
            "local_check_cancel": self._cmd_local_check_cancel,
            "temporal_start": self._cmd_temporal_start,
            "temporal_stop": self._cmd_temporal_stop,
            "photo": self._cmd_photo,
            "start_scan": self._cmd_start_scan,
            "stop_scan": self._cmd_stop_scan,
        }
        while True:
            try:
                command, payload = self.app.command_queue.get_nowait()
            except queue.Empty:
                break
            handler = handlers.get(command)
            if handler is not None:
                handler(state, payload)
            if command == "shutdown":
                break

    def _process_frame(
        self,
        state: SimpleNamespace,
        frame: Any,
        motion_executor: Any,
    ) -> None:
        with self.app.frame_lock:
            self.app.latest_frame = frame
        if state.local is not None and not state.local.closing.is_set():
            state.local.submit(frame)
        if state.temporal_burst is not None:
            if state.temporal_burst.append(frame, state.camera.intrinsics):
                self.app.event_queue.put(
                    (
                        "temporal_progress",
                        {
                            "frames": len(state.temporal_burst.frames),
                            "total": state.temporal_burst.count,
                        },
                    )
                )
            if state.temporal_burst.complete:
                self.app._dispatch_temporal(state.temporal_burst)
                state.temporal_burst = None
        if state.scan is None:
            return
        now = self.dependencies.time.monotonic()
        if not state.scan.fixed_camera:
            self.dependencies.harvest_capture_motion(state.scan)
        if now >= state.scan.next_sample:
            self.dependencies.save_scan_frame(state.scan, frame)
            state.scan.last_preview = frame.color_bgr.copy()
            state.scan.next_sample += 1.0 / state.scan.fps
            if not state.scan.fixed_camera:
                if state.scan.motion_reference_bgr is None:
                    state.scan.motion_reference_bgr = frame.color_bgr.copy()
                elif now >= state.scan.next_motion_check and state.scan.motion_future is None:
                    state.scan.motion_future = motion_executor.submit(
                        self.dependencies.capture_feedback,
                        state.scan.motion_reference_bgr,
                        frame.color_bgr.copy(),
                        frame.depth_mm.copy(),
                        state.scan.depth_min_mm,
                        state.scan.depth_max_mm,
                        scale=self.dependencies.MOTION_CHECK_SCALE,
                    )
                    state.scan.motion_reference_bgr = frame.color_bgr.copy()
                    state.scan.next_motion_check = (
                        now + self.dependencies.MOTION_CHECK_INTERVAL_S
                    )
            elapsed = now - state.scan.started
            self.app.event_queue.put(
                (
                    "scan_progress",
                    {
                        "frames": state.scan.frame_count,
                        "max_frames": state.scan.max_frames,
                        "elapsed": elapsed,
                        "duration": state.scan.duration_s,
                        "processing_mode": state.scan.processing_mode,
                        "capture_quality": state.scan.capture_quality,
                        "motion_state": state.scan.motion_state,
                    },
                )
            )
        if (
            now - state.scan.started >= state.scan.duration_s
            or state.scan.frame_count >= state.scan.max_frames
        ):
            if not state.scan.fixed_camera:
                self.dependencies.harvest_capture_motion(
                    state.scan,
                    wait_timeout_s=self.dependencies.MOTION_FINAL_WAIT_S,
                )
            self._finish_scan(state)

    def run(self) -> None:
        state = SimpleNamespace(
            camera=None,
            scan=None,
            temporal_burst=None,
            local=None,
        )
        motion_executor = self.dependencies.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="FemtoMotion",
        )
        try:
            state.camera = self.dependencies.OrbbecAlignedCamera(
                self.dependencies.APP_DEPTH_MODE,
                color_size=self.dependencies.APP_COLOR_SIZE,
            ).start()
            first_frame = self.dependencies.wait_for_aligned_frame(state.camera)
            if first_frame is None:
                raise RuntimeError(
                    "Camera started but produced no aligned RGB-D frame before the startup deadline: "
                    + json.dumps(
                        state.camera.read_diagnostics_snapshot(),
                        ensure_ascii=False,
                    )
                )
            with self.app.frame_lock:
                self.app.latest_frame = first_frame
            ready_metadata = dict(state.camera.metadata)
            ready_metadata["_intrinsics"] = state.camera.intrinsics
            ready_metadata["_rgb_distortion"] = dict(
                state.camera.calibration["rgb_distortion"]
            )
            ready_metadata["read_diagnostics"] = (
                state.camera.read_diagnostics_snapshot()
            )
            self.app.event_queue.put(("camera_ready", ready_metadata))
            while not self.app.stop_event.is_set():
                self.drain_commands(state)
                if self.app.stop_event.is_set():
                    break
                if (
                    state.temporal_burst is not None
                    and self.dependencies.time.monotonic()
                    >= state.temporal_burst.deadline
                ):
                    self.app._dispatch_temporal(state.temporal_burst, fuse=False)
                    state.temporal_burst = None
                frame = state.camera.read(1000)
                if frame is None:
                    continue
                self._process_frame(state, frame, motion_executor)
            if state.scan is not None and state.scan.frame_count:
                state.scan.reconstruct_after = False
                if not state.scan.fixed_camera:
                    self.dependencies.harvest_capture_motion(
                        state.scan,
                        wait_timeout_s=self.dependencies.MOTION_FINAL_WAIT_S,
                    )
                self.dependencies.finalize_scan(state.scan, state.camera)
        except Exception as error:
            if (state.local is not None and state.local.thread.is_alive()
                    and not getattr(state.local,'acquisition_complete',False)):
                state.local.abort("camera_error", str(error))
            self.app.event_queue.put(
                (
                    "camera_error",
                    f"{error}\n\n{self.dependencies.traceback.format_exc(limit=4)}",
                )
            )
        finally:
            if (state.local is not None and state.local.thread.is_alive()
                    and (self.app.stop_event.is_set() or not getattr(state.local,'acquisition_complete',False))):
                state.local.finish(False)
            if state.temporal_burst is not None:
                self.app._dispatch_temporal(state.temporal_burst, fuse=False)
            if state.camera is not None:
                state.camera.stop()
            motion_executor.shutdown(wait=False, cancel_futures=True)
            self.app.event_queue.put(("camera_stopped", None))

    def dispatch_temporal(self, burst: Any, *, fuse: bool = True) -> None:
        from femto.app.app_temporal import process_burst, process_keyframe_burst

        target = (
            process_keyframe_burst
            if getattr(burst, "build_model", False)
            else process_burst
        )
        self.app.temporal_thread = self.dependencies.threading.Thread(
            target=target,
            args=(
                burst,
                self.dependencies.DATASETS_ROOT,
                self.app.event_queue,
            ),
            kwargs={"fuse": fuse},
            name="FemtoTemporal",
            daemon=False,
        )
        self.app.temporal_thread.start()

    def photo_worker(self, frame: Any, camera: Any) -> None:
        self.app.event_queue.put(("photo_started", None))
        try:
            output = self.dependencies.save_rgbd_photo(frame, camera)
            self.app.event_queue.put(("photo_saved", str(output)))
        except Exception as error:
            self.app.event_queue.put(("error", f"拍照保存失败：{error}"))

    def reconstruction_worker(self, dataset: Path, processing_mode: str) -> None:
        try:
            if processing_mode not in self.dependencies.SCAN_MODE_KEYS:
                raise ValueError(self.app._t("app.capture.single_only"))
            if processing_mode == "keyframe":
                stamp = self.dependencies.time.strftime("%Y%m%d_%H%M%S")
                reconstruction_dataset, processing_report = (
                    self.dependencies.build_fixed_keyframe_dataset(
                        dataset,
                        dataset / "derived" / f"fixed_keyframe_{stamp}",
                    )
                )
            else:
                manifest = json.loads(
                    (dataset / "manifest.json").read_text(encoding="utf-8")
                )
                if manifest.get("parameters", {}).get("fixed_camera") is not False:
                    raise ValueError(
                        self.app._t("app.capture.moving_requires_pending")
                    )
                reconstruction_dataset, processing_report = dataset, {}
            args = argparse.Namespace(
                mode="latest",
                dataset=reconstruction_dataset,
                datasets_root=self.dependencies.DATASETS_ROOT,
                outputs_root=self.dependencies.OUTPUTS_ROOT,
                synthetic_frames=14,
                no_viewer=True,
            )
            result, summary = self.dependencies.run_demo(args)
            self.app.event_queue.put(
                (
                    "reconstruction_done",
                    {
                        "result": str(result),
                        "reconstruction": summary["reconstruction"],
                        "metrics": summary["reconstruction_metrics"],
                        "processing_mode": processing_mode,
                        "processing_report": processing_report,
                    },
                )
            )
        except Exception as error:
            self.app.event_queue.put(
                (
                    "reconstruction_error",
                    f"{error}\n\n{self.dependencies.traceback.format_exc(limit=5)}",
                )
            )
