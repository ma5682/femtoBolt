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
            if state.local is not None and state.local.thread.is_alive():
                state.local.abort("camera_error", str(error))
            self.app.event_queue.put(
                (
                    "camera_error",
                    f"{error}\n\n{self.dependencies.traceback.format_exc(limit=4)}",
                )
            )
        finally:
            if state.local is not None and state.local.thread.is_alive():
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

    def handle_status_event(self, event: str, payload: Any) -> bool:
        app = self.app
        if event.startswith("local_"):
            app.local_panel.event(event, payload)
        elif event == "camera_ready":
            app.camera_ready = True
            app.device_info = dict(payload)
            app.camera_intrinsics = app.device_info.pop("_intrinsics", None)
            app.camera_rgb_distortion = app.device_info.pop(
                "_rgb_distortion",
                None,
            )
            app._update_device_label()
            app._set_status("app.camera_ready")
            app._set_detail("app.camera_ready_detail")
        elif event.startswith("temporal_"):
            if event == "temporal_progress":
                app.progress_var.set(100 * payload["frames"] / payload["total"])
                app._status_token = None
                app.status_var.set(
                    app._t("app.temporal")
                    + f" {payload['frames']}/{payload['total']}"
                )
            elif event == "temporal_processing":
                app.temporal_collecting = False
                app._status_token = None
                app.status_var.set(app._t("app.temporal_processing"))
            elif event in ("temporal_finished", "temporal_error"):
                app.temporal_busy = False
                app.temporal_collecting = False
                app.processing = False
                app.keyframe_building = False
                app.latest_temporal = (
                    {
                        "status": "error",
                        "output": payload.get("source"),
                        "error": payload["error"],
                    }
                    if event == "temporal_error"
                    else payload
                )
                app._status_token = None
                app.status_var.set(
                    app._t("app.temporal")
                    + " · "
                    + payload.get("status", "error")
                )
                app._set_raw_detail(
                    payload.get("output") or payload.get("error", "")
                )
                app.progress_var.set(
                    100 if payload.get("status") == "completed" else 0
                )
            if (
                app.temporal_panel is not None
                and app.temporal_panel.window.winfo_exists()
            ):
                app.temporal_panel.refresh(event, payload)
        elif event == "keyframe_stage":
            app.temporal_collecting = False
            app._set_action_state()
            app._status_token = None
            labels = {
                "saving": "正在保存原始RGB-D…",
                "quality": "Depth Quality / Hole Filling…",
                "fusion": "正在融合：Temporal Fusion…",
                "cloud": "正在生成点云…",
                "geometry_inputs": "正在生成几何及3DGS输入（不启动训练）…",
            }
            app.status_var.set(labels[payload])
        elif event == "keyframe_stopped":
            app.processing = False
            app.temporal_busy = False
            app.temporal_collecting = False
            app.keyframe_building = False
            app._status_token = None
            app.status_var.set("采集已停止")
            app._set_raw_detail("采集已停止，原始证据保留：" + payload)
        elif event == "photo_started":
            app.photo_saving = True
            app._set_status("app.photo_saving")
        elif event == "photo_saved":
            app.photo_saving = False
            app.latest_photo = Path(payload)
            app._set_status("app.photo_saved")
            app._set_raw_detail(str(app.latest_photo))
        elif event == "single_gs_progress":
            app.progress_var.set(100 * payload["step"] / payload["total"])
            app._set_detail(
                "app.single_progress",
                **{
                    key: payload[key]
                    for key in (
                        "step",
                        "total",
                        "gaussians",
                        "cuda_reserved_mib",
                        "cuda_free_mib",
                    )
                },
            )
        elif event == "single_gs_done":
            app.processing = False
            app.latest_single_gs = Path(payload)
            app.progress_var.set(100)
            app._set_status("app.single_done")
            app._set_raw_detail(str(app.latest_single_gs))
        elif event == "single_gs_error":
            app.processing = False
            app._set_status("app.single_failed")
            app._set_raw_detail(str(payload))
        else:
            return False
        return True

    def handle_reconstruction_done(self, payload: dict[str, Any]) -> None:
        app = self.app
        app.temporal_busy = False
        app.temporal_collecting = False
        app.keyframe_building = False
        app.processing = False
        app.latest_result = Path(payload["result"])
        from femto.pipeline.reconstruction_sources import remember

        source_mode = remember(self.dependencies.OUTPUTS_ROOT, app.latest_result)
        if source_mode == "keyframe":
            app.latest_keyframe_result = app.latest_result
        elif source_mode == "moving":
            app.latest_moving_result = app.latest_result
        if payload.get("source_dataset"):
            app.latest_dataset = Path(payload["source_dataset"])
        metrics = payload["metrics"]
        processing_report = payload.get("processing_report") or {}
        depth_ratio = processing_report.get("selected_valid_depth_ratio")
        too_near_ratio = processing_report.get("selected_too_near_depth_ratio")
        no_return_ratio = processing_report.get("selected_no_depth_return_ratio")
        if depth_ratio is not None and depth_ratio < 0.35:
            if too_near_ratio is not None and too_near_ratio >= 0.08:
                app._set_status("app.result.too_near")
            elif no_return_ratio is not None and no_return_ratio >= 0.35:
                app._set_status("app.result.no_return")
            else:
                app._set_status("app.result.low_coverage")
        else:
            app._set_status("app.result.done")
        app._set_detail(
            "app.result.metrics",
            points=metrics["fused_point_count"],
            triangles=metrics["mesh_triangles"],
            depth_value=depth_ratio,
            near_value=too_near_ratio,
            none_value=no_return_ratio,
            path=app.latest_result,
        )

    def handle_scan_event(self, event: str, payload: dict[str, Any]) -> None:
        app = self.app
        if event == "scan_started":
            app.capturing = True
            app.latest_result = None
            app.latest_dataset = Path(payload["session"])
            processing_mode = payload.get("processing_mode", "keyframe")
            app._set_status(f"app.scan_started.{processing_mode}")
            app._set_detail(
                "app.target_frames",
                frames=payload["max_frames"],
                path=app.latest_dataset,
            )
            app.progress_var.set(0.0)
        elif event == "scan_progress":
            by_time = payload["elapsed"] / max(payload["duration"], 1e-6)
            by_frames = payload["frames"] / max(payload["max_frames"], 1)
            app.progress_var.set(
                min(100.0, max(by_time, by_frames) * 100.0)
            )
            remaining = max(0.0, payload["duration"] - payload["elapsed"])
            detail_key = "app.scan_progress"
            detail_values: dict[str, Any] = {
                "frames": payload["frames"],
                "maximum": payload["max_frames"],
                "remaining": remaining,
            }
            if payload.get("processing_mode") == "moving":
                detail_key = "app.scan_progress.moving"
                detail_values["motion"] = app._t(
                    "app.camera_motion."
                    + payload.get("motion_state", "unknown")
                )
                quality = payload.get("capture_quality") or {}
                detail_values["motion"] += " · " + app._t(
                    "app.capture_quality."
                    + quality.get("guidance", "pending")
                )
            app._set_detail(detail_key, **detail_values)
        elif event == "scan_finished":
            app.capturing = False
            app.latest_dataset = Path(payload["session"])
            app.progress_var.set(100.0)
            if payload["frames"] < 2:
                app._set_status("app.scan_too_short")
                app._set_raw_detail(str(app.latest_dataset))
            elif payload.get("reconstruct", True):
                processing_mode = str(
                    payload.get("processing_mode", "keyframe")
                )
                if (
                    processing_mode == "moving"
                    and payload.get("motion_state") == "not_moved"
                ):
                    self.dependencies.messagebox.showwarning(
                        app._t("app.dialog.camera_motion_title"),
                        app._t("app.dialog.camera_motion_body"),
                    )
                app.processing = True
                app._set_status(f"app.reconstruct.{processing_mode}")
                app._set_detail(f"app.reconstruct_detail.{processing_mode}")
                app.reconstruction_thread = self.dependencies.threading.Thread(
                    target=app._reconstruction_worker,
                    args=(app.latest_dataset, processing_mode),
                    name="FemtoReconstruction",
                    daemon=True,
                )
                app.reconstruction_thread.start()

    def poll_events(self) -> None:
        app = self.app
        if not app.root.winfo_exists():
            return
        app._poll_point_cloud_preview()
        while True:
            try:
                event, payload = app.event_queue.get_nowait()
            except queue.Empty:
                break
            if app._handle_status_event(event, payload):
                app._set_action_state()
                continue
            if event in {"scan_started", "scan_progress", "scan_finished"}:
                self.handle_scan_event(event, payload)
            elif event == "reconstruction_done":
                app._handle_reconstruction_done(payload)
            elif event == "reconstruction_error":
                app.temporal_busy = False
                app.temporal_collecting = False
                app.keyframe_building = False
                app.processing = False
                app._set_status("app.reconstruct_failed")
                app._set_raw_detail(str(payload).split("\n", 1)[0])
                self.dependencies.messagebox.showwarning(
                    app._t("app.dialog.reconstruct_title"),
                    app._t(
                        "app.dialog.reconstruct_body",
                        error=str(payload).split("\n\n", 1)[0],
                    ),
                )
            elif event == "3dgs_ready":
                app._set_status("app.3dgs_training")
                app._set_detail("app.3dgs_training_detail", **payload)
            elif event == "3dgs_progress":
                app._set_detail(
                    "app.3dgs_progress",
                    step=payload["step"],
                    total=payload["total"],
                    gaussians=payload["gaussians"],
                    loss=payload["loss"],
                    elapsed=payload["elapsed_s"],
                    reserved=payload.get("cuda_reserved_mib", 0),
                    free=payload.get("cuda_free_mib", 0),
                    cap=payload.get("max_gaussians", 0),
                )
            elif event == "3dgs_done":
                app.processing = False
                app.latest_gsplat_model = Path(payload["output"])
                app._set_status(
                    "app.3dgs_done"
                    if payload["accepted"]
                    else "app.3dgs_rejected"
                )
                app._set_detail("app.3dgs_done_detail", **payload["detail"])
            elif event == "3dgs_error":
                app.processing = False
                app._set_status("app.3dgs_failed")
                failure_message = str(payload).split("\n\nTraceback", 1)[0]
                app._set_raw_detail(failure_message)
                self.dependencies.messagebox.showwarning(
                    app._t("app.dialog.3dgs_not_ready"),
                    failure_message,
                )
            elif event in {"error", "camera_error"}:
                app.photo_saving = False
                if event == "camera_error":
                    app.camera_ready = False
                    app._set_status("app.camera_failed")
                app._set_raw_detail(str(payload).split("\n", 1)[0])
                self.dependencies.messagebox.showerror(
                    "Femto Bolt",
                    str(payload).split("\n\n", 1)[0],
                )
            elif event == "camera_stopped":
                app.camera_ready = False
                writer_active = (
                    app.temporal_thread and app.temporal_thread.is_alive()
                )
                if app.temporal_busy and not writer_active:
                    app.temporal_busy = False
                    app.temporal_collecting = False
                    app.processing = False
                    app.keyframe_building = False
                    if (
                        app.temporal_panel is not None
                        and app.temporal_panel.window.winfo_exists()
                    ):
                        app.temporal_panel.refresh(
                            "temporal_error",
                            {
                                "error": "Camera stopped before burst completion",
                                "source": None,
                            },
                        )
            app._set_action_state()
        app.root.after(80, app._poll_events)

    def toggle_scan(self) -> None:
        app = self.app
        if getattr(app, "local_busy", False):
            return
        if getattr(app, "keyframe_building", False):
            if app.temporal_collecting:
                app.command_queue.put(("temporal_stop", None))
            return
        if app.capturing:
            app.command_queue.put(("stop_scan", {"reconstruct": True}))
            return
        if not app.camera_ready or app.processing:
            return
        if app.scan_mode_var.get() == "keyframe" and hasattr(
            app,
            "keyframe_quality_var",
        ):
            if app.photo_saving:
                return
            names = (
                "快速模式 1帧",
                "标准模式 5帧",
                "高质量模式 10帧",
                "精细模式 15帧",
            )
            count = (1, 5, 10, 15)[names.index(app.keyframe_quality_var.get())]
            app.processing = True
            app.temporal_busy = True
            app.temporal_collecting = True
            app.keyframe_building = True
            app.latest_result = None
            app.progress_var.set(0)
            app._set_action_state()
            app.command_queue.put(
                (
                    "temporal_start",
                    {
                        "count": count,
                        "mode": "quality",
                        "build_model": True,
                    },
                )
            )
            return
        if app.scan_mode_var.get() not in self.dependencies.SCAN_MODE_KEYS:
            self.dependencies.messagebox.showinfo(
                "Femto Bolt",
                app._t("app.capture.single_only"),
            )
            return
        try:
            duration = float(app.duration_var.get())
            fps = float(app.capture_fps_var.get())
        except (TypeError, ValueError, self.dependencies.tk.TclError):
            self.dependencies.messagebox.showerror(
                app._t("app.dialog.parameter_title"),
                app._t("app.dialog.parameter_number"),
            )
            return
        if not 3.0 <= duration <= 30.0 or not 1.0 <= fps <= 15.0:
            self.dependencies.messagebox.showerror(
                app._t("app.dialog.parameter_title"),
                app._t("app.dialog.parameter_range"),
            )
            return
        app.command_queue.put(
            (
                "start_scan",
                {
                    "duration_s": duration,
                    "fps": fps,
                    "processing_mode": app.scan_mode_var.get(),
                },
            )
        )
