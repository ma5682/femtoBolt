"""Open3D live display of the main app's RGB-D stream, with no camera ownership.

Interaction follows the user's E:/测试/05_point_cloud.py viewer. Geometry stays
in the project's calibrated color-camera frame: metres, +X right, +Y down, +Z
forward. Display framing and false colors never modify capture evidence.
"""
from __future__ import annotations
from femto.paths import LOG_ROOT

from dataclasses import asdict
import multiprocessing as mp
import os
from pathlib import Path
import queue
import sys
import threading
import time
import traceback
from typing import Any

import cv2
import numpy as np
import open3d as o3d

from femto.ui_i18n import tr

MIN_DEPTH_M = 0.25
MAX_DEPTH_M = 2.21
PREVIEW_INTERVAL_S = 0.20
GEOMETRY_NAME = "live_rgbd"


class CalibratedCloudBuilder:
    """Cache K6 undistorted rays, retaining every finite in-range pixel."""

    def __init__(self) -> None:
        self._signature = None
        self._rays = None

    def build(self, packet: dict) -> o3d.geometry.PointCloud:
        depth = packet["depth"]
        color = packet["rgb"]
        intr = packet["intrinsics"]
        dist = packet["distortion"]
        if color.shape != (*depth.shape, 3) or depth.shape != (intr["height"], intr["width"]):
            raise ValueError("Live RGB-D dimensions do not match the aligned calibration")
        if "BROWN_CONRADY_K6" not in str(dist.get("model", "")):
            raise ValueError("Live point cloud requires the RGB BROWN_CONRADY_K6 calibration")
        coefficients = tuple(float(dist[k]) for k in ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"))
        signature = (depth.shape, intr["fx"], intr["fy"], intr["cx"], intr["cy"], coefficients)
        if signature != self._signature:
            rows, cols = np.indices(depth.shape, dtype=np.float32)
            pixels = np.stack((cols, rows), axis=-1).reshape(-1, 1, 2)
            matrix = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1]], dtype=np.float64)
            self._rays = cv2.undistortPoints(pixels, matrix, np.asarray(coefficients)).reshape(-1, 2)
            self._signature = signature
        flat = depth.reshape(-1)
        valid = np.isfinite(flat) & (flat >= MIN_DEPTH_M * 1000) & (flat <= MAX_DEPTH_M * 1000)
        z = flat[valid].astype(np.float64) / 1000.0
        xyz = np.column_stack((self._rays[valid] * z[:, None], z))
        finite = np.isfinite(xyz).all(axis=1)
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(xyz[finite])
        cloud.colors = o3d.utility.Vector3dVector(color.reshape(-1, 3)[valid][finite, ::-1].astype(np.float64) / 255.0)
        return cloud


def focus_bounds(cloud: o3d.geometry.PointCloud) -> o3d.geometry.AxisAlignedBoundingBox:
    """Use the most populated depth layer only to select the view target."""
    points = np.asarray(cloud.points)
    if not len(points):
        return o3d.geometry.AxisAlignedBoundingBox()
    selected = points
    if len(points) >= 50:
        counts, edges = np.histogram(points[:, 2], bins=np.arange(MIN_DEPTH_M, MAX_DEPTH_M + 0.05, 0.05))
        occupied = counts >= max(20, int(counts.max() * 0.01))
        starts = np.flatnonzero(np.diff(np.r_[False, occupied, False].astype(int)) == 1)
        ends = np.flatnonzero(np.diff(np.r_[False, occupied, False].astype(int)) == -1)
        if len(starts):
            start, end = max(zip(starts, ends), key=lambda pair: counts[pair[0]:pair[1]].sum())
            candidate = points[(points[:, 2] >= edges[start] - 0.05) & (points[:, 2] <= edges[end] + 0.05)]
            if len(candidate) >= 50:
                selected = candidate
    low, high = np.percentile(selected, [0.5, 99.5], axis=0)
    return o3d.geometry.AxisAlignedBoundingBox(low, np.maximum(high, low + 0.001))


def depth_colors(points: np.ndarray) -> np.ndarray:
    t = np.clip((points[:, 2] - MIN_DEPTH_M) / (MAX_DEPTH_M - MIN_DEPTH_M), 0, 1)
    return np.column_stack((1 - t, 0.25 + 0.55 * (1 - np.abs(t * 2 - 1)), t))


class LivePointCloudPreview:
    """Bounded, nonblocking parent-side transport to a spawned renderer."""

    def __init__(self, language: str) -> None:
        ctx = mp.get_context("spawn")
        self.frames = ctx.Queue(maxsize=1)
        self.events = ctx.Queue(maxsize=8)
        self.stopped = ctx.Event()
        self.language = ctx.Value("i", int(language == "en"))
        self.process = ctx.Process(target=_run_viewer, args=(self.frames, self.events, self.stopped, self.language), name="FemtoLivePointCloud", daemon=True)
        self.last_sent = 0.0
        self.last_frame_key = None
        self.closed = False
        self.process.start()

    def set_language(self, language: str) -> None:
        self.language.value = int(language == "en")

    def submit(self, frame: Any, intrinsics: Any, distortion: dict) -> bool:
        now = time.monotonic()
        key = (frame.frame_index, frame.depth_timestamp_us)
        if self.closed or self.stopped.is_set() or now - self.last_sent < PREVIEW_INTERVAL_S or key == self.last_frame_key:
            return False
        try:
            self.frames.put_nowait({"rgb": frame.color_bgr, "depth": frame.depth_mm, "intrinsics": asdict(intrinsics), "distortion": distortion})
        except queue.Full:
            return False
        self.last_sent = now
        self.last_frame_key = key
        return True

    def poll(self) -> list[tuple[str, Any]]:
        events = []
        while True:
            try:
                events.append(self.events.get_nowait())
            except queue.Empty:
                break
        if not self.process.is_alive() and self.process.exitcode not in (None, 0):
            events.append(("error", f"Open3D preview exited ({self.process.exitcode}); see Log/live_point_cloud.log"))
        return events

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.stopped.set()
        # Never wait for the renderer on the UI thread. Termination is bounded
        # and targets only this instance's child if graceful shutdown fails.
        def reap() -> None:
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=2)
            for channel in (self.frames, self.events):
                channel.cancel_join_thread()
                channel.close()
        threading.Thread(target=reap, name="FemtoPreviewCleanup", daemon=True).start()


class LiveCloudWindow:
    """All Open3D GUI work stays on the child's main thread."""

    def __init__(self, frames, events, stopped, language) -> None:
        from open3d.visualization import gui, rendering

        self.gui, self.rendering = gui, rendering
        self.frames, self.events, self.stopped, self.language_value = frames, events, stopped, language
        self.language = "en" if language.value else "zh"
        self.app = gui.Application.instance
        self.window = self.app.create_window(tr(self.language, "app.point_cloud.title"), 1280, 820)
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        self.scene.scene.set_background([0.035, 0.05, 0.07, 1.0])
        self.scene.scene.show_axes(False)
        self.scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA_SPHERE)
        self.window.add_child(self.scene)
        em = self.window.theme.font_size
        self.panel = gui.Vert(0.65 * em, gui.Margins(em, em, em, em))
        self.labels = {}
        for key in ("title", "profile", "stats"):
            label = gui.Label("")
            self.labels[key] = label
            self.panel.add_child(label)
        self.buttons = {}
        for key, action in (("color", self.toggle_color), ("sensor", self.sensor_view), ("fit_all", self.fit_all)):
            button = gui.Button("")
            button.set_on_clicked(action)
            self.buttons[key] = button
            self.panel.add_child(button)
        self.point_size_label = gui.Label("")
        self.panel.add_child(self.point_size_label)
        self.point_size = gui.Slider(gui.Slider.INT)
        self.point_size.set_limits(1, 6)
        self.point_size.int_value = 2
        self.point_size.set_on_value_changed(self.change_point_size)
        self.panel.add_child(self.point_size)
        axes_row = gui.Horiz(0.5 * em)
        self.axes = gui.Checkbox("")
        self.axes.set_on_checked(lambda checked: self.scene.scene.show_axes(checked))
        axes_row.add_child(self.axes)
        self.axes_label = gui.Label("")
        axes_row.add_child(self.axes_label)
        self.panel.add_child(axes_row)
        self.hint = gui.Label("")
        self.panel.add_child(self.hint)
        self.window.add_child(self.panel)
        self.material = rendering.MaterialRecord()
        self.material.shader = "defaultUnlit"
        self.material.point_size = 2.0
        self.use_rgb = True
        self.cloud = None
        self.rgb = None
        self.focus = None
        self.geometry_added = False
        self.initial_fit = True
        self.close_requested = False
        self.pending = None
        self.pending_error = None
        self.lock = threading.Lock()
        self.last_data = 0.0
        self.frame_count = 0
        self.started = time.monotonic()
        self.stats = None
        self.waiting = True
        self.worker = threading.Thread(target=self._convert_frames, name="LiveCloudGeometry", daemon=True)
        self.window.set_on_layout(self._layout)
        self.window.set_on_close(self._close)
        self.window.set_on_tick_event(self._tick)
        self.apply_language()
        self.worker.start()

    def emit(self, kind: str, value: Any) -> None:
        try:
            self.events.put_nowait((kind, value))
        except queue.Full:
            pass

    def apply_language(self) -> None:
        t = lambda key: tr(self.language, "app.point_cloud." + key)
        self.window.title = t("title")
        self.labels["title"].text = t("title")
        self.labels["profile"].text = "WFOV_UNBINNED 1024×1024@15\n0.25–2.21 m"
        self.buttons["color"].text = t("rgb" if self.use_rgb else "depth")
        self.buttons["sensor"].text = t("sensor")
        self.buttons["fit_all"].text = t("fit_all")
        self.point_size_label.text = t("point_size")
        self.axes_label.text = t("axes")
        self.hint.text = t("controls")
        self._update_stats()

    def _update_stats(self) -> None:
        key = "app.point_cloud.waiting" if self.waiting else "app.point_cloud.stats"
        self.labels["stats"].text = tr(self.language, key, **(self.stats or {}))

    def _layout(self, _context) -> None:
        rect = self.window.content_rect
        side = min(24 * self.window.theme.font_size, int(rect.width * 0.38))
        self.scene.frame = self.gui.Rect(rect.x, rect.y, max(1, rect.width - side), rect.height)
        self.panel.frame = self.gui.Rect(rect.get_right() - side, rect.y, side, rect.height)

    def _convert_frames(self) -> None:
        builder = CalibratedCloudBuilder()
        while not self.stopped.is_set():
            try:
                packet = self.frames.get(timeout=0.10)
            except queue.Empty:
                continue
            try:
                cloud = builder.build(packet)
                bounds = focus_bounds(cloud)
                with self.lock:
                    self.pending = (cloud, bounds)
            except Exception:
                with self.lock:
                    self.pending_error = traceback.format_exc()
                return

    def _tick(self) -> bool:
        if self.stopped.is_set():
            # Let conversion finish and leave this tick before destroying the
            # native window. Closing it inside its own tick can invalidate the
            # renderer callback on Windows/Open3D 0.19.
            if not self.worker.is_alive() and not self.close_requested:
                self.close_requested = True
                self.app.post_to_main_thread(self.window, self.app.quit)
            return False
        try:
            return self._refresh()
        except Exception:
            self.emit("error", traceback.format_exc())
            self.stopped.set()
            return False

    def _refresh(self) -> bool:
        dirty = False
        language = "en" if self.language_value.value else "zh"
        if self.language != language:
            self.language = language
            self.apply_language()
            dirty = True
        with self.lock:
            pending, self.pending = self.pending, None
            error = self.pending_error
        if error:
            raise RuntimeError(error)
        if pending is not None:
            self.cloud, self.focus = pending
            self.rgb = np.asarray(self.cloud.colors).copy()
            self.last_data = time.monotonic()
            self.waiting = False
            self.frame_count += 1
            self._show_cloud()
            count = len(self.cloud.points)
            self.stats = {"points": count, "fps": self.frame_count / max(self.last_data - self.started, 0.001)}
            self._update_stats()
            if count and self.initial_fit:
                self.sensor_view()
                self.initial_fit = False
            self.emit("frame", {**self.stats, "unit": "metres"})
            dirty = True
        elif not self.waiting and time.monotonic() - self.last_data > 2.0:
            self.waiting = True
            self._update_stats()
            dirty = True
        return dirty

    def _show_cloud(self) -> None:
        if self.geometry_added:
            self.scene.scene.remove_geometry(GEOMETRY_NAME)
            self.geometry_added = False
        if self.cloud is not None and len(self.cloud.points):
            colors = self.rgb if self.use_rgb else depth_colors(np.asarray(self.cloud.points))
            self.cloud.colors = o3d.utility.Vector3dVector(colors)
            # Disable Open3D's optional fast downsampled interaction copy.
            self.scene.scene.add_geometry(GEOMETRY_NAME, self.cloud, self.material, False)
            self.geometry_added = True

    def toggle_color(self) -> None:
        self.use_rgb = not self.use_rgb
        self._show_cloud()
        self.apply_language()

    def change_point_size(self, value: float) -> None:
        self.material.point_size = float(value)
        if self.geometry_added:
            self.scene.scene.modify_geometry_material(GEOMETRY_NAME, self.material)

    def sensor_view(self) -> None:
        if self.cloud is None or not len(self.cloud.points):
            return
        self.scene.setup_camera(60.0, self.cloud.get_axis_aligned_bounding_box(), self.focus.get_center())
        self.scene.look_at(self.focus.get_center().astype(np.float32), np.zeros(3, np.float32), np.array([0, -1, 0], np.float32))
        # setup_camera fits an orbit view; restore clipping for an eye at origin.
        self.scene.scene.camera.set_projection(60.0, max(self.scene.frame.width, 1) / max(self.scene.frame.height, 1), 0.01, 10.0, self.rendering.Camera.FovType.Vertical)
        self.window.post_redraw()

    def fit_all(self) -> None:
        if self.cloud is None or not len(self.cloud.points):
            return
        bounds = self.cloud.get_axis_aligned_bounding_box()
        center = bounds.get_center()
        aspect = max(self.scene.frame.width, 1) / max(self.scene.frame.height, 1)
        extents = bounds.get_extent()
        distance = 1.15 * max(extents[1], extents[0] / aspect, 0.02) / (2 * np.tan(np.deg2rad(30))) + extents[2] / 2
        self.scene.setup_camera(60.0, bounds, center)
        self.scene.look_at(center.astype(np.float32), (center - [0, 0, distance]).astype(np.float32), np.array([0, -1, 0], np.float32))
        self.window.post_redraw()

    def _close(self) -> bool:
        self.stopped.set()
        # Defer an OS close request through the same worker/renderer teardown.
        return self.close_requested


def _run_viewer(frames, events, stopped, language) -> None:
    log = LOG_ROOT / "live_point_cloud.log"
    log.parent.mkdir(exist_ok=True)
    with log.open("a", encoding="utf-8", buffering=1) as stream:
        if sys.stdout is None:
            sys.stdout = stream
        if sys.stderr is None:
            sys.stderr = stream
        window = None
        try:
            from open3d.visualization import gui
            from femto.viewer.scene_demo_viewer import _open3d_ascii_resources
            app = gui.Application.instance
            app.initialize(str(_open3d_ascii_resources()))
            font = gui.FontDescription("sans-serif")
            font.add_typeface_for_language(str(Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/msyh.ttc"), "zh")
            app.set_font(gui.Application.DEFAULT_FONT_ID, font)
            window = LiveCloudWindow(frames, events, stopped, language)
            window.emit("ready", None)
            app.run()
        except Exception:
            error = traceback.format_exc()
            stream.write(error + "\n")
            try:
                events.put_nowait(("error", error))
            except queue.Full:
                pass
        finally:
            stopped.set()
            if window is not None:
                window.worker.join(timeout=2)
            # Drain the one pending input so the parent's Queue feeder can
            # finish its write and release its pipe on close/reopen.
            while True:
                try:
                    frames.get(timeout=0.10)
                except (queue.Empty, EOFError, OSError):
                    break
