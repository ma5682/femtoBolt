"""Single-window Femto Bolt RGB-D capture and reconstruction control panel."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import traceback
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import open3d as o3d
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox, ttk

from femto.pipeline.complete_demo import run_demo
from femto.viewer.live_point_cloud import LivePointCloudPreview
from femto.gsplat.gsplat_backend import (
    PROFILE_KEYS as THREE_DGS_PROFILE_KEYS,
    PROFILES,
    assess_training_input,
    latest_trained_model,
    open_viewer as open_gsplat_viewer,
    runtime_status as gsplat_runtime_status,
    train as train_direct_gsplat,
)
from femto.io_utils import write_json
from femto.hardware.orbbec_rgbd import AlignedFrame, DEPTH_MODE_SPECS, OrbbecAlignedCamera
from femto.temporal.scene_keyframe import build_fixed_keyframe_dataset
from femto.capture.scene_capture import create_session, depth_preview, enable_windows_dpi_awareness
from femto.dataset.calibration_files import write_calibration
from femto.tracking.scene_pose_estimator import assess_rgb_motion_pair
from femto.tracking.moving_quality import capture_feedback
from femto.ui_i18n import LANGUAGE_NAMES, load_language, save_language, tr
from femto.paths import DATASETS_ROOT, LOG_ROOT, OUTPUTS_ROOT, PHOTOS_ROOT, REPO_ROOT as ROOT
# Capture-session lifecycle lives in femto.capture.scan_session; these names stay importable
# from this module for tests and external callers written against the flat layout.
from femto.capture.scan_session import (  # noqa: F401
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


SCAN_MODE_KEYS = ("keyframe", "moving")
# One native unbinned depth profile for every capture entry in this app.
APP_DEPTH_MODE = "wide_detail"
APP_COLOR_SIZE = (1920, 1080)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Femto Bolt all-in-one RGB-D control panel")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Connect, save one RGB-D photo and exit without opening the GUI",
    )
    return parser.parse_args()


class FemtoControlApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.language = load_language()
        self.root.title(tr(self.language, "app.window_title"))
        self.root.geometry("1600x900")
        self.root.minsize(1280, 720)
        self.root.configure(bg="#0f141c")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.command_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.event_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.frame_lock = threading.Lock()
        self.latest_frame: AlignedFrame | None = None
        self.stop_event = threading.Event()
        self.camera_thread: threading.Thread | None = None
        self.reconstruction_thread: threading.Thread | None = None
        self.photo_thread: threading.Thread | None = None
        self.temporal_thread: threading.Thread | None = None
        self.temporal_busy = False
        self.keyframe_building = False
        self.keyframe_quality_var = tk.StringVar(value='高质量模式 10帧')
        self.temporal_collecting = False
        self.temporal_panel = None
        self.latest_temporal = None
        self.camera_ready = False
        self.local_busy = False
        self.capturing = False
        self.processing = False
        self.photo_saving = False
        self.latest_photo: Path | None = None
        self.latest_dataset: Path | None = None
        self.latest_result: Path | None = None
        self.latest_keyframe_result: Path | None = None
        self.latest_moving_result: Path | None = None
        self.latest_gsplat_model: Path | None = None
        self.latest_single_gs: Path | None = None
        self.device_info: dict[str, Any] | None = None
        self.camera_intrinsics: Any | None = None
        self.preview_photo: ImageTk.PhotoImage | None = None
        self.point_cloud_preview: LivePointCloudPreview | None = None
        self.camera_rgb_distortion: dict | None = None
        self.preview_counter = 0
        self.preview_started = time.monotonic()

        self._status_token: tuple[str, dict[str, Any]] | None = (
            "app.connecting",
            {},
        )
        self._detail_token: tuple[str, dict[str, Any]] | None = (
            "app.connecting_detail",
            {},
        )
        self.status_var = tk.StringVar(value=tr(self.language, "app.connecting"))
        self.detail_var = tk.StringVar(value=tr(self.language, "app.connecting_detail"))
        self.device_var = tk.StringVar(value=tr(self.language, "app.device_waiting"))
        self.frame_var = tk.StringVar(value=tr(self.language, "app.preview_zero"))
        self.language_display_var = tk.StringVar(value=LANGUAGE_NAMES[self.language])
        self.duration_var = tk.DoubleVar(value=10.0)
        self.capture_fps_var = tk.DoubleVar(value=6.0)
        self.scan_mode_var = tk.StringVar(value="keyframe")
        self.three_dgs_profile_var = tk.StringVar(value="low_vram")
        self.scan_mode_display_var = tk.StringVar(
            value=tr(self.language, "app.scan.keyframe")
        )
        self.three_dgs_profile_display_var = tk.StringVar(
            value=tr(self.language, "app.3dgs_profile.low_vram")
        )
        self.progress_var = tk.DoubleVar(value=0.0)

        self._load_latest_paths()
        self._build_ui()
        self._set_action_state()
        self._start_camera()
        self.root.after(30, self._update_preview)
        self.root.after(80, self._poll_events)

    def _load_latest_paths(self) -> None:
        from femto.pipeline.reconstruction_sources import restore
        sources = restore(OUTPUTS_ROOT)
        self.latest_keyframe_result = sources['keyframe']
        self.latest_moving_result = sources['moving']
        from femto.gsplat.scene_single_gs import MARKER
        try:
            from femto.gsplat.scene_single_gs_viewer import load_model_metadata
            candidate = Path(MARKER.read_text(encoding='utf-8').strip())
            load_model_metadata(candidate)
            self.latest_single_gs = candidate
        except (OSError, ValueError, KeyError):
            self.latest_single_gs = None
        markers = (
            (PHOTOS_ROOT / "latest_photo.txt", "photo"),
            (DATASETS_ROOT / "latest_scene.txt", "dataset"),
            (OUTPUTS_ROOT / "latest_complete_demo.txt", "reconstruction"),
        )
        for marker, kind in markers:
            try:
                value = Path(marker.read_text(encoding="utf-8-sig").strip()).resolve()
            except (OSError, ValueError):
                continue
            if not value.exists():
                continue
            if kind == "photo":
                self.latest_photo = value
            elif kind == "dataset":
                self.latest_dataset = value
            elif kind == "reconstruction" and (value / "reconstruction_report.json").is_file():
                self.latest_result = value.parent
        try:
            self.latest_gsplat_model = latest_trained_model().output_dir
        except (OSError, ValueError, json.JSONDecodeError):
            self.latest_gsplat_model = None

    def _build_advanced_shell(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "TNotebook",
            background="#0f141c",
            borderwidth=0,
            tabmargins=(0, 0, 0, 0),
        )
        style.configure(
            "TNotebook.Tab",
            background="#161d27",
            foreground="#9fb2c4",
            borderwidth=0,
            padding=(18, 8),
            font=("Microsoft YaHei UI", 9),
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", "#1d3547"), ("active", "#203c50")],
            foreground=[("selected", "#eaf7fb"), ("active", "#eaf7fb")],
        )
        style.configure(
            "Dark.Horizontal.TProgressbar",
            troughcolor="#1b2734",
            background="#06d6a0",
            bordercolor="#1b2734",
            lightcolor="#06d6a0",
            darkcolor="#06d6a0",
        )
        self.workbench = ttk.Notebook(self.root)
        self.workbench.pack(fill=tk.BOTH, expand=True)
        local_host = tk.Frame(self.workbench, bg='#0f141c')
        legacy_host = tk.Frame(self.workbench, bg='#0f141c')
        self.workbench.add(local_host, text='局部移动补采 / Local capture')
        self.workbench.add(legacy_host, text='固定采集与高级功能 / Advanced')
        from femto.app.app_local_panel import LocalPanel
        self.local_panel = LocalPanel(self, local_host)
        shell = tk.Frame(legacy_host, bg="#0f141c")
        shell.pack(fill=tk.BOTH, expand=True, padx=18, pady=18)
        shell.grid_columnconfigure(0, weight=1)
        shell.grid_columnconfigure(1, minsize=360)
        shell.grid_rowconfigure(1, weight=1)

        header = tk.Frame(shell, bg="#0f141c")
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 14))
        self.heading_label = tk.Label(
            header,
            text=tr(self.language, "app.heading"),
            bg="#0f141c",
            fg="#f2f7fb",
            font=("Microsoft YaHei UI", 19, "bold"),
        )
        self.heading_label.pack(anchor="w")
        tk.Label(
            header,
            textvariable=self.device_var,
            bg="#0f141c",
            fg="#8fa8bf",
            font=("Microsoft YaHei UI", 10),
        ).pack(anchor="w", pady=(5, 0))

        preview_frame = tk.Frame(shell, bg="#05080c", highlightbackground="#26394d", highlightthickness=1)
        preview_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 16))
        self.preview_label = tk.Label(
            preview_frame,
            text=tr(self.language, "app.camera_starting"),
            bg="#05080c",
            fg="#72869a",
            font=("Microsoft YaHei UI", 16),
        )
        self.preview_label.pack(fill=tk.BOTH, expand=True)

        panel_host = tk.Frame(shell, bg="#111a24")
        panel_host.grid(row=1, column=1, sticky="nsew")
        self.panel_canvas = tk.Canvas(panel_host, bg="#111a24", highlightthickness=0,
                                      width=500, height=1, yscrollincrement=24)
        self.panel_scrollbar = ttk.Scrollbar(panel_host, orient=tk.VERTICAL,
                                             command=self.panel_canvas.yview)
        self.panel_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.panel_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.panel_canvas.configure(yscrollcommand=self.panel_scrollbar.set)
        panel = tk.Frame(self.panel_canvas, bg="#111a24", padx=12, pady=10)
        panel_window = self.panel_canvas.create_window((0, 0), window=panel, anchor='nw')
        panel.bind('<Configure>', lambda event: self.panel_canvas.configure(
            scrollregion=self.panel_canvas.bbox('all'), width=panel.winfo_reqwidth()))
        self.panel_canvas.bind('<Configure>', lambda event: self.panel_canvas.itemconfigure(
            panel_window, width=event.width))
        panel_header = tk.Frame(panel, bg="#111a24")
        panel_header.pack(fill=tk.X)
        self.panel_label = tk.Label(
            panel_header,
            text=tr(self.language, "app.panel"),
            bg="#111a24",
            fg="#f2f7fb",
            font=("Microsoft YaHei UI", 16, "bold"),
        )
        self.panel_label.pack(side=tk.LEFT, anchor="w")
        self.language_combo = ttk.Combobox(
            panel_header,
            textvariable=self.language_display_var,
            values=[LANGUAGE_NAMES["zh"], LANGUAGE_NAMES["en"]],
            state="readonly",
            width=9,
            font=("Microsoft YaHei UI", 9),
        )
        self.language_combo.pack(side=tk.RIGHT, anchor="e", pady=(2, 0))
        self.language_combo.bind("<<ComboboxSelected>>", self._language_changed)
        tk.Label(
            panel,
            textvariable=self.frame_var,
            bg="#111a24",
            fg="#8fa8bf",
            font=("Microsoft YaHei UI", 10),
        ).pack(anchor="w", pady=(2, 8))
        return panel

    def _model_dependencies(self) -> SimpleNamespace:
        return SimpleNamespace(
            PROFILES=PROFILES,
            ROOT=ROOT,
            THREE_DGS_PROFILE_KEYS=THREE_DGS_PROFILE_KEYS,
            assess_training_input=assess_training_input,
            gsplat_runtime_status=gsplat_runtime_status,
            write_json=write_json,
            messagebox=messagebox,
            open_gsplat_viewer=open_gsplat_viewer,
            subprocess=subprocess,
            threading=threading,
            traceback=traceback,
            train_direct_gsplat=train_direct_gsplat,
        )

    def _model_service(self):
        from femto.app.app_model_panel import ModelPanel

        dependencies = self._model_dependencies()
        if not hasattr(self, "model_panel"):
            self.model_panel = ModelPanel(self, dependencies)
        else:
            self.model_panel.dependencies = dependencies
        return self.model_panel

    def _build_model_controls(self, panel):
        self._model_service().build(panel)

    def _build_ui(self) -> None:
        panel = self._build_advanced_shell()

        photo_actions = tk.Frame(panel,bg='#111a24')
        photo_actions.pack(fill=tk.X,pady=(0,6))
        photo_actions.grid_columnconfigure(0,weight=1)
        self.photo_button = self._button(
            photo_actions, tr(self.language, "app.photo"), self.take_photo, "#2467b2", "#2d78cc"
        )
        self.photo_button.grid(row=0,column=0,sticky='ew')
        self.temporal_entry_button = self._button(photo_actions, '固定时域融合 / Temporal fusion',
            self.open_temporal_panel, '#247a8f', '#2b91aa')
        self.temporal_entry_button.grid(row=1,column=0,sticky='ew',pady=(4,0))


        self.point_cloud_button = self._button(
            panel,
            tr(self.language, "app.point_cloud.open"),
            self.toggle_point_cloud_preview,
            "#247a8f",
            "#2b91aa",
        )
        self.point_cloud_button.pack(fill=tk.X, pady=(0, 6))

        self.depth_mode_label = tk.Label(
            panel, text=tr(self.language, "app.depth.fixed"),
            bg="#111a24", fg="#a9bdcf", anchor="w", justify=tk.LEFT,
            font=("Microsoft YaHei UI", 9),
        )
        self.depth_mode_label.pack(fill=tk.X, pady=(0, 6))

        settings = tk.Frame(panel, bg="#172330", padx=10, pady=8)
        settings.pack(fill=tk.X, pady=(0, 6))
        self.capture_mode_label = tk.Label(
            settings,
            text=tr(self.language, "app.capture_mode"),
            bg="#172330",
            fg="#dbe7f2",
            font=("Microsoft YaHei UI", 10, "bold"),
        )
        self.capture_mode_label.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))
        self.scan_mode_combo = ttk.Combobox(
            settings,
            textvariable=self.scan_mode_display_var,
            values=[tr(self.language, f"app.scan.{key}") for key in SCAN_MODE_KEYS],
            state="readonly",
            font=("Microsoft YaHei UI", 9),
        )
        self.scan_mode_combo.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 5))
        self.scan_mode_combo.bind("<<ComboboxSelected>>", self._scan_mode_combo_changed)
        self.duration_label = tk.Label(
            settings, text=tr(self.language, "app.duration"), bg="#172330", fg="#a9bdcf"
        )
        self.duration_label.grid(row=2, column=0, sticky="w", pady=(4, 0))
        self.duration_spin = tk.Spinbox(
            settings, from_=3, to=30, increment=1, textvariable=self.duration_var,
            width=8, bg="#0d151e", fg="#f2f7fb", buttonbackground="#26394d",
            insertbackground="white", relief=tk.FLAT,
        )
        self.duration_spin.grid(row=2, column=1, sticky="e", pady=(4, 0))
        self.fps_label = tk.Label(
            settings, text=tr(self.language, "app.save_fps"), bg="#172330", fg="#a9bdcf"
        )
        self.fps_label.grid(row=3, column=0, sticky="w", pady=(6, 0))
        self.fps_spin = tk.Spinbox(
            settings, from_=1, to=15, increment=1, textvariable=self.capture_fps_var,
            width=8, bg="#0d151e", fg="#f2f7fb", buttonbackground="#26394d",
            insertbackground="white", relief=tk.FLAT,
        )
        self.fps_spin.grid(row=3, column=1, sticky="e", pady=(6, 0))
        settings.grid_columnconfigure(0, weight=1)
        self.keyframe_quality_label=tk.Label(settings,text=self._t('app.capture_quality'),
            bg='#172330',fg='#dbe7f2',font=('Microsoft YaHei UI',9,'bold'))
        self.keyframe_quality_label.grid(row=6,column=0,columnspan=2,sticky='w',pady=(8,3))
        self.keyframe_quality_combo=ttk.Combobox(settings,textvariable=self.keyframe_quality_var,
            values=('快速模式 1帧','标准模式 5帧','高质量模式 10帧','精细模式 15帧'),state='readonly',font=('Microsoft YaHei UI',9))
        self.keyframe_quality_combo.grid(row=7,column=0,columnspan=2,sticky='ew')

        self.scan_button = self._button(
            panel, self._scan_button_label(), self.toggle_scan, "#168f58", "#1baa68", height=2
        )
        self.scan_button.pack(fill=tk.X, pady=(0, 6))
        self.progress = ttk.Progressbar(
            panel,
            style="Dark.Horizontal.TProgressbar",
            variable=self.progress_var,
            maximum=100.0,
        )
        self.progress.pack(fill=tk.X, pady=(0, 6))

        self._build_model_controls(panel)

        status = tk.Frame(panel, bg="#0c141d", padx=8, pady=6)
        status.pack(fill=tk.X)
        tk.Label(
            status,
            textvariable=self.status_var,
            bg="#0c141d",
            fg="#75e0a5",
            justify=tk.LEFT,
            anchor="w",
            wraplength=300,
            font=("Microsoft YaHei UI", 10, "bold"),
        ).pack(fill=tk.X)
        tk.Label(
            status,
            textvariable=self.detail_var,
            bg="#0c141d",
            fg="#a3b6c8",
            justify=tk.LEFT,
            anchor="w",
            wraplength=300,
            font=("Microsoft YaHei UI", 9),
        ).pack(fill=tk.X, pady=(3, 0))

        # Scope wheel handling to this panel; run before combobox/spinbox class
        # bindings so scrolling does not accidentally change capture settings.
        wheel_tag = 'PanelWheel_' + str(id(self.panel_canvas))
        def scroll_panel(event):
            if getattr(event, 'num', None) == 4:
                units = -3
            elif getattr(event, 'num', None) == 5:
                units = 3
            elif event.delta:
                units = -int(event.delta / 120) * 3 if abs(event.delta) >= 120 else (-1 if event.delta > 0 else 1)
            else:
                return 'break'
            if self.panel_canvas.yview() != (0.0, 1.0):
                self.panel_canvas.yview_scroll(units, 'units')
            return 'break'
        for sequence in ('<MouseWheel>', '<Button-4>', '<Button-5>'):
            self.root.bind_class(wheel_tag, sequence, scroll_panel)
        def attach_wheel(widget):
            widget.bindtags((wheel_tag, *widget.bindtags()))
            for child in widget.winfo_children():
                attach_wheel(child)
        attach_wheel(self.panel_canvas)

    @staticmethod
    def _button(
        parent: tk.Widget,
        text: str,
        command: Any,
        color: str,
        active: str,
        height: int = 1,
    ) -> tk.Button:
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=color,
            activebackground=active,
            fg="white",
            activeforeground="white",
            disabledforeground="#718194",
            relief=tk.FLAT,
            bd=0,
            height=height,
            cursor="hand2",
            highlightthickness=0,
            padx=12,
            pady=6,
            font=("Microsoft YaHei UI", 10, "bold"),
        )

    def _t(self, key: str, **values: Any) -> str:
        return tr(self.language, key, **values)

    def _render_token(self, key: str, values: dict[str, Any]) -> str:
        if key != "app.result.metrics":
            return self._t(key, **values)
        formatted = dict(values)
        depth_value = formatted.pop("depth_value", None)
        near_value = formatted.pop("near_value", None)
        none_value = formatted.pop("none_value", None)
        formatted["depth"] = (
            self._t("app.metric.depth", value=depth_value * 100.0)
            if depth_value is not None
            else ""
        )
        formatted["near"] = (
            self._t("app.metric.near", value=near_value * 100.0)
            if near_value is not None
            else ""
        )
        formatted["none"] = (
            self._t("app.metric.none", value=none_value * 100.0)
            if none_value is not None
            else ""
        )
        return self._t(key, **formatted)

    def _set_status(self, key: str, **values: Any) -> None:
        self._status_token = (key, dict(values))
        self.status_var.set(self._render_token(key, dict(values)))

    def _set_detail(self, key: str, **values: Any) -> None:
        self._detail_token = (key, dict(values))
        self.detail_var.set(self._render_token(key, dict(values)))

    def _set_raw_detail(self, value: str) -> None:
        self._detail_token = None
        self.detail_var.set(value)

    def _update_device_label(self) -> None:
        if self.device_info is None:
            self.device_var.set(self._t("app.device_waiting"))
            return
        payload = self.device_info
        spec = DEPTH_MODE_SPECS[APP_DEPTH_MODE]
        range_m = payload.get("depth_range_m", [spec["min_depth_mm"] / 1000.0, spec["max_depth_mm"] / 1000.0])
        self.device_var.set(
            self._t(
                "app.device_ready",
                device=payload.get("device", "Orbbec Femto Bolt"),
                firmware=payload.get("firmware", self._t("app.unknown")),
                mode=payload.get("depth_mode", spec["name"]),
                minimum=range_m[0],
                maximum=range_m[1],
            )
        )

    def _language_changed(self, _event: Any = None) -> None:
        selected = self.language_display_var.get()
        self.language = "en" if selected == LANGUAGE_NAMES["en"] else "zh"
        save_language(self.language)
        self._apply_language()

    def _apply_language(self) -> None:
        self.root.title(self._t("app.window_title"))
        self.heading_label.configure(text=self._t("app.heading"))
        self.preview_label.configure(text=self._t("app.camera_starting"))
        self.panel_label.configure(text=self._t("app.panel"))
        self.capture_mode_label.configure(text=self._t("app.capture_mode"))
        self.duration_label.configure(text=self._t("app.duration"))
        self.fps_label.configure(text=self._t("app.save_fps"))
        self.three_dgs_profile_label.configure(text=self._t("app.3dgs_profile"))
        self.photo_button.configure(text=self._t("app.photo"))
        self.keyframe_quality_label.configure(text=self._t('app.capture_quality'))
        self.point_cloud_button.configure(
            text=self._t(
                "app.point_cloud.close"
                if self.point_cloud_preview is not None
                else "app.point_cloud.open"
            )
        )
        self.depth_mode_label.configure(text=self._t("app.depth.fixed"))
        if self.point_cloud_preview is not None:
            self.point_cloud_preview.set_language(self.language)
        self.view_button.configure(text=self._t(f"app.view_model.{self.scan_mode_var.get()}"))
        self.train_3dgs_button.configure(text=self._t("app.train_3dgs"))
        self.single_train_button.configure(text=self._t('app.single_train'))
        self.single_view_button.configure(text=self._t('app.single_view'))
        self.view_3dgs_button.configure(text=self._t("app.view_3dgs"))
        self.gsplat_report_button.configure(text=self._t("app.gsplat_report"))
        self.gsplat_folder_button.configure(text=self._t("app.gsplat_folder"))
        self.report_button.configure(text=self._t("app.report"))
        self.folder_button.configure(text=self._t("app.folder"))
        self.scan_mode_combo.configure(
            values=[self._t(f"app.scan.{key}") for key in SCAN_MODE_KEYS]
        )
        self.three_dgs_profile_combo.configure(
            values=[self._t(f"app.3dgs_profile.{key}") for key in THREE_DGS_PROFILE_KEYS]
        )
        self.scan_mode_display_var.set(self._t(f"app.scan.{self.scan_mode_var.get()}"))
        self.three_dgs_profile_display_var.set(
            self._t(f"app.3dgs_profile.{self.three_dgs_profile_var.get()}")
        )
        self._update_device_label()
        if self._status_token is not None:
            key, values = self._status_token
            self.status_var.set(self._render_token(key, values))
        if self._detail_token is not None:
            key, values = self._detail_token
            self.detail_var.set(self._render_token(key, values))
        self._set_action_state()

    def _start_camera(self) -> None:
        if self.camera_thread and self.camera_thread.is_alive():
            return
        self.stop_event.clear()
        self.camera_thread = threading.Thread(
            target=self._camera_worker,
            name="FemtoCamera",
            daemon=True,
        )
        self.camera_thread.start()

    def _camera_dependencies(self) -> SimpleNamespace:
        return SimpleNamespace(
            ActiveScan=ActiveScan,
            APP_COLOR_SIZE=APP_COLOR_SIZE,
            APP_DEPTH_MODE=APP_DEPTH_MODE,
            DATASETS_ROOT=DATASETS_ROOT,
            MOTION_CHECK_INTERVAL_S=MOTION_CHECK_INTERVAL_S,
            MOTION_CHECK_SCALE=MOTION_CHECK_SCALE,
            MOTION_FINAL_WAIT_S=MOTION_FINAL_WAIT_S,
            OUTPUTS_ROOT=OUTPUTS_ROOT,
            OrbbecAlignedCamera=OrbbecAlignedCamera,
            SCAN_MODE_KEYS=SCAN_MODE_KEYS,
            ThreadPoolExecutor=ThreadPoolExecutor,
            build_fixed_keyframe_dataset=build_fixed_keyframe_dataset,
            capture_feedback=capture_feedback,
            copy_frame=copy_frame,
            create_session=create_session,
            finalize_scan=finalize_scan,
            harvest_capture_motion=harvest_capture_motion,
            messagebox=messagebox,
            run_demo=run_demo,
            save_rgbd_photo=save_rgbd_photo,
            save_scan_frame=save_scan_frame,
            threading=threading,
            tk=tk,
            time=time,
            traceback=traceback,
            wait_for_aligned_frame=wait_for_aligned_frame,
            write_calibration=write_calibration,
        )

    def _camera_service(self):
        from femto.app.camera_worker import CameraWorker

        return CameraWorker(self, self._camera_dependencies())

    def _camera_worker(self) -> None:
        self._camera_service().run()

    def _dispatch_temporal(self, burst, *, fuse=True):
        self._camera_service().dispatch_temporal(burst, fuse=fuse)

    def open_temporal_panel(self):
        from femto.app.app_temporal_panel import TemporalPanel
        if self.temporal_panel is not None and self.temporal_panel.window.winfo_exists():
            self.temporal_panel.window.lift()
        else:
            self.temporal_panel = TemporalPanel(self)

    def start_temporal(self, count, mode):
        if not self.camera_ready or self.capturing or self.processing or self.photo_saving:
            return False
        if count not in (5,10,15) or mode not in ('fast','quality'):
            raise ValueError('Invalid temporal options')
        self.temporal_busy = self.temporal_collecting = True
        self.processing = True
        self.latest_temporal = None
        self.progress_var.set(0)
        self._set_action_state()
        self.command_queue.put(('temporal_start',dict(count=count,mode=mode)))
        return True

    def _photo_worker(
        self, frame: AlignedFrame, camera: OrbbecAlignedCamera
    ) -> None:
        self._camera_service().photo_worker(frame, camera)

    def _reconstruction_worker(self, dataset: Path, processing_mode: str) -> None:
        self._camera_service().reconstruction_worker(dataset, processing_mode)

    def _update_preview(self) -> None:
        if not self.root.winfo_exists():
            return
        frame: AlignedFrame | None
        with self.frame_lock:
            frame = self.latest_frame
        if frame is not None:
            color = frame.color_bgr
            depth_spec = DEPTH_MODE_SPECS[APP_DEPTH_MODE]
            minimum_depth_mm = float(depth_spec["min_depth_mm"])
            maximum_depth_mm = float(depth_spec["max_depth_mm"])
            finite_positive = np.isfinite(frame.depth_mm) & (frame.depth_mm > 0)
            valid_depth = (
                finite_positive
                & (frame.depth_mm >= minimum_depth_mm)
                & (frame.depth_mm <= maximum_depth_mm)
            )
            valid_depth_ratio = float(valid_depth.mean())
            too_near_ratio = float((finite_positive & (frame.depth_mm < minimum_depth_mm)).mean())
            too_far_ratio = float((finite_positive & (frame.depth_mm > maximum_depth_mm)).mean())
            no_return_ratio = float((~finite_positive).mean())
            depth = depth_preview(
                np.where(finite_positive, frame.depth_mm, np.nan),
                minimum_depth_mm,
                max(maximum_depth_mm, 1.0),
            )
            local_visible = hasattr(self, 'workbench') and self.workbench.index(self.workbench.select()) == 0
            target_label = self.local_panel.preview if local_visible else self.preview_label
            target_width = max(360, target_label.winfo_width())
            target_height = max(240, target_label.winfo_height())
            depth_width = max(180, int(round(target_width * 0.29)))
            rgb_width = target_width - depth_width
            rgb_height = max(1, int(round(color.shape[0] * rgb_width / color.shape[1])))
            depth_height = max(1, int(round(depth.shape[0] * depth_width / depth.shape[1])))
            rgb_small = cv2.resize(color, (rgb_width, rgb_height), interpolation=cv2.INTER_AREA)
            depth_small = cv2.resize(depth, (depth_width, depth_height), interpolation=cv2.INTER_NEAREST)
            canvas_height = max(rgb_height, depth_height)
            composed = np.zeros((canvas_height, target_width, 3), dtype=np.uint8)
            composed[:rgb_height, :rgb_width] = rgb_small
            composed[:depth_height, rgb_width : rgb_width + depth_width] = depth_small
            cv2.line(composed, (rgb_width, 0), (rgb_width, canvas_height), (255, 255, 255), 2)
            cv2.putText(
                composed,
                "RGB",
                (14, 31),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                composed,
                "DEPTH mm",
                (rgb_width + 12, 31),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            if local_visible:
                composed = depth if self.local_panel.view == 'depth' else color
            scale = min(target_width / composed.shape[1], target_height / composed.shape[0])
            if scale < 1.0:
                composed = cv2.resize(
                    composed,
                    (int(round(composed.shape[1] * scale)), int(round(composed.shape[0] * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            rgb_image = cv2.cvtColor(composed, cv2.COLOR_BGR2RGB)
            self.preview_photo = ImageTk.PhotoImage(Image.fromarray(rgb_image))
            if not local_visible or self.local_panel.view in ('preview', 'depth'):
                target_label.configure(image=self.preview_photo, text="")
            self._update_point_cloud_preview(frame)
            self.preview_counter += 1
            elapsed = time.monotonic() - self.preview_started
            if elapsed >= 1.0:
                self.frame_var.set(
                    self._t(
                        "app.preview_stats",
                        fps=self.preview_counter / elapsed,
                        valid=valid_depth_ratio * 100.0,
                        near=too_near_ratio * 100.0,
                        far=too_far_ratio * 100.0,
                        none=no_return_ratio * 100.0,
                    )
                )
                self.preview_counter = 0
                self.preview_started = time.monotonic()
        self.root.after(40, self._update_preview)

    def _handle_status_event(self,event,payload):
        return self._camera_service().handle_status_event(event, payload)

    def _handle_reconstruction_done(self,payload):
        self._camera_service().handle_reconstruction_done(payload)

    def _poll_events(self) -> None:
        self._camera_service().poll_events()

    def _set_action_state(self) -> None:
        self._model_service().update_action_state()

    def _scan_button_label(self) -> str:
        mode = self.scan_mode_var.get()
        return self._t(f"app.scan_button.{mode}")

    def _scan_mode_combo_changed(self, _event: Any = None) -> None:
        selected = self.scan_mode_display_var.get()
        for key in SCAN_MODE_KEYS:
            if self._t(f"app.scan.{key}") == selected:
                self.scan_mode_var.set(key)
                break
        self._mode_changed()

    def _three_dgs_profile_combo_changed(self, _event: Any = None) -> None:
        self._model_service().profile_changed()

    def _mode_changed(self) -> None:
        if not self.capturing and not self.processing:
            if self.scan_mode_var.get() == "moving":
                self.duration_var.set(15.0)
                self.capture_fps_var.set(6.0)
            self.scan_button.configure(text=self._scan_button_label())
            mode = self.scan_mode_var.get()
            self._set_detail(f"app.mode.{mode}_help")
            self._set_action_state()

    def take_photo(self) -> None:
        if self.camera_ready and not self.capturing and not self.processing and not self.photo_saving:
            self.photo_saving = True
            self._set_action_state()
            self.command_queue.put(("photo", None))

    def toggle_point_cloud_preview(self) -> None:
        if self.point_cloud_preview is not None:
            self._close_point_cloud_preview()
            return
        try:
            self.point_cloud_preview = LivePointCloudPreview(self.language)
        except Exception as error:
            messagebox.showerror(self._t("app.point_cloud.title"), str(error))
            return
        self.point_cloud_button.configure(text=self._t("app.point_cloud.close"))

    def _close_point_cloud_preview(self) -> None:
        preview, self.point_cloud_preview = self.point_cloud_preview, None
        if preview is not None:
            preview.close()
        self.point_cloud_button.configure(text=self._t("app.point_cloud.open"))

    def _poll_point_cloud_preview(self) -> None:
        preview = self.point_cloud_preview
        if preview is None:
            return
        errors = [str(value) for kind, value in preview.poll() if kind == "error"]
        if errors or not preview.process.is_alive():
            self._close_point_cloud_preview()
        if errors:
            messagebox.showerror(self._t("app.point_cloud.title"), errors[0])

    def _update_point_cloud_preview(self, frame: AlignedFrame) -> None:
        preview = self.point_cloud_preview
        if preview is not None and self.camera_intrinsics is not None and self.camera_rgb_distortion is not None:
            preview.submit(frame, self.camera_intrinsics, self.camera_rgb_distortion)

    def toggle_scan(self) -> None:
        self._camera_service().toggle_scan()

    def _selected_model_result(self) -> Path | None:
        return self._model_service().selected_model_result()

    def view_model(self) -> None:
        self._model_service().view_model()

    @staticmethod
    def _format_metric(value: Any, spec: str, suffix: str = "") -> str:
        from femto.app.app_model_panel import ModelPanel

        return ModelPanel.format_metric(value, spec, suffix)

    def _train_3dgs_worker(self, profile: str, demo_root: Path, iterations: int) -> None:
        self._model_service().train_3dgs_worker(profile, demo_root, iterations)

    def train_3dgs(self) -> None:
        self._model_service().train_3dgs()

    def train_single_gs(self) -> None:
        self._model_service().train_single_gs()

    def view_single_gs(self) -> None:
        self._model_service().view_single_gs()

    def view_3dgs(self) -> None:
        self._model_service().view_3dgs()

    def open_gsplat_report(self) -> None:
        self._model_service().open_gsplat_report()

    def open_gsplat_folder(self) -> None:
        self._model_service().open_gsplat_folder()

    def open_report(self) -> None:
        self._model_service().open_report()

    def open_data_folder(self) -> None:
        self._model_service().open_data_folder()

    def close(self) -> None:
        if self.capturing:
            if not messagebox.askyesno(
                self._t("app.dialog.exit_title"), self._t("app.dialog.exit_capture")
            ):
                return
        if self.processing:
            if not messagebox.askyesno(
                self._t("app.dialog.exit_title"),
                self._t("app.dialog.exit_processing"),
            ):
                return
        from femto.planning.planning_workspace import close_workspaces
        if not close_workspaces(self.root):return
        self.stop_event.set()
        self.command_queue.put(("shutdown", None))
        if self.temporal_collecting and self.camera_thread:
            # Let the owner hand any partial burst to the non-daemon evidence
            # writer before Tk exits. No SDK calls from this UI thread.
            self.camera_thread.join(timeout=3)
        if self.point_cloud_preview is not None:
            self._close_point_cloud_preview()
        self.root.destroy()


def smoke_test() -> int:
    camera: OrbbecAlignedCamera | None = None
    started = time.monotonic()
    try:
        with OrbbecAlignedCamera(APP_DEPTH_MODE, color_size=APP_COLOR_SIZE) as active_camera:
            camera = active_camera
            frame = wait_for_aligned_frame(camera)
            if frame is None:
                raise RuntimeError(
                    "Camera produced no aligned RGB-D frame before the startup deadline"
                )
            output = save_rgbd_photo(frame, camera)
    except Exception as error:
        payload: dict[str, Any] = {
            "status": "failed",
            "error": str(error),
            "elapsed_s": time.monotonic() - started,
        }
        if camera is not None:
            payload["connection_type"] = camera.metadata.get("connection_type")
            payload["read_diagnostics"] = camera.read_diagnostics_snapshot()
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 2
    print(
        json.dumps(
            {
                "status": "passed",
                "photo": str(output),
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
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def main() -> int:
    args = parse_args()
    if args.smoke_test:
        return smoke_test()
    # pythonw.exe starts without console streams. Native libraries such as Open3D
    # may emit warnings while exporting; give them real streams instead of None.
    log_dir = LOG_ROOT
    log_dir.mkdir(parents=True, exist_ok=True)
    if sys.stdout is None:
        sys.stdout = (log_dir / "femto_control_stdout.log").open("a", encoding="utf-8", buffering=1)
    if sys.stderr is None:
        sys.stderr = (log_dir / "femto_control_stderr.log").open("a", encoding="utf-8", buffering=1)
    enable_windows_dpi_awareness()
    root = tk.Tk()
    FemtoControlApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
