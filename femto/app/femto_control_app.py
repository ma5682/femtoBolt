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
from femto.io_utils import write_json as json_dump
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
        self.root.geometry("1460x900")
        self.root.minsize(1120, 720)
        self.root.configure(bg="#0b1118")
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
            "Dark.Horizontal.TProgressbar",
            troughcolor="#1a2633",
            background="#42d392",
            bordercolor="#1a2633",
            lightcolor="#42d392",
            darkcolor="#42d392",
        )
        self.workbench = ttk.Notebook(self.root)
        self.workbench.pack(fill=tk.BOTH, expand=True)
        local_host = tk.Frame(self.workbench, bg='#0b121a')
        legacy_host = tk.Frame(self.workbench, bg='#0b1118')
        self.workbench.add(local_host, text='局部移动补采 / Local capture')
        self.workbench.add(legacy_host, text='固定采集与高级功能 / Advanced')
        from femto.app.app_local_panel import LocalPanel
        self.local_panel = LocalPanel(self, local_host)
        shell = tk.Frame(legacy_host, bg="#0b1118")
        shell.pack(fill=tk.BOTH, expand=True, padx=18, pady=18)
        shell.grid_columnconfigure(0, weight=1)
        shell.grid_columnconfigure(1, minsize=360)
        shell.grid_rowconfigure(1, weight=1)

        header = tk.Frame(shell, bg="#0b1118")
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 14))
        self.heading_label = tk.Label(
            header,
            text=tr(self.language, "app.heading"),
            bg="#0b1118",
            fg="#f2f7fb",
            font=("Microsoft YaHei UI", 19, "bold"),
        )
        self.heading_label.pack(anchor="w")
        tk.Label(
            header,
            textvariable=self.device_var,
            bg="#0b1118",
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

    def _build_model_controls(self,panel):
        self.view_button = self._button(
            panel, tr(self.language, "app.view_model"), self.view_model, "#6d48a8", "#8058c1"
        )
        self.view_button.pack(fill=tk.X, pady=(0, 4))
        gaussian_buttons = tk.Frame(panel, bg="#111a24")
        gaussian_buttons.pack(fill=tk.X, pady=(0, 4))
        gaussian_buttons.grid_columnconfigure(0, weight=1)
        gaussian_buttons.grid_columnconfigure(1, weight=1)
        self.three_dgs_profile_label = tk.Label(gaussian_buttons,
            text=self._t('app.3dgs_profile'), bg='#111a24', fg='#dbe7f2')
        self.three_dgs_profile_label.grid(row=0, column=0, columnspan=2, sticky='w')
        self.three_dgs_profile_combo = ttk.Combobox(gaussian_buttons,
            textvariable=self.three_dgs_profile_display_var,
            values=[self._t(f'app.3dgs_profile.{key}') for key in THREE_DGS_PROFILE_KEYS],
            state='readonly', font=('Microsoft YaHei UI', 9))
        self.three_dgs_profile_combo.grid(row=1, column=0, columnspan=2, sticky='ew')
        self.three_dgs_profile_combo.bind('<<ComboboxSelected>>', self._three_dgs_profile_combo_changed)
        self.train_3dgs_button = self._button(
            gaussian_buttons, tr(self.language, "app.train_3dgs"), self.train_3dgs, "#9b5d16", "#b66d19"
        )
        self.train_3dgs_button.grid(row=2, column=0, sticky="ew", padx=(0, 4))
        self.view_3dgs_button = self._button(
            gaussian_buttons, tr(self.language, "app.view_3dgs"), self.view_3dgs, "#71449a", "#8551b5"
        )
        self.view_3dgs_button.grid(row=2, column=1, sticky="ew", padx=(4, 0))
        self.gsplat_report_button = self._button(
            gaussian_buttons, tr(self.language, "app.gsplat_report"), self.open_gsplat_report, "#48596b", "#566b81"
        )
        self.gsplat_report_button.grid(row=3, column=0, sticky="ew", padx=(0, 4), pady=(4, 0))
        self.gsplat_folder_button = self._button(
            gaussian_buttons, tr(self.language, "app.gsplat_folder"), self.open_gsplat_folder, "#394958", "#465a6d"
        )
        self.gsplat_folder_button.grid(row=3, column=1, sticky="ew", padx=(4, 0), pady=(4, 0))
        self.moving_source_label = tk.Label(gaussian_buttons, bg='#111a24', fg='#a3b6c8', wraplength=600, anchor='w', justify='left')
        self.moving_source_label.grid(row=4, column=0, columnspan=2, sticky='ew')
        self.single_train_button = self._button(gaussian_buttons, self._t('app.single_train'), self.train_single_gs, '#9b5d16', '#b66d19')
        self.single_train_button.grid(row=5, column=0, sticky='ew', padx=(0,4), pady=(4,0))
        self.single_view_button = self._button(gaussian_buttons, self._t('app.single_view'), self.view_single_gs, '#71449a', '#8551b5')
        self.single_view_button.grid(row=5, column=1, sticky='ew', padx=(4,0), pady=(4,0))
        self.single_source_label = tk.Label(gaussian_buttons, bg='#111a24', fg='#a3b6c8', wraplength=600, anchor='w', justify='left')
        self.single_source_label.grid(row=6,column=0,columnspan=2,sticky='ew')
        result_buttons = tk.Frame(panel, bg="#111a24")
        result_buttons.pack(fill=tk.X, pady=(0, 5))
        result_buttons.grid_columnconfigure(0, weight=1)
        result_buttons.grid_columnconfigure(1, weight=1)
        self.report_button = self._button(
            result_buttons, tr(self.language, "app.report"), self.open_report, "#48596b", "#566b81"
        )
        self.report_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.folder_button = self._button(
            result_buttons, tr(self.language, "app.folder"), self.open_data_folder, "#394958", "#465a6d"
        )
        self.folder_button.grid(row=0, column=1, sticky="ew", padx=(4, 0))

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
            font=("Microsoft YaHei UI", 11, "bold"),
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

    def _drain_camera_commands(self, camera, scan, temporal_burst, local):
        while True:
            try:
                command, payload = self.command_queue.get_nowait()
            except queue.Empty:
                break
            if command == "shutdown":
                self.stop_event.set()
                break
            if command == 'local_start':
                requested_session = payload.get('session_id') if isinstance(payload, dict) else None
                if scan is not None or temporal_burst is not None or (local is not None and local.thread.is_alive()):
                    self.event_queue.put(('local_error',dict(session_id=requested_session,source=None,error='Another capture is active')))
                    continue
                try:
                    from femto.capture.local_capture import LocalCapture
                    local = LocalCapture(camera, DATASETS_ROOT, OUTPUTS_ROOT, self.event_queue,
                                         session_id=requested_session, shutdown_event=self.stop_event,
                                         cancel_event=payload.get('cancel_event') if isinstance(payload,dict) else None)
                except Exception as error:
                    self.event_queue.put(('local_error',dict(session_id=requested_session,source=None,error=str(error))))
            if command in ('local_request', 'local_pause', 'local_finish') and isinstance(payload, dict):
                if local is None or payload.get('session_id') != local.session_id:
                    continue
                payload = payload.get('value')
            if command == 'local_request' and local is not None:
                local.request()
            if command == 'local_pause' and local is not None:
                local.pause(bool(payload))
            if command == 'local_finish' and local is not None:
                local.finish(bool(payload))
            if command in ('start_scan','temporal_start','photo') and local is not None and local.thread.is_alive():
                continue
            if command == 'temporal_start':
                if scan is not None or temporal_burst is not None or (getattr(self,'temporal_thread',None) and self.temporal_thread.is_alive()):
                    self.event_queue.put(('temporal_error',dict(error='Another capture is active',source=None)))
                    continue
                from femto.app.app_temporal import TemporalBurst
                temporal_burst = TemporalBurst(camera,int(payload['count']),str(payload['mode']))
                temporal_burst.build_model = bool(payload.get('build_model',False))
                self.event_queue.put(('temporal_progress',dict(frames=0,total=temporal_burst.count)))
            if command == 'temporal_stop' and temporal_burst is not None:
                self._dispatch_temporal(temporal_burst,fuse=False)
                temporal_burst = None
            if command == "photo":
                if temporal_burst is not None:
                    continue
                with self.frame_lock:
                    snapshot = copy_frame(self.latest_frame) if self.latest_frame is not None else None
                if snapshot is None:
                    self.event_queue.put(("error", "相机还没有可保存的 RGB-D 帧。"))
                elif self.photo_thread is None or not self.photo_thread.is_alive():
                    self.photo_thread = threading.Thread(
                        target=self._photo_worker,
                        args=(snapshot, camera),
                        name="FemtoPhoto",
                        daemon=True,
                    )
                    self.photo_thread.start()
            if command == "start_scan" and scan is None:
                if temporal_burst is not None or (getattr(self,'temporal_thread',None) and self.temporal_thread.is_alive()):
                    continue
                duration_s = float(payload["duration_s"])
                fps = float(payload["fps"])
                processing_mode = str(payload.get("processing_mode", "keyframe"))
                if processing_mode not in SCAN_MODE_KEYS:
                    self.event_queue.put(("error", self._t("app.capture.single_only")))
                    continue
                fixed_camera = processing_mode == "keyframe"
                max_frames = max(2, int(round(duration_s * fps)))
                session = create_session(DATASETS_ROOT)
                if camera.intrinsics is None:
                    raise RuntimeError("Camera intrinsics are unavailable")
                write_calibration(session, camera, camera.intrinsics)
                now = time.monotonic()
                scan = ActiveScan(
                    session=session,
                    frames_log=session / "frames.jsonl",
                    started=now,
                    next_sample=now,
                    duration_s=duration_s,
                    fps=fps,
                    max_frames=max_frames,
                    fixed_camera=fixed_camera,
                    processing_mode=processing_mode,
                    depth_min_mm=camera.min_depth_mm,
                    depth_max_mm=camera.max_depth_mm,
                    reconstruct_after=True,
                    next_motion_check=now + MOTION_CHECK_INTERVAL_S,
                )
                self.event_queue.put(
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
            if command == "stop_scan" and scan is not None:
                scan.reconstruct_after = bool(payload.get("reconstruct", True))
                harvest_capture_motion(scan, wait_timeout_s=MOTION_FINAL_WAIT_S)
                finished = finalize_scan(scan, camera)
                self.event_queue.put(
                    (
                        "scan_finished",
                        {
                            "session": str(finished),
                            "frames": scan.frame_count,
                            "reconstruct": scan.reconstruct_after,
                            "fixed_camera": scan.fixed_camera,
                            "processing_mode": scan.processing_mode,
                            "capture_quality": scan.capture_quality,
                            "motion_state": scan.motion_state,
                            "motion_report": scan.motion_report,
                        },
                    )
                )
                scan = None
        return scan, temporal_burst, local

    def _camera_worker(self) -> None:
        camera: OrbbecAlignedCamera | None = None
        scan: ActiveScan | None = None
        temporal_burst = None
        local = None
        motion_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="FemtoMotion")
        try:
            camera = OrbbecAlignedCamera(APP_DEPTH_MODE, color_size=APP_COLOR_SIZE).start()
            first_frame = wait_for_aligned_frame(camera)
            if first_frame is None:
                raise RuntimeError(
                    "Camera started but produced no aligned RGB-D frame before the startup deadline: "
                    + json.dumps(camera.read_diagnostics_snapshot(), ensure_ascii=False)
                )
            with self.frame_lock:
                self.latest_frame = first_frame
            ready_metadata = dict(camera.metadata)
            ready_metadata["_intrinsics"] = camera.intrinsics
            ready_metadata["_rgb_distortion"] = dict(camera.calibration["rgb_distortion"])
            ready_metadata["read_diagnostics"] = camera.read_diagnostics_snapshot()
            self.event_queue.put(("camera_ready", ready_metadata))
            while not self.stop_event.is_set():
                scan, temporal_burst, local = self._drain_camera_commands(
                    camera, scan, temporal_burst, local
                )

                if self.stop_event.is_set():
                    break
                if temporal_burst is not None and time.monotonic()>=temporal_burst.deadline:
                    self._dispatch_temporal(temporal_burst,fuse=False)
                    temporal_burst = None
                frame = camera.read(1000)
                if frame is None:
                    continue
                with self.frame_lock:
                    self.latest_frame = frame
                if local is not None and not local.closing.is_set():
                    local.submit(frame)
                if temporal_burst is not None:
                    if temporal_burst.append(frame,camera.intrinsics):
                        self.event_queue.put(('temporal_progress',dict(frames=len(temporal_burst.frames),total=temporal_burst.count)))
                    if temporal_burst.complete:
                        self._dispatch_temporal(temporal_burst)
                        temporal_burst = None
                if scan is not None:
                    now = time.monotonic()
                    if not scan.fixed_camera:
                        harvest_capture_motion(scan)
                    if now >= scan.next_sample:
                        save_scan_frame(scan, frame)
                        scan.last_preview = frame.color_bgr.copy()
                        scan.next_sample += 1.0 / scan.fps
                        if not scan.fixed_camera:
                            if scan.motion_reference_bgr is None:
                                scan.motion_reference_bgr = frame.color_bgr.copy()
                            elif now >= scan.next_motion_check and scan.motion_future is None:
                                scan.motion_future = motion_executor.submit(
                                    capture_feedback,
                                    scan.motion_reference_bgr,
                                    frame.color_bgr.copy(),
                                    frame.depth_mm.copy(),
                                    scan.depth_min_mm,
                                    scan.depth_max_mm,
                                    scale=MOTION_CHECK_SCALE,
                                )
                                scan.motion_reference_bgr = frame.color_bgr.copy()
                                scan.next_motion_check = now + MOTION_CHECK_INTERVAL_S
                        elapsed = now - scan.started
                        self.event_queue.put(
                            (
                                "scan_progress",
                                {
                                    "frames": scan.frame_count,
                                    "max_frames": scan.max_frames,
                                    "elapsed": elapsed,
                                    "duration": scan.duration_s,
                                    "processing_mode": scan.processing_mode,
                                    "capture_quality": scan.capture_quality,
                                    "motion_state": scan.motion_state,
                                },
                            )
                        )
                    if now - scan.started >= scan.duration_s or scan.frame_count >= scan.max_frames:
                        if not scan.fixed_camera:
                            harvest_capture_motion(scan, wait_timeout_s=MOTION_FINAL_WAIT_S)
                        finished = finalize_scan(scan, camera)
                        self.event_queue.put(
                            (
                                "scan_finished",
                                {
                                    "session": str(finished),
                                    "frames": scan.frame_count,
                                    "reconstruct": scan.reconstruct_after,
                                    "fixed_camera": scan.fixed_camera,
                                    "processing_mode": scan.processing_mode,
                                    "capture_quality": scan.capture_quality,
                                    "motion_state": scan.motion_state,
                                    "motion_report": scan.motion_report,
                                },
                            )
                        )
                        scan = None
            if scan is not None and scan.frame_count:
                scan.reconstruct_after = False
                if not scan.fixed_camera:
                    harvest_capture_motion(scan, wait_timeout_s=MOTION_FINAL_WAIT_S)
                finalize_scan(scan, camera)
        except Exception as error:
            if local is not None and local.thread.is_alive():
                local.abort('camera_error', str(error))
            self.event_queue.put(("camera_error", f"{error}\n\n{traceback.format_exc(limit=4)}"))
        finally:
            if local is not None and local.thread.is_alive():
                local.finish(False)
            if temporal_burst is not None:
                self._dispatch_temporal(temporal_burst,fuse=False)
            if camera is not None:
                camera.stop()
            motion_executor.shutdown(wait=False, cancel_futures=True)
            self.event_queue.put(("camera_stopped", None))

    def _dispatch_temporal(self, burst, *, fuse=True):
        from femto.app.app_temporal import process_burst, process_keyframe_burst
        target=process_keyframe_burst if getattr(burst,'build_model',False) else process_burst
        self.temporal_thread = threading.Thread(target=target,args=(burst,DATASETS_ROOT,self.event_queue),
                                                kwargs=dict(fuse=fuse),name='FemtoTemporal',daemon=False)
        self.temporal_thread.start()

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
        self.event_queue.put(("photo_started", None))
        try:
            output = save_rgbd_photo(frame, camera)
            self.event_queue.put(("photo_saved", str(output)))
        except Exception as error:
            self.event_queue.put(("error", f"拍照保存失败：{error}"))

    def _reconstruction_worker(self, dataset: Path, processing_mode: str) -> None:
        try:
            if processing_mode not in SCAN_MODE_KEYS:
                raise ValueError(self._t("app.capture.single_only"))
            if processing_mode == "keyframe":
                stamp = time.strftime("%Y%m%d_%H%M%S")
                reconstruction_dataset, processing_report = build_fixed_keyframe_dataset(
                    dataset,
                    dataset / "derived" / f"fixed_keyframe_{stamp}",
                )
            else:
                # Keep every raw RGB-D frame. run_demo estimates missing poses
                # in a derived directory and validates them before fusion.
                manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
                if manifest.get("parameters", {}).get("fixed_camera") is not False:
                    raise ValueError(self._t("app.capture.moving_requires_pending"))
                reconstruction_dataset, processing_report = dataset, {}
            args = argparse.Namespace(
                mode="latest",
                dataset=reconstruction_dataset,
                datasets_root=DATASETS_ROOT,
                outputs_root=OUTPUTS_ROOT,
                synthetic_frames=14,
                no_viewer=True,
            )
            result, summary = run_demo(args)
            self.event_queue.put(
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
            self.event_queue.put(
                (
                    "reconstruction_error",
                    f"{error}\n\n{traceback.format_exc(limit=5)}",
                )
            )

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
        if event.startswith('local_'):
            self.local_panel.event(event,payload)
        elif event == "camera_ready":
            self.camera_ready = True
            self.device_info = dict(payload)
            self.camera_intrinsics = self.device_info.pop("_intrinsics", None)
            self.camera_rgb_distortion = self.device_info.pop("_rgb_distortion", None)
            self._update_device_label()
            self._set_status("app.camera_ready")
            self._set_detail("app.camera_ready_detail")
        elif event.startswith('temporal_'):
            if event == 'temporal_progress':
                self.progress_var.set(100*payload['frames']/payload['total'])
                self._status_token = None
                self.status_var.set(self._t('app.temporal')+f" {payload['frames']}/{payload['total']}")
            elif event == 'temporal_processing':
                self.temporal_collecting = False
                self._status_token = None
                self.status_var.set(self._t('app.temporal_processing'))
            elif event in ('temporal_finished','temporal_error'):
                self.temporal_busy = self.temporal_collecting = self.processing = self.keyframe_building = False
                self.latest_temporal = (dict(status='error',output=payload.get('source'),error=payload['error'])
                                        if event == 'temporal_error' else payload)
                self._status_token = None
                self.status_var.set(self._t('app.temporal')+' · '+payload.get('status','error'))
                self._set_raw_detail(payload.get('output') or payload.get('error',''))
                self.progress_var.set(100 if payload.get('status')=='completed' else 0)
            if self.temporal_panel is not None and self.temporal_panel.window.winfo_exists():
                self.temporal_panel.refresh(event,payload)
        elif event=='keyframe_stage':
            self.temporal_collecting=False
            self._set_action_state()
            self._status_token=None
            labels={'saving':'正在保存原始RGB-D…','quality':'Depth Quality / Hole Filling…',
                    'fusion':'正在融合：Temporal Fusion…','cloud':'正在生成点云…',
                    'geometry_inputs':'正在生成几何及3DGS输入（不启动训练）…'}
            self.status_var.set(labels[payload])
        elif event=='keyframe_stopped':
            self.processing=self.temporal_busy=self.temporal_collecting=self.keyframe_building=False
            self._status_token=None
            self.status_var.set('采集已停止')
            self._set_raw_detail('采集已停止，原始证据保留：'+payload)
        elif event == "photo_started":
            self.photo_saving = True
            self._set_status("app.photo_saving")
        elif event == "photo_saved":
            self.photo_saving = False
            self.latest_photo = Path(payload)
            self._set_status("app.photo_saved")
            self._set_raw_detail(str(self.latest_photo))
        elif event == 'single_gs_progress':
            self.progress_var.set(100*payload['step']/payload['total'])
            self._set_detail('app.single_progress', **{k:payload[k] for k in ('step','total','gaussians','cuda_reserved_mib','cuda_free_mib')})
        elif event == 'single_gs_done':
            self.processing = False
            self.latest_single_gs = Path(payload)
            self.progress_var.set(100)
            self._set_status('app.single_done')
            self._set_raw_detail(str(self.latest_single_gs))
        elif event == 'single_gs_error':
            self.processing = False
            self._set_status('app.single_failed')
            self._set_raw_detail(str(payload))
        else:
            return False
        return True

    def _handle_reconstruction_done(self,payload):
        self.temporal_busy=self.temporal_collecting=self.keyframe_building=False
        self.processing = False
        self.latest_result = Path(payload["result"])
        from femto.pipeline.reconstruction_sources import remember
        source_mode = remember(OUTPUTS_ROOT, self.latest_result)
        if source_mode == 'keyframe':
            self.latest_keyframe_result = self.latest_result
        elif source_mode == 'moving':
            self.latest_moving_result = self.latest_result
        if payload.get('source_dataset'): self.latest_dataset=Path(payload['source_dataset'])
        metrics = payload["metrics"]
        processing_report = payload.get("processing_report") or {}
        depth_ratio = processing_report.get("selected_valid_depth_ratio")
        too_near_ratio = processing_report.get("selected_too_near_depth_ratio")
        no_return_ratio = processing_report.get("selected_no_depth_return_ratio")
        if depth_ratio is not None and depth_ratio < 0.35:
            if too_near_ratio is not None and too_near_ratio >= 0.08:
                self._set_status("app.result.too_near")
            elif no_return_ratio is not None and no_return_ratio >= 0.35:
                self._set_status("app.result.no_return")
            else:
                self._set_status("app.result.low_coverage")
        else:
            self._set_status("app.result.done")
        self._set_detail(
            "app.result.metrics",
            points=metrics["fused_point_count"],
            triangles=metrics["mesh_triangles"],
            depth_value=depth_ratio,
            near_value=too_near_ratio,
            none_value=no_return_ratio,
            path=self.latest_result,
        )

    def _poll_events(self) -> None:
        if not self.root.winfo_exists():
            return
        self._poll_point_cloud_preview()
        while True:
            try:
                event, payload = self.event_queue.get_nowait()
            except queue.Empty:
                break
            if self._handle_status_event(event,payload):
                self._set_action_state()
                continue
            if event == "scan_started":
                self.capturing = True
                # A failed new scan must not leave an old model as its train input.
                self.latest_result = None
                self.latest_dataset = Path(payload["session"])
                processing_mode = payload.get("processing_mode", "keyframe")
                self._set_status(f"app.scan_started.{processing_mode}")
                self._set_detail(
                    "app.target_frames", frames=payload["max_frames"], path=self.latest_dataset
                )
                self.progress_var.set(0.0)
            elif event == "scan_progress":
                by_time = payload["elapsed"] / max(payload["duration"], 1e-6)
                by_frames = payload["frames"] / max(payload["max_frames"], 1)
                self.progress_var.set(min(100.0, max(by_time, by_frames) * 100.0))
                remaining = max(0.0, payload["duration"] - payload["elapsed"])
                detail_key = "app.scan_progress"
                detail_values: dict[str, Any] = {
                    "frames": payload["frames"],
                    "maximum": payload["max_frames"],
                    "remaining": remaining,
                }
                if payload.get("processing_mode") == "moving":
                    detail_key = "app.scan_progress.moving"
                    detail_values["motion"] = self._t(
                        f"app.camera_motion.{payload.get('motion_state', 'unknown')}"
                    )
                    quality = payload.get("capture_quality") or {}
                    detail_values["motion"] += " · " + self._t("app.capture_quality." + quality.get("guidance", "pending"))
                self._set_detail(detail_key, **detail_values)
            elif event == "scan_finished":
                self.capturing = False
                self.latest_dataset = Path(payload["session"])
                self.progress_var.set(100.0)
                if payload["frames"] < 2:
                    self._set_status("app.scan_too_short")
                    self._set_raw_detail(str(self.latest_dataset))
                elif payload.get("reconstruct", True):
                    processing_mode = str(payload.get("processing_mode", "keyframe"))
                    if processing_mode == "moving" and payload.get("motion_state") == "not_moved":
                        messagebox.showwarning(
                            self._t("app.dialog.camera_motion_title"),
                            self._t("app.dialog.camera_motion_body"),
                        )
                    self.processing = True
                    self._set_status(f"app.reconstruct.{processing_mode}")
                    self._set_detail(f"app.reconstruct_detail.{processing_mode}")
                    self.reconstruction_thread = threading.Thread(
                        target=self._reconstruction_worker,
                        args=(self.latest_dataset, processing_mode),
                        name="FemtoReconstruction",
                        daemon=True,
                    )
                    self.reconstruction_thread.start()
            elif event == "reconstruction_done":
                self._handle_reconstruction_done(payload)
            elif event == "reconstruction_error":
                self.temporal_busy=self.temporal_collecting=self.keyframe_building=False
                self.processing = False
                self._set_status("app.reconstruct_failed")
                self._set_raw_detail(str(payload).split("\n", 1)[0])
                messagebox.showwarning(
                    self._t("app.dialog.reconstruct_title"),
                    self._t(
                        "app.dialog.reconstruct_body",
                        error=str(payload).split("\n\n", 1)[0],
                    ),
                )
            elif event == "3dgs_ready":
                self._set_status("app.3dgs_training")
                self._set_detail("app.3dgs_training_detail", **payload)
            elif event == "3dgs_progress":
                self._set_detail(
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
                self.processing = False
                self.latest_gsplat_model = Path(payload["output"])
                self._set_status("app.3dgs_done" if payload["accepted"] else "app.3dgs_rejected")
                self._set_detail("app.3dgs_done_detail", **payload["detail"])
            elif event == "3dgs_error":
                self.processing = False
                self._set_status("app.3dgs_failed")
                failure_message = str(payload).split("\n\nTraceback", 1)[0]
                self._set_raw_detail(failure_message)
                messagebox.showwarning(
                    self._t("app.dialog.3dgs_not_ready"), failure_message
                )
            elif event in {"error", "camera_error"}:
                self.photo_saving = False
                if event == "camera_error":
                    self.camera_ready = False
                    self._set_status("app.camera_failed")
                self._set_raw_detail(str(payload).split("\n", 1)[0])
                messagebox.showerror("Femto Bolt", str(payload).split("\n\n", 1)[0])
            elif event == "camera_stopped":
                self.camera_ready = False
                if self.temporal_busy and not (self.temporal_thread and self.temporal_thread.is_alive()):
                    self.temporal_busy = self.temporal_collecting = self.processing = self.keyframe_building = False
                    if self.temporal_panel is not None and self.temporal_panel.window.winfo_exists():
                        self.temporal_panel.refresh('temporal_error',dict(error='Camera stopped before burst completion',source=None))
            self._set_action_state()
        self.root.after(80, self._poll_events)

    def _set_action_state(self) -> None:
        if hasattr(self,'local_panel'):
            self.local_panel.refresh()
        camera_action = (
            tk.NORMAL
            if self.camera_ready
            and not self.processing
            and not self.capturing
            and not self.photo_saving
            else tk.DISABLED
        )
        self.photo_button.configure(state=camera_action)
        if hasattr(self,'temporal_entry_button'):
            self.temporal_entry_button.configure(state=camera_action)
        self.keyframe_quality_combo.configure(state='readonly' if camera_action==tk.NORMAL and self.scan_mode_var.get()=='keyframe' else 'disabled')
        fixed = getattr(self, 'latest_keyframe_result', None)
        moving = getattr(self, 'latest_moving_result', None)
        self.single_train_button.configure(state=tk.NORMAL if fixed and not self.processing and not self.capturing and not self.photo_saving else tk.DISABLED)
        from femto.pipeline.reconstruction_sources import describe
        self.single_source_label.configure(text=self._t('app.single_source', source=describe(fixed)))
        self.moving_source_label.configure(text=self._t('app.moving_source', source=describe(moving)))
        self.single_view_button.configure(state=tk.NORMAL if self.latest_single_gs and not self.processing and not self.capturing else tk.DISABLED)
        self.point_cloud_button.configure(
            state=tk.NORMAL if self.camera_ready else tk.DISABLED
        )
        timed_capture=camera_action == tk.NORMAL and self.scan_mode_var.get()!='keyframe'
        self.duration_spin.configure(state=tk.NORMAL if timed_capture else tk.DISABLED)
        self.fps_spin.configure(state=tk.NORMAL if timed_capture else tk.DISABLED)
        combo_state = "readonly" if camera_action == tk.NORMAL else "disabled"
        self.scan_mode_combo.configure(state=combo_state)
        self.three_dgs_profile_combo.configure(
            state="readonly" if not self.processing and not self.capturing else "disabled"
        )
        if getattr(self, 'local_busy', False):
            self.scan_button.configure(text=self._t('app.scan_button.processing'), state=tk.DISABLED)
        elif self.capturing or (self.keyframe_building and self.temporal_collecting):
            self.scan_button.configure(
                text=self._t("app.scan_button.stop"), state=tk.NORMAL, bg="#b65334"
            )
        elif self.processing:
            self.scan_button.configure(
                text=self._t("app.scan_button.processing"), state=tk.DISABLED, bg="#48596b"
            )
        else:
            self.scan_button.configure(
                text=self._scan_button_label(),
                state=camera_action,
                bg="#168f58",
            )
        self.view_button.configure(
            text=self._t(f"app.view_model.{self.scan_mode_var.get()}"),
            state=tk.NORMAL if self._selected_model_result() else tk.DISABLED)
        self.train_3dgs_button.configure(
            state=tk.NORMAL if moving and not self.processing and not self.capturing and not self.photo_saving else tk.DISABLED
        )
        self.view_3dgs_button.configure(
            state=tk.NORMAL if self.latest_gsplat_model and not self.processing else tk.DISABLED
        )
        for button in (self.gsplat_report_button, self.gsplat_folder_button):
            button.configure(state=tk.NORMAL if self.latest_gsplat_model else tk.DISABLED)
        self.report_button.configure(state=tk.NORMAL if self.latest_result else tk.DISABLED)
        self.folder_button.configure(
            state=tk.NORMAL if (self.latest_photo or self.latest_dataset or self.latest_result) else tk.DISABLED
        )

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
        selected = self.three_dgs_profile_display_var.get()
        for key in THREE_DGS_PROFILE_KEYS:
            if self._t(f"app.3dgs_profile.{key}") == selected:
                self.three_dgs_profile_var.set(key)
                break

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
        if getattr(self,'local_busy',False):
            return
        if getattr(self,'keyframe_building',False):
            if self.temporal_collecting: self.command_queue.put(('temporal_stop',None))
            return
        if self.capturing:
            self.command_queue.put(("stop_scan", {"reconstruct": True}))
            return
        if not self.camera_ready or self.processing:
            return
        if self.scan_mode_var.get()=='keyframe' and hasattr(self,'keyframe_quality_var'):
            if self.photo_saving: return
            names=('快速模式 1帧','标准模式 5帧','高质量模式 10帧','精细模式 15帧')
            count=(1,5,10,15)[names.index(self.keyframe_quality_var.get())]
            self.processing=self.temporal_busy=self.temporal_collecting=self.keyframe_building=True
            self.latest_result=None
            self.progress_var.set(0)
            self._set_action_state()
            self.command_queue.put(('temporal_start',dict(count=count,mode='quality',build_model=True)))
            return
        if self.scan_mode_var.get() not in SCAN_MODE_KEYS:
            messagebox.showinfo("Femto Bolt", self._t("app.capture.single_only"))
            return
        try:
            duration = float(self.duration_var.get())
            fps = float(self.capture_fps_var.get())
        except (TypeError, ValueError, tk.TclError):
            messagebox.showerror(
                self._t("app.dialog.parameter_title"),
                self._t("app.dialog.parameter_number"),
            )
            return
        if not 3.0 <= duration <= 30.0 or not 1.0 <= fps <= 15.0:
            messagebox.showerror(
                self._t("app.dialog.parameter_title"),
                self._t("app.dialog.parameter_range"),
            )
            return
        self.command_queue.put(
            (
                "start_scan",
                {
                    "duration_s": duration,
                    "fps": fps,
                    "processing_mode": self.scan_mode_var.get(),
                },
            )
        )

    def _selected_model_result(self) -> Path | None:
        attribute = {'keyframe': 'latest_keyframe_result',
                     'moving': 'latest_moving_result'}.get(self.scan_mode_var.get())
        return getattr(self, attribute, None) if attribute else None

    def view_model(self) -> None:
        result = self._selected_model_result()
        if not result:
            return
        reconstruction = result / "reconstruction"
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "femto.viewer.scene_demo_viewer",
                str(reconstruction),
                "--lang",
                self.language,
            ],
            cwd=str(ROOT),
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )

    @staticmethod
    def _format_metric(value: Any, spec: str, suffix: str = "") -> str:
        """Render an evaluation number, or a dash when the metric was not produced."""
        if not isinstance(value, (int, float)) or not np.isfinite(value):
            return "-"
        return f"{value:{spec}}{suffix}"

    def _train_3dgs_worker(self, profile: str, demo_root: Path, iterations: int) -> None:
        def on_progress(record: dict[str, Any]) -> None:
            self.event_queue.put(
                (
                    "3dgs_progress",
                    {
                        "step": int(record.get("step", 0)),
                        "total": iterations,
                        "gaussians": int(record.get("gaussians", 0)),
                        "cuda_reserved_mib": float(record.get("cuda_reserved_mib", 0)),
                        "cuda_free_mib": float(record.get("cuda_free_mib", 0)),
                        "max_gaussians": int(record.get("max_gaussians", 0)),
                        "loss": float(record.get("loss", 0.0)),
                        "elapsed_s": float(record.get("elapsed_s", 0.0)),
                    },
                )
            )

        try:
            from femto.gsplat.gsplat_backend import prepare_training_input
            demo_root = prepare_training_input(demo_root)
            data = assess_training_input(demo_root, require_real=True)
            if not data["ready"]:
                raise ValueError(self._t("app.dialog.3dgs_multiview_body", reasons="\n".join(data["reasons"])))
            runtime = gsplat_runtime_status()
            if not runtime["ready"] or not runtime.get("lpips_ready", False):
                raise RuntimeError(self._t("app.dialog.3dgs_runtime_body") + "\n" + str(runtime.get("error") or runtime.get("lpips_backbone", {}).get("error")))
            self.event_queue.put(("3dgs_ready", {"frames": data["frame_count"], "span": data["translation_span_m"], "profile": self._t(f"app.3dgs_profile.{profile}")}))
            output_dir, report = train_direct_gsplat(
                demo_root, profile=profile, on_progress=on_progress, require_real=True
            )
            aggregate = (report.get("evaluation") or {}).get("aggregate") or {}
            depth_rmse_m = aggregate.get("depth_rmse_m")
            coverage = aggregate.get("depth_coverage_ratio")
            self.event_queue.put(
                (
                    "3dgs_done",
                    {
                        "output": str(output_dir),
                        "accepted": bool(report.get("accepted", False)),
                        "detail": {
                            "gaussians": f"{int(report.get('final_gaussians', 0)):,}",
                            "psnr": self._format_metric(aggregate.get("psnr"), ".2f"),
                            "ssim": self._format_metric(aggregate.get("ssim"), ".3f"),
                            "lpips": self._format_metric(aggregate.get("lpips"), ".4f"),
                            "depth_rmse": self._format_metric(
                                depth_rmse_m * 1000.0 if isinstance(depth_rmse_m, (int, float)) else None,
                                ".0f",
                                " mm",
                            ),
                            "coverage": self._format_metric(
                                coverage * 100.0 if isinstance(coverage, (int, float)) else None, ".1f", "%"
                            ),
                            "path": str(output_dir),
                        },
                    },
                )
            )
        except Exception as error:
            self.event_queue.put(("3dgs_error", f"{error}\n\n{traceback.format_exc(limit=4)}"))

    def train_3dgs(self) -> None:
        source = getattr(self, 'latest_moving_result', None)
        if self.processing or self.capturing or self.photo_saving or not source:
            return
        profile = self.three_dgs_profile_var.get()
        self.processing = True
        self._set_status("app.3dgs_checking")
        self._set_detail("app.3dgs_checking_detail")
        self._set_action_state()
        threading.Thread(
            target=self._train_3dgs_worker,
            args=(profile, source, int(PROFILES[profile].iterations)),
            name="FemtoDirectGsplatTraining",
            daemon=True,
        ).start()

    def train_single_gs(self) -> None:
        source = getattr(self, 'latest_keyframe_result', None)
        if self.processing or self.capturing or self.photo_saving or not source:
            return
        photo = source
        self.processing = True
        self.progress_var.set(0)
        self._set_status('app.single_training')
        self._set_raw_detail(str(photo))
        self._set_action_state()
        def worker():
            try:
                from femto.gsplat.scene_single_gs import launch
                output = launch(photo, on_progress=lambda row:self.event_queue.put(('single_gs_progress',row)))
                self.event_queue.put(('single_gs_done',str(output)))
            except Exception as error:
                self.event_queue.put(('single_gs_error',str(error)))
        threading.Thread(target=worker,name='FemtoSingleGs',daemon=True).start()

    def view_single_gs(self) -> None:
        if not self.latest_single_gs: return
        try:
            from femto.gsplat.scene_single_gs_viewer import open_viewer
            process = open_viewer(self.latest_single_gs, self.language)
            output = self.latest_single_gs
            def check():
                if process.poll() not in (None,0):
                    messagebox.showerror('3DGS',str(output/'viewer_stdout.log'))
            self.root.after(15000,check)
        except Exception as error:
            messagebox.showerror('3DGS',str(error))

    def view_3dgs(self) -> None:
        if not self.latest_gsplat_model:
            return
        try:
            process = open_gsplat_viewer(self.latest_gsplat_model, language=self.language)
            self._set_status("app.3dgs_view_start")
            self._set_detail("app.3dgs_view_detail")
            def check_startup() -> None:
                if process.poll() not in (None, 0):
                    messagebox.showwarning(
                        self._t("app.view_3dgs"),
                        self._t("app.3dgs_view_failed", path=str(self.latest_gsplat_model / "viewer_stdout.log")),
                    )
            self.root.after(15000, check_startup)
        except Exception as error:
            messagebox.showwarning(self._t("app.view_3dgs"), str(error))

    def open_gsplat_report(self) -> None:
        if self.latest_gsplat_model:
            from femto.gsplat.scene_gsplat_acceptance import build_acceptance_report

            path = self.latest_gsplat_model / "acceptance_report.json"
            try:
                json_dump(path, build_acceptance_report(self.latest_gsplat_model))
                os.startfile(path)
            except (OSError, ValueError) as error:
                messagebox.showwarning(self._t("app.gsplat_report"), str(error))

    def open_gsplat_folder(self) -> None:
        if self.latest_gsplat_model and self.latest_gsplat_model.is_dir():
            os.startfile(self.latest_gsplat_model)

    def open_report(self) -> None:
        if self.latest_result:
            report = self.latest_result / "result_quality.html"
            if not report.exists():
                report = self.latest_result / "demo_report.html"
            if report.exists():
                os.startfile(report)

    def open_data_folder(self) -> None:
        target = self.latest_result or self.latest_dataset or self.latest_photo
        if target and target.exists():
            os.startfile(target)

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
