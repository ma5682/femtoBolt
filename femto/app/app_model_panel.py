"""3D model controls and actions for the main control app."""
from __future__ import annotations

import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import tkinter as tk
from tkinter import ttk

from femto.ui_i18n import tr


class ModelPanel:
    """Own model-related widgets, training actions, viewers, and reports."""

    def __init__(self, app: Any, dependencies: SimpleNamespace) -> None:
        self.app = app
        self.dependencies = dependencies

    def build(self, parent: Any) -> None:
        app = self.app
        app.view_button = app._button(
            parent,
            tr(app.language, "app.view_model"),
            app.view_model,
            "#6d48a8",
            "#8058c1",
        )
        app.view_button.pack(fill=tk.X, pady=(0, 4))
        self._build_gaussian_controls(parent)
        self._build_result_controls(parent)

    def _build_gaussian_controls(self, parent: Any) -> None:
        app = self.app
        gaussian_buttons = tk.Frame(parent, bg="#111a24")
        gaussian_buttons.pack(fill=tk.X, pady=(0, 4))
        gaussian_buttons.grid_columnconfigure(0, weight=1)
        gaussian_buttons.grid_columnconfigure(1, weight=1)
        app.three_dgs_profile_label = tk.Label(
            gaussian_buttons,
            text=app._t("app.3dgs_profile"),
            bg="#111a24",
            fg="#dbe7f2",
        )
        app.three_dgs_profile_label.grid(
            row=0,
            column=0,
            columnspan=2,
            sticky="w",
        )
        app.three_dgs_profile_combo = ttk.Combobox(
            gaussian_buttons,
            textvariable=app.three_dgs_profile_display_var,
            values=[
                app._t(f"app.3dgs_profile.{key}")
                for key in self.dependencies.THREE_DGS_PROFILE_KEYS
            ],
            state="readonly",
            font=("Microsoft YaHei UI", 9),
        )
        app.three_dgs_profile_combo.grid(
            row=1,
            column=0,
            columnspan=2,
            sticky="ew",
        )
        app.three_dgs_profile_combo.bind(
            "<<ComboboxSelected>>",
            app._three_dgs_profile_combo_changed,
        )
        app.train_3dgs_button = app._button(
            gaussian_buttons,
            tr(app.language, "app.train_3dgs"),
            app.train_3dgs,
            "#9b5d16",
            "#b66d19",
        )
        app.train_3dgs_button.grid(
            row=2,
            column=0,
            sticky="ew",
            padx=(0, 4),
        )
        app.view_3dgs_button = app._button(
            gaussian_buttons,
            tr(app.language, "app.view_3dgs"),
            app.view_3dgs,
            "#71449a",
            "#8551b5",
        )
        app.view_3dgs_button.grid(
            row=2,
            column=1,
            sticky="ew",
            padx=(4, 0),
        )
        app.gsplat_report_button = app._button(
            gaussian_buttons,
            tr(app.language, "app.gsplat_report"),
            app.open_gsplat_report,
            "#48596b",
            "#566b81",
        )
        app.gsplat_report_button.grid(
            row=3,
            column=0,
            sticky="ew",
            padx=(0, 4),
            pady=(4, 0),
        )
        app.gsplat_folder_button = app._button(
            gaussian_buttons,
            tr(app.language, "app.gsplat_folder"),
            app.open_gsplat_folder,
            "#394958",
            "#465a6d",
        )
        app.gsplat_folder_button.grid(
            row=3,
            column=1,
            sticky="ew",
            padx=(4, 0),
            pady=(4, 0),
        )
        app.moving_source_label = tk.Label(
            gaussian_buttons,
            bg="#111a24",
            fg="#a3b6c8",
            wraplength=600,
            anchor="w",
            justify="left",
        )
        app.moving_source_label.grid(
            row=4,
            column=0,
            columnspan=2,
            sticky="ew",
        )
        app.single_train_button = app._button(
            gaussian_buttons,
            app._t("app.single_train"),
            app.train_single_gs,
            "#9b5d16",
            "#b66d19",
        )
        app.single_train_button.grid(
            row=5,
            column=0,
            sticky="ew",
            padx=(0, 4),
            pady=(4, 0),
        )
        app.single_view_button = app._button(
            gaussian_buttons,
            app._t("app.single_view"),
            app.view_single_gs,
            "#71449a",
            "#8551b5",
        )
        app.single_view_button.grid(
            row=5,
            column=1,
            sticky="ew",
            padx=(4, 0),
            pady=(4, 0),
        )
        app.single_source_label = tk.Label(
            gaussian_buttons,
            bg="#111a24",
            fg="#a3b6c8",
            wraplength=600,
            anchor="w",
            justify="left",
        )
        app.single_source_label.grid(
            row=6,
            column=0,
            columnspan=2,
            sticky="ew",
        )

    def _build_result_controls(self, parent: Any) -> None:
        app = self.app
        result_buttons = tk.Frame(parent, bg="#111a24")
        result_buttons.pack(fill=tk.X, pady=(0, 5))
        result_buttons.grid_columnconfigure(0, weight=1)
        result_buttons.grid_columnconfigure(1, weight=1)
        app.report_button = app._button(
            result_buttons,
            tr(app.language, "app.report"),
            app.open_report,
            "#48596b",
            "#566b81",
        )
        app.report_button.grid(
            row=0,
            column=0,
            sticky="ew",
            padx=(0, 4),
        )
        app.folder_button = app._button(
            result_buttons,
            tr(app.language, "app.folder"),
            app.open_data_folder,
            "#394958",
            "#465a6d",
        )
        app.folder_button.grid(
            row=0,
            column=1,
            sticky="ew",
            padx=(4, 0),
        )

    def update_action_state(self) -> None:
        app = self.app
        if hasattr(app, "local_panel"):
            app.local_panel.refresh()
        camera_action = (
            tk.NORMAL
            if app.camera_ready
            and not app.processing
            and not app.capturing
            and not app.photo_saving
            else tk.DISABLED
        )
        app.photo_button.configure(state=camera_action)
        if hasattr(app, "temporal_entry_button"):
            app.temporal_entry_button.configure(state=camera_action)
        app.keyframe_quality_combo.configure(
            state=(
                "readonly"
                if camera_action == tk.NORMAL
                and app.scan_mode_var.get() == "keyframe"
                else "disabled"
            )
        )
        fixed = getattr(app, "latest_keyframe_result", None)
        moving = getattr(app, "latest_moving_result", None)
        app.single_train_button.configure(
            state=(
                tk.NORMAL
                if fixed
                and not app.processing
                and not app.capturing
                and not app.photo_saving
                else tk.DISABLED
            )
        )
        from femto.pipeline.reconstruction_sources import describe

        app.single_source_label.configure(
            text=app._t("app.single_source", source=describe(fixed))
        )
        app.moving_source_label.configure(
            text=app._t("app.moving_source", source=describe(moving))
        )
        app.single_view_button.configure(
            state=(
                tk.NORMAL
                if app.latest_single_gs and not app.processing and not app.capturing
                else tk.DISABLED
            )
        )
        app.point_cloud_button.configure(
            state=tk.NORMAL if app.camera_ready else tk.DISABLED
        )
        timed_capture = (
            camera_action == tk.NORMAL and app.scan_mode_var.get() != "keyframe"
        )
        app.duration_spin.configure(
            state=tk.NORMAL if timed_capture else tk.DISABLED
        )
        app.fps_spin.configure(state=tk.NORMAL if timed_capture else tk.DISABLED)
        combo_state = "readonly" if camera_action == tk.NORMAL else "disabled"
        app.scan_mode_combo.configure(state=combo_state)
        app.three_dgs_profile_combo.configure(
            state=(
                "readonly"
                if not app.processing and not app.capturing
                else "disabled"
            )
        )
        if getattr(app, "local_busy", False):
            app.scan_button.configure(
                text=app._t("app.scan_button.processing"),
                state=tk.DISABLED,
            )
        elif app.capturing or (app.keyframe_building and app.temporal_collecting):
            app.scan_button.configure(
                text=app._t("app.scan_button.stop"),
                state=tk.NORMAL,
                bg="#b65334",
            )
        elif app.processing:
            app.scan_button.configure(
                text=app._t("app.scan_button.processing"),
                state=tk.DISABLED,
                bg="#48596b",
            )
        else:
            app.scan_button.configure(
                text=app._scan_button_label(),
                state=camera_action,
                bg="#168f58",
            )
        app.view_button.configure(
            text=app._t(f"app.view_model.{app.scan_mode_var.get()}"),
            state=tk.NORMAL if self.selected_model_result() else tk.DISABLED,
        )
        app.train_3dgs_button.configure(
            state=(
                tk.NORMAL
                if moving
                and not app.processing
                and not app.capturing
                and not app.photo_saving
                else tk.DISABLED
            )
        )
        app.view_3dgs_button.configure(
            state=(
                tk.NORMAL
                if app.latest_gsplat_model and not app.processing
                else tk.DISABLED
            )
        )
        for button in (app.gsplat_report_button, app.gsplat_folder_button):
            button.configure(
                state=tk.NORMAL if app.latest_gsplat_model else tk.DISABLED
            )
        app.report_button.configure(
            state=tk.NORMAL if app.latest_result else tk.DISABLED
        )
        app.folder_button.configure(
            state=(
                tk.NORMAL
                if app.latest_photo or app.latest_dataset or app.latest_result
                else tk.DISABLED
            )
        )

    def profile_changed(self) -> None:
        selected = self.app.three_dgs_profile_display_var.get()
        for key in self.dependencies.THREE_DGS_PROFILE_KEYS:
            if self.app._t(f"app.3dgs_profile.{key}") == selected:
                self.app.three_dgs_profile_var.set(key)
                break

    def selected_model_result(self) -> Path | None:
        attribute = {
            "keyframe": "latest_keyframe_result",
            "moving": "latest_moving_result",
        }.get(self.app.scan_mode_var.get())
        return getattr(self.app, attribute, None) if attribute else None

    def view_model(self) -> None:
        result = self.selected_model_result()
        if not result:
            return
        reconstruction = result / "reconstruction"
        self.dependencies.subprocess.Popen(
            [
                sys.executable,
                "-m",
                "femto.viewer.scene_demo_viewer",
                str(reconstruction),
                "--lang",
                self.app.language,
            ],
            cwd=str(self.dependencies.ROOT),
            creationflags=(
                self.dependencies.subprocess.CREATE_NO_WINDOW
                if os.name == "nt"
                else 0
            ),
        )

    @staticmethod
    def format_metric(value: Any, spec: str, suffix: str = "") -> str:
        if not isinstance(value, (int, float)) or not np.isfinite(value):
            return "-"
        return f"{value:{spec}}{suffix}"

    def train_3dgs_worker(
        self,
        profile: str,
        demo_root: Path,
        iterations: int,
    ) -> None:
        app = self.app

        def on_progress(record: dict[str, Any]) -> None:
            app.event_queue.put(
                (
                    "3dgs_progress",
                    {
                        "step": int(record.get("step", 0)),
                        "total": iterations,
                        "gaussians": int(record.get("gaussians", 0)),
                        "cuda_reserved_mib": float(
                            record.get("cuda_reserved_mib", 0)
                        ),
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
            data = self.dependencies.assess_training_input(
                demo_root,
                require_real=True,
            )
            if not data["ready"]:
                raise ValueError(
                    app._t(
                        "app.dialog.3dgs_multiview_body",
                        reasons="\n".join(data["reasons"]),
                    )
                )
            runtime = self.dependencies.gsplat_runtime_status()
            if not runtime["ready"] or not runtime.get("lpips_ready", False):
                raise RuntimeError(
                    app._t("app.dialog.3dgs_runtime_body")
                    + "\n"
                    + str(
                        runtime.get("error")
                        or runtime.get("lpips_backbone", {}).get("error")
                    )
                )
            app.event_queue.put(
                (
                    "3dgs_ready",
                    {
                        "frames": data["frame_count"],
                        "span": data["translation_span_m"],
                        "profile": app._t(f"app.3dgs_profile.{profile}"),
                    },
                )
            )
            output_dir, report = self.dependencies.train_direct_gsplat(
                demo_root,
                profile=profile,
                on_progress=on_progress,
                require_real=True,
            )
            aggregate = (report.get("evaluation") or {}).get("aggregate") or {}
            depth_rmse_m = aggregate.get("depth_rmse_m")
            coverage = aggregate.get("depth_coverage_ratio")
            app.event_queue.put(
                (
                    "3dgs_done",
                    {
                        "output": str(output_dir),
                        "accepted": bool(report.get("accepted", False)),
                        "detail": {
                            "gaussians": f"{int(report.get('final_gaussians', 0)):,}",
                            "psnr": self.format_metric(
                                aggregate.get("psnr"),
                                ".2f",
                            ),
                            "ssim": self.format_metric(
                                aggregate.get("ssim"),
                                ".3f",
                            ),
                            "lpips": self.format_metric(
                                aggregate.get("lpips"),
                                ".4f",
                            ),
                            "depth_rmse": self.format_metric(
                                depth_rmse_m * 1000.0
                                if isinstance(depth_rmse_m, (int, float))
                                else None,
                                ".0f",
                                " mm",
                            ),
                            "coverage": self.format_metric(
                                coverage * 100.0
                                if isinstance(coverage, (int, float))
                                else None,
                                ".1f",
                                "%",
                            ),
                            "path": str(output_dir),
                        },
                    },
                )
            )
        except Exception as error:
            app.event_queue.put(
                (
                    "3dgs_error",
                    f"{error}\n\n{self.dependencies.traceback.format_exc(limit=4)}",
                )
            )

    def train_3dgs(self) -> None:
        app = self.app
        source = getattr(app, "latest_moving_result", None)
        if app.processing or app.capturing or app.photo_saving or not source:
            return
        profile = app.three_dgs_profile_var.get()
        app.processing = True
        app._set_status("app.3dgs_checking")
        app._set_detail("app.3dgs_checking_detail")
        app._set_action_state()
        self.dependencies.threading.Thread(
            target=app._train_3dgs_worker,
            args=(
                profile,
                source,
                int(self.dependencies.PROFILES[profile].iterations),
            ),
            name="FemtoDirectGsplatTraining",
            daemon=True,
        ).start()

    def train_single_gs(self) -> None:
        app = self.app
        source = getattr(app, "latest_keyframe_result", None)
        if app.processing or app.capturing or app.photo_saving or not source:
            return
        photo = source
        app.processing = True
        app.progress_var.set(0)
        app._set_status("app.single_training")
        app._set_raw_detail(str(photo))
        app._set_action_state()

        def worker() -> None:
            try:
                from femto.gsplat.scene_single_gs import launch

                output = launch(
                    photo,
                    on_progress=lambda row: app.event_queue.put(
                        ("single_gs_progress", row)
                    ),
                )
                app.event_queue.put(("single_gs_done", str(output)))
            except Exception as error:
                app.event_queue.put(("single_gs_error", str(error)))

        self.dependencies.threading.Thread(
            target=worker,
            name="FemtoSingleGs",
            daemon=True,
        ).start()

    def view_single_gs(self) -> None:
        app = self.app
        if not app.latest_single_gs:
            return
        try:
            from femto.gsplat.scene_single_gs_viewer import open_viewer

            process = open_viewer(app.latest_single_gs, app.language)
            output = app.latest_single_gs

            def check() -> None:
                if process.poll() not in (None, 0):
                    self.dependencies.messagebox.showerror(
                        "3DGS",
                        str(output / "viewer_stdout.log"),
                    )

            app.root.after(15000, check)
        except Exception as error:
            self.dependencies.messagebox.showerror("3DGS", str(error))

    def view_3dgs(self) -> None:
        app = self.app
        if not app.latest_gsplat_model:
            return
        try:
            process = self.dependencies.open_gsplat_viewer(
                app.latest_gsplat_model,
                language=app.language,
            )
            app._set_status("app.3dgs_view_start")
            app._set_detail("app.3dgs_view_detail")

            def check_startup() -> None:
                if process.poll() not in (None, 0):
                    self.dependencies.messagebox.showwarning(
                        app._t("app.view_3dgs"),
                        app._t(
                            "app.3dgs_view_failed",
                            path=str(
                                app.latest_gsplat_model / "viewer_stdout.log"
                            ),
                        ),
                    )

            app.root.after(15000, check_startup)
        except Exception as error:
            self.dependencies.messagebox.showwarning(
                app._t("app.view_3dgs"),
                str(error),
            )

    def open_gsplat_report(self) -> None:
        app = self.app
        if app.latest_gsplat_model:
            from femto.gsplat.scene_gsplat_acceptance import build_acceptance_report

            path = app.latest_gsplat_model / "acceptance_report.json"
            try:
                self.dependencies.json_dump(
                    path,
                    build_acceptance_report(app.latest_gsplat_model),
                )
                os.startfile(path)
            except (OSError, ValueError) as error:
                self.dependencies.messagebox.showwarning(
                    app._t("app.gsplat_report"),
                    str(error),
                )

    def open_gsplat_folder(self) -> None:
        model = self.app.latest_gsplat_model
        if model and model.is_dir():
            os.startfile(model)

    def open_report(self) -> None:
        result = self.app.latest_result
        if result:
            report = result / "result_quality.html"
            if not report.exists():
                report = result / "demo_report.html"
            if report.exists():
                os.startfile(report)

    def open_data_folder(self) -> None:
        app = self.app
        target = app.latest_result or app.latest_dataset or app.latest_photo
        if target and target.exists():
            os.startfile(target)
