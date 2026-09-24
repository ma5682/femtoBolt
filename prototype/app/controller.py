from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from PySide6.QtCore import QObject, QTimer, Signal
from app.app_state import AppState
from app.navigation import VIEWS, actions, navigation_reason
from models.capture_state import CaptureState
from models.planning_state import PlanningState
from models.task_state import RegionState
from mock.mock_patient import sample_patient
from mock.mock_capture import SCENARIOS, quality_passes
from mock.mock_planning import mock_route, sample_regions
from mock.plan_store import MockPlanStore


class WorkflowController(QObject):
    changed = Signal()
    history_changed = Signal()

    def __init__(self, records_dir: Path, parent=None):
        super().__init__(parent)
        self.state = AppState(patient=sample_patient())
        self.store = MockPlanStore(records_dir)
        self.undo_stack = []
        self._pending = ""
        self.project_data = None
        self.capture_timer = QTimer(self)
        self.capture_timer.setInterval(650)
        self.capture_timer.timeout.connect(self.tick_capture)
        self.operation_timer = QTimer(self)
        self.operation_timer.setSingleShot(True)
        self.operation_timer.timeout.connect(self.complete_operation)

    def publish(self):
        self.changed.emit()

    def message(self, text, tone="warning"):
        self.state.message, self.state.message_tone = text, tone
        self.publish()

    def set_scenario(self, scenario):
        if scenario not in SCENARIOS:
            return
        self.capture_timer.stop()
        self.operation_timer.stop()
        self._pending = ""
        self.undo_stack.clear()
        self.state = AppState(patient=sample_patient(), scenario=scenario)
        if scenario == "parameter":
            self.state.plan.parameters.direct_spacing = "120"
        self.publish()

    def load_replay(self, data):
        self.capture_timer.stop()
        self.operation_timer.stop()
        self.project_data = data
        s = AppState(patient=sample_patient(), replay_mode=True, replay_label=data.label)
        s.replay_stats = dict(data.trajectory.get("statistics", {}))
        s.replay_stats.update({
            "frame_count": data.report.get("frame_count", 0),
            "valid_depth_ratio_mean": data.report.get("metrics", {}).get("valid_depth_ratio_mean"),
            "created_local": data.report.get("created_local", ""),
            "execution_ready": data.trajectory.get("execution_ready", False),
            "physical_accuracy_verified": data.trajectory.get("physical_accuracy_verified", False),
            "treatment_coverage_claim": data.trajectory.get("statistics", {}).get("treatment_coverage_claim", ""),
            "geometry_support_ratio": data.region.get("region", {}).get("geometry_support", {}).get("support_ratio"),
        })
        s.replay_warnings = list(data.warnings)
        s.capture.phase = "passed"
        s.capture.frames = int(data.report.get("frame_count", 3))
        s.capture.reference_ready = True
        s.capture.model_revision = 1
        support = data.region.get("region", {}).get("selection_support", {})
        stats = data.trajectory.get("statistics", {})
        unconfirmed = int(stats.get("geometry_unconfirmed_pixel_count", support.get("geometry_unconfirmed_pixel_count", 0)) or 0)
        # A replay is read-only: unresolved geometry is a visible warning, not
        # silently presented as a fully verified region.
        s.region.support = "partial" if unconfirmed else "valid"
        s.region.geometry_repaired = True
        s.region.revision = 1
        s.plan.phase = "ready" if data.trajectory.get("statistics", {}).get("gap_count", 0) == 0 else "failed"
        s.plan.revision = 1
        s.plan.generated_at = datetime.fromtimestamp(data.trajectory_dir.stat().st_mtime).isoformat(timespec="seconds")
        s.plan.source_model_revision = 1
        s.plan.source_region_revision = 1
        s.plan.parameters.direct_spacing = str(data.trajectory.get("planner", {}).get("parameters", {}).get("row_spacing_mm", 5.0))
        if data.warnings:
            s.message = "项目回放已载入；原始记录保留校验提示，请按只读结果核对。"
            s.message_tone = "warning"
        self.state = s
        self.publish()

    def go(self, stage):
        reason = navigation_reason(self.state, stage)
        if reason:
            self.message(reason)
            return
        self.state.stage = stage
        self.state.view = VIEWS[stage][0]
        self.state.issue_focus = ""
        self.state.message = ""
        self.publish()

    def invalidate_plan(self):
        p = self.state.plan
        p.phase, p.segments, p.saved_id, p.saved_path = "empty", [], "", ""
        p.generated_at = ""

    def dispatch(self, key):
        primary, secondary = actions(self.state)
        allowed = {a.key for a in [primary, *secondary] if a.enabled}
        if key not in allowed:
            self.message("请先完成当前步骤的检查。")
            return
        if self.state.replay_mode:
            if key == "close_replay":
                self.project_data = None
                self.state = AppState(patient=sample_patient())
                self.publish()
                return
            if key.startswith("to_"):
                self.go({"to_capture": 0, "to_region": 1, "to_planning": 2, "to_confirm": 3}[key])
            return
        self.state.message = ""
        if key == "start":
            self.start_capture()
        elif key == "finish":
            self.capture_timer.stop()
            self.state.capture.phase = "checking"
            self.begin_operation("capture")
        elif key == "restart":
            params = deepcopy(self.state.plan.parameters)
            self.state.capture = CaptureState()
            self.state.region = RegionState()
            self.state.plan = PlanningState(parameters=params)
            self.undo_stack.clear()
            self.start_capture()
        elif key == "repair_capture":
            self.state.capture.repaired = True
            self.start_capture()
        elif key == "repair_geometry":
            self.state.capture.return_to_region = True
            self.state.stage = 0
            self.state.view = "点云"
            self.state.issue_focus = "geometry"
            self.start_capture()
        elif key.startswith("to_"):
            self.go({"to_capture": 0, "to_region": 1, "to_planning": 2}[key])
        elif key == "focus_capture":
            self.focus("capture")
        elif key == "undo":
            self.undo()
        elif key == "plan_region":
            self.state.region.support = "checking"
            self.begin_operation("region")
        elif key == "generate":
            self.generate()
        elif key == "repair_route":
            self.state.plan.repaired = True
            self.generate()
        elif key == "confirm":
            self.go(3)
        elif key == "save":
            self.save()

    def start_capture(self):
        self.invalidate_plan()
        self.state.capture.reference_ready = True
        self.state.capture.phase = "capturing"
        self.state.capture.frames = max(1, self.state.capture.frames)
        self.capture_timer.start()
        self.publish()

    def tick_capture(self):
        if self.state.capture.phase != "capturing":
            self.capture_timer.stop()
            return
        self.state.capture.frames = min(6, self.state.capture.frames + 1)
        self.publish()

    def begin_operation(self, name):
        self._pending = name
        self.operation_timer.start(650)
        self.publish()

    def complete_operation(self):
        pending, self._pending = self._pending, ""
        self.operation_timer.stop()
        s = self.state
        if pending == "capture":
            good = quality_passes(s.scenario, s.capture.repaired)
            s.capture.phase = "passed" if good else "failed"
            if good:
                s.capture.model_revision += 1
                if s.capture.return_to_region:
                    s.region.geometry_repaired = True
                    s.region.support = "valid"
                elif s.region.has_work:
                    s.region.support = "unchecked"
                s.issue_focus = ""
            else:
                s.issue_focus = "capture"
        elif pending == "region":
            missing = s.scenario == "geometry" and not s.region.geometry_repaired
            s.region.support = "missing" if missing else "valid"
            if missing:
                s.issue_focus = "geometry"
                s.view = "三维模型"
            else:
                s.stage, s.view = 2, VIEWS[2][0]
                self.generate()
                return
        elif pending == "plan":
            spacing, error = s.plan.parameters.spacing()
            if error:
                s.plan.phase = "empty"
                self.message(error, "error")
                return
            failed = s.scenario == "route" and not s.plan.repaired
            s.plan.segments = mock_route(s.region.strokes, spacing, failed)
            s.plan.phase = "failed" if failed or not s.plan.segments else "ready"
            s.plan.revision += 1
            s.plan.generated_at = datetime.now().isoformat(timespec="seconds")
            s.plan.source_model_revision = s.capture.model_revision
            s.plan.source_region_revision = s.region.revision
            s.issue_focus = "route" if failed else ""
        self.publish()

    def generate(self):
        s = self.state
        _, error = s.plan.parameters.spacing()
        if s.capture.phase != "passed" or s.region.support != "valid":
            self.message("当前模型或选区尚未通过检查。", "error")
            return
        self.invalidate_plan()
        if error:
            self.message(error, "error")
            return
        s.plan.phase = "generating"
        self.begin_operation("plan")

    def set_regions(self, strokes):
        if self.state.replay_mode or self.state.stage != 1 or self.state.busy:
            return
        self.undo_stack.append(deepcopy(self.state.region.strokes))
        self.undo_stack = self.undo_stack[-30:]
        self.state.region.strokes = deepcopy(strokes)
        self.region_modified()

    def region_modified(self):
        self.state.region.revision += 1
        self.state.region.support = "unchecked"
        self.state.issue_focus = ""
        self.state.message = ""
        self.invalidate_plan()
        self.publish()

    def undo(self):
        if self.state.stage == 1 and self.undo_stack and not self.state.busy:
            self.state.region.strokes = self.undo_stack.pop()
            self.region_modified()

    def clear_regions(self):
        if self.state.replay_mode:
            return
        self.set_regions([])

    def load_example(self):
        if self.state.replay_mode:
            return
        self.set_regions(sample_regions())

    def set_tool(self, tool):
        if not self.state.replay_mode and tool in ("work", "forbid", "erase") and not self.state.busy:
            self.state.tool = tool
            self.publish()

    def set_view(self, view):
        if view in VIEWS[self.state.stage]:
            self.state.view = view
            self.publish()

    def set_axes(self, enabled):
        self.state.show_axes = bool(enabled)
        self.publish()

    def set_parameter(self, name, value):
        if self.state.replay_mode or self.state.stage != 2 or self.state.busy or name not in ("mode", "direct_spacing", "effective_width", "overlap"):
            return
        params = self.state.plan.parameters
        if getattr(params, name) == value:
            return
        setattr(params, name, value)
        self.invalidate_plan()
        self.state.message = ""
        self.publish()

    def focus(self, issue):
        self.state.issue_focus = issue
        self.publish()

    def save(self):
        s = self.state
        if s.stage != 3 or not s.plan_current or s.plan.saved_id:
            self.message("当前规划尚未通过检查，或该版本已经保存。", "error")
            return
        identifier = "mock-plan-" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:6]
        spacing, _ = s.plan.parameters.spacing()
        record = {
            "id": identifier, "kind": "mock_planning_preview",
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "patient": asdict(s.patient), "capture": asdict(s.capture),
            "region": asdict(s.region), "plan": asdict(s.plan),
            "final_spacing_mm": spacing,
            "spacing_source": "直接指定" if s.plan.parameters.mode == "direct" else "按宽度计算",
            "parameters_are_placeholders": True,
            "unused_parameters": {"treatment_depth": None, "shell_diameter": None, "shell_clearance": None, "used": False},
            "execution_ready": False, "geometry_only": True,
        }
        try:
            path = self.store.save(record)
        except OSError as exc:
            self.message("保存失败，请检查原型记录目录是否可写：" + str(exc), "error")
            return
        s.plan.saved_id, s.plan.saved_path = identifier, str(path)
        self.publish()
        self.history_changed.emit()
