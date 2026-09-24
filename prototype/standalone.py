"""Generated standalone mock workstation. Run: python standalone.py"""

from dataclasses import dataclass


@dataclass
class CaptureState:
    phase: str = "idle"
    frames: int = 0
    reference_ready: bool = False
    repaired: bool = False
    return_to_region: bool = False
    model_revision: int = 0

    @property
    def stability(self) -> int:
        return 0 if self.phase == "idle" else 46 if self.phase == "failed" else 92 if self.phase == "passed" else min(88, 62 + self.frames * 4)

    @property
    def support(self) -> int:
        return 0 if self.phase == "idle" else 71 if self.phase == "failed" else 96 if self.phase == "passed" else min(93, self.frames * 16)

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Patient:
    name: str = "张*某"
    sex: str = "女"
    age: int = 38
    identifier: str = "PT-202609-001"
    task: str = "腹部区域三维规划"


@dataclass
class Stroke:
    kind: str
    points: list[tuple[float, float]]


@dataclass
class RegionState:
    strokes: list[Stroke] = field(default_factory=list)
    support: str = "unchecked"
    geometry_repaired: bool = False
    revision: int = 0

    @property
    def has_work(self) -> bool:
        return any(s.kind == "work" and s.points for s in self.strokes)

    @property
    def has_forbidden(self) -> bool:
        return any(s.kind == "forbid" and s.points for s in self.strokes)

from dataclasses import dataclass, field
from math import isfinite


@dataclass
class PlanningParameters:
    mode: str = "direct"
    direct_spacing: str = "2.4"
    effective_width: str = "3.0"
    overlap: str = "20"

    def spacing(self) -> tuple[float | None, str]:
        try:
            if self.mode == "direct":
                value = float(self.direct_spacing)
            else:
                width, overlap = float(self.effective_width), float(self.overlap)
                if not isfinite(width) or width <= 0:
                    return None, "有效作用宽度必须大于 0 mm。"
                if not isfinite(overlap) or not 0 <= overlap <= 90:
                    return None, "重叠率须在 0–90% 之间。"
                value = width * (1 - overlap / 100)
            if not isfinite(value) or not 0.5 <= value <= 100:
                return None, "最终行距须在 0.5–100 mm 之间。"
            return value, ""
        except (ValueError, TypeError):
            return None, "请输入有效数值。"


@dataclass
class RouteSegment:
    kind: str
    points: list[tuple[float, float]]


@dataclass
class PlanningState:
    parameters: PlanningParameters = field(default_factory=PlanningParameters)
    phase: str = "empty"
    segments: list[RouteSegment] = field(default_factory=list)
    repaired: bool = False
    revision: int = 0
    generated_at: str = ""
    source_model_revision: int = -1
    source_region_revision: int = -1
    saved_id: str = ""
    saved_path: str = ""

from dataclasses import dataclass, field


@dataclass
class AppState:
    patient: Patient = field(default_factory=Patient)
    capture: CaptureState = field(default_factory=CaptureState)
    region: RegionState = field(default_factory=RegionState)
    plan: PlanningState = field(default_factory=PlanningState)
    stage: int = 0
    scenario: str = "normal"
    view: str = "RGB"
    tool: str = "work"
    issue_focus: str = ""
    message: str = ""
    message_tone: str = "normal"
    show_axes: bool = False
    replay_mode: bool = False
    replay_label: str = ""
    replay_stats: dict = field(default_factory=dict)
    replay_warnings: list[str] = field(default_factory=list)

    @property
    def quality_state(self) -> str:
        """Compact gate state used by the prototype UI: pass, warn or block."""
        if self.replay_mode:
            if self.capture.phase == "failed" or self.region.support == "missing":
                return "block"
            if self.region.support == "partial" or self.plan.phase == "failed":
                return "warn"
            return "pass"
        if self.capture.phase == "failed" or self.region.support == "missing":
            return "block"
        if self.capture.phase in ("checking",) or self.region.support in ("checking",):
            return "warn"
        if self.plan.phase == "failed":
            return "warn"
        return "pass"

    @property
    def busy(self) -> bool:
        return self.capture.phase == "checking" or self.region.support == "checking" or self.plan.phase == "generating"

    @property
    def plan_current(self) -> bool:
        return (
            self.plan.phase == "ready"
            and self.capture.phase == "passed"
            and self.region.support == "valid"
            and self.plan.source_model_revision == self.capture.model_revision
            and self.plan.source_region_revision == self.region.revision
            and not self.plan.parameters.spacing()[1]
        )



def sample_patient() -> Patient:
    return Patient()

"""Deterministic demonstration values; no sensor access."""

SCENARIOS = {
    "normal": "正常流程",
    "quality": "数据缺失需补采",
    "geometry": "三维几何缺失",
    "route": "禁区干涉 / 轨迹断开",
    "parameter": "参数超限",
}


def quality_passes(scenario: str, repaired: bool) -> bool:
    return scenario != "quality" or repaired

"""2D demonstration lines in brush coordinates, not a 3D treatment planner."""
from math import hypot


def distance_to_stroke(x: float, y: float, stroke: Stroke) -> float:
    if not stroke.points:
        return float("inf")
    best = min(hypot(x - px, y - py) for px, py in stroke.points)
    for a, b in zip(stroke.points, stroke.points[1:]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        t = max(0, min(1, ((x-a[0])*dx + (y-a[1])*dy) / (dx*dx + dy*dy or 1)))
        best = min(best, hypot(x-a[0]-t*dx, y-a[1]-t*dy))
    return best


def sample_regions() -> list[Stroke]:
    return [
        Stroke("work", [(325, 300), (495, 300), (495, 344), (325, 344), (325, 388), (495, 388), (495, 432), (325, 432)]),
        Stroke("forbid", [(555, 355), (555, 407)]),
    ]


def mock_route(strokes: list[Stroke], spacing: float, broken: bool) -> list[RouteSegment]:
    work = [s for s in strokes if s.kind == "work"]
    forbid = [s for s in strokes if s.kind == "forbid"]
    points = [p for s in work for p in s.points]
    if not points:
        return []
    def allowed(x, y):
        return any(distance_to_stroke(x, y, s) <= 29 for s in work) and all(distance_to_stroke(x, y, s) > 36 for s in forbid)
    x0, x1 = min(x for x, _ in points)-28, max(x for x, _ in points)+28
    y0, y1 = min(y for _, y in points)-22, max(y for _, y in points)+22
    # Visual density responds to spacing. This is an illustrative scale only.
    pitch = max(3, spacing * 6)
    rows = max(1, int((y1-y0)/pitch)+1)
    result = []
    previous = None
    for row in range(rows):
        y = (y0+y1)/2 if rows == 1 else y0 + row*pitch
        runs, run = [], []
        for col in range(int((x1-x0)/3)+1):
            x = x0 + col*3
            if allowed(x, y):
                run.append((x, y))
            elif run:
                runs.append(run)
                run = []
        if run:
            runs.append(run)
        if row % 2:
            runs = [list(reversed(r)) for r in reversed(runs)]
        for run in runs:
            if len(run) < 2:
                continue
            start, end = run[0], run[-1]
            if previous:
                samples = [(previous[0]+(start[0]-previous[0])*i/30, previous[1]+(start[1]-previous[1])*i/30) for i in range(31)]
                if all(allowed(x, yy) for x, yy in samples):
                    result.append(RouteSegment("connect", [previous, start]))
            if broken and row == rows//2:
                mid = ((start[0]+end[0])/2, y)
                direction = 1 if end[0] > start[0] else -1
                result.extend([RouteSegment("scan", [start, (mid[0]-14*direction, y)]), RouteSegment("scan", [(mid[0]+14*direction, y), end])])
            else:
                result.append(RouteSegment("scan", [start, end]))
            previous = end
    return result

"""Persist demonstration plans only; never accesses production outputs."""
import json
import os
from pathlib import Path


class MockPlanStore:
    def __init__(self, directory: Path):
        self.directory = directory

    def save(self, record: dict) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / (record["id"] + ".json")
        temporary = path.with_suffix(".tmp")
        try:
            temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
        return path

    def records(self) -> list[dict]:
        records = []
        if self.directory.exists():
            for path in self.directory.glob("mock-plan-*.json"):
                try:
                    item = json.loads(path.read_text(encoding="utf-8"))
                    if item.get("kind") == "mock_planning_preview":
                        records.append(item)
                except (OSError, ValueError, TypeError):
                    continue
        return sorted(records, key=lambda r: r.get("saved_at", ""), reverse=True)

from dataclasses import dataclass

STAGES = ("三维采集", "区域标记", "轨迹规划", "运行预览")
VIEWS = (("RGB", "深度", "点云", "模型"), ("固定 RGB", "三维模型"), ("三维模型", "RGB 路线图"), ("三维模型", "RGB 路线图"))


@dataclass(frozen=True)
class Action:
    key: str
    text: str
    enabled: bool = True
    reason: str = ""
    destructive: bool = False


def navigation_reason(s: AppState, stage: int) -> str:
    if not 0 <= stage < 4:
        return "无效步骤。"
    if s.replay_mode:
        return ""
    if s.busy:
        return "正在检查，请稍候。"
    if s.capture.phase == "capturing" and stage != 0:
        return "请先完成采集和质量检查。"
    if stage >= 1 and s.capture.phase != "passed":
        return "请先完成采集质量检查。"
    if stage >= 2 and s.region.support != "valid":
        return "请先标记规划区并通过三维支持检查。"
    if stage == 3 and not s.plan_current:
        return "请先生成轨迹并处理阻断项。"
    return ""


def actions(s: AppState) -> tuple[Action, list[Action]]:
    if s.replay_mode:
        if s.stage == 0:
            return Action("to_region", "下一步：区域标记  →"), [Action("close_replay", "退出项目回放")]
        if s.stage == 1:
            return Action("to_planning", "查看轨迹规划"), [Action("to_capture", "返回采集数据"), Action("close_replay", "退出项目回放")]
        if s.stage == 2:
            return Action("to_confirm", "查看规划确认"), [Action("to_region", "返回规划区"), Action("close_replay", "退出项目回放")]
        return Action("close_replay", "退出项目回放"), [Action("to_planning", "返回轨迹规划")]
    if s.stage == 0:
        if s.capture.phase == "idle":
            return Action("start", "建立参考位置并开始采集"), []
        if s.capture.phase == "capturing":
            return Action("finish", "完成采集并检查质量", s.capture.frames >= 3, "至少等待 3 个模拟视角"), [Action("restart", "重新开始", destructive=True)]
        if s.capture.phase == "checking":
            return Action("", "正在检查采集质量…", False, "检查完成后继续"), []
        if s.capture.phase == "failed":
            return Action("repair_capture", "继续补采"), [Action("focus_capture", "查看缺失位置")]
        return Action("to_region", "返回已画区域" if s.region.has_work else "进入区域标记"), [Action("restart", "重新采集", destructive=True)]
    if s.stage == 1:
        if s.region.support == "checking":
            return Action("", "正在检查三维支持…", False, "检查完成后继续"), []
        if s.region.support == "missing":
            return Action("repair_geometry", "返回补采"), [Action("undo", "撤销上一笔")]
        return Action("plan_region", "确认标记并检查三维几何", s.region.has_work, "请先在画面中标出规划区"), [Action("to_capture", "返回采集")]
    if s.stage == 2:
        if s.plan.phase == "generating":
            return Action("", "正在计算路径…", False, "正在计算规划预览"), []
        if s.plan.phase == "failed":
            return Action("repair_route", "重新规划"), [Action("to_region", "修改区域")]
        if not s.plan_current:
            _, error = s.plan.parameters.spacing()
            return Action("generate", "计算轨迹路径", not error, error), [Action("to_region", "修改区域")]
        return Action("confirm", "进入最终确认"), [Action("to_region", "修改区域")]
    return Action("save", "规划方案已保存" if s.plan.saved_id else "保存规划方案（预览）", s.plan_current and not s.plan.saved_id, "当前版本已保存" if s.plan.saved_id else "请先处理阻断项"), [Action("to_region", "修改区域"), Action("to_planning", "修改规划参数")]

from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from PySide6.QtCore import QObject, QTimer, Signal


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

"""Read-only adapter for the repository's published reconstruction artifacts.

The adapter validates the evidence links before exposing anything to the UI. It
does not call the production planner and never writes to the source folders.
"""
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import struct


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() and path.exists() else root / value


@dataclass
class ProjectData:
    label: str
    dataset: Path
    reconstruction: Path
    rgb_path: Path
    depth_path: Path
    mesh_path: Path
    region_mesh_path: Path
    points_path: Path
    region_path: Path
    trajectory_dir: Path
    report: dict
    local_geometry: dict
    region: dict
    trajectory: dict
    vertices: list[tuple[float, float, float]] = field(default_factory=list)
    colors: list[tuple[int, int, int]] = field(default_factory=list)
    faces: list[tuple[int, int, int]] = field(default_factory=list)
    region_vertices: list[tuple[float, float, float]] = field(default_factory=list)
    region_colors: list[tuple[int, int, int]] = field(default_factory=list)
    region_faces: list[tuple[int, int, int]] = field(default_factory=list)
    cloud_points: list[tuple[float, float, float]] = field(default_factory=list)
    cloud_colors: list[tuple[int, int, int]] = field(default_factory=list)
    route_groups: list[tuple[str, list[tuple[float, float, float]]]] = field(default_factory=list)
    verified_files: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _read_binary_ply(path: Path, load_faces: bool = False):
    with path.open("rb") as stream:
        header = []
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"PLY header incomplete: {path.name}")
            decoded = line.decode("ascii", errors="strict").strip()
            header.append(decoded)
            if decoded == "end_header":
                break
        if "format binary_little_endian 1.0" not in header:
            raise ValueError(f"仅支持 binary_little_endian PLY: {path.name}")
        vertex_count = next(int(row.split()[2]) for row in header if row.startswith("element vertex "))
        face_count = next((int(row.split()[2]) for row in header if row.startswith("element face ")), 0)
        properties = []
        in_vertex = False
        for row in header:
            if row == "element vertex " + str(vertex_count):
                in_vertex = True
                continue
            if row.startswith("element ") and in_vertex:
                break
            if in_vertex and row.startswith("property "):
                properties.append(row.split()[1:])
        formats = {"double": "d", "float": "f", "uchar": "B", "uint8": "B"}
        try:
            fmt = "<" + "".join(formats[item[0]] for item in properties)
        except KeyError as exc:
            raise ValueError(f"PLY property unsupported: {exc}") from exc
        stride = struct.calcsize(fmt)
        xyz_indices = [index for index, item in enumerate(properties) if item[-1] in ("x", "y", "z")]
        rgb_indices = [index for index, item in enumerate(properties) if item[-1] in ("red", "green", "blue")]
        vertices, colors = [], []
        for _ in range(vertex_count):
            record = stream.read(stride)
            if len(record) != stride:
                raise ValueError(f"PLY vertex data incomplete: {path.name}")
            values = struct.unpack(fmt, record)
            vertices.append(tuple(float(values[index]) for index in xyz_indices))
            colors.append(tuple(int(values[index]) for index in rgb_indices) if len(rgb_indices) == 3 else (136, 154, 165))
        faces = []
        if load_faces:
            for _ in range(face_count):
                count_bytes = stream.read(1)
                if not count_bytes:
                    raise ValueError(f"PLY face data incomplete: {path.name}")
                count = struct.unpack("<B", count_bytes)[0]
                indices = struct.unpack("<" + "I" * count, stream.read(4 * count))
                if count >= 3:
                    faces.append((int(indices[0]), int(indices[1]), int(indices[2])))
        return vertices, colors, faces


def _choose_annotation(reconstruction: Path) -> tuple[Path, Path]:
    candidates = []
    for region_path in reconstruction.glob("annotations/*/region_1.json"):
        folder = region_path.parent
        trajectories = list(folder.glob("trajectory_*/trajectory.json"))
        if trajectories:
            candidate = max(trajectories, key=lambda item: item.stat().st_mtime)
            candidates.append((candidate.stat().st_mtime, region_path, candidate.parent))
    if not candidates:
        raise FileNotFoundError("没有找到带轨迹的区域导出记录")
    _, region_path, trajectory_dir = max(candidates, key=lambda item: item[0])
    return region_path, trajectory_dir


def _verify(data: ProjectData) -> None:
    verified = data.verified_files
    expected_region_file = data.region.get("file_sha256", {}).get(data.region_mesh_path.name)
    if expected_region_file and _sha256(data.region_mesh_path) != expected_region_file:
        data.warnings.append("区域导出网格 SHA-256 与区域记录不一致，未标记为可信回放。")
    elif data.region_mesh_path.exists():
        verified[str(data.region_mesh_path)] = _sha256(data.region_mesh_path)
    else:
        data.warnings.append("区域导出网格不存在，无法核验轨迹来源。")
    if data.mesh_path.exists():
        verified[str(data.mesh_path)] = _sha256(data.mesh_path)
    region_file_hashes = data.region.get("file_sha256", {})
    for filename, expected in region_file_hashes.items():
        path = data.region_path.parent / filename
        if not path.exists() or _sha256(path) != expected:
            data.warnings.append(f"区域文件校验失败：{filename}")
        else:
            verified[str(path)] = expected
    files_manifest = data.trajectory_dir / "trajectory_files.json"
    if files_manifest.exists():
        manifest = _json(files_manifest)
        for filename, expected in manifest.get("files", {}).items():
            path = data.trajectory_dir / filename
            if not path.exists() or _sha256(path) != expected:
                data.warnings.append(f"轨迹文件校验失败：{filename}")
            else:
                verified[str(path)] = expected
    source = data.trajectory.get("source", {})
    if source.get("region_manifest_sha256") and _sha256(data.region_path) != source["region_manifest_sha256"]:
        data.warnings.append("轨迹来源区域记录校验失败。")
    if source.get("mesh_file_sha256") and _sha256(data.region_mesh_path) != source["mesh_file_sha256"]:
        data.warnings.append("轨迹来源区域网格校验失败。")
    region = data.region.get("region", {})
    reference = region.get("view_reference", {})
    for path, expected, label in (
        (data.rgb_path, reference.get("rgb_sha256"), "参考 RGB"),
        (data.depth_path, reference.get("depth_sha256"), "参考深度"),
    ):
        if expected and _sha256(path) != expected:
            data.warnings.append(f"{label}校验失败。")
        elif expected:
            verified[str(path)] = expected
    origin = data.local_geometry.get("reference_provenance", {})
    pose = data.dataset / "derived" / "local_posed" / "pose" / "000000.json"
    if origin.get("reference_frame_id") != 0 or origin.get("unit") != "metre" or origin.get("coordinate_frame") != "world_rdf":
        data.warnings.append("W0/RDF/米坐标绑定不完整。")
    if origin.get("reference_pose_sha256") and pose.exists() and _sha256(pose) != origin["reference_pose_sha256"]:
        data.warnings.append("参考位姿 SHA-256 校验失败。")
    if data.trajectory.get("execution_ready") is not False:
        data.warnings.append("轨迹执行就绪标记不符合当前项目约束。")
    if data.region.get("physical_accuracy_verified") is not False:
        data.warnings.append("区域物理精度验证状态不是 false。")


def _read_route(trajectory: dict):
    groups = []
    for item in trajectory.get("ordered_items", []):
        if item.get("kind") != "scan_segment":
            continue
        points = [tuple(float(v) for v in point["xyz_m"]) for point in item.get("points", []) if point.get("xyz_m")]
        if len(points) >= 2:
            groups.append(("scan", points))
    return groups


def load_project_data(project_root: Path, reconstruction: Path | None = None) -> ProjectData:
    project_root = Path(project_root)
    dataset = next(project_root.glob("datasets/scene_*_local_*"), None)
    reconstruction = Path(reconstruction) if reconstruction else project_root / "outputs" / "complete_demo_20260917_102931" / "reconstruction"
    if dataset is None or not reconstruction.exists():
        raise FileNotFoundError("没有找到可回放的 datasets/scene_* 或 reconstruction")
    region_path, trajectory_dir = _choose_annotation(reconstruction)
    region = _json(region_path)
    trajectory = _json(trajectory_dir / "trajectory.json")
    mesh_path = reconstruction / "scene_tsdf_mesh_m.ply"
    region_mesh_path = region_path.parent / "region_1_mesh_m.ply"
    points_path = reconstruction / "scene_fused_points_m.ply"
    rgb_path = dataset / "rgb" / "000000.png"
    depth_path = dataset / "depth" / "000000.png"
    report = _json(reconstruction / "reconstruction_report.json")
    local_geometry = _json(reconstruction / "local_geometry.json")
    vertices, colors, faces = _read_binary_ply(mesh_path, load_faces=True)
    region_vertices, region_colors, region_faces = _read_binary_ply(region_mesh_path, load_faces=True)
    cloud_points, cloud_colors, _ = _read_binary_ply(points_path, load_faces=False)
    data = ProjectData(
        label=f"项目回放 · {dataset.name}", dataset=dataset, reconstruction=reconstruction,
        rgb_path=rgb_path, depth_path=depth_path, mesh_path=mesh_path, points_path=points_path,
        region_mesh_path=region_mesh_path,
        region_path=region_path, trajectory_dir=trajectory_dir, report=report,
        local_geometry=local_geometry, region=region, trajectory=trajectory,
        vertices=vertices, colors=colors, faces=faces,
        region_vertices=region_vertices, region_colors=region_colors, region_faces=region_faces,
        cloud_points=cloud_points,
        cloud_colors=cloud_colors, route_groups=_read_route(trajectory),
    )
    _verify(data)
    return data

"""Display derivatives in memory; originals are never rewritten."""
import json
import struct
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtCore import QRect


def depth_heatmap(path, low_mm=250, high_mm=2210):
    raw = QImage(str(path))
    if raw.isNull():
        raise ValueError("无法读取深度图像")
    raw = raw.convertToFormat(QImage.Format_Grayscale16)
    source = bytes(raw.constBits())
    stride = raw.bytesPerLine()
    sample = max(1, raw.width() // 960)
    width, height = raw.width() // sample, raw.height() // sample
    output = bytearray(width*height*3)
    palette = []
    for i in range(256):
        color = QColor.fromHsvF((1-i/255)*0.67, .83, .96)
        palette.append((color.red(), color.green(), color.blue()))
    for y in range(height):
        values = struct.unpack_from("<"+"H"*raw.width(), source, y*sample*stride)
        for x in range(width):
            mm = values[x*sample]
            color = (10, 15, 20) if mm == 0 else palette[min(255, max(0, int((mm-low_mm)/(high_mm-low_mm)*255)))]
            at = (y*width+x)*3
            output[at:at+3] = bytes(color)
    return QImage(bytes(output), width, height, width*3, QImage.Format_RGB888).copy()


def region_image(data):
    """Nearest-neighbour RGB remap into the exported region's undistorted pixels."""
    region = data.region["region"]
    calibration = data.dataset / "calibration"
    intrinsic = json.loads((calibration / "color_intrinsic.json").read_text(encoding="utf-8"))
    distortion = json.loads((calibration / "distortion.json").read_text(encoding="utf-8"))["rgb"]
    if "BROWN_CONRADY" not in distortion.get("model", ""):
        raise ValueError("当前回放不支持该 RGB 畸变模型")
    k = region["view_reference"]["undistorted_intrinsic"]
    original = QImage(str(data.rgb_path)).convertToFormat(QImage.Format_RGB888)
    if original.isNull():
        raise ValueError("无法读取参考 RGB")
    full_w, full_h = region["view_reference"]["image_size"]
    if (original.width(), original.height()) != (full_w, full_h):
        raise ValueError("参考 RGB 尺寸与选区记录不一致")
    # Half-resolution is a display-only derivative in the same pixel coordinate frame.
    sample = 2
    w, h = full_w // sample, full_h // sample
    raw = bytes(original.constBits())
    stride = original.bytesPerLine()
    output = bytearray(w*h*3)
    d = distortion
    for y in range(h):
        yn = (y*sample-k[1][2])/k[1][1]
        yy = yn*yn
        for x in range(w):
            xn = (x*sample-k[0][2])/k[0][0]
            r2 = xn*xn+yy
            radial = (1+d["k1"]*r2+d["k2"]*r2*r2+d["k3"]*r2*r2*r2)/(1+d["k4"]*r2+d["k5"]*r2*r2+d["k6"]*r2*r2*r2)
            xd = xn*radial+2*d["p1"]*xn*yn+d["p2"]*(r2+2*xn*xn)
            yd = yn*radial+d["p1"]*(r2+2*yy)+2*d["p2"]*xn*yn
            sx = round(xd*intrinsic["fx"]+intrinsic["cx"])
            sy = round(yd*intrinsic["fy"]+intrinsic["cy"])
            if 0 <= sx < full_w and 0 <= sy < full_h:
                at = (y*w+x)*3
                source_at = sy*stride+sx*3
                output[at:at+3] = raw[source_at:source_at+3]
    base = QImage(bytes(output), w, h, w*3, QImage.Format_RGB888).copy()
    selected = base.copy()
    p = QPainter(selected)
    runs = region["selection_support"]["class_runs"]
    offset = 0
    for cls, count in zip(runs[::2], runs[1::2]):
        end = offset+count
        if cls:
            color = QColor(22, 197, 125, 75) if cls == 1 else QColor(193, 95, 248, 220)
            start = offset
            while start < end:
                row, col = divmod(start, full_w)
                stop = min(end, (row+1)*full_w)
                if row % sample == 0:
                    p.fillRect(QRect(col//sample, row//sample, max(1, (stop-start+sample-1)//sample), 1), color)
                start = stop
        offset = end
    p.end()
    if offset != full_w*full_h:
        raise ValueError("选区支持分类长度不正确")
    return base, selected

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget


def text_label(text="", name="", wrap=False):
    label = QLabel(text)
    label.setObjectName(name)
    label.setWordWrap(wrap)
    return label


def tone(widget, value):
    if widget.property("tone") != value:
        widget.setProperty("tone", value)
        widget.style().unpolish(widget)
        widget.style().polish(widget)


class StatusPanel(QWidget):
    focus_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("statusPanel")
        self.column = QVBoxLayout(self)
        self.column.setContentsMargins(14, 14, 14, 12)
        self.column.setSpacing(8)

    def section(self, title):
        icons = {"采集信息": "▣", "数据质量": "▥", "空间数据": "◇", "采集状态": "▣", "三维模型检查": "◇", "规划参数": "⌘", "规划预览": "□", "运行预览": "□"}
        label = text_label(f"{icons.get(title, '·')}  {title}", "sectionTitle")
        self.column.addWidget(label)
        return label

    def divider(self):
        line = QFrame()
        line.setObjectName("divider")
        line.setFixedHeight(1)
        self.column.addWidget(line)

    def row(self, title):
        item = QWidget()
        layout = QHBoxLayout(item)
        layout.setContentsMargins(0, 3, 0, 3)
        layout.setSpacing(8)
        layout.addWidget(text_label(title, "rowName"))
        value = text_label("—", "rowValue")
        value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(value, 1)
        self.column.addWidget(item)
        return value

    def note(self, text="", kind="muted"):
        label = text_label(text, kind, True)
        self.column.addWidget(label)
        return label

    def issue(self, text, key):
        button = QPushButton(text)
        button.setObjectName("issueButton")
        button.setMinimumHeight(44)
        button.clicked.connect(lambda: self.focus_requested.emit(key))
        self.column.addWidget(button)
        return button

    def finish(self):
        self.column.addStretch(1)

    def value(self, widget, text, color=""):
        widget.setText(text)
        tone(widget, color)

from pathlib import Path
from PySide6.QtCore import Qt, Signal, QSignalBlocker
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QComboBox, QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout


class PatientHeader(QFrame):
    scenario_requested = Signal(str)
    history_requested = Signal()
    settings_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("patientHeader")
        self.setFixedHeight(88)
        row = QHBoxLayout(self)
        row.setContentsMargins(18, 12, 18, 12)
        row.setSpacing(12)
        logo = QLabel()
        logo.setObjectName("brandLogo")
        logo.setFixedSize(148, 46)
        asset = Path(__file__).resolve().parent / "assets" / "uziyuan_logo.png"
        if not asset.exists():
            asset = Path(__file__).resolve().parent.parent / "assets" / "uziyuan_logo.png"
        pixmap = QPixmap(str(asset))
        if not pixmap.isNull():
            logo.setPixmap(pixmap.scaled(142, 40, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            logo.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        row.addWidget(logo)
        separator = QFrame()
        separator.setObjectName("headerSeparator")
        separator.setFrameShape(QFrame.VLine)
        row.addWidget(separator)
        brand = QVBoxLayout()
        brand.setSpacing(1)
        brand.addWidget(text_label("三维采集与轨迹规划工作台", "brand"))
        brand.addWidget(text_label("医疗三维工作站", "brandCaption"))
        row.addLayout(brand)
        patient = QVBoxLayout()
        patient.setSpacing(4)
        self.patient = text_label("", "patientName")
        self.task = text_label("", "muted")
        patient.addWidget(self.patient)
        patient.addWidget(self.task)
        row.addLayout(patient, 1)
        device = QVBoxLayout()
        device.setSpacing(4)
        self.device_status = text_label("●  相机已连接", "deviceReady")
        self.sensor_status = text_label("RGB 正常  ·  深度正常  /  模拟", "muted")
        device.addWidget(self.device_status)
        device.addWidget(self.sensor_status)
        self.source = text_label("Mock 演示", "modeBadge")
        device.addWidget(self.source)
        # Connection/quality telemetry is shown once in the bottom status bar;
        # repeating it in the title bar makes the header feel like a dashboard.
        device_widget = QFrame()
        device_widget.setLayout(device)
        device_widget.setVisible(False)
        row.addWidget(device_widget)
        demo = QVBoxLayout()
        demo.setSpacing(4)
        demo.addWidget(text_label("研发场景模拟", "muted"))
        self.scenario = QComboBox()
        self.scenario.setObjectName("scenario")
        for key, name in SCENARIOS.items():
            self.scenario.addItem(name, key)
        self.scenario.addItem("项目回放 · 只读", "replay")
        self.scenario.setMinimumWidth(124)
        self.scenario.currentIndexChanged.connect(lambda: self.scenario_requested.emit(self.scenario.currentData()))
        demo.addWidget(self.scenario)
        row.addLayout(demo)
        for text, signal in (("历史", self.history_requested), ("设置", self.settings_requested)):
            button = QPushButton(text)
            button.setObjectName("quietButton")
            button.setMinimumSize(52, 40)
            button.clicked.connect(signal.emit)
            row.addWidget(button)

    def update_state(self, s):
        if s.replay_mode:
            self.patient.setText(f"项目：{s.replay_label or '已保存回放'}")
            self.task.setText("本地路径：项目数据   ·   只读回放")
        else:
            self.patient.setText(f"项目：{s.patient.identifier}")
            self.task.setText(f"本地路径：prototype/demo_records   ·   {s.patient.task}")
        with QSignalBlocker(self.scenario):
            self.scenario.setCurrentIndex(self.scenario.findData("replay" if s.replay_mode else s.scenario))
        self.scenario.setEnabled(not s.replay_mode)
        self.source.setText("项目回放 · 只读" if s.replay_mode else "Mock 演示")
        self.device_status.setText("●  已保存数据可读" if s.replay_mode else "●  相机已连接")
        self.sensor_status.setText("RGB / 深度 / 网格 · 只读" if s.replay_mode else "RGB 正常  ·  深度正常  /  模拟")

from PySide6.QtCore import Qt, Signal, QRectF, QPointF
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QFrame, QPushButton, QVBoxLayout


class WorkflowStep(QPushButton):
    """Timeline step rendered like an industrial workstation navigator."""

    def __init__(self, index, title, subtitle, parent=None):
        super().__init__(parent)
        self.index = index
        self.title = title
        self.subtitle = subtitle
        self.active = False
        self.warning = False
        self.setObjectName("workflowStep")
        self.setFlat(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(82)

    def set_state(self, active, warning):
        self.active = bool(active)
        self.warning = bool(warning)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        rect = self.rect()
        if self.active:
            p.fillRect(rect, QColor("#142936"))
            p.fillRect(QRectF(0, 0, 3, rect.height()), QColor("#12bfe6"))
        center = QPointF(30, 30)
        if self.index < 3:
            p.setPen(QPen(QColor("#334656"), 1))
            p.drawLine(QPointF(center.x(), 51), QPointF(center.x(), rect.height()))
        ring = QColor("#12bfe6") if self.active else QColor("#596d7d")
        p.setBrush(QColor("#0d1721"))
        p.setPen(QPen(ring, 1.5))
        p.drawEllipse(center, 21, 21)
        p.setPen(ring if self.active else QColor("#b3c0ca"))
        p.setFont(QFont("Segoe UI", 11, QFont.DemiBold))
        p.drawText(QRectF(9, 18, 42, 25), Qt.AlignCenter, f"{self.index + 1:02d}")
        p.setPen(QColor("#e7edf4") if self.active else QColor("#c2cdd6"))
        p.setFont(QFont("Microsoft YaHei", 13, QFont.DemiBold if self.active else QFont.Normal))
        p.drawText(QPointF(64, 28), self.title)
        p.setPen(QColor("#8998a8"))
        p.setFont(QFont("Microsoft YaHei", 11))
        p.drawText(QPointF(64, 52), self.subtitle)
        if self.warning:
            p.setBrush(QColor("#e9b949"))
            p.setPen(Qt.NoPen)
            p.drawEllipse(QPointF(rect.width() - 18, 24), 4, 4)
        p.end()


class WorkflowSidebar(QFrame):
    step_requested = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("workflowSidebar")
        self.setFixedWidth(220)
        col = QVBoxLayout(self)
        col.setContentsMargins(16, 18, 14, 14)
        col.setSpacing(8)
        col.addWidget(text_label("操作流程", "sidebarHeading"))
        self.buttons = []
        self.subtitles = ("采集场景三维数据", "标记目标区域", "生成执行轨迹", "仿真与效果检查")
        for index, title in enumerate(STAGES):
            button = WorkflowStep(index, title, self.subtitles[index], self)
            button.setAttribute(Qt.WA_AlwaysShowToolTips)
            button.clicked.connect(lambda _=False, i=index: self.step_requested.emit(i))
            col.addWidget(button)
            self.buttons.append(button)
        col.addStretch()
        self.mode_badge = text_label("研发规划预览", "prototypeBadge")
        self.mode_note = text_label("模拟患者与设备\n不用于治疗执行", "muted", True)
        col.addWidget(self.mode_badge)
        col.addWidget(self.mode_note)

    def update_state(self, s):
        self.mode_badge.setText("项目回放 · 只读" if s.replay_mode else "研发规划预览")
        self.mode_note.setText("原始数据只读\n不可执行" if s.replay_mode else "模拟患者与设备\n不用于治疗执行")
        for i, btn in enumerate(self.buttons):
            reason = navigation_reason(s, i)
            active = s.stage == i
            warning = (i == 1 and s.region.support == "partial") or (i >= 2 and s.plan.phase == "failed")
            btn.set_state(active, warning)
            btn.setEnabled(active or not reason)
            btn.setToolTip(reason)
            tone(btn, "active" if active else "warning" if warning else "")

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QPushButton


class ActionBar(QFrame):
    action_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("actionBar")
        row = QHBoxLayout(self)
        row.setContentsMargins(16, 7, 16, 7)
        row.setSpacing(10)
        self.notice = text_label("", "actionNotice", True)
        self.notice.setMinimumWidth(230)
        row.addWidget(self.notice)
        self.telemetry = text_label("", "statusTelemetry")
        row.addWidget(self.telemetry, 1)
        self.secondary = []
        for _ in range(2):
            btn = QPushButton()
            btn.setObjectName("secondaryButton")
            btn.setMinimumHeight(46)
            btn.clicked.connect(lambda _=False, b=btn: self.action_requested.emit(b.property("action")))
            row.addWidget(btn)
            self.secondary.append(btn)
        self.primary = QPushButton()
        self.primary.setObjectName("primaryButton")
        self.primary.setMinimumSize(200, 48)
        self.primary.clicked.connect(lambda: self.action_requested.emit(self.primary.property("action")))
        row.addWidget(self.primary)

    def set_telemetry(self, text):
        self.telemetry.setText(text)

    def update_state(self, primary, secondary, notice, severity, telemetry=""):
        status = "▱  项目已加载" if severity == "normal" else ("！  " + notice)
        self.notice.setText(status)
        tone(self.notice, severity)
        if telemetry:
            self.telemetry.setText(telemetry)
        for btn, action in [(self.primary, primary), *zip(self.secondary, secondary)]:
            btn.setText(action.text)
            btn.setProperty("action", action.key)
            btn.setEnabled(action.enabled)
            btn.setToolTip(action.reason if not action.enabled else "")
        for index, btn in enumerate(self.secondary):
            btn.setVisible(index < len(secondary))

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFrame, QPushButton, QVBoxLayout


class ToolPalette(QFrame):
    tool_requested = Signal(str)
    undo_requested = Signal()
    clear_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("toolPalette")
        col = QVBoxLayout(self)
        col.setContentsMargins(5, 5, 5, 5)
        col.setSpacing(4)
        self.buttons = {}
        for key, text in (("work", "＋  规划区"), ("forbid", "⊘  禁入区"), ("erase", "－  擦除"), ("undo", "↶  撤销"), ("clear", "清空标记")):
            btn = QPushButton(text)
            btn.setObjectName("toolButton")
            btn.setMinimumSize(96, 44)
            if key == "undo":
                btn.clicked.connect(self.undo_requested.emit)
            elif key == "clear":
                btn.clicked.connect(self.clear_requested.emit)
            else:
                btn.clicked.connect(lambda _=False, k=key: self.tool_requested.emit(k))
            col.addWidget(btn)
            self.buttons[key] = btn
        self.setFixedSize(112, 254)

    def update_state(self, s, can_undo):
        self.setVisible(s.stage == 1 and not s.replay_mode)
        self.setEnabled(not s.busy and not s.replay_mode)
        for key, button in self.buttons.items():
            tone(button, "active" if key == s.tool else "")
        self.buttons["undo"].setEnabled(can_undo and not s.busy)
        self.buttons["clear"].setEnabled(bool(s.region.strokes) and not s.busy)

from PySide6.QtWidgets import QProgressBar


class CapturePage(StatusPanel):
    """Compact Inspector for acquisition, quality and spatial data."""

    def __init__(self):
        super().__init__()
        self.section("采集信息")
        self.phase = self.row("采集状态")
        self.resolution = self.row("相机分辨率")
        self.fps = self.row("帧率 FPS")
        self.capture_time = self.row("采集时间")
        self.frames = self.row("有效视角")
        self.divider()

        self.section("数据质量")
        self.coverage = self.row("覆盖率")
        self.coverage_bar = self.metric_bar()
        self.stability = self.row("有效深度")
        self.progress = self.metric_bar()
        self.views_bar = self.metric_bar()
        self.divider()

        self.section("空间数据")
        self.depth = self.row("深度数据")
        self.pose = self.row("相机位姿")
        self.reference = self.row("空间基准")
        self.surface = self.row("表面模型")
        self.problem = self.issue("查看缺失位置  →", "capture")
        self.hint = self.note("", "panelHint")
        self.result_note = self.note("", "panelHint")
        self.hint.setVisible(False)
        self.result_note.setVisible(False)
        self.finish()

    def metric_bar(self):
        bar = QProgressBar()
        bar.setObjectName("inspectorProgress")
        bar.setRange(0, 100)
        bar.setTextVisible(False)
        bar.setFixedHeight(8)
        self.column.addWidget(bar)
        return bar

    @staticmethod
    def _percent(value, default=0):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return default
        return value * 100 if 0 <= value <= 1 else value

    def update_state(self, s):
        c = s.capture
        names = {"idle": "待开始", "capturing": "采集中", "checking": "检查中", "passed": "已就绪", "failed": "需补采"}
        self.value(self.phase, names[c.phase], "bad" if c.phase == "failed" else "good" if c.phase == "passed" else "")
        self.value(self.resolution, "1920 × 1080")
        self.value(self.fps, "30")
        if s.replay_mode:
            created = str(s.replay_stats.get("created_local", ""))
            self.value(self.capture_time, created.replace("T", " ")[:16] or "—")
        else:
            self.value(self.capture_time, "演示采集")
        self.value(self.frames, f"{c.frames} / 3" if c.frames else "0 / 3", "good" if c.frames >= 3 else "")
        self.views_bar.setValue(min(100, int(c.frames / 3 * 100)))

        if s.replay_mode:
            coverage = self._percent(s.replay_stats.get("geometry_support_ratio"), 96)
            depth = self._percent(s.replay_stats.get("valid_depth_ratio_mean"), 75)
        else:
            coverage = c.support
            depth = c.stability if c.reference_ready else 0
        coverage = max(0, min(100, int(coverage)))
        depth = max(0, min(100, int(depth)))
        self.value(self.coverage, f"{coverage}%")
        self.coverage_bar.setValue(coverage)
        self.value(self.stability, f"{depth}%", "warning" if depth < 90 and s.replay_mode else "")
        self.progress.setValue(depth)

        checked = c.phase in ("passed", "failed")
        if s.replay_mode:
            self.value(self.depth, "已获取", "good")
            self.value(self.pose, "已建立", "good")
            self.value(self.reference, "已对齐", "good")
            self.value(self.surface, "仅供校对", "warning")
        else:
            status = "已获取" if checked else "待检查"
            status_tone = "good" if checked else ""
            self.value(self.depth, status, status_tone)
            self.value(self.pose, "已建立" if checked else "待建立", status_tone)
            self.value(self.reference, "已对齐" if c.reference_ready else "待建立", "good" if c.reference_ready else "")
            self.value(self.surface, "已生成" if checked else "待检查", status_tone)
        self.problem.setVisible(c.phase == "failed")

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QPushButton


class RegionPage(StatusPanel):
    example_requested = Signal()

    def __init__(self):
        super().__init__()
        self.section("当前标记")
        self.work = self.row("规划区")
        self.forbidden = self.row("禁入区")
        self.note("绿色：目标规划区域\n红色：探头或路线不得进入的区域", "panelHint")
        self.divider()
        self.section("三维支持检查")
        self.support = self.row("三维支持")
        self.issue_button = self.issue("定位缺失三维位置  →", "geometry")
        self.hint = self.note("", "panelHint")
        self.divider()
        self.section("标记操作")
        self.note("在画面中按住左键涂抹。擦除工具删除命中的一笔；撤销可恢复上一笔。")
        self.example = QPushButton("载入示例选区")
        self.example.setObjectName("quietButton")
        self.example.setMinimumHeight(44)
        self.example.clicked.connect(self.example_requested.emit)
        self.column.addWidget(self.example)
        self.example_note = self.note("示例用于演示操作；可继续添加或修改。")
        self.finish()

    def update_state(self, s):
        r = s.region
        self.value(self.work, "✓ 已标记" if r.has_work else "待标记", "good" if r.has_work else "")
        self.value(self.forbidden, "已标记" if r.has_forbidden else "尚未标记")
        names = {
            "unchecked": "待检查", "checking": "检查中…", "valid": "✓ 已确认",
            "partial": "⚠ 部分未确认", "missing": "阻断 · 局部缺失",
        }
        self.value(self.support, names.get(r.support, "待检查"), "bad" if r.support in ("missing", "partial") else "good" if r.support == "valid" else "")
        self.issue_button.setVisible(r.support == "missing")
        unconfirmed = int(s.replay_stats.get("geometry_unconfirmed_pixel_count", 0) or 0)
        ratio = s.replay_stats.get("geometry_support_ratio")
        ratio_text = f"几何像素支持约 {float(ratio) * 100:.1f}%；" if ratio is not None else ""
        self.hint.setText(
            "该区域缺少可靠三维数据，暂时无法生成轨迹。补采后会保留已画区域。"
            if r.support == "missing" else
            f"{ratio_text}部分像素缺少深度参考或存在冲突（约 {unconfirmed} 个）；当前仅可作为规划预览，不能视为治疗覆盖已确认。"
            if r.support == "partial" else
            "生成轨迹前会自动检查选区的三维支持。"
        )
        self.example.setVisible(not s.replay_mode)
        self.example_note.setVisible(not s.replay_mode)
        self.example.setEnabled(not s.busy and not s.replay_mode)

from PySide6.QtCore import Signal, QSignalBlocker
from PySide6.QtWidgets import QButtonGroup, QHBoxLayout, QLineEdit, QRadioButton, QToolButton, QWidget


class PlanningPage(StatusPanel):
    parameter_changed = Signal(str, str)

    def __init__(self):
        super().__init__()
        self.section("扫描行距")
        self.modes = QButtonGroup(self)
        self.direct = QRadioButton("直接指定行距")
        self.width = QRadioButton("按有效作用宽度计算")
        self.modes.addButton(self.direct)
        self.modes.addButton(self.width)
        for button, mode in ((self.direct, "direct"), (self.width, "width")):
            button.setMinimumHeight(30)
            button.toggled.connect(lambda checked, m=mode: self.parameter_changed.emit("mode", m) if checked else None)
            self.column.addWidget(button)
        self.inputs = {}
        for key, name in (("direct_spacing", "行距 / mm"), ("effective_width", "有效作用宽度 / mm"), ("overlap", "重叠率 / %")):
            item = QWidget()
            row = QHBoxLayout(item)
            row.setContentsMargins(0, 2, 0, 2)
            row.addWidget(text_label(name, "rowName"), 1)
            field = QLineEdit()
            field.setObjectName("parameterInput")
            field.setFixedWidth(85)
            field.setMinimumHeight(36)
            field.textEdited.connect(lambda value, k=key: self.parameter_changed.emit(k, value))
            row.addWidget(field)
            self.column.addWidget(item)
            self.inputs[key] = field
        self.spacing = self.note("", "spacingValue")
        self.error = self.note("", "inlineError")
        self.note("演示占位值，非设备参数。宽度 × (1 − 重叠率)；行距范围 0.5–100 mm，重叠率 0–90%。")
        self.divider()
        self.section("规划检查")
        self.region = self.row("规划区")
        self.forbidden = self.row("禁入区")
        self.geometry = self.row("三维几何")
        self.continuity = self.row("扫描线")
        self.route_issue = self.issue("定位路线缺口  →", "route")
        self.edge_issue = self.issue("查看未覆盖边缘  →", "edge")
        self.note("连接段不计入扫描；没有真实有效作用宽度时，不计算治疗覆盖率。")
        self.unused = QToolButton()
        self.unused.setObjectName("detailsButton")
        self.unused.setText("＋ 待用参数")
        self.unused.setCheckable(True)
        self.unused.setMinimumHeight(40)
        self.column.addWidget(self.unused)
        self.unused_text = self.note("治疗深度：暂未参与规划\n外壳尺寸：暂未参与规划\n外壳避让余量：暂未参与规划\n这些参数仅记录，未来用于焦点层、接触与避让检查；不用于当前扫描行距。")
        self.unused_text.hide()
        self.unused.toggled.connect(self.unused_text.setVisible)
        self.finish()

    def update_state(self, s):
        p = s.plan
        for btn, checked in ((self.direct, p.parameters.mode == "direct"), (self.width, p.parameters.mode == "width")):
            with QSignalBlocker(btn):
                btn.setChecked(checked)
            btn.setEnabled(not s.busy and not s.replay_mode)
        for key, field in self.inputs.items():
            value = getattr(p.parameters, key)
            if field.text() != value:
                with QSignalBlocker(field):
                    field.setText(value)
            field.setEnabled(not s.busy and not s.replay_mode and ((key == "direct_spacing") == (p.parameters.mode == "direct")))
        spacing, error = p.parameters.spacing()
        self.spacing.setText("最终行距  —" if error else f"最终行距  {spacing:.2f} mm")
        tone(self.spacing, "bad" if error else "good")
        self.error.setText(error)
        self.error.setVisible(bool(error))
        ready = p.phase in ("ready", "failed")
        stats = s.replay_stats
        self.value(self.region, "✓ 原始规划记录" if s.replay_mode and ready else "✓ 已规划" if ready else "等待生成", "good" if ready else "")
        self.value(self.forbidden, "原始禁入区记录" if s.replay_mode and ready else "✓ 未进入 · 模拟" if ready else "等待检查", "good" if ready else "")
        if s.replay_mode and ready:
            geometry = stats.get("geometry_status", "已记录")
            geometry_label = "部分未确认" if "unconfirmed" in geometry else "已确认"
            self.value(self.geometry, geometry_label, "bad" if "unconfirmed" in geometry else "good")
            gaps = int(stats.get("gap_count", 0))
            continuity = f"局部断开 · {gaps} 个缺口" if gaps else "✓ 连续"
            self.value(self.continuity, continuity, "bad" if gaps else "good")
        else:
            self.value(self.geometry, "✓ 有效 · 模拟", "good")
            self.value(self.continuity, "局部断开" if p.phase == "failed" else "✓ 段内连续 · 模拟" if ready else "等待生成", "bad" if p.phase == "failed" else "good" if ready else "")
        self.route_issue.setVisible(p.phase == "failed")
        self.edge_issue.setVisible(ready and (not s.replay_mode or stats.get("coverage_status") == "partial"))



class ConfirmationPage(StatusPanel):
    def __init__(self):
        super().__init__()
        self.section("运行预览")
        self.patient = self.row("患者")
        self.task = self.row("目标区域")
        self.model = self.row("采集模型")
        self.execution = self.row("可执行性")
        self.divider()
        self.section("规划摘要")
        self.spacing = self.row("最终行距")
        self.source = self.row("行距来源")
        self.width = self.row("有效作用宽度")
        self.overlap = self.row("重叠率")
        self.length = self.row("轨迹长度")
        self.version = self.row("规划版本")
        self.time = self.row("生成时间")
        self.divider()
        self.section("检查结果")
        self.checks = self.note("", "checkSummary")
        self.warning = self.issue("未覆盖区域需核对  →", "edge")
        self.note("当前仅为规划预览。治疗深度、工具姿态、接触、碰撞和治疗头覆盖均未参与。", "panelHint")
        self.saved = self.note("", "savedNotice")
        self.finish()

    def update_state(self, s):
        p = s.plan
        spacing, _ = p.parameters.spacing()
        self.value(self.patient, f"{s.patient.name} · {s.patient.identifier}")
        self.value(self.task, "规划区（目标区域）")
        self.value(self.model, f"本次采集 · {s.capture.model_revision:02d}")
        self.value(self.execution, "不可执行 · 规划预览", "bad" if s.replay_mode or not s.plan_current else "")
        self.value(self.spacing, f"{spacing:.2f} mm" if spacing is not None else "—", "good")
        self.value(self.source, "原始记录 · 直接指定" if s.replay_mode else "直接指定" if p.parameters.mode == "direct" else "按宽度计算")
        self.value(self.width, "未配置 · 不用于覆盖率" if s.replay_mode else f"{p.parameters.effective_width} mm · 占位")
        self.value(self.overlap, "未提供" if s.replay_mode else f"{p.parameters.overlap}% · 占位")
        if s.replay_mode:
            total = s.replay_stats.get("total_length_m")
            self.value(self.length, f"{float(total):.2f} m · 原始记录" if total is not None else "—")
        else:
            self.value(self.length, "1.8 m · 模拟值")
        self.value(self.version, f"预览 {p.revision:02d}")
        timestamp = p.generated_at.replace("T", " ")[5:] or "—"
        self.value(self.time, f"{timestamp} · 文件时间" if s.replay_mode and timestamp != "—" else timestamp)
        if s.replay_mode:
            stats = s.replay_stats
            gaps = int(stats.get("gap_count", 0))
            coverage = str(stats.get("coverage_status", "未提供"))
            self.warning.setText(f"存在 {gaps} 个路线缺口，查看轨迹端点  →" if gaps else "查看边缘未覆盖位置  →")
            self.warning.setVisible(bool(gaps or coverage == "partial"))
            self.checks.setText(
                "✓ 规划区与三维支持记录已载入\n"
                f"• 覆盖状态：{coverage}\n"
                f"• 扫描缺口：{gaps} 个；路线仅供只读核对"
            )
        else:
            self.warning.setText("边缘未覆盖区域需核对  →")
            self.warning.setVisible(True)
            self.checks.setText("✓ 规划区与三维支持已检查\n✓ 禁入区未进入（模拟）\n✓ 扫描段连续（模拟）" if s.plan_current else "当前规划需要重新检查。")
        self.saved.setText(
            "项目回放为只读记录，不保存、不覆盖原始数据。"
            if s.replay_mode else
            "✓ 本版本已保存，可在历史中查看。" if p.saved_id else
            "保存后保留当前模型来源、区域、参数与检查结果。"
        )

"""The UI consumes this contract. A VTK implementation can replace the factory."""
from copy import deepcopy
from dataclasses import dataclass, field
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget


@dataclass
class ViewportScene:
    stage: int = 0
    view: str = "RGB"
    tool: str = "work"
    strokes: list[Stroke] = field(default_factory=list)
    segments: list[RouteSegment] = field(default_factory=list)
    capture_phase: str = "idle"
    capture_progress: int = 0
    scenario: str = "normal"
    region_complete: bool = False
    geometry_repaired: bool = False
    trajectory_generated: bool = False
    trajectory_failed: bool = False
    issue_focus: str = ""
    editable: bool = False
    show_axes: bool = False


def scene_from_state(s):
    return ViewportScene(
        stage=s.stage, view=s.view, tool=s.tool,
        strokes=deepcopy(s.region.strokes), segments=deepcopy(s.plan.segments),
        capture_phase={"passed": "quality_good", "failed": "quality_bad"}.get(s.capture.phase, s.capture.phase),
        capture_progress=s.capture.frames, scenario=s.scenario,
        region_complete=s.region.support in ("valid", "missing"),
        geometry_repaired=s.region.geometry_repaired,
        trajectory_generated=s.plan.phase in ("ready", "failed"),
        trajectory_failed=s.plan.phase == "failed",
        issue_focus=s.issue_focus,
        editable=s.stage == 1 and not s.busy,
        show_axes=s.show_axes,
    )


class Medical3DViewport(QWidget):
    regions_edited = Signal(object)

    def set_scene(self, scene: ViewportScene) -> None:
        raise NotImplementedError

    def reset_view(self) -> None:
        raise NotImplementedError


def create_viewport(parent=None) -> Medical3DViewport:
    return MockViewport(parent)

"""QPainter demonstration viewport. Workflow decisions belong to the controller."""
from copy import deepcopy
from math import sin
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QRadialGradient
from PySide6.QtWidgets import QLabel


class MockViewport(Medical3DViewport):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.state = ViewportScene()
        self.setMinimumSize(520, 380)
        self.setMouseTracking(True)
        self._drawing = None
        self._zoom = 1.0
        self._pan = QPointF()
        self._pan_last = None
        self._scale, self._offset = 1.0, QPointF()
        # Enable a quiet breathing highlight only while an actionable issue
        # is present.  It draws attention without distracting from planning.
        self._pulse_phase = 0.0
        self._pulse_timer = QTimer(self)
        self._pulse_timer.setInterval(55)
        self._pulse_timer.timeout.connect(self._pulse_tick)
        self.tag = QLabel(self)
        self.tag.setObjectName("viewportTag")
        self.hint = QLabel(self)
        self.hint.setObjectName("viewportHint")
        self.hint.setWordWrap(True)
        self.hint.setFixedWidth(224)
        self.legend = QLabel(self)
        self.legend.setObjectName("viewportLegend")
        self.setToolTip("左键绘制规划区 · 右键平移 · 滚轮缩放")

    def set_scene(self, scene):
        self.state = deepcopy(scene)
        self._drawing = None
        self.tag.setText(f"{scene.view} · 实时画面")
        missing = scene.scenario == "geometry" and scene.region_complete and not scene.geometry_repaired
        problem = "该区域缺少可靠三维数据。已画选区保留，补采后可继续。" if missing else "左侧三维数据缺失，请移动相机继续补采。" if scene.capture_phase == "quality_bad" else "扫描线在高亮位置断开，请返回规划检查处理。" if scene.trajectory_failed else ""
        self.hint.setText(problem)
        self.hint.setVisible(bool(problem))
        if scene.stage == 0:
            legend = "绿色 · 可靠采集   灰色 · 未采集   黄色 · 待检查   红色 · 不可用"
        elif scene.stage == 1:
            legend = "绿色 · 规划区   红色 · 禁入区   紫色 · 缺失三维"
        else:
            legend = "黄色 · 扫描段   蓝色虚线 · 不可治疗连接   红色 · 禁入区"
        self.legend.setText(legend)
        self.legend.setVisible(False)
        has_issue = bool(problem) or self.state.issue_focus in ("geometry", "route", "edge")
        if has_issue:
            if not self._pulse_timer.isActive():
                self._pulse_timer.start()
        else:
            self._pulse_timer.stop()
        self._position_overlays()
        self.setCursor(Qt.CrossCursor if scene.editable else Qt.ArrowCursor)
        self.update()

    def reset_view(self):
        self._zoom = 1.0
        self._pan = QPointF()
        self.update()

    def _pulse_tick(self):
        self._pulse_phase = (self._pulse_phase + 0.22) % (2 * 3.141592653589793)
        self.update()

    def _pulse_strength(self) -> float:
        return 0.5 + 0.5 * sin(self._pulse_phase)

    def _position_overlays(self):
        self.tag.adjustSize()
        self.tag.move(18, 16)
        self.hint.adjustSize()
        self.hint.move(self.width()-self.hint.width()-18, 16)
        self.legend.adjustSize()
        self.legend.move(18, self.height()-self.legend.height()-14)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._position_overlays()

    def _view_transform(self):
        scale = min(self.width()/850, self.height()/700) * self._zoom
        offset = QPointF((self.width()-850*scale)/2, (self.height()-700*scale)/2) + self._pan
        self._scale, self._offset = scale, offset
        return scale, offset

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor("#080d12"))
        if self.state.view not in ("RGB", "固定 RGB", "RGB 路线图"):
            self._draw_space_grid(p)
        scale, offset = self._view_transform()
        p.translate(offset)
        p.scale(scale, scale)
        self._draw_surface(p)
        if self.state.stage == 0:
            self._draw_capture(p)
        else:
            p.save()
            p.setClipPath(self._surface_path())
            self._draw_strokes(p)
            if self.state.stage >= 2:
                self._draw_route(p)
            p.restore()
            if self.state.stage == 1:
                self._draw_geometry_issue(p)
            elif self.state.trajectory_failed or self.state.issue_focus in ("route", "edge"):
                x, y = self._work_center()
                if self.state.issue_focus == "edge" and not self.state.trajectory_failed:
                    self._issue_patch(p, x-60, y+70, QColor("#ffd166"), "边缘未覆盖")
                else:
                    self._issue_patch(p, x, y, QColor("#ef476f"), "断开")
        p.end()

    def _draw_capture(self, p):
        if self.state.capture_phase == "idle":
            return
        p.save()
        p.setClipPath(self._surface_path())
        p.setPen(Qt.NoPen)
        height = 60 + self.state.capture_progress*74
        p.fillRect(QRectF(180, 112, 480, min(500, height)), QColor(34, 201, 151, 68))
        if self.state.capture_phase == "capturing":
            p.setPen(QPen(QColor("#06d6a0"), 1.8))
            p.drawLine(QPointF(180, min(600, 112+height)), QPointF(650, min(600, 112+height)))
            p.setBrush(QColor(255, 209, 102, 70))
            p.setPen(Qt.NoPen)
            p.drawEllipse(QRectF(480, 380, 90, 130))
        p.restore()
        if self.state.capture_phase == "quality_bad":
            self._issue_patch(p, 315, 490, QColor("#ef476f"), "待补采")
        if self.state.issue_focus == "geometry":
            self._draw_strokes(p)
            self._draw_geometry_issue(p)

    def _draw_strokes(self, p):
        for stroke in self.state.strokes:
            if not stroke.points:
                continue
            color = QColor(6, 214, 160, 112) if stroke.kind == "work" else QColor(239, 71, 111, 165)
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(color, 66, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            if len(stroke.points) == 1:
                p.drawPoint(QPointF(*stroke.points[0]))
            else:
                curve = QPainterPath(QPointF(*stroke.points[0]))
                for point in stroke.points[1:]:
                    curve.lineTo(QPointF(*point))
                p.drawPath(curve)

    def _draw_route(self, p):
        p.setBrush(Qt.NoBrush)
        for segment in self.state.segments:
            p.setPen(QPen(QColor("#fcca46" if segment.kind == "scan" else "#48cae4"), 2.6 if segment.kind == "scan" else 1.8, Qt.SolidLine if segment.kind == "scan" else Qt.DashLine, Qt.RoundCap))
            path = QPainterPath(QPointF(*segment.points[0]))
            for point in segment.points[1:]:
                path.lineTo(QPointF(*point))
            p.drawPath(path)

    def _scene_point(self, position):
        scale, offset = self._view_transform()
        return ((position.x()-offset.x())/scale, (position.y()-offset.y())/scale)

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            self._pan_last = event.position()
            self.setCursor(Qt.ClosedHandCursor)
            return
        if not self.state.editable or event.button() != Qt.LeftButton:
            return super().mousePressEvent(event)
        x, y = self._scene_point(event.position())
        if not self._surface_path().contains(QPointF(x, y)):
            return
        if self.state.tool == "erase":
            candidates = [(distance_to_stroke(x, y, stroke), i) for i, stroke in enumerate(self.state.strokes)]
            if candidates and min(candidates)[0] <= 40:
                self.state.strokes.pop(min(candidates)[1])
                self.regions_edited.emit(deepcopy(self.state.strokes))
            return
        self._drawing = Stroke(self.state.tool, [(x, y)])
        self.state.strokes.append(self._drawing)
        self.update()

    def mouseMoveEvent(self, event):
        if self._pan_last is not None:
            self._pan += event.position() - self._pan_last
            self._pan_last = event.position()
            self.update()
            return
        if self._drawing is not None:
            x, y = self._scene_point(event.position())
            if self._surface_path().contains(QPointF(x, y)):
                self._drawing.points.append((x, y))
                self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton:
            self._pan_last = None
            self.setCursor(Qt.CrossCursor if self.state.editable else Qt.ArrowCursor)
            return
        if self._drawing is not None:
            self._drawing = None
            self.regions_edited.emit(deepcopy(self.state.strokes))
            self.update()
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        self._zoom = min(2.2, max(0.65, self._zoom * (1.1 if event.angleDelta().y() > 0 else 1/1.1)))
        self.update()
        event.accept()

    def _draw_space_grid(self, p: QPainter) -> None:
        w, h = self.width(), self.height()
        horizon = h * 0.55
        center = w * 0.5
        p.setPen(QPen(QColor(0, 180, 216, 14), 1))
        for x in range(0, w + 1, 42):
            p.drawLine(QPointF(x, 0), QPointF(x, horizon))
        for y in range(0, int(horizon), 42):
            p.drawLine(QPointF(0, y), QPointF(w, y))
        p.setPen(QPen(QColor(72, 202, 228, 22), 1))
        for x in range(-w, 2 * w + 1, max(48, w // 13)):
            p.drawLine(QPointF(center, horizon), QPointF(x, h))
        for fraction in (0.08, 0.16, 0.27, 0.40, 0.57, 0.77, 1.0):
            y = horizon + (h - horizon) * fraction
            p.drawLine(QPointF(0, y), QPointF(w, y))
        if not self.state.show_axes:
            return
        origin = QPointF(83, h - 78)
        axes = ((74, 0, "X", "#ef476f"), (0, -64, "Y", "#06d6a0"), (-42, 27, "Z", "#48cae4"))
        p.setFont(QFont("Segoe UI", 9, QFont.DemiBold))
        for dx, dy, name, color in axes:
            pen = QPen(QColor(color), 2)
            p.setPen(pen)
            target = origin + QPointF(dx, dy)
            p.drawLine(origin, target)
            p.drawEllipse(target, 2.5, 2.5)
            p.drawText(target + QPointF(5, -4), name)
        p.setPen(QColor("#9aabb9"))
        p.drawText(origin + QPointF(-13, 22), "参考")

    def _surface_path(self) -> QPainterPath:
        path = QPainterPath(QPointF(367, 58))
        path.cubicTo(391, 66, 440, 66, 463, 58)
        path.cubicTo(466, 82, 471, 106, 490, 115)
        path.cubicTo(548, 121, 606, 144, 638, 182)
        path.cubicTo(656, 204, 655, 234, 640, 252)
        path.cubicTo(626, 271, 608, 275, 590, 278)
        path.cubicTo(578, 354, 584, 456, 570, 534)
        path.cubicTo(567, 560, 562, 581, 548, 596)
        path.cubicTo(494, 620, 338, 620, 282, 596)
        path.cubicTo(268, 581, 262, 560, 259, 534)
        path.cubicTo(246, 453, 253, 351, 239, 278)
        path.cubicTo(208, 275, 186, 259, 176, 237)
        path.cubicTo(166, 209, 184, 177, 213, 157)
        path.cubicTo(246, 135, 292, 120, 340, 115)
        path.cubicTo(359, 106, 363, 82, 367, 58)
        path.closeSubpath()
        return path

    def _draw_surface(self, p: QPainter) -> None:
        body = self._surface_path()
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(6, 214, 160, 12))
        
        p.setBrush(QColor(0, 0, 0, 95))
        p.drawEllipse(QRectF(220, 617, 390, 26))
        if self.state.view in ("RGB", "固定 RGB", "RGB 路线图"):
            surface_color = "#26323b"
        elif self.state.view == "深度":
            surface_color = "#173445"
        else:
            surface_color = "#172530"
        for width, alpha in ((3, 18),):
            p.setPen(QPen(QColor(0, 180, 216, alpha), width))
            p.setBrush(Qt.NoBrush)
            p.drawPath(body)
        p.setBrush(QColor(surface_color))
        p.setPen(QPen(QColor("#8ba6ae"), 1.4))
        p.drawPath(body)
        p.save()
        p.setClipPath(body)
        sheen = QRadialGradient(QPointF(418, 310), 290)
        sheen.setColorAt(0, QColor(57, 166, 183, 34))
        sheen.setColorAt(0.8, QColor(0, 0, 0, 0))
        p.fillRect(QRectF(165, 55, 500, 565), sheen)
        p.setBrush(Qt.NoBrush)
        rgb = self.state.view in ("RGB", "固定 RGB", "RGB 路线图")
        for y in range(150, 610, 29):
            narrowing = abs(y - 355) * 0.07
            curve = QPainterPath(QPointF(207 + narrowing, y))
            curve.cubicTo(335, y + 18, 498, y + 18, 624 - narrowing, y)
            p.setPen(QPen(QColor(105, 198, 209, 8 if rgb else 48), 1))
            p.drawPath(curve)
        for x in range(250, 620, 34):
            curve = QPainterPath(QPointF(x, 112))
            curve.cubicTo(x - 31, 270, x + 29, 420, x, 605)
            p.setPen(QPen(QColor(91, 170, 185, 5 if rgb else 30), 1))
            p.drawPath(curve)
        midline = QPainterPath(QPointF(415, 150))
        midline.cubicTo(406, 260, 420, 420, 415, 570)
        p.setPen(QPen(QColor(123, 221, 225, 41), 1))
        p.drawPath(midline)
        p.setPen(QPen(QColor(131, 218, 223, 60), 1))
        p.setBrush(QColor("#112a35"))
        p.drawEllipse(QPointF(415, 407), 3.4, 3.4)
        p.restore()
        if self.state.view == "点云":
            p.save()
            p.setClipPath(body)
            p.setPen(QPen(QColor("#b5f0ed"), 2))
            for y in range(95, 608, 12):
                for x in range(180 + (y % 3) * 4, 653, 12):
                    p.drawPoint(x, y)
            p.restore()
        p.setPen(QColor("#85a4b1"))
        p.setFont(QFont("Microsoft YaHei", 9))
        p.drawText(QPointF(270, 675), "模拟患者表面 · 仅用于交互演示")

    def _draw_geometry_issue(self, p: QPainter) -> None:
        if self.state.scenario == "geometry" and self.state.region_complete and not self.state.geometry_repaired:
            x, y = self._work_center()
            self._issue_patch(p, x, y, QColor("#9d4edd"), "缺失几何")

    def _work_center(self) -> tuple[float, float]:
        pts = [point for stroke in self.state.strokes if stroke.kind == "work" for point in stroke.points]
        if not pts:
            return 390, 430
        return (sum(point[0] for point in pts) / len(pts), sum(point[1] for point in pts) / len(pts))

    def _issue_patch(self, p: QPainter, x: float, y: float, color: QColor, label: str) -> None:
        pulse = self._pulse_strength()
        fill_alpha = 78 + int(60 * pulse)
        outer_alpha = 120 + int(90 * pulse)
        radius_x, radius_y = 48 + 7 * pulse, 37 + 5 * pulse
        p.setBrush(QColor(color.red(), color.green(), color.blue(), fill_alpha))
        p.setPen(QPen(QColor(color.red(), color.green(), color.blue(), outer_alpha), 3.5))
        p.drawEllipse(QPointF(x, y), radius_x, radius_y)
        # A second, faint ring reads as a breathing alert at a glance.
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor(color.red(), color.green(), color.blue(), 42 + int(55 * pulse)), 1.4))
        p.drawEllipse(QPointF(x, y), radius_x + 9 + 4 * pulse, radius_y + 7 + 3 * pulse)
        p.setBrush(QColor("#161d27"))
        p.setPen(QPen(QColor(color.red(), color.green(), color.blue(), 220), 2))
        p.drawRoundedRect(QRectF(x + 35, y - 50, 112, 30), 4, 4)
        p.setPen(QColor("#ffffff"))
        p.drawText(QRectF(x + 35, y - 50, 112, 30), Qt.AlignCenter, label)

"""Software display of saved project vertices. No reconstruction or replanning."""
from math import sin, cos, radians, ceil
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import QWidget


class ReplayViewport(QWidget):
    regions_edited = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(520, 380)
        self.data = None
        self.stage = 0
        self.mode = "RGB"
        self.zoom = 1.0
        self.pan = QPointF()
        self.yaw = self.pitch = 0.0
        self.drag = None
        self.focus = ""
        self.images = {}
        self.cache = None
        self.cache_key = None
        self.route_groups = []
        self._pulse_phase = 0.0
        self._pulse_timer = QTimer(self)
        self._pulse_timer.setInterval(55)
        self._pulse_timer.timeout.connect(self._pulse_tick)
        self._redraw_timer = QTimer(self)
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.setInterval(25)
        self._redraw_timer.timeout.connect(self.update)
        self.setToolTip("左键旋转三维视图 · 右键平移 · 滚轮缩放")

    def set_data(self, data, images, route_groups):
        self.data, self.images, self.route_groups = data, images, route_groups
        self._pulse_timer.stop()
        self._pulse_phase = 0.0
        raw_faces = {int(index) for index in data.region["region"].get("triangle_indices", [])}
        self.selected_faces = raw_faces
        # Selection indices are usually indices in the full reconstruction mesh;
        # some exports use the compact region mesh instead.  Detect the format
        # from the index range so both saved variants replay correctly.
        self.region_mesh_selection = bool(self.selected_faces) and max(self.selected_faces) < len(data.region_faces)
        source_faces = data.region_faces if self.region_mesh_selection else data.faces
        if any(index < 0 or index >= len(source_faces) for index in self.selected_faces):
            raise ValueError("区域三角形索引超出回放网格范围")
        self.selected_vertices = {i for face in self.selected_faces for i in source_faces[face]}
        self.reset_view()

    def set_view(self, stage, mode):
        if (self.stage, self.mode) != (stage, mode):
            self.stage, self.mode = stage, mode
            self.focus = ""
            self._pulse_timer.stop()
            self.reset_view()
        else:
            self.update()

    def reset_view(self):
        self.zoom, self.pan = 1.0, QPointF()
        self.yaw = self.pitch = 0.0
        self.cache_key = None
        self.update()

    def focus_issue(self, kind):
        self.focus = "gap" if kind == "route" else kind
        if self.focus in ("gap", "geometry", "capture", "edge"):
            self._pulse_timer.start()
        else:
            self._pulse_timer.stop()
        self.update()

    def _pulse_tick(self):
        self._pulse_phase = (self._pulse_phase + 0.22) % (2 * 3.141592653589793)
        self.update()

    def _pulse_strength(self) -> float:
        return 0.5 + 0.5 * sin(self._pulse_phase)

    def is_3d(self):
        return self.mode in ("模型", "三维模型", "点云")

    def project(self, point):
        x, y, z = (point[i]-self.center[i] for i in range(3))
        a, b = radians(self.yaw), radians(self.pitch)
        x, z = x*cos(a)+z*sin(a), -x*sin(a)+z*cos(a)
        y, z = y*cos(b)-z*sin(b), y*sin(b)+z*cos(b)
        return (self.width()/2+self.pan.x()+x*self.scale, self.height()/2+self.pan.y()+y*self.scale, z)

    def prepare_projection(self):
        use_region = self.stage >= 1 and self.region_mesh_selection
        vertices = self.data.region_vertices if use_region else self.data.vertices
        fitting = [vertices[i] for i in self.selected_vertices] if self.stage >= 1 else vertices
        if not fitting:
            fitting = vertices
        bounds = [(min(p[i] for p in fitting), max(p[i] for p in fitting)) for i in range(3)]
        self.center = [(a+b)/2 for a, b in bounds]
        self.scale = min((self.width()-90)/max(.01, bounds[0][1]-bounds[0][0]), (self.height()-100)/max(.01, bounds[1][1]-bounds[1][0]))*.86*self.zoom

    def render_geometry(self):
        key = (self.width(), self.height(), self.stage, self.mode, self.yaw, self.pitch, self.zoom, self.pan.x(), self.pan.y())
        if key == self.cache_key:
            return self.cache
        self.prepare_projection()
        image = QImage(self.size(), QImage.Format_RGB32)
        image.fill(QColor("#0b0e11"))
        p = QPainter(image)
        p.setClipRect(image.rect())
        if self.mode == "点云":
            cloud = list(zip(self.data.cloud_points, self.data.cloud_colors))
            if len(cloud) > 12000:
                step = ceil(len(cloud) / 12000)
                cloud = cloud[::step]
            items = [(self.project(point), color) for point, color in cloud]
            for (x, y, _), color in sorted(items, key=lambda item: -item[0][2]):
                p.setPen(QPen(QColor(*color), 2))
                p.drawPoint(QPointF(x, y))
        else:
            use_region = self.stage >= 1 and self.region_mesh_selection
            vertices = self.data.region_vertices if use_region else self.data.vertices
            colors = self.data.region_colors if use_region else self.data.colors
            faces = self.data.region_faces if use_region else self.data.faces
            projected = [self.project(point) for point in vertices]
            indices = list(self.selected_faces if self.stage >= 1 else range(len(faces)))
            # A software prototype should stay responsive on ordinary clinical
            # workstations.  The saved mesh remains untouched; this only limits
            # the number of painter polygons used for the interactive preview.
            if len(indices) > 3200:
                step = ceil(len(indices) / 3200)
                indices = indices[::step]
            indices = sorted(indices, key=lambda index: -sum(projected[i][2] for i in faces[index]))
            p.setPen(Qt.NoPen)
            for index in indices:
                face = faces[index]
                coordinates = [projected[i] for i in face]
                if all(x < -50 or x > self.width()+50 or y < -50 or y > self.height()+50 for x, y, _ in coordinates):
                    continue
                rgb = tuple(sum(colors[i][channel] for i in face)//3 for channel in range(3))
                if self.stage >= 1:
                    rgb = (int(rgb[0]*.65), int(rgb[1]*.65+55), int(rgb[2]*.65+22))
                p.setBrush(QColor(*rgb))
                p.drawPolygon(QPolygonF([QPointF(x, y) for x, y, _ in coordinates]))
            if self.stage >= 2:
                p.setRenderHint(QPainter.Antialiasing)
                p.setBrush(Qt.NoBrush)
                for role, points in self.route_groups:
                    if len(points) < 2:
                        continue
                    p.setPen(QPen(QColor("#fcca46" if role == "scan" else "#48cae4"), 1.45, Qt.SolidLine if role == "scan" else Qt.DashLine))
                    poly = QPainterPath()
                    for index, point in enumerate(points):
                        x, y, _ = self.project(point)
                        if index == 0:
                            poly.moveTo(x, y)
                        else:
                            poly.lineTo(x, y)
                    p.drawPath(poly)
        p.end()
        self.cache, self.cache_key = image, key
        return image

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#0b0e11"))
        if not self.data:
            p.setPen(QColor("#b8cdd8"))
            p.drawText(self.rect(), Qt.AlignCenter, "正在读取已保存的项目数据…")
            p.end()
            return
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        if self.is_3d():
            p.drawImage(0, 0, self.render_geometry())
            self.draw_gaps(p)
            hint = "左键旋转 · 右键平移 · 滚轮缩放"
        else:
            key = "route" if self.mode == "RGB 路线图" else "selected" if self.mode in ("选区 RGB", "固定 RGB") else "depth" if self.mode == "深度" else "rgb"
            img = self.images[key]
            ratio = min(self.width()/img.width(), (self.height()-62)/img.height())*self.zoom
            rect = QRectF((self.width()-img.width()*ratio)/2+self.pan.x(), (self.height()-img.height()*ratio)/2+self.pan.y(), img.width()*ratio, img.height()*ratio)
            p.drawImage(rect, img)
            hint = "右键平移 · 滚轮缩放"
            if key == "depth":
                hint += "  |  深度色标 250–2210 mm；黑色为无效值"
        self.draw_focus_marker(p)
        if self.focus == "geometry":
            pulse = self._pulse_strength()
            left = self.width() * 0.24
            top = self.height() * 0.18
            width = self.width() * 0.52
            height = self.height() * 0.58
            p.setBrush(QColor(157, 78, 221, 18 + int(20 * pulse)))
            p.setPen(QPen(QColor(157, 78, 221, 125 + int(80 * pulse)), 2.0 + 1.2 * pulse))
            p.drawRoundedRect(QRectF(left, top, width, height), 10, 10)
            p.setBrush(QColor(22, 29, 39, 230))
            p.setPen(QPen(QColor(157, 78, 221, 200), 1))
            p.drawRoundedRect(QRectF(left + 14, top + 14, 188, 30), 5, 5)
            p.setPen(QColor("#eadcff"))
            p.drawText(QRectF(left + 24, top + 14, 168, 30), Qt.AlignVCenter, "三维表面支持未确认")
        p.setPen(QColor("#e7edf4"))
        p.fillRect(QRectF(12, 12, 188, 28), QColor(11, 17, 24, 220))
        p.drawText(QRectF(22, 12, 178, 28), Qt.AlignVCenter, f"{self.mode} · 项目回放")
        p.end()

    def draw_gaps(self, p):
        if self.stage < 2:
            return
        p.setRenderHint(QPainter.Antialiasing)
        pulse = self._pulse_strength() if self.focus == "gap" else 0.0
        p.setBrush(QColor("#161d27"))
        gaps = [item for item in self.data.trajectory["ordered_items"] if item["kind"] == "gap"]
        for item in gaps:
            for key in ("previous_point", "next_point"):
                x, y, _ = self.project(item[key]["xyz_m"])
                radius = 3 + 2 * pulse
                p.setPen(QPen(QColor(239, 71, 111, 185 + int(60 * pulse)), 1.6 + 1.0 * pulse))
                p.drawEllipse(QPointF(x, y), radius, radius)
                if self.focus == "gap":
                    p.setBrush(Qt.NoBrush)
                    p.setPen(QPen(QColor(239, 71, 111, 50 + int(85 * pulse)), 1.1))
                    p.drawEllipse(QPointF(x, y), radius + 5 + 3 * pulse, radius + 5 + 3 * pulse)
                    p.setBrush(QColor("#161d27"))
        if self.focus == "gap":
            p.setPen(QColor("#ffd166"))
            p.drawText(22, 68, "粉色圆点：断开段端点（缺口之间不连线）")

    def draw_focus_marker(self, p):
        """Pulse a focused geometry/edge area on both mesh and image views."""
        if self.focus not in ("geometry", "edge"):
            return
        pulse = self._pulse_strength()
        color = QColor("#9d4edd" if self.focus == "geometry" else "#ffd166")
        if self.is_3d() and getattr(self, "selected_vertices", None):
            vertices = self.data.region_vertices if self.region_mesh_selection else self.data.vertices
            points = [self.project(vertices[index]) for index in self.selected_vertices if index < len(vertices)]
            if points:
                cx = sum(point[0] for point in points) / len(points)
                cy = sum(point[1] for point in points) / len(points)
            else:
                cx, cy = self.width() / 2, self.height() / 2
        else:
            cx, cy = self.width() / 2, self.height() / 2
        radius = 34 + 8 * pulse
        p.setRenderHint(QPainter.Antialiasing)
        p.setBrush(QColor(color.red(), color.green(), color.blue(), 52 + int(36 * pulse)))
        p.setPen(QPen(QColor(color.red(), color.green(), color.blue(), 145 + int(80 * pulse)), 2.2))
        p.drawEllipse(QPointF(cx, cy), radius, radius * .72)
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor(color.red(), color.green(), color.blue(), 48 + int(70 * pulse)), 1.2))
        p.drawEllipse(QPointF(cx, cy), radius + 10 + 4 * pulse, radius * .72 + 7 + 3 * pulse)

    def mousePressEvent(self, event):
        if event.button() in (Qt.LeftButton, Qt.RightButton):
            self.drag = (event.button(), event.position())

    def mouseMoveEvent(self, event):
        if self.drag:
            button, previous = self.drag
            delta = event.position()-previous
            self.drag = (button, event.position())
            if button == Qt.LeftButton and self.is_3d():
                self.yaw += delta.x()*.35
                self.pitch = min(85, max(-85, self.pitch+delta.y()*.35))
            elif button == Qt.RightButton:
                self.pan += delta
            self._redraw_timer.start()

    def mouseReleaseEvent(self, event):
        self.drag = None
        self._redraw_timer.stop()
        self.update()

    def wheelEvent(self, event):
        self.zoom = max(.35, min(6, self.zoom*(1.15 if event.angleDelta().y() > 0 else 1/1.15)))
        self.update()
        event.accept()

def current_notice(s):
    if s.replay_mode:
        if s.stage == 2 and s.plan.phase == "failed":
            return "项目回放 · 只读核对：原始扫描路线存在局部断开，请查看粉色端点。", "warning"
        if s.stage == 3 and s.plan.phase == "failed":
            return "项目回放 · 只读结果：当前规划存在断开段，不能作为执行路线。", "error"
        if s.stage == 1:
            if s.region.support == "partial":
                return "项目回放 · 只读警告：部分三维支持未确认，请查看选区 RGB 与支持记录。", "warning"
            return "项目回放 · 只读核对：已显示原始区域与三维支持记录。", "normal"
        return "项目回放 · 只读核对：已载入保存的采集数据。", "normal"
    if s.message:
        return s.message, s.message_tone
    if s.stage == 0:
        if s.capture.return_to_region and s.capture.phase == "capturing":
            return "已保留区域标记。请补采高亮位置，完成后返回区域标记。", "warning"
        return {
            "idle": ("请调整相机位置，使目标区域完整进入画面。", "normal"),
            "capturing": ("正在采集。保持目标稳定，缓慢移动相机补采不同视角。", "normal"),
            "checking": ("正在检查深度、相机位姿和表面连续性…", "normal"),
            "passed": ("采集检查通过。请确认目标表面完整，再进入区域标记。", "normal"),
            "failed": ("左侧存在三维数据缺失，请继续补采高亮位置。", "error"),
        }[s.capture.phase]
    if s.stage == 1:
        return {
            "unchecked": ("标出规划区与禁入区，完成后确认标记并检查三维几何。", "normal"),
            "checking": ("正在检查选区对应的三维几何，已画标记会保留。", "normal"),
            "missing": ("紫色区域缺少可靠三维数据，请返回补采。", "error"),
            "valid": ("三维支持已确认。可继续修改标记或生成规划预览。", "normal"),
            "partial": ("部分三维支持未确认。可查看规划预览，但不能视为治疗覆盖已确认。", "warning"),
        }[s.region.support]
    if s.stage == 2:
        _, error = s.plan.parameters.spacing()
        if error:
            return error, "error"
        if s.plan.phase == "generating":
            return "正在计算轨迹路径…", "normal"
        if s.plan.phase == "failed":
            return "扫描线局部断开，请检查高亮位置并重新规划。", "error"
        if s.plan_current:
            return "请核对规划区、禁入区和边缘未覆盖位置，再进入最终确认。", "warning"
        return "参数已更新。请重新生成轨迹，核对变化后的扫描行距。", "normal"
    if s.plan.saved_id:
        return "规划已保存，可在历史中查看。当前记录是研发规划预览。", "normal"
    return "请核对任务、区域、行距及检查结果，确认后保存规划方案（预览）。", "normal"

"""Application composition. Pages emit intents; the controller owns transitions."""
from pathlib import Path
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QImage, QKeySequence, QShortcut
from PySide6.QtWidgets import QCheckBox, QDockWidget, QFrame, QHBoxLayout, QListWidget, QMainWindow, QMessageBox, QPushButton, QScrollArea, QStackedWidget, QTextBrowser, QVBoxLayout, QWidget


class MainWindow(QMainWindow):
    def __init__(self, records_dir=None, viewport_factory=create_viewport):
        super().__init__()
        self.setWindowTitle("Femto Bolt · 三维采集与轨迹规划工作台")
        self.resize(1600, 900)
        self.setMinimumSize(1280, 760)
        source_dir = Path(__file__).resolve().parent
        directory = records_dir or (source_dir.parent if source_dir.name == "app" else source_dir) / "demo_records"
        self.controller = WorkflowController(Path(directory), self)
        self.project_replay = None
        # In the modular app source_dir is <project>/prototype/app; in the
        # generated standalone it is <project>/prototype.
        project_root = source_dir.parents[1] if source_dir.name == "app" else source_dir.parent
        try:
            self.project_replay = load_project_data(project_root)
        except Exception as exc:
            self.replay_error = str(exc)
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        column = QVBoxLayout(root)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        self.header = PatientHeader()
        self.header.history_requested.connect(self.show_history)
        self.header.settings_requested.connect(self.show_settings)
        self.header.scenario_requested.connect(self.change_scenario)
        column.addWidget(self.header)
        self.alert = text_label("", "globalStatus")
        # Status is intentionally kept in the bottom action dock.  A second
        # full-width alert band made the clinical viewport feel compressed.
        self.alert.setVisible(False)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        self.sidebar = WorkflowSidebar()
        self.sidebar.step_requested.connect(self.controller.go)
        body.addWidget(self.sidebar)
        center = QWidget()
        self.center = center
        center.setObjectName("center")
        work = QVBoxLayout(center)
        self.work_layout = work
        work.setContentsMargins(0, 0, 0, 0)
        work.setSpacing(0)
        title_bar = QFrame()
        title_bar.setObjectName("titleBar")
        title_bar.setFixedHeight(94)
        title_row = QHBoxLayout(title_bar)
        title_row.setContentsMargins(16, 10, 16, 9)
        title_col = QVBoxLayout()
        title_col.setSpacing(3)
        self.page_title = text_label("", "pageTitle")
        self.page_subtitle = text_label("", "pageSubtitle")
        title_col.addWidget(self.page_title)
        title_col.addWidget(self.page_subtitle)
        title_row.addLayout(title_col, 1)
        self.reference = text_label("", "referenceStatus")
        self.reference.setVisible(False)
        title_row.addWidget(self.reference)
        work.addWidget(title_bar)

        view_bar = QFrame()
        view_bar.setObjectName("viewBar")
        view_bar.setFixedHeight(54)
        view_row = QHBoxLayout(view_bar)
        view_row.setContentsMargins(16, 4, 16, 4)
        view_row.setSpacing(4)
        self.view_labels = {"RGB": "▣  RGB", "深度": "≋  深度", "点云": "⠿  点云", "模型": "◇  模型"}
        self.view_buttons = []
        for _ in range(4):
            btn = QPushButton()
            btn.setObjectName("viewButton")
            btn.setMinimumSize(92, 44)
            btn.clicked.connect(lambda _=False, b=btn: self.controller.set_view(b.property("view_name")))
            view_row.addWidget(btn)
            self.view_buttons.append(btn)
        view_row.addStretch()
        self.fit_button = QPushButton("适配")
        self.zoom_button = QPushButton("⌕")
        self.one_button = QPushButton("1:1")
        self.full_button = QPushButton("□")
        for button, tip in (
            (self.fit_button, "适配窗口"),
            (self.zoom_button, "缩放"),
            (self.one_button, "原始比例"),
            (self.full_button, "全屏显示"),
        ):
            button.setObjectName("viewTool")
            button.setMinimumSize(44, 40)
            button.setToolTip(tip)
            view_row.addWidget(button)
        self.reset_button = QPushButton("复位视图")
        self.reset_button.setObjectName("quietButton")
        self.reset_button.setMinimumHeight(40)
        self.reset_button.setToolTip("恢复默认视图")
        view_row.addWidget(self.reset_button)
        work.addWidget(view_bar)

        if self.project_replay is not None:
            self.viewport = ReplayViewport(center)
            try:
                rgb, selected = region_image(self.project_replay)
                depth = depth_heatmap(self.project_replay.depth_path)
                route_image = QImage(str(self.project_replay.trajectory_dir / "route_diagram.png"))
                if route_image.isNull():
                    route_image = selected
                self.viewport.set_data(self.project_replay, {
                    "rgb": QImage(str(self.project_replay.rgb_path)),
                    "selected": selected,
                    "depth": depth,
                    "route": route_image,
                }, self.project_replay.route_groups)
                self.controller.load_replay(self.project_replay)
            except Exception as exc:
                self.project_replay = None
                self.replay_error = str(exc)
                self.viewport = viewport_factory(center)
        else:
            self.viewport = viewport_factory(center)
        self.viewport.regions_edited.connect(self.controller.set_regions)
        self.reset_button.clicked.connect(self.viewport.reset_view)
        self.fit_button.clicked.connect(self.viewport.reset_view)
        self.one_button.clicked.connect(self.viewport.reset_view)
        self.full_button.clicked.connect(self.toggle_maximized)
        work.addWidget(self.viewport, 1)
        self.palette = ToolPalette(self.viewport)
        self.palette.move(12, 12)
        self.palette.tool_requested.connect(self.controller.set_tool)
        self.palette.undo_requested.connect(self.controller.undo)
        self.palette.clear_requested.connect(self.clear_regions)
        self.viewport.installEventFilter(self)
        body.addWidget(center, 1)

        self.pages = [CapturePage(), RegionPage(), PlanningPage(), ConfirmationPage()]
        self.page_stack = QStackedWidget()
        for page in self.pages:
            scroll = QScrollArea()
            scroll.setObjectName("rightPanel")
            scroll.setWidgetResizable(True)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            scroll.setWidget(page)
            self.page_stack.addWidget(scroll)
            page.focus_requested.connect(self.controller.focus)
        self.page_stack.setMinimumWidth(280)
        self.page_stack.setMaximumWidth(320)
        self.page_stack.setFixedWidth(300)
        body.addWidget(self.page_stack)
        self.pages[1].example_requested.connect(self.load_example)
        self.pages[2].parameter_changed.connect(self.controller.set_parameter)
        column.addLayout(body, 1)
        self.action_bar = ActionBar()
        self.action_bar.action_requested.connect(self.perform_action)
        column.addWidget(self.action_bar)
        self.build_drawers()
        shortcut = QShortcut(QKeySequence.Undo, self)
        shortcut.activated.connect(self.controller.undo)
        self.controller.changed.connect(self.refresh)
        self.controller.history_changed.connect(self.refresh_history)
        self.refresh()

    def eventFilter(self, watched, event):
        if hasattr(self, "viewport") and watched is self.viewport and event.type() == QEvent.Resize:
            self.palette.move(12, 12)
        return super().eventFilter(watched, event)

    def refresh(self):
        s = self.controller.state
        if not s.replay_mode and isinstance(self.viewport, ReplayViewport):
            self._switch_to_mock_viewport()
        self.header.update_state(s)
        self.sidebar.update_state(s)
        self.page_title.setText(("三维采集", "区域标记", "轨迹规划", "运行预览")[s.stage])
        self.page_subtitle.setText((
            "调整相机位置，采集完整的目标表面",
            "在固定视角标出目标规划区和禁入区",
            "先核对位置和路线，再调整扫描行距",
            "仿真检查当前任务与规划结果",
        )[s.stage])
        self.reference.setText(
            "●  项目回放 · 空间基准已锁定"
            if s.replay_mode else "●  空间基准已锁定" if s.capture.reference_ready else "空间基准待建立"
        )
        tone(self.reference, "good" if s.capture.reference_ready else "")
        self.page_stack.setCurrentIndex(s.stage)
        self.pages[s.stage].update_state(s)
        names = VIEWS[s.stage]
        for i, btn in enumerate(self.view_buttons):
            btn.setVisible(i < len(names))
            if i < len(names):
                btn.setProperty("view_name", names[i])
                btn.setText(self.view_labels.get(names[i], names[i]))
                tone(btn, "active" if names[i] == s.view else "")
        if s.replay_mode and isinstance(self.viewport, ReplayViewport):
            self.viewport.set_view(s.stage, s.view)
            self.viewport.focus_issue(s.issue_focus)
        else:
            self.viewport.set_scene(scene_from_state(s))
        self.palette.update_state(s, bool(self.controller.undo_stack))
        self.palette.raise_()
        notice, severity = current_notice(s)
        self.alert.setText(("●  " if severity == "normal" else "！  ") + notice)
        tone(self.alert, severity)
        primary, secondary = actions(s)
        self.action_bar.update_state(primary, secondary, notice, severity, self._status_summary(s))
        if s.replay_mode:
            self.developer_info.setText("数据源：项目已保存回放（演示自动选择）\n坐标：空间基准 · metres\n视口：只读项目数据\n临床可执行性：阻断\n规划：保留原始缺口 / 覆盖状态")
        else:
            self.developer_info.setText("参考坐标：空间基准\n原型独立运行，未连接设备。\n视口：QPainter MockViewport\n记录：仅保存演示规划 JSON")

    def _switch_to_mock_viewport(self):
        """Leave read-only replay without leaving a ReplayViewport behind."""
        old = self.viewport
        self.work_layout.removeWidget(old)
        old.removeEventFilter(self)
        old.deleteLater()
        self.viewport = create_viewport(self.center)
        self.work_layout.addWidget(self.viewport, 1)
        try:
            self.reset_button.clicked.disconnect()
        except RuntimeError:
            pass
        self.reset_button.clicked.connect(self.viewport.reset_view)
        try:
            self.fit_button.clicked.disconnect()
            self.one_button.clicked.disconnect()
        except RuntimeError:
            pass
        self.fit_button.clicked.connect(self.viewport.reset_view)
        self.one_button.clicked.connect(self.viewport.reset_view)
        self.viewport.regions_edited.connect(self.controller.set_regions)
        self.viewport.installEventFilter(self)
        self.palette.deleteLater()
        self.palette = ToolPalette(self.viewport)
        self.palette.move(12, 12)
        self.palette.tool_requested.connect(self.controller.set_tool)
        self.palette.undo_requested.connect(self.controller.undo)
        self.palette.clear_requested.connect(self.clear_regions)
        self.project_replay = None

    @staticmethod
    def _status_summary(s):
        if s.replay_mode:
            project = s.replay_label.replace("项目回放 · ", "")
            return f"{project}  |  ● 相机：回放  |  FPS —  |  分辨率 —  |  GPU —  |  存储 —"
        return f"{s.patient.identifier}  |  ● 相机：已连接  |  FPS 30  |  1920 × 1080  |  GPU 28%  |  存储 12.4 / 100 GB"

    def toggle_maximized(self):
        self.showNormal() if self.isMaximized() else self.showMaximized()

    def ask_discard(self, title, text):
        return QMessageBox.question(self, title, text, QMessageBox.Yes | QMessageBox.No, QMessageBox.No) == QMessageBox.Yes

    def perform_action(self, key):
        primary, secondary = actions(self.controller.state)
        action = next((a for a in [primary, *secondary] if a.key == key), None)
        if not action or not action.enabled:
            return
        if action.destructive and not self.ask_discard("重新开始采集", "重新采集会清除当前采集草稿、选区及未保存规划。已保存的历史记录会保留。继续？"):
            return
        self.controller.dispatch(key)

    def clear_regions(self):
        if self.controller.state.region.strokes and self.ask_discard("清空区域标记", "清空当前规划区和禁入区？可使用撤销恢复。"):
            self.controller.clear_regions()

    def load_example(self):
        if self.controller.state.region.strokes and not self.ask_discard("替换区域标记", "用示例替换当前标记？可使用撤销恢复。"):
            return
        self.controller.load_example()

    def change_scenario(self, key):
        s = self.controller.state
        if key == s.scenario:
            return
        if s.capture.reference_ready and not s.plan.saved_id:
            if not self.ask_discard("切换原型场景", "切换会丢弃当前采集与未保存规划，并载入新的模拟任务。继续？"):
                self.header.update_state(s)
                return
        self.controller.set_scenario(key)

    def build_drawers(self):
        self.history_dock = QDockWidget("规划历史 · 演示记录", self)
        self.history_dock.setObjectName("historyDock")
        self.history_dock.setAllowedAreas(Qt.RightDockWidgetArea)
        history = QWidget()
        col = QVBoxLayout(history)
        col.setContentsMargins(18, 18, 18, 18)
        col.addWidget(text_label("本机保存的研发规划预览", "sectionTitle"))
        col.addWidget(text_label("只读查看，不改变当前任务。", "muted"))
        self.history_list = QListWidget()
        self.history_list.setMinimumWidth(280)
        self.history_list.currentRowChanged.connect(self.show_record)
        col.addWidget(self.history_list, 1)
        self.history_detail = QTextBrowser()
        col.addWidget(self.history_detail, 1)
        self.history_dock.setWidget(history)
        self.addDockWidget(Qt.RightDockWidgetArea, self.history_dock)
        self.history_dock.hide()
        self.settings_dock = QDockWidget("显示设置", self)
        self.settings_dock.setObjectName("settingsDock")
        settings = QWidget()
        settings_col = QVBoxLayout(settings)
        settings_col.setContentsMargins(18, 18, 18, 18)
        checkbox = QCheckBox("显示参考坐标轴")
        checkbox.setMinimumHeight(42)
        checkbox.toggled.connect(self.set_axes)
        self.axes_box = checkbox
        settings_col.addWidget(checkbox)
        settings_col.addWidget(text_label("研发信息", "sectionTitle"))
        self.developer_info = text_label("", "muted", True)
        settings_col.addWidget(self.developer_info)
        settings_col.addStretch()
        self.settings_dock.setWidget(settings)
        self.addDockWidget(Qt.RightDockWidgetArea, self.settings_dock)
        self.settings_dock.hide()
        self.history_records = []

    def set_axes(self, enabled):
        self.controller.set_axes(enabled)

    def show_history(self):
        self.settings_dock.hide()
        self.history_dock.setWindowTitle("项目回放 · 只读" if self.controller.state.replay_mode else "规划历史 · 演示记录")
        self.refresh_history()
        self.history_dock.setVisible(not self.history_dock.isVisible())

    def show_settings(self):
        self.history_dock.hide()
        self.settings_dock.setVisible(not self.settings_dock.isVisible())

    def refresh_history(self):
        if self.controller.state.replay_mode:
            self.history_records = []
            self.history_list.clear()
            self.history_list.addItem("当前项目回放\n只读数据，不写入规划历史")
            self.history_detail.setPlainText(
                f"{self.controller.state.replay_label}\n\n"
                "这是已保存的项目回放，不是新的规划版本。\n"
                "原始数据、区域和轨迹不会被当前原型修改。"
            )
            return
        self.history_records = self.controller.store.records()
        self.history_list.clear()
        for r in self.history_records:
            self.history_list.addItem(f"{r['saved_at'].replace('T', ' ')}\n{r['patient']['name']} · 腹部 · 预览 {r['plan']['revision']:02d}")
        if self.history_records:
            self.history_list.setCurrentRow(0)
        else:
            self.history_detail.setPlainText("暂无已保存规划。\n完成规划确认后，记录会显示在这里。")

    def show_record(self, index):
        if not 0 <= index < len(self.history_records):
            return
        r = self.history_records[index]
        self.history_detail.setPlainText(
            f"研发规划预览 / 只读\n\n患者：{r['patient']['name']}\n任务：{r['patient']['task']}\n行距：{r['final_spacing_mm']:.2f} mm\n来源：{r['spacing_source']}\n参数性质：占位值，非设备参数\n生成：{r['plan']['generated_at'].replace('T', ' ')}\n\n包含规划区、禁入区、模拟轨迹和模型来源。\n未包含治疗执行控制。"
        )

    def closeEvent(self, event):
        s = self.controller.state
        if s.replay_mode:
            self.controller.capture_timer.stop()
            self.controller.operation_timer.stop()
            event.accept()
            return
        if s.plan.phase in ("ready", "failed") and not s.plan.saved_id:
            if not self.ask_discard("关闭工作台", "当前规划尚未保存，关闭会丢弃该草稿。继续？"):
                event.ignore()
                return
        self.controller.capture_timer.stop()
        self.controller.operation_timer.stop()
        event.accept()

MEDICAL_QSS = '/* Medical Dark Pro — high contrast, touch-ready Qt workstation theme */\nQWidget { font-family: "Segoe UI", "Microsoft YaHei", sans-serif; font-size: 13px; color: #e6edf3; selection-background-color: #1d596a; }\nQWidget#root, QWidget#center { background: #0b1118; }\nQFrame#patientHeader { background: #111923; border-bottom: 1px solid #263443; }\nQLabel#brandLogo { background: transparent; }\nQFrame#headerSeparator { background: #263443; max-width: 1px; }\nQLabel#brand { color: #e6edf3; font-size: 18px; font-weight: 700; letter-spacing: .2px; }\nQLabel#brandCaption { color: #86a2b5; font-size: 11px; }\nQLabel#patientName { color: #f1f7fb; font-size: 14px; font-weight: 600; }\nQLabel#muted { color: #8b9aaa; font-size: 11px; }\nQLabel#deviceReady { color: #22c997; font-size: 13px; font-weight: 600; }\nQLabel#modeBadge { color: #8b9aaa; font-size: 11px; padding-top: 2px; }\nQComboBox#scenario { background: #151f2b; color: #e6edf3; border: 1px solid #263443; border-radius: 5px; padding: 6px 9px; min-height: 28px; }\nQComboBox#scenario:hover { border-color: #3c8795; }\nQComboBox#scenario QAbstractItemView { background: #111923; color: #e6edf3; selection-background-color: #1d596a; }\nQLabel#globalStatus { background: #102c2d; color: #a5f3d4; border-bottom: 1px solid #1d655d; padding: 5px 22px; font-size: 12px; }\nQLabel#globalStatus[tone="warning"] { background: #342b16; color: #ffd166; border-color: #806320; }\nQLabel#globalStatus[tone="error"] { background: #351c29; color: #ff9ab4; border-color: #8a2d4b; }\nQFrame#workflowSidebar { background: #0f171f; border-right: 1px solid #263443; }\nQLabel#sidebarHeading { color: #7e98aa; font-size: 11px; letter-spacing: 1px; padding: 0 10px 10px 10px; }\nQPushButton#workflowStep { background: transparent; border: none; padding: 0; text-align: left; }\nQPushButton#workflowStep:hover { background: #12232d; }\nQPushButton#workflowStep:disabled { color: #607181; }\nQPushButton#stepButton { background: transparent; border: 1px solid transparent; border-radius: 5px; text-align: left; padding: 6px 7px; font-size: 11px; color: #a8bdca; }\nQPushButton#stepButton:hover { background: #18232f; border-color: #2d4557; }\nQPushButton#stepButton[tone="active"] { background: #172f39; border: none; border-left: 2px solid #10bfe3; color: #e6edf3; font-weight: 600; }\nQPushButton#stepButton[tone="done"] { color: #79e4c2; }\nQPushButton#stepButton[tone="warning"] { color: #ffd166; }\nQPushButton#stepButton:disabled { color: #5f7483; }\nQLabel#prototypeBadge { color: #70d5e4; border-top: 1px solid #2b3b4d; padding-top: 12px; font-size: 11px; }\nQFrame#titleBar { background: #111923; border-bottom: 1px solid #263443; }\nQLabel#pageTitle { color: #e6edf3; font-size: 20px; font-weight: 650; }\nQLabel#pageSubtitle { color: #8b9aaa; font-size: 12px; }\nQLabel#referenceStatus { color: #8b9aaa; font-size: 11px; }\nQLabel#referenceStatus[tone="good"] { color: #68dfbd; }\nQFrame#viewBar { background: #151f2b; border-bottom: 1px solid #263443; }\nQPushButton#viewButton { background: transparent; color: #8ea7b7; border: 1px solid transparent; border-radius: 4px; padding: 4px 11px; }\nQPushButton#viewButton:hover { background: #182a33; color: #d9e8ed; }\nQPushButton#viewButton[tone="active"] { background: #18353e; border-color: #2f7180; color: #e6edf3; }\nQPushButton#viewTool { background: transparent; color: #aab7c2; border: none; border-radius: 4px; padding: 4px 9px; font-size: 12px; }\nQPushButton#viewTool:hover { background: #182a33; color: #e6edf3; }\nQScrollArea#rightPanel { background: #111923; border: none; }\nQWidget#statusPanel { background: #111923; }\nQLabel#sectionTitle { color: #e7edf4; font-size: 17px; font-weight: 600; padding: 5px 0 6px 0; }\nQLabel#rowName { color: #8998a8; font-size: 13px; }\nQLabel#rowValue { color: #e7edf4; font-size: 13px; font-weight: 500; }\nQLabel#rowValue[tone="good"], QLabel#phaseLabel[tone="good"] { color: #06d6a0; }\nQLabel#rowValue[tone="warning"], QLabel#phaseLabel[tone="warning"] { color: #ffd166; }\nQLabel#rowValue[tone="bad"], QLabel#phaseLabel[tone="bad"] { color: #ef476f; }\nQLabel#phaseLabel { color: #dcebf2; font-size: 18px; font-weight: 650; padding: 4px 0 8px 0; }\nQLabel#panelHint { background: transparent; border-left: 2px solid #2f7185; padding: 6px 9px; color: #9eb1bd; font-size: 11px; }\nQLabel#checkSummary { color: #9deacb; font-size: 12px; padding: 5px 0; }\nQLabel#savedNotice { color: #8fd2ca; font-size: 12px; padding-top: 8px; }\nQPushButton#inspectionCard { background: transparent; border: none; border-left: 2px solid #2a3b4f; border-radius: 4px; color: #cfe2ea; text-align: left; padding: 7px 10px; }\nQPushButton#inspectionCard:hover { background: #18232f; border-left-color: #3c8795; }\nQPushButton#inspectionCard[tone="good"] { border-left: 3px solid #06d6a0; color: #a8efd7; }\nQPushButton#inspectionCard[tone="warning"] { border-left: 3px solid #ffd166; color: #ffe5a2; }\nQPushButton#inspectionCard[tone="bad"] { border-left: 3px solid #ef476f; color: #ffb1c3; }\nQLabel#spacingValue { color: #7de7f1; font-size: 19px; font-weight: 650; padding: 8px 0; }\nQLabel#spacingValue[tone="bad"], QLabel#inlineError { color: #ff8daa; }\nQLabel#inlineError { font-size: 12px; }\nQFrame#divider { background: #202a37; border: none; }\nQPushButton#issueButton { background: #3a2d18; color: #ffd166; border: 1px solid #8d7028; border-radius: 5px; text-align: left; padding: 8px 10px; font-size: 12px; }\nQPushButton#issueButton:hover { background: #4b381b; border-color: #ffd166; }\nQPushButton#quietButton, QToolButton#detailsButton { background: #17212c; color: #b9d0dc; border: 1px solid #385067; border-radius: 5px; padding: 5px 11px; }\nQPushButton#quietButton:hover, QToolButton#detailsButton:hover { background: #1d3948; border-color: #3c8795; }\nQPushButton#quietButton:disabled { color: #607584; border-color: #2b3b4b; }\nQLineEdit#parameterInput { background: #0f1822; color: #e0f8fb; border: 1px solid #42627a; border-radius: 5px; padding: 2px 8px; font-size: 14px; }\nQLineEdit#parameterInput:focus { border: 1px solid #3c8795; }\nQLineEdit#parameterInput:disabled { color: #617888; border-color: #2b3b4b; background: #141d27; }\nQRadioButton { spacing: 8px; font-size: 12px; }\nQRadioButton::indicator { width: 14px; height: 14px; border-radius: 8px; border: 1px solid #5d7b8d; background: #101a24; }\nQRadioButton::indicator:checked { background: #2e8492; border: 2px solid #c9f7ff; }\nQProgressBar#captureProgress { background: #0c141d; border: 1px solid #365268; border-radius: 4px; min-height: 22px; color: #e0f2f2; text-align: center; font-size: 11px; }\nQProgressBar#captureProgress::chunk { background: #22c997; border-radius: 3px; }\nQProgressBar#inspectorProgress { background: #14202c; border: none; border-radius: 4px; min-height: 8px; max-height: 8px; }\nQProgressBar#inspectorProgress::chunk { background: #12bfe6; border-radius: 4px; }\nQFrame#actionBar { background: #111923; border-top: 1px solid #263443; }\nQLabel#actionNotice { color: #b9eedb; font-size: 12px; padding: 0 8px; }\nQLabel#actionNotice[tone="warning"] { color: #ffd166; }\nQLabel#actionNotice[tone="error"] { color: #ff9ab4; }\nQLabel#statusTelemetry { color: #8193a2; font-size: 11px; padding-left: 8px; }\nQPushButton#primaryButton { background: #10bfe3; border: 1px solid #10bfe3; border-radius: 5px; color: #06151b; font-size: 14px; font-weight: 700; padding: 7px 20px; }\nQPushButton#primaryButton:hover { background: #25c9e7; border-color: #67dced; }\nQPushButton#primaryButton:pressed { background: #0e98b4; }\nQPushButton#primaryButton:disabled { background: #2d4350; border-color: #415867; color: #8aa0aa; }\nQPushButton#secondaryButton { background: #192632; border: 1px solid #4a6275; border-radius: 6px; padding: 7px 16px; color: #d7e8ef; }\nQPushButton#secondaryButton:hover { background: #214052; border-color: #456273; }\nQFrame#toolPalette { background: rgba(21, 31, 43, 240); border: 1px solid #263443; border-radius: 6px; }\nQPushButton#toolButton { background: transparent; border: 1px solid transparent; border-radius: 4px; color: #cbe0e8; padding: 5px 7px; text-align: left; font-size: 12px; }\nQPushButton#toolButton[tone="active"] { background: #1a3a46; border-color: #3c8795; color: #ffffff; }\nQPushButton#toolButton:hover { background: #1c3742; }\nQPushButton#toolButton:disabled { color: #607b89; }\nQLabel#viewportTag { background: rgba(8, 17, 26, 225); color: #c3e8ef; padding: 6px 9px; border-radius: 5px; font-size: 11px; }\nQLabel#viewportHint { background: #3b202b; color: #ffb2c3; border-left: 3px solid #ef476f; border-radius: 5px; padding: 10px; font-size: 12px; }\nQLabel#viewportLegend { background: rgba(8, 17, 26, 225); color: #c3dbe4; padding: 7px 9px; border-radius: 5px; font-size: 11px; }\nQDockWidget { color: #dcebf2; background: #161d27; }\nQDockWidget > QWidget { background: #161d27; }\nQListWidget, QTextBrowser { background: #111a24; border: 1px solid #30465a; color: #d0e0e7; padding: 7px; }\nQListWidget::item { min-height: 54px; padding: 8px; border-bottom: 1px solid #26384a; }\nQListWidget::item:selected { background: #07566a; color: #ffffff; }\nQScrollBar:vertical { background: #111a24; width: 8px; }\nQScrollBar::handle:vertical { background: #42627a; min-height: 24px; border-radius: 4px; }\n'

"""Launch with: python main.py"""

import sys
from pathlib import Path

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication




def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("三维采集与轨迹规划工作台 · 原型")
    app.setFont(QFont("Microsoft YaHei", 10))
    qss = None
    app.setStyleSheet(MEDICAL_QSS)
    window = MainWindow(records_dir=Path(__file__).resolve().parent / "demo_records")
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
