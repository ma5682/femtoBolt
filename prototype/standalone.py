"""Generated standalone mock workstation. Run: python standalone.py"""

"""Fixed navy workstation palette. Keep medical.qss in sync with these tokens."""
COLORS = {
    "background": "#08131F", "panel": "#0D1B29", "secondary": "#102233",
    "control": "#142A3D", "hover": "#193B55", "current": "#086E91",
    "primary": "#087DDE", "primary_hover": "#0A91F5", "primary_pressed": "#0567BC",
    "text": "#EAF2F8", "muted": "#91A9BB", "subtle": "#637B8D",
    "border": "#20384A", "good": "#22D3A6", "warning": "#F2B84B",
    "error": "#F06464", "pending": "#F29B45", "viewport": "#06101A",
    "header": "#0B1927", "footer": "#0C2030", "grid": "#173042",
}
SIZES = {"button": 42, "tool": 40, "primary": 52, "step": 80, "status_row": 36}
TYPE = {"title": 28, "state": 24, "panel": 18, "step": 17,
        "button": 15, "body": 14, "description": 13, "helper": 12}

"""Small workstation icons rendered with Qt paths, independent of UI fonts."""

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap


def workstation_icon(name: str, color: str = "#91A9BB") -> QIcon:
    """Return a 24 px vector-style icon backed by a crisp 2× raster.

    Drawing in logical pixels keeps the stroke width consistent across the
    toolbar. Unknown names raise early so a typo cannot create an empty button.
    """
    supported = {"fit", "zoom", "pan", "fullscreen", "cube"}
    if name not in supported:
        raise ValueError(f"Unknown workstation icon: {name}")
    pixmap = QPixmap(48, 48)
    pixmap.setDevicePixelRatio(2.0)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setPen(QPen(QColor(color), 1.6, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    painter.setBrush(Qt.NoBrush)

    def polyline(*points):
        path = QPainterPath(QPointF(*points[0]))
        for point in points[1:]:
            path.lineTo(QPointF(*point))
        painter.drawPath(path)

    if name in ("fit", "fullscreen"):
        inset, length = (4, 4) if name == "fit" else (3, 5)
        outer = 24 - inset
        polyline((inset, inset + length), (inset, inset), (inset + length, inset))
        polyline((outer - length, inset), (outer, inset), (outer, inset + length))
        polyline((outer, outer - length), (outer, outer), (outer - length, outer))
        polyline((inset + length, outer), (inset, outer), (inset, outer - length))
        if name == "fit":
            painter.drawRoundedRect(QRectF(8, 8, 8, 8), 0.6, 0.6)
    elif name == "zoom":
        painter.drawEllipse(QRectF(3.5, 3.5, 12.5, 12.5))
        polyline((14.5, 14.5), (20.5, 20.5))
        polyline((6.8, 9.75), (12.7, 9.75))
        polyline((9.75, 6.8), (9.75, 12.7))
    elif name == "pan":
        polyline((12, 3), (12, 21))
        polyline((3, 12), (21, 12))
        polyline((9, 6), (12, 3), (15, 6))
        polyline((9, 18), (12, 21), (15, 18))
        polyline((6, 9), (3, 12), (6, 15))
        polyline((18, 9), (21, 12), (18, 15))
    else:  # cube
        polyline((12, 2.8), (20, 7.3), (20, 16.7), (12, 21.2), (4, 16.7), (4, 7.3), (12, 2.8))
        polyline((4, 7.3), (12, 12), (20, 7.3))
        polyline((12, 12), (12, 21.2))

    painter.end()
    return QIcon(pixmap)

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

STAGES = ("三维采集", "区域标记", "轨迹规划", "规划确认")
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
        label = text_label(title, "sectionTitle")
        self.column.addWidget(label)
        return label

    def divider(self):
        line = QFrame()
        line.setObjectName("divider")
        line.setFixedHeight(1)
        self.column.addWidget(line)

    def row(self, title):
        item = QWidget()
        item.setMinimumHeight(36)
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
        brand.setSpacing(4)
        brand.addWidget(text_label("三维采集与轨迹规划工作台", "brand"))
        self.reference_status = text_label("参考位置未建立", "referenceStatus")
        brand.addWidget(self.reference_status)
        row.addLayout(brand, 1)
        # Keep task metadata available on hover rather than adding a second
        # title-sized information block to the header.
        self.patient = text_label("", "patientName")
        self.task = text_label("", "muted")
        self.patient.hide(); self.task.hide()
        demo = QVBoxLayout()
        demo.setSpacing(4)
        demo.addWidget(text_label("研发场景模拟", "muted"))
        self.scenario = QComboBox()
        self.scenario.setObjectName("scenario")
        for key, name in SCENARIOS.items():
            self.scenario.addItem(name, key)
        self.scenario.setFixedSize(184, 42)
        self.scenario.currentIndexChanged.connect(lambda: self.scenario_requested.emit(self.scenario.currentData()))
        demo.addWidget(self.scenario)
        row.addLayout(demo)
        for text, signal in (("历史记录", self.history_requested), ("高级工具", self.settings_requested)):
            button = QPushButton(text)
            button.setObjectName("quietButton")
            button.setFixedHeight(42)
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
            if s.replay_mode and self.scenario.findData("replay") < 0:
                self.scenario.addItem("项目回放 · 只读", "replay")
            self.scenario.setCurrentIndex(self.scenario.findData("replay" if s.replay_mode else s.scenario))
            if not s.replay_mode and self.scenario.findData("replay") >= 0:
                self.scenario.removeItem(self.scenario.findData("replay"))
        self.scenario.setEnabled(not s.replay_mode)
        established = s.capture.reference_ready
        self.reference_status.setText("●  项目回放 · 参考位置已记录" if s.replay_mode else
                                      "●  参考位置已建立 · 模拟采集" if established else "○  参考位置未建立 · 原型演示")
        tone(self.reference_status, "good" if established else "warning")
        self.reference_status.setToolTip(self.patient.text()+"\n"+self.task.text())

from PySide6.QtCore import Qt, Signal, QRectF, QPointF
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QFrame, QHBoxLayout, QPushButton, QScrollArea, QVBoxLayout, QWidget



def _font(size, bold=False):
    font = QFont('Microsoft YaHei')
    font.setPixelSize(size)
    font.setWeight(QFont.DemiBold if bold else QFont.Normal)
    return font


class WorkflowStep(QPushButton):
    """A milestone; clicking only requests navigation within the existing flow."""

    def __init__(self, index, title, subtitle, parent=None):
        super().__init__(parent)
        self.index, self.title, self.subtitle = index, title, subtitle
        self.active = self.warning = self.completed = self.blocked = False
        self.setObjectName('workflowStep')
        self.setFlat(True)
        self.setFixedHeight(80)
        self.setCursor(Qt.PointingHandCursor)
        self.setAccessibleName(f'{index + 1:02d} {title}')

    def set_state(self, active, warning=False, completed=False, blocked=False):
        self.active, self.warning = bool(active), bool(warning)
        self.completed, self.blocked = bool(completed), bool(blocked)
        self.setAccessibleDescription('需处理' if blocked else '当前' if active else '已完成' if completed else '待处理' if warning else '未执行')
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        rect = QRectF(self.rect()).adjusted(.5, .5, -.5, -.5)
        if self.active:
            p.setBrush(QColor('#103D57'))
            p.setPen(QPen(QColor('#087FA8'), 1))
            p.drawRoundedRect(rect, 5, 5)
        elif self.underMouse() and self.isEnabled():
            p.setBrush(QColor('#142A3D'))
            p.setPen(Qt.NoPen)
            p.drawRoundedRect(rect, 5, 5)
        if self.hasFocus():
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(QColor('#91A9BB'), 1, Qt.DotLine))
            p.drawRoundedRect(rect.adjusted(2, 2, -2, -2), 4, 4)
        p.setPen(QPen(QColor('#355368'), 1))
        if self.index:
            p.drawLine(QPointF(24, 0), QPointF(24, 10))
        if self.index < 5:
            p.drawLine(QPointF(24, 44), QPointF(24, 80))
        edge, fill, ink = '#4D687C', '#172B3E', '#91A9BB'
        marker = f'{self.index + 1:02d}'
        if self.completed:
            edge, fill, ink, marker = '#22D3A6', '#10352F', '#22D3A6', '✓'
        if self.active:
            edge, fill, ink, marker = '#13D7DC', '#13D7DC', '#06212A', f'{self.index + 1:02d}'
        if self.blocked:
            edge, fill, ink, marker = '#F06464', '#392734', '#F06464', '!'
        elif self.warning:
            edge, fill, ink, marker = '#F2B84B', '#3A3426', '#F2B84B', '!'
        p.setPen(QPen(QColor(edge), 1.25))
        p.setBrush(QColor(fill))
        p.drawEllipse(QPointF(24, 27), 16, 16)
        p.setPen(QColor(ink))
        p.setFont(_font(15, True))
        if marker == '✓':
            p.setPen(QPen(QColor(ink), 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            p.drawLine(QPointF(17, 27), QPointF(22, 32))
            p.drawLine(QPointF(22, 32), QPointF(31, 22))
        else:
            p.drawText(QRectF(8, 11, 32, 32), Qt.AlignCenter, marker)
        p.setFont(_font(17, self.active or self.completed))
        p.setPen(QColor('#EAF2F8' if self.active or self.completed else '#91A9BB'))
        p.drawText(QPointF(51, 29), self.title)
        p.setFont(_font(13))
        p.setPen(QColor('#91A9BB' if self.active else '#637B8D'))
        p.drawText(QPointF(51, 53), self.subtitle)
        p.end()


class WorkflowSidebar(QFrame):
    step_requested = Signal(int)
    STAGE_MAP = (0, 0, 0, 1, 2, 3)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('workflowSidebar')
        self.setFixedWidth(236)
        col = QVBoxLayout(self)
        col.setContentsMargins(10, 16, 10, 12)
        col.setSpacing(6)
        col.addWidget(text_label('工作流程', 'sidebarHeading'))
        self.scroll = QScrollArea()
        self.scroll.setObjectName('workflowScroll')
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content = QWidget()
        content.setObjectName('workflowSteps')
        steps = QVBoxLayout(content)
        steps.setContentsMargins(0, 0, 0, 0)
        steps.setSpacing(0)
        self.buttons = []
        self.subtitles = ('设置相机与目标参考', '多角度补充采集', '确认三维表面质量', '圈定目标与避让区域', '计算并核对扫描路线', '核对参数与保存方案')
        titles = ('建立参考位置', '移动补采', '检查模型', '区域标记', '轨迹规划', '规划确认')
        for index, title in enumerate(titles):
            button = WorkflowStep(index, title, self.subtitles[index], content)
            button.setAttribute(Qt.WA_AlwaysShowToolTips)
            button.clicked.connect(lambda _=False, stage=self.STAGE_MAP[index]: self.step_requested.emit(stage))
            steps.addWidget(button)
            self.buttons.append(button)
        steps.addStretch(1)
        self.scroll.setWidget(content)
        col.addWidget(self.scroll, 1)
        self._visible_milestone = None
        device = QFrame()
        device.setObjectName('devicePanel')
        device_col = QVBoxLayout(device)
        device_col.setContentsMargins(10, 10, 10, 8)
        device_col.setSpacing(0)
        head = QHBoxLayout()
        head.addWidget(text_label('设备状态', 'deviceTitle'))
        self.device_status = text_label('● 未连接', 'deviceRowValue')
        head.addWidget(self.device_status, 1, Qt.AlignRight)
        device_col.addLayout(head)
        self.device_values = {}
        for key, icon, name in (('camera', '▣', '相机'), ('rgb', '▧', 'RGB 图像'), ('depth', '◇', '深度图像'), ('reference', '⊙', '参考位置')):
            row = QWidget()
            row.setMinimumHeight(30)
            layout = QHBoxLayout(row)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(text_label(name, 'deviceRowName'))
            value = text_label('—', 'deviceRowValue')
            layout.addWidget(value, 1, Qt.AlignRight)
            device_col.addWidget(row)
            self.device_values[key] = value
        col.addWidget(device)
        self.mode_badge = text_label('研发模拟 · 不连接设备', 'prototypeBadge')
        self.mode_note = text_label('仅供交互预览，不用于治疗执行', 'muted', True)
        self.mode_badge.hide()
        self.mode_note.hide()
        device.setToolTip("原型演示：设备与采集状态为模拟；仅供交互预览。")

    @staticmethod
    def current_milestone(s):
        if s.stage > 0:
            return s.stage + 2
        return 0 if s.capture.phase == 'idle' else 1 if s.capture.phase == 'capturing' else 2

    def update_state(self, s):
        current = self.current_milestone(s)
        completed = (s.capture.reference_ready, s.capture.phase in ('checking', 'passed', 'failed'), s.capture.phase == 'passed', s.region.support == 'valid', s.plan_current, bool(s.plan.saved_id))
        for i, btn in enumerate(self.buttons):
            stage = self.STAGE_MAP[i]
            reason = navigation_reason(s, stage)
            if stage == 0 and i > current and s.stage == 0:
                reason = '请按底部主操作完成当前采集步骤。'
            active = i == current
            blocked = (i == 2 and s.capture.phase == 'failed') or (i == 3 and s.region.support == 'missing') or (i == 4 and s.plan.phase == 'failed')
            warning = i == 3 and s.region.support == 'partial'
            btn.set_state(active, warning, completed[i], blocked)
            btn.setEnabled((active and not s.busy) or not reason)
            btn.setToolTip(reason or ('已完成' if completed[i] and not active else '当前步骤' if active else '查看此步骤'))
            tone(btn, 'bad' if blocked else 'warning' if warning else 'current' if active else 'good' if completed[i] else 'idle')
        if self._visible_milestone != current:
            self.scroll.ensureWidgetVisible(self.buttons[current], 0, 8)
            self._visible_milestone = current
        if s.replay_mode:
            self.mode_badge.setText('项目回放 · 只读')
            self.mode_note.setText('显示已保存数据，不连接相机')
            status, status_tone = '● 未连接', 'idle'
            data = {'camera': ('○ 未连接', 'idle'), 'rgb': ('● 回放数据', 'current'), 'depth': ('● 回放数据', 'current'), 'reference': ('● 已载入', 'current')}
        else:
            self.mode_badge.setText('研发模拟 · 不连接设备')
            self.mode_note.setText('仅供交互预览，不用于治疗执行')
            active = s.capture.phase != 'idle'
            status, status_tone = ('● 模拟中', 'current') if active else ('○ 未就绪', 'idle')
            data = {'camera': ('● 模拟' if active else '○ 未就绪', 'current' if active else 'idle'), 'rgb': ('● 模拟画面' if active else '○ 未获取', 'current' if active else 'idle'), 'depth': ('● 模拟画面' if active else '○ 未获取', 'current' if active else 'idle'), 'reference': ('● 模拟已建' if s.capture.reference_ready else '○ 未建立', 'good' if s.capture.reference_ready else 'idle')}
        self.device_status.setText(status)
        tone(self.device_status, status_tone)
        for key, (text, color) in data.items():
            self.device_values[key].setText(text)
            tone(self.device_values[key], color)

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QVBoxLayout, QPushButton, QToolButton, QMenu


class ActionBar(QFrame):
    action_requested = Signal(str)
    details_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("actionBar")
        self.setFixedHeight(80)
        row = QHBoxLayout(self)
        row.setContentsMargins(16, 7, 16, 7)
        row.setSpacing(10)
        self.notice = text_label("", "actionNotice", True)
        self.notice.setMinimumWidth(230)
        copy = QVBoxLayout()
        copy.setSpacing(3)
        copy.addWidget(self.notice)
        self.telemetry = text_label("", "statusTelemetry")
        copy.addWidget(self.telemetry)
        row.addLayout(copy, 1)
        self.details = QToolButton()
        self.details.setObjectName("detailsButton")
        self.details.setText("状态详情")
        self.details.setFixedHeight(42)
        self.details.clicked.connect(self.details_requested.emit)
        row.addWidget(self.details)
        self.more = QToolButton()
        self.more.setObjectName("moreButton")
        self.more.setText("更多")
        self.more.setFixedHeight(42)
        self.more.setPopupMode(QToolButton.InstantPopup)
        self.more_menu = QMenu(self.more)
        self._menu_signature = None
        self.more.setMenu(self.more_menu)
        row.addWidget(self.more)
        self.secondary = []
        for _ in range(2):
            btn = QPushButton()
            btn.setObjectName("secondaryButton")
            btn.setFixedHeight(42)
            btn.clicked.connect(lambda _=False, b=btn: self.action_requested.emit(b.property("action")))
            btn.hide()
            self.secondary.append(btn)
        self.primary = QPushButton()
        self.primary.setObjectName("primaryButton")
        self.primary.setProperty("role", "primary")
        self.primary.setFixedSize(290, 52)
        self.primary.clicked.connect(lambda: self.action_requested.emit(self.primary.property("action")))
        row.addWidget(self.primary)

    def set_telemetry(self, text):
        self.telemetry.setText(text)

    def update_state(self, primary, secondary, notice, severity, telemetry=""):
        status = ("●  " if severity == "normal" else "！  ") + notice
        self.notice.setText(status)
        tone(self.notice, severity)
        if telemetry:
            self.telemetry.setText(telemetry)
        for btn, action in [(self.primary, primary), *zip(self.secondary, secondary)]:
            btn.setText(action.text)
            btn.setProperty("action", action.key)
            btn.setEnabled(action.enabled)
            btn.setToolTip(action.reason if not action.enabled else "")
        signature = tuple((a.key, a.text, a.enabled, a.reason) for a in secondary)
        # Capture refreshes every 650 ms; keep an open menu usable across ticks.
        if signature != self._menu_signature:
            self.more_menu.clear()
            for action in secondary:
                item = self.more_menu.addAction(action.text)
                item.setEnabled(action.enabled)
                item.setToolTip(action.reason)
                item.triggered.connect(lambda _=False, key=action.key: self.action_requested.emit(key))
            self._menu_signature = signature
        self.more.setEnabled(bool(secondary))

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
        for key, text in (("work", "规划区"), ("forbid", "禁入区"), ("erase", "擦除"), ("undo", "撤销"), ("clear", "清空标记")):
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



class CapturePage(StatusPanel):
    """Acquisition Inspector with one prominent phase and factual data rows."""

    def __init__(self):
        super().__init__()
        self.section('当前状态')
        self.phase = self.note('等待建立参考位置', 'phaseLabel')
        self.hint = self.note('保持相机与目标稳定\n在下方开始模拟采集', 'panelHint')
        self.divider()
        self.section('采集信息')
        self.camera = self.row('相机状态')
        self.rgb = self.row('RGB 图像')
        self.depth = self.row('深度图像')
        self.surface = self.row('三维表面')
        self.reference = self.row('参考位置')
        self.pose = self.reference
        self.divider()
        self.section('当前采集')
        self.frames = self.row('有效视角帧数')
        self.frames.setObjectName('metricValue')
        self.minimum_frames = self.row('建议最少帧数')
        self.value(self.minimum_frames, '3 帧')
        self.problem = self.issue('查看待补采位置  →', 'capture')
        self.problem.hide()
        self.result_note = self.note('补采会保留参考位置。\n完成后检查三维表面质量。', 'panelHint')
        self.finish()

    def update_state(self, s):
        c = s.capture
        names = {'idle': '等待建立参考位置', 'capturing': '正在模拟采集', 'checking': '正在检查模型', 'passed': '三维模型已就绪', 'failed': '质量不足，需补采'}
        hints = {'idle': '保持相机与目标稳定\n在下方开始模拟采集', 'capturing': '多角度补充目标表面\n至少采集 3 个模拟视角', 'checking': '正在检查三维表面\n检查完成后可继续', 'passed': '模型检查已完成\n可继续标记目标区域', 'failed': '发现三维数据缺失\n请定位缺失处后继续补采'}
        phase_tone = 'bad' if c.phase == 'failed' else 'good' if c.phase == 'passed' else 'current' if c.phase in ('capturing', 'checking') else 'idle'
        self.value(self.phase, names.get(c.phase, '等待确认'), phase_tone)
        self.hint.setText(hints.get(c.phase, '请按底部提示继续操作'))
        self.value(self.frames, f'{c.frames} 帧', 'good' if c.frames >= 3 else 'idle')
        if s.replay_mode:
            self.value(self.phase, '查看已保存采集', 'current')
            self.hint.setText('当前为项目数据回放\n相机未连接，不执行采集')
            self.value(self.camera, '○ 未连接', 'idle')
            self.value(self.rgb, '● 回放数据', 'current')
            self.value(self.depth, '● 回放数据', 'current')
            self.value(self.reference, '● 已载入', 'current')
            self.value(self.surface, '仅供校对', 'warning')
            self.result_note.setText('仅显示已保存采集与检查结果。')
        else:
            active = c.phase != 'idle'
            self.value(self.camera, '● 模拟中' if active else '○ 未就绪', 'current' if active else 'idle')
            self.value(self.rgb, '● 模拟画面' if active else '○ 未获取', 'current' if active else 'idle')
            self.value(self.depth, '● 模拟画面' if active else '○ 未获取', 'current' if active else 'idle')
            self.value(self.reference, '● 模拟已建立' if c.reference_ready else '○ 未建立', 'good' if c.reference_ready else 'idle')
            surface = {'idle': '等待采集', 'capturing': '采集中', 'checking': '检查中', 'passed': '● 模拟已生成', 'failed': '● 模拟有缺失'}
            self.value(self.surface, surface.get(c.phase, '等待确认'), phase_tone)
            self.result_note.setText('补采会保留参考位置。\n完成后检查三维表面质量。')
        self.problem.setVisible(c.phase == 'failed')

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
        self.value(self.work, "● 已标记" if r.has_work else "待标记", "good" if r.has_work else "")
        self.value(self.forbidden, "已标记" if r.has_forbidden else "尚未标记")
        names = {
            "unchecked": "待检查", "checking": "检查中…", "valid": "● 已确认",
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
        self.value(self.region, "● 原始规划记录" if s.replay_mode and ready else "● 已规划" if ready else "等待生成", "good" if ready else "")
        self.value(self.forbidden, "原始禁入区记录" if s.replay_mode and ready else "● 未进入 · 模拟" if ready else "等待检查", "good" if ready else "")
        if s.replay_mode and ready:
            geometry = stats.get("geometry_status", "已记录")
            geometry_label = "部分未确认" if "unconfirmed" in geometry else "已确认"
            self.value(self.geometry, geometry_label, "bad" if "unconfirmed" in geometry else "good")
            gaps = int(stats.get("gap_count", 0))
            continuity = f"局部断开 · {gaps} 个缺口" if gaps else "● 连续"
            self.value(self.continuity, continuity, "bad" if gaps else "good")
        else:
            self.value(self.geometry, "● 有效 · 模拟", "good")
            self.value(self.continuity, "局部断开" if p.phase == "failed" else "● 段内连续 · 模拟" if ready else "等待生成", "bad" if p.phase == "failed" else "good" if ready else "")
        self.route_issue.setVisible(p.phase == "failed")
        self.edge_issue.setVisible(ready and (not s.replay_mode or stats.get("coverage_status") == "partial"))



class ConfirmationPage(StatusPanel):
    def __init__(self):
        super().__init__()
        self.section("规划确认")
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
                "● 规划区与三维支持记录已载入\n"
                f"• 覆盖状态：{coverage}\n"
                f"• 扫描缺口：{gaps} 个；路线仅供只读核对"
            )
        else:
            self.warning.setText("边缘未覆盖区域需核对  →")
            self.warning.setVisible(True)
            self.checks.setText("● 规划区与三维支持已检查\n● 禁入区未进入（模拟）\n● 扫描段连续（模拟）" if s.plan_current else "当前规划需要重新检查。")
        self.saved.setText(
            "项目回放为只读记录，不保存、不覆盖原始数据。"
            if s.replay_mode else
            "● 本版本已保存，可在历史中查看。" if p.saved_id else
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
    reference_ready: bool = False
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
        reference_ready=s.capture.reference_ready,
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

    def set_navigation_tool(self, tool: str) -> None:
        raise NotImplementedError

    def zoom_in(self) -> None:
        raise NotImplementedError


def create_viewport(parent=None) -> Medical3DViewport:
    return MockViewport(parent)

"""Illustrative QPainter viewport; coordinates stay stable for marked regions."""
from copy import deepcopy
from math import cos, exp, pi, sin
from random import Random

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPainterPath, QPen, QRadialGradient
from PySide6.QtWidgets import QLabel



def color(name, alpha=None):
    result = QColor(COLORS.get(name, name))
    if alpha is not None:
        result.setAlpha(alpha)
    return result


def font(size, weight=QFont.Normal):
    result = QFont("Microsoft YaHei")
    result.setPixelSize(size)
    result.setWeight(weight)
    return result


class MockViewport(Medical3DViewport):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.state = ViewportScene()
        self.setMinimumSize(520, 380)
        self.setMouseTracking(True)
        self._drawing = None
        self._navigation_tool = "select"
        self._zoom = 1.0
        self._pan = QPointF()
        self._pan_last = None
        self._scale, self._offset = 1.0, QPointF()
        self._pulse_phase = 0.0
        self._pulse_timer = QTimer(self)
        self._pulse_timer.setInterval(55)
        self._pulse_timer.timeout.connect(self._pulse_tick)
        self._surface_points = self._make_surface_points()
        self.tag = QLabel(self)
        self.tag.setObjectName("viewportTag")
        self.tag.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.hint = QLabel(self)
        self.hint.setObjectName("viewportHint")
        self.hint.setWordWrap(True)
        self.hint.setFixedWidth(228)
        self.hint.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.legend = QLabel(self)
        self.legend.setObjectName("viewportLegend")
        self.legend.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.legend.hide()
        self.setToolTip("模拟数据 · 左键标记 · 右键平移 · 滚轮缩放")
        self.set_scene(self.state)

    def set_scene(self, scene):
        self.state = deepcopy(scene)
        self._drawing = None
        self.tag.setText(f"{scene.view} · {'等待画面' if self._is_empty() else '模拟数据'}")
        missing = scene.scenario == "geometry" and scene.region_complete and not scene.geometry_repaired
        problem = (
            "三维数据缺失\n已画区域保留，返回补采后继续。" if missing
            else "采集质量不足\n移动相机补充高亮区域。" if scene.capture_phase == "quality_bad"
            else "轨迹断开\n请返回规划检查高亮位置。" if scene.trajectory_failed
            else ""
        )
        self.hint.setText(problem)
        self.hint.setVisible(bool(problem))
        if problem or scene.issue_focus in ("geometry", "route", "edge"):
            if not self._pulse_timer.isActive():
                self._pulse_timer.start()
        else:
            self._pulse_timer.stop()
        self._position_overlays()
        self._update_cursor()
        self.update()

    def _is_empty(self):
        return self.state.stage == 0 and self.state.capture_phase == "idle"

    def set_navigation_tool(self, tool):
        if tool not in ("select", "pan", "zoom"):
            raise ValueError(f"未知视图工具: {tool}")
        self._navigation_tool = tool
        self._pan_last = None
        self._update_cursor()

    def _update_cursor(self):
        if self._navigation_tool == "pan":
            cursor = Qt.OpenHandCursor
        elif self._navigation_tool == "zoom" or self.state.editable:
            cursor = Qt.CrossCursor
        else:
            cursor = Qt.ArrowCursor
        self.setCursor(cursor)

    def zoom_in(self):
        self._zoom = min(2.2, self._zoom * 1.15)
        self.update()

    def reset_view(self):
        self._zoom = 1.0
        self._pan = QPointF()
        self.update()

    def _pulse_tick(self):
        self._pulse_phase = (self._pulse_phase + 0.22) % (2 * pi)
        self.update()

    def _pulse_strength(self):
        return 0.5 + 0.5 * sin(self._pulse_phase)

    def _position_overlays(self):
        self.tag.adjustSize()
        self.tag.move(136 if self.state.stage == 1 else 16, 14)
        self.hint.adjustSize()
        self.hint.move(max(16, self.width() - self.hint.width() - 16), 14)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._position_overlays()

    def _view_transform(self):
        scale = min(self.width() / 850, self.height() / 700) * self._zoom
        offset = QPointF((self.width() - 850 * scale) / 2, (self.height() - 700 * scale) / 2) + self._pan
        self._scale, self._offset = scale, offset
        return scale, offset

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), color("viewport"))
        self._draw_space_grid(p)
        if self._is_empty():
            self._draw_empty(p)
            p.end()
            return
        scale, offset = self._view_transform()
        p.save()
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
                    self._issue_patch(p, x - 60, y + 70, color("warning"), "边缘未覆盖")
                else:
                    self._issue_patch(p, x, y, color("error"), "轨迹断开")
        if self.state.reference_ready:
            self._draw_reference(p)
        p.restore()
        if self.state.show_axes or self.state.view in ("三维", "点云", "模型", "三维模型"):
            self._draw_axes(p)
        self._draw_legend(p)
        p.setFont(font(12))
        p.setPen(color("subtle"))
        p.drawText(QRectF(16, self.height() - 28, max(100, self.width() - 245), 18), Qt.AlignLeft | Qt.AlignVCenter, "模拟表面 · 仅用于交互演示")
        p.end()

    def _draw_space_grid(self, p):
        w, h = self.width(), self.height()
        p.save()
        p.setPen(QPen(color("grid", 76), 1))
        horizon = h * 0.57
        for x in range(0, w + 1, 64):
            p.drawLine(QPointF(x, 0), QPointF(x, horizon))
        for y in range(0, int(horizon), 64):
            p.drawLine(QPointF(0, y), QPointF(w, y))
        for x in range(-w, 2 * w + 1, max(70, w // 10)):
            p.drawLine(QPointF(w * .5, horizon - h * .12), QPointF(x, h))
        for fraction in (.06, .16, .30, .49, .72, 1.0):
            y = horizon + (h - horizon) * fraction
            p.drawLine(QPointF(0, y), QPointF(w, y))
        p.restore()

    def _draw_empty(self, p):
        p.save()
        cx, cy = self.width() / 2, self.height() / 2
        frame_w = min(self.width() * .70, 520)
        frame_h = min(self.height() * .54, 310)
        rect = QRectF(cx - frame_w / 2, cy - frame_h / 2 - 14, frame_w, frame_h)
        p.setPen(QPen(color("border"), 1.5))
        corner = 25
        for x, y, sx, sy in ((rect.left(), rect.top(), 1, 1), (rect.right(), rect.top(), -1, 1), (rect.left(), rect.bottom(), 1, -1), (rect.right(), rect.bottom(), -1, -1)):
            p.drawLine(QPointF(x, y), QPointF(x + sx * corner, y))
            p.drawLine(QPointF(x, y), QPointF(x, y + sy * corner))
        icon_center = QPointF(cx, cy - 50)
        p.setPen(QPen(color("subtle"), 1.5))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(QRectF(cx - 28, cy - 70, 56, 39), 4, 4)
        p.drawEllipse(icon_center, 10, 10)
        p.drawLine(QPointF(cx - 12, cy - 70), QPointF(cx - 6, cy - 77))
        p.drawLine(QPointF(cx - 6, cy - 77), QPointF(cx + 10, cy - 77))
        p.drawLine(QPointF(cx + 10, cy - 77), QPointF(cx + 16, cy - 70))
        p.setFont(font(22, QFont.DemiBold))
        p.setPen(color("text"))
        p.drawText(QRectF(cx - 210, cy - 5, 420, 36), Qt.AlignCenter, "等待相机画面")
        p.setFont(font(14))
        p.setPen(color("muted"))
        p.drawText(QRectF(cx - 220, cy + 37, 440, 25), Qt.AlignCenter, "请确认 Femto Bolt 已连接")
        p.setPen(QPen(color("subtle", 130), 1))
        cross_y = rect.bottom() - 21
        p.drawLine(QPointF(cx - 5, cross_y), QPointF(cx + 5, cross_y))
        p.drawLine(QPointF(cx, cross_y - 5), QPointF(cx, cross_y + 5))
        p.setFont(font(12))
        p.setPen(color("subtle"))
        p.drawText(QRectF(18, self.height() - 33, self.width() - 36, 20), Qt.AlignLeft | Qt.AlignVCenter, "参考位置未建立  /  W0 待建立")
        p.restore()

    @staticmethod
    def _surface_path():
        path = QPainterPath(QPointF(369, 59))
        path.cubicTo(393, 66, 438, 66, 461, 59)
        path.cubicTo(462, 88, 471, 107, 490, 117)
        path.cubicTo(522, 127, 578, 134, 608, 156)
        path.cubicTo(638, 177, 651, 212, 647, 240)
        path.cubicTo(635, 263, 613, 277, 588, 282)
        path.cubicTo(579, 326, 566, 357, 566, 408)
        path.cubicTo(565, 457, 557, 480, 563, 518)
        path.cubicTo(567, 551, 574, 579, 551, 600)
        path.cubicTo(500, 621, 335, 621, 281, 600)
        path.cubicTo(258, 579, 265, 551, 269, 518)
        path.cubicTo(275, 480, 267, 457, 266, 408)
        path.cubicTo(266, 357, 253, 326, 244, 282)
        path.cubicTo(219, 277, 197, 263, 185, 240)
        path.cubicTo(181, 212, 194, 177, 224, 156)
        path.cubicTo(254, 134, 310, 127, 342, 117)
        path.cubicTo(361, 107, 368, 88, 369, 59)
        path.closeSubpath()
        return path

    def _make_surface_points(self):
        rng = Random(732)
        path = self._surface_path()
        points = []
        for y in range(58, 618, 5):
            for x in range(180, 652, 5):
                px, py = x + rng.uniform(-2, 2), y + rng.uniform(-2, 2)
                if not path.contains(QPointF(px, py)):
                    continue
                center_light = max(0, cos((px - 416) / 240 * pi / 2))
                chest = exp(-((py - 233) / 90) ** 2) * (.65 + .35 * cos((px - 416) / 78))
                side_light = .10 * sin(px * .043 + py * .012)
                strength = max(.18, min(1, .20 + .49 * center_light + .19 * chest + side_light + rng.uniform(-.14, .14)))
                points.append((px, py, strength, rng.uniform(.7, 1.2)))
        return points

    @staticmethod
    def _acquired_path():
        path = QPainterPath(QPointF(338, 218))
        path.cubicTo(371, 195, 414, 210, 438, 199)
        path.cubicTo(466, 192, 483, 214, 478, 237)
        path.cubicTo(497, 248, 484, 277, 491, 302)
        path.cubicTo(482, 322, 490, 353, 474, 369)
        path.cubicTo(480, 393, 458, 410, 465, 435)
        path.cubicTo(452, 453, 464, 480, 440, 494)
        path.cubicTo(414, 504, 397, 484, 375, 491)
        path.cubicTo(353, 479, 325, 474, 320, 450)
        path.cubicTo(301, 428, 319, 406, 302, 384)
        path.cubicTo(287, 361, 305, 338, 297, 313)
        path.cubicTo(293, 286, 317, 271, 314, 249)
        path.cubicTo(312, 234, 326, 230, 338, 218)
        path.closeSubpath()
        return path

    def _draw_surface(self, p):
        body = self._surface_path()
        rgb = self.state.view in ("RGB", "固定 RGB", "RGB 路线图", "选区 RGB")
        cloud = self.state.view == "点云"
        depth_mode = self.state.view == "深度"
        mesh = not (rgb or cloud or depth_mode)
        p.save()
        p.setPen(Qt.NoPen)
        p.setBrush(color("panel", 120))
        p.drawEllipse(QRectF(235, 624, 365, 18))
        gradient = QLinearGradient(220, 0, 610, 0)
        gradient.setColorAt(0, color("viewport"))
        gradient.setColorAt(.30, color("subtle" if rgb else "secondary"))
        gradient.setColorAt(.62, color("muted" if rgb else "hover" if mesh else "control"))
        gradient.setColorAt(1, color("viewport"))
        p.setBrush(gradient)
        p.setPen(QPen(color("muted", 90), 1.1))
        p.drawPath(body)
        p.setClipPath(body)
        sheen = QRadialGradient(QPointF(387, 254), 270)
        sheen.setColorAt(0, color("text" if rgb else "muted", 40 if rgb else 35))
        sheen.setColorAt(1, color("viewport", 0))
        p.fillRect(QRectF(175, 55, 485, 570), sheen)
        if rgb or mesh:
            # Broad matte highlights describe volume without implying a photo.
            for cx, cy, radius, strength in ((334, 238, 116, 22), (490, 238, 116, 17), (405, 429, 155, 22)):
                contour = QRadialGradient(QPointF(cx, cy), radius)
                contour.setColorAt(0, color("text", strength))
                contour.setColorAt(1, color("text", 0))
                p.fillRect(QRectF(cx - radius, cy - radius, radius * 2, radius * 2), contour)
        if depth_mode:
            depth = QLinearGradient(0, 70, 0, 620)
            depth.setColorAt(0, color("primary_pressed"))
            depth.setColorAt(.25, color("primary_hover"))
            depth.setColorAt(.58, color("good"))
            depth.setColorAt(.84, color("warning"))
            depth.setColorAt(1, color("pending"))
            p.fillPath(body, depth)
        p.setBrush(Qt.NoBrush)
        if mesh or depth_mode:
            for y in range(136, 613, 23 if mesh else 35):
                curve = QPainterPath(QPointF(178, y))
                curve.cubicTo(294, y + 31, 505, y + 31, 655, y - 7)
                p.setPen(QPen(color("muted", 94) if mesh else color("viewport", 45), .9))
                p.drawPath(curve)
        if mesh:
            for x in range(230, 616, 31):
                curve = QPainterPath(QPointF(x, 118))
                curve.cubicTo(x + (416 - x) * .10, 283, x + (416 - x) * .19, 393, x, 612)
                p.setPen(QPen(color("muted", 62), .8))
                p.drawPath(curve)
        if cloud:
            for px, py, strength, radius in self._surface_points:
                alpha = int(44 + strength * 150)
                p.setPen(QPen(color("muted" if strength < .55 else "text", alpha), radius * 1.1))
                p.drawPoint(QPointF(px, py))
        p.restore()

    def _draw_capture(self, p):
        progress = min(1.0, max(.12, self.state.capture_progress / 6))
        acquired = self._acquired_path()
        p.save()
        p.setClipPath(self._surface_path())
        p.setClipRect(QRectF(175, 90, 490, 130 + 420 * progress), Qt.IntersectClip)
        p.setBrush(color("good", 25 if self.state.view == "点云" else 42 if self.state.view != "深度" else 0))
        p.setPen(QPen(color("good", 165), 1.35))
        p.drawPath(acquired)
        if self.state.view == "点云":
            for px, py, strength, radius in self._surface_points:
                if acquired.contains(QPointF(px, py)):
                    p.setPen(QPen(color("good", int(95 + 150 * strength)), radius * 1.4))
                    p.drawPoint(QPointF(px, py))
                elif 470 < px < 518 + 7 * sin(py / 22) and 208 < py < 502:
                    p.setPen(QPen(color("pending", 185), radius * 1.25))
                    p.drawPoint(QPointF(px, py))
        else:
            pending = QPainterPath(QPointF(489, 212))
            pending.cubicTo(525, 262, 513, 313, 515, 363)
            pending.cubicTo(506, 406, 521, 460, 484, 497)
            p.setPen(QPen(color("pending", 160), 2, Qt.DashLine))
            p.setBrush(Qt.NoBrush)
            p.drawPath(pending)
        p.restore()
        if self.state.capture_phase == "quality_bad":
            self._issue_patch(p, 315, 490, color("warning"), "待补采")
        if self.state.issue_focus == "geometry":
            p.save()
            p.setClipPath(self._surface_path())
            self._draw_strokes(p)
            p.restore()
            self._draw_geometry_issue(p)

    def _draw_strokes(self, p):
        for stroke in self.state.strokes:
            if not stroke.points:
                continue
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(color("good", 102) if stroke.kind == "work" else color("error", 125), 66, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
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
            if not segment.points:
                continue
            p.setPen(QPen(color("warning" if segment.kind == "scan" else "primary"), 2.6 if segment.kind == "scan" else 1.8, Qt.SolidLine if segment.kind == "scan" else Qt.DashLine, Qt.RoundCap))
            path = QPainterPath(QPointF(*segment.points[0]))
            for point in segment.points[1:]:
                path.lineTo(QPointF(*point))
            p.drawPath(path)

    def _draw_reference(self, p):
        p.save()
        origin = QPointF(415, 392)
        p.setPen(QPen(color("primary_hover"), 1.6))
        p.setBrush(color("viewport", 180))
        p.drawEllipse(origin, 10, 10)
        p.drawLine(origin + QPointF(-18, 0), origin + QPointF(18, 0))
        p.drawLine(origin + QPointF(0, -18), origin + QPointF(0, 18))
        p.setFont(font(14, QFont.DemiBold))
        p.setPen(color("text"))
        p.drawText(origin + QPointF(17, -14), "W0")
        p.restore()

    def _draw_axes(self, p):
        p.save()
        origin = QPointF(44, self.height() - 66)
        p.setFont(font(12, QFont.DemiBold))
        for dx, dy, name, tone in ((38, 14, "X", "error"), (29, -20, "Y", "good"), (0, -43, "Z", "primary")):
            target = origin + QPointF(dx, dy)
            p.setPen(QPen(color(tone), 1.7))
            p.drawLine(origin, target)
            p.drawEllipse(target, 1.8, 1.8)
            p.drawText(target + QPointF(5, -2), name)
        p.restore()

    def _draw_legend(self, p):
        p.save()
        if self.state.stage == 0:
            rows = [("good", "已采区域"), ("pending", "待补采"), ("primary", "参考位置（W0）" if self.state.reference_ready else "参考位置待建立")]
        elif self.state.stage == 1:
            rows = [("good", "规划区"), ("error", "禁入区")]
            if self.state.scenario == "geometry" and not self.state.geometry_repaired:
                rows.append(("warning", "缺失三维数据"))
        else:
            rows = [("warning", "扫描段"), ("primary", "连接段（不计覆盖）"), ("error", "禁入区")]
        width, height = 190, 18 + 25 * len(rows)
        x, y = self.width() - width - 16, self.height() - height - 16
        p.setBrush(color("panel", 238))
        p.setPen(QPen(color("border"), 1))
        p.drawRoundedRect(QRectF(x, y, width, height), 5, 5)
        p.setFont(font(12))
        for index, (tone, label) in enumerate(rows):
            cy = y + 19 + index * 25
            p.setPen(Qt.NoPen)
            p.setBrush(color(tone))
            p.drawEllipse(QPointF(x + 17, cy), 4, 4)
            p.setPen(color("muted"))
            p.drawText(QRectF(x + 30, cy - 10, width - 36, 20), Qt.AlignVCenter, label)
        p.restore()

    def _draw_geometry_issue(self, p):
        if self.state.scenario == "geometry" and self.state.region_complete and not self.state.geometry_repaired:
            x, y = self._work_center()
            self._issue_patch(p, x, y, color("warning"), "缺失三维")

    def _work_center(self):
        points = [point for stroke in self.state.strokes if stroke.kind == "work" for point in stroke.points]
        if not points:
            return 390, 430
        return sum(point[0] for point in points) / len(points), sum(point[1] for point in points) / len(points)

    def _issue_patch(self, p, x, y, tone, label):
        p.save()
        pulse = self._pulse_strength()
        fill, outline = QColor(tone), QColor(tone)
        fill.setAlpha(35 + int(25 * pulse))
        outline.setAlpha(145 + int(85 * pulse))
        p.setBrush(fill)
        p.setPen(QPen(outline, 1.6))
        p.drawEllipse(QPointF(x, y), 48 + 5 * pulse, 37 + 4 * pulse)
        p.setBrush(color("panel", 240))
        p.drawRoundedRect(QRectF(x + 35, y - 50, 112, 30), 4, 4)
        p.setFont(font(13, QFont.DemiBold))
        p.setPen(color("text"))
        p.drawText(QRectF(x + 35, y - 50, 112, 30), Qt.AlignCenter, label)
        p.restore()

    def _scene_point(self, position):
        scale, offset = self._view_transform()
        return (position.x() - offset.x()) / scale, (position.y() - offset.y()) / scale

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton or (event.button() == Qt.LeftButton and self._navigation_tool == "pan"):
            self._pan_last = event.position()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        if event.button() == Qt.LeftButton and self._navigation_tool == "zoom":
            self.zoom_in()
            event.accept()
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
                self.update()
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
        if self._pan_last is not None:
            self._pan_last = None
            self._update_cursor()
            return
        if self._drawing is not None:
            self._drawing = None
            self.regions_edited.emit(deepcopy(self.state.strokes))
            self.update()
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        self._zoom = min(2.2, max(.65, self._zoom * (1.1 if event.angleDelta().y() > 0 else 1 / 1.1)))
        self.update()
        event.accept()

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
        self.navigation_tool = "select"
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

    def set_navigation_tool(self, tool):
        if tool not in ("select", "pan", "zoom"):
            raise ValueError(f"未知视图工具: {tool}")
        self.navigation_tool = tool
        self.drag = None
        self.setCursor(Qt.OpenHandCursor if tool == "pan" else Qt.CrossCursor if tool == "zoom" else Qt.ArrowCursor)

    def zoom_in(self):
        self.zoom = min(6, self.zoom * 1.15)
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
        image.fill(QColor(COLORS["viewport"]))
        p = QPainter(image)
        p.setClipRect(image.rect())
        self.draw_grid(p)
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
                    p.setPen(QPen(QColor(COLORS["warning" if role == "scan" else "primary"]), 1.45, Qt.SolidLine if role == "scan" else Qt.DashLine))
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
        p.fillRect(self.rect(), QColor(COLORS["viewport"]))
        if not self.data:
            self.draw_grid(p)
            p.setPen(QColor(COLORS["muted"]))
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
            p.setBrush(QColor(242, 184, 75, 18 + int(20 * pulse)))
            p.setPen(QPen(QColor(242, 184, 75, 125 + int(80 * pulse)), 2.0))
            p.drawRoundedRect(QRectF(left, top, width, height), 5, 5)
            p.setBrush(QColor(13, 27, 41, 230))
            p.setPen(QPen(QColor(COLORS["warning"]), 1))
            p.drawRoundedRect(QRectF(left + 14, top + 14, 188, 30), 5, 5)
            p.setPen(QColor(COLORS["text"]))
            p.drawText(QRectF(left + 24, top + 14, 168, 30), Qt.AlignVCenter, "三维表面支持未确认")
        p.setPen(QColor(COLORS["text"]))
        p.fillRect(QRectF(12, 12, 188, 28), QColor(13, 27, 41, 230))
        p.drawText(QRectF(22, 12, 178, 28), Qt.AlignVCenter, f"{self.mode} · 项目回放")
        p.end()

    def draw_grid(self, p):
        """A visual orientation backdrop, not a measured coordinate grid."""
        p.save()
        grid = QColor(COLORS["grid"])
        grid.setAlpha(76)
        p.setPen(QPen(grid, 1))
        horizon = self.height() * .57
        for x in range(-self.width(), 2 * self.width(), max(72, self.width() // 10)):
            p.drawLine(QPointF(self.width() * .5, horizon), QPointF(x, self.height()))
        for fraction in (.08, .18, .33, .53, .77, 1):
            y = horizon + (self.height() - horizon) * fraction
            p.drawLine(QPointF(0, y), QPointF(self.width(), y))
        p.restore()

    def draw_gaps(self, p):
        if self.stage < 2:
            return
        p.setRenderHint(QPainter.Antialiasing)
        pulse = self._pulse_strength() if self.focus == "gap" else 0.0
        p.setBrush(QColor(COLORS["panel"]))
        gaps = [item for item in self.data.trajectory["ordered_items"] if item["kind"] == "gap"]
        for item in gaps:
            for key in ("previous_point", "next_point"):
                x, y, _ = self.project(item[key]["xyz_m"])
                radius = 3 + 2 * pulse
                p.setPen(QPen(QColor(240, 100, 100, 185 + int(60 * pulse)), 1.6 + 1.0 * pulse))
                p.drawEllipse(QPointF(x, y), radius, radius)
                if self.focus == "gap":
                    p.setBrush(Qt.NoBrush)
                    p.setPen(QPen(QColor(240, 100, 100, 50 + int(85 * pulse)), 1.1))
                    p.drawEllipse(QPointF(x, y), radius + 5 + 3 * pulse, radius + 5 + 3 * pulse)
                    p.setBrush(QColor(COLORS["panel"]))
        if self.focus == "gap":
            p.setPen(QColor(COLORS["warning"]))
            p.drawText(22, 68, "红色圆点：断开段端点（缺口之间不连线）")

    def draw_focus_marker(self, p):
        """Pulse a focused geometry/edge area on both mesh and image views."""
        if self.focus not in ("geometry", "edge"):
            return
        pulse = self._pulse_strength()
        color = QColor(COLORS["warning"])
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
        if event.button() == Qt.LeftButton and self.navigation_tool == "zoom":
            self.zoom_in()
            event.accept()
            return
        if event.button() in (Qt.LeftButton, Qt.RightButton):
            self.drag = (event.button(), event.position())

    def mouseMoveEvent(self, event):
        if self.drag:
            button, previous = self.drag
            delta = event.position()-previous
            self.drag = (button, event.position())
            if button == Qt.LeftButton and self.is_3d() and self.navigation_tool == "select":
                self.yaw += delta.x()*.35
                self.pitch = min(85, max(-85, self.pitch+delta.y()*.35))
            elif button == Qt.RightButton or self.navigation_tool == "pan":
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
            return "项目回放 · 只读核对：原始扫描路线存在局部断开，请查看高亮端点。", "warning"
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
            "missing": ("高亮区域缺少可靠三维数据，请返回补采。", "error"),
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
from PySide6.QtCore import QEvent, QSize, Qt
from PySide6.QtGui import QImage, QKeySequence, QShortcut
from PySide6.QtWidgets import QCheckBox, QDialog, QDockWidget, QFrame, QHBoxLayout, QListWidget, QMainWindow, QMenu, QMessageBox, QPushButton, QScrollArea, QStackedWidget, QTextBrowser, QVBoxLayout, QWidget


class MainWindow(QMainWindow):
    def __init__(self, records_dir=None, viewport_factory=create_viewport, use_project_replay=False):
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
        if use_project_replay:
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
        work.setContentsMargins(10, 8, 10, 8)
        work.setSpacing(6)
        title_bar = QFrame()
        title_bar.setObjectName("titleBar")
        title_bar.setFixedHeight(64)
        title_row = QHBoxLayout(title_bar)
        title_row.setContentsMargins(4, 4, 4, 4)
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
        view_row.setContentsMargins(6, 5, 6, 5)
        view_row.setSpacing(4)
        self.view_labels = {"RGB": "▣  RGB", "深度": "≋  深度", "点云": "⠿  点云", "模型": "◇  模型"}
        self.view_buttons = []
        for _ in range(3):
            btn = QPushButton()
            btn.setObjectName("viewButton")
            btn.setFixedHeight(42)
            btn.setMinimumWidth(110)
            btn.clicked.connect(lambda _=False, b=btn: self.choose_view(b.property("view_name")))
            view_row.addWidget(btn)
            self.view_buttons.append(btn)
        view_row.addStretch()
        self.fit_button = QPushButton("⊡")
        self.zoom_button = QPushButton("＋")
        self.one_button = QPushButton("↔")
        self.full_button = QPushButton("⛶")
        for button, icon, tip in (
            (self.fit_button, "fit", "适配窗口"),
            (self.zoom_button, "zoom", "放大视图；滚轮可缩小"),
            (self.one_button, "pan", "平移视图；再次点击返回选择"),
            (self.full_button, "fullscreen", "全屏显示"),
        ):
            button.setText("")
            button.setIcon(workstation_icon(icon))
            button.setIconSize(QSize(22, 22))
            button.setAccessibleName(tip)
            button.setObjectName("viewTool")
            button.setFixedSize(40, 40)
            button.setToolTip(tip)
            view_row.addWidget(button)
        self.reset_button = QPushButton()
        self.reset_button.setIcon(workstation_icon("cube"))
        self.reset_button.setIconSize(QSize(22, 22))
        self.reset_button.setAccessibleName("三维显示方式")
        self.reset_button.setObjectName("viewTool")
        self.reset_button.setFixedSize(40,40)
        self.reset_button.setToolTip("三维显示方式")
        mode_menu = QMenu(self.reset_button)
        for mode in ("点云", "模型"):
            mode_menu.addAction(mode, lambda mode=mode: self.controller.set_view(mode))
        self.reset_button.setMenu(mode_menu)
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
        self._navigation_tool = "select"
        self.fit_button.clicked.connect(lambda: self.viewport.reset_view())
        self.zoom_button.clicked.connect(lambda: self.viewport.zoom_in())
        self.one_button.clicked.connect(self.toggle_pan)
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
        self.action_bar.details_requested.connect(self.show_status_details)
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
        self.page_title.setText(("三维采集", "区域标记", "轨迹规划", "规划确认")[s.stage])
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
        names = ("RGB", "深度", "三维预览") if s.stage == 0 else VIEWS[s.stage]
        for i, btn in enumerate(self.view_buttons):
            btn.setVisible(i < len(names))
            if i < len(names):
                btn.setProperty("view_name", names[i])
                btn.setText(names[i])
                selected = names[i] == s.view or names[i] == "三维预览" and s.view in ("点云","模型")
                tone(btn, "active" if selected else "")
        self.reset_button.setVisible(s.stage == 0 and s.view in ("点云","模型"))
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
        self._navigation_tool = "select"
        tone(self.one_button, "")
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
            return f"项目回放 · 只读  |  {project}  |  相机未连接"
        return f"原型演示 · 未连接设备  |  {s.patient.identifier}  |  模拟采集 {s.capture.frames} 帧"

    def choose_view(self, name):
        self.controller.set_view("点云" if name == "三维预览" else name)

    def toggle_pan(self):
        self._navigation_tool = "select" if self._navigation_tool == "pan" else "pan"
        self.viewport.set_navigation_tool(self._navigation_tool)
        tone(self.one_button, "active" if self._navigation_tool == "pan" else "")

    def show_status_details(self):
        s=self.controller.state
        notice, severity=current_notice(s)
        dialog=QDialog(self)
        dialog.setWindowTitle("当前任务状态")
        dialog.resize(580,360)
        column=QVBoxLayout(dialog)
        content=QTextBrowser()
        content.setPlainText(notice+"\n\n"+self._status_summary(s)+"\n\n"+self.developer_info.text())
        column.addWidget(content)
        close=QPushButton("关闭");close.clicked.connect(dialog.accept)
        column.addWidget(close,alignment=Qt.AlignRight)
        dialog.exec()

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

MEDICAL_QSS = '/* Fixed navy workstation palette; QPainter tokens: styles/tokens.py. */\nQWidget { font-family: "Microsoft YaHei", "Segoe UI"; font-size: 14px; color: #EAF2F8; selection-background-color: #086E91; selection-color: #EAF2F8; }\nQMainWindow, QWidget#root, QWidget#appRoot, QWidget#center { background: #08131F; }\nQLabel { background: transparent; }\nQFrame#patientHeader { background: #0B1927; border-bottom: 1px solid #20384A; }\nQLabel#brand { font-size: 28px; font-weight: 700; }\nQLabel#brandLogo { background: transparent; }\nQFrame#headerSeparator { background: #20384A; max-width: 1px; }\nQLabel#brandCaption, QLabel#patientName { color: #91A9BB; font-size: 13px; }\nQLabel#muted, QLabel[role="secondary"] { color: #91A9BB; font-size: 13px; }\nQLabel#modeBadge, QLabel#deviceReady { color: #91A9BB; font-size: 12px; }\nQLabel#referenceStatus { font-size: 14px; }\nQFrame[panel="primary"], QFrame#workflowSidebar { background: #0D1B29; border: 1px solid #20384A; }\nQFrame[panel="secondary"], QFrame#devicePanel { background: #102233; border: 1px solid #20384A; border-radius: 5px; }\nQFrame#workflowSidebar { border-left: none; border-top: none; border-bottom: none; }\nQScrollArea#stepScroll, QScrollArea#stepScroll > QWidget > QWidget { background: #0D1B29; border: none; }\nQScrollArea#workflowScroll, QWidget#workflowSteps { background: #0D1B29; border: none; }\nQLabel#sidebarHeading, QLabel#deviceTitle, QLabel#sectionTitle, QLabel[role="panelTitle"] { color: #EAF2F8; font-size: 18px; font-weight: 600; }\nQLabel#sidebarHeading { padding: 0 8px 2px 8px; }\nQLabel#deviceRowName { color: #91A9BB; font-size: 12px; }\nQLabel#deviceRowValue { font-size: 12px; }\nQLabel#prototypeBadge { color: #637B8D; font-size: 12px; }\nQPushButton#workflowStep { background: transparent; border: none; padding: 0; min-height: 80px; max-height: 80px; text-align: left; }\nQPushButton#workflowStep:focus { outline: none; }\nQFrame#titleBar { background: #08131F; }\nQLabel#pageTitle { color: #EAF2F8; font-size: 24px; font-weight: 700; }\nQLabel#pageSubtitle { color: #91A9BB; font-size: 13px; }\nQFrame#viewBar { background: #102233; border: 1px solid #20384A; border-radius: 5px; }\nQPushButton, QToolButton { min-height: 40px; padding: 0 16px; background: #142A3D; color: #EAF2F8; border: 1px solid #29465D; border-radius: 5px; font-size: 15px; }\nQPushButton:hover, QToolButton:hover { background: #193B55; border-color: #3E6682; }\nQPushButton:pressed, QToolButton:pressed { background: #103D57; border-color: #087FA8; }\nQPushButton:focus, QToolButton:focus { border-color: #0A91F5; }\nQPushButton#viewButton { min-width: 90px; padding: 0 10px; }\nQPushButton#viewButton[tone="active"], QPushButton[state="active"], QPushButton#toolButton[tone="active"] { background: #0A587A; border-color: #10A8D4; color: #EAF2F8; }\nQPushButton#viewTool { min-height: 38px; max-height: 38px; min-width: 38px; max-width: 38px; padding: 0; }\nQPushButton#viewTool[tone="active"] { background: #086E91; border-color: #10A8D4; }\nQPushButton#quietButton, QToolButton#detailsButton, QToolButton#moreButton { background: #102233; border-color: #20384A; color: #91A9BB; }\nQPushButton#quietButton:hover, QToolButton#detailsButton:hover, QToolButton#moreButton:hover { background: #193B55; color: #EAF2F8; border-color: #3E6682; }\nQPushButton#primaryButton, QPushButton[role="primary"] { min-height: 0; background: #087DDE; border: 1px solid #1597F5; border-radius: 6px; padding: 0 16px; color: #EAF2F8; font-size: 16px; font-weight: 600; }\nQPushButton#primaryButton:hover, QPushButton[role="primary"]:hover { background: #0A91F5; }\nQPushButton#primaryButton:pressed, QPushButton[role="primary"]:pressed { background: #0567BC; }\nQPushButton:disabled, QToolButton:disabled, QPushButton#primaryButton:disabled, QPushButton#viewButton:disabled, QPushButton#quietButton:disabled, QPushButton[role="primary"]:disabled { background: #112231; color: #52697B; border-color: #1C3345; }\nQPushButton#secondaryButton { background: #142A3D; }\nQScrollArea#rightPanel, QWidget#statusPanel { background: #0D1B29; border: none; }\nQScrollArea#rightPanel { border-left: 1px solid #20384A; }\nQLabel#sectionTitle { padding: 5px 0 6px 0; }\nQLabel#phaseLabel { font-size: 24px; font-weight: 700; padding: 4px 0 8px 0; }\nQLabel#metricValue { font-size: 24px; font-weight: 600; }\nQLabel#rowName { color: #91A9BB; font-size: 14px; }\nQLabel#rowValue { font-size: 14px; }\nQLabel#panelHint { color: #91A9BB; font-size: 13px; padding: 4px 0; }\nQLabel#checkSummary, QLabel#savedNotice { color: #22D3A6; font-size: 13px; padding: 5px 0; }\nQFrame#divider { background: #20384A; border: none; }\nQLabel[tone="good"], QLabel[state="good"] { color: #22D3A6; }\nQLabel[tone="warning"], QLabel[state="warning"] { color: #F2B84B; }\nQLabel[tone="bad"], QLabel[tone="error"], QLabel[state="error"] { color: #F06464; }\nQLabel[tone="current"] { color: #0A91F5; }\nQLabel[tone="idle"] { color: #637B8D; }\nQLabel[tone="pending"] { color: #F29B45; }\nQPushButton#inspectionCard { min-height: 42px; background: transparent; border: none; border-left: 2px solid #20384A; color: #EAF2F8; text-align: left; padding: 7px 10px; }\nQPushButton#inspectionCard:hover { background: #102233; }\nQPushButton#inspectionCard[tone="good"] { border-left-color: #22D3A6; }\nQPushButton#inspectionCard[tone="warning"] { border-left-color: #F2B84B; }\nQPushButton#inspectionCard[tone="bad"] { border-left-color: #F06464; }\nQPushButton#issueButton { background: #102233; border-color: #F29B45; color: #F2B84B; padding: 6px 10px; text-align: left; font-size: 13px; }\nQPushButton#issueButton:hover { background: #193B55; }\nQLabel#spacingValue { font-size: 24px; font-weight: 600; color: #EAF2F8; padding: 8px 0; }\nQLabel#spacingValue[tone="bad"], QLabel#inlineError { color: #F06464; }\nQLabel#inlineError { font-size: 13px; }\nQLineEdit, QLineEdit#parameterInput { background: #102233; color: #EAF2F8; border: 1px solid #29465D; border-radius: 5px; padding: 2px 8px; font-size: 15px; min-height: 34px; }\nQLineEdit:focus, QLineEdit#parameterInput:focus { border-color: #0A91F5; }\nQLineEdit:disabled, QLineEdit#parameterInput:disabled { background: #112231; color: #52697B; border-color: #1C3345; }\nQRadioButton, QCheckBox { spacing: 8px; min-height: 34px; font-size: 14px; }\nQRadioButton::indicator { width: 16px; height: 16px; border-radius: 9px; border: 1px solid #637B8D; background: #102233; }\nQRadioButton::indicator:checked { background: #087DDE; border: 2px solid #91A9BB; }\nQComboBox { background: #142A3D; color: #EAF2F8; border: 1px solid #29465D; border-radius: 5px; padding: 0 12px; min-height: 40px; }\nQComboBox:hover { background: #193B55; }\nQComboBox::drop-down { border: none; width: 22px; }\nQComboBox QAbstractItemView { background: #102233; color: #EAF2F8; selection-background-color: #086E91; padding: 4px; }\nQProgressBar#captureProgress { background: #142A3D; border: 1px solid #20384A; border-radius: 4px; min-height: 22px; text-align: center; }\nQProgressBar#captureProgress::chunk, QProgressBar#inspectorProgress::chunk { background: #22D3A6; border-radius: 3px; }\nQProgressBar#inspectorProgress { background: #142A3D; border: none; border-radius: 4px; min-height: 8px; max-height: 8px; }\nQFrame#actionBar { background: #0C2030; border-top: 1px solid #20384A; }\nQLabel#actionNotice { color: #EAF2F8; font-size: 14px; padding: 0 4px; }\nQLabel#actionNotice[tone="warning"] { color: #F2B84B; }\nQLabel#actionNotice[tone="error"] { color: #F06464; }\nQLabel#statusTelemetry { color: #637B8D; font-size: 12px; }\nQFrame#toolPalette { background: rgba(16, 34, 51, 240); border: 1px solid #20384A; border-radius: 5px; }\nQPushButton#toolButton { background: transparent; padding: 0 7px; text-align: left; font-size: 14px; }\nQLabel#viewportTag { background: rgba(13, 27, 41, 225); color: #91A9BB; padding: 6px 9px; border-radius: 4px; font-size: 12px; }\nQLabel#viewportHint { background: #102233; color: #F2B84B; border-left: 3px solid #F29B45; border-radius: 4px; padding: 10px; font-size: 13px; }\nQLabel#viewportLegend { background: rgba(13, 27, 41, 230); color: #EAF2F8; border: 1px solid #20384A; padding: 10px 12px; border-radius: 5px; font-size: 13px; }\nQDockWidget, QDockWidget > QWidget, QDialog, QMessageBox { color: #EAF2F8; background: #0D1B29; }\nQListWidget, QTextBrowser { background: #102233; border: 1px solid #20384A; color: #EAF2F8; padding: 8px; }\nQListWidget::item { min-height: 48px; padding: 8px; border-bottom: 1px solid #20384A; }\nQListWidget::item:selected { background: #086E91; }\nQMenu { background: #102233; border: 1px solid #20384A; padding: 5px; }\nQMenu::item { padding: 10px 18px; }\nQMenu::item:selected { background: #193B55; }\nQMenu::item:disabled { color: #637B8D; }\nQScrollBar:vertical { background: #0D1B29; width: 8px; }\nQScrollBar::handle:vertical { background: #29465D; min-height: 24px; border-radius: 4px; }\nQScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }\nQScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }\nQToolTip { background: #142A3D; color: #EAF2F8; border: 1px solid #29465D; padding: 6px; }\n'

"""Launch with: python main.py"""

import sys
import argparse
from pathlib import Path

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication




def main() -> int:
    parser = argparse.ArgumentParser(description="三维采集与规划工作台 · 独立界面原型")
    parser.add_argument("--replay", action="store_true", help="只读载入已保存项目；默认使用 Mock 演示")
    args, qt_args = parser.parse_known_args()
    app = QApplication([sys.argv[0], *qt_args])
    app.setApplicationName("三维采集与轨迹规划工作台 · 原型")
    app.setFont(QFont("Microsoft YaHei", 10))
    qss = None
    app.setStyleSheet(MEDICAL_QSS)
    window = MainWindow(records_dir=Path(__file__).resolve().parent / "demo_records", use_project_replay=args.replay)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
