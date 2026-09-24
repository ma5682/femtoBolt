"""Public state-model entry points used by the high-fidelity prototype.

The production prototype keeps its state classes in :mod:`app.app_state` and
the small, focused state modules next to it.  This module is intentionally a
thin compatibility layer so that pages/widgets can use the conventional
``models.app_state`` import without creating a second source of truth.
"""

from enum import IntEnum

from app.app_state import AppState
from models.capture_state import CaptureState
from models.planning_state import PlanningParameters, PlanningState, RouteSegment
from models.task_state import Patient, RegionState, Stroke


class WorkflowStage(IntEnum):
    """The four clinician-facing stages of the prototype workflow."""

    CAPTURE = 0
    REGION = 1
    PLANNING = 2
    CONFIRM = 3


# Names used by a few early prototype screens and external smoke tests.
Stage = WorkflowStage
WorkflowState = AppState

SCENARIO_LABELS = {
    "normal": "正常流程",
    "quality": "数据缺失需补采",
    "geometry": "三维几何缺失",
    "route": "禁区干涉 / 轨迹断开",
    "parameter": "参数超限",
}


def stage_label(stage: int | WorkflowStage) -> str:
    """Return a short Chinese label for a workflow stage."""

    labels = ("三维采集", "区域标记", "轨迹规划", "运行预览")
    try:
        return labels[int(stage)]
    except (TypeError, ValueError, IndexError):
        return "未知步骤"


def scenario_label(scenario: str) -> str:
    """Return a display label without exposing the internal scenario key."""

    return SCENARIO_LABELS.get(scenario, scenario or "正常流程")


__all__ = [
    "AppState",
    "WorkflowState",
    "WorkflowStage",
    "Stage",
    "Patient",
    "CaptureState",
    "RegionState",
    "Stroke",
    "PlanningParameters",
    "PlanningState",
    "RouteSegment",
    "SCENARIO_LABELS",
    "stage_label",
    "scenario_label",
]
