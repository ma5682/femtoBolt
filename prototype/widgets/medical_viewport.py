"""The UI consumes this contract. A VTK implementation can replace the factory."""
from copy import deepcopy
from dataclasses import dataclass, field
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget
from models.task_state import Stroke
from models.planning_state import RouteSegment


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
    from widgets.viewport import MockViewport
    return MockViewport(parent)
