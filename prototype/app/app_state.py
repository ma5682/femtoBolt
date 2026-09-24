from dataclasses import dataclass, field
from models.capture_state import CaptureState
from models.task_state import Patient, RegionState
from models.planning_state import PlanningState


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
