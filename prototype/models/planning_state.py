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
