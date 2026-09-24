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
