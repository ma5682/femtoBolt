"""Deterministic visual data for the standalone clinician-facing prototype.

This is display data only.  Coordinates are screen-space pixels and must not
be interpreted as patient coordinates or treatment geometry.  The existing
mock providers remain the source of workflow behaviour; this module simply
provides a stable, richer scene description for alternate screens and smoke
tests.
"""

from dataclasses import dataclass
from math import cos, pi, sin
from random import Random

from mock.mock_capture import SCENARIOS, quality_passes
from mock.mock_planning import distance_to_stroke, mock_route
from mock.mock_patient import sample_patient
from mock.mock_planning import sample_regions
from models.planning_state import RouteSegment
from models.task_state import Patient, Stroke


@dataclass(frozen=True)
class MockScene:
    """A compact scene bundle consumed by optional viewport implementations."""

    point_cloud: list[tuple[float, float, float]]
    contour: list[tuple[float, float]]
    defects: list[tuple[float, float]]
    route: list[RouteSegment]
    strokes: list[Stroke]
    patient: Patient
    scenario: str = "normal"


def generate_contour(cx: float = 520, cy: float = 390, width: float = 270, height: float = 360, points: int = 96) -> list[tuple[float, float]]:
    """Generate a smooth front-view torso outline in viewport coordinates."""

    points = max(16, int(points))
    result: list[tuple[float, float]] = []
    for i in range(points):
        a = (2 * pi * i) / points
        # A torso-like ellipse with a slightly narrower waist.
        shoulder = 1.0 + 0.06 * cos(2 * a)
        waist = 1.0 - 0.10 * cos(a)
        scale = shoulder * waist
        result.append((cx + width * 0.5 * scale * cos(a), cy + height * 0.5 * sin(a)))
    return result


def generate_point_cloud(
    count: int = 720,
    cx: float = 520,
    cy: float = 390,
    width: float = 270,
    height: float = 360,
    seed: int = 17,
) -> list[tuple[float, float, float]]:
    """Generate a deterministic, gently noisy torso point cloud.

    The third coordinate is a display depth in millimetres-like units.  It is
    deliberately not a world or patient coordinate.
    """

    rng = Random(seed)
    result: list[tuple[float, float, float]] = []
    for _ in range(max(32, int(count))):
        a = rng.random() * 2 * pi
        radius = (rng.random() ** 0.5)
        x = cx + width * 0.5 * radius * cos(a) + rng.uniform(-2.5, 2.5)
        y = cy + height * 0.5 * radius * sin(a) + rng.uniform(-2.5, 2.5)
        z = 8.0 + 20.0 * (1.0 - radius) + rng.uniform(-1.3, 1.3)
        result.append((x, y, z))
    return result


def generate_defects(scenario: str = "normal", cx: float = 520, cy: float = 390) -> list[tuple[float, float]]:
    """Return display-space defect markers for the selected demo scenario."""

    if scenario in ("quality", "geometry"):
        return [(cx - 90, cy - 112), (cx + 104, cy - 24), (cx + 55, cy + 126)]
    if scenario == "route":
        return [(cx - 12, cy + 6), (cx + 4, cy + 6)]
    if scenario == "parameter":
        return [(cx, cy)]
    return []


def build_scene(scenario: str = "normal", spacing: float = 2.4) -> MockScene:
    """Build all visual mock data with one deterministic call."""

    strokes = sample_regions()
    return MockScene(
        point_cloud=generate_point_cloud(),
        contour=generate_contour(),
        defects=generate_defects(scenario),
        route=mock_route(strokes, spacing, broken=scenario == "route"),
        strokes=strokes,
        patient=sample_patient(),
        scenario=scenario,
    )


__all__ = [
    "SCENARIOS",
    "quality_passes",
    "distance_to_stroke",
    "mock_route",
    "sample_patient",
    "sample_regions",
    "MockScene",
    "generate_contour",
    "generate_point_cloud",
    "generate_defects",
    "build_scene",
]
