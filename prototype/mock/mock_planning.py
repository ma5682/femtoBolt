"""2D demonstration lines in brush coordinates, not a 3D treatment planner."""
from math import hypot
from models.planning_state import RouteSegment
from models.task_state import Stroke


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
