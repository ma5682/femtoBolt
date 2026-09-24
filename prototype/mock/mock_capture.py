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
