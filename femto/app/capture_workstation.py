"""Small, widget-free helpers shared by the acquisition workbench.

The clinician-facing UI is implemented by :mod:`femto.app.qt_capture`.
This module deliberately contains no GUI toolkit imports so data and tests can
use the workflow contract in headless environments.
"""
from __future__ import annotations

from typing import Any

STATUS_COLORS = dict(normal='#06d6a0', current='#268bda', warning='#ffd166', error='#ef476f', idle='#8fa2b5')


def workflow_progress(*, state: str, count: int = 0, active: bool = False,
                      result: Any = None, failure: Any = None) -> list[str]:
    """Return the four acquisition stages represented by the Qt stepper."""
    del active
    blocked = state in {"lost", "tracking_lost", "camera_error"} or failure is not None
    stage = (
        3 if state == "mark" else
        2 if result is not None or state in {"processing", "loading_review", "ready"} else
        1 if count > 0 else 0
    )
    statuses = ["complete" if index < stage else "pending" for index in range(4)]
    statuses[stage] = "blocked" if blocked else "paused" if state == "cancelled" else "current"
    return statuses


__all__ = ["STATUS_COLORS", "workflow_progress"]
