"""Export catalog API and Qt compatibility entry points.

Filesystem discovery, project registration, and record status formatting live
in :mod:`femto.planning.catalog`.  The active PySide6 history and report widgets live in femto.app.qt_history;
this module keeps only catalog functions and clear compatibility errors for
removed standalone callers.
"""
from __future__ import annotations

from typing import Any

from femto.planning.catalog import (
    history_record_status, local_asset, register_project, scan_exports, scan_region_exports,
)


def show_history(parent: Any, reconstruction: Any) -> Any:
    """Retired compatibility name; use the Qt HistoryDialog from the app layer."""
    raise RuntimeError("历史记录查看器已迁移到 femto.app.qt_history.HistoryDialog")


def show_region_library(parent: Any, reconstruction: Any) -> Any:
    raise RuntimeError("区域历史查看器已迁移到 femto.app.qt_history.HistoryDialog")


def _show_planning_report(parent: Any, text: str) -> Any:
    raise RuntimeError("规划报告查看器已迁移到 femto.app.qt_history.ReportDialog")

def _history_record_status(row: dict, rows: list[dict]) -> str:
    return history_record_status(row, rows)


__all__ = [
    'history_record_status', 'local_asset', 'register_project',
    'scan_exports', 'scan_region_exports', '_history_record_status',
]
