"""Compact quality/inspection card panel for the prototype shell."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QLabel, QPushButton, QVBoxLayout

from widgets.status_panel import tone


class InspectionPanel(QFrame):
    """Display workflow checks as readable, clickable status cards.

    The panel intentionally contains no business logic.  Clicking a warning
    emits its key so the host can focus the corresponding viewport area.
    """

    focus_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("inspectionPanel")
        column = QVBoxLayout(self)
        column.setContentsMargins(16, 16, 16, 16)
        column.setSpacing(8)
        self.title = QLabel("质控检查")
        self.title.setObjectName("sectionTitle")
        column.addWidget(self.title)
        self.cards: dict[str, QPushButton] = {}
        for key, label in (
            ("capture", "三维采集质量"),
            ("geometry", "三维表面支持"),
            ("route", "轨迹连续性"),
            ("edge", "边缘覆盖"),
        ):
            card = QPushButton()
            card.setObjectName("inspectionCard")
            card.setMinimumHeight(48)
            card.setText(f"{label}\n—")
            card.setToolTip(label)
            card.clicked.connect(lambda _=False, k=key: self.focus_requested.emit(k))
            column.addWidget(card)
            self.cards[key] = card
        column.addStretch(1)

    @staticmethod
    def _set(card: QPushButton, label: str, value: str, status: str = ""):
        card.setText(f"{label}\n{value}")
        tone(card, status)

    def update_state(self, state):
        """Refresh cards from either the current AppState or a compatible object."""

        capture = getattr(state, "capture", None)
        region = getattr(state, "region", None)
        plan = getattr(state, "plan", None)
        replay = bool(getattr(state, "replay_mode", False))
        stats = getattr(state, "replay_stats", {}) or {}
        capture_phase = getattr(capture, "phase", "idle")
        self._set(self.cards["capture"], "三维采集质量", "已记录" if replay else ("已通过" if capture_phase == "passed" else "待检查"), "good" if replay or capture_phase == "passed" else "warning")
        support = getattr(region, "support", "unchecked")
        self._set(self.cards["geometry"], "三维表面支持", "部分未确认" if support == "partial" else "已确认" if support == "valid" else "阻断" if support == "missing" else "待检查", "bad" if support == "missing" else "warning" if support == "partial" else "good" if support == "valid" else "")
        gaps = int(stats.get("gap_count", 0) or 0)
        failed = getattr(plan, "phase", "empty") == "failed"
        self._set(self.cards["route"], "轨迹连续性", f"{gaps} 个缺口" if replay and gaps else "存在缺口" if failed else "已检查" if replay else "待生成", "bad" if gaps or failed else "good" if replay else "")
        coverage = stats.get("coverage_status") if replay else None
        self._set(self.cards["edge"], "边缘覆盖", "部分覆盖" if coverage == "partial" else "未提供" if replay else "待计算", "warning" if coverage == "partial" else "")


__all__ = ["InspectionPanel"]
