from PySide6.QtCore import Qt, Signal, QRectF, QPointF
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QFrame, QPushButton, QVBoxLayout
from app.navigation import STAGES, navigation_reason
from widgets.status_panel import text_label, tone


class WorkflowStep(QPushButton):
    """Timeline step rendered like an industrial workstation navigator."""

    def __init__(self, index, title, subtitle, parent=None):
        super().__init__(parent)
        self.index = index
        self.title = title
        self.subtitle = subtitle
        self.active = False
        self.warning = False
        self.setObjectName("workflowStep")
        self.setFlat(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(82)

    def set_state(self, active, warning):
        self.active = bool(active)
        self.warning = bool(warning)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        rect = self.rect()
        if self.active:
            p.fillRect(rect, QColor("#142936"))
            p.fillRect(QRectF(0, 0, 3, rect.height()), QColor("#12bfe6"))
        center = QPointF(30, 30)
        if self.index < 3:
            p.setPen(QPen(QColor("#334656"), 1))
            p.drawLine(QPointF(center.x(), 51), QPointF(center.x(), rect.height()))
        ring = QColor("#12bfe6") if self.active else QColor("#596d7d")
        p.setBrush(QColor("#0d1721"))
        p.setPen(QPen(ring, 1.5))
        p.drawEllipse(center, 21, 21)
        p.setPen(ring if self.active else QColor("#b3c0ca"))
        p.setFont(QFont("Segoe UI", 11, QFont.DemiBold))
        p.drawText(QRectF(9, 18, 42, 25), Qt.AlignCenter, f"{self.index + 1:02d}")
        p.setPen(QColor("#e7edf4") if self.active else QColor("#c2cdd6"))
        p.setFont(QFont("Microsoft YaHei", 13, QFont.DemiBold if self.active else QFont.Normal))
        p.drawText(QPointF(64, 28), self.title)
        p.setPen(QColor("#8998a8"))
        p.setFont(QFont("Microsoft YaHei", 11))
        p.drawText(QPointF(64, 52), self.subtitle)
        if self.warning:
            p.setBrush(QColor("#e9b949"))
            p.setPen(Qt.NoPen)
            p.drawEllipse(QPointF(rect.width() - 18, 24), 4, 4)
        p.end()


class WorkflowSidebar(QFrame):
    step_requested = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("workflowSidebar")
        self.setFixedWidth(220)
        col = QVBoxLayout(self)
        col.setContentsMargins(16, 18, 14, 14)
        col.setSpacing(8)
        col.addWidget(text_label("操作流程", "sidebarHeading"))
        self.buttons = []
        self.subtitles = ("采集场景三维数据", "标记目标区域", "生成执行轨迹", "仿真与效果检查")
        for index, title in enumerate(STAGES):
            button = WorkflowStep(index, title, self.subtitles[index], self)
            button.setAttribute(Qt.WA_AlwaysShowToolTips)
            button.clicked.connect(lambda _=False, i=index: self.step_requested.emit(i))
            col.addWidget(button)
            self.buttons.append(button)
        col.addStretch()
        self.mode_badge = text_label("研发规划预览", "prototypeBadge")
        self.mode_note = text_label("模拟患者与设备\n不用于治疗执行", "muted", True)
        col.addWidget(self.mode_badge)
        col.addWidget(self.mode_note)

    def update_state(self, s):
        self.mode_badge.setText("项目回放 · 只读" if s.replay_mode else "研发规划预览")
        self.mode_note.setText("原始数据只读\n不可执行" if s.replay_mode else "模拟患者与设备\n不用于治疗执行")
        for i, btn in enumerate(self.buttons):
            reason = navigation_reason(s, i)
            active = s.stage == i
            warning = (i == 1 and s.region.support == "partial") or (i >= 2 and s.plan.phase == "failed")
            btn.set_state(active, warning)
            btn.setEnabled(active or not reason)
            btn.setToolTip(reason)
            tone(btn, "active" if active else "warning" if warning else "")
