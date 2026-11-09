"""Shared workflow navigation; destinations are supplied by the owning window."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal, QRectF, QPointF
from PySide6.QtGui import QPainter, QPen, QColor, QRadialGradient
from PySide6.QtWidgets import QFrame, QHBoxLayout, QPushButton, QSizePolicy


STEPS = ('局部采集', '检查模型', '核对选区', '生成路线', '核对保存', '标记/重建')
# Stable semantic IDs retain historical navigation contracts; displayed order
# follows the new acquisition -> image selection -> local model workflow.
STEP_ORDER = (1, 6, 2, 3, 4, 5)
# Internal semantic destinations remain stable for saved-view navigation.
# Only four groups are presented to the operator.
STAGE_GROUPS = ((1,), (6,), (2, 3, 4), (5,))
STAGE_TITLES = ('采集', '涂抹选区', '模型与路线', '保存')
VISIBLE_STEPS = (1, 6, 2, 5)
HINTS = (
    '接纳一帧后即可提前完成；最多 3 个有效视角，第三帧保存后自动进入固定参考图。',
    '查看缺失与待检查范围，再核对三维模型中的目标和边界。',
    '涂抹选入工作区域，擦除排除；扫描范围不等于工作区域。',
    '核对参数并计算路线，参数或区域变化后重新计算。',
    '查看待处理项，明确核对后保存；历史修改另存新版本。',
    '在固定参考 RGB 图上涂抹，只重建最终选中的实测表面。',
)


class StepButton(QPushButton):
    """Paint the current position without implying earlier quality approval."""
    def __init__(self, index, title):
        position = next(i+1 for i, group in enumerate(STAGE_GROUPS) if index in group)
        super().__init__(f'{position:02d}  {title}')
        self.index, self.title = position, title
        self.setFixedHeight(54)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        current = self.property('state') == 'current'
        color = QColor('#20d7ff' if current else '#6193af' if self.isEnabled() else '#43667e')
        cy = self.height()/2
        x = 25.
        if current:
            glow = QRadialGradient(QPointF(x, cy), 27)
            glow.setColorAt(0, QColor('#087daa'))
            glow.setColorAt(1, QColor(0, 150, 230, 0))
            painter.setPen(Qt.NoPen); painter.setBrush(glow)
            painter.drawEllipse(QPointF(x, cy), 27, 27)
        painter.setPen(QPen(color, 1.3)); painter.setBrush(QColor('#0a2d40' if current else '#0b2132'))
        painter.drawEllipse(QPointF(x, cy), 17, 17)
        font = painter.font(); font.setPixelSize(16); font.setBold(current); painter.setFont(font)
        painter.setPen(QColor('#effaff' if current else '#87afc7'))
        painter.drawText(QRectF(x-17, cy-17, 34, 34), Qt.AlignCenter, f'{self.index:02d}')
        font.setPixelSize(14); painter.setFont(font)
        painter.drawText(QRectF(51, 0, 80, self.height()), Qt.AlignVCenter, self.title)
        if self.index < len(STAGE_GROUPS) and self.width() > 147:
            painter.setPen(QPen(color, 1.2))
            painter.drawLine(QPointF(138, cy), QPointF(self.width()-6, cy))
        if self.hasFocus() and self.isEnabled():
            painter.setBrush(Qt.NoBrush);painter.setPen(QPen(QColor('#20d7ff'), 1, Qt.DotLine))
            painter.drawRoundedRect(QRectF(self.rect()).adjusted(1,1,-1,-1), 6,6)


class WorkflowProgress(QFrame):
    """Offer explicit destinations without inferring that earlier checks passed."""
    step_requested = Signal(int)
    refreshed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('workflowProgress')
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 8, 16, 8)
        layout.setSpacing(6)
        self.buttons = []
        self.labels = self.buttons
        self.current_step = None
        self._note = ''
        self._available = frozenset()
        self._reasons = {}
        self._state = None
        for index, title in enumerate(STEPS, start=1):
            button = StepButton(index, title)
            button.setParent(self)
            button.setObjectName('workflowPosition')
            button.setMinimumWidth(0)
            button.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
            button.setFocusPolicy(Qt.StrongFocus)
            button.clicked.connect(lambda checked=False, number=index: self.step_requested.emit(number))
            self.buttons.append(button)
            button.hide()
        self.stage_buttons = []
        for position,number in enumerate(VISIBLE_STEPS,1):
            button = StepButton(number, STAGE_TITLES[position-1])
            button.setObjectName('workflowPosition')
            button.setSizePolicy(QSizePolicy.Ignored,QSizePolicy.Fixed)
            button.clicked.connect(lambda checked=False, number=number:
                self.step_requested.emit(self._destination(number)))
            self.stage_buttons.append(button)
            layout.addWidget(button,1)
        self.set_step(None)

    def _destination(self, number):
        if number not in VISIBLE_STEPS:
            return number
        group = STAGE_GROUPS[VISIBLE_STEPS.index(number)]
        return next((step for step in reversed(group) if step in self._available), number)

    def set_step(self, index: int | None, note: str = '') -> None:
        self.current_step = index
        self._note = note
        self._render()
        self.refreshed.emit()

    def set_navigation(self, available, reasons=None) -> None:
        self._available = frozenset(available)
        self._reasons = dict(reasons or {})
        self._render()

    def _render(self):
        state = (self.current_step, self._note, self._available, tuple(sorted(self._reasons.items())))
        if self._state == state:
            return
        self._state = state
        self.setAccessibleDescription('可点击可用步骤切换；流程位置不代表质量通过。' + self._note)
        for number, button in enumerate(self.buttons, start=1):
            active = number == self.current_step
            enabled = number in self._available and not active
            button.setEnabled(enabled)
            button.setCursor(Qt.PointingHandCursor if enabled else Qt.ArrowCursor)
            button.setProperty('state', 'current' if active else 'available' if enabled else 'idle')
            status = '当前步骤' if active else '可点击切换' if enabled else self._reasons.get(number, '此步骤尚不可切换')
            button.setAccessibleName(f'第 {STEP_ORDER.index(number)+1} 步：{STEPS[number-1]}，{status}')
            button.setToolTip(HINTS[number-1] + '\n' + (self._note if active and self._note else status))
            button.style().unpolish(button)
            button.style().polish(button)
        for position, (number, group) in enumerate(zip(VISIBLE_STEPS, STAGE_GROUPS), 1):
            button = self.stage_buttons[position-1]
            active = self.current_step in group
            target = self._destination(number)
            enabled = not active and target in self._available
            button.setEnabled(enabled)
            button.setProperty('state','current' if active else 'available' if enabled else 'idle')
            button.setCursor(Qt.PointingHandCursor if enabled else Qt.ArrowCursor)
            status = '当前阶段' if active else '可点击切换' if enabled else self._reasons.get(target,'此阶段尚不可切换')
            button.setAccessibleName(f'第 {position} 步：{button.title}，{status}')
            button.setToolTip((self._note if active and self._note else HINTS[target-1])+'\n'+status)
            button.style().unpolish(button)
            button.style().polish(button)
