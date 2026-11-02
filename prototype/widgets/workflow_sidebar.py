from PySide6.QtCore import Qt, Signal, QRectF, QPointF
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QFrame, QHBoxLayout, QPushButton, QScrollArea, QVBoxLayout, QWidget

from app.navigation import navigation_reason
from widgets.status_panel import text_label, tone


def _font(size, bold=False):
    font = QFont('Microsoft YaHei')
    font.setPixelSize(size)
    font.setWeight(QFont.DemiBold if bold else QFont.Normal)
    return font


class WorkflowStep(QPushButton):
    """A milestone; clicking only requests navigation within the existing flow."""

    def __init__(self, index, title, subtitle, parent=None):
        super().__init__(parent)
        self.index, self.title, self.subtitle = index, title, subtitle
        self.active = self.warning = self.completed = self.blocked = False
        self.setObjectName('workflowStep')
        self.setFlat(True)
        self.setFixedHeight(80)
        self.setCursor(Qt.PointingHandCursor)
        self.setAccessibleName(f'{index + 1:02d} {title}')

    def set_state(self, active, warning=False, completed=False, blocked=False):
        self.active, self.warning = bool(active), bool(warning)
        self.completed, self.blocked = bool(completed), bool(blocked)
        self.setAccessibleDescription('需处理' if blocked else '当前' if active else '已完成' if completed else '待处理' if warning else '未执行')
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        rect = QRectF(self.rect()).adjusted(.5, .5, -.5, -.5)
        if self.active:
            p.setBrush(QColor('#103D57'))
            p.setPen(QPen(QColor('#087FA8'), 1))
            p.drawRoundedRect(rect, 5, 5)
        elif self.underMouse() and self.isEnabled():
            p.setBrush(QColor('#142A3D'))
            p.setPen(Qt.NoPen)
            p.drawRoundedRect(rect, 5, 5)
        if self.hasFocus():
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(QColor('#91A9BB'), 1, Qt.DotLine))
            p.drawRoundedRect(rect.adjusted(2, 2, -2, -2), 4, 4)
        p.setPen(QPen(QColor('#355368'), 1))
        if self.index:
            p.drawLine(QPointF(24, 0), QPointF(24, 10))
        if self.index < 5:
            p.drawLine(QPointF(24, 44), QPointF(24, 80))
        edge, fill, ink = '#4D687C', '#172B3E', '#91A9BB'
        marker = f'{self.index + 1:02d}'
        if self.completed:
            edge, fill, ink, marker = '#22D3A6', '#10352F', '#22D3A6', '✓'
        if self.active:
            edge, fill, ink, marker = '#13D7DC', '#13D7DC', '#06212A', f'{self.index + 1:02d}'
        if self.blocked:
            edge, fill, ink, marker = '#F06464', '#392734', '#F06464', '!'
        elif self.warning:
            edge, fill, ink, marker = '#F2B84B', '#3A3426', '#F2B84B', '!'
        p.setPen(QPen(QColor(edge), 1.25))
        p.setBrush(QColor(fill))
        p.drawEllipse(QPointF(24, 27), 16, 16)
        p.setPen(QColor(ink))
        p.setFont(_font(15, True))
        if marker == '✓':
            p.setPen(QPen(QColor(ink), 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            p.drawLine(QPointF(17, 27), QPointF(22, 32))
            p.drawLine(QPointF(22, 32), QPointF(31, 22))
        else:
            p.drawText(QRectF(8, 11, 32, 32), Qt.AlignCenter, marker)
        p.setFont(_font(17, self.active or self.completed))
        p.setPen(QColor('#EAF2F8' if self.active or self.completed else '#91A9BB'))
        p.drawText(QPointF(51, 29), self.title)
        p.setFont(_font(13))
        p.setPen(QColor('#91A9BB' if self.active else '#637B8D'))
        p.drawText(QPointF(51, 53), self.subtitle)
        p.end()


class WorkflowSidebar(QFrame):
    step_requested = Signal(int)
    STAGE_MAP = (0, 0, 0, 1, 2, 3)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('workflowSidebar')
        self.setFixedWidth(236)
        col = QVBoxLayout(self)
        col.setContentsMargins(10, 16, 10, 12)
        col.setSpacing(6)
        col.addWidget(text_label('工作流程', 'sidebarHeading'))
        self.scroll = QScrollArea()
        self.scroll.setObjectName('workflowScroll')
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content = QWidget()
        content.setObjectName('workflowSteps')
        steps = QVBoxLayout(content)
        steps.setContentsMargins(0, 0, 0, 0)
        steps.setSpacing(0)
        self.buttons = []
        self.subtitles = ('设置相机与目标参考', '多角度补充采集', '确认三维表面质量', '圈定目标与避让区域', '计算并核对扫描路线', '核对参数与保存方案')
        titles = ('建立参考位置', '移动补采', '检查模型', '区域标记', '轨迹规划', '规划确认')
        for index, title in enumerate(titles):
            button = WorkflowStep(index, title, self.subtitles[index], content)
            button.setAttribute(Qt.WA_AlwaysShowToolTips)
            button.clicked.connect(lambda _=False, stage=self.STAGE_MAP[index]: self.step_requested.emit(stage))
            steps.addWidget(button)
            self.buttons.append(button)
        steps.addStretch(1)
        self.scroll.setWidget(content)
        col.addWidget(self.scroll, 1)
        self._visible_milestone = None
        device = QFrame()
        device.setObjectName('devicePanel')
        device_col = QVBoxLayout(device)
        device_col.setContentsMargins(10, 10, 10, 8)
        device_col.setSpacing(0)
        head = QHBoxLayout()
        head.addWidget(text_label('设备状态', 'deviceTitle'))
        self.device_status = text_label('● 未连接', 'deviceRowValue')
        head.addWidget(self.device_status, 1, Qt.AlignRight)
        device_col.addLayout(head)
        self.device_values = {}
        for key, icon, name in (('camera', '▣', '相机'), ('rgb', '▧', 'RGB 图像'), ('depth', '◇', '深度图像'), ('reference', '⊙', '参考位置')):
            row = QWidget()
            row.setMinimumHeight(30)
            layout = QHBoxLayout(row)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(text_label(name, 'deviceRowName'))
            value = text_label('—', 'deviceRowValue')
            layout.addWidget(value, 1, Qt.AlignRight)
            device_col.addWidget(row)
            self.device_values[key] = value
        col.addWidget(device)
        self.mode_badge = text_label('研发模拟 · 不连接设备', 'prototypeBadge')
        self.mode_note = text_label('仅供交互预览，不用于治疗执行', 'muted', True)
        self.mode_badge.hide()
        self.mode_note.hide()
        device.setToolTip("原型演示：设备与采集状态为模拟；仅供交互预览。")

    @staticmethod
    def current_milestone(s):
        if s.stage > 0:
            return s.stage + 2
        return 0 if s.capture.phase == 'idle' else 1 if s.capture.phase == 'capturing' else 2

    def update_state(self, s):
        current = self.current_milestone(s)
        completed = (s.capture.reference_ready, s.capture.phase in ('checking', 'passed', 'failed'), s.capture.phase == 'passed', s.region.support == 'valid', s.plan_current, bool(s.plan.saved_id))
        for i, btn in enumerate(self.buttons):
            stage = self.STAGE_MAP[i]
            reason = navigation_reason(s, stage)
            if stage == 0 and i > current and s.stage == 0:
                reason = '请按底部主操作完成当前采集步骤。'
            active = i == current
            blocked = (i == 2 and s.capture.phase == 'failed') or (i == 3 and s.region.support == 'missing') or (i == 4 and s.plan.phase == 'failed')
            warning = i == 3 and s.region.support == 'partial'
            btn.set_state(active, warning, completed[i], blocked)
            btn.setEnabled((active and not s.busy) or not reason)
            btn.setToolTip(reason or ('已完成' if completed[i] and not active else '当前步骤' if active else '查看此步骤'))
            tone(btn, 'bad' if blocked else 'warning' if warning else 'current' if active else 'good' if completed[i] else 'idle')
        if self._visible_milestone != current:
            self.scroll.ensureWidgetVisible(self.buttons[current], 0, 8)
            self._visible_milestone = current
        if s.replay_mode:
            self.mode_badge.setText('项目回放 · 只读')
            self.mode_note.setText('显示已保存数据，不连接相机')
            status, status_tone = '● 未连接', 'idle'
            data = {'camera': ('○ 未连接', 'idle'), 'rgb': ('● 回放数据', 'current'), 'depth': ('● 回放数据', 'current'), 'reference': ('● 已载入', 'current')}
        else:
            self.mode_badge.setText('研发模拟 · 不连接设备')
            self.mode_note.setText('仅供交互预览，不用于治疗执行')
            active = s.capture.phase != 'idle'
            status, status_tone = ('● 模拟中', 'current') if active else ('○ 未就绪', 'idle')
            data = {'camera': ('● 模拟' if active else '○ 未就绪', 'current' if active else 'idle'), 'rgb': ('● 模拟画面' if active else '○ 未获取', 'current' if active else 'idle'), 'depth': ('● 模拟画面' if active else '○ 未获取', 'current' if active else 'idle'), 'reference': ('● 模拟已建' if s.capture.reference_ready else '○ 未建立', 'good' if s.capture.reference_ready else 'idle')}
        self.device_status.setText(status)
        tone(self.device_status, status_tone)
        for key, (text, color) in data.items():
            self.device_values[key].setText(text)
            tone(self.device_values[key], color)
