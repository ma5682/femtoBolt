"""Minimal read-only preview while preparing the planning destination."""
from __future__ import annotations

from PySide6.QtCore import Qt, QRectF, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import (
    QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton,
    QSizePolicy, QStackedWidget, QVBoxLayout, QWidget,
)


STAGES = ('读取采集数据', '准备三维模型', '检查选区与缺口')
STAGE_MESSAGES = tuple('[planning-load:%d]' % index for index in range(len(STAGES)))


class PreparationSpinner(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(34, 34)
        self.angle = 0
        self.timer = QTimer(self)
        self.timer.setInterval(40)
        self.timer.timeout.connect(self.advance)

    def advance(self):
        self.angle = (self.angle + 12) % 360
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = QRectF(self.rect()).adjusted(4, 4, -4, -4)
        painter.setPen(QPen(QColor('#234a5d'), 3))
        painter.drawEllipse(rect)
        painter.setPen(QPen(QColor('#10c9e8'), 3, Qt.SolidLine, Qt.RoundCap))
        painter.drawArc(rect, -self.angle * 16, 100 * 16)


class PreparationReference(QWidget):
    """Display a detached image only; never expose a previous model's canvas."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = QImage()
        self.empty_text = '参考图加载中…'
        self.setMinimumSize(400, 260)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAccessibleName('只读选区参考图')

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor('#06111a'))
        if self.image.isNull():
            painter.setPen(QColor('#93adbe'))
            painter.drawText(self.rect(), Qt.AlignCenter, self.empty_text)
            return
        # Keep the legend aligned with the image, including at narrow sizes.
        available = self.size()
        available.setHeight(max(1, available.height()-32))
        size = self.image.size().scaled(available, Qt.KeepAspectRatio)
        rect = QRectF((self.width()-size.width())/2, (available.height()-size.height())/2,
                      size.width(), size.height())
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.drawImage(rect, self.image)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(QColor('#00d4a3'), 1.5))
        painter.drawRoundedRect(QRectF(rect.left(), rect.bottom()+12, 28, 14), 3, 3)
        painter.setPen(QColor('#d6e5ee'))
        painter.drawText(QRectF(rect.left()+40, rect.bottom()+7, 100, 24), Qt.AlignVCenter, '标记区域')


class PlanningPreparation(QWidget):
    cancel_requested = Signal()
    retry_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.stage = 0
        self.state = 'idle'
        self.last_progress = ''
        body = QVBoxLayout(self)
        body.setContentsMargins(24, 8, 24, 12)
        body.setSpacing(10)
        self.banner = QFrame()
        self.banner.setObjectName('planningPreparationBanner')
        banner = QGridLayout(self.banner)
        banner.setContentsMargins(0, 4, 0, 12)
        banner.setHorizontalSpacing(16)
        for column in range(3):banner.setColumnStretch(column, 1)
        self.reference_label = QLabel('采集图像 · 区域标记预览')
        self.reference_label.setObjectName('preparationReferenceHeading')
        banner.addWidget(self.reference_label, 0, 0, Qt.AlignLeft)
        status = QWidget()
        status_layout = QHBoxLayout(status)
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setSpacing(12)
        self.spinner = PreparationSpinner()
        self.title = QLabel()
        self.title.setObjectName('preparationTitle')
        self.title.setTextFormat(Qt.PlainText)
        status_layout.addWidget(self.spinner)
        status_layout.addWidget(self.title)
        banner.addWidget(status, 0, 1, Qt.AlignCenter)
        actions = QWidget()
        buttons = QHBoxLayout(actions)
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.setSpacing(14)
        self.readonly_label = QLabel('只读')
        self.readonly_label.setProperty('role', 'secondary')
        buttons.addWidget(self.readonly_label)
        self.retry = QPushButton('重试')
        self.retry.clicked.connect(self.retry_requested)
        buttons.addWidget(self.retry)
        self.cancel = QPushButton('取消')
        self.cancel.setMinimumWidth(80)
        self.cancel.clicked.connect(self.cancel_requested)
        buttons.addWidget(self.cancel)
        banner.addWidget(actions, 0, 2, Qt.AlignRight)
        body.addWidget(self.banner)
        self.reference = PreparationReference()
        self.views = QStackedWidget()
        self.views.addWidget(self.reference)
        self.viewport = None
        body.addWidget(self.views, 1)

    def start(self, preview=None):
        self.clear_model()
        self.state = 'running'
        self.stage = 0
        self.last_progress = ''
        self.reference.image = QImage() if preview is None else preview.copy()
        self.reference.empty_text = '参考图加载中…'
        self.reference.update()
        self.title.setText('正在准备三维模型…')
        self.reference_label.setText('采集图像 · 区域标记预览')
        self.readonly_label.setText('只读')
        self.retry.hide()
        self.cancel.setText('取消')
        self.cancel.setEnabled(True)
        self.spinner.show()
        self.spinner.timer.start()

    def show_model(self, snapshot, assessment):
        """The source surface is visible while selection records finish below."""
        if self.state != 'running':return False
        try:
            if self.viewport is None:
                from femto.app.qt_scene_panel import create_viewport
                self.viewport = create_viewport(self)
                self.views.addWidget(self.viewport)
            self.viewport.set_snapshot(snapshot)
            self.views.setCurrentWidget(self.viewport)
            self.reference_label.setText('三维模型 · 选区检查中')
            self.title.setText('模型已显示，正在完成检查…')
            self.readonly_label.setText(f"一致 {assessment['ratio']:.1%} · 待检查 {assessment['pending']} 像素")
            self.readonly_label.setToolTip('按选区参考像素计算；选区检查完成后可生成路线。')
            return True
        except Exception:
            # Missing VTK/graphics support must not fail geometry preparation.
            self.clear_model()
            return False

    def clear_model(self):
        self.views.setCurrentWidget(self.reference)
        self.reference_label.setText('采集图像 · 区域标记预览')
        self.readonly_label.setText('只读')
        self.readonly_label.setToolTip('')
        if self.viewport is not None:
            self.viewport.shutdown()
            self.views.removeWidget(self.viewport)
            self.viewport.deleteLater()
            self.viewport = None

    def progress(self, message):
        # Retain worker phases for diagnostics without adding clinical UI text.
        if self.state != 'running':return
        if message in STAGE_MESSAGES:
            self.stage = max(self.stage, STAGE_MESSAGES.index(message))
        else:
            self.last_progress = str(message)

    def cancelling(self):
        self.state = 'cancelling'
        self.title.setText('正在取消…')
        self.cancel.setEnabled(False)
        self.retry.hide()

    def failed(self, error):
        self.state = 'failed'
        self.last_progress = str(error)
        self.stop()
        self.spinner.hide()
        self.title.setText('模型准备失败，请重试')
        self.reference.empty_text = '暂无参考图'
        self.reference.update()
        self.cancel.setText('返回')
        self.cancel.setEnabled(True)
        self.retry.show()

    def stop(self):
        self.spinner.timer.stop()
        self.clear_model()
