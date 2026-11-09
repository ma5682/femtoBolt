"""Image-only selection before reconstruction; no placeholder mesh or XYZ."""
from PySide6.QtCore import Signal, Qt, QSize
from PySide6.QtGui import QShortcut, QKeySequence
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSpinBox, QSlider, QFrame

from femto.app.qt_planning_canvas import PlanningCanvas
from femto.geometry.local_region import DEFAULT_MARGIN_MM


class CaptureRegionEditor(QWidget):
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('captureRegionEditor')
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.reference = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(12)
        heading = QHBoxLayout(); heading.setSpacing(14)
        accent = QFrame(); accent.setObjectName('regionHeadingAccent'); accent.setFixedSize(5,24)
        heading.addWidget(accent)
        self.title = QLabel('涂抹选区'); self.title.setObjectName('regionHeading')
        heading.addWidget(self.title); heading.addWidget(self._separator())
        self.frame_count = QLabel(); self.frame_count.setObjectName('regionFrameCount')
        heading.addWidget(self.frame_count); heading.addStretch()
        layout.addLayout(heading)
        self.note = QLabel('在图像上涂抹需要规划的区域。')
        self.note.setObjectName('regionInstruction')
        self.note.setTextFormat(Qt.PlainText)
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        toolbar = QFrame(); toolbar.setObjectName('regionToolbar')
        row = QHBoxLayout(toolbar); row.setContentsMargins(0,8,0,8); row.setSpacing(10)
        self.paint = self._button('涂抹','brush')
        self.erase = self._button('擦除','eraser')
        for button in (self.paint,self.erase):
            button.setCheckable(True)
            row.addWidget(button)
        row.addWidget(self._separator())
        self.canvas = PlanningCanvas()
        self.canvas.brush_shape = 'round'
        self.canvas.work_color = (0,210,195,60)
        self.canvas.selection_outline = True
        self.canvas.background_color = '#0D171F'
        self.canvas.show_zoom = False
        self.paint.clicked.connect(lambda: self.set_erase(False))
        self.erase.clicked.connect(lambda: self.set_erase(True))
        self.brush_label = QLabel('笔刷大小'); row.addWidget(self.brush_label)
        self.brush_slider = QSlider(Qt.Horizontal); self.brush_slider.setObjectName('regionBrushSlider')
        self.brush_slider.setRange(4,200); self.brush_slider.setValue(40)
        self.brush_slider.setMinimumWidth(90); self.brush_slider.setMaximumWidth(180)
        self.brush_slider.setAccessibleName('笔刷大小')
        row.addWidget(self.brush_slider,1)
        self.brush = QSpinBox();self.brush.setRange(4,200);self.brush.setValue(40)
        self.brush.setObjectName('regionBrushValue'); self.brush.setFixedWidth(52)
        self.brush.setButtonSymbols(QSpinBox.NoButtons); self.brush.setAlignment(Qt.AlignCenter)
        self.brush.setAccessibleName('笔刷直径（屏幕像素）')
        self.brush.setToolTip('笔刷直径，单位：屏幕像素')
        self.brush.valueChanged.connect(self._brush_changed)
        self.brush_slider.valueChanged.connect(self.brush.setValue)
        row.addWidget(self.brush)
        row.addWidget(self._separator())
        self.undo = self._button('撤销','undo'); self.redo = self._button('重做','redo')
        self.clear = self._button('清空','trash'); self.reset = self._button('复位视图','fit')
        for button, action, shortcut in ((self.undo,self.canvas.undo,'Ctrl+Z'),
                                         (self.redo,self.canvas.redo,'Ctrl+Y'),
                                         (self.clear,self.canvas.clear_work,None)):
            button.clicked.connect(action)
            if shortcut:
                key = QShortcut(QKeySequence(shortcut), self)
                key.setContext(Qt.WidgetWithChildrenShortcut); key.activated.connect(action)
                button.setToolTip(shortcut)
            row.addWidget(button)
        row.addStretch()
        self.zoom_label = QLabel('100%'); self.zoom_label.setMinimumWidth(46)
        self.zoom_label.setAlignment(Qt.AlignRight|Qt.AlignVCenter)
        row.addWidget(self.zoom_label); row.addWidget(self._separator())
        self.reset.clicked.connect(self.canvas.reset_view); row.addWidget(self.reset)
        self.canvas.zoom_changed.connect(lambda zoom:self.zoom_label.setText(f'{zoom*100:.0f}%'))
        self.advanced_toggle = QPushButton('失败详情')
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.hide()
        row.addWidget(self.advanced_toggle)
        layout.addWidget(toolbar)
        self.advanced = QWidget()
        options = QHBoxLayout(self.advanced)
        options.setContentsMargins(0,0,0,0)
        # The physician chooses a target, not an engineering support band.
        # Keep the internal value for the existing capture worker contract.
        self.margin = QSpinBox(self.advanced);self.margin.setRange(0,0);self.margin.setValue(int(DEFAULT_MARGIN_MM))
        self.margin.hide()
        self.error_details=QLabel();self.error_details.setWordWrap(True)
        self.error_details.setTextFormat(Qt.PlainText)
        self.error_details.setTextInteractionFlags(Qt.TextSelectableByMouse)
        options.addWidget(self.error_details,1)
        self.advanced.hide()
        self.advanced_toggle.toggled.connect(self.advanced.setVisible)
        self.margin.valueChanged.connect(self.changed)
        layout.addWidget(self.advanced)
        layout.addWidget(self.canvas,1)
        self.canvas.changed.connect(self.changed)
        self.canvas.changed.connect(self._update_actions)
        self.set_erase(False)
        self._update_actions()

    @staticmethod
    def _separator():
        line = QFrame(); line.setObjectName('regionToolSeparator'); line.setFixedSize(1,24)
        return line

    def _button(self, text, icon):
        from femto.app.qt_icons import line_icon
        button = QPushButton(text, self)
        button.setIcon(line_icon(icon, foreground='#EDF4F4')); button.setIconSize(QSize(20,20))
        button.setMinimumHeight(40)
        return button

    def _brush_changed(self, value):
        self.canvas.brush_size = value
        self.brush_slider.setValue(value)
        self.canvas.update()

    def _update_actions(self):
        self.undo.setEnabled(bool(self.canvas._undo))
        self.redo.setEnabled(bool(self.canvas._redo))
        self.clear.setEnabled(self.canvas.work is not None and bool(self.canvas.work.any()))

    def set_erase(self, erase):
        self.canvas.finish_stroke()
        self.canvas.erase = erase
        self.paint.setChecked(not erase);self.erase.setChecked(erase)
        self.canvas.update()

    def install(self, reference, error=''):
        # A failed reconstruction retains exact strokes, undo history and zoom.
        same = self.reference is not None and self.reference.binding == reference.binding
        self.reference = reference
        if not same:
            self.canvas.set_model(reference)
        self.frame_count.setText(f'已保存 {reference.frame_count} 帧')
        self.error_details.setText(error)
        self.error_details.setVisible(bool(error))
        self.advanced_toggle.setVisible(bool(error))
        self.advanced_toggle.setChecked(bool(error))
        self.advanced.setVisible(bool(error))
        self.note.setText('局部重建未完成。可检查选区后重试；详细原因见“失败详情”。' if error else
            '在图像上涂抹需要规划的区域。')
        self.setEnabled(True)
        self._update_actions()
