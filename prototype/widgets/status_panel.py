from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget


def text_label(text="", name="", wrap=False):
    label = QLabel(text)
    label.setObjectName(name)
    label.setWordWrap(wrap)
    return label


def tone(widget, value):
    if widget.property("tone") != value:
        widget.setProperty("tone", value)
        widget.style().unpolish(widget)
        widget.style().polish(widget)


class StatusPanel(QWidget):
    focus_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("statusPanel")
        self.column = QVBoxLayout(self)
        self.column.setContentsMargins(14, 14, 14, 12)
        self.column.setSpacing(8)

    def section(self, title):
        icons = {"采集信息": "▣", "数据质量": "▥", "空间数据": "◇", "采集状态": "▣", "三维模型检查": "◇", "规划参数": "⌘", "规划预览": "□", "运行预览": "□"}
        label = text_label(f"{icons.get(title, '·')}  {title}", "sectionTitle")
        self.column.addWidget(label)
        return label

    def divider(self):
        line = QFrame()
        line.setObjectName("divider")
        line.setFixedHeight(1)
        self.column.addWidget(line)

    def row(self, title):
        item = QWidget()
        layout = QHBoxLayout(item)
        layout.setContentsMargins(0, 3, 0, 3)
        layout.setSpacing(8)
        layout.addWidget(text_label(title, "rowName"))
        value = text_label("—", "rowValue")
        value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(value, 1)
        self.column.addWidget(item)
        return value

    def note(self, text="", kind="muted"):
        label = text_label(text, kind, True)
        self.column.addWidget(label)
        return label

    def issue(self, text, key):
        button = QPushButton(text)
        button.setObjectName("issueButton")
        button.setMinimumHeight(44)
        button.clicked.connect(lambda: self.focus_requested.emit(key))
        self.column.addWidget(button)
        return button

    def finish(self):
        self.column.addStretch(1)

    def value(self, widget, text, color=""):
        widget.setText(text)
        tone(widget, color)
