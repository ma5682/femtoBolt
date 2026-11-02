from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFrame, QPushButton, QVBoxLayout
from widgets.status_panel import tone


class ToolPalette(QFrame):
    tool_requested = Signal(str)
    undo_requested = Signal()
    clear_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("toolPalette")
        col = QVBoxLayout(self)
        col.setContentsMargins(5, 5, 5, 5)
        col.setSpacing(4)
        self.buttons = {}
        for key, text in (("work", "规划区"), ("forbid", "禁入区"), ("erase", "擦除"), ("undo", "撤销"), ("clear", "清空标记")):
            btn = QPushButton(text)
            btn.setObjectName("toolButton")
            btn.setMinimumSize(96, 44)
            if key == "undo":
                btn.clicked.connect(self.undo_requested.emit)
            elif key == "clear":
                btn.clicked.connect(self.clear_requested.emit)
            else:
                btn.clicked.connect(lambda _=False, k=key: self.tool_requested.emit(k))
            col.addWidget(btn)
            self.buttons[key] = btn
        self.setFixedSize(112, 254)

    def update_state(self, s, can_undo):
        self.setVisible(s.stage == 1 and not s.replay_mode)
        self.setEnabled(not s.busy and not s.replay_mode)
        for key, button in self.buttons.items():
            tone(button, "active" if key == s.tool else "")
        self.buttons["undo"].setEnabled(can_undo and not s.busy)
        self.buttons["clear"].setEnabled(bool(s.region.strokes) and not s.busy)
