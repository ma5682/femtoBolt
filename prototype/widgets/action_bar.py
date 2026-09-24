from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QPushButton
from widgets.status_panel import text_label, tone


class ActionBar(QFrame):
    action_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("actionBar")
        row = QHBoxLayout(self)
        row.setContentsMargins(16, 7, 16, 7)
        row.setSpacing(10)
        self.notice = text_label("", "actionNotice", True)
        self.notice.setMinimumWidth(230)
        row.addWidget(self.notice)
        self.telemetry = text_label("", "statusTelemetry")
        row.addWidget(self.telemetry, 1)
        self.secondary = []
        for _ in range(2):
            btn = QPushButton()
            btn.setObjectName("secondaryButton")
            btn.setMinimumHeight(46)
            btn.clicked.connect(lambda _=False, b=btn: self.action_requested.emit(b.property("action")))
            row.addWidget(btn)
            self.secondary.append(btn)
        self.primary = QPushButton()
        self.primary.setObjectName("primaryButton")
        self.primary.setMinimumSize(200, 48)
        self.primary.clicked.connect(lambda: self.action_requested.emit(self.primary.property("action")))
        row.addWidget(self.primary)

    def set_telemetry(self, text):
        self.telemetry.setText(text)

    def update_state(self, primary, secondary, notice, severity, telemetry=""):
        status = "▱  项目已加载" if severity == "normal" else ("！  " + notice)
        self.notice.setText(status)
        tone(self.notice, severity)
        if telemetry:
            self.telemetry.setText(telemetry)
        for btn, action in [(self.primary, primary), *zip(self.secondary, secondary)]:
            btn.setText(action.text)
            btn.setProperty("action", action.key)
            btn.setEnabled(action.enabled)
            btn.setToolTip(action.reason if not action.enabled else "")
        for index, btn in enumerate(self.secondary):
            btn.setVisible(index < len(secondary))
