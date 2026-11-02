from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QVBoxLayout, QPushButton, QToolButton, QMenu
from widgets.status_panel import text_label, tone


class ActionBar(QFrame):
    action_requested = Signal(str)
    details_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("actionBar")
        self.setFixedHeight(80)
        row = QHBoxLayout(self)
        row.setContentsMargins(16, 7, 16, 7)
        row.setSpacing(10)
        self.notice = text_label("", "actionNotice", True)
        self.notice.setMinimumWidth(230)
        copy = QVBoxLayout()
        copy.setSpacing(3)
        copy.addWidget(self.notice)
        self.telemetry = text_label("", "statusTelemetry")
        copy.addWidget(self.telemetry)
        row.addLayout(copy, 1)
        self.details = QToolButton()
        self.details.setObjectName("detailsButton")
        self.details.setText("状态详情")
        self.details.setFixedHeight(42)
        self.details.clicked.connect(self.details_requested.emit)
        row.addWidget(self.details)
        self.more = QToolButton()
        self.more.setObjectName("moreButton")
        self.more.setText("更多")
        self.more.setFixedHeight(42)
        self.more.setPopupMode(QToolButton.InstantPopup)
        self.more_menu = QMenu(self.more)
        self._menu_signature = None
        self.more.setMenu(self.more_menu)
        row.addWidget(self.more)
        self.secondary = []
        for _ in range(2):
            btn = QPushButton()
            btn.setObjectName("secondaryButton")
            btn.setFixedHeight(42)
            btn.clicked.connect(lambda _=False, b=btn: self.action_requested.emit(b.property("action")))
            btn.hide()
            self.secondary.append(btn)
        self.primary = QPushButton()
        self.primary.setObjectName("primaryButton")
        self.primary.setProperty("role", "primary")
        self.primary.setFixedSize(290, 52)
        self.primary.clicked.connect(lambda: self.action_requested.emit(self.primary.property("action")))
        row.addWidget(self.primary)

    def set_telemetry(self, text):
        self.telemetry.setText(text)

    def update_state(self, primary, secondary, notice, severity, telemetry=""):
        status = ("●  " if severity == "normal" else "！  ") + notice
        self.notice.setText(status)
        tone(self.notice, severity)
        if telemetry:
            self.telemetry.setText(telemetry)
        for btn, action in [(self.primary, primary), *zip(self.secondary, secondary)]:
            btn.setText(action.text)
            btn.setProperty("action", action.key)
            btn.setEnabled(action.enabled)
            btn.setToolTip(action.reason if not action.enabled else "")
        signature = tuple((a.key, a.text, a.enabled, a.reason) for a in secondary)
        # Capture refreshes every 650 ms; keep an open menu usable across ticks.
        if signature != self._menu_signature:
            self.more_menu.clear()
            for action in secondary:
                item = self.more_menu.addAction(action.text)
                item.setEnabled(action.enabled)
                item.setToolTip(action.reason)
                item.triggered.connect(lambda _=False, key=action.key: self.action_requested.emit(key))
            self._menu_signature = signature
        self.more.setEnabled(bool(secondary))
