from pathlib import Path
from PySide6.QtCore import Qt, Signal, QSignalBlocker
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QComboBox, QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout
from widgets.status_panel import text_label, tone
from mock.mock_capture import SCENARIOS


class PatientHeader(QFrame):
    scenario_requested = Signal(str)
    history_requested = Signal()
    settings_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("patientHeader")
        self.setFixedHeight(88)
        row = QHBoxLayout(self)
        row.setContentsMargins(18, 12, 18, 12)
        row.setSpacing(12)
        logo = QLabel()
        logo.setObjectName("brandLogo")
        logo.setFixedSize(148, 46)
        asset = Path(__file__).resolve().parent / "assets" / "uziyuan_logo.png"
        if not asset.exists():
            asset = Path(__file__).resolve().parent.parent / "assets" / "uziyuan_logo.png"
        pixmap = QPixmap(str(asset))
        if not pixmap.isNull():
            logo.setPixmap(pixmap.scaled(142, 40, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            logo.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        row.addWidget(logo)
        separator = QFrame()
        separator.setObjectName("headerSeparator")
        separator.setFrameShape(QFrame.VLine)
        row.addWidget(separator)
        brand = QVBoxLayout()
        brand.setSpacing(4)
        brand.addWidget(text_label("三维采集与轨迹规划工作台", "brand"))
        self.reference_status = text_label("参考位置未建立", "referenceStatus")
        brand.addWidget(self.reference_status)
        row.addLayout(brand, 1)
        # Keep task metadata available on hover rather than adding a second
        # title-sized information block to the header.
        self.patient = text_label("", "patientName")
        self.task = text_label("", "muted")
        self.patient.hide(); self.task.hide()
        demo = QVBoxLayout()
        demo.setSpacing(4)
        demo.addWidget(text_label("研发场景模拟", "muted"))
        self.scenario = QComboBox()
        self.scenario.setObjectName("scenario")
        for key, name in SCENARIOS.items():
            self.scenario.addItem(name, key)
        self.scenario.setFixedSize(184, 42)
        self.scenario.currentIndexChanged.connect(lambda: self.scenario_requested.emit(self.scenario.currentData()))
        demo.addWidget(self.scenario)
        row.addLayout(demo)
        for text, signal in (("历史记录", self.history_requested), ("高级工具", self.settings_requested)):
            button = QPushButton(text)
            button.setObjectName("quietButton")
            button.setFixedHeight(42)
            button.clicked.connect(signal.emit)
            row.addWidget(button)

    def update_state(self, s):
        if s.replay_mode:
            self.patient.setText(f"项目：{s.replay_label or '已保存回放'}")
            self.task.setText("本地路径：项目数据   ·   只读回放")
        else:
            self.patient.setText(f"项目：{s.patient.identifier}")
            self.task.setText(f"本地路径：prototype/demo_records   ·   {s.patient.task}")
        with QSignalBlocker(self.scenario):
            if s.replay_mode and self.scenario.findData("replay") < 0:
                self.scenario.addItem("项目回放 · 只读", "replay")
            self.scenario.setCurrentIndex(self.scenario.findData("replay" if s.replay_mode else s.scenario))
            if not s.replay_mode and self.scenario.findData("replay") >= 0:
                self.scenario.removeItem(self.scenario.findData("replay"))
        self.scenario.setEnabled(not s.replay_mode)
        established = s.capture.reference_ready
        self.reference_status.setText("●  项目回放 · 参考位置已记录" if s.replay_mode else
                                      "●  参考位置已建立 · 模拟采集" if established else "○  参考位置未建立 · 原型演示")
        tone(self.reference_status, "good" if established else "warning")
        self.reference_status.setToolTip(self.patient.text()+"\n"+self.task.text())
