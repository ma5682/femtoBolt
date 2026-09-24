from pathlib import Path
from PySide6.QtCore import Qt, Signal, QSignalBlocker
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QComboBox, QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout
from widgets.status_panel import text_label
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
        brand.setSpacing(1)
        brand.addWidget(text_label("三维采集与轨迹规划工作台", "brand"))
        brand.addWidget(text_label("医疗三维工作站", "brandCaption"))
        row.addLayout(brand)
        patient = QVBoxLayout()
        patient.setSpacing(4)
        self.patient = text_label("", "patientName")
        self.task = text_label("", "muted")
        patient.addWidget(self.patient)
        patient.addWidget(self.task)
        row.addLayout(patient, 1)
        device = QVBoxLayout()
        device.setSpacing(4)
        self.device_status = text_label("●  相机已连接", "deviceReady")
        self.sensor_status = text_label("RGB 正常  ·  深度正常  /  模拟", "muted")
        device.addWidget(self.device_status)
        device.addWidget(self.sensor_status)
        self.source = text_label("Mock 演示", "modeBadge")
        device.addWidget(self.source)
        # Connection/quality telemetry is shown once in the bottom status bar;
        # repeating it in the title bar makes the header feel like a dashboard.
        device_widget = QFrame()
        device_widget.setLayout(device)
        device_widget.setVisible(False)
        row.addWidget(device_widget)
        demo = QVBoxLayout()
        demo.setSpacing(4)
        demo.addWidget(text_label("研发场景模拟", "muted"))
        self.scenario = QComboBox()
        self.scenario.setObjectName("scenario")
        for key, name in SCENARIOS.items():
            self.scenario.addItem(name, key)
        self.scenario.addItem("项目回放 · 只读", "replay")
        self.scenario.setMinimumWidth(124)
        self.scenario.currentIndexChanged.connect(lambda: self.scenario_requested.emit(self.scenario.currentData()))
        demo.addWidget(self.scenario)
        row.addLayout(demo)
        for text, signal in (("历史", self.history_requested), ("设置", self.settings_requested)):
            button = QPushButton(text)
            button.setObjectName("quietButton")
            button.setMinimumSize(52, 40)
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
            self.scenario.setCurrentIndex(self.scenario.findData("replay" if s.replay_mode else s.scenario))
        self.scenario.setEnabled(not s.replay_mode)
        self.source.setText("项目回放 · 只读" if s.replay_mode else "Mock 演示")
        self.device_status.setText("●  已保存数据可读" if s.replay_mode else "●  相机已连接")
        self.sensor_status.setText("RGB / 深度 / 网格 · 只读" if s.replay_mode else "RGB 正常  ·  深度正常  /  模拟")
