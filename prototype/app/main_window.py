"""Application composition. Pages emit intents; the controller owns transitions."""
from pathlib import Path
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QImage, QKeySequence, QShortcut
from PySide6.QtWidgets import QCheckBox, QDockWidget, QFrame, QHBoxLayout, QListWidget, QMainWindow, QMessageBox, QPushButton, QScrollArea, QStackedWidget, QTextBrowser, QVBoxLayout, QWidget
from app.controller import WorkflowController
from app.navigation import STAGES, VIEWS, actions
from app.presentation import current_notice
from pages.capture_page import CapturePage
from pages.region_page import RegionPage
from pages.planning_page import PlanningPage
from pages.confirmation_page import ConfirmationPage
from widgets.patient_header import PatientHeader
from widgets.workflow_sidebar import WorkflowSidebar
from widgets.action_bar import ActionBar
from widgets.tool_palette import ToolPalette
from widgets.medical_viewport import create_viewport, scene_from_state
from widgets.status_panel import text_label, tone
from widgets.replay_viewport import ReplayViewport
from adapters.project_data import load_project_data
from adapters.replay_images import depth_heatmap, region_image


class MainWindow(QMainWindow):
    def __init__(self, records_dir=None, viewport_factory=create_viewport):
        super().__init__()
        self.setWindowTitle("Femto Bolt · 三维采集与轨迹规划工作台")
        self.resize(1600, 900)
        self.setMinimumSize(1280, 760)
        source_dir = Path(__file__).resolve().parent
        directory = records_dir or (source_dir.parent if source_dir.name == "app" else source_dir) / "demo_records"
        self.controller = WorkflowController(Path(directory), self)
        self.project_replay = None
        # In the modular app source_dir is <project>/prototype/app; in the
        # generated standalone it is <project>/prototype.
        project_root = source_dir.parents[1] if source_dir.name == "app" else source_dir.parent
        try:
            self.project_replay = load_project_data(project_root)
        except Exception as exc:
            self.replay_error = str(exc)
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        column = QVBoxLayout(root)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        self.header = PatientHeader()
        self.header.history_requested.connect(self.show_history)
        self.header.settings_requested.connect(self.show_settings)
        self.header.scenario_requested.connect(self.change_scenario)
        column.addWidget(self.header)
        self.alert = text_label("", "globalStatus")
        # Status is intentionally kept in the bottom action dock.  A second
        # full-width alert band made the clinical viewport feel compressed.
        self.alert.setVisible(False)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        self.sidebar = WorkflowSidebar()
        self.sidebar.step_requested.connect(self.controller.go)
        body.addWidget(self.sidebar)
        center = QWidget()
        self.center = center
        center.setObjectName("center")
        work = QVBoxLayout(center)
        self.work_layout = work
        work.setContentsMargins(0, 0, 0, 0)
        work.setSpacing(0)
        title_bar = QFrame()
        title_bar.setObjectName("titleBar")
        title_bar.setFixedHeight(94)
        title_row = QHBoxLayout(title_bar)
        title_row.setContentsMargins(16, 10, 16, 9)
        title_col = QVBoxLayout()
        title_col.setSpacing(3)
        self.page_title = text_label("", "pageTitle")
        self.page_subtitle = text_label("", "pageSubtitle")
        title_col.addWidget(self.page_title)
        title_col.addWidget(self.page_subtitle)
        title_row.addLayout(title_col, 1)
        self.reference = text_label("", "referenceStatus")
        self.reference.setVisible(False)
        title_row.addWidget(self.reference)
        work.addWidget(title_bar)

        view_bar = QFrame()
        view_bar.setObjectName("viewBar")
        view_bar.setFixedHeight(54)
        view_row = QHBoxLayout(view_bar)
        view_row.setContentsMargins(16, 4, 16, 4)
        view_row.setSpacing(4)
        self.view_labels = {"RGB": "▣  RGB", "深度": "≋  深度", "点云": "⠿  点云", "模型": "◇  模型"}
        self.view_buttons = []
        for _ in range(4):
            btn = QPushButton()
            btn.setObjectName("viewButton")
            btn.setMinimumSize(92, 44)
            btn.clicked.connect(lambda _=False, b=btn: self.controller.set_view(b.property("view_name")))
            view_row.addWidget(btn)
            self.view_buttons.append(btn)
        view_row.addStretch()
        self.fit_button = QPushButton("适配")
        self.zoom_button = QPushButton("⌕")
        self.one_button = QPushButton("1:1")
        self.full_button = QPushButton("□")
        for button, tip in (
            (self.fit_button, "适配窗口"),
            (self.zoom_button, "缩放"),
            (self.one_button, "原始比例"),
            (self.full_button, "全屏显示"),
        ):
            button.setObjectName("viewTool")
            button.setMinimumSize(44, 40)
            button.setToolTip(tip)
            view_row.addWidget(button)
        self.reset_button = QPushButton("复位视图")
        self.reset_button.setObjectName("quietButton")
        self.reset_button.setMinimumHeight(40)
        self.reset_button.setToolTip("恢复默认视图")
        view_row.addWidget(self.reset_button)
        work.addWidget(view_bar)

        if self.project_replay is not None:
            self.viewport = ReplayViewport(center)
            try:
                rgb, selected = region_image(self.project_replay)
                depth = depth_heatmap(self.project_replay.depth_path)
                route_image = QImage(str(self.project_replay.trajectory_dir / "route_diagram.png"))
                if route_image.isNull():
                    route_image = selected
                self.viewport.set_data(self.project_replay, {
                    "rgb": QImage(str(self.project_replay.rgb_path)),
                    "selected": selected,
                    "depth": depth,
                    "route": route_image,
                }, self.project_replay.route_groups)
                self.controller.load_replay(self.project_replay)
            except Exception as exc:
                self.project_replay = None
                self.replay_error = str(exc)
                self.viewport = viewport_factory(center)
        else:
            self.viewport = viewport_factory(center)
        self.viewport.regions_edited.connect(self.controller.set_regions)
        self.reset_button.clicked.connect(self.viewport.reset_view)
        self.fit_button.clicked.connect(self.viewport.reset_view)
        self.one_button.clicked.connect(self.viewport.reset_view)
        self.full_button.clicked.connect(self.toggle_maximized)
        work.addWidget(self.viewport, 1)
        self.palette = ToolPalette(self.viewport)
        self.palette.move(12, 12)
        self.palette.tool_requested.connect(self.controller.set_tool)
        self.palette.undo_requested.connect(self.controller.undo)
        self.palette.clear_requested.connect(self.clear_regions)
        self.viewport.installEventFilter(self)
        body.addWidget(center, 1)

        self.pages = [CapturePage(), RegionPage(), PlanningPage(), ConfirmationPage()]
        self.page_stack = QStackedWidget()
        for page in self.pages:
            scroll = QScrollArea()
            scroll.setObjectName("rightPanel")
            scroll.setWidgetResizable(True)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            scroll.setWidget(page)
            self.page_stack.addWidget(scroll)
            page.focus_requested.connect(self.controller.focus)
        self.page_stack.setMinimumWidth(280)
        self.page_stack.setMaximumWidth(320)
        self.page_stack.setFixedWidth(300)
        body.addWidget(self.page_stack)
        self.pages[1].example_requested.connect(self.load_example)
        self.pages[2].parameter_changed.connect(self.controller.set_parameter)
        column.addLayout(body, 1)
        self.action_bar = ActionBar()
        self.action_bar.action_requested.connect(self.perform_action)
        column.addWidget(self.action_bar)
        self.build_drawers()
        shortcut = QShortcut(QKeySequence.Undo, self)
        shortcut.activated.connect(self.controller.undo)
        self.controller.changed.connect(self.refresh)
        self.controller.history_changed.connect(self.refresh_history)
        self.refresh()

    def eventFilter(self, watched, event):
        if hasattr(self, "viewport") and watched is self.viewport and event.type() == QEvent.Resize:
            self.palette.move(12, 12)
        return super().eventFilter(watched, event)

    def refresh(self):
        s = self.controller.state
        if not s.replay_mode and isinstance(self.viewport, ReplayViewport):
            self._switch_to_mock_viewport()
        self.header.update_state(s)
        self.sidebar.update_state(s)
        self.page_title.setText(("三维采集", "区域标记", "轨迹规划", "运行预览")[s.stage])
        self.page_subtitle.setText((
            "调整相机位置，采集完整的目标表面",
            "在固定视角标出目标规划区和禁入区",
            "先核对位置和路线，再调整扫描行距",
            "仿真检查当前任务与规划结果",
        )[s.stage])
        self.reference.setText(
            "●  项目回放 · 空间基准已锁定"
            if s.replay_mode else "●  空间基准已锁定" if s.capture.reference_ready else "空间基准待建立"
        )
        tone(self.reference, "good" if s.capture.reference_ready else "")
        self.page_stack.setCurrentIndex(s.stage)
        self.pages[s.stage].update_state(s)
        names = VIEWS[s.stage]
        for i, btn in enumerate(self.view_buttons):
            btn.setVisible(i < len(names))
            if i < len(names):
                btn.setProperty("view_name", names[i])
                btn.setText(self.view_labels.get(names[i], names[i]))
                tone(btn, "active" if names[i] == s.view else "")
        if s.replay_mode and isinstance(self.viewport, ReplayViewport):
            self.viewport.set_view(s.stage, s.view)
            self.viewport.focus_issue(s.issue_focus)
        else:
            self.viewport.set_scene(scene_from_state(s))
        self.palette.update_state(s, bool(self.controller.undo_stack))
        self.palette.raise_()
        notice, severity = current_notice(s)
        self.alert.setText(("●  " if severity == "normal" else "！  ") + notice)
        tone(self.alert, severity)
        primary, secondary = actions(s)
        self.action_bar.update_state(primary, secondary, notice, severity, self._status_summary(s))
        if s.replay_mode:
            self.developer_info.setText("数据源：项目已保存回放（演示自动选择）\n坐标：空间基准 · metres\n视口：只读项目数据\n临床可执行性：阻断\n规划：保留原始缺口 / 覆盖状态")
        else:
            self.developer_info.setText("参考坐标：空间基准\n原型独立运行，未连接设备。\n视口：QPainter MockViewport\n记录：仅保存演示规划 JSON")

    def _switch_to_mock_viewport(self):
        """Leave read-only replay without leaving a ReplayViewport behind."""
        old = self.viewport
        self.work_layout.removeWidget(old)
        old.removeEventFilter(self)
        old.deleteLater()
        self.viewport = create_viewport(self.center)
        self.work_layout.addWidget(self.viewport, 1)
        try:
            self.reset_button.clicked.disconnect()
        except RuntimeError:
            pass
        self.reset_button.clicked.connect(self.viewport.reset_view)
        try:
            self.fit_button.clicked.disconnect()
            self.one_button.clicked.disconnect()
        except RuntimeError:
            pass
        self.fit_button.clicked.connect(self.viewport.reset_view)
        self.one_button.clicked.connect(self.viewport.reset_view)
        self.viewport.regions_edited.connect(self.controller.set_regions)
        self.viewport.installEventFilter(self)
        self.palette.deleteLater()
        self.palette = ToolPalette(self.viewport)
        self.palette.move(12, 12)
        self.palette.tool_requested.connect(self.controller.set_tool)
        self.palette.undo_requested.connect(self.controller.undo)
        self.palette.clear_requested.connect(self.clear_regions)
        self.project_replay = None

    @staticmethod
    def _status_summary(s):
        if s.replay_mode:
            project = s.replay_label.replace("项目回放 · ", "")
            return f"{project}  |  ● 相机：回放  |  FPS —  |  分辨率 —  |  GPU —  |  存储 —"
        return f"{s.patient.identifier}  |  ● 相机：已连接  |  FPS 30  |  1920 × 1080  |  GPU 28%  |  存储 12.4 / 100 GB"

    def toggle_maximized(self):
        self.showNormal() if self.isMaximized() else self.showMaximized()

    def ask_discard(self, title, text):
        return QMessageBox.question(self, title, text, QMessageBox.Yes | QMessageBox.No, QMessageBox.No) == QMessageBox.Yes

    def perform_action(self, key):
        primary, secondary = actions(self.controller.state)
        action = next((a for a in [primary, *secondary] if a.key == key), None)
        if not action or not action.enabled:
            return
        if action.destructive and not self.ask_discard("重新开始采集", "重新采集会清除当前采集草稿、选区及未保存规划。已保存的历史记录会保留。继续？"):
            return
        self.controller.dispatch(key)

    def clear_regions(self):
        if self.controller.state.region.strokes and self.ask_discard("清空区域标记", "清空当前规划区和禁入区？可使用撤销恢复。"):
            self.controller.clear_regions()

    def load_example(self):
        if self.controller.state.region.strokes and not self.ask_discard("替换区域标记", "用示例替换当前标记？可使用撤销恢复。"):
            return
        self.controller.load_example()

    def change_scenario(self, key):
        s = self.controller.state
        if key == s.scenario:
            return
        if s.capture.reference_ready and not s.plan.saved_id:
            if not self.ask_discard("切换原型场景", "切换会丢弃当前采集与未保存规划，并载入新的模拟任务。继续？"):
                self.header.update_state(s)
                return
        self.controller.set_scenario(key)

    def build_drawers(self):
        self.history_dock = QDockWidget("规划历史 · 演示记录", self)
        self.history_dock.setObjectName("historyDock")
        self.history_dock.setAllowedAreas(Qt.RightDockWidgetArea)
        history = QWidget()
        col = QVBoxLayout(history)
        col.setContentsMargins(18, 18, 18, 18)
        col.addWidget(text_label("本机保存的研发规划预览", "sectionTitle"))
        col.addWidget(text_label("只读查看，不改变当前任务。", "muted"))
        self.history_list = QListWidget()
        self.history_list.setMinimumWidth(280)
        self.history_list.currentRowChanged.connect(self.show_record)
        col.addWidget(self.history_list, 1)
        self.history_detail = QTextBrowser()
        col.addWidget(self.history_detail, 1)
        self.history_dock.setWidget(history)
        self.addDockWidget(Qt.RightDockWidgetArea, self.history_dock)
        self.history_dock.hide()
        self.settings_dock = QDockWidget("显示设置", self)
        self.settings_dock.setObjectName("settingsDock")
        settings = QWidget()
        settings_col = QVBoxLayout(settings)
        settings_col.setContentsMargins(18, 18, 18, 18)
        checkbox = QCheckBox("显示参考坐标轴")
        checkbox.setMinimumHeight(42)
        checkbox.toggled.connect(self.set_axes)
        self.axes_box = checkbox
        settings_col.addWidget(checkbox)
        settings_col.addWidget(text_label("研发信息", "sectionTitle"))
        self.developer_info = text_label("", "muted", True)
        settings_col.addWidget(self.developer_info)
        settings_col.addStretch()
        self.settings_dock.setWidget(settings)
        self.addDockWidget(Qt.RightDockWidgetArea, self.settings_dock)
        self.settings_dock.hide()
        self.history_records = []

    def set_axes(self, enabled):
        self.controller.set_axes(enabled)

    def show_history(self):
        self.settings_dock.hide()
        self.history_dock.setWindowTitle("项目回放 · 只读" if self.controller.state.replay_mode else "规划历史 · 演示记录")
        self.refresh_history()
        self.history_dock.setVisible(not self.history_dock.isVisible())

    def show_settings(self):
        self.history_dock.hide()
        self.settings_dock.setVisible(not self.settings_dock.isVisible())

    def refresh_history(self):
        if self.controller.state.replay_mode:
            self.history_records = []
            self.history_list.clear()
            self.history_list.addItem("当前项目回放\n只读数据，不写入规划历史")
            self.history_detail.setPlainText(
                f"{self.controller.state.replay_label}\n\n"
                "这是已保存的项目回放，不是新的规划版本。\n"
                "原始数据、区域和轨迹不会被当前原型修改。"
            )
            return
        self.history_records = self.controller.store.records()
        self.history_list.clear()
        for r in self.history_records:
            self.history_list.addItem(f"{r['saved_at'].replace('T', ' ')}\n{r['patient']['name']} · 腹部 · 预览 {r['plan']['revision']:02d}")
        if self.history_records:
            self.history_list.setCurrentRow(0)
        else:
            self.history_detail.setPlainText("暂无已保存规划。\n完成规划确认后，记录会显示在这里。")

    def show_record(self, index):
        if not 0 <= index < len(self.history_records):
            return
        r = self.history_records[index]
        self.history_detail.setPlainText(
            f"研发规划预览 / 只读\n\n患者：{r['patient']['name']}\n任务：{r['patient']['task']}\n行距：{r['final_spacing_mm']:.2f} mm\n来源：{r['spacing_source']}\n参数性质：占位值，非设备参数\n生成：{r['plan']['generated_at'].replace('T', ' ')}\n\n包含规划区、禁入区、模拟轨迹和模型来源。\n未包含治疗执行控制。"
        )

    def closeEvent(self, event):
        s = self.controller.state
        if s.replay_mode:
            self.controller.capture_timer.stop()
            self.controller.operation_timer.stop()
            event.accept()
            return
        if s.plan.phase in ("ready", "failed") and not s.plan.saved_id:
            if not self.ask_discard("关闭工作台", "当前规划尚未保存，关闭会丢弃该草稿。继续？"):
                event.ignore()
                return
        self.controller.capture_timer.stop()
        self.controller.operation_timer.stop()
        event.accept()
