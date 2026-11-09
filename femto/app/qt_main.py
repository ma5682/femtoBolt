"""PySide6 entry for the real acquisition, annotation and planning workflow."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import traceback

from PySide6.QtCore import Qt, QTimer, QSize
from PySide6.QtGui import QFont, QPixmap, QFontDatabase
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QFrame, QHBoxLayout, QLabel, QMainWindow,
    QMessageBox, QPushButton, QStackedWidget, QVBoxLayout, QWidget,
)

from femto.paths import LOG_ROOT, OUTPUTS_ROOT, REPO_ROOT
from femto.data_access import DataLease
from femto.data_backup import assert_no_pending_restore
from femto.app.qt_capture import CapturePage
from femto.app.qt_planning import PlanningPage
from femto.app.qt_review import ReviewPage
from femto.app.qt_history import HistoryDialog, AdvancedToolsDialog
from femto.app.qt_common import breakable_text
from femto.app.qt_execution import ExecutionPage
from femto.app.qt_icons import line_icon


def apply_theme(app):
    app.setStyle('Fusion')
    # Qt's font database can be empty in a clean/offscreen runtime even when
    # Windows has the CJK font installed. Register it explicitly when present;
    # the family list in the QSS still provides portable fallbacks elsewhere.
    for font_path in (Path(r'C:\Windows\Fonts\msyh.ttc'),
                      Path(r'C:\Windows\Fonts\NotoSansSC-VF.ttf')):
        if font_path.is_file():
            QFontDatabase.addApplicationFont(str(font_path))
    app.setFont(QFont('Microsoft YaHei UI', 10))
    app.setStyleSheet((Path(__file__).parent / 'styles' / 'medical.qss').read_text(encoding='utf-8'))


class WorkstationWindow(QMainWindow):
    """Own page lifetimes so returning to capture keeps the current draft."""
    def __init__(self, *, connect_camera=True, project=None):
        super().__init__()
        self.setWindowTitle('悠自元 · 三维感知与治疗规划工作站')
        self.resize(1600, 940)
        self.setMinimumSize(1120, 740)
        self._planning_source = None
        self._pending_sources = {}
        self._planning_load_origin = None
        self._close_pending = False
        self._close_timer = QTimer(self)
        self._close_timer.setInterval(250)
        self._close_timer.timeout.connect(self.close)
        root = QWidget(self)
        root.setObjectName('appRoot')
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        header = QFrame()
        header.setObjectName('patientHeader')
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(0)
        brand_row = QWidget()
        brand_row.setObjectName('brandRow')
        bar = QHBoxLayout(brand_row)
        bar.setContentsMargins(16, 6, 16, 6)
        bar.setSpacing(12)
        logo = QLabel()
        logo.setObjectName('brandLogo')
        logo.setFixedSize(156, 44)
        logo.setAlignment(Qt.AlignCenter)
        pixmap = QPixmap(str(Path(__file__).parent / 'assets' / 'uziyuan_logo_header.png'))
        if not pixmap.isNull():
            # Display the supplied artwork's lockup without its transparent
            # canvas margins; keep the original source asset intact.
            lockup = pixmap.copy(245, 145, 1685, 450)
            logo.setPixmap(lockup.scaled(152, 42, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            logo.setText('悠自元')
        bar.addWidget(logo)
        texts = QVBoxLayout()
        texts.setSpacing(2)
        title = QLabel('三维感知与治疗规划工作站')
        title.setObjectName('workstationTitle')
        texts.addWidget(title)
        self.subtitle = QLabel('三维采集  ·  区域标记  ·  轨迹规划  ·  方案确认')
        self.subtitle.setObjectName('workflowHint')
        self.subtitle.setWordWrap(True)
        self.subtitle.setTextFormat(Qt.PlainText)
        texts.addWidget(self.subtitle)
        bar.addLayout(texts, 1)
        self.project_badge = QLabel('当前采集  ·  未绑定病例')
        self.project_badge.setObjectName('projectBadge')
        self.project_badge.setTextFormat(Qt.PlainText)
        self.project_badge.setProperty('tone', 'pending')
        self.project_badge.hide()
        self.execution_badge = QLabel('研发预览  ·  不可执行')
        self.execution_badge.setObjectName('modeBadge')
        self.execution_badge.setToolTip('当前界面只生成和核对规划结果，不直接驱动治疗设备。')
        self.execution_badge.hide()
        header_layout.addWidget(brand_row)
        navigation = QFrame()
        navigation.setObjectName('workspaceNavigation')
        nav = QHBoxLayout(navigation)
        nav.setContentsMargins(20, 0, 20, 0)
        nav.setSpacing(6)
        self.capture_button = QPushButton('采集工作台')
        self.capture_button.setObjectName('headerButton')
        self.capture_button.clicked.connect(self.show_capture)
        nav.addWidget(self.capture_button)
        self.open_button = QPushButton('打开项目')
        self.open_button.setObjectName('headerButton')
        self.open_button.clicked.connect(self.choose_project)
        nav.addWidget(self.open_button)
        nav.addStretch()
        self.history_button = QPushButton('历史记录')
        self.history_button.setObjectName('headerButton')
        self.history_button.clicked.connect(self.show_history)
        self.history_button.setVisible(True)
        nav.addWidget(self.history_button)
        self.devices_button = QPushButton('设备与模拟')
        self.devices_button.setObjectName('headerButton')
        self.devices_button.clicked.connect(self.show_devices)
        nav.addWidget(self.devices_button)
        for button, name in ((self.capture_button, 'workspace'), (self.open_button, 'folder'),
                             (self.history_button, 'history'), (self.devices_button, 'device')):
            button.setIcon(line_icon(name))
            button.setProperty('iconName', name)
            button.setIconSize(QSize(20, 20))
        self.advanced_button = QPushButton('高级工具')
        self.advanced_button.setObjectName('headerButton')
        self.advanced_button.clicked.connect(self.show_advanced)
        self.advanced_button.setVisible(False)
        nav.addWidget(self.advanced_button)
        header_layout.addWidget(navigation)
        layout.addWidget(header)

        self.pages = QStackedWidget()
        self.capture = CapturePage(connect_camera=connect_camera)
        self.planning = PlanningPage()
        self.review = ReviewPage()
        self.execution = ExecutionPage()
        self.pages.addWidget(self.capture)
        self.pages.addWidget(self.planning)
        self.pages.addWidget(self.review)
        self.pages.addWidget(self.execution)
        layout.addWidget(self.pages, 1)
        self._header_nav_buttons = (self.capture_button, self.open_button,
                                    self.history_button, self.devices_button, self.advanced_button)
        self._set_header_active(self.capture_button)
        self.capture.planning_requested.connect(self.open_planning)
        self.capture.planning_preview_requested.connect(lambda source:self.open_planning(source,review_model=True))
        self.capture.history_requested.connect(self.show_history)
        self.capture.status_changed.connect(self._capture_context_changed)
        self.planning.back_requested.connect(self.show_capture)
        self.planning.saved.connect(self.plan_saved)
        self.planning.review_requested.connect(self.open_review)
        self.planning.new_capture_requested.connect(self.new_task)
        self.planning.history_requested.connect(lambda:self.show_history(all_plans=True))
        self.review.edit_requested.connect(lambda source:self.open_planning(source,new_version=True))
        self.review.capture_requested.connect(lambda source:self.show_capture(source=source))
        self.review.history_requested.connect(lambda:self.show_history(all_plans=True))
        self.review.new_capture_requested.connect(self.new_task)
        self.review.simulation_requested.connect(self.simulate_saved_plan)
        self.planning.rescan_requested.connect(lambda:self.rescan_region(self.planning))
        self.review.rescan_requested.connect(lambda:self.rescan_region(self.review))
        self.review.loaded.connect(self._review_loaded)
        # Only remember a source after the asynchronous load has completed.  A
        # failed load must remain retryable from the same path.
        self.planning.loaded.connect(self._planning_loaded)
        self.planning.load_return_requested.connect(self._return_from_planning_load)
        self.planning.load_started.connect(self._remember_planning_load_origin)
        self.planning.status_changed.connect(self._sync_context)
        self.pages.currentChanged.connect(self._sync_context)
        self.review.runner.finished.connect(lambda: self._context_finished(self.review))
        self.planning.runner.finished.connect(lambda: self._context_finished(self.planning))
        self.capture.review_task.finished.connect(self._sync_context)
        self.capture.task_name.textChanged.connect(self._sync_context)
        from femto.app.qt_workflow_navigation import WorkflowNavigation
        self.workflow_navigation = WorkflowNavigation(self)
        self._sync_context()
        if project is not None:
            QTimer.singleShot(0, lambda: self.open_review(str(project)))

    def _set_header_active(self, active):
        """Keep the current workbench destination visible in the header."""
        for button in self._header_nav_buttons:
            state = 'active' if button is active else 'idle'
            if button.property('state') == state:
                continue
            button.setProperty('state', state)
            name = button.property('iconName')
            if name:
                button.setIcon(line_icon(name, active=button is active))
            # Dynamic QSS properties need a polish cycle to update immediately.
            button.style().unpolish(button)
            button.style().polish(button)

    def open_planning(self, source, *, new_version=False,review_model=False):
        if not self.execution.can_leave():
            QMessageBox.information(self, '模拟任务进行中', '请先请求停止并等待结果，结束当前模拟任务，再切换项目。')
            return False
        if not new_version and self.pages.currentWidget() is self.capture and self.workflow_navigation.resume_planning(source):
            return True
        path = Path(source).resolve()
        if not self.capture.can_leave():
            QMessageBox.information(
                self, '采集进行中',
                '请完成或取消本次采集后再打开规划。已接纳的采集数据会保留。')
            return False
        if self._planning_source != path or new_version:
            if not self.planning.can_leave():
                QMessageBox.information(
                    self, '规划任务进行中',
                    '请等待当前任务完成，或在规划页点击“取消任务”后再切换项目。')
                return False
            revision=getattr(self.review.model,'revision_id',None) if new_version and self.review.source==path else None
            self._pending_sources[self.planning] = path
            options={'review_model':True} if review_model else {}
            origin=self.pages.currentWidget()
            if origin is self.capture and self.capture.session.result is not None:
                result=Path(self.capture.session.result).resolve()
                if path in (result,result/'reconstruction'):
                    options.update(preview=self.capture.scan_roi_preview,
                                   loading_context=self.capture.task_name.text())
                    prepared=self.capture.take_prepared_review(path)
                    if prepared is not None:options['prepared']=prepared
            if not self.planning.load_source(path,force=new_version,restore_draft=not new_version,new_version=new_version,expected_revision=revision,**options):
                self._pending_sources.pop(self.planning, None)
                self._planning_load_origin=None
                return False
            if not self.planning.busy:
                self._pending_sources.pop(self.planning, None)
        self.pages.setCurrentWidget(self.planning)
        self._sync_context()
        self._set_header_active(self._planning_load_origin[1] if self._planning_load_origin else self.open_button)
        return True

    def _remember_planning_load_origin(self):
        header=next((button for button in self._header_nav_buttons if button.property('state')=='active'),self.open_button)
        self._planning_load_origin=(self.pages.currentWidget(),header)

    def _return_from_planning_load(self):
        origin,header=self._planning_load_origin or (self.capture,self.capture_button)
        self._planning_load_origin=None
        self._pending_sources.pop(self.planning,None)
        if not self._close_pending:
            self.pages.setCurrentWidget(origin)
            self._set_header_active(header)
            self._sync_context()

    def open_review(self,source, *, expected_record=None):
        if not self.capture.can_leave() or self.planning.busy or self.review.busy or not self.execution.can_leave():
            QMessageBox.information(self,'任务进行中','请先结束当前采集或后台任务，再查看保存结果。')
            return False
        self._pending_sources[self.review] = Path(source).resolve()
        started = self.review.load_source(source) if expected_record is None else self.review.load_source(source, expected_record=expected_record)
        if not started:
            self._pending_sources.pop(self.review, None)
            return False
        self.pages.setCurrentWidget(self.review)
        self._sync_context()
        self._set_header_active(self.history_button)
        return True

    def _review_loaded(self,source):
        self._pending_sources.pop(self.review, None)
        self._sync_context()

    def _context_finished(self, page):
        self._pending_sources.pop(page, None)
        self._sync_context()

    def _sync_context(self, *_):
        """Describe only the visible page and its committed data source."""
        page = self.pages.currentWidget()
        preparing=self.planning.loading_source is not None
        for button in self._header_nav_buttons:button.setEnabled(not preparing)
        if page is self.execution:
            self.project_badge.setText('模拟联调 · 不可执行')
            self.subtitle.setText('模拟设备与事件回看 · 无真实运动及超声输出')
            return
        pending = self._pending_sources.get(page)
        if page is self.planning and preparing:
            source=self.planning.loading_source
            mode='准备模型'
            name=self.planning._loading_request['loading_context'] or source.name
        elif pending is not None and page.busy:
            mode, name = '正在核验来源', pending.name
            source = pending
        elif page is self.capture:
            state = self.capture.session
            source = state.result or state.source
            if state.result:
                mode = '历史采集 · 只读' if self.capture._history_replay else '本次采集结果'
                if self.capture.review_task.busy or state.state == 'review_load_error':
                    name = Path(state.result).name + (' · 读取中' if self.capture.review_task.busy else ' · 预览读取失败，请重试')
                else:
                    name = self.capture.task_name.text()
            else:
                mode = '界面预览 · 未连接设备' if self.capture.preview_mode else '采集工作台'
                name = self.capture.task_name.text()
        else:
            source = page.source
            task = getattr(page.model, 'task_context', None) or {}
            mode = '历史结果 · 只读' if page is self.review else '规划编辑 · 研发预览'
            name = task.get('name') or (Path(source).name if source else '尚未加载数据')
        self.project_badge.setText(mode + ' · 未绑定病例')
        self.project_badge.setToolTip(str(source) if source else '尚无已加载的数据来源')
        self.subtitle.setText(breakable_text(f"任务：{name or '未命名'}"))
        self.subtitle.setToolTip(str(source) if source else '请填写任务名称；不会自动识别或绑定病例。')
        if hasattr(self, 'workflow_navigation'):
            self.workflow_navigation.refresh()

    def new_task(self):
        if self.review.busy or not self.planning.can_leave() or not self.capture.can_leave() or not self.execution.can_leave():
            QMessageBox.information(self,'任务进行中','请先完成或取消当前任务，再新建采集。')
            return False
        self.capture.new_capture()
        self.pages.setCurrentWidget(self.capture)
        self._set_header_active(self.capture_button)
        self._capture_context_changed()
        return True

    def show_capture(self, *, source=None):
        if self.planning.busy or self.review.busy or not self.execution.can_leave():
            QMessageBox.information(self,'任务进行中','请等待后台任务完成或取消后再切换。')
            return False
        current=self.pages.currentWidget()
        if source is None and current is self.review:
            source=getattr(self.review.model,'capture_source',None) or self.review.source
        elif source is None and current is self.planning:
            source=getattr(self.planning.model,'capture_source',None) or self.planning.current_source
        if source and not self.capture.restore_review_for_source(source):
            QMessageBox.information(self,'无法回显关联采集',
                '此记录的关联采集尚未核验或当前读取尚未完成。继续在当前页面查看，或点击“新建采集”开始独立任务；不会显示其他项目的采集结果。')
            return False
        self.workflow_navigation.remember_capture_return(current, source)
        self.pages.setCurrentWidget(self.capture)
        self._set_header_active(self.capture_button)
        self._capture_context_changed()
        return True

    def _capture_context_changed(self, *_):
        if self.pages.currentWidget() is not self.capture:
            return
        self._sync_context()

    def choose_project(self):
        dialog = QMessageBox(self)
        dialog.setWindowTitle('打开项目')
        dialog.setText('选择要打开的已采数据或保存方案。')
        reconstruction = dialog.addButton('重建项目', QMessageBox.ActionRole)
        archive = dialog.addButton('区域 / 路线方案', QMessageBox.ActionRole)
        dialog.addButton('取消', QMessageBox.RejectRole)
        dialog.exec()
        if dialog.clickedButton() is reconstruction:
            path = QFileDialog.getExistingDirectory(self, '选择结果或 reconstruction 目录', str(OUTPUTS_ROOT))
        elif dialog.clickedButton() is archive:
            path, _ = QFileDialog.getOpenFileName(self, '打开已保存方案', str(OUTPUTS_ROOT),
                                                  '规划方案 (*.zip *.json)')
        else:
            return
        if path:
            self.open_review(path)

    def rescan_region(self, page):
        from femto.app.region_quality import rescan_context
        context=rescan_context(page.model)
        if (context is None or self.review.busy or not self.planning.can_leave()
                or not self.capture.can_leave() or not self.execution.can_leave()):
            return False
        task=dict(context);task.pop('gap_followup',None)
        followup=getattr(page,'_gap_followup_request',None)
        if followup is not None:
            from femto.dataset.gap_followup import checked
            from femto.planning.planning_revision import digest
            try:
                followup=checked(followup,task['task_id'])
                if digest(getattr(page,'plan',None))!=followup['origin_plan_sha256']:
                    raise ValueError('缺口来源方案已改变，请重新选择处理范围')
            except ValueError as error:
                page._status(str(error));return False
            task['gap_followup']=followup
        self.capture.new_capture(task=task)
        self.pages.setCurrentWidget(self.capture)
        self._set_header_active(self.capture_button);self._sync_context()
        if followup is not None:page._gap_followup_request=None
        return True

    def show_history(self, *, all_plans=False):
        dialog = HistoryDialog(self)
        if all_plans:dialog.show_all_plans()
        # Keep the picker until the requested model is actually installed.
        dialog.planning_requested.connect(
            lambda source: self._open_planning_from_history(dialog, source))
        dialog.exec()
        dialog.deleteLater()

    def _open_planning_from_history(self, dialog, source):
        """Keep selection and retry controls until this exact read succeeds."""
        if dialog.opening:
            return False
        target = Path(source).resolve()
        dialog.set_opening(True, '正在读取并核验所选记录；成功后进入只读回看…')

        def loaded(path):
            if Path(path).resolve() == target and not dialog._closing_requested:
                dialog.accept()

        def progress(message):
            if not dialog._open_cancelling:
                dialog.status.setText(str(message))

        def finished():
            self.review.loaded.disconnect(loaded)
            self.review.runner.progress.disconnect(progress)
            self.review.runner.finished.disconnect(finished)
            dialog.cancel_open_requested.disconnect(self.review.runner.cancel)
            dialog.set_opening(False, self.review.status.text() + '\n可重试所选记录，或选择其他记录。')

        self.review.loaded.connect(loaded)
        self.review.runner.progress.connect(progress)
        self.review.runner.finished.connect(finished)
        dialog.cancel_open_requested.connect(self.review.runner.cancel)
        if not self.open_review(source):
            finished()
            dialog.status.setText('未开始读取；请先完成当前任务，再重试所选记录。')
            return False
        return True

    def show_advanced(self):
        dialog = AdvancedToolsDialog(self)
        dialog.exec()

    def show_devices(self):
        if not self.capture.can_leave() or self.planning.busy or self.review.busy:
            QMessageBox.information(self, '任务进行中', '请先结束采集或后台读取，再进入模拟联调。')
            return False
        self.pages.setCurrentWidget(self.execution)
        self._set_header_active(self.devices_button)
        self._sync_context()
        return True

    def simulate_saved_plan(self, source):
        if not self.execution.can_leave() or self.execution.busy:
            QMessageBox.information(self, '模拟任务尚未结束', '请先撤销冻结或请求停止并等待结果，再选择保存方案。')
            return False
        plan = self.review.plan if self.review.source == Path(source).resolve() else None
        if not plan or not plan.get('surface_trajectory'):
            QMessageBox.information(self, '没有已保存路线', '请先打开并核验包含三维表面路线的已保存方案。')
            return False
        expected_revision = getattr(self.review.model, 'revision_id', None)
        expected_sha256 = plan['surface_trajectory'].get('result_sha256')
        if not self.show_devices():return False
        return self.execution.use_saved_plan(source, expected_revision=expected_revision,
                                             expected_result_sha256=expected_sha256)

    def plan_saved(self, path):
        if self.pages.currentWidget() is not self.planning:
            return
        self.project_badge.setProperty('tone', 'pending')
        self.project_badge.style().unpolish(self.project_badge)
        self.project_badge.style().polish(self.project_badge)
        self.project_badge.setText('研发方案已保存 · 未绑定病例')
        self.project_badge.setToolTip(str(path))
        self.subtitle.setText(breakable_text('当前方案：' + Path(path).name))

    def _planning_loaded(self, source):
        path = Path(source).resolve()
        self._planning_source = path
        self._planning_load_origin = None
        self._pending_sources.pop(self.planning, None)
        self._sync_context()

    def closeEvent(self, event):
        if self.review.busy or not self.planning.can_leave() or not self.capture.can_leave() or not self.execution.can_leave():
            event.ignore()
            if not self._close_pending:
                answer=QMessageBox.question(self, '任务尚未结束',
                    '采集、模型读取、规划或模拟任务正在运行。是否结束任务并关闭？\n已记录的原始数据与模拟事件会保留。选择“否”继续等待。',
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if answer != QMessageBox.Yes:return
                self._close_pending=True
                self.setEnabled(False)
                self.subtitle.setText('正在取消任务，完成后自动关闭；已记录的原始数据保留。')
                self.planning.runner.cancel()
                self.review.runner.cancel()
                self.capture.review_task.cancel()
                self.execution.shutdown()
                if self.capture.session.active:self.capture.finish(False)
                self._close_timer.start()
            return
        # Always stop both owners. Short-circuiting here can leave the camera
        # host alive when a planning worker needs another close attempt.
        planning_stopped = self.planning.shutdown()
        if not planning_stopped and self.planning.journal and self.planning.journal.status() == 'error':
            event.ignore()
            self._close_timer.stop();self._close_pending=False
            self.setEnabled(True)
            QMessageBox.warning(self,'草稿未保存','草稿保存失败，窗口保持打开。请在规划页重试保存或另存方案。')
            return
        review_stopped = self.review.shutdown()
        capture_stopped = self.capture.shutdown()
        execution_stopped = self.execution.shutdown()
        if not planning_stopped or not capture_stopped or not review_stopped or not execution_stopped:
            event.ignore()
            if self._close_pending:return
            self._close_pending=True
            self.setEnabled(False)
            self.subtitle.setText('正在结束后台任务与模拟实例，完成后自动关闭。')
            self._close_timer.start()
            return
        self._close_timer.stop()
        event.accept()


def _logs():
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    if sys.stdout is None:
        sys.stdout = (LOG_ROOT / 'qt_workstation_stdout.log').open('a', encoding='utf-8', buffering=1)
    if sys.stderr is None:
        sys.stderr = (LOG_ROOT / 'qt_workstation_stderr.log').open('a', encoding='utf-8', buffering=1)


def main():
    parser = argparse.ArgumentParser(description='Femto Bolt PySide6 工作台')
    parser.add_argument('--ui-preview', action='store_true', help='打开真实界面，不连接相机')
    parser.add_argument('--project', type=Path, help='打开已采项目、便携方案或原生轨迹')
    args = parser.parse_args()
    _logs()
    app = QApplication(sys.argv[:1])
    app.setApplicationName('Femto Bolt Workstation')
    apply_theme(app)

    def report_exception(kind, value, tb):
        traceback.print_exception(kind, value, tb)
        QMessageBox.critical(None, '操作未完成', f'{value}\n详细信息已写入运行日志。')
    sys.excepthook = report_exception
    lease=DataLease(REPO_ROOT)
    try:
        lease.acquire()
        assert_no_pending_restore(REPO_ROOT)
    except RuntimeError as error:
        lease.release()
        QMessageBox.warning(None,'数据维护中',str(error))
        return 2
    try:
        window = WorkstationWindow(connect_camera=not args.ui_preview, project=args.project)
        window.data_lease = lease
        window.show()
        return app.exec()
    finally:
        lease.release()


if __name__ == '__main__':
    raise SystemExit(main())
