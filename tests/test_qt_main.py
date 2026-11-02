"""Smoke checks for the clinician-facing Qt shell."""
from __future__ import annotations

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QCloseEvent
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMessageBox, QDialog

from femto.app.qt_main import WorkstationWindow, apply_theme
from femto.app.qt_history import HistoryDialog
from unittest.mock import patch


class WorkstationShellTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        apply_theme(cls.app)

    def test_close_attempt_shuts_down_capture_when_planning_needs_retry(self):
        window = WorkstationWindow(connect_camera=False)
        try:
            event = QCloseEvent()
            with patch.object(window.planning, 'can_leave', return_value=True), \
                 patch.object(window.capture, 'can_leave', return_value=True), \
                 patch.object(window.planning, 'shutdown', return_value=False) as planning_shutdown, \
                 patch.object(window.capture, 'shutdown', return_value=True) as capture_shutdown, \
                 patch.object(QMessageBox, 'information'):
                window.closeEvent(event)
            self.assertTrue(planning_shutdown.called)
            self.assertTrue(capture_shutdown.called)
            self.assertFalse(event.isAccepted())
        finally:
            window.planning.shutdown()
            window.capture.shutdown()
            window.close()

    def test_preview_shell_opens_with_capture_step_active(self):
        window = WorkstationWindow(connect_camera=False)
        try:
            self.assertEqual(window.pages.currentWidget(), window.capture)
            self.assertEqual(window.capture_button.property('state'), 'active')
            self.assertEqual(window.open_button.property('state'), 'idle')
            self.assertTrue(window.project_badge.isHidden())
            self.assertTrue(window.execution_badge.isHidden())
            self.assertIn('未绑定病例', window.project_badge.text())
            self.assertIn('不可执行', window.execution_badge.text())
            self.assertFalse(window.history_button.isHidden())
            self.assertTrue(window.advanced_button.isHidden())
            window._set_header_active(window.open_button)
            self.assertEqual(window.open_button.property('state'), 'active')
            self.assertEqual(window.capture_button.property('state'), 'idle')
            self.assertTrue(window.capture.can_leave())
            self.assertFalse(window.planning.busy)
            self.assertFalse(window.planning.primary.isEnabled())
        finally:
            self.assertTrue(window.planning.shutdown())
            self.assertTrue(window.capture.shutdown())
            window.close()

    def test_returning_from_planning_restores_historical_capture_source(self):
        """The planning back action must repopulate the capture review source."""
        window = WorkstationWindow(connect_camera=False)
        try:
            source = Path("C:/history/complete_demo/reconstruction")
            window.planning.source = source
            window.pages.setCurrentWidget(window.planning)
            with patch.object(window.capture, "restore_review_for_source", return_value=True) as restore:
                window.show_capture()
            restore.assert_called_once_with(source)
            self.assertIs(window.pages.currentWidget(), window.capture)
        finally:
            window.planning.shutdown()
            window.capture.shutdown()
            window.close()

    def test_history_picker_closes_only_after_open_is_accepted(self):
        window = WorkstationWindow(connect_camera=False)
        try:
            with TemporaryDirectory() as root:
                rejected = HistoryDialog(window, root=root)
                with patch.object(window, 'open_review', return_value=False):
                    window._open_planning_from_history(rejected, 'missing')
                self.assertNotEqual(rejected.result(), QDialog.DialogCode.Accepted)
                self.assertFalse(rejected.opening)
                rejected.close()

                accepted = HistoryDialog(window, root=root)
                with patch.object(window, 'open_review', return_value=True):
                    window._open_planning_from_history(accepted, 'history')
                self.assertNotEqual(accepted.result(), QDialog.DialogCode.Accepted)
                self.assertTrue(accepted.opening)
                window.review.loaded.emit(str(Path('history').resolve()))
                self.assertEqual(accepted.result(), QDialog.DialogCode.Accepted)
                window.review.runner.finished.emit()
                self.assertFalse(accepted.opening)
                accepted.close()
        finally:
            window.planning.shutdown()
            window.capture.shutdown()
            window.close()

    def test_task_edits_and_return_clear_previous_saved_context(self):
        window = WorkstationWindow(connect_camera=False)
        try:
            window.capture.task_name.setText('右肩补扫')
            self.assertIn('右肩补扫', window.subtitle.text())
            self.assertNotIn('部位：', window.subtitle.text())
            window.pages.setCurrentWidget(window.planning)
            window.plan_saved('C:/history/old-plan.zip')
            self.assertIn('old-plan.zip', window.subtitle.text())
            window.pages.setCurrentWidget(window.capture)
            self.assertNotIn('old-plan.zip', window.subtitle.text())
            self.assertIn('右肩补扫', window.subtitle.text())
            self.assertIn('未连接设备', window.project_badge.text())
        finally:
            window.close()

    def test_hidden_page_notifications_do_not_replace_current_task(self):
        window = WorkstationWindow(connect_camera=False)
        try:
            window.capture.task_name.setText('当前任务')
            before = (window.project_badge.text(), window.subtitle.text())
            window._planning_loaded('C:/history/other-record')
            window._review_loaded('C:/history/other-review')
            window.plan_saved('C:/history/other-plan.zip')
            self.assertEqual(before, (window.project_badge.text(), window.subtitle.text()))
        finally:
            window.close()

    def test_historical_capture_context_does_not_claim_live_device(self):
        window = WorkstationWindow(connect_camera=False)
        try:
            window.capture._history_replay = True
            window.capture.session.result = Path('C:/history/record-a')
            window.capture.session.state = 'review_load_error'
            window.capture.task_name.setText('不应冒用的旧任务')
            window._sync_context()
            self.assertIn('历史采集', window.project_badge.text())
            self.assertIn('只读', window.project_badge.text())
            self.assertIn('读取失败', window.subtitle.text())
            self.assertNotIn('不应冒用的旧任务', window.subtitle.text())
            self.assertIn('record-a', window.project_badge.toolTip())
        finally:
            window.close()

    def test_reused_planning_source_does_not_leave_pending_load_label(self):
        window = WorkstationWindow(connect_camera=False)
        try:
            source = Path('C:/history/cached-record')
            with patch.object(window.planning, 'load_source', return_value=True):
                self.assertTrue(window.open_planning(source))
            self.assertNotIn(window.planning, window._pending_sources)
            window.planning.runner.busy = True
            window._sync_context()
            self.assertNotIn('正在核验来源', window.project_badge.text())
        finally:
            window.planning.runner.busy = False
            window.close()

    def test_task_labels_are_plain_text_and_not_html(self):
        window = WorkstationWindow(connect_camera=False)
        try:
            window.capture.task_name.setText('<b>任务甲</b>')
            self.assertEqual(window.subtitle.textFormat(), Qt.PlainText)
            self.assertIn('<b>任务甲</b>', window.subtitle.text())
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
