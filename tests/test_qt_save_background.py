"""A published route ends the save wait; its editable project keeps every gate closed."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from contextlib import contextmanager
import threading
import unittest
from unittest.mock import PropertyMock, patch

from PySide6.QtCore import QEventLoop, QTimer, Signal
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QMessageBox, QWidget

import test_qt_planning as planning_fixture
import test_route_layout as layout_fixture
from femto.app.qt_workflow_navigation import FINISHING_SAVE
from femto.planning import region_project, surface_workspace


class FakeViewport(QWidget):
    """The real viewport needs a display; the panel logic does not."""

    picked = Signal(dict)

    def set_scene(self, *args, **kwargs): pass
    def reset_camera(self): pass
    def shutdown(self): pass


class HeldProject:
    """Hold the real worker between the published route and its editable project."""

    def wait(self, predicate, seconds=60):
        loop = QEventLoop(); poll = QTimer(); deadline = QTimer()
        poll.setInterval(10); deadline.setSingleShot(True)
        poll.timeout.connect(lambda: loop.quit() if predicate() else None)
        deadline.timeout.connect(loop.quit)
        try:
            poll.start(); deadline.start(int(seconds*1000))
            if not predicate(): loop.exec()
        finally:
            poll.stop(); deadline.stop()
        self.assertTrue(predicate())

    @contextmanager
    def held_project(self, *, fail=False):
        original = region_project.save_project
        entered, released, threads = threading.Event(), threading.Event(), []
        def save(*args, **kwargs):
            threads.append(threading.get_ident())
            entered.set(); released.wait(30)
            if fail: raise OSError('injected project disk failure')
            return original(*args, **kwargs)
        with patch.object(region_project, 'save_project', save):
            try: yield entered, released, threads
            finally: released.set()

    def start_save(self, page, entered):
        page.save_plan()
        self.wait(lambda: page.finishing_save is not None and entered.is_set())
        return page.finishing_save


class SaveBackgroundPageTests(HeldProject, unittest.TestCase):
    @classmethod
    def setUpClass(cls): planning_fixture.QtPlanningTests.setUpClass()

    def setUp(self):
        self.case = planning_fixture.QtPlanningTests(); self.case.setUp()
        self.addCleanup(self.case.doCleanups)
        self.case.mark(); self.page = self.case.page
        self.page.generate(); self.case.wait_task()
        self.page.set_stage(2); self.page.reviewed.setChecked(True)
        self.saved = []; self.page.saved.connect(self.saved.append)

    def finish(self):
        self.wait(lambda: not self.page.busy)

    def test_route_is_released_while_the_same_worker_finishes_the_project(self):
        page = self.page; plan = page.plan; journal = page.journal
        native = surface_workspace.export; exports = []
        def export(*args, **kwargs):
            exports.append(threading.get_ident()); return native(*args, **kwargs)
        with patch.object(surface_workspace, 'export', export), self.held_project() as (entered, released, threads):
            folder = self.start_save(page, entered)
            self.assertTrue(page.busy)
            self.assertTrue((folder/'trajectory.json').is_file())
            self.assertFalse((folder/'planning.region.zip').exists())
            # Nothing is committed before the editable project exists.
            self.assertIsNone(page.archive); self.assertTrue(page.dirty); self.assertEqual(self.saved, [])
            self.assertIs(page.journal, journal); self.assertFalse(page.saved_actions.isVisible())
            self.assertIn('三维路线已保存', page.status_label.text())
            self.assertIn('三维路线已保存', page.save_summary.text())
            self.assertEqual(page.primary.text(), '正在写入可编辑工程…')
            # Viewing is released; inputs and every task stay locked.
            self.assertTrue(page.viewable()); self.assertFalse(page.usable())
            self.assertIs(page.scene_panel._scene[2], page.plan)
            for control in (page.head, page.metadata, page.reviewed, page.primary, page.save_as_button,
                            page.save_report_button, page.open_button):
                self.assertFalse(control.isEnabled())
            self.assertFalse(page.canvas.editable)
            self.assertTrue(page.cancel_button.isVisible()); self.assertTrue(page.cancel_button.isEnabled())
            self.assertIn('三维路线文件保留', page.cancel_button.toolTip())
            page.save_plan(); page.generate(); page.show_report(); page.save_as(); page.verify_coverage()
            self.assertEqual(len(exports), 1); self.assertIs(page.plan, plan)
            self.assertFalse(page.can_leave())
            self.assertIn('可编辑工程正在写入', page.status_label.text())
            self.assertTrue(page.tabs.isTabEnabled(1)); self.assertTrue(page.previous.isEnabled())
            page.previous.click()
            self.assertEqual(page._stage, 1)
            self.assertFalse(page.head.isEnabled()); self.assertFalse(page.canvas.editable)
            self.assertTrue(page.play_button.isEnabled())
            self.assertEqual(page.primary.text(), '下一步：核对并保存'); self.assertTrue(page.primary.isEnabled())
            page.tabs.setCurrentIndex(0)
            self.assertEqual(page._stage, 0)
            self.assertFalse(page.canvas.editable); self.assertFalse(page.tools.isEnabled())
            self.assertFalse(page.primary.isEnabled())
            # Toggling the read-only view keeps the writing notice on screen.
            notice = page.status_label.text()
            self.assertIn('三维路线已保存', notice)
            with patch('femto.app.qt_scene_panel.create_viewport', side_effect=FakeViewport):
                page.scene_panel.mode.setCurrentIndex(1); page.scene_panel.mode.setCurrentIndex(0)
            self.assertEqual(page.status_label.text(), notice)
            # The first stage's "back" leaves for capture, so it stays locked.
            left = []; page.back_requested.connect(left.append)
            self.assertFalse(page.previous.isEnabled()); page.go_back(); page.primary_action()
            self.assertEqual((page._stage, left), (0, []))
            page.tabs.setCurrentIndex(1); page.primary.click()
            self.assertEqual(page._stage, 2)
            # A region-first model returns to capture from the route stage.
            with patch.object(type(page), 'fixed_selection', new_callable=PropertyMock, return_value=True):
                page.refresh()
                self.assertTrue(page.previous.isEnabled()); self.assertFalse(page.tabs.isTabEnabled(0))
                page.go_back()
                self.assertEqual(page._stage, 1)
                self.assertFalse(page.previous.isEnabled()); page.go_back()
                self.assertEqual((page._stage, left), (1, []))
                page.primary.click()
                self.assertEqual(page._stage, 2)
            page.refresh()
            self.assertIs(page.plan, plan); self.assertEqual(len(exports), 1)
            released.set(); self.case.wait_task()
        self.assertEqual(exports, threads); self.assertNotEqual(exports[0], threading.get_ident())
        self.assertIsNone(page.finishing_save)
        self.assertEqual(page.archive, folder); self.assertFalse(page.dirty)
        self.assertEqual(self.saved, [str(folder)])
        self.assertTrue((folder/'planning.region.zip').is_file())
        self.assertIn('规划方案已保存', page.status_label.text())
        self.assertEqual(page.primary.text(), '规划方案已保存')
        self.assertTrue(page.saved_actions.isVisible()); self.assertTrue(page.saved_actions.isEnabled())
        self.assertFalse(journal.path.exists())
        self.assertIs(page.plan, plan); self.assertTrue(page.usable())

    def test_project_failure_after_release_keeps_the_unsaved_plan_retryable(self):
        page = self.page; plan = page.plan
        with self.held_project(fail=True) as (entered, released, _):
            folder = self.start_save(page, entered)
            page.previous.click()
            released.set(); self.finish()
        self.assertIsNone(page.finishing_save)
        self.assertIn('可编辑工程保存未完成', page._error)
        self.assertIn('可编辑工程保存未完成', page.status_label.text())
        self.assertIsNone(page.archive); self.assertTrue(page.dirty); self.assertEqual(self.saved, [])
        self.assertTrue((folder/'trajectory.json').is_file())
        self.assertFalse((folder/'planning.region.zip').exists())
        self.assertIs(page.plan, plan); self.assertTrue(page.usable())
        self.assertEqual(page._stage, 1); self.assertTrue(page.reviewed.isChecked())
        page.set_stage(2)
        self.assertTrue(page.primary.isEnabled())
        page.primary.click(); self.case.wait_task()
        self.assertNotEqual(page.archive, folder)
        self.assertTrue((page.archive/'planning.region.zip').is_file())
        self.assertEqual(self.saved, [str(page.archive)])

    def test_cancel_during_the_project_write_ends_the_released_state(self):
        page = self.page; plan = page.plan
        with self.held_project() as (entered, released, _):
            folder = self.start_save(page, entered)
            page.previous.click()
            page.cancel_button.click()
            self.assertTrue(page.busy); self.assertIsNone(page.finishing_save)
            self.assertFalse(page.viewable())
            self.assertIn('正在取消', page.status_label.text())
            self.assertFalse(page.previous.isEnabled()); self.assertFalse(page.primary.isEnabled())
            self.assertFalse(any(page.tabs.isTabEnabled(index) for index in range(3)))
            page.tabs.setCurrentIndex(2)
            self.assertEqual(page._stage, 1)
            released.set(); self.finish()
        self.assertIn('任务已取消', page.status_label.text()); self.assertFalse(page._error)
        self.assertIsNone(page.archive); self.assertTrue(page.dirty); self.assertEqual(self.saved, [])
        self.assertFalse((folder/'planning.region.zip').exists())
        self.assertIs(page.plan, plan); self.assertTrue(page.usable())

    def test_late_or_foreign_route_notice_never_releases_another_task(self):
        page = self.page
        with self.held_project() as (entered, released, _):
            folder = self.start_save(page, entered)
            token = page._save_token
            released.set(); self.case.wait_task()
        self.assertEqual(page.archive, folder); self.assertIsNone(page._save_token)
        entered, released = threading.Event(), threading.Event()
        def held_report(*args):
            entered.set(); released.wait(30); return 'report'
        with patch.object(surface_workspace, 'report_text', held_report), patch('femto.app.qt_planning.show_text'):
            page.show_report(); self.wait(entered.is_set)
            try:
                for stale in (token, None, object()):
                    page._native_route_published(stale, folder)
                self.assertTrue(page.busy)
                self.assertIsNone(page.finishing_save); self.assertIsNone(page._finishing_save)
                self.assertFalse(page.viewable()); self.assertFalse(page.previous.isEnabled())
            finally:
                released.set(); self.finish()

    def test_route_published_after_a_cancel_request_stays_a_plain_cancellation(self):
        page = self.page; native = surface_workspace.export
        entered, released = threading.Event(), threading.Event()
        def export(*args, **kwargs):
            folder = native(*args, **kwargs)
            entered.set(); released.wait(30)
            return folder
        with patch.object(surface_workspace, 'export', export):
            page.save_plan(); self.wait(entered.is_set)
            self.assertIsNone(page.finishing_save)
            page.cancel_button.click()
            released.set(); self.finish()
        self.assertIsNone(page._finishing_save)
        self.assertIn('任务已取消', page.status_label.text()); self.assertNotIn('三维路线已保存', page.status_label.text())
        self.assertIsNone(page.archive); self.assertTrue(page.dirty); self.assertEqual(self.saved, [])


class SaveBackgroundWindowTests(HeldProject, unittest.TestCase):
    def setUp(self):
        self.fixture = layout_fixture.RouteLayoutTests()
        self.fixture.setUpClass(); self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.window = self.fixture.window; self.page = self.fixture.page
        self.fixture.generate()
        self.page.set_stage(2); self.page.reviewed.setChecked(True)

    def test_released_save_keeps_project_switching_blocked(self):
        page = self.page; window = self.window; navigation = window.workflow_navigation
        with self.held_project() as (entered, released, _), patch.object(QMessageBox, 'information'):
            folder = self.start_save(page, entered)
            available, reasons = navigation.destinations(page)
            self.assertEqual(available, {3, 4, 5})
            self.assertEqual(set(reasons.values()), {FINISHING_SAVE})
            self.assertEqual([button.isEnabled() for button in page.workflow.buttons],
                             [False, False, True, True, False, False])
            self.assertIn(FINISHING_SAVE, page.workflow.buttons[0].toolTip())
            self.assertTrue(navigation.navigate(page, 4)); self.assertEqual(page._stage, 1)
            for step in (1, 2, 6):
                self.assertFalse(navigation.navigate(page, step))
            self.assertFalse(window.new_task()); self.assertFalse(window.show_capture())
            self.assertFalse(window.show_devices()); self.assertFalse(window.open_review(folder))
            self.assertFalse(window.open_planning(folder/'other.region.zip'))
            self.assertIs(window.pages.currentWidget(), page)
            self.assertFalse(page.runner._cancel.is_set()); self.assertEqual(page.finishing_save, folder)
            released.set(); self.fixture.wait_task()
        self.assertEqual(navigation._blocked_reason(), '')
        self.assertEqual(page.archive, folder); self.assertTrue((folder/'planning.region.zip').is_file())

    def test_close_request_waits_for_the_project_then_closes(self):
        page = self.page; window = self.window
        with self.held_project() as (entered, released, _), \
                patch.object(QMessageBox, 'question', side_effect=AssertionError('The write must not be cancelled')):
            folder = self.start_save(page, entered)
            for _ in range(2):
                event = QCloseEvent(); window.closeEvent(event)
                self.assertFalse(event.isAccepted())
            self.assertFalse(window.isEnabled()); self.assertTrue(window._close_timer.isActive())
            # Later write progress must not replace the reason the window is locked.
            page._task_progress('3/4 写入方案清单与压缩包')
            self.assertIn('三维路线已保存 · 3/4', page.status_label.text())
            self.assertIn('写完后自动关闭', window.subtitle.text())
            self.assertFalse(page.runner._cancel.is_set()); self.assertEqual(page.finishing_save, folder)
            released.set()
            self.wait(lambda: not window.isVisible())
        self.assertFalse(window._close_timer.isActive())
        self.assertFalse(page.busy); self.assertFalse(page._error)
        self.assertEqual(page.archive, folder); self.assertFalse(page.dirty)
        self.assertTrue((folder/'planning.region.zip').is_file())

    def test_failed_project_keeps_the_window_open_after_a_close_request(self):
        page = self.page; window = self.window
        with self.held_project(fail=True) as (entered, released, _), \
                patch.object(QMessageBox, 'question', side_effect=AssertionError('The write must not be cancelled')), \
                patch.object(QMessageBox, 'warning') as warning:
            self.start_save(page, entered)
            event = QCloseEvent(); window.closeEvent(event)
            self.assertFalse(event.isAccepted()); self.assertFalse(window.isEnabled())
            released.set()
            self.wait(lambda: warning.called)
        warning.assert_called_once()
        self.assertTrue(window.isVisible()); self.assertTrue(window.isEnabled())
        self.assertFalse(window._close_pending); self.assertFalse(window._close_timer.isActive())
        self.assertIsNone(window._close_saving)
        self.assertNotIn('自动关闭', window.subtitle.text())
        self.assertIn('可编辑工程保存未完成', page.status_label.text())
        self.assertIsNone(page.archive); self.assertTrue(page.dirty)


if __name__ == '__main__':
    unittest.main()
