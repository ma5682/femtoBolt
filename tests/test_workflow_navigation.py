"""Workflow clicks preserve data, recheck gates and never restart acquisition."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from copy import deepcopy
from pathlib import Path
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
from PySide6.QtCore import QEventLoop, QTimer, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QMessageBox

from femto.app.qt_capture import CaptureState
import test_route_layout as layout_fixture


class WorkflowNavigationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = layout_fixture.RouteLayoutTests()
        self.fixture.setUpClass(); self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.window = self.fixture.window
        self.page = self.fixture.page
        self.capture = self.window.capture
        self.addCleanup(self.stop_capture)

    def stop_capture(self):
        self.capture.session.active = False
        self.capture.session.finishing = False

    def wait_runner(self, runner):
        loop = QEventLoop(); deadline = QTimer()
        deadline.setSingleShot(True); deadline.timeout.connect(loop.quit)
        runner.finished.connect(loop.quit)
        try:
            deadline.start(60000)
            if runner.busy: loop.exec()
        finally:
            deadline.stop(); runner.finished.disconnect(loop.quit)
        self.assertFalse(runner.busy)
        self.fixture.settle()

    def wait_event(self, event):
        loop = QEventLoop(); timer = QTimer(); deadline = QTimer()
        timer.timeout.connect(lambda: loop.quit() if event.is_set() else None)
        deadline.setSingleShot(True); deadline.timeout.connect(loop.quit)
        timer.start(10); deadline.start(5000)
        if not event.is_set(): loop.exec()
        timer.stop(); deadline.stop()
        self.assertTrue(event.is_set())

    def click(self, page, step):
        button = page.workflow.buttons[step-1]
        self.assertTrue(button.isEnabled(), button.toolTip())
        QTest.mouseClick(button, Qt.LeftButton)
        self.fixture.settle()

    def capture_data(self):
        points = np.array([[0., 0., .5], [.01, .01, .5]])
        cloud = (points, np.ones_like(points))
        return dict(clouds={'final': cloud, 'initial': cloud, 'accumulated': cloud},
                    surface=None, quality={}, frame_count=3,
                    scan_roi_quality=dict(status='checked', supported_pixels=3,
                                          pending_pixels=1, requested_pixels=4),
                    scan_roi_preview=np.full((64, 64, 3), 120, np.uint8))

    def install_capture(self):
        root = self.page.model.reconstruction.parent
        self.capture.session = CaptureState(session_id='synthetic-navigation', result=root,
                                            state='ready', count=3)
        self.capture.clouds = self.capture_data()['clouds']
        self.capture.task_context = deepcopy(self.page.model.task_context)
        self.capture.refresh()
        return root

    def test_mouse_and_keyboard_switches_keep_region_route_and_review_confirmation(self):
        self.fixture.generate(); page = self.page
        page.set_stage(2); page.reviewed.setChecked(True)
        plan = deepcopy(page.plan); work = page.canvas.work.copy()
        forbidden = page.canvas.forbidden.copy(); head = page.head.record()
        model, journal, source = page.model, page.journal, page.source
        self.click(page, 3)
        self.assertEqual(page._stage, 0); self.assertTrue(page.canvas.editable)
        button = page.workflow.buttons[3]; button.setFocus()
        QTest.keyClick(button, Qt.Key_Space); self.fixture.settle()
        self.assertEqual(page._stage, 1)
        self.click(page, 5)
        self.assertEqual(page._stage, 2)
        self.assertTrue(page.reviewed.isChecked()); self.assertTrue(page.usable())
        self.assertIs(page.model, model); self.assertIs(page.journal, journal)
        self.assertEqual(page.source, source); self.assertEqual(page.plan, plan)
        self.assertEqual(page.head.record(), head)
        np.testing.assert_array_equal(page.canvas.work, work)
        np.testing.assert_array_equal(page.canvas.forbidden, forbidden)

    def test_parameter_and_region_edits_disable_downstream_steps(self):
        from femto.planning.treatment_head import head_parameters
        self.page.head.set_record(head_parameters('direct_row_spacing',row_spacing_mm=5))
        self.fixture.generate(); page = self.page; plan = page.plan
        page.set_stage(2); self.click(page, 4)
        page.head.fields['row_spacing_mm'].setText('7')
        self.assertFalse(page.usable()); self.assertIs(page.plan, plan)
        self.assertFalse(page.workflow.buttons[4].isEnabled())
        self.assertFalse(self.window.workflow_navigation.navigate(page, 5))
        with patch('femto.app.qt_planning.surface_workspace.export') as export:
            page.reviewed.setChecked(True); page.save_plan()
        export.assert_not_called()
        page.head.fields['row_spacing_mm'].setText('5')
        self.assertFalse(page.reviewed.isChecked())
        self.click(page, 3)
        page.canvas.work[40:42, 40:42] = 0
        page.canvas.rebuild_layers(); page._selection_changed()
        self.assertTrue(page.selection_dirty); self.assertIsNone(page.plan)
        self.assertFalse(page.workflow.buttons[3].isEnabled())
        self.assertFalse(page.workflow.buttons[4].isEnabled())
        self.assertFalse(self.window.workflow_navigation.navigate(page, 4))

    def test_save_in_progress_blocks_clicks_until_cancellation_finishes(self):
        self.fixture.generate(); page = self.page
        page.set_stage(2); page.reviewed.setChecked(True)
        plan = page.plan; entered = threading.Event(); released = threading.Event()
        def delayed_export(model, region, current_plan, check, progress, *, target_session=None):
            from femto.planning.save_validation import SaveValidationSession
            self.assertIsInstance(target_session, SaveValidationSession)
            entered.set(); released.wait(10); check()
            raise AssertionError('The test save should have been cancelled')
        try:
            with patch('femto.app.qt_planning.surface_workspace.export', side_effect=delayed_export):
                page.save_plan(); self.wait_event(entered)
                self.assertTrue(page.busy)
                self.assertTrue(all(not button.isEnabled() for button in page.workflow.buttons))
                self.assertFalse(self.window.workflow_navigation.navigate(page, 3))
                page.workflow.step_requested.emit(0)
                self.assertIs(self.window.pages.currentWidget(), page)
                self.assertEqual(page._stage, 2); self.assertIs(page.plan, plan)
                page.runner.cancel(); released.set(); self.wait_runner(page.runner)
        finally:
            released.set(); self.wait_runner(page.runner)
        self.assertFalse(page._error)
        self.assertTrue(page.workflow.buttons[2].isEnabled())
        self.assertIs(page.plan, plan); self.assertIsNone(page.archive)

    def test_async_capture_replay_returns_to_exact_draft_without_camera_commands(self):
        self.fixture.generate(); page = self.page
        root = Path(self.fixture.fixture.temp.name)
        portable = root/'navigation.region.zip'
        page.model.capture_source = page.model.reconstruction
        with patch('femto.app.qt_planning.QFileDialog.getSaveFileName', return_value=(str(portable), '')), \
             patch('femto.app.qt_planning.OUTPUTS_ROOT', root/'outputs'):
            page.save_as(); self.fixture.wait_task()
        self.assertEqual(page.source, portable.resolve())
        page.note.setText('保留在便携工程中的未保存备注'); page._metadata_changed()
        page.set_stage(2)
        model, plan, source, journal = page.model, page.plan, page.source, page.journal
        work = page.canvas.work.copy(); head = page.head.record()
        entered = threading.Event(); released = threading.Event()
        def delayed_read(*args):
            entered.set(); released.wait(10); return self.capture_data()
        try:
            with patch('femto.app.qt_capture.read_review', side_effect=delayed_read), \
                 patch('femto.dataset.task_context.read_task', return_value=deepcopy(model.task_context)), \
                 patch.object(self.capture, 'send') as send, \
                 patch.object(page, 'load_source') as reload:
                self.click(page, 1); self.wait_event(entered)
                self.assertIs(self.window.pages.currentWidget(), self.capture)
                self.assertTrue(self.capture.review_task.busy)
                self.assertTrue(all(not button.isEnabled() for button in self.capture.workflow.buttons))
                released.set(); self.wait_runner(self.capture.review_task)
                self.assertEqual(self.capture.workflow.current_step, 1)
                self.assertEqual(self.capture.view, 'initial')
                self.assertFalse(self.capture.workflow_view_notice.isVisible())
                self.assertTrue(self.capture.history_metrics.isVisible())
                self.assertTrue(self.capture.history_rescan.isVisible())
                self.assertFalse(self.capture.inspector_scroll.isVisible())
                self.assertFalse(hasattr(self.capture,'body_site_button'))
                self.assertEqual(self.capture.view, 'initial')
                self.assertEqual(self.capture.workflow.current_step, 1)
                self.assertFalse(self.capture.session.active)
                self.click(self.capture, 5)
                self.assertIs(self.window.pages.currentWidget(), page)
                self.assertEqual(page._stage, 2)
                self.click(page, 1)
                self.capture.primary.click(); self.fixture.settle()
                self.assertEqual(self.capture.workflow.current_step, 2)
                with patch('femto.app.qt_capture.QMessageBox.question', return_value=QMessageBox.Yes):
                    self.capture.primary.click(); self.fixture.settle()
                self.assertIs(self.window.pages.currentWidget(), page)
                self.assertEqual(page._stage, 0)
                reload.assert_not_called(); send.assert_not_called()
        finally:
            released.set(); self.wait_runner(self.capture.review_task)
        self.assertIs(page.model, model); self.assertIs(page.plan, plan)
        self.assertIs(page.journal, journal); self.assertEqual(page.source, source)
        self.assertEqual(page.head.record(), head)
        self.assertEqual(page.note.text(), '保留在便携工程中的未保存备注')
        np.testing.assert_array_equal(page.canvas.work, work)

    def test_failed_capture_preview_can_return_to_loaded_draft(self):
        self.fixture.generate(); page = self.page; page.set_stage(2)
        model, plan, source = page.model, page.plan, page.source
        work = page.canvas.work.copy()
        with patch('femto.app.qt_capture.read_review', side_effect=OSError('合成读取失败')), \
             patch.object(page, 'load_source') as reload, \
             patch.object(self.capture, 'send') as send:
            self.click(page, 1); self.wait_runner(self.capture.review_task)
            self.assertEqual(self.capture.session.state, 'review_load_error')
            self.assertFalse(self.capture.review_ready())
            self.assertFalse(self.capture.workflow.buttons[0].isEnabled())
            self.assertFalse(self.capture.workflow.buttons[1].isEnabled())
            self.click(self.capture, 5)
            self.assertIs(self.window.pages.currentWidget(), page)
            reload.assert_not_called(); send.assert_not_called()
        self.assertIs(page.model, model); self.assertIs(page.plan, plan)
        self.assertEqual(page.source, source); self.assertEqual(page._stage, 2)
        np.testing.assert_array_equal(page.canvas.work, work)

    def test_missing_capture_and_new_capture_cannot_reuse_unrelated_planning(self):
        self.fixture.generate(); page = self.page
        original = page.source
        page.model.capture_source = None
        page.source = Path(self.fixture.fixture.temp.name)/'detached.region.zip'
        self.install_capture(); page.refresh()
        for index in (1, 2):
            self.assertFalse(page.workflow.buttons[index-1].isEnabled())
        self.assertFalse(self.window.workflow_navigation.navigate(page, 0))
        self.assertIs(self.window.pages.currentWidget(), page)
        page.source = original; page.refresh()
        self.click(page, 2)
        self.assertTrue(self.capture.workflow.buttons[3].isEnabled())
        with patch.object(self.capture, 'send') as send:
            self.capture.new_capture()
        send.assert_not_called()
        self.assertIsNone(self.capture.session.result)
        self.assertFalse(self.capture.workflow.buttons[3].isEnabled())
        self.assertFalse(self.capture.workflow.buttons[4].isEnabled())
        self.assertFalse(self.window.workflow_navigation.navigate(self.capture, 4))
        self.assertIs(self.window.pages.currentWidget(), self.capture)

    def test_history_steps_remain_read_only_and_do_not_start_an_edit(self):
        self.fixture.generate(); model = self.page.model
        model.saved_plan = deepcopy(self.page.plan); model.capture_source = None
        review = self.window.review
        review.install_model(model, Path(self.fixture.fixture.temp.name)/'portable.region.zip')
        self.window.pages.setCurrentWidget(review)
        mask = review.canvas.work.copy(); plan = deepcopy(review.plan)
        edits = []; review.edit_requested.connect(edits.append)
        for step in (3, 4, 5, 2):
            self.click(review, step)
            self.assertEqual(review.workflow.current_step, step)
            self.assertFalse(review.canvas.editable)
            QTest.mouseClick(review.canvas, Qt.LeftButton, pos=review.canvas.rect().center())
        np.testing.assert_array_equal(review.canvas.work, mask)
        self.assertEqual(review.plan, plan); self.assertEqual(edits, [])
        self.assertFalse(review.workflow.buttons[0].isEnabled())

    def test_history_model_round_trip_returns_to_same_viewer(self):
        review = self.window.review
        model = self.page.model
        model.capture_source = model.reconstruction
        review.install_model(model, model.reconstruction)
        self.window.pages.setCurrentWidget(review)
        review.show_workflow_step(2)
        source = review.source
        from femto.app.capture_viewport import render_scene
        def slow_render(*args, **kwargs):
            # Rendering must survive several timer refreshes without the
            # hidden inspector/alert resizing the viewport underneath it.
            time.sleep(.2)
            return render_scene(*args, **kwargs)
        with patch('femto.app.qt_capture.read_review', return_value=self.capture_data()), \
             patch('femto.app.qt_capture.render_scene', side_effect=slow_render), \
             patch('femto.dataset.task_context.read_task', return_value=deepcopy(model.task_context)), \
             patch.object(review, 'load_source') as reload, \
             patch.object(self.capture, 'send') as send, \
             patch.object(review, 'show_workflow_step', wraps=review.show_workflow_step) as show:
            # Both the navigation strip and the footer must return to the
            # original historical page, including after a detour through 01.
            for via_footer in (False, True):
                self.click(review, 1)
                self.wait_runner(self.capture.review_task)
                self.assertIs(self.window.pages.currentWidget(), self.capture)
                self.assertEqual(self.capture.workflow.current_step, 1)
                deadline = time.monotonic() + 5
                while self.capture.viewport.image.isNull() and time.monotonic() < deadline:
                    QTest.qWait(20)
                self.assertFalse(self.capture.viewport.image.isNull())
                serial = self.capture._render_serial
                QTest.qWait(300)
                self.assertEqual(self.capture._render_serial, serial)
                self.assertFalse(self.capture.inspector_scroll.isVisible())
                self.assertFalse(self.capture.alert.isVisible())
                if via_footer:
                    self.capture.primary.click(); self.fixture.settle()
                else:
                    self.click(self.capture, 2)
                self.assertIs(self.window.pages.currentWidget(), review)
                self.assertEqual(review.workflow.current_step, 2)
                self.assertIs(review.model, model)
                self.assertEqual(review.source, source)
                self.assertFalse(review.canvas.editable)
                self.assertEqual(self.window.history_button.property('state'), 'active')
            self.assertEqual(show.call_count, 2)
            reload.assert_not_called(); send.assert_not_called()

    def test_first_entry_to_planning_still_requires_model_confirmation(self):
        self.install_capture(); self.window.pages.setCurrentWidget(self.capture)
        self.capture.scan_roi_quality = None; self.capture.refresh()
        self.assertIn('尚未核验', self.capture.guidance.text())
        with patch('femto.app.qt_capture.QMessageBox.question', return_value=QMessageBox.No) as confirm:
            self.click(self.capture, 3)
        confirm.assert_called_once()
        self.assertIs(self.window.pages.currentWidget(), self.capture)
        with patch('femto.app.qt_capture.QMessageBox.question', return_value=QMessageBox.Yes) as confirm:
            self.click(self.capture, 3)
        confirm.assert_called_once()
        self.assertIs(self.window.pages.currentWidget(), self.page)

    def test_active_or_paused_capture_cannot_jump_or_reset_its_reference(self):
        self.window.pages.setCurrentWidget(self.capture)
        record = dict(state='tracking', reference='synthetic-fixed-W0')
        self.capture.session = CaptureState(session_id='synthetic-active', active=True,
                                            state='tracking', count=3, record=record)
        with patch.object(self.capture, 'send') as send:
            for paused in (False, True):
                self.capture.session.paused = paused; self.capture.refresh()
                self.assertEqual(self.capture.workflow.current_step, 1)
                self.assertTrue(all(not button.isEnabled() for button in self.capture.workflow.buttons))
                self.assertFalse(self.window.workflow_navigation.navigate(self.capture, 0))
                self.assertIs(self.capture.session.record, record)
                self.assertEqual(self.capture.session.session_id, 'synthetic-active')
        send.assert_not_called()


if __name__ == '__main__':
    unittest.main()
