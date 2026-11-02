"""Clinician navigation and controls; synthetic frames, no device connection."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch

import numpy as np
from PySide6.QtCore import QEventLoop, QTimer, QPoint
from PySide6.QtWidgets import QApplication

from femto.app.qt_main import WorkstationWindow, apply_theme
from femto.app.qt_capture import CaptureState
from test_scan_roi import frame


class ClinicianWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        apply_theme(cls.app)

    def setUp(self):
        self.window = WorkstationWindow(connect_camera=False)
        self.page = self.window.capture
        self.window.resize(1120, 740)
        self.window.show()
        self.app.processEvents()
        self.addCleanup(self.close_window)

    def close_window(self):
        self.page.session.active = False
        self.window.planning.dirty = False
        self.window.close()
        self.app.processEvents()

    def install_frame(self, depth=None):
        initial = frame(depth)
        self.page.host.latest_frame = initial
        self.page.host.latest_received = time.monotonic()
        self.page.device_info['depth_range_m'] = [.25, 2.21]
        return initial

    def wait_review(self):
        loop = QEventLoop(); deadline = QTimer()
        deadline.setSingleShot(True); deadline.timeout.connect(loop.quit)
        self.page.review_task.finished.connect(loop.quit)
        try:
            deadline.start(5000)
            if self.page.review_task.busy: loop.exec()
        finally:
            deadline.stop(); self.page.review_task.finished.disconnect(loop.quit)
        self.assertFalse(self.page.review_task.busy)

    def test_confirmed_site_opens_range_tool_but_only_explicit_start_sends_full_frame(self):
        initial = self.install_frame()
        with patch.object(self.page, 'live_frame_valid', return_value=True), patch.object(self.page, 'send', return_value=True) as send:
            self.page.choose_scan_roi()
            self.assertEqual(self.page.view, 'preview')
            self.assertTrue(self.page.viewport.roi_enabled)
            self.assertFalse(self.page.primary.isEnabled())
            send.assert_not_called()
            self.page._roi_changed([.125, .125, .875, .875])
            self.assertTrue(self.page.primary.isEnabled())
            send.assert_not_called()
            self.page.primary.click()
            send.assert_called_once()
            self.assertEqual(send.call_args.args, ('local_start',))
            sent = send.call_args.kwargs
            self.assertEqual(sent['scan_roi']['rect_norm'], [.125, .125, .875, .875])
            self.assertFalse(sent['scan_roi']['raw_rgbd_cropped'])
            np.testing.assert_array_equal(sent['initial_frame'].color_bgr, initial.color_bgr)
            np.testing.assert_array_equal(sent['initial_frame'].depth_mm, initial.depth_mm)
            self.assertEqual(self.page.workflow.current_step, 1)

    def test_range_with_missing_depth_still_blocks_start(self):
        self.install_frame(np.zeros((64, 64), np.uint16))
        with patch.object(self.page, 'live_frame_valid', return_value=True), patch.object(self.page, 'send') as send:
            self.page.choose_scan_roi()
            self.page._roi_changed([.125, .125, .875, .875])
            self.assertFalse(self.page.primary.isEnabled())
            self.page.primary_action()
            send.assert_not_called()
            self.assertFalse(self.page.session.active)
            self.assertIn('有效深度', self.page.roi_message)

    def test_fixed_footer_keeps_finish_visible_and_original_admission_rules(self):
        self.page.session = CaptureState(active=True, count=1, state='tracking', record={
            'state':'tracking', 'received_monotonic':time.monotonic()})
        with patch.object(self.page, 'live_frame_valid', return_value=True):
            self.page.refresh(); self.app.processEvents()
            self.assertTrue(self.page.finish_button.isVisible())
            self.assertFalse(self.page.finish_button.isEnabled())
            self.page.session.count = 3
            self.page.refresh(); self.app.processEvents()
            self.assertTrue(self.page.finish_button.isEnabled())
            for button in (self.page.primary, self.page.finish_button, self.page.pause_button, self.page.cancel_button):
                origin = button.mapTo(self.window, QPoint(0, 0))
                self.assertTrue(self.window.rect().contains(origin))
                self.assertTrue(self.window.rect().contains(origin + QPoint(button.width()-1, button.height()-1)))
            self.assertEqual(self.page.primary.parentWidget(), self.page.finish_button.parentWidget())
            self.page.session.paused = True; self.page.refresh()
            self.assertFalse(self.page.finish_button.isEnabled())
            self.assertFalse(self.page.primary.isEnabled())
            self.page.session.paused = False
            self.page.session.record['received_monotonic'] = time.monotonic()-3
            self.page.refresh()
            self.assertFalse(self.page.finish_button.isEnabled())
            self.assertFalse(self.page.primary.isEnabled())

    def load_review(self, quality=None, preview=None):
        temp = TemporaryDirectory(); self.addCleanup(temp.cleanup)
        result = Path(temp.name); (result/'reconstruction').mkdir()
        points = np.array([[0., 0., .5], [.01, .01, .5]])
        data = dict(clouds={'final':(points, np.ones_like(points))}, surface=None,
                    quality={}, scan_roi_quality=quality, scan_roi_preview=preview)
        with patch('femto.app.qt_capture.read_review', return_value=data):
            self.assertTrue(self.page.restore_review_for_source(result/'reconstruction'))
            self.wait_review()
        return result

    def test_loaded_missing_map_is_shown_first_and_switching_views_keeps_evidence(self):
        quality = dict(status='checked', supported_pixels=3, pending_pixels=1, requested_pixels=4)
        self.load_review(quality, np.full((64, 64, 3), 120, np.uint8))
        self.assertEqual(self.page.view, 'scan_roi')
        self.assertTrue(self.page.views['scan_roi'].isChecked())
        self.assertEqual(self.page.workflow.current_step, 2)
        self.assertIn('缺失或待检查', self.page.guidance.text())
        self.page.views['3d'].click()
        self.assertEqual(self.page.view, 'final')
        self.page.views['scan_roi'].click()
        self.assertEqual(self.page.view, 'scan_roi')
        self.assertIs(self.page.scan_roi_quality, quality)
        self.page.planning_requested.disconnect(self.window.open_planning)
        requested = []; self.page.planning_requested.connect(requested.append)
        with patch('femto.app.qt_capture.QMessageBox.question', return_value=16384) as question:
            self.page.primary.click()
        question.assert_called_once()
        self.assertIn('1 个参考图像像素缺失', question.call_args.args[2])
        self.assertEqual(len(requested), 1)

    def test_unknown_scan_coverage_is_not_presented_as_pass(self):
        self.load_review()
        self.assertEqual(self.page.view, 'final')
        self.assertFalse(self.page.views['scan_roi'].isEnabled())
        self.assertIn('尚未核验', self.page.guidance.text())
        self.assertIsNone(self.page.scan_roi_quality)
        self.assertTrue(self.page._history_replay)

    def test_live_frame_absence_does_not_enable_range_on_site_confirmation(self):
        with patch.object(self.page, 'live_frame_valid', return_value=False), patch.object(self.page, 'send') as send:
            self.page.choose_scan_roi()
            self.assertFalse(self.page.viewport.roi_enabled)
            self.assertFalse(self.page.primary.isEnabled())
            self.assertFalse(self.page.scan_roi_button.isEnabled())
            send.assert_not_called()

    def test_planning_and_history_share_steps_without_claiming_capture_completed(self):
        from test_qt_planning import QtPlanningTests
        fixture = QtPlanningTests(); fixture.setUpClass(); fixture.setUp()
        try:
            fixture.mark()
            page = fixture.page
            self.assertEqual(page.workflow.current_step, 4)
            self.assertIn('05 / 06', page.step_heading.text())
            page.go_back()
            self.assertEqual(page.workflow.current_step, 3)
            self.assertEqual([label.text() for label in page.workflow.labels],
                             [label.text() for label in self.page.workflow.labels])
            self.window.review.install_model(fixture.model, fixture.model.reconstruction)
            self.window.pages.setCurrentWidget(self.window.review)
            self.assertEqual(self.window.review.workflow.current_step, 2)
            self.assertTrue(all('✓' not in label.text() for label in page.workflow.labels))
            self.assertFalse(self.window.review.canvas.editable)
        finally:
            fixture.page.dirty = False; fixture.doCleanups()

    def test_preparation_remains_reachable_with_inspector_scrolled(self):
        self.install_frame()
        with patch.object(self.page, 'live_frame_valid', return_value=True), patch.object(self.page, 'send') as send:
            self.page.refresh(); self.app.processEvents()
            self.assertFalse(self.page.details_panel.isVisible())
            self.assertFalse(self.page.count_label.isVisible())
            self.assertFalse(self.page.quality_button.isVisible())
            self.assertFalse(self.page.report_button.isVisible())
            # A collapsed parent hides descendants without setting their explicit hidden flag.
            self.assertTrue(self.page.details_toggle.isVisible())
            self.assertFalse(self.page.task_name.isVisible())
            self.assertFalse(hasattr(self.page,'body_records_button'))
            self.assertFalse(hasattr(self.window,'body_records_button'))
            self.assertTrue(self.page.task_name.text().strip())
            self.assertFalse(self.page.inspector_scroll.isHidden())
            scroll = self.page.inspector_scroll.verticalScrollBar()
            scroll.setValue(scroll.maximum())
            button = self.page.prepare_action
            self.assertTrue(button.isVisible())
            self.assertTrue(button.isEnabled())
            origin = button.mapTo(self.window, QPoint(0, 0))
            self.assertTrue(self.window.rect().contains(origin))
            self.assertTrue(self.window.rect().contains(origin + QPoint(button.width()-1, button.height()-1)))
            button.click()
            self.assertFalse(self.page.inspector_scroll.isHidden())
            self.assertTrue(self.page.viewport.roi_enabled)
            self.assertFalse(self.page.primary.isVisible())
            self.assertFalse(self.page.prepare_action.isEnabled())
            send.assert_not_called()
            self.page._roi_changed([.125, .125, .875, .875])
            self.assertTrue(self.page.primary.isVisible())
            self.assertTrue(self.page.primary.isEnabled())
            self.assertFalse(self.page.prepare_action.isVisible())
            send.assert_not_called()

    def test_preparation_reconnect_does_not_bypass_frame_or_roi_admission(self):
        self.install_frame()
        with patch.object(self.page, 'live_frame_valid', return_value=False), patch.object(self.page, 'send') as send:
            self.page.choose_scan_roi()
            self.assertEqual(self.page.prepare_action.text(), '连接相机')
            self.assertTrue(self.page.prepare_action.isEnabled())
            with patch.object(self.page, 'connect_camera') as connect:
                self.page.prepare_action.click()
            connect.assert_called_once()
            self.assertFalse(self.page.viewport.roi_enabled)
            self.assertFalse(self.page.primary.isEnabled())
            send.assert_not_called()
        with patch.object(self.page, 'live_frame_valid', return_value=True), patch.object(self.page, 'send') as send:
            self.page.refresh()
            self.assertEqual(self.page.prepare_action.text(), '框选扫描范围')
            self.page.prepare_action.click()
            self.assertTrue(self.page.viewport.roi_enabled)
            self.page.views['depth'].click()
            self.assertFalse(self.page.viewport.roi_enabled)
            self.assertTrue(self.page.prepare_action.isEnabled())
            self.page.prepare_action.click()
            self.assertEqual(self.page.view, 'preview')
            self.assertTrue(self.page.viewport.roi_enabled)
            send.assert_not_called()

    def test_decluttering_keeps_blocking_alert_and_quality_evidence_visible(self):
        self.page.session = CaptureState(active=True, count=3, state='tracking', record={
            'state':'tracking', 'received_monotonic':time.monotonic()-3})
        with patch.object(self.page, 'live_frame_valid', return_value=True):
            self.page.refresh(); self.app.processEvents()
            self.assertTrue(self.page.alert.isVisible())
            self.assertIn('位姿更新已过期', self.page.alert.text())
            self.assertFalse(self.page.guidance.isVisible())
            self.assertTrue(self.page.count_label.isVisible())
            self.assertTrue(self.page.quality_button.isVisible())
            self.assertFalse(self.page.primary.isEnabled())
            self.assertFalse(self.page.finish_button.isEnabled())
            self.page.session.record['received_monotonic'] = time.monotonic()
            self.page.refresh()
            self.assertFalse(self.page.alert.isVisible())
            self.assertTrue(self.page.guidance.isVisible())
            self.assertFalse(self.page.primary.isEnabled())  # Three accepted views reach the capture cap.
            self.assertTrue(self.page.finish_button.isEnabled())


if __name__ == '__main__':
    unittest.main()
