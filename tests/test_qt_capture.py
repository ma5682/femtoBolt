from __future__ import annotations

import os
import queue
from types import SimpleNamespace
import time
import threading
from pathlib import Path
import unittest
import uuid
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
    from PySide6.QtGui import QImage, QMouseEvent
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    from femto.app.capture_viewport import sample_aligned_frame_cloud
    from femto.app.qt_capture import CapturePage, CaptureState, CaptureViewport, read_review
    from femto.geometry.pose_geometry import Intrinsics
    HAVE_QT = True
except ModuleNotFoundError:
    HAVE_QT = False


@unittest.skipUnless(HAVE_QT, "PySide6 is required for the Qt workbench")
class CaptureStateTests(unittest.TestCase):
    def test_auto_finish_is_session_bound_and_does_not_claim_model_success(self):
        state = CaptureState(); state.begin(); sid = state.session_id
        state.state = 'tracking'; state.count = 3
        state.record = {'state': 'tracking', 'received_monotonic': time.monotonic()}
        self.assertFalse(state.accept('local_finishing', {'session_id': 'old', 'reason': 'accepted_view_limit'}))
        self.assertTrue(state.fresh_pose())
        self.assertTrue(state.accept('local_finishing', {'session_id': sid, 'reason': 'accepted_view_limit'}))
        self.assertTrue(state.active); self.assertTrue(state.finishing)
        self.assertEqual(state.finish_reason, 'accepted_view_limit')
        self.assertFalse(state.fresh_pose()); self.assertIsNone(state.result)
        state.accept('local_processing', {'session_id': sid})
        state.accept('local_error', {'session_id': sid, 'error': 'test_reconstruction_failed'})
        self.assertFalse(state.active); self.assertIsNone(state.result)

    def test_stage_check_keeps_capture_live_and_rejects_late_or_wrong_stage(self):
        state=CaptureState();state.begin();sid=state.session_id
        state.state='tracking'
        state.record={'state':'tracking','received_monotonic':time.monotonic()}
        self.assertTrue(state.accept('local_checkpoint_queued',{'session_id':sid}))
        self.assertTrue(state.accept('local_checkpoint_started',{'session_id':sid,'stage_id':'stage-one'}))
        self.assertTrue(state.active)
        self.assertTrue(state.fresh_pose())
        stage={'stage_id':'stage-one','session_id':sid,'status':'checked'}
        self.assertFalse(state.accept('local_checkpoint_ready',{'session_id':sid,'stage_id':'wrong','checkpoint':stage}))
        self.assertTrue(state.accept('local_checkpoint_ready',{'session_id':sid,'stage_id':'stage-one','checkpoint':stage}))
        self.assertIsNone(state.result)
        self.assertEqual(state.state,'tracking')
        self.assertFalse(state.checkpoint_busy)
        state.accept('local_finished',{'session_id':sid,'result':'D:/final'})
        self.assertFalse(state.accept('local_checkpoint_ready',{'session_id':sid,'stage_id':'stage-one','checkpoint':stage}))

    def test_reference_session_ignores_stale_events_and_requires_fresh_pose(self):
        state = CaptureState()
        state.begin()
        sid = state.session_id
        self.assertFalse(state.accept("local_pose", {"session_id": uuid.uuid4().hex,
                                                       "record": {"state": "tracking"},
                                                       "accepted": 99}))
        self.assertEqual(state.count, 0)
        state.accept("local_pose", {"session_id": sid,
                                     "record": {"state": "tracking",
                                                "received_monotonic": time.monotonic(),
                                                "local_workspace": {"within_limit": True}},
                                     "accepted": 1})
        self.assertTrue(state.fresh_pose())
        state.record["received_monotonic"] -= 3
        self.assertFalse(state.fresh_pose())

    def test_finish_event_publishes_result_and_closes_capture(self):
        state = CaptureState()
        state.begin()
        sid = state.session_id
        state.accept("local_finished", {"session_id": sid, "source": "D:/session",
                                         "result": "D:/output/reconstruction"})
        self.assertFalse(state.active)
        self.assertEqual(state.state, "ready")
        self.assertEqual(state.result.name, "reconstruction")

    def test_same_session_pose_events_cannot_regress_count_or_frame(self):
        state = CaptureState()
        state.begin()
        sid = state.session_id
        self.assertTrue(state.accept("local_pose", {
            "session_id": sid,
            "record": {"state": "tracking", "frame_id": 8,
                        "received_monotonic": 8.0},
            "accepted": 3,
        }))
        self.assertFalse(state.accept("local_pose", {
            "session_id": sid,
            "record": {"state": "uncertain", "frame_id": 7,
                        "received_monotonic": 7.0},
            "accepted": 2,
        }))
        self.assertEqual(state.count, 3)
        self.assertEqual(state.record["frame_id"], 8)
        self.assertEqual(state.state, "tracking")

    def test_malformed_completion_does_not_claim_success(self):
        state = CaptureState()
        state.begin()
        sid = state.session_id
        self.assertTrue(state.accept("local_finished", {"session_id": sid}))
        self.assertFalse(state.active)
        self.assertEqual(state.state, "lost")
        self.assertEqual(state.failure["error"], "missing_result")


@unittest.skipUnless(HAVE_QT, "PySide6 is required for the Qt workbench")
class CapturePageTests(unittest.TestCase):
    def test_three_views_block_more_requests_and_preserve_historical_counts(self):
        page = CapturePage(connect_camera=False)
        try:
            page.session.begin(); page.session.state = 'tracking'; page.session.count = 3
            page.session.record = {'state': 'tracking', 'received_monotonic': time.monotonic()}
            with patch.object(page, 'live_frame_valid', return_value=True), patch.object(page, 'send') as send:
                page.refresh()
                self.assertEqual(page.count_label.text(), '3 / 3 帧')
                self.assertFalse(page.primary.isEnabled())
                self.assertTrue(all(segment.property('accepted') for segment in page.frame_steps))
                page.request_view(); send.assert_not_called()
                page.handle_event('local_finishing', {'session_id': page.session.session_id,
                                                     'reason': 'accepted_view_limit'})
                page.refresh()
                self.assertFalse(page.finish_button.isEnabled())
                self.assertTrue(page.cancel_button.isEnabled())
                self.assertIn('自动结束', page._capture_message(page.session, True, False, False)[1])
            page.session.active = False; page.session.finishing = False
            page.session.count = 6; page.refresh()
            self.assertEqual(page.count_label.text(), '6 帧')
        finally:
            page.session.active = False
            page.shutdown()

    def test_finish_available_after_one_valid_frame_but_not_before(self):
        page=CapturePage(connect_camera=False)
        try:
            page.session.begin();page.session.state='tracking'
            page.session.record={'state':'tracking','received_monotonic':time.monotonic()}
            with patch.object(page,'live_frame_valid',return_value=True),patch.object(page,'send',return_value=True) as send:
                page.refresh();self.assertFalse(page.finish_button.isEnabled())
                page.finish();send.assert_not_called()
                page.session.count=1;page.refresh();self.assertTrue(page.finish_button.isEnabled())
                page.finish();send.assert_called_once_with('local_finish',True)
        finally:
            page.session.active=False;page.shutdown()

    def test_stage_request_rejection_does_not_claim_busy_without_backend_ack(self):
        page=CapturePage(connect_camera=False)
        try:
            page.session.begin();page.session.count=3
            with patch.object(page.session,'fresh_pose',return_value=True), \
                    patch.object(page,'send',return_value=False) as send:
                page.check_current()
                send.assert_called_once_with('local_check')
                self.assertFalse(page.session.checkpoint_busy)
        finally:
            page.session.active=False
            page.shutdown()

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_preview_mode_is_safe_without_hardware(self):
        page = CapturePage(connect_camera=False)
        self.assertFalse(page.camera_ready)
        self.assertTrue(page.can_leave())
        self.assertEqual("开始采集", page.primary.text())
        self.assertTrue(page.shutdown())

    def test_live_frame_cloud_previews_before_reference_and_clears_stale_camera(self):
        page = CapturePage(connect_camera=False)
        try:
            intrinsics = Intrinsics(12, 8, 10., 10., 5.5, 3.5)
            distortion = dict(model='BROWN_CONRADY_K6',
                              k1=0., k2=0., p1=0., p2=0.,
                              k3=0., k4=0., k5=0., k6=0.)
            frame = SimpleNamespace(depth_mm=np.full((8, 12), 1000., dtype=np.float32),
                                    color_bgr=np.full((8, 12, 3), [0, 0, 255], dtype=np.uint8))
            page.handle_event('camera_ready', dict(_intrinsics=intrinsics,
                _rgb_distortion=distortion, depth_range_m=[.25, 2.21]))
            with patch.object(page.host, 'is_running', return_value=True):
                with page.host.frame_lock:
                    page.host.latest_frame = frame
                    page.host.latest_received = time.monotonic()
                page.set_view('frame_cloud')
                page.poll()
                deadline = time.monotonic() + 5
                while page.render_task.busy and time.monotonic() < deadline:
                    QTest.qWait(10)
                self.assertFalse(page.render_task.busy)
                self.assertFalse(page.viewport.image.isNull())
                self.assertEqual(page.session.count, 0)
                self.assertFalse(page.cloud_choice.isVisible())
                self.assertIn('未配准、未累计', page.preview_note.text())
                page.session.session_id = 'preview-fit'
                page.preview_mailbox = queue.Queue()
                page.preview_mailbox.put(dict(session_id='preview-fit',
                    points=np.array([[0., 0., 1.], [.02, .01, 1.]]),
                    colors=np.full((2, 3), 255, dtype=np.uint8),
                    initial_points=np.array([[0., 0., 1.]]),
                    initial_colors=np.full((1, 3), 255, dtype=np.uint8)))
                page.poll()
                self.assertIsNotNone(page.center)
                page.set_view('3d')
                self.assertEqual(page.view, 'live')
                self.assertIn('本次采集累计点云', page.preview_note.text())
                page.set_view('frame_cloud')
                with page.host.frame_lock:
                    page.host.latest_received -= 3
                page.poll()
                self.assertTrue(page.viewport.image.isNull())
        finally:
            self.assertTrue(page.shutdown())

    def test_live_frame_cloud_uses_aligned_rgb_k6_and_depth_range(self):
        intrinsics = Intrinsics(3, 1, 2., 2., 1., 0.)
        distortion = dict(model='BROWN_CONRADY_K6',
                          k1=0., k2=0., p1=0., p2=0.,
                          k3=0., k4=0., k5=0., k6=0.)
        points, colors = sample_aligned_frame_cloud(
            np.array([[1000., 0., 3000.]], dtype=np.float32),
            np.array([[[0, 0, 255], [0, 0, 0], [255, 0, 0]]], dtype=np.uint8),
            intrinsics, distortion, 250., 2210.)
        np.testing.assert_allclose(points, [[-.5, 0., 1.]])
        np.testing.assert_allclose(colors, [[1., 0., 0.]])

    def test_unreadable_review_cannot_enter_planning(self):
        page = CapturePage(connect_camera=False)
        try:
            page.session.result = Path('C:/history/complete_demo')
            page.session.state = 'review_load_error'
            page.clouds['final'] = (np.asarray([[0., 0., .5]]), np.ones((1, 3)))
            page.refresh()
            self.assertFalse(page.primary.isEnabled())
            requested = []
            page.planning_requested.connect(requested.append)
            with patch.object(page, 'start') as start:
                page.primary_action()
            self.assertFalse(requested)
            start.assert_not_called()
            page.session.state = 'ready'
            page.refresh()
            self.assertTrue(page.primary.isEnabled())
            with patch('femto.app.qt_capture.QMessageBox.question',return_value=16384):
                page.primary_action()
            self.assertEqual(len(requested), 1)
        finally:
            self.assertTrue(page.shutdown())

    def test_saved_fused_cloud_is_available_as_historical_accumulation(self):
        with TemporaryDirectory() as temp:
            result = Path(temp)
            cloud = SimpleNamespace(points=np.array([[0., 0., .5], [.01, 0., .5]]),
                                    colors=np.array([[1., 0., 0.], [0., 1., 0.]]))
            with patch('femto.geometry.fusion.read_geometry_safe', return_value=cloud), \
                 patch('femto.app.qt_capture.read_review_surface', return_value=None), \
                 patch('femto.local.local_processing.result_quality', return_value={}):
                review = read_review(result, threading.Event(), lambda _: None)
            self.assertIs(review['clouds']['accumulated'], review['clouds']['final'])
            self.assertEqual(len(review['clouds']['accumulated'][0]), 2)

    def test_workstation_preview_constructs_without_camera(self):
        # Exercise the real shell as a clinician would first open it.  This
        # catches state-banner ordering errors that page-only tests cannot see.
        from femto.app.qt_main import WorkstationWindow
        window = WorkstationWindow(connect_camera=False)
        try:
            self.assertIs(window.pages.currentWidget(), window.capture)
            self.assertTrue(window.capture.can_leave())
            self.assertTrue(window.planning.can_leave())
            self.assertEqual("开始采集", window.capture.primary.text())
        finally:
            window.close()
            self.app.processEvents()

    def test_page_does_not_accept_finish_without_live_frame(self):
        page = CapturePage(connect_camera=False)
        page.camera_ready = True
        page.session.begin()
        page.session.record = {"state": "tracking", "received_monotonic": time.monotonic(),
                               "local_workspace": {"within_limit": True}}
        page.session.state = "tracking"
        page.session.count = 2
        page.finish(True)
        self.assertTrue(page.session.active)
        self.assertEqual(page.session.notice, "camera_not_ready")
        self.assertTrue(page.shutdown())

    def test_paused_capture_cannot_finish_until_resumed(self):
        """A paused acquisition must keep the clinician in the recoverable flow."""
        page = CapturePage(connect_camera=False)
        page.camera_ready = True
        page.session.begin()
        page.session.record = {"state": "tracking", "received_monotonic": time.monotonic(),
                               "local_workspace": {"within_limit": True}}
        page.session.state = "tracking"
        page.session.count = 3
        page.session.paused = True
        page.refresh()
        self.assertFalse(page.finish_button.isEnabled())
        self.assertIn("继续补采", page.finish_button.toolTip())
        page.finish(True)
        self.assertTrue(page.session.active)
        self.assertEqual(page.session.notice, "paused_finish")
        self.assertTrue(page.shutdown())

    def test_dead_camera_cannot_start_from_a_fresh_but_stale_frame(self):
        page = CapturePage(connect_camera=False)
        try:
            page.camera_ready = True
            page.host.camera_thread = SimpleNamespace(is_alive=lambda: False)
            with page.host.frame_lock:
                page.host.latest_frame = object()
                page.host.latest_received = time.monotonic()
            self.assertFalse(page.live_frame_valid())
            page.start()
            self.assertFalse(page.session.active)
            self.assertEqual(page.session.state, "waiting")
            self.assertTrue(page.host.command_queue.empty())
        finally:
            self.assertTrue(page.shutdown())

    def test_camera_stop_before_local_start_releases_session_for_retry(self):
        page = CapturePage(connect_camera=False)
        try:
            page.session.begin()
            page.handle_event("camera_stopped", None)
            self.assertFalse(page.session.active)
            self.assertEqual(page.session.state, "camera_error")
            self.assertEqual(page.session.failure["error"], "camera_not_ready")
            self.assertEqual("重新开始采集", page.primary.text())
        finally:
            self.assertTrue(page.shutdown())

    def test_finish_requires_live_camera_even_when_pose_is_fresh(self):
        page = CapturePage(connect_camera=False)
        try:
            page.session.begin()
            page.session.state = "tracking"
            page.session.count = 3
            page.session.record = {
                "state": "tracking",
                "received_monotonic": time.monotonic(),
                "local_workspace": {"within_limit": True},
            }
            page.camera_ready = True
            page.host.camera_thread = SimpleNamespace(is_alive=lambda: False)
            page.finish(True)
            self.assertTrue(page.session.active)
            self.assertEqual(page.session.notice, "camera_not_ready")
        finally:
            self.assertTrue(page.shutdown())

    def test_restore_historical_result_populates_three_d_review(self):
        """Returning from historical planning keeps its model visible in capture.

        A historical reconstruction is not a live CaptureState session.  The
        capture page still needs to bind the result root and run the same
        bounded review loader so the ``final`` point cloud is available after
        the planning page's "上一步" action.
        """
        page = CapturePage(connect_camera=False)
        try:
            with TemporaryDirectory() as temp:
                result = Path(temp) / "complete_demo_20260926"
                reconstruction = result / "reconstruction"
                reconstruction.mkdir(parents=True)
                review = {
                    "clouds": {
                        "accumulated": (
                            np.asarray([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]], dtype=float),
                            np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float),
                        ),
                        "final": (
                            np.asarray([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]], dtype=float),
                            np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float),
                        )
                    },
                    "surface": None,
                    "quality": {"accepted": True},
                }
                with patch("femto.app.qt_capture.read_review", return_value=review):
                    page.restore_review_for_source(reconstruction)
                    deadline = time.monotonic() + 5
                    while page.review_task.busy and time.monotonic() < deadline:
                        QTest.qWait(10)
                    self.assertFalse(page.review_task.busy)
                self.assertEqual(page.session.result, result.resolve())
                self.assertFalse(page.session.active)
                self.assertEqual(page.session.state, "ready")
                self.assertIn("final", page.clouds)
                self.assertEqual(page.view, "final")
                self.assertTrue(page.viewport.is_cloud)
                page.cloud_choice.setCurrentIndex(0)
                self.assertEqual(page.cloud_choice.currentData(), 'accumulated')
                self.assertEqual(page.view, 'accumulated')
                self.assertEqual(len(page.clouds[page.view][0]), 2)
                self.assertIn('历史累计点云', page.preview_note.text())
                self.assertIn('历史三维记录', page.device_label.text())
                self.assertIn('历史采集记录', page.capture_heading.text())
                self.assertFalse(page.views['frame_cloud'].isEnabled())
        finally:
            self.assertTrue(page.shutdown())

    def test_reference_check_first_still_frames_all_historical_clouds(self):
        page = CapturePage(connect_camera=False)
        try:
            with TemporaryDirectory() as temp:
                result = Path(temp)
                (result / 'reconstruction').mkdir()
                points = np.array([[.1, .2, .3], [.3, .4, .5]])
                colors = np.ones((2, 3))
                review = dict(
                    clouds={key: (points, colors) for key in
                            ('accumulated', 'initial', 'final')},
                    surface=None, quality={'accepted': True},
                    scan_roi_quality={'status': 'checked', 'supported_pixels': 90,
                                      'pending_pixels': 10, 'requested_pixels': 100},
                    scan_roi_preview=np.zeros((10, 10, 3), dtype=np.uint8))
                with patch('femto.app.qt_capture.read_review', return_value=review):
                    page.restore_review_for_source(result)
                    deadline = time.monotonic() + 5
                    while page.review_task.busy and time.monotonic() < deadline:
                        QTest.qWait(10)
                self.assertFalse(page.review_task.busy)
                self.assertEqual(page.view, 'scan_roi')
                np.testing.assert_allclose(page.center, [.2, .3, .4])
                self.assertAlmostEqual(page.span, .2)
                # Exercise the actual renderer: populated clouds must not take
                # its empty-grid branch (which also requires a valid center).
                from femto.app.capture_viewport import render_scene
                with patch('femto.app.qt_capture.render_scene', wraps=render_scene) as rendered:
                    for index, key in enumerate(('accumulated', 'initial', 'final')):
                        page.cloud_choice.setCurrentIndex(index)
                        page.set_view('3d')
                        deadline = time.monotonic() + 5
                        while (page._render_dirty or page.render_task.busy) and time.monotonic() < deadline:
                            page.poll()
                            QTest.qWait(10)
                        self.assertEqual(page.view, key)
                        self.assertFalse(page.viewport.image.isNull())
                        self.assertGreater(len(rendered.call_args.args[2]), 0)
                        np.testing.assert_allclose(rendered.call_args.kwargs['center'], [.2, .3, .4])
        finally:
            self.assertTrue(page.shutdown())

    def test_restore_review_for_region_project_is_noop(self):
        """Standalone editable plans do not masquerade as capture results."""
        page = CapturePage(connect_camera=False)
        try:
            with TemporaryDirectory() as temp:
                source = Path(temp) / "planning.region.zip"
                source.write_bytes(b"placeholder")
                self.assertFalse(page.restore_review_for_source(source))
                self.assertIsNone(page.session.result)
                self.assertEqual(page.clouds, {})
        finally:
            self.assertTrue(page.shutdown())


@unittest.skipUnless(HAVE_QT, "PySide6 is required for the Qt workbench")
class CaptureViewportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_rgb_drag_selects_normalized_scan_range(self):
        viewport = CaptureViewport()
        viewport.resize(400, 300)
        viewport.show()
        try:
            viewport.set_image(QImage(200, 100, QImage.Format_RGB888))
            viewport.roi_enabled = True
            selected = []
            viewport.roi_changed.connect(selected.append)
            QTest.mousePress(viewport, Qt.LeftButton, pos=QPoint(80, 90))
            QTest.mouseMove(viewport, QPoint(320, 210))
            QTest.mouseRelease(viewport, Qt.LeftButton, pos=QPoint(320, 210))
            self.assertEqual(len(selected), 1)
            np.testing.assert_allclose(selected[0], [.2, .2, .8, .8], atol=.01)
        finally:
            viewport.close()

    def test_xyz_viewport_ignores_right_drag_pan(self):
        viewport = CaptureViewport()
        viewport.is_cloud = True
        press = QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(10, 10),
                            Qt.MouseButton.RightButton, Qt.MouseButton.RightButton,
                            Qt.KeyboardModifier.NoModifier)
        move = QMouseEvent(QEvent.Type.MouseMove, QPointF(100, 80),
                           Qt.MouseButton.RightButton, Qt.MouseButton.RightButton,
                           Qt.KeyboardModifier.NoModifier)
        release = QMouseEvent(QEvent.Type.MouseButtonRelease, QPointF(100, 80),
                              Qt.MouseButton.RightButton, Qt.MouseButton.NoButton,
                              Qt.KeyboardModifier.NoModifier)
        viewport.mousePressEvent(press)
        viewport.mouseMoveEvent(move)
        viewport.mouseReleaseEvent(release)
        np.testing.assert_array_equal(viewport.pan, [0., 0.])


if __name__ == "__main__":
    unittest.main()
