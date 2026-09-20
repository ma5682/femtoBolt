"""Tracking solver retry, immutable origin, failure evidence and visible UI state.

These software fixtures do not assert physical camera accuracy.
"""
import gc
import json
from pathlib import Path
import queue
import tempfile
import time
import tkinter as tk
import unittest
from unittest import mock
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageTk

from local_capture import LocalCapture
from scene_live_tracker import LiveTracker, PairEstimator, PairEstimationError
from test_segmented_capture import make_frame, make_camera


def translation(x):
    result = np.eye(4)
    result[0, 3] = x
    return result


class PairRetryTests(unittest.TestCase):
    def pair(self):
        pair = PairEstimator.__new__(PairEstimator)
        pair.jacobian, pair.option = object(), object()
        return pair

    def test_retry_solves_both_directions_and_returns_solution_not_seed(self):
        pair = self.pair()
        seed, solution = translation(-.03), translation(-.045)
        answers = [(True, translation(.31), None), (True, np.eye(4), None),
                   (True, solution, None), (True, np.linalg.inv(solution), None)]
        evaluation = SimpleNamespace(fitness=.95, inlier_rmse=.005)
        with mock.patch('scene_live_tracker.o3d.pipelines.odometry.compute_rgbd_odometry', side_effect=answers) as solve, \
                mock.patch('scene_live_tracker.o3d.pipelines.registration.evaluate_registration', return_value=evaluation):
            result, metrics = pair.estimate(('a', 'k', 'ca'), ('b', 'k', 'cb'), initial_transform=seed)
        np.testing.assert_array_equal(result, solution)
        self.assertFalse(np.array_equal(result, seed))
        self.assertEqual(solve.call_count, 4)
        for call, expected in zip(solve.call_args_list, [np.eye(4), np.eye(4), seed, np.linalg.inv(seed)]):
            np.testing.assert_array_equal(call.args[3], expected)
        self.assertEqual(metrics['initialization'], 'previous_valid_step')
        self.assertTrue(metrics['cycle']['passed'])
        failed = metrics['attempts'][0]
        self.assertFalse(failed['passed'])
        self.assertEqual(failed['reason'], 'pair_quality')
        self.assertIn('cycle_consistency', failed['metrics']['rejected_checks'])
        self.assertIn('translation_step', failed['metrics']['rejected_checks'])
        json.dumps(metrics, allow_nan=False)

    def test_bad_or_absent_seed_cannot_create_unchecked_pose_or_extra_attempt(self):
        pair = self.pair()
        for seed in (None, np.eye(4), np.full((4, 4), np.nan), translation(.3)):
            with self.subTest(seed=str(seed)), mock.patch.object(pair, '_estimate_once',
                    side_effect=PairEstimationError('pair_quality', {'rejected_checks': ['fitness']})) as solve:
                with self.assertRaises(PairEstimationError) as caught:
                    pair.estimate(None, None, initial_transform=seed)
                self.assertEqual(solve.call_count, 1)
                self.assertEqual(len(caught.exception.attempts), 1)

    def test_retry_failure_keeps_failed_metrics_and_has_only_two_attempts(self):
        pair = self.pair()
        with mock.patch.object(pair, '_estimate_once', side_effect=PairEstimationError(
                'pair_quality', {'rejected_checks': ['cycle_consistency']})) as solve:
            with self.assertRaises(PairEstimationError) as caught:
                pair.estimate(None, None, initial_transform=translation(.03))
        self.assertEqual(solve.call_count, 2)
        self.assertEqual(caught.exception.metrics['rejected_checks'], ['cycle_consistency'])

    def test_original_quality_thresholds_still_reject_and_json_metrics_are_finite(self):
        pair = self.pair()
        rotation = np.eye(4)
        angle = np.radians(26)
        rotation[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        cases = [(translation(.251), .9, .01, 'translation_step'),
                 (rotation, .9, .01, 'rotation_step'), (np.eye(4), .149, .01, 'fitness'),
                 (np.eye(4), .9, .061, 'rmse'), (np.eye(4), np.nan, .01, 'fitness'),
                 (np.eye(4), .9, np.inf, 'rmse')]
        for transform, fitness, rmse, check in cases:
            with self.subTest(check=check, fitness=fitness, rmse=rmse), \
                    mock.patch('scene_live_tracker.o3d.pipelines.odometry.compute_rgbd_odometry',
                        side_effect=[(True, transform, None), (True, np.linalg.inv(transform), None)]), \
                    mock.patch('scene_live_tracker.o3d.pipelines.registration.evaluate_registration',
                        return_value=SimpleNamespace(fitness=fitness, inlier_rmse=rmse)):
                with self.assertRaises(PairEstimationError) as caught:
                    pair.estimate((None, None, None), (None, None, None))
                self.assertIn(check, caught.exception.metrics['rejected_checks'])
                json.dumps(caught.exception.metrics, allow_nan=False)


class TrackerContinuityTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.
        self.estimator = mock.Mock()
        self.estimator.prepare.side_effect = lambda frame: frame.frame_index
        self.tracker = LiveTracker(self.estimator, 250, 2210, clock=lambda: self.now)
        self.quality = mock.patch('scene_live_tracker.frame_quality', return_value=dict(usable=True))
        self.quality.start()
        self.addCleanup(self.quality.stop)

    def process(self, frame_id, seconds):
        self.now = seconds
        return self.tracker.process(make_frame([500]*10, frame_id+1), frame_id, seconds)

    def test_short_rejection_keeps_origin_and_previous_valid_reference(self):
        first, next_step = translation(-.03), translation(-.01)
        self.estimator.estimate.side_effect = [(first, {}), PairEstimationError('pair_quality'), (next_step, {})]
        self.process(7, 100.)
        self.process(8, 100.2)
        bad = self.process(9, 100.4)
        self.assertIsNone(bad['T_WC'])
        self.assertEqual(bad['state'], 'uncertain')
        good = self.process(10, 100.6)
        self.assertEqual(good['state'], 'tracking')
        self.assertEqual(good['reference_frame_id'], 7)
        self.assertAlmostEqual(good['xyz_mm'][0], 40.)
        calls = self.estimator.estimate.call_args_list
        self.assertIsNone(calls[0].kwargs['initial_transform'])
        np.testing.assert_array_equal(calls[1].kwargs['initial_transform'], first)
        np.testing.assert_array_equal(calls[2].kwargs['initial_transform'], first)
        self.assertEqual(calls[2].args, (9, 11))  # frame 9's failed preparation was not retained

    def test_five_rejected_frames_still_latch_loss_not_five_solver_attempts(self):
        self.process(0, 100.)
        self.estimator.estimate.side_effect = PairEstimationError('pair_quality',
            {'rejected_checks': ['cycle_consistency']}, [{'passed': False}]*2)
        for frame_id in range(1, 6):
            record = self.process(frame_id, 100.+frame_id*.1)
            self.assertEqual(record['state'], 'lost' if frame_id==5 else 'uncertain')
            self.assertIsNone(record['T_WC'])
        self.assertEqual(record['loss_context']['consecutive_failures'], 5)
        self.assertEqual(record['loss_context']['current_pair_failure']['rejected_checks'], ['cycle_consistency'])
        self.estimator.estimate.side_effect = None
        self.estimator.estimate.return_value = (np.eye(4), {})
        count = self.estimator.estimate.call_count
        self.assertEqual(self.process(6, 100.6)['state'], 'lost')
        self.assertEqual(self.estimator.estimate.call_count, count)
        self.assertEqual(self.tracker.reference_id, 0)

    def test_two_second_gap_keeps_specific_preceding_failure(self):
        self.process(0, 100.)
        self.estimator.estimate.side_effect = PairEstimationError('pair_quality',
            {'rejected_checks': ['cycle_consistency']})
        self.process(1, 100.5)
        record = self.process(2, 102.01)
        self.assertEqual(record['state'], 'lost')
        self.assertEqual(record['rejection_reason'], 'tracking_gap')
        self.assertEqual(record['loss_context']['preceding_rejection']['reason'], 'pair_quality')
        self.assertEqual(record['loss_context']['last_valid_frame_id'], 0)
        self.assertEqual(self.estimator.estimate.call_count, 1)

    def test_solver_result_arriving_after_age_limit_cannot_advance_pose(self):
        self.process(0, 100.)
        def slow(*args, **kwargs):
            self.now = 102.3
            return translation(-.04), {'initialization': 'previous_valid_step'}
        self.estimator.estimate.side_effect = slow
        record = self.process(1, 100.1)
        self.assertEqual(record['state'], 'lost')
        self.assertEqual(record['rejection_reason'], 'processing_stale')
        self.assertIsNone(record['T_WC'])
        self.assertIsNone(self.tracker.previous_step)
        self.assertGreater(record['loss_context']['frame_age_s'], 2.)
        np.testing.assert_array_equal(self.tracker.pose, np.eye(4))
        self.assertEqual(self.tracker.last_valid_frame_id, 0)


class CaptureFailureTests(unittest.TestCase):
    def setUp(self):
        gc.collect()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.events = queue.Queue()
        self.root = Path(self.temp.name)

    def wait_for(self, kind):
        deadline = time.monotonic()+6
        while time.monotonic()<deadline:
            event, payload = self.events.get(timeout=6)
            if event==kind:return payload
        self.fail('Missing event '+kind)

    def capture(self):
        capture = LocalCapture(make_camera('wide_detail', 10), self.root/'data', self.root/'out', self.events)
        self.addCleanup(lambda: (capture.finish(False), capture.thread.join(6)))
        capture.submit(make_frame([500]*10, 1))
        self.wait_for('local_pose')
        return capture

    @mock.patch('complete_demo.run_demo')
    @mock.patch('scene_live_tracker.frame_quality', return_value=dict(usable=True))
    @mock.patch('local_capture.PairEstimator')
    def test_no_input_failure_has_persistent_reason_and_no_fake_frame(self, pair, quality, demo):
        capture = self.capture()
        payload = self.wait_for('local_error')
        capture.thread.join(5)
        self.assertFalse(capture.thread.is_alive())
        demo.assert_not_called()
        report = json.loads((capture.session/'derived/local_capture_report.json').read_text())
        failure = json.loads((capture.session/'derived/local_failure.json').read_text())
        self.assertEqual(report['frame_count'], 1)
        self.assertEqual(report['termination']['reason'], 'no_frames')
        self.assertEqual(failure['termination'], report['termination'])
        self.assertEqual(failure['accepted_frames'], [0])
        self.assertIsNone(failure['termination']['frame_id'])
        self.assertEqual(payload['failure'], failure)
        self.assertFalse(failure['successful_model_published'])
        row = json.loads(capture.scan.frames_log.read_text().splitlines()[0])
        self.assertTrue((capture.session/row['rgb']).exists())
        self.assertIsNone(json.loads((capture.session/row['pose']).read_text())['T_WC'])

    @mock.patch('complete_demo.run_demo')
    @mock.patch('scene_live_tracker.frame_quality', return_value=dict(usable=True))
    @mock.patch('local_capture.PairEstimator')
    def test_device_error_is_not_overwritten_by_shutdown_cancellation(self, pair, quality, demo):
        capture = self.capture()
        capture.abort('camera_error', 'device read failed')
        capture.finish(False)
        payload = self.wait_for('local_error')
        capture.thread.join(5)
        self.assertEqual(payload['error'], 'camera_error')
        self.assertEqual(payload['failure']['device_error'], 'device read failed')
        self.assertEqual(json.loads((capture.session/'derived/local_capture_report.json').read_text())['state'], 'error')
        demo.assert_not_called()

    @mock.patch('scene_live_tracker.frame_quality', return_value=dict(usable=True))
    @mock.patch('local_capture.PairEstimator')
    def test_report_separates_retry_from_rejected_frames_and_records_stage_timing(self, pair, quality):
        pair.return_value.estimate.return_value = (translation(-.03),
            {'initialization': 'previous_valid_step', 'attempts': [{'passed': False}, {'passed': True}]})
        capture = self.capture()
        capture.submit(make_frame([500]*10, 2))
        self.wait_for('local_pose')
        capture.finish(False)
        self.wait_for('local_cancelled')
        capture.thread.join(5)
        report = json.loads((capture.session/'derived/local_capture_report.json').read_text())
        self.assertEqual(report['retried_pair_count'], 1)
        self.assertEqual(report['recovered_pair_count'], 1)
        self.assertEqual(report['rejected_frame_count'], 0)
        for key in ('queue_wait', 'raw_write', 'pose_processing'):
            self.assertGreaterEqual(report['timing_ms'][key]['mean'], 0.)
        self.assertEqual(report['max_gap_s'], 2.)

    def test_sdk_read_error_is_forwarded_to_the_active_local_session(self):
        import femto_control_app as control
        from test_temporal_gui import TemporalGuiTests
        app = TemporalGuiTests().bare_app()
        camera = make_camera('wide_detail', 10)
        camera.start.return_value = camera
        camera.read.side_effect = [make_frame([500]*10, 1), RuntimeError('USB read fixture failure')]
        app.command_queue.put(('local_start', None))
        local = mock.Mock()
        local.thread.is_alive.return_value = True
        with mock.patch.object(control, 'OrbbecAlignedCamera', return_value=camera) as factory, \
                mock.patch('local_capture.LocalCapture', return_value=local):
            app._camera_worker()
        factory.assert_called_once()
        local.abort.assert_called_once_with('camera_error', 'USB read fixture failure')
        self.assertTrue(any(kind=='camera_error' for kind, _ in app.event_queue.queue))

    @mock.patch('femto_control_app.save_scan_frame', side_effect=OSError('raw storage fixture failure'))
    @mock.patch('local_capture.PairEstimator')
    def test_raw_write_failure_is_reported_as_error_without_a_successful_pose(self, pair, save):
        capture = LocalCapture(make_camera('wide_detail', 10), self.root/'data', self.root/'out', self.events)
        self.addCleanup(lambda: (capture.finish(False), capture.thread.join(6)))
        capture.submit(make_frame([500]*10, 1))
        payload = self.wait_for('local_error')
        capture.thread.join(5)
        self.assertEqual(payload['error'], 'raw storage fixture failure')
        report = json.loads((capture.session/'derived/local_capture_report.json').read_text())
        self.assertEqual(report['state'], 'error')
        self.assertEqual(report['error'], payload['error'])
        self.assertEqual(report['valid_pose_count'], 0)
        self.assertEqual(report['accepted_frames'], [])

    def test_finalization_updates_only_the_actual_session_root_marker(self):
        import femto_control_app as control
        protected = self.root/'unrelated_live_data'
        protected.mkdir()
        marker = protected/'latest_scene.txt'
        marker.write_bytes(b'existing-user-session')
        with mock.patch.object(LocalCapture, '_run'):
            capture = LocalCapture(make_camera('wide_detail', 10), self.root/'isolated', self.root/'out', self.events)
            capture.thread.join(5)
        with mock.patch.object(control, 'DATASETS_ROOT', protected):
            control.save_scan_frame(capture.scan, make_frame([500]*10, 1), lossless=True)
            control.finalize_scan(capture.scan, capture.camera)
        self.assertEqual(marker.read_bytes(), b'existing-user-session')
        self.assertEqual((capture.session.parent/'latest_scene.txt').read_text(encoding='utf-8'), str(capture.session))


class TrackingStatusUITests(unittest.TestCase):
    def setUp(self):
        gc.collect()
        import femto_control_app as control
        with mock.patch.object(control.FemtoControlApp, '_start_camera'), \
                mock.patch.object(control.FemtoControlApp, '_load_latest_paths'):
            self.root = tk.Tk()
            self.app = control.FemtoControlApp(self.root)
        self.root.geometry('1120x720')
        self.panel = self.app.local_panel
        self.app.camera_ready = True
        self.panel.start()
        self.addCleanup(self.close)

    def close(self):
        for timer in self.root.tk.call('after', 'info'):self.root.after_cancel(timer)
        self.root.destroy()
        del self.panel, self.app, self.root
        gc.collect()

    def pose_event(self, state='tracking', **extra):
        record = dict(state=state, reference_frame_id=0, received_monotonic=time.monotonic(),
                      xyz_mm=[10., 20., 30.], pitch_yaw_roll_deg=[1.,2.,3.], latency_ms=120.)
        record.update(extra)
        self.panel.event('local_pose', dict(session_id=self.panel.session_id, source='fixture', record=record, accepted=3))
        self.panel.tick()

    def test_invalid_pose_pause_and_recovery_remain_in_same_session(self):
        self.pose_event()
        session = self.panel.session_id
        self.assertIn('+10.0', self.panel.pose.get())
        self.panel.pause()
        self.assertIn('暂停', self.panel.hint.get())
        self.pose_event('uncertain', rejection_reason='pair_quality', pair_failure={'rejected_checks': ['cycle_consistency']})
        self.assertNotIn('+10.0', self.panel.pose.get())
        self.assertIn('前后向', self.panel.hint.get())
        self.assertEqual(str(self.panel.buttons['add']['state']), 'disabled')
        self.pose_event()
        self.assertEqual(self.panel.session_id, session)
        self.assertTrue(self.panel.paused)
        self.panel.pause()
        self.assertEqual(str(self.panel.buttons['add']['state']), 'normal')

    def test_relocalizing_hides_pose_and_recovery_clears_old_notice(self):
        self.pose_event();self.panel.count=3;self.panel.notice='duplicate';self.panel.notice_until=time.monotonic()+5
        self.pose_event('relocalizing',continuity=dict(remaining_window_s=12.),status_updated_monotonic=time.monotonic())
        self.assertNotIn('+10.0',self.panel.pose.get());self.assertIn('重定位',self.panel.hint.get())
        self.assertEqual(str(self.panel.buttons['add']['state']),'disabled')
        self.assertEqual(str(self.panel.buttons['finish']['state']),'disabled')
        self.pose_event(continuity=dict(relocalized=True))
        self.assertEqual(self.panel.notice,'relocalized');self.assertIn('重新停稳',self.panel.hint.get())
        self.app.language='en';self.panel.refresh_status();self.assertIn('Relocalized',self.panel.hint.get())

    def test_expired_display_is_invalid_without_claiming_session_ended(self):
        self.pose_event(received_monotonic=time.monotonic()-3)
        self.assertNotIn('+10.0', self.panel.pose.get())
        self.assertIn('过期', self.panel.hint.get())
        self.assertTrue(self.panel.active)
        self.assertEqual(str(self.panel.buttons['add']['state']), 'disabled')

    def test_insufficient_views_keep_session_open_until_three_are_accepted(self):
        panel = self.panel
        session = panel.session_id
        for count in (0, 1, 2):
            panel.count = count
            panel.refresh()
            self.assertEqual(str(panel.buttons['finish']['state']), 'disabled')
            queued = self.app.command_queue.qsize()
            panel.finish(True)
            self.assertEqual(self.app.command_queue.qsize(), queued)
            self.assertTrue(panel.active)
            self.assertFalse(panel.finishing)
            self.assertEqual(panel.session_id, session)
            self.assertEqual(panel.notice, 'need_more_views')
        self.pose_event()
        self.assertEqual(str(panel.buttons['finish']['state']), 'normal')
        panel.finish(True)
        self.assertTrue(panel.finishing)
        command, payload = self.app.command_queue.get_nowait()
        while not self.app.command_queue.empty():
            command, payload = self.app.command_queue.get_nowait()
        self.assertEqual((command, payload), ('local_finish', dict(session_id=session, value=True)))

    def test_cancel_remains_available_before_three_views(self):
        self.panel.count = 1
        self.panel.refresh()
        self.assertEqual(str(self.panel.buttons['cancel']['state']), 'normal')
        self.panel.finish(False)
        self.assertTrue(self.panel.finishing)
        self.assertEqual(list(self.app.command_queue.queue)[-1],
                         ('local_finish', dict(session_id=self.panel.session_id, value=False)))

    def test_every_terminal_error_keeps_old_pose_cleared_on_later_ticks(self):
        for error in ('calibration_changed', 'final_pose_invalid', 'disk write failed'):
            with self.subTest(error=error):
                if not self.panel.active:self.panel.start()
                self.pose_event()
                self.assertIn('+10.0', self.panel.pose.get())
                self.panel.event('local_error', dict(session_id=self.panel.session_id,
                    source='fixture', error=error))
                for _ in range(2):
                    self.panel.tick()
                    self.assertNotIn('+10.0', self.panel.pose.get())
                    self.assertEqual(self.panel.age.get(), self.panel.text('age_failed'))
                self.assertFalse(self.panel.active)

    def test_loss_reason_stays_visible_with_large_image_in_both_languages(self):
        self.pose_event()
        image = ImageTk.PhotoImage(Image.new('RGB', (2000, 1200), '#102030'))
        self.panel.photo = image
        self.panel.preview.configure(image=image)
        context = dict(reason='tracking_gap', preceding_rejection=dict(reason='pair_quality',
                       pair_failure={'rejected_checks': ['cycle_consistency']}))
        self.panel.notice, self.panel.notice_until = 'duplicate', time.monotonic()+3
        self.panel.event('local_error', dict(session_id=self.panel.session_id, source='fixture',
            error='tracking_lost', failure=dict(termination=context)))
        for language, message in [('zh', '跟踪已中断'), ('en', 'Tracking interrupted')]:
            self.app.language = language
            self.panel.refresh();self.panel.tick()
            self.root.update();self.root.update()
            self.assertIn(message, self.panel.hint.get())
            self.assertNotIn('+10.0', self.panel.pose.get())
            self.assertEqual(self.panel.hint_label.cget('bg'), '#622e2c')
            label = self.panel.hint_label
            self.assertTrue(label.winfo_ismapped())
            self.assertGreater(label.winfo_height(), 30)
            self.assertLess(label.winfo_rooty()+label.winfo_height(), self.root.winfo_rooty()+720)
            for key in ('start', 'add', 'pause', 'finish', 'cancel', 'view', 'report'):
                button = self.panel.buttons[key]
                self.assertGreaterEqual(button.winfo_height(), button.winfo_reqheight(), key)
                self.assertGreaterEqual(button.winfo_width(), button.winfo_reqwidth(), key)
                self.assertLessEqual(button.winfo_rooty()+button.winfo_height(), self.root.winfo_rooty()+720)
        self.assertFalse(self.panel.active)
        for key in ('add','pause','finish','view'):
            self.assertEqual(str(self.panel.buttons[key]['state']), 'disabled')
        self.assertEqual(str(self.panel.buttons['report']['state']), 'normal')
        self.assertEqual(str(self.panel.buttons['start']['state']), 'normal')
        previous = self.panel.session_id
        self.panel.start()
        self.assertNotEqual(previous, self.panel.session_id)
        self.assertIsNone(self.panel.failure)
        self.assertNotIn('Tracking interrupted', self.panel.hint.get())


if __name__=='__main__':
    unittest.main()
