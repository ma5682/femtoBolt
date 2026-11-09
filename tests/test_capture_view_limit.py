"""New captures stop on three durable admissions, never on three raw frames."""
import json
from pathlib import Path
import queue
from tempfile import TemporaryDirectory
import threading
import time
import unittest
from unittest import mock

import numpy as np
import open3d as o3d

from femto.capture.local_capture import LocalCapture
from femto.capture.local_capture_policy import MAX_ACCEPTED_VIEWS
from femto.capture.scan_session import save_scan_frame
from femto.geometry.pose_geometry import Intrinsics
from femto.tracking.scene_live_tracker import PreparedRGBD
from test_segmented_capture import make_camera, make_frame


class CaptureViewLimitTests(unittest.TestCase):
    def wait_record(self, capture, count):
        deadline = time.monotonic()+5
        while len(capture.records) < count and capture.thread.is_alive() and time.monotonic() < deadline:
            time.sleep(.005)
        self.assertEqual(len(capture.records), count)

    def exercise(self, mode, *, region_first=False):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            camera = make_camera('wide_detail', 100)
            camera.intrinsics = Intrinsics(100, 100, 100., 100., 49.5, 49.5)
            intrinsic = o3d.camera.PinholeCameraIntrinsic(100, 100, 100., 100., 49.5, 49.5)
            positions = {1: 0., 2: 0., 3: .012, 4: .012, 5: .024, 6: .024}
            third_writing, release_write = threading.Event(), threading.Event()
            rebuilding, release_rebuild = threading.Event(), threading.Event()
            def prepare(frame):
                return PreparedRGBD(None, intrinsic, None, frame.color_bgr,
                    frame.depth_mm, features=positions[frame.frame_index])
            def estimate(previous, current, **kwargs):
                transform = np.eye(4)
                transform[0, 3] = previous.features-current.features
                return transform, {}
            def save(scan, frame, **kwargs):
                if frame.frame_index == 6:
                    third_writing.set()
                    if not release_write.wait(4):
                        raise RuntimeError('Test did not release raw writer')
                    if mode == 'write_failure':
                        raise OSError('test disk full')
                return save_scan_frame(scan, frame, **kwargs)
            def rebuild():
                rebuilding.set()
                if not release_rebuild.wait(4):
                    raise RuntimeError('Test did not release reconstruction')
                return root/'models'/'result', {'frame_count': 3}
            with mock.patch('femto.capture.local_capture.PairEstimator') as estimator, \
                    mock.patch('femto.tracking.scene_live_tracker.frame_quality', return_value={'usable': True}), \
                    mock.patch('femto.capture.scan_session.save_scan_frame', side_effect=save), \
                    mock.patch('femto.pipeline.local_offline.LocalOfflineJob') as job, \
                    mock.patch('femto.pipeline.local_offline.publish_result') as publish:
                estimator.return_value.prepare.side_effect = prepare
                estimator.return_value.estimate.side_effect = estimate
                job.return_value.run.side_effect = rebuild
                job.return_value.report = {'state': 'complete'}
                events = queue.Queue()
                region_first = region_first or mode == 'region_selection'
                capture = LocalCapture(camera, root/'data', root/'models', events,region_first=region_first)
                def submit(index):
                    frame = make_frame([500]*10000, index)
                    frame.depth_mm = frame.depth_mm.reshape(100, 100)
                    frame.color_bgr = frame.color_bgr.reshape(100, 100, 3)
                    capture.submit(frame)
                try:
                    submit(1); self.wait_record(capture, 1)
                    capture.request(); submit(2); self.wait_record(capture, 2)
                    self.assertEqual(len(capture.accepted), 1)  # Duplicate does not count.
                    self.assertEqual(capture.records[-1]['selection']['state'], 'duplicate')
                    capture.request(); submit(3); self.wait_record(capture, 3)
                    self.assertFalse(capture.closing.is_set())  # Three raw frames are insufficient.
                    self.assertEqual(len(capture.accepted), 1)  # Moving, not yet still.
                    submit(4); self.wait_record(capture, 4)
                    self.assertEqual(len(capture.accepted), 2)
                    capture.request(); submit(5); self.wait_record(capture, 5)
                    submit(6)
                    self.assertTrue(third_writing.wait(2))
                    self.assertEqual(len(capture.accepted), 2)
                    self.assertFalse(capture.closing.is_set())
                    job.assert_not_called()  # Cannot finish before the third matching raw receipt.
                    if mode == 'cancel_before_commit':
                        capture.finish(False)
                    release_write.set()
                    if region_first and mode in ('complete', 'region_selection', 'cancel_during_rebuild'):
                        self.wait_record(capture,6)
                        self.assertTrue(capture.awaiting_region.wait(3))
                        self.assertTrue(capture.closing.is_set())
                        self.assertFalse(capture.request())
                        job.assert_not_called()
                        mask=np.zeros((100,100),np.uint8);mask[30:70,30:70]=1
                        self.assertTrue(capture.select_region(mask,40))
                    if mode in ('complete', 'region_selection', 'cancel_during_rebuild'):
                        self.assertTrue(rebuilding.wait(3))
                        self.assertTrue(capture.closing.is_set())
                        self.assertEqual(len(capture.accepted), MAX_ACCEPTED_VIEWS)
                        self.assertFalse(capture.request())
                        submit(6)  # Late camera submissions do not create another raw record.
                        if mode == 'cancel_during_rebuild':
                            capture.finish(False)
                    release_rebuild.set()
                    capture.thread.join(5)
                    self.assertFalse(capture.thread.is_alive())
                    if mode in ('complete','region_selection'):
                        job.assert_called_once(); publish.assert_called_once()
                        self.assertEqual(capture.finish_reason, 'accepted_view_limit')
                        self.assertEqual([r['frame_id'] for r in capture.accepted], [0, 3, 5])
                        self.assertEqual(capture.scan.frame_count, 6)
                        posed = capture.session/'derived/local_posed'
                        rows = [json.loads(s) for s in (posed/'frames.jsonl').read_text().splitlines()]
                        self.assertEqual([r['frame'] for r in rows], [0, 3, 5])
                        manifest = json.loads((posed/'manifest.json').read_text())
                        self.assertEqual(manifest['frame_count'], 3)
                        self.assertEqual(manifest['local_capture']['capture_limit']['maximum_accepted_views'], 3)
                        self.assertTrue(manifest['local_capture']['capture_limit']['automatically_finish'])
                        self.assertEqual(manifest['local_capture']['finish_reason'], 'accepted_view_limit')
                        report = json.loads((capture.session/'derived/local_capture_report.json').read_text())
                        self.assertEqual(report['accepted_frames'], [0, 3, 5])
                        self.assertEqual(report['frame_count'], 6)
                        self.assertFalse(report['capture_limit']['raw_tracking_frames_limited'])
                        for row in rows:
                            pose = json.loads((posed/row['pose']).read_text())
                            self.assertEqual(pose['reference_frame_id'], 0)
                            self.assertEqual(pose['capture_timestamp_us'], row['camera_timestamp_us'])
                        raw_rows = [json.loads(s) for s in capture.scan.frames_log.read_text().splitlines()]
                        for row in raw_rows:
                            raw_pose = json.loads((capture.session/row['pose']).read_text())
                            self.assertFalse(raw_pose['pose_valid'])
                            self.assertIsNone(raw_pose['T_WC'])
                        kinds = [kind for kind, payload in events.queue]
                        self.assertEqual(kinds.count('local_finishing'), 1)
                        self.assertLess(kinds.index('local_finishing'), kinds.index('local_processing'))
                        self.assertIn('local_finished', kinds)
                        if mode == 'region_selection':
                            self.assertLess(kinds.index('local_region_ready'),kinds.index('local_processing'))
                            self.assertTrue(Path(job.call_args.kwargs['region_request']).is_file())
                    else:
                        publish.assert_not_called()
                        self.assertFalse(capture.successful_model_published)
                        self.assertFalse(list((root/'models').glob('latest*.txt')))
                        if mode != 'cancel_during_rebuild':
                            job.assert_not_called()
                            self.assertEqual(len(capture.accepted), 2)
                finally:
                    release_write.set(); release_rebuild.set()
                    capture.finish(False); capture.thread.join(6)

    def test_third_durable_view_auto_finishes_and_exports_exactly_three(self):
        self.exercise('complete')

    def test_three_views_wait_for_painted_region_before_any_offline_mesh(self):
        self.exercise('region_selection')

    def test_third_raw_write_failure_never_starts_reconstruction(self):
        self.exercise('write_failure')

    def test_cancel_before_third_raw_commit_prevents_admission_and_reconstruction(self):
        self.exercise('cancel_before_commit')

    def test_cancel_after_auto_finish_still_prevents_publication(self):
        self.exercise('cancel_during_rebuild')

    def test_region_first_write_failure_does_not_enter_selection_or_publish(self):
        self.exercise('write_failure', region_first=True)

    def test_region_first_cancel_before_raw_commit_prevents_third_admission(self):
        self.exercise('cancel_before_commit', region_first=True)

    def test_region_first_auto_finish_and_reconstruction_remain_cancellable(self):
        self.exercise('cancel_during_rebuild', region_first=True)

    def test_fourth_view_is_rejected_even_with_pending_request(self):
        capture = LocalCapture.__new__(LocalCapture)
        for region_first in (False, True):
            with self.subTest(region_first=region_first):
                capture.region_first = region_first
                capture.accepted = [{}, {}, {}]
                capture.emit = mock.Mock()
                record = {'state': 'tracking', 'T_WC': np.eye(4).tolist()}
                self.assertFalse(capture._select(record, time.monotonic()))
                self.assertEqual(record['selection']['reason'], 'accepted_view_limit')
                self.assertFalse(capture.request())


if __name__ == '__main__':
    unittest.main()
