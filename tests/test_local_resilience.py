"""Meaningful raw-write, preprocessing and interrupted-prefix invariants."""
import copy
import hashlib
import json
from pathlib import Path
import queue
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock

import cv2
import numpy as np

from femto.capture.local_capture import LocalCapture
from femto.pipeline.local_interrupted import assess, prepare
from femto.capture.local_raw_writer import RawFrameWriter
from femto.dataset.local_reference import reference_provenance
from femto.tracking.local_tracking_diagnostics import classify, quality_text, summarize
from femto.tracking.scene_live_tracker import PairEstimator
from femto.io_utils import imread_unicode as _read_image
from femto.geometry.rgbd_frames import prepare_rgbd as _prepare_rgbd
from test_segmented_capture import make_camera, make_frame


class RawReceiptTests(unittest.TestCase):
    def test_slow_raw_receipt_cannot_admit_a_stale_pose_or_claim_it_as_cutoff(self):
        from femto.app.femto_control_app import save_scan_frame
        import gc
        gc.collect()
        def slow(scan, frame, **kwargs):
            time.sleep(2.05)
            save_scan_frame(scan, frame, **kwargs)
        with tempfile.TemporaryDirectory() as td, mock.patch('femto.capture.local_capture.PairEstimator'), \
                mock.patch('femto.tracking.scene_live_tracker.frame_quality', return_value=dict(usable=True)), \
                mock.patch('femto.capture.scan_session.save_scan_frame', side_effect=slow):
            capture=LocalCapture(make_camera('wide_detail', 10), Path(td)/'data', Path(td)/'out', queue.Queue())
            capture.submit(make_frame([500]*10, 1))
            capture.thread.join(6)
            self.assertFalse(capture.thread.is_alive())
            self.assertEqual(capture.accepted, [])
            self.assertEqual(capture.records[0]['rejection_reason'], 'raw_write_stale')
            self.assertIsNone(capture.records[0]['T_WC'])
            self.assertIsNone(capture.records[0]['loss_context']['last_valid_frame_id'])
            self.assertFalse(capture.records[0]['continuity']['current_frame_verified'])

    def test_pose_and_save_overlap_but_admission_waits_for_matching_raw_receipt(self):
        from femto.app.femto_control_app import save_scan_frame
        import gc
        gc.collect()
        release, estimating = threading.Event(), threading.Event()
        def save(scan, frame, **kwargs):
            if not release.wait(3):
                raise RuntimeError('Test did not release writer')
            save_scan_frame(scan, frame, **kwargs)
        with tempfile.TemporaryDirectory() as td, mock.patch('femto.capture.local_capture.PairEstimator') as pair, \
                mock.patch('femto.tracking.scene_live_tracker.frame_quality', return_value=dict(usable=True)), \
                mock.patch('femto.capture.scan_session.save_scan_frame', side_effect=save):
            pair.return_value.prepare.side_effect = lambda frame: estimating.set()
            capture = LocalCapture(make_camera('wide_detail', 10), Path(td)/'data', Path(td)/'out', queue.Queue())
            try:
                capture.submit(make_frame([500]*10, 17))
                self.assertTrue(estimating.wait(2), 'Pose must run while raw writer is waiting')
                self.assertEqual(capture.accepted, [])
                self.assertEqual(capture.records, [])
                release.set()
                deadline = time.monotonic()+3
                while not capture.records and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertEqual(len(capture.accepted), 1)
                raw = json.loads(capture.scan.frames_log.read_text().splitlines()[0])
                record = capture.accepted[0]
                self.assertEqual(record['raw_write_receipt']['capture_timestamp_us'], raw['camera_timestamp_us'])
                self.assertEqual(record['frame_id'], raw['frame'])
                self.assertIsNone(json.loads((capture.session/raw['pose']).read_text())['T_WC'])
            finally:
                release.set(); capture.finish(False); capture.thread.join(5)
            self.assertFalse(capture.thread.is_alive())

    def test_writer_failure_and_receipt_mismatch_cannot_be_ignored(self):
        scan = SimpleNamespace(frame_count=0)
        frame = make_frame([500]*10, 1)
        writer = RawFrameWriter(mock.Mock(side_effect=OSError('disk full')))
        try:
            writer.submit(scan, frame, 0)
            with self.assertRaises(OSError):
                writer.commit({})
        finally:
            writer.close()
        def save(scan, frame, **kwargs):
            scan.frame_count += 1
        writer = RawFrameWriter(save)
        try:
            writer.submit(scan, frame, 0)
            with self.assertRaises(RuntimeError):
                writer.submit(scan, frame, 1)
            with self.assertRaisesRegex(ValueError, 'timestamp mismatch'):
                writer.commit(dict(frame_id=0, capture_timestamp_us=2, color_timestamp_us=1, depth_timestamp_us=1))
        finally:
            writer.close()

    def test_local_copy_omits_unused_native_depth_but_owns_aligned_rgbd(self):
        from femto.app.femto_control_app import copy_frame
        frame = make_frame([500]*10, 1)
        frame.native_depth_mm = np.ones((4, 4))
        snapshot = copy_frame(frame, include_native_depth=False)
        self.assertIsNone(snapshot.native_depth_mm)
        frame.depth_mm[:] = 0; frame.color_bgr[:] = 0
        self.assertTrue(np.all(snapshot.depth_mm == 500))
        self.assertIsNot(snapshot.color_bgr, frame.color_bgr)


class PreparationAndDiagnosticsTests(unittest.TestCase):
    def test_cached_depth_mapping_matches_original_calibrated_preparation(self):
        from femto.pipeline.synthetic_scene_demo import generate_dataset
        with tempfile.TemporaryDirectory() as td:
            source = generate_dataset(Path(td)/'data', frames=4, width=516, height=292)
            row = json.loads((source/'frames.jsonl').read_text().splitlines()[0])
            frame = make_frame([500], 1)
            frame.color_bgr = _read_image(source/row['rgb'], cv2.IMREAD_COLOR)
            frame.depth_mm = _read_image(source/row['depth'], cv2.IMREAD_UNCHANGED)
            before = frame.depth_mm.copy()
            estimator = PairEstimator(source, 500, 3860)
            mx, my, k, _, _ = estimator.maps
            old, intrinsic, color, depth = _prepare_rgbd(frame.color_bgr, frame.depth_mm, mx, my, k, .25, .5, 3.86)
            current = estimator.prepare(frame)
            np.testing.assert_array_equal(current.depth_mm, depth)
            np.testing.assert_array_equal(current.color_bgr, color)
            np.testing.assert_array_equal(np.asarray(current.rgbd.depth), np.asarray(old.depth))
            np.testing.assert_array_equal(current.intrinsic.intrinsic_matrix, intrinsic.intrinsic_matrix)
            np.testing.assert_array_equal(frame.depth_mm, before)
            self.assertIs(current.intrinsic, estimator.prepare(frame).intrinsic)

    def test_diagnostics_separate_low_depth_sync_processing_and_unknown_scene_causes(self):
        r = dict(frame_id=2, state='uncertain', rejection_reason='pair_quality',
                 quality=dict(valid_depth_fraction=.19, sharpness=15),
                 image_diagnostics=dict(weak_texture_suspected=True), color_timestamp_us=100, depth_timestamp_us=6001,
                 timing_ms=dict(raw_write=40, queue_wait=80, pose_processing=700),
                 pair_failure=dict(rejected_checks=['cycle_consistency']), consecutive_failures=2, latency_ms=820)
        d = classify(r)
        self.assertIn('low_depth', d['observed'])
        self.assertIn('weak_texture_suspected', d['observed'])
        self.assertEqual(d['dynamic_object'], 'not_established')
        self.assertEqual(d['dominant_measured_stage'], 'pose_processing')
        self.assertEqual(d['synchronization_delta_us'], 5901)
        summary = summarize([r], 2, 4)
        self.assertEqual(summary['pose_latency_ms']['p50'], 820)
        self.assertEqual(summary['valid_pose_fps'], 0)
        self.assertEqual(summary['dropped_input_frames'], 4)
        self.assertIn('insufficient depth', quality_text(r, 'en'))
        json.dumps(summary, allow_nan=False)


class InterruptedPrefixTests(unittest.TestCase):
    def fixture(self, root):
        from femto.app.femto_control_app import save_scan_frame, finalize_scan
        with mock.patch.object(LocalCapture, '_run'):
            capture = LocalCapture(make_camera('wide_detail', 10), root/'data', root/'out', queue.Queue())
            capture.thread.join()
        records = []
        for i in range(4):
            frame = make_frame([500]*10, i+1)
            save_scan_frame(capture.scan, frame, lossless=True)
            pose = np.eye(4); pose[0, 3] = i*.011
            records.append(dict(frame_id=i, camera_frame_index=frame.frame_index, state='tracking' if i < 3 else 'lost',
                T_WC=pose.tolist() if i < 3 else None, reference_frame_id=0, session_id=capture.session_id,
                capture_timestamp_us=i+1, color_timestamp_us=i+1, depth_timestamp_us=i+1, unit='metre',
                received_monotonic=10+i*.4, latency_ms=20, selected_for_fusion=i < 3))
        finalize_scan(capture.scan, capture.camera)
        capture.records=records; capture.accepted=records[:3]
        capture.reference_provenance=reference_provenance(capture.session_id, records[0], capture.calibration)
        from femto.io_utils import write_json_strict as write_json
        write_json(capture.session/'derived/local_reference.json', capture.reference_provenance)
        (capture.session/'derived/live_tracking.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records), encoding='utf-8')
        tracker=SimpleNamespace(reference_id=0, lost=True, max_gap_s=2, loss_context=dict(reason='tracking_gap', last_valid_frame_id=2))
        capture._write_tracking_report(tracker)
        return capture

    def test_prefix_is_independent_fixed_w0_and_does_not_rewrite_raw_or_latest(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); capture=self.fixture(root)
            before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'data').rglob('*') if p.is_file()}
            dataset, decision=prepare(capture.session, root/'prefix')
            self.assertTrue(decision['eligible']); self.assertEqual(decision['last_trusted_frame_id'], 2)
            manifest=json.loads((dataset/'manifest.json').read_text('utf-8'))
            self.assertEqual(manifest['local_capture']['capture_scope'], 'interrupted_prefix')
            self.assertEqual(manifest['local_capture']['reference_provenance'], capture.reference_provenance)
            self.assertEqual(manifest['frame_count'], 3)
            for p, digest in before.items():
                self.assertEqual(hashlib.sha256(Path(p).read_bytes()).hexdigest(), digest)

    def test_two_views_late_frame_and_cross_reference_are_not_eligible(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);capture=self.fixture(root)
            path=capture.session/'derived/local_capture_report.json'
            report=json.loads(path.read_text())
            report['accepted_frames']=[0, 1];path.write_text(json.dumps(report))
            self.assertIn('need_three_distinct_verified_views', assess(capture.session)[0]['reasons'])
            report['accepted_frames']=[0, 1, 2];report['termination']['last_valid_frame_id']=1
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, 'after the last trustworthy'):
                assess(capture.session)
            report['termination']['last_valid_frame_id']=2;path.write_text(json.dumps(report))
            log=capture.session/'derived/live_tracking.jsonl'
            records=[json.loads(s) for s in log.read_text().splitlines()]
            records[1]['reference_frame_id']=99
            log.write_text(''.join(json.dumps(r)+'\n' for r in records))
            with self.assertRaisesRegex(ValueError, 'different or invalid W0'):
                assess(capture.session)
