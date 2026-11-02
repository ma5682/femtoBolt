"""Software fixtures for session isolation, reference provenance and review loading."""
import hashlib
import gc
import json
from pathlib import Path
import queue
import tempfile
import threading
import time
import unittest
from unittest import mock
import uuid

import numpy as np
import open3d as o3d

from femto.app.qt_capture import read_review
import test_scene_annotation_origin as origin
from femto.capture.local_capture import LocalCapture
from femto.dataset.local_reference import calibration_identity, validate_reference_provenance
from femto.annotation.scene_annotations import AnnotationStore
from test_segmented_capture import make_frame, make_camera


class ReferenceProvenanceTests(unittest.TestCase):
    def setUp(self):
        gc.collect()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def captured(self):
        with mock.patch.object(LocalCapture, '_run'):
            capture = LocalCapture(make_camera('wide_detail', 10), self.root/'data', self.root/'out', queue.Queue())
            capture.thread.join()
        from femto.app.femto_control_app import save_scan_frame, finalize_scan
        for i in range(3):
            save_scan_frame(capture.scan, make_frame([500]*10, i+1), lossless=True)
        finalize_scan(capture.scan, capture.camera)
        capture.accepted = [dict(frame_id=i, T_WC=np.eye(4).tolist(), reference_frame_id=0,
                                 capture_timestamp_us=i+1) for i in range(3)]
        return capture

    def test_export_carries_same_reference_and_calibration_without_raw_writes(self):
        capture = self.captured()
        original = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in capture.session.rglob('*') if path.is_file()}
        posed = capture._export_poses()
        manifest = json.loads((posed/'manifest.json').read_text(encoding='utf-8'))
        proof = manifest['local_capture']['reference_provenance']
        self.assertEqual(proof['calibration'], calibration_identity(capture.session))
        self.assertEqual(proof['calibration'], calibration_identity(posed))
        self.assertEqual(proof['reference_frame_id'], 0)
        rows = [json.loads(line) for line in (posed/'frames.jsonl').read_text().splitlines()]
        for row in rows:
            pose = json.loads((posed/row['pose']).read_text())
            self.assertEqual(pose['reference_provenance'], proof)
            self.assertEqual(pose['capture_timestamp_us'], row['camera_timestamp_us'])
        validate_reference_provenance(proof, posed, rows[0], json.loads((posed/rows[0]['pose']).read_text()))
        for path, sha in original.items():
            self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(), sha)

    def test_export_rejects_wrong_frame_timestamp_reference_and_changed_calibration_before_writing(self):
        capture = self.captured()
        for key, wrong in (('capture_timestamp_us', 999), ('reference_frame_id', 1), ('frame_id', 0)):
            with self.subTest(key=key):
                previous = capture.accepted[1][key]
                capture.accepted[1][key] = wrong
                with self.assertRaises(ValueError):
                    capture._export_poses()
                self.assertFalse((capture.session/'derived/local_posed').exists())
                capture.accepted[1][key] = previous
        calibration = capture.session/'calibration/color_intrinsic.json'
        calibration.write_bytes(calibration.read_bytes() + b'\n')
        with self.assertRaisesRegex(ValueError, 'calibration_changed'):
            capture._export_poses()
        self.assertFalse((capture.session/'derived/local_posed').exists())

    def test_first_valid_frame_proof_and_performance_statistics_are_actual_records(self):
        events = queue.Queue()
        with mock.patch('femto.capture.local_capture.PairEstimator'), mock.patch('femto.tracking.scene_live_tracker.frame_quality',
                side_effect=[dict(usable=False, state='low_texture'), dict(usable=True)]):
            capture = LocalCapture(make_camera('wide_detail', 10), self.root/'data', self.root/'out', events)
            try:
                capture.submit(make_frame([500]*10, 1))
                received = []
                deadline = time.monotonic() + 5
                while len(received) < 2 and time.monotonic() < deadline:
                    kind, payload = events.get(timeout=5)
                    if kind == 'local_pose':
                        received.append(payload)
                        if len(received) == 1:
                            # Receipt order is the invariant here. The latest-only
                            # live mailbox may intentionally replace unread input.
                            capture.submit(make_frame([500]*10, 2))
                self.assertEqual(len(received), 2)
            finally:
                capture.finish(False)
                capture.thread.join(5)
        self.assertFalse(capture.thread.is_alive())
        proof = json.loads((capture.session/'derived/local_reference.json').read_text())
        report = json.loads((capture.session/'derived/local_capture_report.json').read_text())
        self.assertEqual(proof['reference_frame_id'], 1)
        self.assertEqual(proof['capture_timestamp_us'], 2)
        self.assertEqual(report['reference_provenance'], proof)
        self.assertEqual(report['accepted_frames'], [1])
        self.assertEqual(report['valid_pose_count'], 1)
        self.assertEqual(report['rejected_frame_count'], 1)
        self.assertEqual(report['rejection_counts'], dict(low_texture=1))
        self.assertAlmostEqual(report['processed_fps'], 2/report['duration_s'])
        self.assertAlmostEqual(report['latency_mean_ms'], np.mean(report['latency_ms']))
        self.assertEqual(report['tracking_lost_events'], 0)

    def test_model_origin_proof_checks_actual_calibration_and_saved_binding(self):
        capture = self.captured()
        posed = capture._export_poses()
        binding = json.loads((posed/'manifest.json').read_text())['local_capture']
        tracking = capture.session/'derived/local_capture_report.json'
        origin.write_json(tracking, dict(schema='femto-local-tracking-v1',
            session_id=capture.session_id, reference_frame_id=0, accepted_frames=[0,1,2],
            coordinate_frame='world_rdf', unit='metre', transform_direction='T_W0_Ct',
            pose_source='online_rgbd_research_grade', reference_provenance=capture.reference_provenance))
        source = self.root/'model/mesh.ply'
        origin.write_json(source.parent/'local_geometry.json', dict(**binding,
            coordinate_frame='world_rdf', unit='metre', transform_direction='T_W0_Ct', support={}))
        origin.write_json(source.parent/'reconstruction_report.json', dict(
            outputs=dict(mesh=dict(ply=str(source))), source_dataset=str(posed), local_capture=binding))
        store = AnnotationStore(origin.triangle(), source)
        self.assertEqual(store.origin_reference['kind'], 'initial_color_camera')
        self.assertEqual(store.origin_reference['calibration_sha256'], capture.calibration['sha256'])
        saved = store.save(self.root/'saved')
        changed = posed/'calibration/color_intrinsic.json'
        changed.write_bytes(changed.read_bytes() + b'\n')
        other = AnnotationStore(store.mesh, source)
        self.assertEqual(other.origin_reference['provenance'], 'unknown')
        with self.assertRaises(ValueError):
            other.load(saved)




class CameraDispatchTests(unittest.TestCase):
    def setUp(self):
        gc.collect()

    def test_commands_for_other_session_never_touch_current_worker(self):
        import femto.app.femto_control_app as control
        import test_temporal_gui
        app = test_temporal_gui.TemporalGuiTests().bare_app()
        camera = make_camera('wide_detail', 10)
        camera.start.return_value = camera
        reads = 0
        def read(_timeout):
            nonlocal reads
            reads += 1
            if reads > 3:
                app.stop_event.set()
                return None
            return make_frame([500]*10, reads)
        camera.read.side_effect = read
        session = uuid.uuid4().hex
        app.command_queue.put(('local_start', dict(session_id=session)))
        for command in ('local_pause', 'local_request', 'local_finish'):
            app.command_queue.put((command, dict(session_id='old-session', value=True)))
        app.command_queue.put(('local_pause', dict(session_id=session, value=True)))
        local = mock.Mock(session_id=session)
        local.closing.is_set.return_value = False
        with mock.patch.object(control, 'OrbbecAlignedCamera', return_value=camera), \
                mock.patch('femto.capture.local_capture.LocalCapture', return_value=local) as factory:
            app._camera_worker()
        self.assertEqual(factory.call_args.kwargs['session_id'], session)
        local.request.assert_not_called()
        local.pause.assert_called_once_with(True)
        local.finish.assert_called_once_with(False)  # Shutdown only.

    def test_review_sampling_is_bounded_and_source_arrays_unchanged(self):
        cloud = o3d.geometry.PointCloud()
        original = np.column_stack((np.arange(100001)*.001, np.zeros(100001), np.ones(100001)))
        cloud.points = o3d.utility.Vector3dVector(original)
        cloud.colors = o3d.utility.Vector3dVector(np.ones_like(original)*.5)
        with mock.patch('femto.geometry.fusion.read_geometry_safe', return_value=cloud), \
             mock.patch('femto.app.qt_capture.read_review_surface', return_value=None), \
             mock.patch('femto.local.local_processing.result_quality', return_value={}):
            result = read_review(Path('fixture'), threading.Event(), lambda _: None)['clouds']
        self.assertLessEqual(len(result['final'][0]), 35000)
        result['final'][0][:] = 999
        np.testing.assert_array_equal(np.asarray(cloud.points), original)


if __name__ == '__main__':
    unittest.main()
