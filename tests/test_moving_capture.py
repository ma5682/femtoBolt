"""Restored operator flow; no physical camera or training job is started."""
import json
from pathlib import Path
import queue
import threading
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import femto.app.femto_control_app as control
from tests.test_segmented_capture import make_camera, make_frame
from tests.camera_worker_fixture import bare_camera_app


class MovingCaptureTests(unittest.TestCase):
    def app(self):
        return bare_camera_app()

    def test_camera_worker_records_unknown_moving_pose_and_valid_fixed_pose(self):
        for mode in ('moving', 'keyframe'):
            with self.subTest(mode=mode), TemporaryDirectory() as tmp:
                app = self.app()
                app.stop_event = threading.Event()
                app.frame_lock = threading.Lock()
                app.command_queue = queue.Queue()
                app.command_queue.put(('start_scan', dict(duration_s=15, fps=6, processing_mode=mode)))
                camera = make_camera('wide_detail', 3)
                frame = make_frame([500,600,700])
                calls = 0
                def read(timeout):
                    nonlocal calls
                    calls += 1
                    if calls > 2:
                        app.stop_event.set()
                        return None
                    return frame
                camera.read.side_effect = read
                camera.start.return_value = camera
                with mock.patch.object(control, 'DATASETS_ROOT', Path(tmp)), mock.patch.object(control, 'OrbbecAlignedCamera', return_value=camera):
                    app._camera_worker()
                events = list(app.event_queue.queue)
                self.assertFalse([e for e in events if e[0] == 'camera_error'], events)
                started = next(value for name,value in events if name == 'scan_started')
                session = Path(started['session'])
                pose = json.loads((session/'pose/000000.json').read_text())
                manifest = json.loads((session/'manifest.json').read_text())
                self.assertEqual(pose['pose_valid'], mode == 'keyframe')
                self.assertEqual(manifest['parameters']['fixed_camera'], mode == 'keyframe')
                if mode == 'moving':
                    self.assertIsNone(pose['T_WC'])
                    self.assertEqual(manifest['status'], 'captured_pose_pending')
                else:
                    self.assertEqual(pose['T_WC'][0], [1.,0.,0.,0.])
                self.assertTrue((session/'rgb/000000.jpg').exists())
                self.assertTrue((session/'depth/000000.png').exists())
                camera.stop.assert_called_once()

    def test_moving_passes_all_raw_frames_to_pose_pipeline_without_keyframe_selection(self):
        with TemporaryDirectory() as tmp:
            source = Path(tmp)
            manifest = source/'manifest.json'
            manifest.write_text(json.dumps({'parameters':{'fixed_camera':False}}))
            before = manifest.read_bytes()
            app = self.app()
            summary = dict(reconstruction='new/reconstruction', reconstruction_metrics={})
            with mock.patch.object(control,'run_demo',return_value=(Path('new'),summary)) as run, mock.patch.object(control,'build_fixed_keyframe_dataset') as keyframe:
                app._reconstruction_worker(source,'moving')
            args = run.call_args.args[0]
            self.assertEqual(args.dataset,source)
            self.assertEqual(args.mode,'latest')
            self.assertTrue(args.no_viewer)
            keyframe.assert_not_called()
            self.assertEqual(app.event_queue.get_nowait()[0],'reconstruction_done')
            self.assertEqual(manifest.read_bytes(),before)

    def test_fixed_dataset_cannot_be_relabelled_as_moving(self):
        with TemporaryDirectory() as tmp:
            source = Path(tmp)
            (source/'manifest.json').write_text(json.dumps({'parameters':{'fixed_camera':True}}))
            app = self.app()
            with mock.patch.object(control,'run_demo') as run:
                app._reconstruction_worker(source,'moving')
            run.assert_not_called()
            self.assertEqual(app.event_queue.get_nowait()[0],'reconstruction_error')

    def test_pose_failure_does_not_fall_back_to_keyframe_or_old_result(self):
        with TemporaryDirectory() as tmp:
            source = Path(tmp)
            (source/'manifest.json').write_text(json.dumps({'parameters':{'fixed_camera':False}}))
            app = self.app()
            with mock.patch.object(control,'run_demo',side_effect=ValueError('Pose quality BLOCKED')) as run, mock.patch.object(control,'build_fixed_keyframe_dataset') as keyframe:
                app._reconstruction_worker(source,'moving')
            run.assert_called_once()
            keyframe.assert_not_called()
            kind, error = app.event_queue.get_nowait()
            self.assertEqual(kind,'reconstruction_error')
            self.assertIn('Pose quality BLOCKED',error)

    def test_keyframe_processing_still_selects_one_frame(self):
        app = self.app()
        summary = dict(reconstruction='new/reconstruction',reconstruction_metrics={})
        with mock.patch.object(control,'run_demo',return_value=(Path('new'),summary)) as run, mock.patch.object(control,'build_fixed_keyframe_dataset',return_value=(Path('selected'),{})) as keyframe:
            app._reconstruction_worker(Path('raw'),'keyframe')
        keyframe.assert_called_once()
        self.assertEqual(run.call_args.args[0].dataset,Path('selected'))
        self.assertEqual(app.event_queue.get_nowait()[0],'reconstruction_done')
