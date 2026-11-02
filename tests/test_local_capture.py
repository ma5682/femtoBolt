import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import time
import unittest
from unittest import mock
import queue

import numpy as np
import open3d as o3d
from femto.tracking.scene_live_tracker import LiveTracker, display_pose
from femto.capture.local_capture import LocalCapture
from tests.test_segmented_capture import make_frame, make_camera


class TrackerTests(unittest.TestCase):
    def tracker(self, transform=None):
        estimator=mock.Mock()
        estimator.estimate.return_value=(np.eye(4) if transform is None else transform, {})
        return LiveTracker(estimator,250,2210,clock=lambda:10.)

    @mock.patch('femto.tracking.scene_live_tracker.frame_quality',return_value=dict(usable=True))
    def test_origin_and_inverse_translation(self, quality):
        forward=np.eye(4);forward[0,3]=-.03
        tracker=self.tracker(forward)
        first=tracker.process(make_frame([500]*10,1),7,9.5)
        second=tracker.process(make_frame([500]*10,2),8,9.8)
        self.assertEqual(first['reference_frame_id'],7)
        np.testing.assert_allclose(first['T_WC'],np.eye(4))
        self.assertAlmostEqual(second['xyz_mm'][0],30.)
        self.assertEqual(second['reference_frame_id'],7)
        self.assertEqual(json.loads(json.dumps(second))['state'],'tracking')

    def test_composed_angles_follow_documented_axes(self):
        angles=np.radians([7,-4,2])
        r=o3d.geometry.get_rotation_matrix_from_zyx(angles[::-1])
        t=np.eye(4);t[:3,:3]=r
        np.testing.assert_allclose(display_pose(t)['pitch_yaw_roll_deg'],[7,-4,2],atol=1e-8)
        t[:3,:3]=o3d.geometry.get_rotation_matrix_from_zyx([0,np.pi/2,0])
        self.assertIsNone(display_pose(t)['pitch_yaw_roll_deg'])

    @mock.patch('femto.tracking.scene_live_tracker.frame_quality',return_value=dict(usable=True))
    def test_invalid_first_frame_never_becomes_origin(self, quality):
        tracker=self.tracker();frame=make_frame([500]*10)
        frame.color_timestamp_us=7000
        result=tracker.process(frame,0,9.5)
        self.assertIsNone(result['T_WC']);self.assertIsNone(tracker.reference_id)

    @mock.patch('femto.tracking.scene_live_tracker.frame_quality',return_value=dict(usable=True))
    def test_lost_latches_and_does_not_relocalize(self, quality):
        tracker=self.tracker();tracker.process(make_frame([500]*10,1),0,7.9)
        result=tracker.process(make_frame([500]*10,2),1,9.9)
        self.assertEqual(result['state'],'lost');self.assertIsNone(result['T_WC'])

    @mock.patch('femto.tracking.scene_live_tracker.frame_quality',return_value=dict(usable=True))
    def test_duplicate_timestamp_and_pair_failure_do_not_advance_reference(self, quality):
        tracker=self.tracker();frame=make_frame([500]*10,1)
        tracker.process(frame,0,9.1)
        result=tracker.process(frame,1,9.2)
        self.assertIsNone(result['T_WC'])
        tracker.estimator.estimate.side_effect=ValueError('bad pair')
        result=tracker.process(make_frame([500]*10,2),2,9.3)
        self.assertIsNone(result['T_WC']);np.testing.assert_allclose(tracker.pose,np.eye(4))


class CaptureTests(unittest.TestCase):
    def test_overloaded_queue_replaces_old_unprocessed_frame(self):
        c=LocalCapture.__new__(LocalCapture)
        c.closing=threading.Event();c.queue=queue.Queue(maxsize=2);c.dropped=0
        for index in (1,2,3):c.submit(make_frame([500]*10,index))
        self.assertEqual(c.dropped,1)
        self.assertEqual([c.queue.get()[0].frame_index,c.queue.get()[0].frame_index],[2,3])

    def bare(self):
        capture=LocalCapture.__new__(LocalCapture)
        capture.lock=threading.Lock();capture.closing=threading.Event()
        capture.accepted=[dict(T_WC=np.eye(4).tolist())]
        capture.paused=False;capture.requested=10.;capture.emit=mock.Mock()
        return capture

    def test_request_uses_future_frame_and_waits_until_still(self):
        c=self.bare();t=np.eye(4);t[0,3]=.015
        record=dict(state='tracking',T_WC=t.tolist(),step_m=.01,step_deg=0.)
        with mock.patch('femto.capture.local_capture.time.monotonic',return_value=10.5):
            self.assertFalse(c._select(record,9.9))
            self.assertFalse(c._select(record,10.2))
            record['step_m']=0.
            self.assertTrue(c._select(record,10.3))
            self.assertIsNone(c.requested)

    def test_pause_duplicate_and_expired_request(self):
        c=self.bare();r=dict(state='tracking',T_WC=np.eye(4).tolist(),step_m=0.,step_deg=0.)
        with mock.patch('femto.capture.local_capture.time.monotonic',return_value=10.5):
            c.paused=True;self.assertFalse(c._select(r,10.2))
            c.paused=False;self.assertFalse(c._select(r,10.2))
            c.emit.assert_called_with('notice',key='duplicate')
        c.requested=10.
        with mock.patch('femto.capture.local_capture.time.monotonic',return_value=14.):
            self.assertFalse(c._select(r,10.2));self.assertIsNone(c.requested)

    def test_cancel_worker_saves_raw_without_publishing_model(self):
        with TemporaryDirectory() as td, mock.patch('femto.capture.local_capture.PairEstimator') as pair, \
                mock.patch('femto.tracking.scene_live_tracker.frame_quality',return_value=dict(usable=True)), \
                mock.patch('femto.pipeline.complete_demo.run_demo') as reconstruct:
            c=LocalCapture(make_camera('wide_detail',10),Path(td)/'data',Path(td)/'out',queue.Queue())
            c.submit(make_frame([500]*10,1))
            deadline=time.monotonic()+5
            while not c.records and time.monotonic()<deadline:time.sleep(.01)
            c.finish(False);c.thread.join(5)
            self.assertFalse(c.thread.is_alive());reconstruct.assert_not_called()
            frame=json.loads(c.scan.frames_log.read_text().splitlines()[0])
            self.assertTrue(frame['rgb'].endswith('.png'))
            raw=json.loads((c.session/frame['pose']).read_text())
            self.assertFalse(raw['pose_valid']);self.assertIsNone(raw['T_WC'])
            self.assertTrue((c.session/'derived'/'live_tracking.jsonl').exists())

    def test_export_preserves_raw_and_binds_pose_to_same_frame(self):
        with TemporaryDirectory() as td, mock.patch.object(LocalCapture,'_run'):
            c=LocalCapture(make_camera('wide_detail',10),Path(td)/'data',Path(td)/'out',queue.Queue())
            c.thread.join()
            from femto.app.femto_control_app import save_scan_frame,finalize_scan
            for i in range(3):save_scan_frame(c.scan,make_frame([500]*10,i+1),lossless=True)
            finalize_scan(c.scan,c.camera)
            before=c.scan.frames_log.read_bytes()
            c.accepted=[dict(frame_id=i,T_WC=np.eye(4).tolist(),reference_frame_id=0,capture_timestamp_us=i+1) for i in range(3)]
            posed=c._export_poses()
            frames=[json.loads(s) for s in (posed/'frames.jsonl').read_text().splitlines()]
            for i,f in enumerate(frames):
                pose=json.loads((posed/f['pose']).read_text())
                self.assertEqual(pose['capture_timestamp_us'],f['camera_timestamp_us'])
                self.assertEqual(pose['frame'],i)
            self.assertEqual(before,c.scan.frames_log.read_bytes())


class LocalIntegrationTests(unittest.TestCase):
    def test_real_rgbd_pair_backend_on_static_synthetic_frame(self):
        from femto.pipeline.synthetic_scene_demo import generate_dataset
        from femto.tracking.scene_live_tracker import PairEstimator
        from femto.io_utils import imread_unicode
        import cv2
        with TemporaryDirectory() as td:
            dataset=generate_dataset(Path(td)/'data',frames=4,width=512,height=288)
            row=json.loads((dataset/'frames.jsonl').read_text().splitlines()[0])
            frame=make_frame([500])
            frame.color_bgr=imread_unicode(dataset/row['rgb'],cv2.IMREAD_COLOR)
            frame.depth_mm=imread_unicode(dataset/row['depth'],cv2.IMREAD_UNCHANGED)
            pair=PairEstimator(dataset,500,3860)
            prepared=pair.prepare(frame)
            transform,metrics=pair.estimate(prepared,prepared)
            np.testing.assert_allclose(transform,np.eye(4),atol=1e-5)
            self.assertTrue(metrics['cycle']['passed'])

    def test_local_complete_pipeline_keeps_origin_and_outputs_review_cloud(self):
        from femto.pipeline.synthetic_scene_demo import generate_dataset
        from femto.pipeline.complete_demo import run_demo
        with TemporaryDirectory() as td:
            root=Path(td);dataset=generate_dataset(root/'data',frames=5,width=192,height=108)
            rows=[json.loads(s) for s in (dataset/'frames.jsonl').read_text().splitlines()]
            origin=np.asarray(json.loads((dataset/rows[0]['pose']).read_text())['T_WC'])
            for row in rows:
                p=dataset/row['pose'];data=json.loads(p.read_text())
                data['T_WC']=(np.linalg.inv(origin)@np.asarray(data['T_WC'])).tolist()
                p.write_text(json.dumps(data))
            p=dataset/'manifest.json';manifest=json.loads(p.read_text())
            manifest['local_capture']=dict(measured_only=True,session_id='test',reference_frame_id=0)
            p.write_text(json.dumps(manifest))
            result,summary=run_demo(argparse.Namespace(mode='latest',dataset=dataset,datasets_root=root/'data',outputs_root=root/'out',no_viewer=True))
            self.assertTrue((result/'reconstruction'/'initial_observed_points_m.ply').is_file())
            self.assertTrue((result/'result_quality.html').is_file())
            self.assertEqual(summary['frame_count'],5)
            self.assertEqual(summary['data_kind'],'synthetic')

    def test_measured_geometry_and_annotation_lineage(self):
        from femto.pipeline.synthetic_scene_demo import generate_dataset
        from femto.pipeline.scene_reconstruction import reconstruct_scene
        from femto.geometry.fusion import read_geometry_safe
        from femto.annotation.scene_annotations import AnnotationStore
        with TemporaryDirectory() as td:
            root=Path(td);dataset=generate_dataset(root/'data',frames=5,width=192,height=108)
            manifest=json.loads((dataset/'manifest.json').read_text())
            manifest['local_capture']=dict(measured_only=True,session_id='test',reference_frame_id=0)
            (dataset/'manifest.json').write_text(json.dumps(manifest))
            report=reconstruct_scene(dataset,root/'result',image_scale=1.,voxel_m=.015,sdf_trunc_m=.045)
            self.assertTrue((root/'result'/'initial_observed_points_m.ply').is_file())
            context=json.loads((root/'result'/'local_geometry.json').read_text())
            self.assertFalse(context['inferred_repairs_integrated'])
            self.assertGreater(context['support']['triangles_after'],0)
            self.assertEqual(report['depth_quality']['integrations_per_frame'],[1]*5)
            self.assertFalse(report['depth_quality']['filled_depth_used'])
            self.assertTrue(all(row['camera_point_count'] is not None for row in report['frames']))
            self.assertTrue(all('accumulated_voxel_downsample_applied' in row for row in report['frames']))
            self.assertTrue(context['support']['post_retain_component_filter']['applied'])
            self.assertEqual(context['display_artifacts']['role'],'display_subset_not_measurement')
            scaled=reconstruct_scene(dataset,root/'scaled',image_scale=.5,voxel_m=.015,sdf_trunc_m=.045)
            self.assertFalse(scaled['local_capture']['support']['reused_prepared_depth'])
            mesh_path=Path(report['outputs']['mesh']['ply'])
            mesh=read_geometry_safe(mesh_path,mesh=True)
            store=AnnotationStore(mesh,mesh_path);saved=store.save(root/'annotations')
            store.load(saved)
            store.local_context['session_id']='other'
            with self.assertRaises(ValueError):store.load(saved)


    def test_local_camera_dispatch_does_not_open_second_sdk(self):
        import femto.app.femto_control_app as control
        from tests.test_temporal_gui import TemporalGuiTests
        app=TemporalGuiTests().bare_app()
        camera=make_camera('wide_detail',10);camera.start.return_value=camera
        calls=0
        def read(timeout):
            nonlocal calls
            calls+=1
            if calls>4:app.stop_event.set();return None
            return make_frame([500]*10,calls)
        camera.read.side_effect=read
        app.command_queue.put(('local_start',None))
        local=mock.Mock();local.closing.is_set.return_value=False
        with mock.patch.object(control,'OrbbecAlignedCamera',return_value=camera) as factory, \
                mock.patch('femto.capture.local_capture.LocalCapture',return_value=local) as worker:
            app._camera_worker()
        factory.assert_called_once();worker.assert_called_once()
        self.assertGreater(local.submit.call_count,0)
        local.finish.assert_called_with(False)
        self.assertFalse([e for e in app.event_queue.queue if e[0]=='camera_error'])
