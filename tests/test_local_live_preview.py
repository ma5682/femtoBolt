import json
from pathlib import Path
import queue
import tempfile
import time
import tkinter as tk
import unittest
from unittest import mock

import numpy as np

from femto.capture.local_live_preview import ObservedAccumulator,LivePreviewWorker
from tests.test_segmented_capture import make_camera,make_frame


class AccumulatorTests(unittest.TestCase):
    def test_camera_to_origin_transform_and_new_samples_have_source(self):
        a=ObservedAccumulator(voxel_m=.01)
        p=np.array([[0,0,1],[.02,0,1]])
        colors=np.array([[10,20,30],[40,50,60]],dtype=np.uint8)
        a.add(p,colors,0,np.eye(4))
        t=np.eye(4);t[0,3]=.02
        update=a.add(p,colors,7,t)
        self.assertEqual(update['added_samples'],1)
        np.testing.assert_allclose(a.points,[[0,0,1],[.02,0,1],[.04,0,1]])
        self.assertEqual(a.sources.tolist(),[0,0,7])
        np.testing.assert_array_equal(a.colors[-1],colors[1])
        np.testing.assert_allclose(a.initial[0],p)

    def test_missing_depth_and_repeated_voxels_do_not_invent_surface(self):
        a=ObservedAccumulator(voxel_m=.01)
        points=np.array([[0,0,0],[0,0,-1],[np.nan,0,1],[0,0,1],[.001,0,1]])
        first=a.add(points,np.zeros((5,3),np.uint8),0,np.eye(4))
        self.assertEqual(first['added_samples'],1)
        next_frame=a.add(points,np.zeros((5,3),np.uint8),1,np.eye(4))
        self.assertEqual(next_frame['added_samples'],0)
        np.testing.assert_allclose(a.points,[[0,0,1]])

    def test_budget_is_bounded_and_snapshot_is_independent(self):
        a=ObservedAccumulator(voxel_m=.001,max_points=3)
        p=np.column_stack((np.arange(10)*.01,np.zeros(10),np.ones(10)))
        update=a.add(p,np.zeros((10,3),np.uint8),0,np.eye(4))
        self.assertTrue(update['budget_limited']);self.assertEqual(len(a.points),3)
        snap=a.snapshot(update,display_budget=2)
        self.assertEqual(len(snap['points']),2)
        snap['points'][:]=123
        self.assertLess(a.points.max(),2)
        update=a.add(p+.02,np.zeros((10,3),np.uint8),1,np.eye(4))
        self.assertEqual(update['added_samples'],0)

    def test_invalid_pose_duplicate_frame_leave_accumulator_unchanged(self):
        a=ObservedAccumulator();p=np.array([[0,0,1]])
        a.add(p,np.zeros((1,3),np.uint8),0,np.eye(4))
        with self.assertRaises(ValueError):a.add(p,np.zeros((1,3),np.uint8),0,np.eye(4))
        t=np.eye(4);t[0,0]=2
        with self.assertRaises(ValueError):a.add(p,np.zeros((1,3),np.uint8),1,t)
        self.assertEqual(a.frame_ids,{0})


class WorkerTests(unittest.TestCase):
    def test_source_scan_handles_bom_without_hiding_blocked_imports(self):
        from commercial_compliance_check import imported_modules, BLOCKED_IMPORT_PREFIXES
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'bom.py'
            path.write_text('import pymeshlab\nimport numpy\n',encoding='utf-8-sig')
            modules=imported_modules(path)
            self.assertIn('numpy',modules)
            self.assertTrue(any(m.startswith(BLOCKED_IMPORT_PREFIXES) for m in modules))

    def test_saved_rgbd_in_same_origin_latest_mailbox_and_raw_unchanged(self):
        from femto.app.femto_control_app import ActiveScan,save_scan_frame
        from femto.capture.scene_capture import write_calibration
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            for name in ('rgb','depth','pose','derived'):(root/name).mkdir()
            camera=make_camera('wide_detail',20)
            write_calibration(root,camera,camera.intrinsics)
            scan=ActiveScan(root,root/'frames.jsonl',0,0,300,0,0,False,'moving',250,2210)
            for i in range(3):save_scan_frame(scan,make_frame([500]*20,i+1),lossless=True)
            raw=(root/'depth/000000.png').read_bytes()
            worker=LivePreviewWorker(root,'s',250,2210)
            try:
                self.assertFalse(worker.submit(dict(state='lost',selected_for_fusion=True)))
                for i in range(3):
                    t=np.eye(4);t[1,3]=i*.02
                    self.assertTrue(worker.submit(dict(state='tracking',selected_for_fusion=True,frame_id=i,
                        capture_timestamp_us=i+1,T_WC=t.tolist())))
            finally:
                worker.close();worker.thread.join(10)
            self.assertFalse(worker.thread.is_alive())
            self.assertFalse(worker.failed)
            snapshot=worker.mailbox.get_nowait()
            self.assertEqual(snapshot['accepted_preview_frames'],3)
            self.assertEqual(snapshot['session_id'],'s')
            self.assertEqual(worker.mailbox.qsize(),0)
            self.assertEqual(raw,(root/'depth/000000.png').read_bytes())
            report=json.loads((root/'derived/live_preview_report.json').read_text())
            self.assertEqual(report['status'],'closed')

    def test_corrupt_input_reports_preview_failure_without_modifying_raw(self):
        from femto.capture.scene_capture import write_calibration
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);write_calibration(root,make_camera('wide_detail',20),make_camera('wide_detail',20).intrinsics)
            worker=LivePreviewWorker(root,'bad',250,2210)
            worker.submit(dict(state='tracking',selected_for_fusion=True,frame_id=0,
                capture_timestamp_us=1,T_WC=np.eye(4).tolist()))
            worker.close();worker.thread.join(10)
            self.assertTrue(worker.failed)
            self.assertEqual(worker.mailbox.get_nowait()['status'],'error')


class PreviewGuiTests(unittest.TestCase):
    def test_live_snapshot_controls_highlight_and_session_isolation(self):
        import femto.app.femto_control_app as control
        with mock.patch.object(control.FemtoControlApp,'_start_camera'),mock.patch.object(control.FemtoControlApp,'_load_latest_paths'):
            root=tk.Tk();root.withdraw()
            try:
                app=control.FemtoControlApp(root);panel=app.local_panel
                panel.session_id='current';panel.preview_mailbox=queue.Queue(maxsize=1)
                p=np.array([[0,0,1],[.02,0,1]])
                snapshot=dict(status='ready',session_id='wrong',points=p,colors=np.full((2,3),100,np.uint8),
                    sources=np.array([0,1]),initial_points=p[:1],initial_colors=np.full((1,3),100,np.uint8),
                    frame_id=1,accepted_preview_frames=2,point_count=2,added_samples=1,budget_limited=False)
                panel.preview_mailbox.put(snapshot);panel.consume_preview()
                self.assertNotIn('live',panel.clouds)
                snapshot['session_id']='current';panel.count=2
                panel.preview_mailbox.put(snapshot);panel.consume_preview()
                self.assertIn('live',panel.clouds)
                self.assertEqual(str(panel.buttons['initial']['state']),'normal')
                self.assertEqual(str(panel.buttons['view']['state']),'disabled')
                panel.set_view('live');panel.highlight.set(False);panel.draw_cloud()
                self.assertEqual(panel.view,'live')
                self.assertIn('2/2',panel.preview_status.get())
                center=panel.center.copy();panel.set_view('initial')
                np.testing.assert_array_equal(center,panel.center)
                panel.event('local_error',dict(session_id='current',source=None,error='tracking_lost'))
            finally:
                for timer in root.tk.call('after','info'):root.after_cancel(timer)
                root.destroy()

    def test_capture_notification_survives_next_pose_update(self):
        import femto.app.femto_control_app as control
        with mock.patch.object(control.FemtoControlApp,'_start_camera'),mock.patch.object(control.FemtoControlApp,'_load_latest_paths'):
            root=tk.Tk();root.withdraw()
            try:
                app=control.FemtoControlApp(root);panel=app.local_panel
                panel.active=True;panel.session_id='s'
                panel.event('local_notice',dict(session_id='s',source=None,key='duplicate'))
                panel.event('local_pose',dict(session_id='s',source=None,record=dict(state='tracking'),accepted=1))
                self.assertTrue(panel.hint.get().startswith(panel.text('duplicate')))
                self.assertIn('状态更新',panel.hint.get())
                panel.notice=None;panel.state='processing';panel.finishing=True
                panel.record=dict(state='tracking',received_monotonic=time.monotonic()-10,
                    xyz_mm=[0,0,0],pitch_yaw_roll_deg=[0,0,0])
                panel.refresh();panel.tick()
                self.assertEqual(panel.hint.get(),panel.text('processing'))
            finally:
                for timer in root.tk.call('after','info'):root.after_cancel(timer)
                root.destroy()
