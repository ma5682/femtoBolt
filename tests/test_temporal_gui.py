import json
from pathlib import Path
import queue
from tempfile import TemporaryDirectory
import threading
import unittest
from unittest import mock

import numpy as np
import tkinter as tk

import femto.app.femto_control_app as control
from femto.app.app_temporal import TemporalBurst, process_burst
from tests.test_segmented_capture import make_camera, make_frame


class TemporalGuiTests(unittest.TestCase):
    def bare_app(self):
        app=control.FemtoControlApp.__new__(control.FemtoControlApp)
        app.language='zh'; app.event_queue=queue.Queue(); app.command_queue=queue.Queue()
        app.stop_event=threading.Event(); app.frame_lock=threading.Lock()
        app.temporal_thread=None
        return app

    def test_main_camera_is_only_owner_and_preview_continues_after_burst(self):
        app=self.bare_app()
        app.command_queue.put(('temporal_start',dict(count=5,mode='fast')))
        camera=make_camera('wide_detail',3); camera.start.return_value=camera
        calls=0; dispatched=[]
        def read(timeout):
            nonlocal calls
            calls+=1
            if calls>9: app.stop_event.set(); return None
            return make_frame([500,600,700],calls)
        camera.read.side_effect=read
        app._dispatch_temporal=lambda burst,**kw:dispatched.append((burst,kw))
        with mock.patch.object(control,'OrbbecAlignedCamera',return_value=camera) as factory:
            app._camera_worker()
        self.assertEqual(factory.call_count,1)
        self.assertEqual(len(dispatched),1)
        self.assertTrue(dispatched[0][0].complete)
        self.assertGreater(app.latest_frame.frame_index,dispatched[0][0].frames[-1].frame_index)
        self.assertFalse([e for e in app.event_queue.queue if e[0]=='camera_error'])

    def test_cancel_saves_partial_without_fusion(self):
        camera=make_camera('wide_detail',3)
        burst=TemporalBurst(camera,5,'fast')
        frame=make_frame([500,600,700],1)
        burst.append(frame,camera.intrinsics)
        frame.depth_mm[:]=900
        self.assertEqual(burst.frames[0].depth_mm[0,0],500)
        with TemporaryDirectory() as td, mock.patch('femto.temporal.scene_temporal_fusion.build_temporal_dataset') as fuse:
            events=queue.Queue(); process_burst(burst,Path(td),events,fuse=False)
            result=list(events.queue)[-1][1]
            self.assertEqual(result['status'],'cancelled_or_incomplete')
            source=Path(result['source'])
            self.assertTrue((source/'raw/frame_001_depth_mm.png').exists())
            self.assertEqual(json.loads((source/'manifest.json').read_text())['frame_count'],1)
            fuse.assert_not_called()

    def test_camera_stop_hands_partial_burst_to_writer(self):
        app=self.bare_app(); app.command_queue.put(('temporal_start',dict(count=15,mode='quality')))
        camera=make_camera('wide_detail',3); camera.start.return_value=camera
        calls=0; dispatched=[]
        def read(timeout):
            nonlocal calls
            calls+=1
            if calls>3: app.stop_event.set(); return None
            return make_frame([500,600,700],calls)
        camera.read.side_effect=read
        app._dispatch_temporal=lambda burst,**kwargs:dispatched.append((burst,kwargs))
        with mock.patch.object(control,'OrbbecAlignedCamera',return_value=camera): app._camera_worker()
        self.assertEqual(len(dispatched),1)
        self.assertFalse(dispatched[0][1]['fuse'])
        self.assertEqual(len(dispatched[0][0].frames),2)

    def test_error_worker_event_restores_reviewable_failure(self):
        burst=TemporalBurst(make_camera('wide_detail',3),5,'fast')
        with TemporaryDirectory() as td,mock.patch.object(burst,'save',side_effect=OSError('disk full')):
            events=queue.Queue(); process_burst(burst,Path(td),events)
        self.assertEqual(list(events.queue)[-1][0],'temporal_error')
        self.assertIn('disk full',list(events.queue)[-1][1]['error'])

    def test_panel_options_and_busy_state_without_camera(self):
        with mock.patch.object(control.FemtoControlApp,'_start_camera'),mock.patch.object(control.FemtoControlApp,'_load_latest_paths'), \
             mock.patch('femto.app.app_temporal_panel.TemporalPanel.restore_last'):
            root=tk.Tk(); root.withdraw()
            try:
                app=control.FemtoControlApp(root)
                app.camera_ready=True
                app.open_temporal_panel(); panel=app.temporal_panel
                root.update()
                self.assertEqual(tuple(panel.count_box['values']),('5','10','15'))
                panel.count.set('10'); panel.mode.set('quality'); panel.start()
                self.assertTrue(app.processing); self.assertTrue(app.temporal_collecting)
                self.assertEqual(app.command_queue.get_nowait(),('temporal_start',dict(count=10,mode='quality')))
                self.assertFalse(app.start_temporal(5,'fast'))
                panel.cancel()
                self.assertEqual(app.command_queue.get_nowait()[0],'temporal_stop')
                app.event_queue.put(('temporal_finished',dict(status='cancelled_or_incomplete',source='unused',output='unused')))
                app._poll_events()
                self.assertFalse(app.processing); self.assertFalse(app.temporal_busy)
                self.assertIsNone(app.latest_result)
                app.event_queue.put(('temporal_error',dict(error='test failure',source=None)))
                app._poll_events()
                panel.window.destroy()
                app.open_temporal_panel()
                self.assertIn('test failure',app.temporal_panel.status.get())
            finally:
                for timer in root.tk.call('after','info'): root.after_cancel(timer)
                root.destroy()

    def test_completed_result_buttons_and_no_old_model_reassignment(self):
        with TemporaryDirectory() as td, mock.patch.object(control.FemtoControlApp,'_start_camera'), \
             mock.patch.object(control.FemtoControlApp,'_load_latest_paths'),mock.patch('femto.app.app_temporal_panel.TemporalPanel.restore_last'):
            root=tk.Tk(); root.withdraw()
            try:
                app=control.FemtoControlApp(root); app.open_temporal_panel()
                output=Path(td)
                for name in ('fused_pointcloud.ply','comparison.png','temporal_report.json'): (output/name).write_text('test')
                report=dict(frame_count=5,original_invalid_rate=.1,after_temporal_invalid_rate=.08,reference_point_count=90,point_count=92)
                app.event_queue.put(('temporal_finished',dict(status='completed',output=str(output),source=str(output),report=report)))
                app._poll_events(); app.temporal_panel.update_controls()
                self.assertEqual(str(app.temporal_panel.buttons['cloud']['state']),'normal')
                with mock.patch('femto.app.app_temporal_panel.subprocess.Popen') as launch:
                    app.temporal_panel.open_result('cloud')
                    self.assertEqual(launch.call_args.args[0][1:3],['-m','femto.viewer.segmented_cloud_viewer'])
                self.assertIsNone(app.latest_result)
            finally:
                for timer in root.tk.call('after','info'): root.after_cancel(timer)
                root.destroy()

    def test_visible_panel_layout_at_default_window_size(self):
        with mock.patch.object(control.FemtoControlApp,'_start_camera'),mock.patch.object(control.FemtoControlApp,'_load_latest_paths'), \
             mock.patch('femto.app.app_temporal_panel.TemporalPanel.restore_last'):
            root=tk.Tk()
            try:
                app=control.FemtoControlApp(root); app.open_temporal_panel()
                root.update()
                button=app.photo_button
                self.assertGreater(button.winfo_width(),70)
                self.assertLess(button.winfo_rootx()+button.winfo_width(),root.winfo_rootx()+root.winfo_width()+1)
                panel=app.temporal_panel
                for widget in (panel.start_button,panel.cancel_button,*panel.buttons.values()):
                    self.assertTrue(widget.winfo_ismapped())
                    self.assertLessEqual(widget.winfo_rootx()+widget.winfo_width(),panel.window.winfo_rootx()+panel.window.winfo_width())
                    self.assertLessEqual(widget.winfo_rooty()+widget.winfo_height(),panel.window.winfo_rooty()+panel.window.winfo_height())
            finally:
                for timer in root.tk.call('after','info'): root.after_cancel(timer)
                root.destroy()
