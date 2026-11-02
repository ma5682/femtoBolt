import json
from pathlib import Path
import queue
from tempfile import TemporaryDirectory
import threading
import unittest
from unittest import mock

import numpy as np

import femto.app.femto_control_app as control
from femto.app.app_temporal import TemporalBurst, process_burst
from tests.test_segmented_capture import make_camera, make_frame
from tests.camera_worker_fixture import bare_camera_app


class TemporalGuiTests(unittest.TestCase):
    def bare_app(self):
        return bare_camera_app()

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
