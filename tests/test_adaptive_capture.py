"""Early one/two-view finish and historical multiview reconstruction."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase,mock
import json
import queue
import time

import numpy as np
import open3d as o3d

from femto.capture.local_capture import LocalCapture
from femto.geometry.pose_geometry import Intrinsics
from femto.tracking.scene_live_tracker import PreparedRGBD
from test_segmented_capture import make_camera,make_frame


class AdaptiveCaptureTests(TestCase):
    def exercise_manual(self,count):
        with TemporaryDirectory() as directory:
            root=Path(directory);camera=make_camera('wide_detail',10000)
            camera.intrinsics=Intrinsics(100,100,100.,100.,49.5,49.5)
            intr=o3d.camera.PinholeCameraIntrinsic(100,100,100.,100.,49.5,49.5)
            positions=[(0.,0.),(.012,0.),(.024,0.),(.024,.012)]
            def prepare(frame):
                return PreparedRGBD(None,intr,None,frame.color_bgr,frame.depth_mm,
                    features=np.array(positions[(frame.frame_index)//2]))
            def estimate(previous,current,**kwargs):
                transform=np.eye(4);transform[:2,3]=previous.features-current.features
                return transform,{}
            with mock.patch('femto.capture.local_capture.PairEstimator') as estimator, \
                    mock.patch('femto.tracking.scene_live_tracker.frame_quality',return_value={'usable':True}), \
                    mock.patch('femto.pipeline.local_offline.LocalOfflineJob') as offline:
                estimator.return_value.prepare.side_effect=prepare
                estimator.return_value.estimate.side_effect=estimate
                capture=LocalCapture(camera,root/'data',root/'models',queue.Queue(),region_first=True)
                def submit(index):
                    frame=make_frame([500]*10000,index)
                    frame.depth_mm=frame.depth_mm.reshape(100,100)
                    frame.color_bgr=frame.color_bgr.reshape(100,100,3)
                    before=len(capture.records);capture.submit(frame)
                    deadline=time.monotonic()+4
                    while len(capture.records)==before and capture.thread.is_alive() and time.monotonic()<deadline:
                        time.sleep(.005)
                    self.assertEqual(len(capture.records),before+1)
                try:
                    submit(1)
                    for view in range(1,count):
                        capture.request();submit(view*2);submit(view*2+1)
                        self.assertEqual(len(capture.accepted),view+1)
                        self.assertFalse(capture.closing.is_set())
                    offline.assert_not_called()
                    capture.finish(True)
                    self.assertTrue(capture.awaiting_region.wait(5))
                    self.assertEqual(capture.finish_reason,'user_finish')
                    rows=[json.loads(line) for line in (capture.session/'derived/local_posed/frames.jsonl').read_text().splitlines()]
                    self.assertEqual(len(rows),count)
                    offline.assert_not_called()  # Wait for paint even with one view.
                finally:
                    capture.finish(False);capture.thread.join(5)
                self.assertFalse(capture.thread.is_alive())

    def test_single_view_can_finish(self):
        self.exercise_manual(1)

    def test_two_views_can_finish_before_the_limit(self):
        self.exercise_manual(2)

    def test_four_view_real_offline_job_and_publication(self):
        from test_region_first_reconstruction import dataset_fixture
        from femto.geometry.local_region import load_reference,save_request
        from femto.pipeline.local_offline import LocalOfflineJob,publish_result
        from femto.io_utils import load_json_object
        from femto.annotation.fixed_rgb_geometry import FixedRGBModel
        with TemporaryDirectory() as directory:
            root=Path(directory);capture,dataset=dataset_fixture(root,4)
            reference=load_reference(dataset);mask=np.zeros(reference.depth.shape,np.uint8);mask[40:80,60:100]=1
            request=save_request(reference,mask,root/'request')
            job=LocalOfflineJob(dataset,root/'job',root/'outputs',capture.session_id,lambda:False,region_request=request)
            result,summary=job.run()
            report=load_json_object(result/'reconstruction/reconstruction_report.json')
            self.assertEqual(report['frame_count'],4)
            self.assertEqual(report['parameters']['mesh_method'],'tsdf')
            self.assertEqual(len(report['reconstruction_scope']['frames']),4)
            publish_result(result,root/'outputs',capture.session_id)
            self.assertEqual(load_json_object(result/'publication.json')['status'],'published')
            model=FixedRGBModel(result/'reconstruction')
            self.assertEqual(model.view_reference['session_id'],capture.session_id)

    def test_invalid_single_pose_cannot_be_selected_for_reconstruction(self):
        from test_region_first_reconstruction import dataset_fixture
        from femto.geometry.local_region import load_reference
        from femto.io_utils import load_json_object,write_json_strict
        with TemporaryDirectory() as directory:
            _,dataset=dataset_fixture(Path(directory),1)
            pose=load_json_object(dataset/'pose/000000.json');pose['pose_valid']=False
            write_json_strict(dataset/'pose/000000.json',pose)
            with self.assertRaisesRegex(ValueError,'valid T_WC'):load_reference(dataset)
