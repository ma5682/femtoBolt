import json
import unittest
from concurrent.futures import Future
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock
import numpy as np
from femto.tracking.moving_quality import frame_quality, cycle_consistency, select_keyframes, capture_feedback
import femto.tracking.scene_pose_estimator as estimator
from femto.app.femto_control_app import ActiveScan, harvest_capture_motion

class MovingQualityTests(unittest.TestCase):
    def test_depth_and_blur_hints_do_not_modify_source(self):
        rgb = np.zeros((80,100,3), np.uint8)
        depth = np.full((80,100), 1000, np.uint16)
        original = depth.copy()
        self.assertEqual(frame_quality(rgb,depth,250,2210)['state'], 'blur')
        self.assertEqual(frame_quality(rgb,np.zeros_like(depth),250,2210)['state'], 'low_depth')
        np.testing.assert_array_equal(depth, original)

    def test_cycle_rejects_inconsistent_translation_and_rotation(self):
        forward=np.eye(4); forward[0,3]=.03
        self.assertTrue(cycle_consistency(forward,np.linalg.inv(forward))['passed'])
        self.assertFalse(cycle_consistency(forward,np.eye(4))['passed'])
        angle=np.deg2rad(4); rotation=np.eye(4)
        rotation[:2,:2]=[[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]
        self.assertFalse(cycle_consistency(rotation,np.eye(4))['passed'])

    def test_small_steps_accumulate_relative_to_last_keyframe(self):
        poses=[]
        for i in range(8):
            pose=np.eye(4); pose[0,3]=i*.006; poses.append(pose)
        self.assertEqual(select_keyframes(poses), [0,2,4,6])
        self.assertEqual(select_keyframes([np.eye(4)]*20),[0])

    def test_current_warning_not_hidden_by_past_movement(self):
        scan=ActiveScan(session=Path('test'),frames_log=Path('test/frames.jsonl'),
            started=0,next_sample=0,duration_s=15,fps=6,max_frames=90,
            fixed_camera=False,processing_mode='moving',depth_min_mm=250,depth_max_mm=2210)
        scan.motion_state='moved'; scan.motion_report=dict(reliable=True,appears_fixed=False)
        scan.motion_future=Future()
        scan.motion_future.set_result(dict(reliable=False,appears_fixed=None,capture_quality={'guidance':'blur'}))
        harvest_capture_motion(scan)
        self.assertEqual(scan.motion_state,'moved')
        self.assertEqual(scan.capture_quality['guidance'],'blur')
        self.assertEqual(len(scan.quality_history),1)

    def run_pipeline(self, root, *, inconsistent=False, bad_after=None):
        source=root/'raw';source.mkdir(); (source/'calibration').mkdir()
        (source/'manifest.json').write_text(json.dumps(dict(schema='femto-full-scene-rgbd-v1',parameters={'depth_min_mm':250,'depth_max_mm':2210})))
        frames=[dict(frame=i,rgb=f'rgb{i}',depth=f'depth{i}',pose_valid=False) for i in range(8)]
        (source/'frames.jsonl').write_text(''.join(json.dumps(f)+'\n' for f in frames))
        before=(source/'frames.jsonl').read_bytes()
        rgb=np.random.default_rng(12).integers(0,256,(80,100,3),dtype=np.uint8)
        depth=np.full((80,100),1000,np.uint16)
        def read(path, flag):
            if path.name.startswith('rgb'):return rgb.copy()
            index=int(path.name[5:])
            return np.zeros_like(depth) if bad_after is not None and index>=bad_after else depth.copy()
        calls=0
        def odometry(*args):
            nonlocal calls
            forward=calls%2==0;calls+=1
            pose=np.eye(4);pose[0,3]=.006 if forward else (-.006 if not inconsistent else .06)
            return True,pose,np.eye(6)
        with mock.patch.object(estimator,'o3d') as engine, mock.patch.object(estimator,'assess_background_motion',return_value={'reliable':True,'appears_fixed':False}), mock.patch.object(estimator,'_distortion_maps',return_value=(None,None,np.eye(3),100,80)), mock.patch.object(estimator,'_read_image',side_effect=read), mock.patch.object(estimator,'_prepare_rgbd',return_value=(object(),object(),None,None)):
            engine.pipelines.odometry.compute_rgbd_odometry.side_effect=odometry
            engine.pipelines.registration.evaluate_registration.return_value=mock.Mock(fitness=.8,inlier_rmse=.002)
            try:
                return estimator.estimate_poses(source,root/'posed')
            finally:
                self.assertEqual(before,(source/'frames.jsonl').read_bytes())

    def test_pipeline_tracks_intermediates_but_exports_diverse_keyframes(self):
        with TemporaryDirectory() as tmp:
            output,report=self.run_pipeline(Path(tmp))
            self.assertEqual(report['accepted_frame_ordinals'],[0,2,4,6])
            self.assertEqual(len(report['pairs']),7)
            self.assertEqual(len((output/'frames.jsonl').read_text().splitlines()),4)
            self.assertEqual(report['keyframe_selection']['redundant_frame_ordinals'],[1,3,5,7])

    def test_bad_cycles_or_long_depth_loss_block_without_output_pose(self):
        for options in ({'inconsistent':True},{'bad_after':2}):
            with self.subTest(options=options), TemporaryDirectory() as tmp:
                root=Path(tmp)
                with self.assertRaisesRegex(ValueError,'Tracking lost'):
                    self.run_pipeline(root,**options)
                report=json.loads((root/'posed/pose_quality_report.json').read_text())
                self.assertEqual(report['status'],'BLOCKED')
                self.assertFalse((root/'posed/pose').exists())
                self.assertFalse((root/'raw/derived/latest_posed_dataset.txt').exists())
