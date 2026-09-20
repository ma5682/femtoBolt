"""State-machine/geometry fixtures; no physical-accuracy claims."""
import json
import threading
import unittest
from unittest import mock
from types import SimpleNamespace
import numpy as np
import open3d as o3d
from femto.tracking.scene_live_tracker import LiveTracker,PairEstimator,PreparedRGBD,PairEstimationError
from femto.capture.local_capture import LocalCapture
from femto.tracking.local_stability_feedback import CandidateStability
from tests.test_segmented_capture import make_frame


def pose(x=0.,angle=0.):
    t=np.eye(4);t[:3,:3]=o3d.geometry.get_rotation_matrix_from_xyz([.01*angle,.02*angle,.03*angle]);t[0,3]=x
    return t


def prepared():
    return PreparedRGBD(None,SimpleNamespace(intrinsic_matrix=np.eye(3)),None,np.zeros((12,16,3),np.uint8),np.full((12,16),500,np.uint16))


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.now=10.;self.est=mock.Mock();self.est.feature_seed=False
        self.est.prepare.side_effect=lambda f:prepared()
        self.est.estimate.return_value=(np.eye(4),{})
        self.est.feature_seed_transform.return_value=(np.eye(4),dict(inliers=20))
        self.tr=LiveTracker(self.est,250,2210,clock=lambda:self.now,keyframe_anchor=True,relocalization=True)
        self.quality=mock.patch('femto.tracking.scene_live_tracker.frame_quality',return_value=dict(usable=True));self.quality.start();self.addCleanup(self.quality.stop)
        self.frame_id=0
    def step(self,delta=.1,**kw):
        self.now+=delta;f=make_frame([500]*10,self.frame_id+1)
        for k,v in kw.items():setattr(f,k,v)
        r=self.tr.process(f,self.frame_id,self.now);self.frame_id+=1;return r
    def origin(self):
        self.step();self.assertTrue(self.tr.add_keyframe(self.tr.previous,self.tr.pose,self.tr.last_valid_frame_id))
    def lose(self):
        self.origin();self.now+=2.1;r=self.tr.watchdog('tracking_gap')
        self.assertEqual(r['state'],'relocalizing');self.assertIsNone(r['T_WC']);return r
    def test_two_fresh_confirmations_preserve_reference_and_reset_motion(self):
        self.lose();first=self.step();self.assertEqual(first['state'],'relocalizing');self.assertIsNone(first['T_WC'])
        second=self.step();self.assertEqual(second['state'],'tracking');self.assertTrue(second['continuity']['relocalized'])
        self.assertEqual(second['reference_frame_id'],0);self.assertTrue(second['motion_discontinuity'])
        self.assertIsNone(self.tr.previous_step);json.dumps(second,allow_nan=False)
    def test_bad_sync_breaks_confirmation_run(self):
        self.lose();self.step();bad=self.step(color_timestamp_us=9999999)
        self.assertEqual(bad['rejection_reason'],'rgb_depth_sync');self.assertFalse(self.tr.relocalization_candidates)
        self.assertEqual(self.step()['state'],'relocalizing');self.assertEqual(self.step()['state'],'tracking')
    def test_repeated_timestamp_does_not_count(self):
        self.lose();r=self.step();t=r['capture_timestamp_us']
        bad=self.step(timestamp_us=t)
        self.assertEqual(bad['rejection_reason'],'nonmonotonic_timestamp');self.assertIsNone(bad['T_WC'])
    def test_expired_input_cannot_attempt_recovery(self):
        self.lose();self.now+=3
        r=self.tr.process(make_frame([500]*10,20),20,self.now-2.1)
        self.assertEqual(r['rejection_reason'],'stale_frame');self.est.feature_seed_transform.assert_not_called()
    def test_sparse_confirmations_do_not_recover(self):
        self.lose();self.step();r=self.step(delta=2.1)
        self.assertEqual(r['state'],'relocalizing');self.assertEqual(len(self.tr.relocalization_candidates),1)
    def test_watchdog_timeout_keeps_original_reason(self):
        self.lose();self.now+=15.;r=self.tr.watchdog()
        self.assertEqual(r['state'],'lost');self.assertEqual(r['loss_context']['original_reason'],'tracking_gap')
        self.assertEqual(r['loss_context']['reason'],'relocalization_timeout')
    def test_computation_crossing_deadline_cannot_recover(self):
        self.lose();self.step();self.now=self.tr.relocalizing_since+14.8
        def slow(*a,**k):self.now+=.5;return np.eye(4),{}
        self.est.estimate.side_effect=slow;r=self.step(delta=.01)
        self.assertEqual(r['state'],'lost');self.assertIsNone(r['T_WC'])
    def test_remaining_window_includes_processing_time(self):
        self.lose();received=self.now+.1;self.now=received
        def slow(*a,**k):self.now+=2.;return np.eye(4),{}
        self.est.estimate.side_effect=slow
        r=self.tr.process(make_frame([500]*10,20),20,received)
        expected=self.tr.relocalization_window_s-(self.now-self.tr.relocalizing_since)
        self.assertEqual(r['state'],'relocalizing')
        self.assertAlmostEqual(r['continuity']['remaining_window_s'],expected)
        self.assertAlmostEqual(r['relocalization']['remaining_window_s'],expected)
    def test_conflicting_keyframe_hypotheses_rejected(self):
        self.lose();self.tr.keyframes.append(dict(frame_id=2,pose=pose(.03),prepared=prepared()))
        r=self.step();self.assertEqual(r['rejection_reason'],'relocalization_ambiguous')
        self.est.estimate.assert_not_called()
    def test_dense_solution_cannot_silently_disagree_with_seed(self):
        self.lose();self.est.estimate.return_value=(pose(.04),{})
        r=self.step();self.assertEqual(r['rejection_reason'],'relocalization_seed_disagreement');self.assertIsNone(r['T_WC'])
    def test_keyframe_identity_and_pose_must_match_current_verified_frame(self):
        self.origin();self.assertFalse(self.tr.add_keyframe(self.tr.previous,self.tr.pose,0))
        self.assertFalse(self.tr.add_keyframe(prepared(),self.tr.pose,0))
        self.assertFalse(self.tr.add_keyframe(self.tr.previous,pose(.01),0))
    def test_anchor_prediction_direction_with_noncommuting_transforms(self):
        self.origin();kf=pose(.02,1.);candidate=pose(.04,-2.)
        self.tr.keyframes[0]['pose']=kf;self.tr.previous=prepared()
        forward=np.linalg.inv(candidate)@kf;self.est.estimate.return_value=(forward,{})
        adopted,report=self.tr._anchor(prepared(),candidate)
        np.testing.assert_allclose(self.est.estimate.call_args.kwargs['initial_transform'],forward)
        np.testing.assert_allclose(adopted,candidate,atol=1e-12);self.assertTrue(report['used'])
    def test_adopted_step_includes_anchor_correction(self):
        self.origin();self.step();self.est.estimate.side_effect=[(np.eye(4),{}),(pose(-.006),{})]
        r=self.step();self.assertAlmostEqual(r['step_m'],.006);self.assertEqual(r['frame_pair_step_m'],0.)
        self.assertTrue(r['motion_discontinuity'])
    def test_anchor_conflict_rejects_and_failed_anchor_falls_back(self):
        self.origin();self.step();self.est.estimate.side_effect=[(np.eye(4),{}),(pose(-.03),{})]
        r=self.step();self.assertEqual(r['rejection_reason'],'anchor_disagreement');self.assertIsNone(r['T_WC'])
        self.est.estimate.side_effect=[(np.eye(4),{}),PairEstimationError('no_overlap')]
        r=self.step();self.assertEqual(r['state'],'tracking');self.assertFalse(r['anchor']['used'])
    def test_recovery_frame_not_admitted_and_pending_request_cleared(self):
        self.lose();self.step();r=self.step()
        c=LocalCapture.__new__(LocalCapture);c.lock=threading.Lock();c.accepted=[dict(reference_frame_id=0)];c.requested=1.;c.stability=CandidateStability('session')
        self.assertFalse(c._select(r,self.now,self.tr.previous));self.assertIsNone(c.requested)
        self.assertEqual(r['selection']['reason'],'pose_correction_recheck')
        self.assertEqual(c.stability.observe(r,self.now)['state'],'reset')
    def test_invalid_configuration_rejected(self):
        for kw in [dict(relocalization_confirmations=1),dict(relocalization_window_s=float('nan')),dict(relocalization_window_s=16.)]:
            with self.assertRaises(ValueError):LiveTracker(self.est,250,2210,**kw)


class FeatureRetryTests(unittest.TestCase):
    def test_feature_is_third_seed_and_never_returned_as_solution(self):
        est=PairEstimator.__new__(PairEstimator);est.feature_seed=True
        est.feature_seed_transform=mock.Mock(return_value=(pose(.02),dict(inliers=20)))
        est._estimate_once=mock.Mock(side_effect=[PairEstimationError('bad'),PairEstimationError('bad'),(pose(.021),{})])
        result,report=est.estimate(prepared(),prepared(),initial_transform=pose(.01))
        self.assertEqual([r['initialization'] for r in report['attempts']],['identity','previous_valid_step','feature_pnp'])
        self.assertAlmostEqual(result[0,3],.021)
    def test_feature_disabled_by_caller_or_not_needed(self):
        est=PairEstimator.__new__(PairEstimator);est.feature_seed=True;est.feature_seed_transform=mock.Mock()
        est._estimate_once=mock.Mock(return_value=(np.eye(4),{}));est.estimate(prepared(),prepared());est.feature_seed_transform.assert_not_called()
        est._estimate_once.side_effect=PairEstimationError('bad')
        with self.assertRaises(PairEstimationError):est.estimate(prepared(),prepared(),allow_feature_seed=False)
        est.feature_seed_transform.assert_not_called()

    def test_identity_feature_seed_does_not_repeat_identity_dense_attempt(self):
        est=PairEstimator.__new__(PairEstimator);est.feature_seed=True
        est.feature_seed_transform=mock.Mock(return_value=(np.eye(4),dict(inliers=20)))
        est._estimate_once=mock.Mock(side_effect=PairEstimationError('bad'))
        with self.assertRaises(PairEstimationError) as caught:
            est.estimate(prepared(),prepared())
        self.assertEqual(est._estimate_once.call_count,1)
        self.assertEqual(len(caught.exception.attempts),1)

class CaptureRecoveryIntegrationTests(unittest.TestCase):
    def test_worker_keeps_window_open_and_requires_durable_recovery(self):
        import gc,queue,tempfile,time
        from pathlib import Path
        from tests.test_segmented_capture import make_camera
        gc.collect()
        with tempfile.TemporaryDirectory() as temp, mock.patch('femto.capture.local_capture.PairEstimator') as factory, mock.patch('femto.tracking.scene_live_tracker.frame_quality',return_value=dict(usable=True)), mock.patch('femto.capture.local_capture.ViewSelector') as selector:
            est=factory.return_value;est.feature_seed=False;est.prepare.side_effect=lambda f:prepared()
            est.estimate.side_effect=[PairEstimationError('pair_quality')]*5+[(np.eye(4),{})]*3
            est.feature_seed_transform.return_value=(np.eye(4),dict(inliers=20))
            selector.return_value.evaluate.return_value=(object(),dict(state='accepted',reason='fixture'))
            events=queue.Queue();root=Path(temp)
            capture=LocalCapture(make_camera('wide_detail',10),root/'data',root/'out',events,tracker_options=dict(relocalization=True))
            def submit(index):
                capture.submit(make_frame([500]*10,index));deadline=time.monotonic()+6
                while time.monotonic()<deadline:
                    event,payload=events.get(timeout=6)
                    if event=='local_error':self.fail(str(payload))
                    if event=='local_pose' and payload['record'].get('frame_id')==index-1:return payload['record']
                self.fail('No pose record')
            try:
                self.assertEqual(submit(1)['state'],'tracking')
                for i in range(2,7):r=submit(i)
                self.assertEqual(r['state'],'relocalizing');self.assertFalse(capture.closing.is_set())
                self.assertEqual(submit(7)['state'],'relocalizing')
                recovered=submit(8);self.assertTrue(recovered['continuity']['relocalized'])
                self.assertFalse(recovered['selected_for_fusion']);self.assertEqual(len(capture.accepted),1)
                self.assertIn('raw_write_receipt',recovered)
            finally:capture.finish(False);capture.thread.join(8)
            self.assertFalse(capture.thread.is_alive())
            report=json.loads((capture.session/'derived/local_capture_report.json').read_text())
            self.assertTrue(report['relocalization']);self.assertEqual(report['relocalization_events'][-1]['event'],'recovered')
            self.assertFalse(capture.successful_model_published)

class RealFeatureGeometryTests(unittest.TestCase):
    def fixture(self):
        import cv2
        rng=np.random.default_rng(92);height,width=384,512
        color=rng.integers(0,256,(height,width,3),dtype=np.uint8)
        color=cv2.GaussianBlur(color,(3,3),.5)
        k=np.array([[400.,0,255.5],[0,400.,191.5],[0,0,1.]])
        truth=pose(.012,.15);truth[1,3]=.004;truth[2,3]=.003;z=1.2
        h=k@(truth[:3,:3]+np.outer(truth[:3,3],[0,0,1.])/z)@np.linalg.inv(k)
        current=cv2.warpPerspective(color,h,(width,height))
        yy,xx=np.indices((height,width));rays=np.stack(((xx-k[0,2])/400,(yy-k[1,2])/400,np.ones_like(xx)),axis=-1)
        normal=truth[:3,2];depth=(z+normal@truth[:3,3])/(rays@normal)
        intrinsic=SimpleNamespace(intrinsic_matrix=k)
        a=PreparedRGBD(None,intrinsic,None,color,np.full((height,width),1200,np.uint16))
        b=PreparedRGBD(None,intrinsic,None,current,np.rint(depth*1000).astype(np.uint16))
        est=PairEstimator.__new__(PairEstimator);est.minimum=.25;est.maximum=2.21
        return est,a,b,truth
    def test_orb_pnp_real_backend_on_known_planar_motion(self):
        from femto.tracking.moving_quality import rotation_deg
        est,a,b,truth=self.fixture();result,report=est.feature_seed_transform(a,b)
        self.assertIsNotNone(result,report);delta=np.linalg.inv(truth)@result
        self.assertLess(np.linalg.norm(delta[:3,3]),.01);self.assertLess(rotation_deg(delta),2.)
        self.assertGreaterEqual(min(report['inlier_grid_cells']),4)
    def test_depth_disagreement_and_textureless_frame_cannot_seed(self):
        est,a,b,_=self.fixture();b.depth_mm[:]=500
        result,report=est.feature_seed_transform(a,b)
        self.assertIsNone(result);self.assertEqual(report['reason'],'insufficient_distributed_depth_evidence')
        blank=prepared();blank.color_bgr[:]=0
        result,report=est.feature_seed_transform(blank,blank);self.assertIsNone(result)

class SyntheticReplayTests(unittest.TestCase):
    def test_four_frame_replay_is_read_only_and_records_variants(self):
        from pathlib import Path
        import tempfile,shutil
        from femto.pipeline.synthetic_scene_demo import generate_dataset
        from femto.tracking.local_tracking_replay import replay
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);generated=generate_dataset(root/'generated',frames=4,width=512,height=288)
            source=root/'synthetic_source';source.mkdir();(source/'derived').mkdir()
            shutil.copytree(generated/'calibration',source/'calibration')
            initial=json.loads((generated/'frames.jsonl').read_text().splitlines()[0]);rows=[];log=[]
            for index in range(4):
                row=dict(initial,frame=index,camera_frame_index=index,camera_timestamp_us=100000+index*100000,color_timestamp_us=100000+index*100000,depth_timestamp_us=100000+index*100000)
                for key in ('rgb','depth','pose'):row[key]=str(generated/initial[key])
                rows.append(row);log.append(dict(frame_id=index,capture_timestamp_us=row['camera_timestamp_us'],received_monotonic=10+index*.1,state='tracking',T_WC=np.eye(4).tolist(),selected_for_fusion=index==0))
            (source/'frames.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            (source/'derived/live_tracking.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in log))
            (source/'derived/local_capture_report.json').write_text('{}')
            (source/'manifest.json').write_text(json.dumps(dict(purpose='synthetic stationary repeated observation, not physical data',parameters=dict(depth_min_mm=500,depth_max_mm=3860))))
            report=replay(source,root/'replayed',variants=('optimized','all'))
            self.assertTrue(report['source_unchanged']);self.assertEqual(set(report['variants']),{'optimized','all'})
            self.assertFalse(report['physical_accuracy_verified'])
            self.assertEqual(report['variants']['optimized']['valid_pose_frames'],4)

if __name__=='__main__':unittest.main()
