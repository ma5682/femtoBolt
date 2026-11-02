import copy,json,tempfile,unittest,time
from pathlib import Path
from unittest import mock
from types import SimpleNamespace
import numpy as np
import cv2
from tests.test_fixed_rgb_experience import make_model,K,quad
from tests.test_segmented_capture import make_frame
from femto.local.local_geometry_diagnostics import diagnose,render_comparison,save_diagnostics,digest
from femto.tracking.local_tracking_diagnostics import current_feedback
from femto.tracking.scene_live_tracker import LiveTracker
from femto.capture.local_capture import LocalCapture

class CaptureDiagnosticsTests(unittest.TestCase):
    def test_early_sync_rejection_has_current_quality_and_update_time(self):
        e=mock.Mock();t=LiveTracker(e,250,2210,clock=lambda:10.)
        f=make_frame([0]*10,1);f.color_timestamp_us=f.depth_timestamp_us+6000
        r=t.process(f,0,9.9)
        self.assertEqual(r['rejection_reason'],'rgb_depth_sync');self.assertEqual(r['quality']['valid_depth_fraction'],0.)
        self.assertIn('status_updated_at',r);self.assertIsNone(r['T_WC']);e.prepare.assert_not_called()
    def test_recovery_does_not_display_historical_failure(self):
        r=dict(state='tracking',received_monotonic=10.,selected_for_fusion=True,pair_failure={'reason':'old'},
            quality={'valid_depth_fraction':.8},status_updated_at='fresh',reference_frame_id=7)
        d=current_feedback(r,3,10.2);self.assertIsNone(d['current_reason']);self.assertEqual(d['updated_at'],'fresh')
        d=current_feedback(r,3,13.);self.assertEqual(d['current_reason'],'stale_frame');self.assertFalse(d['selected_for_fusion'])
    def test_bad_pose_quality_or_changed_reference_never_selected(self):
        c=LocalCapture.__new__(LocalCapture);c.accepted=[{'reference_frame_id':7}]
        good=dict(state='tracking',T_WC=np.eye(4).tolist(),reference_frame_id=7,quality={'usable':True},continuity={'current_frame_verified':True})
        for change in ({'reference_frame_id':8},{'T_WC':np.full((4,4),np.nan).tolist()},{'quality':{'usable':False}},{'continuity':{'current_frame_verified':False}}):
            r=dict(good,**change);self.assertFalse(c._select(r,10.));self.assertEqual(r['selection']['reason'],'pose_quality_or_reference_mismatch')
    def test_world_point_reprojects_through_inverse_camera_pose(self):
        from femto.local.local_geometry_diagnostics import sample_view
        t=np.eye(4);t[0,3]=.03
        r=sample_view([[.03,0,.5]],t,K,np.full((100,100),500,np.uint16),.25,2.21)
        np.testing.assert_allclose(r['pixel'],[50,50]);self.assertEqual(r['signed_residual_mm'],0.)
    def test_invalid_composed_pose_does_not_advance_origin(self):
        e=mock.Mock();e.estimate.return_value=(np.full((4,4),np.nan),{})
        t=LiveTracker(e,250,2210,clock=lambda:10.)
        with mock.patch('femto.tracking.scene_live_tracker.frame_quality',return_value={'usable':True}):
            t.process(make_frame([500]*10,1),7,9.8);r=t.process(make_frame([500]*10,2),8,9.9)
        self.assertEqual(r['state'],'uncertain');self.assertEqual(r['rejection_reason'],'invalid_composed_pose')
        self.assertEqual(t.reference_id,7);self.assertEqual(t.last_valid_frame_id,7);np.testing.assert_array_equal(t.pose,np.eye(4))

class LocalEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.model=make_model(self.root)
        self.dataset=self.root/'dataset';self.dataset.mkdir();(self.dataset/'derived').mkdir();(self.dataset/'calibration').mkdir()
        from femto.dataset.local_reference import CALIBRATION_FILES,calibration_identity
        for name in CALIBRATION_FILES:(self.dataset/'calibration'/name).write_text('{}')
        def write(name,data):(self.dataset/name).write_text(json.dumps(data),encoding='utf-8')
        frames=[];records=[]
        for i in range(2):
            (self.dataset/f'rgb{i}.png').write_bytes(cv2.imencode('.png',np.full((100,100,3),140,np.uint8))[1].tobytes())
            self.set_depth(i,500)
            pose=dict(frame=i,capture_timestamp_us=i+100,T_WC=np.eye(4).tolist(),pose_valid=True,session_id='synthetic-fixture',reference_frame_id=0)
            write(f'pose{i}.json',pose)
            frames.append(dict(frame=i,rgb=f'rgb{i}.png',depth=f'depth{i}.png',pose=f'pose{i}.json',pose_valid=True,camera_timestamp_us=i+100))
            records.append(dict(session_id='synthetic-fixture',calibration_sha256=calibration_identity(self.dataset)['sha256'],frame_id=i,state='tracking',selected_for_fusion=True,T_WC=pose['T_WC'],reference_frame_id=0,capture_timestamp_us=i+100))
        self.frames=frames;self.records=records
        (self.dataset/'frames.jsonl').write_text('\n'.join(map(json.dumps,frames)))
        (self.dataset/'derived/live_tracking.jsonl').write_text('\n'.join(map(json.dumps,records)))
        write('manifest.json',{'local_capture':{'session_id':'synthetic-fixture'}})
        write('derived/local_capture_report.json',dict(session_id='synthetic-fixture',reference_frame_id=0,accepted_frames=[0,1]))
        local=dict(session_id='synthetic-fixture',reference_frame_id=0,reference_provenance={'calibration':calibration_identity(self.dataset)},support={'tolerance_m':.01},tracking_report=str(self.dataset/'derived/local_capture_report.json'))
        (self.model.reconstruction/'local_geometry.json').write_text(json.dumps(local))
        (self.model.reconstruction/'reconstruction_report.json').write_text(json.dumps(dict(source_dataset=str(self.dataset),frames=frames)))
        self.model.store.origin_reference['source_dataset']=str(self.dataset)
        self.model.view_reference.update(depth_sha256=digest(self.dataset/'depth0.png'),rgb_sha256=digest(self.dataset/'rgb0.png'))
        yy,xx=np.indices((100,100));self.patch=mock.patch('femto.geometry.rgbd_frames.distortion_maps',return_value=(xx.astype(np.float32),yy.astype(np.float32),K,100,100));self.patch.start()
        self.point=self.model.pick(50,50,'test','')
    def tearDown(self):self.patch.stop();self.tmp.cleanup()
    def set_depth(self,index,mm):
        (self.dataset/f'depth{index}.png').write_bytes(cv2.imencode('.png',np.full((100,100),mm,np.uint16))[1].tobytes())
    def test_multi_single_and_missing_reference_are_independent(self):
        r=diagnose(self.model,self.point);self.assertEqual(r['support_kind'],'multi_frame_supported');self.assertEqual(r['support_frame_count'],2)
        self.set_depth(1,600);r=diagnose(self.model,self.point);self.assertEqual(r['support_kind'],'single_frame_supported');self.assertAlmostEqual(r['observations'][1]['signed_residual_mm'],100.)
        self.set_depth(0,0);self.set_depth(1,500);self.model.projection.depth[:]=0;self.model.view_reference['depth_sha256']=digest(self.dataset/'depth0.png')
        r=diagnose(self.model,self.point);self.assertTrue(r['reference_depth_missing']);self.assertEqual(r['support_frame_count'],1)
    def test_unknown_pose_missing_asset_and_reference_tamper(self):
        p=self.dataset/'pose1.json';original=p.read_text();pose=json.loads(original);pose['reference_frame_id']=99;p.write_text(json.dumps(pose))
        self.assertEqual(diagnose(self.model,self.point)['support_kind'],'source_unknown');p.write_text(original)
        (self.dataset/'depth1.png').unlink();r=diagnose(self.model,self.point);self.assertEqual(r['support_kind'],'source_unknown');self.assertEqual(r['checked_frame_count'],1)
        self.set_depth(1,500);self.model.view_reference['depth_sha256']='wrong';r=diagnose(self.model,self.point);self.assertEqual(r['observations'][0]['status'],'unknown')
    def test_unaccepted_frame_not_counted_and_no_geometry_causal_claim(self):
        self.records[1]['selected_for_fusion']=False;(self.dataset/'derived/live_tracking.jsonl').write_text('\n'.join(map(json.dumps,self.records)))
        r=diagnose(self.model,self.point);self.assertEqual(r['support_frame_count'],1);self.assertEqual(r['support_kind'],'source_unknown')
        self.assertEqual(r['diagnosis']['fusion_ghosting']['status'],'not_established');self.assertEqual(r['diagnosis']['texture_alignment']['status'],'not_measured')
    def test_report_export_atomic_failure_and_cancel(self):
        r=diagnose(self.model,self.point);image=render_comparison(self.model,r);folder=save_diagnostics(r,image,self.root/'reports')
        self.assertEqual(json.loads((folder/'report.json').read_text(encoding='utf-8')),r)
        before=list((self.root/'reports').iterdir())
        with mock.patch('os.replace',side_effect=OSError('disk full')):
            with self.assertRaises(OSError):save_diagnostics(r,image,self.root/'reports')
        self.assertEqual(list((self.root/'reports').iterdir()),before)
        with self.assertRaisesRegex(RuntimeError,'cancelled'):diagnose(self.model,self.point,check=lambda:(_ for _ in ()).throw(RuntimeError('cancelled')))

class GeometryProbeTests(unittest.TestCase):
    def test_two_sheets_are_geometry_evidence_not_a_ghosting_diagnosis(self):
        with tempfile.TemporaryDirectory() as tmp:
            mesh=quad();mesh+=quad(z=.53)
            model=make_model(Path(tmp),mesh=mesh);point=model.pick(50,50,'layers','')
            r=diagnose(model,point)
            self.assertEqual(r['diagnosis']['geometry_ray_layers']['status'],'multiple_intersections_observed')
            np.testing.assert_allclose(r['diagnosis']['geometry_ray_layers']['depths_m'],[.5,.53],atol=1e-6)
            self.assertEqual(r['diagnosis']['fusion_ghosting']['status'],'not_established')
            self.assertEqual(r['diagnosis']['texture_alignment']['status'],'not_measured')
            self.assertEqual(r['support_kind'],'source_unknown')


if __name__=='__main__':unittest.main()
