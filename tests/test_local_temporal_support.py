import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import numpy as np
from femto.local.local_temporal_support import support_mask,stationary_window,filtered_depth,prepare
from femto.io_utils import imwrite_unicode
from test_local_temporal_binding import make_source


def records():
    return [dict(frame_id=i,state='tracking',T_WC=np.eye(4).tolist(),reference_frame_id=0,
                 capture_timestamp_us=i*500000+1,color_timestamp_us=i*500000+1,
                 depth_timestamp_us=i*500000+1) for i in range(5)]


class SupportTests(unittest.TestCase):
    def test_report_distinguishes_stationary_windows_without_accepted_frame(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);dataset,raw=make_source(root,accepted=(0,),outlier=False)
            tracking=raw/'derived/live_tracking.jsonl'
            rows=[json.loads(line) for line in tracking.read_text().splitlines() if line.strip()]
            # Frame 0 is the only accepted frame.  Frames 1-4 deliberately move;
            # frames 5-9 form a valid stationary window that cannot support it.
            for index,row in enumerate(rows):
                row['T_WC'][0][3]=index*.003
            for index in range(5,10):
                timestamp=1_000_000+index*400_000
                rows.append(dict(frame_id=index,state='tracking',session_id='temporal-source-fixture',
                    reference_frame_id=0,T_WC=np.eye(4).tolist(),capture_timestamp_us=timestamp,
                    color_timestamp_us=timestamp,depth_timestamp_us=timestamp,latency_ms=10.,
                    quality=dict(usable=True)))
            tracking.write_text(''.join(json.dumps(row)+'\n' for row in rows),encoding='utf-8')
            derived,report=prepare(dataset,root/'window-report')
            self.assertEqual(derived,dataset.resolve())
            self.assertEqual(report['stationary_windows_total'],1)
            self.assertEqual(report['stationary_windows_containing_accepted'],0)
            self.assertEqual(report['window_miss_interpretation'],
                             'stationary_windows_exist_but_do_not_contain_accepted_frames')
            self.assertEqual(report['all_stationary_windows'][0]['source_frame_ids'],[5,6,7,8,9])
            status=report['frames'][0]['candidate_status']
            self.assertIn('candidate_frame_ids',status)
            self.assertIn('nearby_relative_pose',status)
            self.assertIn('stopped_after',status)

    def test_reconstruction_uses_mask_and_keeps_initial_raw_cloud(self):
        from femto.pipeline.scene_reconstruction import reconstruct_scene
        from femto.geometry.fusion import read_geometry_safe
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            raw_depth=np.full((96,128),513,np.uint16);raw_depth[:,:64]=0;raw_depth[32:40,24:32]=513
            posed,_=make_source(root,depth=raw_depth,outlier=False)
            motion=np.zeros(raw_depth.shape,bool);motion[:,:64]=True
            # The original left-half integration assertions stay intact. This
            # source has only 64 valid pixels there, so the real >=80% retention
            # gate is also respected instead of constructing an unsupported mask.
            with mock.patch('femto.local.local_temporal_support.audit_temporal_motion',return_value=(motion,dict(passed=True))):
                source,temporal=prepare(posed,root/'masks')
            self.assertEqual(temporal['applied_frames'],3)
            rows=[json.loads(s) for s in (source/'frames.jsonl').read_text().splitlines()]
            report=reconstruct_scene(source,root/'result',depth_min_m=.25,depth_max_m=2.21)
            initial=read_geometry_safe(root/'result/initial_observed_points_m.ply',mesh=False)
            self.assertGreater(len(initial.points),0)
            self.assertGreater(report['metrics']['fused_point_count'],0)
            self.assertTrue(all(r.get('local_temporal_support') for r in rows))
            # Inspect integration calls rather than inferring support from mesh appearance.
            import femto.pipeline.scene_reconstruction as reconstruction
            original=reconstruction._prepare_rgbd;observations=[]
            def observe(color,depth,*args,**kwargs):
                observations.append(depth.copy());return original(color,depth,*args,**kwargs)
            with mock.patch.object(reconstruction,'_prepare_rgbd',side_effect=observe):
                reconstruct_scene(source,root/'result2',depth_min_m=.25,depth_max_m=2.21)
            self.assertTrue(any(np.all(d[:,:64]==0) and np.any(d[:,64:]>0) for d in observations))
            self.assertTrue(any(np.any(d[:,:64]>0) for d in observations))

    def test_reject_outlier_motion_and_missing_without_changing_retained_values(self):
        depths=[np.full((8,8),500,np.uint16) for _ in range(5)]
        depths[2][0,0]=550;depths[2][1,1]=0;depths[1][2,2]=0;depths[3][2,2]=0
        depths[2][3,3]=503
        motion=np.zeros((8,8),bool);motion[4,4]=True
        before=depths[2].copy();keep=support_mask(depths,2,motion)
        self.assertFalse(keep[0,0]);self.assertFalse(keep[1,1]);self.assertFalse(keep[2,2]);self.assertFalse(keep[4,4])
        self.assertTrue(keep[3,3]);np.testing.assert_array_equal(before,depths[2])
    def test_moving_lost_unsynced_and_duplicate_timestamps_reject_window(self):
        self.assertIsNotNone(stationary_window(records(),2))
        for change in ('motion','lost','sync','timestamp'):
            r=records()
            if change=='motion':r[0]['T_WC'][0][3]=.002
            elif change=='lost':r[0]['state']='lost'
            elif change=='sync':r[0]['color_timestamp_us']+=6000
            else:r[1]['capture_timestamp_us']=r[0]['capture_timestamp_us']
            self.assertIsNone(stationary_window(r,2))
    def test_mask_hash_mismatch_blocks_and_retained_depth_unchanged(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);raw=np.full((5,5),513,np.uint16)
            posed,_=make_source(root,depth=raw,outlier=False)
            motion=np.zeros(raw.shape,bool);motion[0,0]=True
            with mock.patch('femto.local.local_temporal_support.audit_temporal_motion',return_value=(motion,dict(passed=True))):
                dataset,_=prepare(posed,root/'masks')
            frame=next(json.loads(s) for s in (dataset/'frames.jsonl').read_text().splitlines() if json.loads(s)['frame']==2)
            path=Path(frame['local_temporal_support']['mask']);result=filtered_depth(dataset,frame,raw)
            self.assertEqual(result[0,0],0);np.testing.assert_array_equal(result[1:],raw[1:]);self.assertEqual(raw[0,0],513)
            path.write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'hash'):filtered_depth(dataset,frame,raw)
    def test_prepare_exports_masks_preserves_pose_and_source_or_skips_motion(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);dataset,raw=make_source(root,accepted=(0,),outlier_index=0)
            frame=json.loads((dataset/'frames.jsonl').read_text())
            before={p:p.read_bytes() for p in root.rglob('*') if p.is_file()}
            with mock.patch('femto.local.local_temporal_support.audit_temporal_motion',return_value=(np.zeros((20,20),bool),dict(passed=True))):
                derived,report=prepare(dataset,root/'out')
            self.assertEqual(report['applied_frames'],1);self.assertTrue(report['source_unchanged'])
            row=json.loads((derived/'frames.jsonl').read_text());d=np.full((20,20),500,np.uint16);d[0,0]=550
            self.assertEqual(filtered_depth(derived,row,d)[0,0],0)
            self.assertEqual((derived/row['pose']).read_bytes(),before[dataset/frame['pose']])
            self.assertTrue(all(p.read_bytes()==value for p,value in before.items()))
            with mock.patch('femto.local.local_temporal_support.audit_temporal_motion',return_value=(np.zeros((20,20),bool),dict(passed=False))):
                derived,report=prepare(dataset,root/'skip')
            self.assertEqual(derived,dataset.resolve());self.assertEqual(report['applied_frames'],0)
