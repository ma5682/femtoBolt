from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, mock
import json

import numpy as np

from femto.depth.temporal_depth_fusion import confidence_weighted_median, fuse_temporal_depth
from femto.temporal.scene_temporal_fusion import build_temporal_dataset, audit_temporal_motion
from femto.io_utils import imwrite_unicode


class TemporalDepthTests(TestCase):
    def stack(self,n=5): return [np.full((96,128),1000,np.uint16) for _ in range(n)]

    def test_weighted_median_selects_observation_not_average(self):
        self.assertEqual(confidence_weighted_median(np.array([1000,1002,1001,3000]),np.ones(4)),1001)
        self.assertEqual(confidence_weighted_median(np.array([1000,2000]),np.array([.3,1.])),2000)

    def test_stable_real_values_get_one_and_one_off_dropout_recovers(self):
        depths=self.stack(); depths[0][30,40]=0
        fused,c,m,r=fuse_temporal_depth(depths,250,2210)
        self.assertEqual(fused[30,40],1000)
        self.assertEqual(c[0,0],1)
        self.assertAlmostEqual(float(c[30,40]),.8)
        self.assertEqual(m['real_support'][30,40],4)
        self.assertEqual(r['point_count'],fused.size)

    def test_one_real_sample_cannot_be_stable(self):
        depths=self.stack()
        for d in depths[1:]: d[30,40]=0
        fused,c,_,_=fuse_temporal_depth(depths,250,2210)
        self.assertEqual(fused[30,40],0); self.assertEqual(c[30,40],0)

    def test_outlier_rejected_but_competing_surfaces_unknown(self):
        depths=self.stack(); depths[0][30,40]=2000
        fused,c,_,report=fuse_temporal_depth(depths,250,2210)
        self.assertEqual(fused[30,40],1000)
        self.assertLess(c[30,40],1)
        self.assertEqual(report['removed_outlier_observations'],1)
        depths[1][30,40]=2000
        fused,c,_,_=fuse_temporal_depth(depths,250,2210)
        self.assertEqual(fused[30,40],0)

    def test_five_material_patterns_and_counts(self):
        for n in (5,10,15):
            for name in ('table','computer_shell','black_object','phone_glass','transparent'):
                with self.subTest(n=n,name=name):
                    depths=self.stack(n)
                    if name=='transparent':
                        for d in depths: d[20:50,20:50]=0
                    elif name=='phone_glass':
                        for i,d in enumerate(depths): d[20:50,20:50]=800 if i%2 else 1600
                    else:
                        for i,d in enumerate(depths):
                            d[:]=1000+(i%3)-1
                            d[20+i%3,30]=0
                    before=[d.copy() for d in depths]
                    fused,c,_,_=fuse_temporal_depth(depths,250,2210)
                    if name in ('transparent','phone_glass'):
                        self.assertTrue(np.all(fused[20:50,20:50]==0))
                    else:
                        self.assertGreater(np.count_nonzero(fused),np.count_nonzero(depths[0]))
                        self.assertLessEqual(np.max(np.abs(fused.astype(float)-1000)),1)
                    for a,b in zip(depths,before): np.testing.assert_array_equal(a,b)

    def test_local_motion_excluded(self):
        mask=np.zeros((96,128),bool); mask[30:35,40:45]=True
        fused,c,_,_=fuse_temporal_depth(self.stack(),250,2210,motion_mask=mask)
        self.assertTrue(np.all(fused[mask]==0)); self.assertTrue(np.all(c[mask]==0))

    def fixture(self,root):
        root.mkdir()
        rng=np.random.default_rng(42)
        rgb=rng.integers(0,256,(192,256,3),dtype=np.uint8)
        frames=[]
        for i in range(5):
            d=np.full((192,256),1000,np.uint16); d[50+i,60]=0
            imwrite_unicode(root/f'depth/{i}.png',d); imwrite_unicode(root/f'rgb/{i}.png',rgb)
            frames.append(dict(frame=i,depth=f'depth/{i}.png',rgb=f'rgb/{i}.png',camera_timestamp_us=1000+i))
        (root/'calibration').mkdir()
        (root/'manifest.json').write_text(json.dumps(dict(schema='femto-full-scene-rgbd-v1',parameters=dict(fixed_camera=True),camera=dict(depth_range_m=[.25,2.21]))))
        (root/'frames.jsonl').write_text(''.join(json.dumps(f)+'\n' for f in frames))
        (root/'calibration/color_intrinsic.json').write_text(json.dumps(dict(width=256,height=192,fx=200,fy=200,cx=128,cy=96)))
        (root/'calibration/distortion.json').write_text(json.dumps(dict(rgb=dict(model='BROWN_CONRADY_K6',**{k:0 for k in ('k1','k2','k3','k4','k5','k6','p1','p2')}))))
        return root

    def test_complete_replay_preserves_source_and_continuous_ply(self):
        with TemporaryDirectory() as td:
            source=self.fixture(Path(td)/'scene')
            # Flow requires enough texture at the downsampled resolution.
            output=build_temporal_dataset(source)
            report=json.loads((output/'temporal_report.json').read_text())
            self.assertEqual(report['status'],'completed')
            self.assertTrue(report['source_unchanged'])
            self.assertTrue((output/'fused_pointcloud.ply').exists())
            c=np.load(output/'temporal_confidence.npy')
            self.assertAlmostEqual(float(c[50,60]),.8)
            self.assertTrue((output/'per_frame/000/depth_quality.json').exists())

    def test_motion_failure_saves_report_no_geometry(self):
        with TemporaryDirectory() as td:
            source=self.fixture(Path(td)/'scene')
            with mock.patch('femto.temporal.scene_temporal_fusion.assess_rgb_motion_pair',return_value=dict(reliable=True,appears_fixed=False)):
                output=build_temporal_dataset(source)
            self.assertFalse((output/'fused_pointcloud.ply').exists())
            self.assertIn('rejected',json.loads((output/'temporal_report.json').read_text())['status'])

    def test_repeated_timestamps_rejected(self):
        with TemporaryDirectory() as td:
            source=self.fixture(Path(td)/'scene')
            path=source/'frames.jsonl'
            records=[json.loads(l) for l in path.read_text().splitlines()]
            records[1]['camera_timestamp_us']=records[0]['camera_timestamp_us']
            path.write_text(''.join(json.dumps(r)+'\n' for r in records))
            with self.assertRaises(ValueError): build_temporal_dataset(source)

    def test_capture_uses_bounded_read_burst_before_encoding(self):
        from femto.temporal.scene_temporal_fusion import capture_temporal_frames
        from femto.hardware.orbbec_rgbd import AlignedFrame
        from types import SimpleNamespace
        from femto.geometry.pose_geometry import Intrinsics
        events=[]
        distortion=dict(model='BROWN_CONRADY_K6',**{k:0 for k in ('k1','k2','k3','k4','k5','k6','p1','p2')})
        camera=mock.MagicMock()
        camera.__enter__.return_value=camera
        camera.intrinsics=Intrinsics(128,96,100,100,64,48)
        camera.metadata=dict(depth_range_m=[.25,2.21],depth_mode='WFOV_UNBINNED')
        camera.calibration=dict(rgb_distortion=distortion,depth_distortion=distortion,depth_intrinsic={},depth_to_color={})
        def read(timeout):
            events.append('read'); i=events.count('read')
            return AlignedFrame(np.full((96,128,3),128,np.uint8),np.full((96,128),1000,np.float32),i,i,i,i,i,i,i)
        camera.read.side_effect=read
        def write(*args,**kwargs):
            events.append('encode'); return imwrite_unicode(*args,**kwargs)
        with TemporaryDirectory() as td, mock.patch('femto.hardware.orbbec_rgbd.OrbbecAlignedCamera',return_value=camera), \
             mock.patch('femto.temporal.scene_temporal_fusion.imwrite_unicode',side_effect=write):
            output=capture_temporal_frames(Path(td),5)
            self.assertEqual(events[:5],['read']*5)
            self.assertEqual(events.count('read'),5)
            self.assertEqual(len(list((output/'raw').glob('*depth_mm.png'))),5)
            self.assertFalse((output/'derived').exists())
