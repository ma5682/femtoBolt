import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock
import numpy as np
from femto.temporal.temporal_keyframe import fuse_quality_frames, build_temporal_keyframe
from femto.depth.temporal_depth_layer import prepared_depth
from tests import test_temporal_depth_fusion as fixtures


class KeyframeFusionTests(unittest.TestCase):
    def test_continuous_confidence_and_provenance(self):
        raw=[np.full((16,16),1000,np.uint16) for _ in range(5)]
        for d in raw: d[2,2]=0
        filled=[np.where(d==0,1000,d).astype(np.uint16) for d in raw]
        c=[np.where(d==0,.3,.85).astype(np.float32) for d in raw]
        d,w,m,r=fuse_quality_frames(raw,filled,c,250,2210)
        self.assertAlmostEqual(float(w[0,0]),.85)
        self.assertLessEqual(w[2,2],np.float32(.3))
        self.assertEqual(m['original_support_count'][2,2],0)
        self.assertEqual(m['filled_support_count'][2,2],5)
        raw[0][2,2]=2000; filled[0][2,2]=2000; c[0][2,2]=1
        d,w,_,_=fuse_quality_frames(raw,filled,c,250,2210)
        self.assertEqual(d[2,2],0)

    def test_modes_dropouts_outliers_and_edges(self):
        for n in (1,5,10,15):
            raw=[np.full((32,32),1000,np.uint16) for _ in range(n)]
            for d in raw: d[:,16:]=2000; d[4:8,4:8]=0
            if n>1: raw[0][20,20]=3000; raw[0][10,10]=0
            c=[(d>0).astype(np.float32) for d in raw]
            d,w,m,r=fuse_quality_frames(raw,raw,c,250,4000)
            self.assertTrue(np.all(d[4:8,4:8]==0))
            self.assertEqual(d[10,15],1000); self.assertEqual(d[10,16],2000)
            if n>1:
                self.assertEqual(d[20,20],2000); self.assertLess(w[20,20],1)
                self.assertEqual(d[10,10],1000)

    def test_replay_and_downstream_preserve_weights(self):
        from femto.gsplat.scene_single_gs import prepare
        from femto.pipeline.scene_to_nerfstudio import convert
        from femto.pipeline.scene_reconstruction import reconstruct_scene
        with TemporaryDirectory() as td:
            root=Path(td)
            source=fixtures.TemporalDepthTests().fixture(root/'source')
            dataset,report=build_temporal_keyframe(source)
            from femto.depth.temporal_depth_layer import load_temporal_layer
            layer=load_temporal_layer(dataset.parent)
            layer.validate()
            self.assertTrue(report['source_unchanged'])
            frame=json.loads((dataset/'frames.jsonl').read_text())
            c=np.load(dataset/frame['depth_confidence'])
            self.assertTrue(np.any((c>0)&(c<1)))
            output=prepare(dataset,root/'single')
            prepared=np.load(output/'derived/depth_quality/confidence.npy')
            self.assertTrue(np.any((prepared>0)&(prepared<1)))
            convert(dataset,root/'gs',allow_static=True)
            converted=np.load(root/'gs/derived/confidence/000000.npy')
            np.testing.assert_array_equal(prepared,converted)
            result=reconstruct_scene(dataset,root/'geometry',single_frame_quality=True,depth_min_m=.25,depth_max_m=2.21)
            self.assertIn('temporal_fusion',result)
            self.assertIn(b'property float confidence',(root/'geometry/scene_fused_points_m.ply').read_bytes()[:500])
            with np.load(root/'geometry/derived/tsdf_input/000000.npz') as tsdf_input:
                self.assertTrue(np.any((tsdf_input['confidence']>0)&(tsdf_input['confidence']<1)))
            with self.assertRaisesRegex(ValueError,'phase 2'):
                reconstruct_scene(dataset,root/'deferred_tsdf')

    def test_seed_arrays_uses_continuous_confidence_and_omits_unknown(self):
        from femto.gsplat.scene_single_gs import seed_arrays
        d=np.full((4,4),1000,np.uint16); c=np.full((4,4),.67,np.float32); c[0,0]=0
        xyz,_,_,_,w=seed_arrays(d,np.zeros((4,4,3),np.uint8),np.eye(3),16,c)
        self.assertEqual(len(xyz),15)
        np.testing.assert_array_equal(w,np.full(15,.67,np.float32))

    def test_depth_motion_and_shared_layer(self):
        from femto.temporal.temporal_keyframe import assess_temporal_scene
        from femto.depth.temporal_depth_layer import TemporalDepthLayer
        raw=[np.full((32,32),1000,np.uint16) for _ in range(5)]
        for i,d in enumerate(raw): d[10:15,10:15]=1000 if i%2 else 1500
        rgb=[np.zeros((32,32,3),np.uint8) for _ in raw]
        with mock.patch('femto.temporal.temporal_keyframe.audit_temporal_motion',return_value=(np.zeros((32,32),bool),dict(passed=True))):
            mask,report=assess_temporal_scene(raw,rgb,250,2210)
        self.assertTrue(report['scene_motion_detected'])
        self.assertTrue(mask[12,12]); self.assertFalse(mask[0,0])
        d,c,m,r=fuse_quality_frames(raw,raw,[(a>0).astype(np.float32) for a in raw],250,2210,mask)
        layer=TemporalDepthLayer(d,c,m['source_count'],m['temporal_variance'],m['original_support_count'],m['filled_support_count'],m['selected_original'])
        layer.validate()
        self.assertEqual(layer.fused_confidence[12,12],0)
