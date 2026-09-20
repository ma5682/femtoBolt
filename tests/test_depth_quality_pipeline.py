from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase, mock
import json

import numpy as np
import open3d as o3d
from PIL import Image

from femto.app.femto_control_app import save_rgbd_photo
from femto.geometry.fusion import read_geometry_safe
from femto.hardware.orbbec_rgbd import AlignedFrame
from femto.geometry.pose_geometry import Intrinsics
from femto.pipeline.scene_reconstruction import reconstruct_scene
from femto.pipeline.synthetic_scene_demo import generate_dataset


class DepthQualityPipelineTests(TestCase):
    def test_photo_keeps_raw_cloud_and_saves_separate_repaired_cloud(self):
        depth = np.full((60,80),1000,np.float32)
        depth[30,40] = 0
        frame = AlignedFrame(np.full((60,80,3),100,np.uint8), depth, 1,1,1,1,1,1,1)
        distortion = dict(model='BROWN_CONRADY_K6', **{k:0.0 for k in ('k1','k2','k3','k4','k5','k6','p1','p2')})
        camera = SimpleNamespace(intrinsics=Intrinsics(80,60,75,75,40,30),
                                 min_depth_mm=250,max_depth_mm=2210,metadata={},
                                 calibration=dict(rgb_distortion=distortion,depth_distortion=distortion,
                                                  depth_intrinsic={},depth_to_color={}))
        with TemporaryDirectory() as td:
            output = save_rgbd_photo(frame,camera,Path(td))
            with Image.open(output/'depth_mm.png') as im:
                np.testing.assert_array_equal(np.asarray(im),depth)
            raw = read_geometry_safe(output/'point_cloud_m.ply',mesh=False)
            repaired = read_geometry_safe(output/'derived/point_cloud_hole_filled_m.ply',mesh=False)
            self.assertEqual(len(raw.points),4799)
            self.assertEqual(len(repaired.points),4800)
            np.testing.assert_allclose(np.asarray(raw.points)[:,2],1.0)
            manifest = json.loads((output/'manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(manifest['depth_quality']['repaired_pixels'],1)
            for value in manifest['depth_quality']['files'].values():
                self.assertTrue((output/value).is_file())

    def test_tsdf_repeats_only_frames_with_actual_repairs(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            dataset = generate_dataset(root/'datasets',frames=4,width=128,height=96)
            frames = [json.loads(x) for x in (dataset/'frames.jsonl').read_text().splitlines()]
            for i, frame in enumerate(frames):
                depth = np.full((96,128),1000,np.uint16)
                if i == 0:
                    depth[32,48] = 0
                Image.fromarray(depth).save(dataset/frame['depth'])
            real_volume = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=.02,sdf_trunc=.06,color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
            volume = mock.Mock(wraps=real_volume)
            with mock.patch('femto.pipeline.scene_reconstruction.o3d.pipelines.integration.ScalableTSDFVolume',return_value=volume):
                report = reconstruct_scene(dataset,root/'reconstruction',voxel_m=.02,sdf_trunc_m=.06)
            self.assertEqual(volume.integrate.call_count,16)
            depths = [np.asarray(c.args[0].depth) for c in volume.integrate.call_args_list]
            self.assertTrue(all(np.count_nonzero(d)>100 for d in depths[:10]+depths[13:]))
            self.assertTrue(all(np.count_nonzero(d)==1 for d in depths[10:13]))
            self.assertEqual(report['depth_quality']['frames'][0]['repaired_pixels'],1)
            self.assertEqual(report['depth_quality']['integrations_per_frame'],[10,1,1,1])
            self.assertTrue(report['depth_quality']['filled_depth_used'])
