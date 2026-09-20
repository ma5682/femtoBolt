from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib
import json
import unittest

import numpy as np

from femto.geometry.single_frame_quality import clean_depth
from femto.pipeline.scene_reconstruction import reconstruct_scene
from femto.geometry.rgbd_frames import prepare_rgbd
from femto.io_utils import imwrite_unicode
from femto.geometry.fusion import read_geometry_safe
from femto.pipeline.synthetic_scene_demo import generate_dataset
from femto.temporal.scene_keyframe import build_fixed_keyframe_dataset


class SingleFrameQualityTests(unittest.TestCase):
    def test_cleanup_retains_edges_endpoints_and_never_averages(self):
        depth = np.zeros((12, 16), dtype=np.float64)
        depth[1:6, 1:6] = 250
        depth[1:6, 6:11] = 2210
        depth[8, 1:8] = 1000
        depth[10, 12] = 800
        depth[10, 13] = np.nan
        depth[10, 14] = 2211
        original = depth.copy()
        cleaned, report = clean_depth(depth)
        np.testing.assert_array_equal(depth, original)
        np.testing.assert_array_equal(cleaned[1:6, 1:11], depth[1:6, 1:11])
        self.assertTrue(np.all(cleaned[8, 2:7] == 1000))
        self.assertEqual(cleaned[10, 12], 0)
        self.assertEqual(cleaned[10, 14], 0)
        self.assertGreater(report['isolated_pixels_removed'], 0)

    def test_open3d_includes_2210mm_endpoint(self):
        y, x = np.indices((4, 5), dtype=np.float32)
        depth = np.full((4, 5), 2210, np.uint16)
        rgbd, *_ = prepare_rgbd(np.zeros((4, 5, 3), np.uint8), depth, x, y,
                                  np.array([[10., 0, 2], [0, 10., 2], [0, 0, 1]]), 1, .25, 2.21)
        np.testing.assert_allclose(np.asarray(rgbd.depth), 2.21)

    def test_single_frame_exports_evidence_and_quality_without_source_changes(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = generate_dataset(root / 'data', frames=4, width=160, height=96)
            dataset, _ = build_fixed_keyframe_dataset(source, source / 'derived' / 'one')
            depth_path = dataset / 'depth' / '000000.png'
            depth = np.full((96, 160), 1000, np.uint16)
            depth[45, 70] = 1900
            imwrite_unicode(depth_path, depth)
            before = hashlib.sha256(depth_path.read_bytes()).hexdigest()
            output = root / 'result'
            report = reconstruct_scene(dataset, output, image_scale=1, point_stride=1,
                                       depth_min_m=.25, depth_max_m=2.21, single_frame_quality=True)
            self.assertEqual(before, hashlib.sha256(depth_path.read_bytes()).hexdigest())
            self.assertEqual(report['frame_count'], 1)
            self.assertEqual(report['metrics']['raw_point_count'], 96 * 160)
            self.assertEqual(report['metrics']['cleaned_point_count'], 96 * 160 - 1)
            for ext in ('ply', 'obj', 'glb'):
                self.assertTrue((output / ('scene_surface_mesh_m.' + ext)).is_file())
            cloud = read_geometry_safe(output / 'scene_fused_points_m.ply', mesh=False)
            self.assertTrue(np.allclose(np.asarray(cloud.points)[:, 2], 1))
            quality = json.loads((output / 'single_frame_quality.json').read_text())
            self.assertIsNone(quality['accepted'])
            self.assertLess(quality['surface_consistency']['cleaned_to_mesh']['p95_mm'], 1)
            self.assertGreater(quality['surface_consistency']['raw_to_mesh']['max_mm'], 800)
            self.assertEqual(len(json.loads((output / 'source_evidence.json').read_text())['files']),
                             5 + len(list((dataset / 'calibration').glob('*.json'))))
            with self.assertRaises(FileExistsError):
                reconstruct_scene(dataset, output)
            with self.assertRaisesRegex(ValueError, 'exactly one frame'):
                reconstruct_scene(source, root / 'rejected', single_frame_quality=True,
                                  depth_min_m=.25, depth_max_m=2.21)


if __name__ == '__main__':
    unittest.main()
