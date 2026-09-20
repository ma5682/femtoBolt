from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np
import open3d as o3d

from femto.geometry.fusion import read_geometry_safe
from femto.io_utils import imwrite_unicode
from femto.dataset.scene_dataset_validator import validate_pose
from femto.geometry.scene_gaussian import export_gaussian_initialization
from femto.temporal.scene_keyframe import build_fixed_keyframe_dataset
from femto.pipeline.scene_reconstruction import reconstruct_scene
from femto.temporal.scene_static_fusion import build_static_median_dataset
from femto.pipeline.scene_to_nerfstudio import convert
from femto.pipeline.synthetic_scene_demo import _look_at_rdf, generate_dataset


class CompleteDemoTests(unittest.TestCase):
    def test_look_at_pose_is_rigid_rdf_camera_to_world(self) -> None:
        pose = _look_at_rdf(np.array([0.2, -0.3, 0.1]), np.array([0.0, 0.1, 1.5]))
        self.assertIsNone(validate_pose(pose))
        forward = pose[:3, 2]
        expected = np.array([-0.2, 0.4, 1.4])
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(forward, expected)

    def test_gaussian_initialization_has_standard_scalar_properties(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(
                np.array([[0.0, 0.0, 1.0], [0.01, 0.0, 1.0], [0.0, 0.01, 1.0]])
            )
            cloud.colors = o3d.utility.Vector3dVector(
                np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
            )
            result = export_gaussian_initialization(cloud, root, voxel_m=0.001)
            data = Path(result["model"]).read_bytes()
            header = data.split(b"end_header\n", 1)[0].decode("ascii")
            for name in ("f_dc_0", "opacity", "scale_2", "rot_3"):
                self.assertIn(f"property float {name}", header)
            manifest = json.loads((root / "gaussian_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "initialization_only_untrained")
            self.assertFalse(result["trained"])

    def test_known_pose_rgbd_reconstructs_and_converts(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = generate_dataset(root / "datasets", frames=5, width=192, height=108)
            reconstruction = root / "reconstruction"
            report = reconstruct_scene(
                dataset,
                reconstruction,
                image_scale=1.0,
                voxel_m=0.02,
                sdf_trunc_m=0.06,
                point_stride=2,
            )
            self.assertGreater(report["metrics"]["fused_point_count"], 1000)
            self.assertGreater(report["metrics"]["mesh_triangles"], 1000)
            self.assertTrue((reconstruction / "scene_tsdf_mesh_m.glb").is_file())
            self.assertTrue((reconstruction / "gaussian" / "gaussian_init_untrained.ply").is_file())

            cloud = read_geometry_safe(reconstruction / "scene_fused_points_m.ply", mesh=False)
            truth = read_geometry_safe(
                dataset / "ground_truth" / "synthetic_scene_ground_truth_m.ply", mesh=True
            )
            ray_scene = o3d.t.geometry.RaycastingScene()
            ray_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(truth))
            distances = ray_scene.compute_distance(
                o3d.core.Tensor(np.asarray(cloud.points), dtype=o3d.core.Dtype.Float32)
            ).numpy()
            self.assertLess(float(np.quantile(distances, 0.95)), 0.015)

            converted = root / "nerfstudio"
            conversion = convert(dataset, converted)
            self.assertEqual(conversion["frame_count"], 5)
            self.assertTrue((converted / "transforms.json").is_file())

    def test_fixed_camera_temporal_median_is_a_separate_derived_dataset(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = generate_dataset(root / "datasets", frames=5, width=192, height=108)
            frames = [
                json.loads(line)
                for line in (source / "frames.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            first_encoded = np.fromfile(source / frames[0]["depth"], dtype=np.uint8)
            first_depth = cv2.imdecode(first_encoded, cv2.IMREAD_UNCHANGED)
            self.assertIsNotNone(first_depth)
            # Make every observation share the same fixed-camera depth. One transient
            # foreground patch must not change the five-frame temporal median.
            for ordinal, frame in enumerate(frames):
                depth = first_depth.copy()
                if ordinal == 4:
                    depth[depth > 0] = 550
                imwrite_unicode(source / frame["depth"], depth)

            derived, report = build_static_median_dataset(
                source,
                source / "derived" / "fixed_median_test",
                minimum_support_ratio=0.60,
            )
            derived_depth = cv2.imdecode(
                np.fromfile(derived / "depth" / "000000.png", dtype=np.uint8),
                cv2.IMREAD_UNCHANGED,
            )
            np.testing.assert_array_equal(derived_depth, first_depth)
            manifest = json.loads((derived / "manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["parameters"]["fixed_camera"])
            self.assertEqual(manifest["pose"]["source"], "fixed_camera_identity_after_temporal_median")
            self.assertEqual(report["source_frame_count"], 5)
            self.assertTrue((source / "derived" / "latest_fixed_dataset.txt").is_file())

            keyframe, keyframe_report = build_fixed_keyframe_dataset(
                source,
                source / "derived" / "fixed_keyframe_test",
            )
            selected = keyframe_report["selected_source_frame"]
            self.assertNotEqual(selected, 4)
            selected_depth = cv2.imdecode(
                np.fromfile(keyframe / "depth" / "000000.png", dtype=np.uint8),
                cv2.IMREAD_UNCHANGED,
            )
            np.testing.assert_array_equal(selected_depth, first_depth)
            self.assertEqual(
                json.loads((keyframe / "manifest.json").read_text(encoding="utf-8"))["parameters"]["processing_mode"],
                "keyframe",
            )


if __name__ == "__main__":
    unittest.main()
