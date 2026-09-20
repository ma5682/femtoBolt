from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

from femto.pipeline.scene_to_nerfstudio import camera_to_opengl, convert, pose_diversity


class NerfstudioConversionTests(unittest.TestCase):
    def test_camera_axes_change_from_rdf_to_rub(self) -> None:
        converted = camera_to_opengl(np.eye(4))
        np.testing.assert_allclose(converted, np.diag([1.0, -1.0, -1.0, 1.0]))

    def test_pose_diversity_uses_metric_camera_centres(self) -> None:
        first = np.eye(4)
        second = np.eye(4)
        second[0, 3] = 0.12
        translation_span, rotation_span = pose_diversity([first, second])
        self.assertAlmostEqual(translation_span, 0.12)
        self.assertAlmostEqual(rotation_span, 0.0)

    def test_static_format_conversion_is_explicit_and_preserves_uint16_depth(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "scene"
            for child in ("rgb", "depth", "pose", "calibration"):
                (dataset / child).mkdir(parents=True)
            (dataset / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema": "femto-full-scene-rgbd-v1",
                        "frame_count": 1,
                        "camera": {"depth_range_m": [0.5, 3.86]},
                    }
                ),
                encoding="utf-8",
            )
            frame = {
                "frame": 0,
                "camera_timestamp_us": 123,
                "rgb": "rgb/000000.png",
                "depth": "depth/000000.png",
                "pose": "pose/000000.json",
                "pose_valid": True,
                "rgb_depth_timestamp_delta_us": 20,
                "frame_metadata": {"color": {"exposure": 100, "gain": 2}},
            }
            (dataset / "frames.jsonl").write_text(json.dumps(frame) + "\n", encoding="utf-8")
            (dataset / "pose" / "000000.json").write_text(
                json.dumps(
                    {
                        "pose_valid": True,
                        "T_WC": np.eye(4).tolist(),
                        "pose_source": "unit_test",
                    }
                ),
                encoding="utf-8",
            )
            intrinsic = {"width": 8, "height": 6, "fx": 7.0, "fy": 7.0, "cx": 3.5, "cy": 2.5}
            distortion = {
                "rgb": {
                    "model": "OBCameraDistortionModel.BROWN_CONRADY_K6",
                    **{key: 0.0 for key in ("k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2")},
                }
            }
            (dataset / "calibration" / "color_intrinsic.json").write_text(
                json.dumps(intrinsic), encoding="utf-8"
            )
            (dataset / "calibration" / "distortion.json").write_text(
                json.dumps(distortion), encoding="utf-8"
            )
            color = np.full((6, 8, 3), 127, dtype=np.uint8)
            depth = np.full((6, 8), 1000, dtype=np.uint16)
            depth[2, 2] = 0
            depth[2, 2] = 0
            for path, image in (
                (dataset / "rgb" / "000000.png", color),
                (dataset / "depth" / "000000.png", depth),
            ):
                ok, encoded = cv2.imencode(path.suffix, image)
                self.assertTrue(ok)
                path.write_bytes(encoded.tobytes())

            output = root / "converted"
            result = convert(dataset, output, allow_static=True)
            self.assertEqual(result["frame_count"], 1)
            self.assertEqual(result["schema"], "femto-to-nerfstudio-v2")
            self.assertGreater(result["initialization"]["point_count"], 0)
            transforms = json.loads((output / "transforms.json").read_text(encoding="utf-8"))
            self.assertEqual(transforms["ply_file_path"], "rgbd_init_points.ply")
            self.assertTrue((output / transforms["ply_file_path"]).is_file())
            ply_header = (output / transforms["ply_file_path"]).read_bytes().split(b"end_header\n", 1)[0]
            self.assertIn(b"property float x", ply_header)
            self.assertIn(b"property uchar red", ply_header)
            quality = json.loads(
                (output / "quality" / "3dgs_input_report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(quality["color_exposure_metadata"]["count"], 1)
            self.assertFalse(quality["photometric_unstable"])
            np.testing.assert_allclose(
                transforms["frames"][0]["transform_matrix"],
                np.diag([1.0, -1.0, -1.0, 1.0]),
            )
            converted_depth = cv2.imdecode(
                np.fromfile(output / "derived" / "depth" / "000000.png", dtype=np.uint8),
                cv2.IMREAD_UNCHANGED,
            )
            self.assertEqual(converted_depth.dtype, np.uint16)
            self.assertEqual(int(converted_depth[2, 2]), 1000)
            original_depth = cv2.imdecode(
                np.fromfile(output / "depth_original" / "000000.png", dtype=np.uint8),
                cv2.IMREAD_UNCHANGED,
            )
            self.assertEqual(int(original_depth[2, 2]), 0)
            confidence = np.load(output / "derived" / "confidence" / "000000.npy", allow_pickle=False)
            self.assertAlmostEqual(float(confidence[2, 2]), 0.3, places=6)
            source_depth = cv2.imdecode(
                np.fromfile(dataset / "depth" / "000000.png", dtype=np.uint8),
                cv2.IMREAD_UNCHANGED,
            )
            self.assertEqual(int(source_depth[2, 2]), 0)


if __name__ == "__main__":
    unittest.main()
