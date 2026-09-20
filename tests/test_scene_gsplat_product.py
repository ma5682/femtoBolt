from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from commercial_compliance_check import (
    EXCLUDED_SCAN_DIRS,
    EXPECTED_SPZ_EXTENSION_SHA256,
    REQUIRED_LICENSES,
    SPZ_STATIC_COMPONENTS,
)
from femto.gsplat.scene_gsplat_data import CV_CAMERA_TO_OPENGL_CAMERA, load_gsplat_dataset, opengl_c2w_to_rdf, read_xyzrgb_ply
from femto.gsplat.scene_gsplat_export import export_run_spz, write_trained_gaussian_ply
from femto.gsplat.scene_gsplat_metrics import (
    LPIPS_BACKBONE_FILE,
    depth_metrics_numpy,
    install_lpips_backbone,
    lpips_backbone_status,
    psnr_numpy,
    torch_hub_checkpoint_dir,
)


class DirectGsplatProductTests(unittest.TestCase):
    def test_camera_conversion_restores_rdf_projection_and_split_is_disjoint(self) -> None:
        c2w_rdf = np.eye(4)
        c2w_rdf[:3, 3] = [0.1, -0.2, 0.3]
        c2w_gl = c2w_rdf @ CV_CAMERA_TO_OPENGL_CAMERA
        np.testing.assert_allclose(opengl_c2w_to_rdf(c2w_gl), c2w_rdf)

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "images").mkdir()
            (root / "depth").mkdir()
            frames = []
            for index in range(4):
                Image.fromarray(np.full((6, 8, 3), 80 + index, np.uint8)).save(root / "images" / f"{index}.png")
                Image.fromarray(np.full((6, 8), 1000, np.uint16)).save(root / "depth" / f"{index}.png")
                frames.append(
                    {
                        "file_path": f"images/{index}.png",
                        "depth_file_path": f"depth/{index}.png",
                        "transform_matrix": c2w_gl.tolist(),
                    }
                )
            (root / "seed.ply").write_text(
                "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\n"
                "property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n"
                "end_header\n0 0 1 255 128 0\n",
                encoding="ascii",
            )
            (root / "transforms.json").write_text(
                json.dumps(
                    {
                        "camera_model": "OPENCV", "fl_x": 8, "fl_y": 8, "cx": 4, "cy": 3,
                        "w": 8, "h": 6, "depth_unit_scale_factor": 0.001,
                        "ply_file_path": "seed.ply", "frames": frames,
                    }
                ),
                encoding="utf-8",
            )
            dataset = load_gsplat_dataset(root, downscale=2, eval_stride=2)
            self.assertEqual(dataset.frames[0].width, 4)
            self.assertEqual(dataset.frames[0].height, 3)
            self.assertFalse(set(dataset.train_indices) & set(dataset.eval_indices))
            self.assertEqual(set(dataset.train_indices) | set(dataset.eval_indices), set(range(4)))
            points, colors = read_xyzrgb_ply(dataset.seed_path)
            np.testing.assert_allclose(points, [[0, 0, 1]])
            np.testing.assert_allclose(colors, [[1.0, 128 / 255, 0.0]])

    def test_trained_ply_contains_full_sh_layout_and_normalized_quaternion(self) -> None:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "trained.ply"
            report = write_trained_gaussian_ply(
                path,
                means=np.array([[0.0, 0.0, 1.0]], np.float32),
                sh0=np.zeros((1, 1, 3), np.float32),
                sh_rest=np.zeros((1, 15, 3), np.float32),
                opacity_logits=np.array([0.0], np.float32),
                log_scales=np.zeros((1, 3), np.float32),
                quaternions_wxyz=np.array([[2.0, 0.0, 0.0, 0.0]], np.float32),
            )
            raw = path.read_bytes()
            header, body = raw.split(b"end_header\n", 1)
            self.assertIn(b"property float f_rest_44", header)
            self.assertIn(b"property float rot_3", header)
            self.assertEqual(report["property_count"], 62)
            self.assertEqual(len(body), report["property_count"] * 4)
            values = np.frombuffer(body, dtype="<f4")
            np.testing.assert_allclose(values[-4:], [1.0, 0.0, 0.0, 0.0])

    def test_acceptance_metrics_use_metric_depth(self) -> None:
        self.assertTrue(np.isinf(psnr_numpy(np.zeros((2, 2, 3)), np.zeros((2, 2, 3)))))
        metrics = depth_metrics_numpy(np.array([[1.1, 0.0]]), np.array([[1.0, 2.0]]))
        self.assertEqual(metrics["valid_pixels"], 2)
        self.assertEqual(metrics["predicted_pixels"], 1)
        self.assertAlmostEqual(float(metrics["coverage_ratio"]), 0.5)
        self.assertAlmostEqual(float(metrics["mae_m"]), 1.05)

    def test_unrendered_depth_holes_count_as_full_misses(self) -> None:
        # Pixel 0: 1.1 vs 1.0 (0.1 m error). Pixel 1: hole (0) vs 2.0 -> 2.0 m miss, not excluded.
        metrics = depth_metrics_numpy(np.array([[1.1, 0.0]]), np.array([[1.0, 2.0]]))
        expected_rmse = float(np.sqrt((0.1**2 + 2.0**2) / 2.0))
        self.assertAlmostEqual(float(metrics["rmse_m"]), expected_rmse)
        self.assertAlmostEqual(float(metrics["abs_rel"]), (0.1 / 1.0 + 2.0 / 2.0) / 2.0)
        # Pixels without target depth are neither scored nor counted, wherever the render put geometry.
        ignored = depth_metrics_numpy(np.array([[1.1, 3.0]]), np.array([[1.0, 0.0]]))
        self.assertEqual(ignored["valid_pixels"], 1)
        self.assertAlmostEqual(float(ignored["coverage_ratio"]), 1.0)
        self.assertAlmostEqual(float(ignored["mae_m"]), 0.1)
        # A perfect render over all valid pixels reaches full coverage and zero error.
        perfect = depth_metrics_numpy(np.array([[1.0, 2.0]]), np.array([[1.0, 2.0]]))
        self.assertAlmostEqual(float(perfect["coverage_ratio"]), 1.0)
        self.assertAlmostEqual(float(perfect["rmse_m"]), 0.0)

    def test_lpips_backbone_install_requires_official_hash(self) -> None:
        with TemporaryDirectory() as temporary, mock.patch.dict(os.environ, {"TORCH_HOME": temporary}):
            self.assertEqual(torch_hub_checkpoint_dir(), Path(temporary) / "hub" / "checkpoints")
            missing = lpips_backbone_status()
            self.assertFalse(missing["present"])
            self.assertFalse(missing["ready"])
            self.assertIn("install_lpips_weights.cmd", missing["error"])

            payload = b"fake alexnet checkpoint bytes"
            good_hash = hashlib.sha256(payload).hexdigest()
            source = Path(temporary) / "download.pth"
            source.write_bytes(payload)

            with self.assertRaises(ValueError):
                install_lpips_backbone(source, expected_sha256="0" * 64)
            self.assertFalse((torch_hub_checkpoint_dir() / LPIPS_BACKBONE_FILE).exists())

            installed = install_lpips_backbone(source, expected_sha256=good_hash)
            self.assertEqual(installed["status"], "installed")
            target = torch_hub_checkpoint_dir() / LPIPS_BACKBONE_FILE
            self.assertTrue(target.is_file())
            self.assertEqual(install_lpips_backbone(source, expected_sha256=good_hash)["status"], "already_installed")
            ready = lpips_backbone_status(expected_sha256=good_hash)
            self.assertTrue(ready["ready"])
            self.assertTrue(ready["sha256_verified"])

            target.write_bytes(b"tampered")
            tampered = lpips_backbone_status(expected_sha256=good_hash)
            self.assertTrue(tampered["present"])
            self.assertFalse(tampered["ready"])
            self.assertIn("differs", tampered["error"])

    def test_existing_run_spz_export_updates_both_report_compatibility_fields(self) -> None:
        with TemporaryDirectory() as temporary:
            run = Path(temporary) / "direct_gsplat_test"
            model = run / "model"
            model.mkdir(parents=True)
            (model / "trained_gaussians.ply").write_bytes(b"test-ply")
            (run / "training_report.json").write_text(
                json.dumps({"lineage": {"outputs": {}}}), encoding="utf-8"
            )
            (model / "model_manifest.json").write_text(
                json.dumps({"outputs": {}}), encoding="utf-8"
            )
            spz_path = model / "trained_gaussians.spz"
            spz = {
                "status": "exported",
                "path": str(spz_path),
                "ply_gaussian_count": 12,
                "spz_gaussian_count": 12,
                "roundtrip_count_matches": True,
            }
            with mock.patch("femto.gsplat.scene_gsplat_export.export_official_spz", return_value=spz):
                export_run_spz(run)

            report = json.loads((run / "training_report.json").read_text(encoding="utf-8"))
            manifest = json.loads((model / "model_manifest.json").read_text(encoding="utf-8"))
            for payload in (report, report["lineage"], manifest):
                self.assertEqual(payload["model_spz"], str(spz_path))
                self.assertIsNone(payload["spz_error"])
            self.assertTrue(report["spz_conversion"]["roundtrip_count_matches"])
            self.assertTrue(report["lineage"]["outputs"]["spz"]["roundtrip_count_matches"])
            self.assertTrue(manifest["outputs"]["spz"]["roundtrip_count_matches"])

    def test_compliance_scan_skips_gsplat_runtime_and_requires_its_licenses(self) -> None:
        self.assertIn(".venv-gsplat", EXCLUDED_SCAN_DIRS)
        for relative in (
            "torch/LICENSE",
            "torch/NOTICE",
            "torchvision/LICENSE",
            "torchmetrics/LICENSE",
            "lpips/LICENSE",
            "spz/LICENSE",
            "spz/ADOBE_NOTICE.txt",
            "zlib/LICENSE.txt",
            "zstd/LICENSE",
            "zstd/LEGACY_BSD2_NOTICES.txt",
            "divsufsort/LICENSE",
            "nanobind/LICENSE",
            "robin-map/LICENSE",
        ):
            self.assertIn(relative, REQUIRED_LICENSES)
        self.assertEqual(SPZ_STATIC_COMPONENTS["nanobind"]["version"], "3.0.1")
        self.assertEqual(SPZ_STATIC_COMPONENTS["robin-map"]["version"], "1.4.0")
        self.assertEqual(SPZ_STATIC_COMPONENTS["spz-source"]["tag"], "v3.0.0")
        self.assertEqual(SPZ_STATIC_COMPONENTS["zlib"]["version"], "1.3.1")
        self.assertEqual(SPZ_STATIC_COMPONENTS["zstd"]["version"], "1.5.6")
        self.assertEqual(SPZ_STATIC_COMPONENTS["divsufsort-lite"]["covered_by_component"], "zstd")
        self.assertEqual(SPZ_STATIC_COMPONENTS["zstd-legacy-code"]["covered_by_component"], "zstd")
        self.assertEqual(len(EXPECTED_SPZ_EXTENSION_SHA256), 64)


if __name__ == "__main__":
    unittest.main()
