from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import numpy as np

import femto.gsplat.gsplat_backend as gsplat_backend
from femto.gsplat.scene_gsplat_export import read_trained_gaussian_ply, write_trained_gaussian_ply
from femto.gsplat.scene_gsplat_trainer import PROFILES


def _write_demo(root: Path, nerfstudio: Path) -> None:
    (root / "demo_summary.json").write_text(
        json.dumps({"schema": "femto-complete-demo-v1", "nerfstudio": str(nerfstudio)}, ensure_ascii=False),
        encoding="utf-8",
    )


class GsplatBackendTests(unittest.TestCase):
    def test_prepare_converts_legacy_input_without_overwriting(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = root / 'nerfstudio'
            original.mkdir()
            (original / 'transforms.json').write_text('{}')
            manifest = {'source_dataset': str(root / 'source'),
                        'initialization': {'voxel_m': 0.005, 'max_points': 12345}}
            (original / 'conversion_manifest.json').write_text(json.dumps(manifest))
            with mock.patch('femto.pipeline.scene_to_nerfstudio.convert') as conversion:
                prepared = gsplat_backend.prepare_training_input(original)
            self.assertNotEqual(prepared, original)
            self.assertEqual(prepared.parent, original.parent)
            conversion.assert_called_once_with((root / 'source').resolve(), prepared,
                                               seed_voxel_m=0.005, max_seed_points=12345)
            self.assertEqual((original / 'transforms.json').read_text(), '{}')

    def test_prepare_preserves_existing_strict_input(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'transforms.json').write_text(json.dumps({'dataset_split': {}}))
            with mock.patch('femto.pipeline.scene_to_nerfstudio.convert') as conversion:
                self.assertEqual(gsplat_backend.prepare_training_input(root), root.resolve())
            conversion.assert_not_called()

    def test_backend_exposes_the_three_product_profiles(self) -> None:
        self.assertEqual(gsplat_backend.PROFILE_KEYS, ("low_vram", "balanced", "quality"))
        self.assertIs(gsplat_backend.PROFILES, PROFILES)
        self.assertEqual(gsplat_backend.BACKEND_NAME, "direct_gsplat")

    def test_resolve_training_input_accepts_demo_root_and_nerfstudio_dir(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            nerfstudio = root / "nerfstudio"
            nerfstudio.mkdir()
            (nerfstudio / "transforms.json").write_text("{}", encoding="utf-8")
            _write_demo(root, nerfstudio)
            self.assertEqual(gsplat_backend.resolve_training_input(root), nerfstudio.resolve())
            self.assertEqual(gsplat_backend.resolve_training_input(nerfstudio), nerfstudio.resolve())

    def test_resolve_training_input_rejects_a_directory_without_transforms(self) -> None:
        with TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                gsplat_backend.resolve_training_input(Path(temporary))

    def test_assess_reports_a_reason_instead_of_raising(self) -> None:
        with TemporaryDirectory() as temporary:
            assessment = gsplat_backend.assess_training_input(Path(temporary))
        self.assertFalse(assessment["ready"])
        self.assertTrue(assessment["reasons"])
        self.assertEqual(assessment["backend"], "direct_gsplat")

    def test_skipped_local_conversion_does_not_use_a_stale_nested_input(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "demo_summary.json").write_text(
                json.dumps({"nerfstudio": None, "frame_count": 5}), encoding="utf-8"
            )
            stale_input = root / "nerfstudio"
            stale_input.mkdir()
            (stale_input / "transforms.json").write_text("{}", encoding="utf-8")
            with mock.patch.object(gsplat_backend, "assess_direct_input") as direct:
                assessment = gsplat_backend.assess_training_input(root)
            self.assertFalse(assessment["ready"])
            self.assertIsNone(assessment["input"])
            self.assertIn("not generated", assessment["reasons"][0])
            direct.assert_not_called()
            with self.assertRaisesRegex(ValueError, "not generated"):
                gsplat_backend.prepare_training_input(root)
            output = root / "must_not_be_created"
            with mock.patch.object(gsplat_backend.subprocess, "Popen") as launch:
                with self.assertRaisesRegex(ValueError, "not generated"):
                    gsplat_backend.train(root, output_dir=output)
            launch.assert_not_called()
            self.assertFalse(output.exists())

    def test_invalid_summary_inputs_return_a_blocked_assessment(self) -> None:
        cases = [[], None, {"nerfstudio": ""}, {"nerfstudio": "  "},
                 {"nerfstudio": False}, {"nerfstudio": 12},
                 {"nerfstudio": []}, {"nerfstudio": {}}]
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            for summary in cases:
                with self.subTest(summary=summary):
                    (root / "demo_summary.json").write_text(json.dumps(summary), encoding="utf-8")
                    with mock.patch.object(gsplat_backend, "assess_direct_input") as direct:
                        assessment = gsplat_backend.assess_training_input(root)
                    self.assertFalse(assessment["ready"])
                    self.assertIsNone(assessment["input"])
                    self.assertTrue(assessment["reasons"])
                    direct.assert_not_called()

    def test_relative_summary_input_is_bound_to_the_reconstruction(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            nested = root / "nerfstudio"
            nested.mkdir()
            (nested / "transforms.json").write_text("{}", encoding="utf-8")
            (root / "demo_summary.json").write_text(
                json.dumps({"nerfstudio": "nerfstudio"}), encoding="utf-8"
            )
            self.assertEqual(gsplat_backend.resolve_training_input(root), nested.resolve())

    def test_legacy_summary_without_input_field_keeps_nested_resolution(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "demo_summary.json").write_text("{}", encoding="utf-8")
            self.assertEqual(gsplat_backend.resolve_training_input(root), (root / "nerfstudio").resolve())

    def test_training_command_targets_the_runtime_trainer_with_an_explicit_output(self) -> None:
        command = gsplat_backend.build_training_command(
            Path("C:/FemtoGsplatRuntime/Scripts/python.exe"),
            Path("input"),
            Path("output"),
            profile="balanced",
            iterations=800,
            export_spz=False,
        )
        self.assertEqual(command[1:4], ["-m", "femto.gsplat.scene_gsplat_trainer", "train"])
        self.assertIn("--profile", command)
        self.assertEqual(command[command.index("--profile") + 1], "balanced")
        self.assertEqual(command[command.index("--output") + 1], str(Path("output")))
        self.assertEqual(command[command.index("--iterations") + 1], "800")
        self.assertIn("--no-spz", command)

    def test_unknown_profile_is_rejected_before_launching_a_process(self) -> None:
        with self.assertRaises(ValueError):
            gsplat_backend.build_training_command(
                Path("python"), Path("in"), Path("out"), profile="ultra"
            )

    def test_runtime_environment_mirrors_the_documented_cuda_launcher(self) -> None:
        with mock.patch.object(gsplat_backend, "_visual_studio_environment", return_value={}):
            environment = gsplat_backend.runtime_environment()
        self.assertEqual(environment["TORCH_CUDA_ARCH_LIST"], gsplat_backend.TORCH_CUDA_ARCH_LIST)
        self.assertEqual(environment["PYTHONUTF8"], "1")
        self.assertEqual(environment["VSLANG"], "1033")
        # The ASCII runtime junction must precede everything else: CUDA JIT fails
        # under the repository's non-ASCII path.
        self.assertTrue(environment["PATH"].startswith(str(gsplat_backend.ASCII_RUNTIME_ROOT / "Scripts")))
        self.assertIn(gsplat_backend.CUDA_TOOLKIT_VERSION, environment["CUDA_HOME"])

    def test_latest_trained_model_requires_a_real_report_and_ply(self) -> None:
        with TemporaryDirectory() as temporary:
            run = Path(temporary) / "direct_gsplat_test"
            (run / "model").mkdir(parents=True)
            with self.assertRaises(ValueError):
                gsplat_backend.latest_trained_model(run)
            ply = run / "model" / "trained_gaussians.ply"
            (run / "training_report.json").write_text(
                json.dumps({"accepted": True, "model_ply": str(ply), "final_gaussians": 7}),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                gsplat_backend.latest_trained_model(run)
            write_trained_gaussian_ply(
                ply,
                means=np.zeros((1, 3), np.float32),
                sh0=np.zeros((1, 1, 3), np.float32),
                sh_rest=np.zeros((1, 15, 3), np.float32),
                opacity_logits=np.zeros(1, np.float32),
                log_scales=np.zeros((1, 3), np.float32),
                quaternions_wxyz=np.array([[1.0, 0.0, 0.0, 0.0]], np.float32),
            )
            model = gsplat_backend.latest_trained_model(run)
            self.assertTrue(model.accepted)
            self.assertIsNone(model.model_spz)


class TrainedGaussianRoundTripTests(unittest.TestCase):
    def test_trained_ply_round_trips_every_stored_parameter(self) -> None:
        rng = np.random.default_rng(20260904)
        count = 5
        means = rng.normal(size=(count, 3)).astype(np.float32)
        sh0 = rng.normal(size=(count, 1, 3)).astype(np.float32)
        sh_rest = rng.normal(size=(count, 15, 3)).astype(np.float32)
        opacity = rng.normal(size=count).astype(np.float32)
        log_scales = rng.normal(size=(count, 3)).astype(np.float32)
        quaternions = rng.normal(size=(count, 4)).astype(np.float32)
        normalized = quaternions / np.linalg.norm(quaternions, axis=1, keepdims=True)
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "trained.ply"
            write_trained_gaussian_ply(
                path, means=means, sh0=sh0, sh_rest=sh_rest, opacity_logits=opacity,
                log_scales=log_scales, quaternions_wxyz=quaternions,
            )
            model = read_trained_gaussian_ply(path)
        np.testing.assert_allclose(model["means"], means, rtol=1e-6)
        np.testing.assert_allclose(model["sh0"], sh0, rtol=1e-6)
        # The writer flattens (N, K, 3) channel-major; the reader must undo exactly that.
        np.testing.assert_allclose(model["sh_rest"], sh_rest, rtol=1e-6)
        np.testing.assert_allclose(model["opacity_logits"], opacity, rtol=1e-6)
        np.testing.assert_allclose(model["log_scales"], log_scales, rtol=1e-6)
        np.testing.assert_allclose(model["quaternions_wxyz"], normalized, rtol=1e-5)

    def test_reader_rejects_a_plain_xyzrgb_point_cloud(self) -> None:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "seed.ply"
            path.write_text(
                "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\n"
                "property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n"
                "end_header\n0 0 1 255 128 0\n",
                encoding="ascii",
            )
            with self.assertRaises(ValueError):
                read_trained_gaussian_ply(path)


if __name__ == "__main__":
    unittest.main()
