from pathlib import Path
from tempfile import TemporaryDirectory
from contextlib import nullcontext
import json
import sys
import types
import unittest
from unittest import mock

import numpy as np

import scene_gsplat_viewer as viewer


class RenderBudgetTests(unittest.TestCase):
    def test_budget_keeps_full_model_when_memory_is_sufficient(self):
        plan = viewer.plan_render_budget(
            500_000,
            320,
            180,
            free_cuda_bytes=8 * 1024**3,
            total_cuda_bytes=8 * 1024**3,
        )
        self.assertEqual(plan["state"], "full")
        self.assertEqual(plan["effective_gaussian_count"], 500_000)
        self.assertEqual(plan["effective_resolution"], [320, 180])
        self.assertFalse(plan["temporary_sampling"])

    def test_budget_marks_temporary_sampling_on_low_memory(self):
        plan = viewer.plan_render_budget(
            1_000_000,
            960,
            540,
            free_cuda_bytes=128 * 1024**2,
            total_cuda_bytes=256 * 1024**2,
        )
        self.assertIn(plan["state"], {"sampled_gaussians", "reduced_resolution_and_sampled"})
        self.assertLess(plan["effective_gaussian_count"], plan["model_gaussian_count"])
        self.assertTrue(plan["temporary_sampling"])
        self.assertIn("GPU memory budget", plan["reason"])

    def test_budget_reduces_excessive_pixel_count(self):
        plan = viewer.plan_render_budget(
            10_000,
            4000,
            3000,
            free_cuda_bytes=8 * 1024**3,
        )
        self.assertEqual(plan["effective_gaussian_count"], 10_000)
        self.assertLess(plan["effective_resolution"][0], 4000)
        self.assertLessEqual(plan["effective_resolution"][0] * plan["effective_resolution"][1], 1_600_000)
        self.assertIn("pixel budget", plan["reason"])

    def test_budget_is_deterministic_and_reports_requested_values(self):
        first = viewer.plan_render_budget(1000, 640, 360, free_cuda_bytes=512 * 1024**2)
        second = viewer.plan_render_budget(1000, 640, 360, free_cuda_bytes=512 * 1024**2)
        self.assertEqual(first, second)
        self.assertEqual(first["requested_resolution"], [640, 360])
        self.assertEqual(first["model_gaussian_count"], 1000)

    def test_timeout_records_current_frame_before_degrading_next_frame(self):
        class FakeCamera:
            yaw_deg = 0.0

            @staticmethod
            def viewmat():
                return np.eye(4, dtype=np.float32)

        class FakeRenderer:
            def __init__(self, *_args, **_kwargs):
                self.model_gaussian_count = 10_000
                self.sh_degree = 0
                self._rendered_count = 10_000
                self._budget = {
                    "state": "full",
                    "effective_resolution": [320, 180],
                }
                # Attribute access below intentionally raises AttributeError,
                # exercising the existing no-CUDA reporting path.
                self._torch = object()
                self.device = None

            @property
            def budget(self):
                return {
                    **self._budget,
                    "effective_resolution": list(self._budget["effective_resolution"]),
                }

            @property
            def rendered_gaussian_count(self):
                return self._rendered_count

            @staticmethod
            def bounds():
                return np.zeros(3, dtype=np.float64), 1.0

            @staticmethod
            def render(_viewmat, _intrinsic, _width, _height):
                return np.zeros((180, 320, 3), dtype=np.uint8)

            def degrade_for_timeout(self, _width, _height):
                self._rendered_count = 5_000
                self._budget = {
                    "state": "reduced_resolution_and_sampled",
                    "effective_resolution": [224, 126],
                }

        with TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "trained_gaussians.ply"
            model.write_bytes(b"synthetic test model")
            output = root / "orbit"
            with (
                mock.patch.object(viewer, "resolve_model", return_value=(model, {}, {})),
                mock.patch.object(viewer, "training_optics", return_value=(None, None)),
                mock.patch.object(viewer, "TrainedGaussianRenderer", FakeRenderer),
                mock.patch.object(viewer, "initial_camera", return_value=FakeCamera()),
                mock.patch.object(viewer.time, "time", side_effect=[100.0, 101.0, 103.0, 104.0]),
            ):
                report = viewer.render_orbit(
                    model,
                    output,
                    frames=1,
                    width=320,
                    frame_timeout_s=1.0,
                    data_kind="synthetic",
                )

        current_frame = report["frame_status"][0]
        self.assertTrue(current_frame["timed_out"])
        self.assertEqual(current_frame["effective_resolution"], [320, 180])
        self.assertEqual(current_frame["rendered_gaussian_count"], 10_000)
        self.assertEqual(current_frame["degradation_state"], "full")
        self.assertEqual(report["rendered_gaussian_count"], 5_000)
        self.assertEqual(report["degradation"]["state"], "reduced_resolution_and_sampled")
        self.assertIn("peak_cuda_memory_mb", report)
        self.assertIn("peak_cuda_memory_mib", report)
        self.assertIsNone(report["peak_cuda_memory_mib"])

    def test_forced_sampling_recomputes_dynamic_budget_metadata(self):
        renderer = object.__new__(viewer.TrainedGaussianRenderer)
        renderer.max_render_pixels = 1_500_000
        renderer.model_gaussian_count = 1_000_000
        renderer.device = types.SimpleNamespace(type="cpu")
        renderer._forced_width = None
        renderer._forced_count = 500_000
        renderer._active_indices = range(500_000)
        renderer._budget = {}
        renderer._replan(320, 180, reason="test forced sample")
        self.assertEqual(renderer.budget["state"], "sampled_gaussians")
        self.assertTrue(renderer.budget["temporary_sampling"])
        expected = 500_000 * viewer.RENDER_MEMORY_PER_GAUSSIAN + 320 * 180 * viewer.RENDER_MEMORY_PER_PIXEL
        self.assertEqual(renderer.budget["estimated_memory_bytes"], expected)

    def test_render_keeps_degrading_after_more_than_two_ooms(self):
        class FakeTensor:
            def unsqueeze(self, _axis):
                return self

        class FakeCuda:
            @staticmethod
            def empty_cache():
                return None

        class FakeTorch:
            cuda = FakeCuda()
            float32 = np.float32

            @staticmethod
            def no_grad():
                return nullcontext()

            @staticmethod
            def as_tensor(*_args, **_kwargs):
                return FakeTensor()

        attempts = {"count": 0}

        def always_oom(**_kwargs):
            attempts["count"] += 1
            raise RuntimeError("CUDA out of memory in synthetic test")

        fake_gsplat = types.ModuleType("gsplat")
        fake_rendering = types.ModuleType("gsplat.rendering")
        fake_rendering.rasterization = always_oom
        renderer = object.__new__(viewer.TrainedGaussianRenderer)
        renderer._torch = FakeTorch()
        renderer.device = "cuda:0"
        renderer.means = renderer.quats = renderer.scales = renderer.opacities = renderer.colors = None
        renderer.sh_degree = 0
        renderer._budget = {"effective_resolution": [960, 540]}
        renderer.effective_resolution = mock.Mock(return_value=(960, 540))
        renderer.degrade_for_oom = mock.Mock(side_effect=[True, True, True, False])
        with mock.patch.dict(sys.modules, {"gsplat": fake_gsplat, "gsplat.rendering": fake_rendering}):
            with self.assertRaisesRegex(RuntimeError, "out of memory"):
                renderer.render(np.eye(4), np.eye(3), 960, 540)
        self.assertEqual(attempts["count"], 4)
        self.assertEqual(renderer.degrade_for_oom.call_count, 4)

    def test_render_does_not_degrade_for_non_oom_cuda_runtime_error(self):
        class FakeTensor:
            def unsqueeze(self, _axis):
                return self

        class FakeCuda:
            @staticmethod
            def empty_cache():
                return None

        class FakeTorch:
            cuda = FakeCuda()
            float32 = np.float32

            @staticmethod
            def no_grad():
                return nullcontext()

            @staticmethod
            def as_tensor(*_args, **_kwargs):
                return FakeTensor()

        attempts = {"count": 0}

        def fail_with_non_oom_cuda_error(**_kwargs):
            attempts["count"] += 1
            raise RuntimeError("CUDA error: invalid configuration argument")

        fake_gsplat = types.ModuleType("gsplat")
        fake_rendering = types.ModuleType("gsplat.rendering")
        fake_rendering.rasterization = fail_with_non_oom_cuda_error
        renderer = object.__new__(viewer.TrainedGaussianRenderer)
        renderer._torch = FakeTorch()
        renderer.device = "cuda:0"
        renderer.means = renderer.quats = renderer.scales = renderer.opacities = renderer.colors = None
        renderer.sh_degree = 0
        renderer._budget = {"effective_resolution": [960, 540]}
        renderer.effective_resolution = mock.Mock(return_value=(960, 540))
        renderer.degrade_for_oom = mock.Mock(return_value=True)
        with mock.patch.dict(sys.modules, {"gsplat": fake_gsplat, "gsplat.rendering": fake_rendering}):
            with self.assertRaisesRegex(RuntimeError, "invalid configuration argument"):
                renderer.render(np.eye(4), np.eye(3), 960, 540)
        self.assertEqual(attempts["count"], 1)
        renderer.degrade_for_oom.assert_not_called()

    def test_viewer_error_log_is_persisted_for_direct_ply(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "trained_gaussians.ply"
            model.write_bytes(b"ply")
            self.assertEqual(viewer._viewer_log_directory(model), root.resolve())
            path = viewer._append_viewer_log(
                viewer._viewer_log_directory(model),
                {"status": "blocked", "phase": "interactive_render", "error": "synthetic"},
            )
            payload = json.loads(path.read_text(encoding="utf-8-sig").strip())
            self.assertEqual(payload["phase"], "interactive_render")
            self.assertEqual(payload["error"], "synthetic")

    def test_missing_direct_ply_does_not_become_a_directory(self):
        with TemporaryDirectory() as temp:
            target = Path(temp) / "missing_gaussians.ply"
            with self.assertRaisesRegex(ValueError, "Trained Gaussian PLY not found"):
                viewer.resolve_model(target)
            self.assertFalse(target.exists())
            self.assertEqual(viewer._viewer_log_directory(target), target.parent.resolve())


if __name__ == "__main__":
    unittest.main()
