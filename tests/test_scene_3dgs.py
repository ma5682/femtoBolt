from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from femto.gsplat.scene_3dgs import build_training_command


class ThreeDGSCommandTests(unittest.TestCase):
    def test_supported_quality_features_are_enabled_from_installed_help(self) -> None:
        data = {"input": "dataset", "photometric_unstable": True}
        help_text = " ".join(
            (
                "--pipeline.model.rasterize-mode",
                "--pipeline.model.use-scale-regularization",
                "--pipeline.model.use-bilateral-grid",
            )
        )
        with TemporaryDirectory() as temporary:
            command, features = build_training_command(
                "ns-train",
                data,
                Path(temporary),
                profile="balanced",
                iterations=30_000,
                help_text=help_text,
            )
        self.assertIn("antialiasing", features["enabled"])
        self.assertIn("scale_regularization", features["enabled"])
        self.assertIn("bilateral_grid", features["enabled"])
        self.assertIn("antialiased", command)

    def test_unsupported_flags_are_not_passed_to_runtime(self) -> None:
        data = {"input": "dataset", "photometric_unstable": False}
        with TemporaryDirectory() as temporary:
            command, features = build_training_command(
                "ns-train",
                data,
                Path(temporary),
                profile="low_vram",
                iterations=100,
                help_text="usage: ns-train splatfacto",
            )
        self.assertNotIn("--pipeline.model.rasterize-mode", command)
        self.assertEqual(set(features["skipped"]), {"antialiasing", "scale_regularization"})


if __name__ == "__main__":
    unittest.main()
