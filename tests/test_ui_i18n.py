from pathlib import Path
import tempfile
import unittest
from unittest import mock

import ui_i18n


class UiI18nTests(unittest.TestCase):
    def test_required_language_keys_are_symmetric(self) -> None:
        self.assertEqual(set(ui_i18n.STRINGS["zh"]), set(ui_i18n.STRINGS["en"]))

    def test_runtime_formatting_uses_selected_language(self) -> None:
        self.assertEqual(ui_i18n.tr("zh", "viewer.heading"), "三维模型查看器")
        self.assertEqual(ui_i18n.tr("en", "viewer.heading"), "3D Model Viewer")
        self.assertIn(
            "12 accepted views",
            ui_i18n.tr("en", "app.3dgs_training_detail", frames=12, span=0.4, profile="Balanced quality"),
        )

    def test_viewer_strings_do_not_claim_a_point_cloud_display(self) -> None:
        # The trained-model viewer must not be described the way the Open3D
        # diagnostic point display is; the two are different capabilities.
        self.assertIn("gsplat", ui_i18n.tr("en", "gsview.backend", count=1, degree=3, accepted="passed", psnr="20.0"))
        self.assertIn("gsplat", ui_i18n.tr("zh", "gsview.backend", count=1, degree=3, accepted="通过", psnr="20.0"))

    def test_language_preference_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ui_preferences.json"
            with mock.patch("ui_i18n.preference_path", return_value=path):
                ui_i18n.save_language("en")
                self.assertEqual(ui_i18n.load_language(), "en")
                ui_i18n.save_language("zh")
                self.assertEqual(ui_i18n.load_language(), "zh")


if __name__ == "__main__":
    unittest.main()
