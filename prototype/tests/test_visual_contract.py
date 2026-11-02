"""Regression checks for QSS layout, feedback and toolbar interactions.

Run separately from the QCoreApplication controller suite.
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QPalette
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from app.main_window import MainWindow
from widgets.viewport import MockViewport


class VisualContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.app.setStyleSheet((ROOT / "styles/medical.qss").read_text(encoding="utf-8"))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        with patch("app.main_window.load_project_data") as loader:
            self.window = MainWindow(records_dir=Path(self.temp.name))
            loader.assert_not_called()
        self.window.show()
        self.app.processEvents()

    def tearDown(self):
        self.window.controller.capture_timer.stop()
        self.window.controller.operation_timer.stop()
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()
        self.temp.cleanup()

    def test_fixed_controls_survive_stylesheet_polishing_and_resize(self):
        w = self.window
        for width, height in ((1600, 900), (1280, 760)):
            w.resize(width, height)
            self.app.processEvents()
            self.assertEqual((w.width(), w.height()), (width, height))
            self.assertEqual((w.action_bar.primary.width(), w.action_bar.primary.height()), (290, 52))
            self.assertTrue(all(b.height() == 80 for b in w.sidebar.buttons))
            self.assertTrue(all(b.height() == 42 for b in w.view_buttons if b.isVisible()))
            for tool in (w.fit_button, w.zoom_button, w.one_button, w.full_button):
                self.assertEqual((tool.width(), tool.height()), (40, 40))
                self.assertFalse(tool.icon().isNull())
            self.assertGreaterEqual(w.viewport.width(), 520)
            self.assertGreaterEqual(w.viewport.height(), 380)
            self.assertTrue(w.rect().contains(w.action_bar.primary.mapTo(w, QPoint(289, 51))))
        self.assertGreater(w.sidebar.scroll.verticalScrollBar().maximum(), 0)

    def test_semantic_colors_and_gate_feedback(self):
        w = self.window
        c = w.controller
        self.assertIsInstance(w.viewport, MockViewport)
        self.assertTrue(w.viewport._is_empty())
        self.assertFalse(w.sidebar.buttons[3].isEnabled())
        self.assertEqual(w.pages[0].frames.palette().color(QPalette.WindowText).name(), "#637b8d")
        c.set_scenario("quality")
        QTest.mouseClick(w.action_bar.primary, Qt.LeftButton)
        c.capture_timer.stop()
        self.assertFalse(w.action_bar.primary.isEnabled())
        self.assertIn("3", w.action_bar.primary.toolTip())
        restart_action = w.action_bar.more_menu.actions()[0]
        c.tick_capture()
        c.tick_capture()
        self.assertIs(w.action_bar.more_menu.actions()[0], restart_action)
        QTest.mouseClick(w.action_bar.primary, Qt.LeftButton)
        c.operation_timer.stop()
        c.complete_operation()
        self.app.processEvents()
        self.assertEqual(c.state.capture.phase, "failed")
        self.assertEqual(w.pages[0].phase.palette().color(QPalette.WindowText).name(), "#f06464")
        self.assertEqual(w.action_bar.notice.palette().color(QPalette.WindowText).name(), "#f06464")
        self.assertTrue(w.sidebar.buttons[2].blocked)
        self.assertFalse(w.sidebar.buttons[3].isEnabled())

    def test_three_dimensional_view_and_navigation_tools(self):
        w = self.window
        QTest.mouseClick(w.view_buttons[2], Qt.LeftButton)
        self.assertEqual(w.controller.state.view, "点云")
        self.assertTrue(w.reset_button.isVisible())
        w.reset_button.menu().actions()[1].trigger()
        self.assertEqual(w.controller.state.view, "模型")
        QTest.mouseClick(w.zoom_button, Qt.LeftButton)
        self.assertGreater(w.viewport._zoom, 1)
        QTest.mouseClick(w.one_button, Qt.LeftButton)
        self.assertEqual(w.viewport._navigation_tool, "pan")
        QTest.mouseClick(w.one_button, Qt.LeftButton)
        self.assertEqual(w.viewport._navigation_tool, "select")
        QTest.mouseClick(w.fit_button, Qt.LeftButton)
        self.assertEqual(w.viewport._zoom, 1)


if __name__ == "__main__":
    unittest.main()
