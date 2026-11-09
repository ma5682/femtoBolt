"""Regression checks for the compact capture page and empty-selection actions."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import unittest
from PySide6.QtWidgets import QApplication
from femto.app.qt_capture import CapturePage


class UiCleanupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_capture_refresh_never_reopens_removed_sidebar_or_details(self):
        page = CapturePage(connect_camera=False)
        try:
            page.show()
            page.refresh()
            self.assertFalse(hasattr(page, 'details_toggle'))
            self.assertFalse(hasattr(page, 'inspector_dialog'))
            self.assertFalse(page.inspector_scroll.isVisible())
            self.assertFalse(page.task_name.isVisible())
            self.assertEqual(page.count_label.text(), '0 / 3 帧')
            page.hide()
            page.show()
            page.refresh()
            self.assertFalse(page.inspector_scroll.isVisible())
            self.assertTrue(page.frame_progress.isVisible())
        finally:
            page.shutdown()
            page.close()

    def test_empty_selection_actions_follow_clear_and_undo(self):
        from test_qt_planning import QtPlanningTests
        fixture = QtPlanningTests()
        fixture.setUpClass()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        page = fixture.page
        self.assertFalse(page.clear_button.isEnabled())
        self.assertFalse(hasattr(page,'fit_selection_button'))
        page.canvas.work[10:20, 10:20] = 1
        page._selection_changed()
        self.assertTrue(page.clear_button.isEnabled())
        page.clear_button.click()
        self.assertFalse(page.clear_button.isEnabled())
        page.canvas.undo()
        self.assertTrue(page.clear_button.isEnabled())
