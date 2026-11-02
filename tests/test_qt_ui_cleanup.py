"""Regression checks for progressive disclosure and empty-selection actions."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import unittest
from PySide6.QtWidgets import QApplication
from femto.app.qt_capture import CapturePage


class UiCleanupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_capture_details_survive_refresh_and_can_be_collapsed(self):
        page = CapturePage(connect_camera=False)
        try:
            self.assertFalse(page.details_toggle.isHidden())
            self.assertTrue(page.details_panel.isHidden())
            page.details_toggle.click()
            page.refresh()
            self.assertFalse(page.details_panel.isHidden())
            page.details_toggle.click()
            page.refresh()
            self.assertTrue(page.details_panel.isHidden())
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
