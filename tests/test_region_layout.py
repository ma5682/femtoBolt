"""Compact marking controls keep real quality feedback and editing semantics."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')

from copy import deepcopy
import unittest
import numpy as np
from PySide6.QtCore import QEventLoop, QTimer, QPoint
from PySide6.QtWidgets import QApplication

from femto.app.qt_main import WorkstationWindow, apply_theme
import test_qt_planning as planning_fixture
from test_region_quality import task_context


class RegionLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QApplication.instance() or QApplication([]);apply_theme(cls.app)

    def setUp(self):
        fixture=planning_fixture.QtPlanningTests();fixture.setUpClass();fixture.setUp()
        self.fixture=fixture;self.addCleanup(fixture.doCleanups);fixture.page.hide()
        self.window=WorkstationWindow(connect_camera=False)
        self.addCleanup(self.close_window)
        self.page=self.window.planning
        fixture.model.task_context=task_context()
        self.page.install_model(fixture.model,fixture.model.reconstruction,fixture.journal)
        self.window.pages.setCurrentWidget(self.page)
        self.window.resize(1120,740);self.window.show()
        self.wait_quality()

    def close_window(self):
        self.window.close();self.app.processEvents()

    def settle(self):
        loop=QEventLoop();QTimer.singleShot(30,loop.quit);loop.exec()

    def wait_quality(self):
        loop=QEventLoop();deadline=QTimer();deadline.setSingleShot(True)
        deadline.timeout.connect(loop.quit)
        def changed():
            if self.page.quality.result is not None:loop.quit()
        self.page.quality.changed.connect(changed)
        try:
            deadline.start(10000)
            if self.page.quality.result is None:loop.exec()
        finally:
            deadline.stop();self.page.quality.changed.disconnect(changed)
        self.assertIsNotNone(self.page.quality.result);self.settle()

    def assert_inside_inspector(self,widget):
        viewport=self.page.inspector_scroll.viewport()
        origin=widget.mapTo(viewport,QPoint(0,0))
        self.assertTrue(widget.isVisible())
        self.assertTrue(viewport.rect().contains(origin),widget.objectName())
        self.assertTrue(viewport.rect().contains(origin+QPoint(widget.width()-1,widget.height()-1)))

    def test_small_window_shows_tools_and_pending_quality_without_scrolling(self):
        self.page.canvas.work[:]=1;self.page.canvas.rebuild_layers();self.page._selection_changed()
        self.wait_quality()
        self.assertGreater(self.page.quality.result['pending'],0)
        self.assertFalse(self.page.source_details.isVisible())
        self.assertTrue(self.page.more_region.isVisible())
        self.assertEqual(self.page.inspector_scroll.verticalScrollBar().maximum(),0)
        for widget in (*self.page.paint_buttons,self.page.brush,self.page.undo_button,
                       self.page.redo_button,self.page.clear_button,self.page.fit_button,self.page.quality.summary,
                       self.page.quality.focus):
            self.assert_inside_inspector(widget)
        self.assertTrue(self.page.quality.legend.isHidden())
        self.assertIn('一致比例',self.page.quality.summary.text())
        self.assertIn('待检查像素',self.page.quality.summary.accessibleDescription())
        work=self.page.canvas.work.copy();forbidden=self.page.canvas.forbidden.copy()
        self.page.quality.focus.click()
        np.testing.assert_array_equal(self.page.canvas.work,work)
        np.testing.assert_array_equal(self.page.canvas.forbidden,forbidden)
        self.assertTrue(self.page.canvas.support)
        self.assertEqual(self.page._stage,0)

    def test_clear_undo_and_redo_are_distinct_and_keep_legacy_exclusions(self):
        canvas=self.page.canvas
        canvas.work[20:70,20:70]=1;canvas.forbidden[30:35,30:35]=1
        canvas.rebuild_layers();self.page._selection_changed();self.wait_quality()
        work=canvas.work.copy();forbidden=canvas.forbidden.copy()
        self.assertTrue(self.page.clear_button.isVisible())
        self.assertTrue(self.page.legacy_exclusion_notice.isVisible())
        self.page.clear_button.click()
        self.assertFalse(canvas.work.any())
        np.testing.assert_array_equal(canvas.forbidden,forbidden)
        self.page.undo_button.click()
        np.testing.assert_array_equal(canvas.work,work)
        np.testing.assert_array_equal(canvas.forbidden,forbidden)
        self.page.redo_button.click()
        self.assertFalse(canvas.work.any())
        np.testing.assert_array_equal(canvas.forbidden,forbidden)
        self.page.undo_button.click();self.wait_quality()
        self.assertTrue(self.page.more_region.isVisible())
        self.assert_inside_inspector(self.page.quality.summary)

    def test_source_details_do_not_change_masks_context_or_geometry_gate(self):
        context=deepcopy(self.page.model.task_context)
        self.page.canvas.work[:]=0;self.page.canvas.work[:4,:4]=1
        self.page.canvas.rebuild_layers();self.page._selection_changed();self.wait_quality()
        work=self.page.canvas.work.copy()
        self.page.source_toggle.click();self.settle()
        self.assertTrue(self.page.source_label.isVisible())
        self.assertIn('W0 / RDF / m',self.page.source_label.text())
        self.assertIn(context['name'],self.page.source_label.text())
        self.page.source_toggle.click();self.settle()
        self.assertEqual(self.page.model.task_context,context)
        np.testing.assert_array_equal(self.page.canvas.work,work)
        self.page.primary.click()
        loop=QEventLoop();deadline=QTimer();deadline.setSingleShot(True)
        deadline.timeout.connect(loop.quit);self.page.runner.finished.connect(loop.quit)
        try:
            deadline.start(10000)
            if self.page.busy:loop.exec()
        finally:
            deadline.stop();self.page.runner.finished.disconnect(loop.quit)
        self.assertFalse(self.page.busy)
        self.assertIsNone(self.page.region)
        self.assertEqual(self.page._stage,0)
        self.assertIn('没有可核验',self.page.status_label.text())
        self.assertEqual(self.page.quality.result['supported'],0)
        np.testing.assert_array_equal(self.page.canvas.work,work)


if __name__=='__main__':unittest.main()
