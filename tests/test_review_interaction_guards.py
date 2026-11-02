"""Presentation guards use existing task cancellation and saved-data eligibility."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from pathlib import Path
import threading
import unittest

from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication

from femto.app.qt_planning import PlanningPage
from femto.app.qt_review import ReviewPage
import test_qt_planning as fixture_module


class ReviewInteractionGuardsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def wait(self, runner):
        loop = QEventLoop(); deadline = QTimer(); deadline.setSingleShot(True)
        deadline.timeout.connect(loop.quit); runner.finished.connect(loop.quit)
        try:
            deadline.start(5000)
            if runner.busy: loop.exec()
        finally:
            deadline.stop(); runner.finished.disconnect(loop.quit)
        self.assertFalse(runner.busy)

    def check_cancel(self, page, button, status):
        release = threading.Event(); installed = []; failures = []
        try:
            page._status('正在处理测试记录…')
            page.runner.start(lambda cancel, progress: (release.wait(5), 'late')[1],
                              installed.append, failures.append)
            page.refresh()
            button.click()
            self.assertTrue(page.busy)
            self.assertIn('正在取消', status.text())
            self.assertFalse(button.isEnabled())
            # Neither another job nor a late success may bypass cancellation.
            self.assertFalse(page.runner.start(lambda *_: 'duplicate', installed.append))
            release.set(); self.wait(page.runner)
            self.assertEqual(installed, [])
            self.assertEqual(len(failures), 1)
            self.assertTrue(button.isHidden())
            page.runner.start(lambda *_: 'retry', installed.append, failures.append)
            page.refresh()
            self.assertTrue(button.isEnabled())
            self.wait(page.runner)
            self.assertEqual(installed, ['retry'])
        finally:
            release.set(); self.wait(page.runner)
            page.shutdown(); page.close()

    def test_planning_cancel_waits_for_completion_then_allows_retry(self):
        page = PlanningPage()
        self.check_cancel(page, page.cancel_button, page.status_label)

    def test_review_cancel_waits_for_completion_then_allows_retry(self):
        page = ReviewPage()
        self.check_cancel(page, page.cancel, page.status)

    def review_fixture(self, *, region=False):
        case = fixture_module.QtPlanningTests(); case.setUpClass(); case.setUp()
        self.addCleanup(case.doCleanups)
        if region: case.mark()
        page = ReviewPage()
        self.addCleanup(page.close); self.addCleanup(page.shutdown)
        page.install_model(case.page.model, Path(case.temp.name) / 'record')
        return page

    def test_model_only_record_cannot_enter_nonexistent_region_or_route(self):
        page = self.review_fixture()
        self.assertIsNone(page.region); self.assertIsNone(page.plan)
        self.assertFalse(page.layers.model().item(1).isEnabled())
        self.assertFalse(page.layers.model().item(2).isEnabled())
        for index in (1, 2):
            page.layers.setCurrentIndex(index)
            self.assertEqual(page.layers.currentIndex(), 0)
            self.assertEqual(page.workflow.current_step, 2)
            self.assertFalse(page.canvas.editable)

    def test_unconfirmed_review_routes_to_capture_confirmation_not_editing(self):
        page=self.review_fixture()
        source=Path('unconfirmed/reconstruction').resolve()
        page.model.capture_source=source
        page.model.acquisition_completion=dict(confirmed_for_planning=False)
        capture=[];editing=[]
        page.capture_requested.connect(capture.append);page.edit_requested.connect(editing.append)
        page.refresh()
        self.assertEqual(page.edit.text(),'检查并确认最终模型')
        page.edit.click()
        self.assertEqual(capture,[str(source)]);self.assertEqual(editing,[])
        self.assertFalse(page.model.acquisition_completion['confirmed_for_planning'])

    def test_region_without_route_keeps_region_view_and_blocks_route(self):
        page = self.review_fixture(region=True)
        self.assertIsNotNone(page.region); self.assertIsNone(page.plan)
        self.assertTrue(page.layers.model().item(1).isEnabled())
        self.assertFalse(page.layers.model().item(2).isEnabled())
        page.layers.setCurrentIndex(2)
        self.assertEqual(page.layers.currentIndex(), 1)
        self.assertEqual(page.workflow.current_step, 3)
        self.assertFalse(page.show_workflow_step(4))

    def test_unconfirmed_record_without_capture_disables_dead_end_action(self):
        page = self.review_fixture()
        page.model.capture_source = None
        page.model.acquisition_completion = dict(confirmed_for_planning=False)
        page.refresh()
        self.assertFalse(page.edit.isEnabled())
        self.assertIn('关联采集不可用', page.edit.toolTip())
        self.assertFalse(page.quality_toggle.isEnabled())
        self.assertTrue(page.source_details.isHidden())
        page.source_toggle.click()
        self.assertFalse(page.source_details.isHidden())

if __name__ == '__main__':
    unittest.main()
