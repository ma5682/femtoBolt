"""Route presentation uses the existing geometry, state and playback gates."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')

from copy import deepcopy
import unittest
from unittest.mock import patch
from PySide6.QtCore import QEventLoop, QTimer, QPoint
from PySide6.QtWidgets import QApplication

from femto.app.qt_main import WorkstationWindow, apply_theme
from femto.app.qt_planning import HeadForm
import test_qt_planning as planning_fixture
from test_region_quality import task_context


class RouteLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QApplication.instance() or QApplication([]);apply_theme(cls.app)

    def setUp(self):
        fixture=planning_fixture.QtPlanningTests();fixture.setUpClass();fixture.setUp()
        self.fixture=fixture;fixture.page.hide();self.addCleanup(fixture.doCleanups)
        self.window=WorkstationWindow(connect_camera=False);self.addCleanup(self.close_window)
        self.page=self.window.planning
        fixture.model.task_context=task_context()
        self.page.install_model(fixture.model,fixture.model.reconstruction,fixture.journal)
        self.window.pages.setCurrentWidget(self.page);self.window.resize(1120,740);self.window.show()

    def close_window(self):
        self.window.close();self.app.processEvents()

    def settle(self):
        loop=QEventLoop();QTimer.singleShot(30,loop.quit);loop.exec()

    def wait_task(self):
        loop=QEventLoop();deadline=QTimer();deadline.setSingleShot(True)
        deadline.timeout.connect(loop.quit);self.page.runner.finished.connect(loop.quit)
        try:
            deadline.start(60000)
            if self.page.busy:loop.exec()
        finally:
            deadline.stop();self.page.runner.finished.disconnect(loop.quit)
        self.assertFalse(self.page.busy);self.assertFalse(self.page._error);self.settle()

    def generate(self):
        self.page.canvas.work[:]=self.page.model.projection.valid
        self.page.canvas.forbidden[:,48:53]=1
        self.page.canvas.rebuild_layers();self.page._selection_changed()
        self.page.commit_region();self.wait_task()
        self.page.generate();self.wait_task()
        self.assertTrue(self.page.usable())

    def test_generated_route_exposes_playback_and_summary_without_changing_evidence(self):
        self.generate();page=self.page
        plan=deepcopy(page.plan);findings=page.current_findings()
        self.assertFalse(page.parameters_button.isChecked())
        self.assertFalse(page.route_details_toggle.isChecked())
        self.assertFalse(page.warning.isVisible())
        self.assertIn(f'待核对 {len(findings)} 项',page.route_attention.text())
        self.assertIn('目标覆盖尚未完成核验',page.route_attention.text())
        for widget,parent in ((page.play_button,self.window),
                              (page.route_attention,page.inspector_scroll.viewport())):
            origin=widget.mapTo(parent,QPoint(0,0))
            self.assertTrue(widget.isVisible())
            self.assertTrue(parent.rect().contains(origin))
            self.assertTrue(parent.rect().contains(origin+QPoint(widget.width()-1,widget.height()-1)))
        self.assertFalse(page.route_details_toggle.isVisible())
        self.assertFalse(page.route_stats.isVisible())
        self.assertFalse(page.route_extra.isVisible())
        self.assertFalse(page.route_summary.isVisible())
        for item in findings:self.assertIn(item,page.route_attention.toolTip())
        self.assertTrue(all(label.isHidden() for label in page.coverage.labels[:2]))
        self.assertTrue(page.coverage.verify.isVisible())
        self.assertEqual(page.plan,plan)
        page.play_button.click()
        self.assertTrue(page.canvas.timer.isActive());self.assertEqual(page.play_button.text(),'暂停播放')
        page.play_button.click();self.assertFalse(page.canvas.timer.isActive())
        page.canvas._play_index=len(page.canvas._samples)-1;page.canvas._advance();page._refresh_playback()
        self.assertEqual(page.play_button.text(),'重播路线')
        self.assertEqual(page.plan,plan);self.assertEqual(page.current_findings(),findings)
        page.quality.invalidate()
        self.assertIn('几何支持尚未核验',page.route_attention.text())
        self.assertFalse(page.reviewed.isChecked())

    def test_parameter_edit_and_calculation_failure_keep_warnings_visible(self):
        from femto.planning.treatment_head import head_parameters
        self.page.head.set_record(head_parameters('direct_row_spacing',row_spacing_mm=5))
        self.generate();page=self.page;previous=page.plan
        page.parameters_button.click()
        page.head.fields['row_spacing_mm'].setText('7');self.settle()
        self.assertIs(page.plan,previous);self.assertFalse(page.usable())
        self.assertFalse(page.play_button.isEnabled())
        self.assertTrue(page.warning.isVisible());self.assertIn('旧路线',page.warning.text())
        self.assertIn('重新计算',page.status_label.text())
        page.reviewed.setChecked(True)
        with patch('femto.app.qt_planning.surface_workspace.export') as export:page.save_plan()
        export.assert_not_called()
        page.head.fields['effective_treatment_width_mm'].setText('0')
        self.assertFalse(page.primary.isEnabled());self.assertTrue(page.head.error.isVisible())
        page.head.fields['effective_treatment_width_mm'].setText('5')
        page.head.fields['row_spacing_mm'].setText('5')
        self.assertTrue(page.usable());self.assertFalse(page.reviewed.isChecked())
        page._failure(RuntimeError('synthetic calculation failure'))
        self.assertIs(page.plan,previous)
        # The current error stays in the fixed action area and full details;
        # the sidebar retains geometry findings rather than duplicating paths.
        self.assertTrue(page.warning.isVisible())
        self.assertIn('待核对',page.warning.text())
        self.assertTrue(page.action_dock.isVisible())
        self.assertIn('synthetic calculation failure',page.status_label.text())
        self.assertIn('synthetic calculation failure',page.status_label.toolTip())
        self.assertNotIn('synthetic calculation failure',page.warning.text())
        with patch('femto.app.qt_planning.show_text') as details:page.show_details()
        self.assertIn('synthetic calculation failure',details.call_args.args[2])
        self.app.processEvents()
        self.assertLessEqual(page.inspector_scroll.widget().width(),page.inspector_scroll.viewport().width())

    def test_parameter_modes_hide_unused_rows_without_resetting_values_or_limits(self):
        head=HeadForm();head.show();self.addCleanup(head.close);self.settle()
        self.assertEqual(head.mode.currentData(),'fixed_width_auto')
        head.mode.setCurrentIndex(0);self.settle()
        self.assertFalse(head.fields['overlap_percent'].isVisible())
        self.assertEqual(head.record()['overlap_percent'],20)
        head.mode.setCurrentIndex(1);self.settle()
        self.assertFalse(head.fields['row_spacing_mm'].isVisible())
        self.assertTrue(head.fields['overlap_percent'].isVisible())
        head.fields['effective_treatment_width_mm'].setText('8')
        head.fields['overlap_percent'].setText('25')
        self.assertEqual(head.record()['row_spacing_mm'],6)
        head.mode.setCurrentIndex(0);self.settle()
        self.assertEqual(head.record()['row_spacing_mm'],5)
        self.assertEqual(head.record()['overlap_percent'],25)
        self.assertEqual(head.record()['effective_treatment_width_mm'],8)
        self.assertFalse(head.record()['overlap_used_for_row_spacing'])
        head.fields['effective_treatment_width_mm'].setText('0')
        with self.assertRaises(ValueError):head.record()
        self.assertIn('参数无效',head.error.text())


if __name__=='__main__':unittest.main()
