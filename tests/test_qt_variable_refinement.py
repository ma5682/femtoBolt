"""Actual P5 UI route plus inherited asynchronous cancellation/digest guards."""
import test_qt_refinement as qt_base
import test_variable_refinement as core
from PySide6.QtWidgets import QApplication
from copy import deepcopy
from PySide6.QtCore import QEventLoop,QTimer
from unittest.mock import patch
from datetime import date
from pathlib import Path
from PySide6.QtGui import QFontDatabase,QFont


class QtVariableRefinementTests(qt_base.QtRefinementTests):
    @classmethod
    def setUpClass(cls):
        cls.app=QApplication.instance() or QApplication([])
        if core.os.environ.get('FEMTO_ITERATION_PREVIEW'):
            identifier=QFontDatabase.addApplicationFont('C:/Windows/Fonts/msyh.ttc')
            families=QFontDatabase.applicationFontFamilies(identifier)
            if not families:raise RuntimeError('local Chinese preview font unavailable')
            cls.app.setFont(QFont(families[0],9))
        core.VariableRefinementTests.setUpClass()
        fixture=core.VariableRefinementTests
        cls.base=(fixture.final_model,fixture.base[1],fixture.variable,fixture.final_plan)

    @classmethod
    def tearDownClass(cls):core.VariableRefinementTests.tearDownClass()

    def wait_task(self):
        loop=QEventLoop();timer=QTimer();timer.setSingleShot(True);timer.timeout.connect(loop.quit)
        self.page.runner.finished.connect(loop.quit)
        try:
            timer.start(150000)
            if self.page.busy:loop.exec()
        finally:self.page.runner.finished.disconnect(loop.quit);timer.stop()
        self.assertFalse(self.page.busy)

    def test_actual_background_refine_and_restore_preserve_original_route(self):
        original=deepcopy(self.page.plan)
        self.page.refine_coverage();self.wait_task()
        self.assertFalse(self.page._error,self.page.status_label.text())
        self.assertEqual(self.page.plan['surface_trajectory']['planner']['algorithm_version'],core.refinement.ALGORITHM)
        self.assertTrue(self.page.dirty);self.assertIsNone(self.page.archive);self.assertTrue(self.page.usable())
        self.assertFalse(self.page.reviewed.isChecked())
        self.page.restore_coverage_route();self.wait_task()
        self.assertEqual(self.page.plan['surface_trajectory'],original['surface_trajectory'])
        self.assertEqual(self.page.plan['coverage_evidence'],original['coverage_evidence'])

    def test_controls_require_coverage_and_follow_busy_and_refined_state(self):
        super().test_controls_require_coverage_and_follow_busy_and_refined_state()
        output=core.os.environ.get('FEMTO_ITERATION_PREVIEW')
        if output:
            path=Path(output);path.mkdir(parents=True,exist_ok=True)
            self.app.processEvents()
            self.assertTrue(self.page.coverage.grab().save(str(path/'variable_refined_panel.png')))

    def test_expired_refined_route_can_restore_only_when_inputs_still_match(self):
        self.install_refined();require=core.variable_footprint.require_current
        with patch.object(core.variable_footprint,'require_current',side_effect=lambda model:require(model,on=date(2100,1,1))):
            self.page.refresh()
            self.assertFalse(self.page.usable());self.assertTrue(self.page.coverage.restore.isEnabled())
            self.page.head.fields['effective_treatment_width_mm'].setText('10')
            self.assertFalse(self.page.coverage.restore.isEnabled())

    def test_model_expiring_before_ui_install_discards_late_result(self):
        before=self.page.plan;certificate=self.page.model.coverage_certificate
        entered,release,job=self.delayed();require=core.variable_footprint.require_current
        with patch('femto.planning.refinement_workflow.calculate',side_effect=job):
            self.page.refine_coverage();self.assertTrue(entered.wait(2))
            with patch.object(core.variable_footprint,'require_current',side_effect=lambda model:require(model,on=date(2100,1,1))):
                release.set();self.wait_task()
        self.assertIs(self.page.plan,before);self.assertIs(self.page.model.coverage_certificate,certificate)

    def test_changed_depth_without_generation_event_rejects_late_result(self):
        before=self.page.plan;certificate=self.page.model.coverage_certificate
        entered,release,job=self.delayed()
        with patch('femto.planning.refinement_workflow.calculate',side_effect=job):
            self.page.refine_coverage();self.assertTrue(entered.wait(2))
            depth=self.page.model.projection.depth.copy();depth.flat[0]+=1
            self.assertFalse(core.np.array_equal(depth,self.page.model.projection.depth))
            self.page.model.projection.depth=depth
            release.set();self.wait_task()
        self.assertIs(self.page.plan,before);self.assertIs(self.page.model.coverage_certificate,certificate)
