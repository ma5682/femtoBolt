"""P5 editor, actual background application, expiry and late-result guards."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from copy import deepcopy
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication, QTableWidgetItem

from femto.app.qt_planning import PlanningPage
from femto.app.qt_planning_io import clone_model
from femto.app.qt_review import ReviewPage
from femto.app.qt_variable_footprint import VariableFootprintDialog
from femto.planning import variable_width_workflow, variable_footprint
from femto.planning.planning_drafts import Journal
from test_strict_workflow import strict_fixture
from test_variable_footprint import profile


class QtVariableWidthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QApplication.instance() or QApplication([])
        cls.temp=tempfile.TemporaryDirectory();cls.base=strict_fixture(Path(cls.temp.name)/'source')
        model,region,_,plan,_=cls.base
        cls.profile=profile(plan['treatment_head']['effective_treatment_width_mm']/2000)
        cls.variable_model=clone_model(model,freeze_inputs=True)
        cls.variable_plan=variable_width_workflow.calculate(cls.variable_model,region,plan,cls.profile)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def setUp(self):
        self.case=tempfile.TemporaryDirectory();self.addCleanup(self.case.cleanup);self.root=Path(self.case.name)
        model,region,_,plan,_=self.base
        self.model=clone_model(model,freeze_inputs=True);self.region=deepcopy(region)
        self.model.store.regions=[self.region];self.model.saved_plan=deepcopy(plan)
        self.model.reconstruction=self.root/'work';self.model.reconstruction.mkdir()
        self.page=PlanningPage();self.page.resize(1120,740);self.page.show()
        self.page.install_model(self.model,self.root/'source.zip',Journal(self.root/'source.zip',self.root/'drafts'))
        self.page.set_stage(1);self.addCleanup(self.close)

    def close(self):self.page.shutdown();self.page.close();self.app.processEvents()

    def wait_task(self):
        loop=QEventLoop();timer=QTimer();timer.setSingleShot(True);timer.timeout.connect(loop.quit)
        self.page.runner.finished.connect(loop.quit)
        try:
            timer.start(90000)
            if self.page.busy:loop.exec()
        finally:self.page.runner.finished.disconnect(loop.quit);timer.stop()
        self.assertFalse(self.page.busy)

    def install_variable(self):
        self.page.model.target_ledger=self.variable_model.target_ledger
        self.page.model.coverage_certificate=None;self.page.model.coverage_display=None
        self.page.plan=deepcopy(self.variable_plan);self.page.canvas.set_plan(self.page.plan);self.page.refresh()

    def test_editor_preserves_imported_model_without_unit_roundtrip_changes(self):
        dialog=VariableFootprintDialog(10.,self.profile)
        try:
            self.assertEqual(dialog.value(),self.profile)
            dialog.table.setItem(1,1,QTableWidgetItem('0'))
            changed=dialog.value()
            self.assertEqual(changed['knots'][1]['lower_half_width_m'],0.)
            self.assertNotEqual(changed['sha256'],self.profile['sha256'])
            dialog.table.setItem(1,0,QTableWidgetItem('0'))
            dialog.accept_model();self.assertTrue(dialog.error.text());self.assertIsNone(dialog.profile)
        finally:dialog.close()

    def test_actual_apply_restore_clear_coverage_and_require_new_review(self):
        original=deepcopy(self.page.plan)
        self.page._variable_operation(self.profile);self.wait_task()
        self.assertFalse(self.page._error,self.page.status_label.text())
        self.assertEqual(self.page.plan['surface_trajectory']['planner']['algorithm_version'],'nominal-variable-width-surface-v1')
        self.assertNotIn('coverage_evidence',self.page.plan);self.assertIsNone(self.page.model.coverage_certificate)
        self.assertFalse(self.page.reviewed.isChecked());self.assertTrue(self.page.usable());self.assertTrue(self.page.dirty)
        self.assertIn('重新核验',self.page.status_label.text())
        self.page._variable_operation(None);self.wait_task()
        self.assertFalse(self.page._error,self.page.status_label.text())
        self.assertEqual(self.page.plan['surface_trajectory'],original['surface_trajectory'])
        self.assertNotIn('coverage_evidence',self.page.plan)

    def test_controls_follow_model_busy_and_readonly_states(self):
        self.assertTrue(self.page.coverage.variable.isEnabled())
        self.install_variable();panel=self.page.coverage
        self.assertTrue(panel.variable_restore.isEnabled());self.assertFalse(panel.refine.isVisible())
        self.assertIn('名义变宽',panel.profile_summary.text())
        panel.set_context(self.model,self.page.plan,ready=True,busy=True)
        self.assertFalse(panel.variable.isEnabled());self.assertFalse(panel.variable_restore.isEnabled())
        model=clone_model(self.variable_model,freeze_inputs=True);model.saved_plan=deepcopy(self.variable_plan)
        page=ReviewPage();page.resize(1120,740);page.show()
        try:
            page.install_model(model,self.root/'history.zip')
            self.assertFalse(page.coverage.variable.isVisible());self.assertFalse(page.coverage.variable_restore.isVisible())
            self.assertIn('名义变宽',page.coverage.profile_summary.text())
        finally:page.shutdown();page.close();self.app.processEvents()

    def delayed(self):
        entered=threading.Event();release=threading.Event()
        def run(model,region,plan,profile,check,progress):
            entered.set()
            if not release.wait(5):raise RuntimeError('test release timeout')
            check();return deepcopy(self.variable_plan)
        return entered,release,run

    def test_cancel_preserves_current_route_and_all_caches(self):
        before=self.page.plan;cache=self.model.target_ledger,self.model.coverage_certificate,self.model.coverage_display
        entered,release,run=self.delayed()
        with patch.object(variable_width_workflow,'calculate',side_effect=run):
            self.page._variable_operation(self.profile);self.assertTrue(entered.wait(2))
            self.page.runner.cancel();release.set();self.wait_task()
        self.assertIs(self.page.plan,before)
        for a,b in zip(cache,(self.model.target_ledger,self.model.coverage_certificate,self.model.coverage_display)):self.assertIs(a,b)

    def test_input_generation_and_full_plan_changes_both_discard_late_result(self):
        for kind in ('generation','plan'):
            entered,release,run=self.delayed()
            with patch.object(variable_width_workflow,'calculate',side_effect=run):
                self.page._variable_operation(self.profile);self.assertTrue(entered.wait(2))
                if kind=='generation':self.page._input_generation+=1
                else:
                    self.page.plan=deepcopy(self.page.plan);self.page.plan['display_note']='changed'
                before=self.page.plan;release.set();self.wait_task()
                self.assertIs(self.page.plan,before)

    def test_navigation_preserves_model_and_upstream_change_disables_coverage(self):
        self.install_variable();before=deepcopy(self.page.plan)
        self.page.set_stage(0);self.page.set_stage(1)
        self.assertEqual(self.page.plan,before);self.assertTrue(self.page.usable())
        self.page.head.fields['effective_treatment_width_mm'].setText('1.5')
        self.assertFalse(self.page.usable());self.assertFalse(self.page.coverage.verify.isEnabled())
        self.assertFalse(self.page.coverage.variable.isEnabled());self.assertFalse(self.page.coverage.variable_restore.isEnabled())
        self.assertEqual(self.page.plan,before)
