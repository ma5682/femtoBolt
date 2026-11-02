"""P4 Qt opt-in, ordinary-route restoration and asynchronous state guards."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')

from copy import deepcopy
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from PySide6.QtCore import QEventLoop,QTimer
from PySide6.QtWidgets import QApplication
from femto.app.qt_planning import PlanningPage
from femto.app.qt_review import ReviewPage
from femto.app.qt_planning_io import clone_model
from femto.planning.planning_drafts import Journal
from test_strict_workflow import strict_fixture


class QtStrictTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QApplication.instance() or QApplication([])
        cls.temp=tempfile.TemporaryDirectory();cls.base=strict_fixture(Path(cls.temp.name)/'source')

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def setUp(self):
        self.case=tempfile.TemporaryDirectory();self.addCleanup(self.case.cleanup);self.root=Path(self.case.name)
        model,region,before,self.strict,original_cache=self.base
        self.model=clone_model(model,freeze_inputs=True);self.region=deepcopy(region)
        self.model.store.regions=[self.region]
        self.model.target_ledger,self.model.coverage_certificate,self.model.coverage_display=original_cache
        self.model.saved_plan=deepcopy(before)
        self.page=PlanningPage();self.page.resize(1120,740);self.page.show()
        self.page.install_model(self.model,self.root/'record.zip',Journal(self.root/'record.zip',self.root/'drafts'))
        self.page.set_stage(1);self.addCleanup(self.close)

    def close(self):self.page.shutdown();self.page.close();self.app.processEvents()

    def wait_task(self):
        loop=QEventLoop();timer=QTimer();timer.setSingleShot(True);timer.timeout.connect(loop.quit)
        self.page.runner.finished.connect(loop.quit)
        try:
            timer.start(45000)
            if self.page.busy:loop.exec()
        finally:self.page.runner.finished.disconnect(loop.quit);timer.stop()
        self.assertFalse(self.page.busy)

    def install_strict(self):
        model,_,_,plan,_=self.base
        self.page.model.target_ledger=model.target_ledger
        self.page.model.coverage_certificate=model.coverage_certificate
        self.page.model.coverage_display=model.coverage_display
        self.page.plan=deepcopy(plan);self.page.canvas.set_plan(self.page.plan);self.page.refresh()

    def delayed(self):
        entered=threading.Event();release=threading.Event()
        def job(model,region,plan,check,progress):
            entered.set()
            if not release.wait(5):raise RuntimeError('test release timed out')
            check();return deepcopy(self.strict)
        return entered,release,job

    def test_controls_are_explicit_and_follow_version_and_busy_state(self):
        panel=self.page.coverage
        self.assertTrue(panel.strict.isVisible());self.assertTrue(panel.strict.isEnabled())
        self.assertFalse(panel.ordinary.isVisible())
        panel.set_context(self.model,self.page.plan,ready=True,busy=True)
        self.assertFalse(panel.strict.isEnabled())
        self.install_strict()
        self.assertFalse(panel.strict.isVisible());self.assertTrue(panel.ordinary.isEnabled())
        self.assertEqual(panel.refine.text(),'严格补充规划')
        no_coverage=deepcopy(self.page.plan);no_coverage.pop('coverage_evidence')
        panel.set_context(self.model,no_coverage,ready=True)
        self.assertFalse(panel.refine.isEnabled());self.assertTrue(panel.verify.isEnabled())

    def test_actual_background_reconstruction_and_restore_clear_stale_coverage(self):
        original=deepcopy(self.page.plan)
        self.page._strict_operation(False);self.wait_task()
        self.assertFalse(self.page._error,self.page.status_label.text())
        self.assertEqual(self.page.plan['surface_trajectory']['planner']['algorithm_version'],'constructed-surface-route-v1')
        self.assertNotIn('coverage_evidence',self.page.plan);self.assertIsNone(self.page.model.coverage_certificate)
        self.assertTrue(self.page.dirty);self.assertIsNone(self.page.archive)
        self.assertFalse(self.page.reviewed.isChecked());self.assertTrue(self.page.usable())
        self.assertIn('重新核验',self.page.status_label.text())
        self.page._strict_operation(True);self.wait_task()
        self.assertFalse(self.page._error,self.page.status_label.text())
        self.assertEqual(self.page.plan['surface_trajectory'],original['surface_trajectory'])
        self.assertNotIn('coverage_evidence',self.page.plan)

    def test_cancelled_job_keeps_previous_route_and_all_caches(self):
        before=self.page.plan
        cache=self.page.model.target_ledger,self.page.model.coverage_certificate,self.page.model.coverage_display
        entered,release,job=self.delayed()
        with patch('femto.planning.strict_workflow.calculate',side_effect=job):
            self.page._strict_operation(False);self.assertTrue(entered.wait(2))
            self.page.runner.cancel();release.set();self.wait_task()
        self.assertIs(self.page.plan,before)
        for a,b in zip(cache,(self.page.model.target_ledger,self.page.model.coverage_certificate,self.page.model.coverage_display)):
            self.assertIs(a,b)

    def test_new_input_generation_cannot_receive_late_strict_result(self):
        before=self.page.plan;entered,release,job=self.delayed()
        with patch('femto.planning.strict_workflow.calculate',side_effect=job):
            self.page._strict_operation(False);self.assertTrue(entered.wait(2))
            self.page._input_generation+=1;release.set();self.wait_task()
        self.assertIs(self.page.plan,before)

    def test_full_plan_digest_guards_edits_outside_geometry_revision(self):
        entered,release,job=self.delayed()
        with patch('femto.planning.strict_workflow.calculate',side_effect=job):
            self.page._strict_operation(False);self.assertTrue(entered.wait(2))
            self.page.plan=deepcopy(self.page.plan);self.page.plan['display_note']='new edit'
            changed=self.page.plan;release.set();self.wait_task()
        self.assertIs(self.page.plan,changed);self.assertIn('旧重建结果未安装',self.page.status_label.text())

    def test_navigation_preserves_strict_route_but_parameter_change_disables_it(self):
        self.install_strict();before=deepcopy(self.page.plan)
        self.page.set_stage(0);self.page.set_stage(1)
        self.assertEqual(self.page.plan,before);self.assertTrue(self.page.usable())
        self.page.head.fields['effective_treatment_width_mm'].setText('10')
        self.assertFalse(self.page.usable());self.assertFalse(self.page.coverage.ordinary.isEnabled())
        self.assertEqual(self.page.plan,before)

    def test_history_hides_mutations_and_retains_coverage_overlay(self):
        model,_,_,plan,_=self.base
        model=clone_model(model,freeze_inputs=True);model.saved_plan=deepcopy(plan)
        page=ReviewPage();page.resize(1120,740);page.show()
        try:
            page.install_model(model,self.root/'history.zip')
            for button in (page.coverage.strict,page.coverage.ordinary,page.coverage.refine,page.coverage.restore,page.coverage.verify):
                self.assertFalse(button.isVisible())
            page.coverage.show_layer.setChecked(True);self.assertIsNotNone(page.canvas._coverage_image)
        finally:page.shutdown();page.close();self.app.processEvents()
