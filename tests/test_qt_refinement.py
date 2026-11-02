"""P3 Qt lifecycle: frozen jobs, late-result guards, cancellation and rollback."""
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
from femto.planning import refinement_workflow,coverage_workflow
from femto.planning.planning_drafts import Journal
from test_refinement_workflow import refined_fixture


class QtRefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QApplication.instance() or QApplication([])
        cls.temp=tempfile.TemporaryDirectory();cls.base=refined_fixture(Path(cls.temp.name)/'source')

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def setUp(self):
        self.case=tempfile.TemporaryDirectory();self.addCleanup(self.case.cleanup);self.root=Path(self.case.name)
        model,self.region,self.original,self.refined=self.base
        self.model=clone_model(model,freeze_inputs=True);self.region=deepcopy(self.region)
        self.model.store.regions=[self.region]
        self.model.saved_plan=refinement_workflow.restore(self.model,self.region,self.refined)
        self.page=PlanningPage();self.page.resize(1120,740);self.page.show()
        self.page.install_model(self.model,self.root/'record.zip',Journal(self.root/'record.zip',self.root/'drafts'))
        self.page.set_stage(1);self.addCleanup(self.close)

    def close(self):self.page.shutdown();self.page.close();self.app.processEvents()

    def wait_task(self):
        loop=QEventLoop();timer=QTimer();timer.setSingleShot(True);timer.timeout.connect(loop.quit)
        self.page.runner.finished.connect(loop.quit)
        try:
            timer.start(30000)
            if self.page.busy:loop.exec()
        finally:self.page.runner.finished.disconnect(loop.quit);timer.stop()
        self.assertFalse(self.page.busy)

    def install_refined(self):
        model,_,_,plan=self.base
        self.page.model.coverage_certificate=model.coverage_certificate
        self.page.model.target_ledger=model.target_ledger
        self.page.model.coverage_display=model.coverage_display
        self.page.plan=deepcopy(plan);self.page.canvas.set_plan(self.page.plan);self.page.refresh()

    def delayed(self):
        entered=threading.Event();release=threading.Event()
        def job(model,region,plan,check,progress):
            entered.set()
            if not release.wait(5):raise RuntimeError('test release timed out')
            check()
            return deepcopy(self.refined)
        return entered,release,job

    def test_controls_require_coverage_and_follow_busy_and_refined_state(self):
        panel=self.page.coverage
        self.assertTrue(panel.refine.isEnabled());self.assertFalse(panel.restore.isVisible())
        panel.set_context(self.model,self.page.plan,ready=True,busy=True)
        self.assertFalse(panel.refine.isEnabled())
        no_coverage=deepcopy(self.page.plan);no_coverage.pop('coverage_evidence')
        panel.set_context(self.model,no_coverage,ready=True)
        self.assertFalse(panel.refine.isEnabled())
        self.install_refined()
        self.assertTrue(self.page.usable());self.assertTrue(panel.restore.isEnabled())
        self.assertFalse(panel.refine.isVisible());self.assertFalse(panel.verify.isVisible())

    def test_actual_background_refine_and_restore_preserve_original_route(self):
        original=deepcopy(self.page.plan)
        self.page.refine_coverage();self.wait_task()
        self.assertFalse(self.page._error,self.page.status_label.text())
        self.assertEqual(self.page.plan['surface_trajectory']['planner']['algorithm_version'],'coverage-refined-surface-v1')
        self.assertTrue(self.page.dirty);self.assertIsNone(self.page.archive);self.assertTrue(self.page.usable())
        self.assertFalse(self.page.reviewed.isChecked())
        self.page.restore_coverage_route();self.wait_task()
        self.assertEqual(self.page.plan['surface_trajectory'],original['surface_trajectory'])
        self.assertEqual(self.page.plan['coverage_evidence'],original['coverage_evidence'])

    def test_cancelled_refinement_does_not_install_or_replace_caches(self):
        before=self.page.plan;certificate=self.page.model.coverage_certificate
        entered,release,job=self.delayed()
        with patch('femto.planning.refinement_workflow.calculate',side_effect=job):
            self.page.refine_coverage();self.assertTrue(entered.wait(2))
            self.page.runner.cancel();release.set();self.wait_task()
        self.assertIs(self.page.plan,before);self.assertIs(self.page.model.coverage_certificate,certificate)

    def test_new_input_generation_rejects_late_result(self):
        before=self.page.plan
        entered,release,job=self.delayed()
        with patch('femto.planning.refinement_workflow.calculate',side_effect=job):
            self.page.refine_coverage();self.assertTrue(entered.wait(2))
            self.page._input_generation+=1;release.set();self.wait_task()
        self.assertIs(self.page.plan,before)

    def test_whole_plan_digest_rejects_changes_outside_calculation_revision(self):
        entered,release,job=self.delayed()
        with patch('femto.planning.refinement_workflow.calculate',side_effect=job):
            self.page.refine_coverage();self.assertTrue(entered.wait(2))
            self.page.plan=deepcopy(self.page.plan);self.page.plan['display_note']='new local edit'
            changed=self.page.plan;release.set();self.wait_task()
        self.assertIs(self.page.plan,changed)
        self.assertIn('旧补充结果未安装',self.page.status_label.text())

    def test_navigation_keeps_refinement_and_changed_width_disables_it(self):
        self.install_refined();before=deepcopy(self.page.plan)
        self.page.set_stage(0);self.page.set_stage(1)
        self.assertEqual(self.page.plan,before);self.assertTrue(self.page.usable())
        self.page.head.fields['effective_treatment_width_mm'].setText('10')
        self.assertFalse(self.page.usable());self.assertFalse(self.page.coverage.restore.isEnabled())
        self.assertEqual(self.page.plan,before)

    def test_history_is_readonly_and_coverage_layer_stays_synchronized(self):
        model,_,_,plan=self.base
        model=clone_model(model,freeze_inputs=True);model.saved_plan=deepcopy(plan)
        page=ReviewPage();page.resize(1120,740);page.show()
        try:
            page.install_model(model,self.root/'history.zip')
            self.assertFalse(page.coverage.refine.isVisible());self.assertFalse(page.coverage.restore.isVisible())
            page.coverage.show_layer.setChecked(True);self.assertIsNotNone(page.canvas._coverage_image)
            page.install_model(model,self.root/'history.zip')
            self.assertFalse(page.coverage.show_layer.isChecked());self.assertIsNone(page.canvas._coverage_image)
        finally:page.shutdown();page.close();self.app.processEvents()
