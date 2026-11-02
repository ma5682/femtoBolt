"""Product coverage actions, revision guards and history layer semantics."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import json

from copy import copy, deepcopy
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np
from PySide6.QtCore import QEventLoop, QTimer, Signal
from PySide6.QtWidgets import QApplication, QWidget

from femto.app.qt_planning import PlanningPage
from femto.app.qt_review import ReviewPage
from femto.app.qt_planning_io import clone_model
from femto.planning import coverage_workflow as workflow, coverage_certificate as coverage
from femto.planning.planning_drafts import Journal
from test_coverage_workflow import fixture


class CoverageViewport(QWidget):
    picked=Signal(dict)
    def __init__(self,parent=None):
        super().__init__(parent);self.layers=[]
    def set_scene(self,*args,**kwargs):pass
    def set_coverage(self,display,reason=0):self.layers.append((display,reason))
    def focus_coverage(self,reason=0):return reason!=8
    def reset_camera(self):pass
    def shutdown(self):pass


class QtCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QApplication.instance() or QApplication([])
        cls.temp=tempfile.TemporaryDirectory()
        cls.base=fixture(Path(cls.temp.name)/'source')

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def setUp(self):
        self.case=tempfile.TemporaryDirectory();self.addCleanup(self.case.cleanup)
        self.root=Path(self.case.name)
        base,region,original,self.certificate,self.ledger=self.base
        self.model=clone_model(base,freeze_inputs=True)
        self.region=deepcopy(region);self.model.store.regions=[self.region]
        self.plan=workflow.attach(self.model,self.region,original,self.certificate,self.ledger)
        self.model.saved_plan=self.plan
        self.page=PlanningPage();self.page.resize(1120,740);self.page.show()
        self.page.install_model(self.model,self.root/'record.zip',Journal(self.root/'record.zip',self.root/'drafts'))
        self.page.set_stage(1)
        self.addCleanup(self.close_page)

    def close_page(self):
        self.page.shutdown();self.page.close();self.app.processEvents()

    def wait_task(self):
        loop=QEventLoop();timer=QTimer();timer.setSingleShot(True);timer.timeout.connect(loop.quit)
        self.page.runner.finished.connect(loop.quit)
        try:
            timer.start(30000)
            if self.page.busy:loop.exec()
        finally:self.page.runner.finished.disconnect(loop.quit);timer.stop()
        self.assertFalse(self.page.busy)

    def test_three_summaries_and_single_layer_preserve_plan_when_navigating(self):
        panel=self.page.coverage
        self.assertIn('路径：',panel.labels[0].text())
        self.assertIn('原目标几何：',panel.labels[1].text())
        self.assertIn('模型覆盖：',panel.labels[2].text())
        before=deepcopy(self.page.plan)
        panel.show_layer.setChecked(True)
        self.assertIsNotNone(self.page.canvas._coverage_image)
        self.assertFalse(self.page.play_button.isEnabled())
        self.page.set_stage(0)
        self.assertIsNone(self.page.canvas._coverage_image)
        self.page.set_stage(1)
        self.assertIsNotNone(self.page.canvas._coverage_image)
        panel.show_layer.setChecked(False)
        self.assertTrue(self.page.play_button.isEnabled())
        self.assertEqual(self.page.plan,before)

    def test_next_step_guidance_tracks_reason_and_dispatches_existing_operations(self):
        panel=self.page.coverage
        self.assertTrue(panel.task_summary.isVisible())
        self.assertFalse(panel.task_action.isEnabled())
        self.assertIn('类待处理原因',panel.task_summary.text())
        pending=next(task for task in panel.display.tasks.items if task.present and task.reason in (3,4,5,7))
        panel.reason.setCurrentIndex(panel.reason.findData(pending.reason))
        self.assertTrue(panel.task_action.isEnabled())
        self.assertIn('mm²',panel.task_summary.text())
        name='refine_coverage' if pending.reason==3 else 'verify_coverage'
        before=deepcopy(self.page.plan)
        with patch.object(self.page,name) as execute:
            panel.task_action.click();execute.assert_called_once_with()
        self.assertEqual(self.page.plan,before)
        panel.reason.setCurrentIndex(panel.reason.findData(1))
        self.assertFalse(panel.task_action.isEnabled())

    def test_unlocated_task_focuses_reference_without_changing_plan_or_inventing_area(self):
        panel=self.page.coverage
        self.assertGreater(panel.display.tasks.task(8).pixel_count,0)
        panel.reason.setCurrentIndex(panel.reason.findData(8))
        self.assertIn('面积未知',panel.task_summary.text())

        before=deepcopy(self.page.plan)
        with patch('femto.app.qt_gap_inspection.GapInspectionDialog.exec',return_value=0),patch.object(self.page,'verify_coverage') as recalculate,patch.object(self.page,'refine_coverage') as refine:
            panel.task_action.click();recalculate.assert_not_called();refine.assert_not_called()
        self.assertTrue(panel.show_layer.isChecked())
        self.assertIn('不能补造三维坐标或面积',self.page.status_label.text())
        self.assertEqual(self.page.plan,before)

    def test_task_actions_and_exports_are_disabled_for_busy_or_changed_inputs(self):
        panel=self.page.coverage;panel.reason.setCurrentIndex(panel.reason.findData(8))
        panel.set_context(self.model,self.plan,ready=True,busy=True)
        self.assertFalse(panel.task_action.isEnabled());self.assertFalse(panel.tasks_export.isEnabled())
        self.page.refresh()
        self.assertTrue(panel.task_action.isEnabled())
        self.page.head.fields['effective_treatment_width_mm'].setText('10')
        self.assertFalse(panel.task_action.isEnabled());self.assertFalse(panel.tasks_export.isEnabled())
        before=self.page.plan
        with patch.object(self.page,'verify_coverage') as calculate:
            self.page._coverage_task(7);calculate.assert_not_called()
        self.assertIs(self.page.plan,before)

    def test_tasks_export_runs_in_background_and_preserves_route_review_state(self):
        target=self.root/'tasks.json';before=deepcopy(self.page.plan);dirty=self.page.dirty
        from femto.planning import coverage_tasks
        original=coverage_tasks.write;gui_thread=threading.get_ident()
        def write(*args,**kwargs):
            self.assertNotEqual(threading.get_ident(),gui_thread)
            return original(*args,**kwargs)
        with patch('femto.app.qt_planning.QFileDialog.getSaveFileName',return_value=(str(target),'JSON (*.json)')),\
             patch('femto.planning.coverage_tasks.write',side_effect=write):
            self.page.coverage.tasks_export.click();self.wait_task()
        self.assertEqual(json.loads(target.read_bytes()),self.page.coverage.display.tasks.to_dict())
        self.assertEqual(self.page.plan,before);self.assertEqual(self.page.dirty,dirty)
        self.assertIn('已导出',self.page.status_label.text())

    def test_gap_inspection_dispatches_diagnostic_without_replacing_route(self):
        before=deepcopy(self.page.plan)
        with patch('femto.app.qt_gap_inspection.GapInspectionDialog') as dialog,patch.object(self.page,'_audit_reconstruction_gap') as audit:
            dialog.return_value.exec.return_value=1
            dialog.return_value.action='audit'
            self.page._coverage_task(8)
            audit.assert_called_once()
            np.testing.assert_array_equal(audit.call_args.args[0],self.page.coverage.display.mask(8))
        self.assertEqual(self.page.plan,before)

    def test_gap_inspection_rescan_uses_existing_navigation_and_preserves_evidence(self):
        before=deepcopy(self.page.plan)
        with patch('femto.app.qt_gap_inspection.GapInspectionDialog') as dialog,patch.object(self.page,'_prepare_gap_rescan') as prepare:
            dialog.return_value.exec.return_value=1;dialog.return_value.action='rescan'
            self.page._coverage_task(8)
            prepare.assert_called_once()
        self.assertEqual(self.page.plan,before)

    def test_exact_partition_action_follows_numeric_filter_and_uses_background_worker(self):
        panel=self.page.coverage;panel.reason.setCurrentIndex(panel.reason.findData(8))
        panel.pixel_reason.setCurrentIndex(panel.pixel_reason.findData(10))
        with patch('femto.app.qt_gap_inspection.GapInspectionDialog') as dialog,patch.object(self.page,'_continue_original_target') as execute:
            dialog.return_value.exec.return_value=1;dialog.return_value.action='exact_target'
            self.page._coverage_task(8)
            self.assertTrue(dialog.call_args.kwargs['allow_exact_partition'])
            self.assertFalse(dialog.call_args.kwargs['allow_acquisition'])
            execute.assert_called_once_with(exact=True)
        from femto.planning import target_reconstruction
        original=target_reconstruction.calculate;gui_thread=threading.get_ident();before=deepcopy(self.page.plan)
        def calculate(*args,**kwargs):
            self.assertNotEqual(threading.get_ident(),gui_thread)
            return original(*args,**kwargs)
        with patch('femto.planning.target_reconstruction.calculate',side_effect=calculate),\
             patch('femto.app.qt_planning.OUTPUTS_ROOT',self.root/'outputs'):
            self.page._continue_original_target(exact=True);self.wait_task()
        self.assertFalse(self.page._error)
        self.assertEqual(self.page.plan['segments'],before['segments'])
        self.assertNotIn(workflow.KEY,self.page.plan)
        self.assertIsNone(self.model.coverage_display)
        self.assertTrue(self.page.dirty);self.assertFalse(self.page.reviewed.isChecked())
        self.assertLess(self.model.target_ledger.summary['unlocated_pixel_count'],self.ledger.summary['unlocated_pixel_count'])

    def test_real_background_target_continuation_clears_only_old_coverage_and_keeps_route(self):
        from dataclasses import replace
        from femto.planning import target_surface,target_paging,planning_state
        from femto.planning.surface_trajectory import SurfaceTrajectory
        revision=self.plan['planning_context']['revision'];params=self.plan['surface_trajectory']['planner']['parameters']
        arrays,metrics=target_surface._construct(
            {key:self.ledger.arrays[key] for key in target_paging.SOURCE_NAMES},revision,
            target_surface.TargetBudget(max_units=50),lambda:None)
        ledger=target_surface._seal(arrays,revision,target_surface.TargetBudget(max_units=50),metrics,
            params['quality_evidence']['sha256'],params['effective_domain']['sha256'])
        plan=deepcopy(self.plan);plan.pop(workflow.KEY,None)
        plan['planning_context']['target']=ledger.metadata
        trajectory=replace(SurfaceTrajectory.from_dict(plan['surface_trajectory']),planning_context=plan['planning_context'])
        plan['surface_trajectory']=trajectory.to_dict();plan['plan_binding']=planning_state.binding(self.model.mesh,self.region,plan)
        self.model.target_ledger=ledger;self.model.coverage_certificate=None;self.model.coverage_display=None
        certificate,ledger=workflow.calculate(self.model,self.region,plan,budget=coverage.CoverageBudget(max_depth=1))
        self.page.plan=workflow.attach(self.model,self.region,plan,certificate,ledger)
        self.page.canvas.set_plan(self.page.plan);self.page.refresh()
        before=deepcopy(self.page.plan)
        with patch('femto.app.qt_planning.OUTPUTS_ROOT',self.root/'continuations'):
            self.page._continue_original_target();self.wait_task()
        self.assertEqual(self.page.plan['planning_context']['target']['schema'],target_surface.PAGED_SCHEMA)
        self.assertEqual(self.page.plan['surface_trajectory']['ordered_items'],before['surface_trajectory']['ordered_items'])
        self.assertIsNone(self.model.coverage_certificate);self.assertIsNone(self.model.coverage_display)
        self.assertNotIn(workflow.KEY,self.page.plan);self.assertTrue(self.page.dirty)
        self.assertEqual(self.model.target_ledger.summary['unknown_by_reason']['budget_unprocessed'],0)
        self.assertTrue(list((self.root/'continuations').glob('*/previous/coverage_certificate.npz')))

    def test_readonly_history_shows_tasks_without_executable_task_controls(self):
        review=ReviewPage();review.show();self.addCleanup(review.close);self.addCleanup(review.shutdown)
        review.install_model(self.model,self.root/'history.zip')
        panel=review.coverage
        self.assertTrue(panel.task_summary.isVisible())
        self.assertFalse(panel.task_action.isVisible());self.assertFalse(panel.tasks_export.isVisible())
        panel.reason.setCurrentIndex(panel.reason.findData(8))
        self.assertIn('面积未知',panel.task_summary.text())
        self.assertFalse(panel.task_action.isEnabled());self.assertFalse(panel.tasks_export.isEnabled())

    def test_parameter_change_disables_evidence_until_original_inputs_return(self):
        self.page.coverage.show_layer.setChecked(True)
        old=self.page.plan
        self.page.head.fields['effective_treatment_width_mm'].setText('10')
        self.assertFalse(self.page.usable())
        self.assertIsNone(self.page.canvas._coverage_image)
        self.assertFalse(self.page.coverage.show_layer.isEnabled())
        self.assertIs(self.page.plan,old)
        self.page.head.fields['effective_treatment_width_mm'].setText('5')
        self.assertTrue(self.page.usable())
        self.assertIsNotNone(self.page.canvas._coverage_image)

    def test_new_record_clears_old_coverage_overlay(self):
        self.page.coverage.show_layer.setChecked(True)
        other=clone_model(self.model);other.saved_plan=deepcopy(self.base[2])
        self.page.install_model(other,self.root/'other.zip',Journal(self.root/'other.zip',self.root/'drafts'))
        self.page.set_stage(1)
        self.assertIsNone(self.page.canvas._coverage_image)
        self.assertIsNone(self.page.coverage.display)
        self.assertIn('未进行此项核验',self.page.coverage.labels[2].text())

    def test_real_background_calculation_installs_evidence_without_replanning(self):
        self.page.plan=deepcopy(self.base[2]);self.page.canvas.set_plan(self.page.plan);self.page.refresh()
        before=deepcopy(self.page.plan['segments'])
        original=workflow.calculate
        def calculate(*args,**kwargs):
            self.assertNotEqual(threading.get_ident(),self.gui_thread)
            from dataclasses import replace
            budget=replace(kwargs.pop('budget'),max_depth=1)
            return original(*args,**kwargs,budget=budget)
        self.gui_thread=threading.get_ident()
        with patch('femto.planning.coverage_workflow.calculate',side_effect=calculate),\
             patch('femto.planning.surface_workspace.generate',side_effect=AssertionError('replanned')):
            self.page.verify_coverage();self.wait_task()
        self.assertEqual(self.page.plan['segments'],before)
        self.assertIn(workflow.KEY,self.page.plan)
        self.assertTrue(self.page.dirty)
        self.assertIsNone(self.page.archive)
        self.assertTrue(self.page.coverage.show_layer.isChecked())
        self.assertEqual(self.page.plan[workflow.KEY]['inputs']['budget']['seconds'],30)
        self.assertFalse(self.page._error)

    def test_cancel_keeps_previous_certificate_and_display(self):
        self.page.coverage.show_layer.setChecked(True)
        previous=self.page.plan;release=threading.Event()
        def delayed(*args,**kwargs):
            release.wait(5)
            return self.certificate,self.ledger
        with patch('femto.planning.coverage_workflow.calculate',side_effect=delayed):
            self.page.verify_coverage();self.page.runner.cancel();release.set();self.wait_task()
        self.assertIs(self.page.plan,previous)
        self.assertIsNotNone(self.page.canvas._coverage_image)
        self.assertIn('取消',self.page.status_label.text())

    def test_completed_result_cannot_replace_a_new_model(self):
        release=threading.Event()
        def delayed(*args,**kwargs):
            release.wait(5)
            return self.certificate,self.ledger
        with patch('femto.planning.coverage_workflow.calculate',side_effect=delayed):
            self.page.verify_coverage()
            other=clone_model(self.model);other.saved_plan=deepcopy(self.base[2])
            self.page.install_model(other,self.root/'other.zip',Journal(self.root/'other.zip',self.root/'drafts'))
            release.set();self.wait_task()
        self.assertIs(self.page.model,other)
        self.assertNotIn(workflow.KEY,self.page.plan)
        self.assertIn('旧任务结果未安装',self.page.status_label.text())

    def test_same_input_revision_but_replaced_route_discards_pending_result(self):
        release=threading.Event()
        def delayed(*args,**kwargs):
            release.wait(5)
            return self.certificate,self.ledger
        with patch('femto.planning.coverage_workflow.calculate',side_effect=delayed):
            self.page.verify_coverage();replacement=deepcopy(self.base[2]);self.page.plan=replacement
            release.set();self.wait_task()
        self.assertIs(self.page.plan,replacement)
        self.assertIn('旧覆盖结果未安装',self.page.status_label.text())

    def test_readonly_history_shares_evidence_but_never_offers_recomputation(self):
        review=ReviewPage();review.show();self.addCleanup(review.close);self.addCleanup(review.shutdown)
        review.install_model(self.model,self.root/'history.zip')
        self.assertFalse(review.coverage.verify.isVisible())
        self.assertFalse(review.canvas.editable)
        review.coverage.show_layer.setChecked(True)
        self.assertIsNotNone(review.canvas._coverage_image)
        self.assertFalse(review.play.isEnabled())
        review.quality_toggle.setChecked(True)
        self.assertIsNone(review.canvas._coverage_image)
        self.assertFalse(review.coverage.show_layer.isChecked())
        self.assertTrue(review.canvas.support)

    def test_three_dimensional_reason_and_unknown_pixel_focus(self):
        panel=self.page.scene_panel
        with patch('femto.app.qt_scene_panel.create_viewport',side_effect=CoverageViewport):
            panel.mode.setCurrentIndex(1)
            self.page.coverage.show_layer.setChecked(True)
            display,reason=panel.viewport.layers[-1]
            self.assertIs(display,self.model.coverage_display)
            self.page.coverage.reason.setCurrentIndex(8)
            self.page.coverage.focus.click()
            self.assertFalse(panel.is_3d)
            self.assertEqual(self.page.coverage.reason.currentData(),8)
            self.assertIn('不补造三维位置',self.page.status_label.text())
        self.assertEqual(self.page.plan['coverage_evidence'],self.certificate.metadata)

    def test_reinstalling_same_history_model_keeps_layer_controls_and_views_consistent(self):
        review=ReviewPage();review.show();self.addCleanup(review.close);self.addCleanup(review.shutdown)
        source=self.root/'history.zip'
        with patch('femto.app.qt_scene_panel.create_viewport',side_effect=CoverageViewport):
            review.install_model(self.model,source)
            review.scene_panel.mode.setCurrentIndex(1)
            review.coverage.show_layer.setChecked(True)
            self.assertIsNotNone(review.canvas._coverage_image)
            self.assertIs(review.scene_panel.viewport.layers[-1][0],self.model.coverage_display)
            review.install_model(self.model,source)
            self.assertFalse(review.coverage.show_layer.isChecked())
            self.assertIsNone(review.canvas._coverage_image)
            self.assertIsNone(review.scene_panel.viewport.layers[-1][0])
            review.coverage.show_layer.setChecked(True)
            self.assertIsNotNone(review.canvas._coverage_image)
            self.assertIs(review.scene_panel.viewport.layers[-1][0],self.model.coverage_display)
            review.layers.setCurrentIndex(0)
            self.assertIsNone(review.canvas._coverage_image)
            review.layers.setCurrentIndex(2)
            self.assertIsNotNone(review.canvas._coverage_image)
            self.assertIs(review.scene_panel.viewport.layers[-1][0],self.model.coverage_display)
        self.assertEqual(review.plan,self.plan)

    def test_vtk_cell_colors_and_filtering_use_leaf_geometry_without_replacing_mesh(self):
        from femto.app.qt_vtk_viewport import VTKViewport
        from vtkmodules.vtkRenderingCore import vtkRenderer,vtkActor
        class Harness:
            _actor=VTKViewport._actor
            _check_thread=lambda self:None
            _render=lambda self:None
        owner=Harness();owner.renderer=vtkRenderer();owner._actors=[]
        owner._coverage_actor=None;owner._region_actor=vtkActor();owner._route_actor=vtkActor()
        display=self.model.coverage_display
        VTKViewport.set_coverage(owner,display,1)
        data=owner._coverage_actor.GetMapper().GetInput()
        count=int(np.count_nonzero(display.status==1))
        self.assertEqual(data.GetNumberOfCells(),count)
        self.assertEqual(data.GetCellData().GetScalars().GetNumberOfTuples(),count)
        self.assertFalse(owner._region_actor.GetVisibility())
        self.assertFalse(owner._route_actor.GetVisibility())
        VTKViewport.set_coverage(owner,None)
        self.assertIsNone(owner._coverage_actor)
        self.assertTrue(owner._region_actor.GetVisibility())
        self.assertTrue(owner._route_actor.GetVisibility())
