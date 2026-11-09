"""Operator actions and internal semantic navigation stay independent."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from unittest import TestCase
import numpy as np
from PySide6.QtWidgets import QApplication,QPushButton
from femto.app.qt_region_capture import CaptureRegionEditor
from femto.app.qt_workflow import WorkflowProgress
from femto.app.qt_planning import HeadForm
from types import SimpleNamespace


class SimpleLocalWorkflowTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QApplication.instance() or QApplication([])

    def test_clear_is_undoable_and_failure_details_preserve_selection(self):
        editor=CaptureRegionEditor();self.addCleanup(editor.close)
        reference=SimpleNamespace(binding={'source':'test'},rgb=np.zeros((80,100,3),np.uint8),frame_count=1)
        editor.install(reference);editor.show();self.app.processEvents()
        self.assertFalse(editor.margin.isVisible())
        self.assertFalse(editor.advanced_toggle.isVisible())
        editor.canvas.work[20:40,30:50]=1
        before=editor.canvas.work.copy()
        next(b for b in editor.findChildren(QPushButton) if b.text()=='清空').click()
        self.assertFalse(editor.canvas.work.any())
        next(b for b in editor.findChildren(QPushButton) if b.text()=='撤销').click()
        np.testing.assert_array_equal(editor.canvas.work,before)
        editor.install(reference,error='离线失败')
        np.testing.assert_array_equal(editor.canvas.work,before)
        self.assertEqual(editor.margin.value(),0)
        self.assertFalse(editor.margin.isVisible())
        self.assertTrue(editor.error_details.isVisible())
        self.assertEqual(editor.advanced_toggle.text(),'失败详情')
        editor.advanced_toggle.click();self.assertFalse(editor.error_details.isVisible())
        editor.install(reference)
        self.assertFalse(editor.advanced_toggle.isVisible())
        self.assertFalse(editor.error_details.isVisible())
        np.testing.assert_array_equal(editor.canvas.work,before)

    def test_four_stages_do_not_bypass_available_destinations(self):
        progress=WorkflowProgress();self.addCleanup(progress.close)
        progress.show();progress.set_step(6);progress.set_navigation({1})
        self.assertEqual(sum(b.isVisible() for b in progress.stage_buttons),4)
        self.assertFalse(any(b.isVisible() for b in progress.buttons))
        self.assertFalse(progress.stage_buttons[2].isEnabled())
        progress.set_step(5);progress.set_navigation({1,2,3,4,5})
        requests=[];progress.step_requested.connect(requests.append)
        progress.stage_buttons[2].click();self.assertEqual(requests,[4])
        progress.set_step(3);self.assertEqual(progress.stage_buttons[2].property('state'),'current')

    def test_device_width_visible_and_algorithm_options_collapsed(self):
        form=HeadForm();self.addCleanup(form.close)
        form.show();self.app.processEvents()
        original=form.record()
        self.assertTrue(form.fields['effective_treatment_width_mm'].isVisible())
        self.assertFalse(form.mode.isVisible());self.assertFalse(form.layout_choice.isVisible())
        form.extra_button.click()
        self.assertTrue(form.mode.isVisible());self.assertTrue(form.layout_choice.isVisible())
        form.extra_button.click();self.assertEqual(form.record(),original)

    def test_local_default_is_explicit_and_keeps_legacy_registration(self):
        from femto.planning import planning_revision as revision
        model=SimpleNamespace(store=SimpleNamespace(local_context={'reconstruction_scope':{'schema':'test'}}))
        head={'mode':'fixed_width_auto','effective_treatment_width_mm':10.}
        small={'triangle_indices':list(range(4999))}
        large={'triangle_indices':list(range(5000))}
        self.assertEqual(revision.default_algorithms(model,small,head),revision.LOCAL_DENSE_LAYOUT_ALGORITHMS)
        algorithms=revision.default_algorithms(model,large,head)
        self.assertEqual(algorithms,revision.LOCAL_DENSE_LAYOUT_ALGORITHMS)
        self.assertTrue(revision.supported_target_algorithms(algorithms))
        self.assertTrue(revision.supported_target_algorithms(revision.ORDERED_LAYOUT_ALGORITHMS))
        self.assertEqual(algorithms['layout'],revision.ORDERED_LAYOUT_ALGORITHMS['layout'])
        self.assertEqual(algorithms['route'],revision.ORDERED_LAYOUT_ALGORITHMS['route'])
        model.store.local_context={}
        self.assertEqual(revision.default_algorithms(model,large,head),revision.ORDERED_LAYOUT_ALGORITHMS)
