"""Reason-specific reference locators never change proof, selection, or planning."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PySide6.QtWidgets import QApplication
from femto.app.qt_planning import PlanningPage
from femto.app.qt_review import ReviewPage
from femto.app.qt_planning_io import clone_model
from femto.planning import coverage_workflow as workflow
from test_coverage_workflow import fixture


class PixelReasonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QApplication.instance() or QApplication([])
        cls.temp=tempfile.TemporaryDirectory()
        cls.model,cls.region,original,certificate,ledger=fixture(Path(cls.temp.name)/'source')
        cls.plan=workflow.attach(cls.model,cls.region,original,certificate,ledger)
        cls.model.saved_plan=cls.plan

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def page(self, readonly=False):
        page=ReviewPage() if readonly else PlanningPage()
        page.install_model(clone_model(self.model),Path(self.temp.name)/'record.zip')
        if not readonly:page.set_stage(1)
        self.addCleanup(page.close);self.addCleanup(page.shutdown)
        return page

    def test_masks_are_exact_disjoint_and_do_not_invent_xyz(self):
        display=self.model.coverage_display
        combined=np.zeros(display.pixel_state.shape,bool)
        for code in range(2,11):
            mask=display.mask(8,pixel_reason=code)
            np.testing.assert_array_equal(mask,display.mask(8)&(display.reference_status==code))
            self.assertFalse(np.any(combined&mask));combined|=mask
            np.testing.assert_array_equal(display.rgba(8,pixel_reason=code)[...,3]>0,mask)
        np.testing.assert_array_equal(combined,display.mask(8))
        self.assertFalse(display.reference_status.flags.writeable)
        for code in (True,0,1,11,'5'):
            with self.assertRaises(ValueError):display.mask(8,pixel_reason=code)
        with self.assertRaises(ValueError):display.mask(1,pixel_reason=5)
        # Display-only classification fixture: observation and budget cannot mix.
        status=np.zeros(display.reference_status.shape,np.uint8)
        unknown=np.flatnonzero(display.mask(8))
        for i,code in enumerate((2,5,6,9,10)):status.flat[unknown[i]]=code
        sample=replace(display,reference_status=status)
        for i,code in enumerate((2,5,6,9,10)):
            self.assertEqual(int(sample.mask(8,pixel_reason=code).sum()),1)

    def test_selected_reason_focuses_only_its_reference_pixels_and_preserves_plan(self):
        page=self.page();panel=page.coverage
        panel.reason.setCurrentIndex(panel.reason.findData(8))
        self.assertGreater(panel.pixel_reason.count(),1)
        before=deepcopy(page.plan)
        panel.pixel_reason.setCurrentIndex(1)
        code=panel.reference_reason
        with patch.object(page.canvas,'fit_mask') as fit:
            panel.focus.click()
        np.testing.assert_array_equal(fit.call_args.args[0],panel.display.mask(8,pixel_reason=code))
        self.assertIn('当前定位',panel.task_summary.text())
        self.assertEqual(page.plan,before)
        panel.set_context(page.model,page.plan,ready=True,busy=True)
        self.assertFalse(panel.pixel_reason.isEnabled());self.assertFalse(panel.focus.isEnabled())

    def test_context_switch_clears_filter_and_history_supports_readonly_location(self):
        page=self.page(True);panel=page.coverage
        panel.reason.setCurrentIndex(panel.reason.findData(8));panel.pixel_reason.setCurrentIndex(1)
        before=deepcopy(page.plan)
        with patch.object(page.canvas,'fit_mask') as fit:panel.focus.click()
        np.testing.assert_array_equal(fit.call_args.args[0],panel.display.mask(8,pixel_reason=panel.reference_reason))
        self.assertFalse(panel.task_action.isVisible());self.assertEqual(page.plan,before)
        panel.set_context(None,None)
        self.assertIsNone(panel.reference_reason);self.assertFalse(panel.pixel_reason.isEnabled())
        self.assertEqual(panel.pixel_reason.currentData(),0)


if __name__=='__main__':unittest.main()
