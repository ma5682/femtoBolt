import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import unittest
from PySide6.QtWidgets import QApplication
from femto.app.qt_planning import HeadForm
from femto.planning.treatment_head import head_parameters
import numpy as np
from femto.app.qt_planning_canvas import PlanningCanvas
from femto.app.qt_common import rgb_image

class FixedWidthFormTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])

    def test_default_and_historical_modes(self):
        form=HeadForm()
        try:
            self.assertEqual(form.record()['mode'],'fixed_width_auto')
            self.assertIn('最大行距',form.spacing.text())
            self.assertFalse(form.fields['row_spacing_mm'].isEnabled())
            for mode in ('direct_row_spacing','from_effective_width','fixed_width_auto'):
                saved=head_parameters(mode,row_spacing_mm=10)
                form.set_record(saved)
                self.assertEqual(form.record(),saved)
        finally:form.close()

    def test_centerline_visible_over_coverage_only_for_auto_mode(self):
        canvas=PlanningCanvas();canvas.resize(400,300)
        canvas._rgb=rgb_image(np.zeros((30,40,3),np.uint8))
        canvas._runs=[('scan',[(5,15),(35,15)])]
        canvas.show_plan=True
        rgba=np.zeros((30,40,4),np.uint8);rgba[:]=[30,198,137,165]
        canvas.set_coverage_overlay(rgba)
        try:
            canvas.plan={'treatment_head':{'mode':'fixed_width_auto'}}
            auto=canvas.grab().toImage().pixelColor(200,150)
            self.assertGreater(auto.red(),200)
            canvas.plan={'treatment_head':{'mode':'direct_row_spacing'}}
            legacy=canvas.grab().toImage().pixelColor(200,150)
            self.assertLess(legacy.red(),100)
        finally:canvas.close()
