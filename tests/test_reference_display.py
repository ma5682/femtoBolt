"""High-resolution display cannot alter selection coordinates or plan inputs."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
from PIL import Image
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication

from femto.app.qt_planning_canvas import PlanningCanvas
from femto.planning.planning_revision import array_record
from femto.planning.reference_display import install, save, load, ASSET


def fixture():
    rgb=np.arange(103*99*3,dtype=np.uint8).reshape(103,99,3)
    small=rgb[4::8,4::8].copy()
    model=SimpleNamespace(rgb=small,view_reference=dict(derivation=dict(
        schema='reduced-planning-reference-v1',stride=8,source_pixel_offset_xy=[4,4],
        original_image_size=[99,103],original_rgb=array_record(rgb))))
    return model,rgb


class ReferenceDisplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])

    def test_exact_pixels_immutable_and_computation_inputs_unchanged(self):
        model,rgb=fixture();view=deepcopy(model.view_reference);small=model.rgb.copy()
        install(model,rgb)
        np.testing.assert_array_equal(model.display_rgb,rgb)
        np.testing.assert_array_equal(model.rgb,small)
        self.assertEqual(model.view_reference,view)
        self.assertFalse(model.display_rgb.flags.writeable)
        rgb[:]=0;self.assertFalse(np.array_equal(model.display_rgb,rgb))

    def test_pixel_center_mapping_including_non_divisible_image_size(self):
        model,rgb=fixture();install(model,rgb)
        x,y,w,h=model.display_rect
        for u,v in [(0,0),(5,8),(model.rgb.shape[1]-1,model.rgb.shape[0]-1)]:
            self.assertAlmostEqual(x+(8*u+4+.5)/8,u+.5)
            self.assertAlmostEqual(y+(8*v+4+.5)/8,v+.5)
        self.assertEqual((w,h),(99/8,103/8))

    def test_display_and_roundtrip_asset_validation(self):
        model,rgb=fixture();install(model,rgb)
        with tempfile.TemporaryDirectory() as tmp:
            save(model,Path(tmp));other,_=fixture();load(other,Path(tmp))
            np.testing.assert_array_equal(other.display_rgb,rgb)
            wrong=rgb.copy();wrong[0,0]^=255;Image.fromarray(wrong).save(Path(tmp)/ASSET)
            with self.assertRaisesRegex(ValueError,'不一致'):load(other,Path(tmp))

    def test_missing_asset_preserves_old_model(self):
        model,_=fixture()
        with tempfile.TemporaryDirectory() as tmp:load(model,Path(tmp))
        self.assertFalse(hasattr(model,'display_rgb'))

    def test_forged_offset_and_reduced_pixels_rejected(self):
        for kind in ('offset','small','original'):
            model,rgb=fixture()
            if kind=='offset':model.view_reference['derivation']['source_pixel_offset_xy']=[0,0]
            elif kind=='small':model.rgb[0,0]^=255
            else:rgb[0,0]^=255
            with self.subTest(kind=kind),self.assertRaises(ValueError):install(model,rgb)

    def test_mouse_brush_zoom_pan_and_fit_identical_with_hd(self):
        model,rgb=fixture();other=deepcopy(model);install(other,rgb)
        a,b=PlanningCanvas(),PlanningCanvas()
        try:
            for canvas,m in ((a,model),(b,other)):
                canvas.resize(960,600);canvas.set_model(m)
                canvas.zoom=1.7;canvas.pan=QPointF(23,-18)
                pixel=canvas.image_point(canvas.image_rect().center())
                canvas.brush_size=60;canvas._paint(pixel)
                canvas.fit_selection()
            self.assertEqual(a.image_rect(),b.image_rect())
            self.assertEqual(a.image_point(QPointF(450,310)),b.image_point(QPointF(450,310)))
            np.testing.assert_array_equal(a.work,b.work)
            self.assertEqual(b._display_rgb.width(),99)
            b.set_model(model);self.assertTrue(b._display_rgb.isNull())
        finally:a.close();b.close()


if __name__=='__main__':unittest.main()
