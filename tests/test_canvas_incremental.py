"""Incremental display must reproduce a full mask overlay after every edit."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from types import SimpleNamespace
import unittest
import numpy as np
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication
from femto.app.qt_planning_canvas import PlanningCanvas


def pixels(image):
    return np.ndarray((image.height(), image.width(), 4), np.uint8,
                      buffer=image.constBits(), strides=(image.bytesPerLine(), 4, 1)).copy()


class IncrementalCanvasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.canvas = PlanningCanvas()
        self.canvas.resize(803, 601)
        self.canvas.set_model(SimpleNamespace(rgb=np.zeros((91, 123, 3), np.uint8)))
        self.addCleanup(self.canvas.close)

    def check_layers(self):
        for image, mask, color in zip(self.canvas._layers,
                                     (self.canvas.work, self.canvas.forbidden),
                                     ((73,135,225,76), (239,71,111,115))):
            expected = np.zeros((*mask.shape, 4), np.uint8)
            expected[mask.astype(bool)] = color
            np.testing.assert_array_equal(pixels(image), expected)

    def test_fractional_sweeps_all_edges_and_erase(self):
        rng = np.random.default_rng(1104)
        for role in ('work', 'forbidden'):
            self.canvas.role = role
            for zoom in (.4, 1., 6.):
                self.canvas.zoom = zoom
                for erase in (False, True):
                    self.canvas.erase = erase
                    self.canvas._last = None
                    points = [(0.,0.), (122.,90.), (122.,0.), (0.,90.), (61.03125,45.96875)]
                    points += list(rng.uniform([0,0],[122,90],(15,2)).tolist())
                    for point in points:
                        self.canvas._paint(tuple(point))
                        self.check_layers()

    def test_shared_qimage_is_not_mutated(self):
        previous = QImage(self.canvas._layers[0])
        expected = pixels(previous)
        self.canvas._paint((50.5, 50.5))
        np.testing.assert_array_equal(pixels(previous), expected)
        self.check_layers()

    def test_undo_redo_clear_and_new_reference(self):
        self.canvas._before = (self.canvas.work.copy(), self.canvas.forbidden.copy())
        self.canvas._paint((30., 30.))
        self.canvas._paint((90., 70.))
        self.canvas.finish_stroke()
        painted = self.canvas.work.copy()
        self.canvas.undo(); self.check_layers()
        self.assertFalse(self.canvas.work.any())
        self.canvas.redo(); self.check_layers()
        np.testing.assert_array_equal(self.canvas.work, painted)
        self.canvas.clear_work(); self.check_layers()
        self.canvas.undo(); self.check_layers()
        np.testing.assert_array_equal(self.canvas.work, painted)
        self.canvas.set_model(SimpleNamespace(rgb=np.zeros((63, 79, 3), np.uint8)))
        self.canvas._paint((78., 62.)); self.check_layers()
        self.assertFalse(self.canvas._undo)

if __name__ == '__main__':
    unittest.main()
