"""Exact residual maxima and unchanged footprint decisions, not coverage claims."""
import unittest
from unittest.mock import patch

import numpy as np

from femto.annotation.selection_intent import encode_runs
from femto.planning import quality_constrained as quality
from test_refinement_candidates_v2 import evidence


def direct_max(reference, outer):
    x0, y0, x1, y1 = outer
    values = reference.residual[max(0,y0):min(reference.height,y1+1),
                                max(0,x0):min(reference.width,x1+1)]
    return float(values.max())/1e6 if values.size else reference.global_residual


class ResidualTileTests(unittest.TestCase):
    def envelope(self, classes=None):
        rng = np.random.default_rng(9317)
        if classes is None: classes = np.ones((100,100), np.uint8)
        record = evidence(classes)
        residual = rng.integers(0, 10001, classes.shape, dtype=np.int32)
        residual[classes != 1] = 0
        record['residual_um_runs'] = encode_runs(residual)
        record['sha256'] = quality._hash(record)
        return quality.ReferenceEnvelope(record, 5.)

    def test_exact_maxima_for_random_and_off_image_rectangles(self):
        gate = self.envelope()
        rng = np.random.default_rng(311)
        for _ in range(3000):
            x0, y0, x1, y1 = rng.integers(-120, 220, 4).tolist()
            rect = (x0, y0, x1, y1)
            self.assertEqual(gate._residual_max(rect), direct_max(gate, rect), rect)

    def test_peak_on_each_tile_and_rectangle_boundary_is_never_lost(self):
        for y, x in ((0,0), (15,15), (16,16), (31,32), (32,31), (63,63), (64,64), (99,99)):
            gate = self.envelope()
            gate.residual[:] = 0
            gate.residual[y,x] = 9999
            gate.global_residual = .009999
            for rect in ((0,0,99,99), (1,1,98,98), (16,16,63,63), (15,15,64,64)):
                self.assertEqual(gate._residual_max(rect), direct_max(gate,rect), (y,x,rect))

    def test_public_probe_decisions_margins_and_details_equal_direct_reductions(self):
        classes = np.ones((100,100),np.uint8)
        classes[20:30,40:55] = 7
        classes[65:70,10:30] = 0
        gate = self.envelope(classes)
        rng = np.random.default_rng(222)
        for _ in range(400):
            mid = np.r_[rng.uniform(-.2,.2,2), rng.uniform(.3,.8)]
            half = rng.uniform(0,.04)
            width = float(rng.choice([.5,5.,20.,100.]))
            actual = gate.probe(mid,half,details=True,width_mm=width)
            with patch.object(quality.ReferenceEnvelope,'_residual_max',direct_max):
                expected = gate.probe(mid,half,details=True,width_mm=width)
            self.assertEqual(actual, expected)

    def test_work_reduction_keeps_budget_enforced(self):
        gate = self.envelope()
        for _ in range(1000): gate._residual_max((0,0,95,95))
        self.assertLess(gate.sampled_pixels, 96*96*1000//10)
        gate.sampled_pixels = 100000000
        with self.assertRaisesRegex(ValueError,'超过预算'):
            gate._residual_max((0,0,95,95))
        fresh = self.envelope()
        fresh.sampled_pixels = 100000000
        with self.assertRaisesRegex(ValueError,'超过预算'):
            fresh._residual_max((0,0,95,95))


if __name__ == '__main__': unittest.main()
