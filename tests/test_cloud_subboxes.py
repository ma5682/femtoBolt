"""Subcell negative certificates preserve old strict point predicates."""
import unittest
from unittest.mock import patch
import numpy as np
from femto.geometry import cloud_layer_batches as batches


class SubboxTests(unittest.TestCase):
    def test_certificates_at_thresholds_cell_boundaries_and_large_origins(self):
        rng=np.random.default_rng(802)
        for origin in (0., -.010001, .010001, 1e6, -1e6):
            for h in (np.nextafter(.004,0),.004,np.nextafter(.004,np.inf)):
                points=np.r_[rng.uniform(-.02,.02,(180,3)),[[0,0,0],[.01,0,0],[0,0,h]]]+origin
                normals=rng.normal(size=points.shape)
                for i in np.flatnonzero(batches.negative_bounds(points,normals)):
                    delta=points-points[i]
                    near=np.linalg.norm(delta,axis=1)<.01
                    self.assertFalse(np.any(np.abs(delta[near]@normals[i])>.004))

    def test_memory_failure_falls_back_and_cancellation_propagates(self):
        points=np.array([[0.,0.,0.],[.001,.001,.001]])
        normals=np.ones_like(points)
        expected=batches._original_negative_bounds(points,normals)
        with patch.object(batches,'_subbox_negative_bounds',side_effect=MemoryError):
            np.testing.assert_array_equal(batches.negative_bounds(points,normals),expected)
        def cancel():raise RuntimeError('cancelled')
        with self.assertRaisesRegex(RuntimeError,'cancelled'):
            batches.negative_bounds(points,normals,check=cancel)
        with np.errstate(under='raise'),patch.object(batches,'_subbox_negative_bounds',side_effect=AssertionError):
            np.testing.assert_array_equal(batches.negative_bounds(points,normals),
                                          batches._original_negative_bounds(points,normals))
