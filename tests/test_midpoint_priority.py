from copy import deepcopy
import unittest
import numpy as np
from femto.planning import midpoint_priority as module


class MidpointPriorityTests(unittest.TestCase):
    def test_matches_exhaustive_and_preserves_order_and_inputs(self):
        rng=np.random.default_rng(88)
        points=rng.normal(size=(301,3));centers=rng.normal(size=(71,3));radius=.4
        records=[dict(role='scan',xyz_m=p.tolist(),source_face=i) for i,p in enumerate(points)]
        before=deepcopy(records);original_centers=centers.copy()
        got,report=module.select(records,centers,radius)
        expected=np.flatnonzero((np.linalg.norm(points[:,None,:]-centers[None,:,:],axis=2)<=radius).any(axis=1))
        np.testing.assert_array_equal(got,expected);np.testing.assert_array_equal(centers,original_centers)
        self.assertEqual(records,before);self.assertEqual(report['comparison_count'],301*71)
        self.assertEqual(report['semantics'],'float_priority_only_not_coverage_or_physical_distance')

    def test_boundary_empty_and_limits(self):
        records=[dict(role='scan',xyz_m=[0,0,0]),dict(role='scan',xyz_m=[1,0,0])]
        got,_=module.select(records,np.array([[0.,0.,0.]]),1.)
        np.testing.assert_array_equal(got,[0,1])
        got,_=module.select(records,np.empty((0,3)),1.)
        self.assertEqual(len(got),0)
        got,_=module.select([],np.array([[0.,0.,0.]]),1.)
        self.assertEqual(len(got),0)
        with self.assertRaisesRegex(ValueError,'预算'):
            module.select(records,np.array([[0.,0.,0.]]),1.,max_comparisons=1)
        for radius in (0.,float('nan'),float('inf'),True):
            with self.subTest(radius=radius),self.assertRaises(ValueError):module.select(records,np.zeros((1,3)),radius)
        with self.assertRaises(ValueError):module.select([dict(role='connector',xyz_m=[0,0,0])],np.zeros((1,3)),1.)
        with self.assertRaises(ValueError):module.select(records,np.array([[np.nan,0,0]]),1.)

    def test_cancellation_and_distance_overflow(self):
        records=[dict(role='scan',xyz_m=[0,0,0])]
        def cancel():raise RuntimeError('cancelled')
        with self.assertRaisesRegex(RuntimeError,'cancelled'):module.select(records,np.zeros((1,3)),1.,check=cancel)
        with self.assertRaises(ValueError):
            module.select([dict(role='scan',xyz_m=[1e308,0,0])],np.array([[-1e308,0,0]]),1.)
