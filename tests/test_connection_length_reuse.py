"""Arithmetic reuse never bypasses anchored-link validation or mutations."""
from copy import deepcopy
import unittest
from unittest.mock import patch
import numpy as np
from femto.planning import scan_visit_order as order,verified_relocation_research as relocation
from femto.planning.surface_trajectory import ScanSegment

class ConnectionLengthReuseTests(unittest.TestCase):
    def test_original_float_arithmetic_at_different_scales(self):
        rng=np.random.default_rng(12345);cache=relocation._ConnectionLengths()
        for scale in (1e-10,.001,1.,1e8,1e150):
            for count in (1,2,7):
                values=(rng.normal(size=(count+1,3))*scale).tolist()
                tail={'xyz_m':values[0]};samples=[{'xyz_m':p} for p in values[1:]]
                expected=order._connector_length(tail,samples)
                with self.subTest(scale=scale,count=count):
                    self.assertEqual(cache.measure(tail,samples),expected)
                    self.assertEqual(cache.measure(deepcopy(tail),deepcopy(samples)),expected)

    def test_identical_coordinates_reuse_but_mutations_recompute(self):
        cache=relocation._ConnectionLengths();tail={'xyz_m':[0.,0.,0.]}
        samples=[{'xyz_m':[.003,.004,0.]},{'xyz_m':[.006,.008,0.]}]
        original=order._connector_length
        with patch.object(order,'_connector_length',wraps=original) as measured:
            self.assertEqual(cache.measure(tail,samples),original(tail,samples))
            self.assertEqual(cache.measure(deepcopy(tail),deepcopy(samples)),original(tail,samples))
            self.assertEqual(measured.call_count,1)
            samples[0]['xyz_m'][2]=.001
            self.assertEqual(cache.measure(tail,samples),original(tail,samples))
            tail['xyz_m'][0]=.002
            self.assertEqual(cache.measure(tail,samples),original(tail,samples))
            self.assertEqual(measured.call_count,3)

    def test_other_numeric_types_and_cache_budget_use_original_path(self):
        cache=relocation._ConnectionLengths();tail={'xyz_m':[0,0,0]};samples=[{'xyz_m':[3.,4.,0.]}]
        self.assertEqual(cache.measure(tail,samples),order._connector_length(tail,samples));self.assertFalse(cache.values)
        tail={'xyz_m':np.array([0.,0.,0.])}
        self.assertEqual(cache.measure(tail,samples),order._connector_length(tail,samples));self.assertFalse(cache.values)
        with patch.object(relocation,'MAX_CACHE_POINTS',0):
            tail={'xyz_m':[0.,0.,0.]}
            self.assertEqual(cache.measure(tail,samples),order._connector_length(tail,samples));self.assertFalse(cache.values)

    def test_anchored_metadata_and_point_budget_are_checked_after_cache_hit(self):
        def point(x,row,face=0,role='scan'):
            return dict(xyz_m=[x,0.,.5],row_index=row,triangle_index=face,role=role)
        segments=[ScanSegment(0,0,[point(0.,0),point(.001,0)],0),
                  ScanSegment(1,1,[point(.004,1),point(.005,1)],0)]
        visits=[(0,False),(1,False)]
        links=[dict(from_segment=0,to_segment=1,connected=True,added_edges=2,
                    samples=[point(.002,1,role='connector'),point(.004,1,role='connector')],reason='verified_surface_path')]
        cache=relocation._ConnectionLengths()
        expected=order.anchored_objective(segments,visits,links,budget=20)
        self.assertEqual(order.anchored_objective(segments,visits,links,budget=20,_lengths=cache),expected)
        self.assertEqual(order.anchored_objective(segments,visits,links,budget=20,_lengths=cache),expected)
        def cancelled():raise RuntimeError('cancelled')
        with self.assertRaisesRegex(RuntimeError,'cancelled'):
            order.anchored_objective(segments,visits,links,cancelled,budget=20,_lengths=cache)
        with self.assertRaisesRegex(ValueError,'预算'):
            order.anchored_objective(segments,visits,links,budget=2,_lengths=cache)
        for key,value in (('role','scan'),('row_index',0),('xyz_m',[.006,0.,.5])):
            damaged=deepcopy(links);damaged[0]['samples'][-1][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                order.anchored_objective(segments,visits,damaged,budget=20,_lengths=cache)

if __name__=='__main__':unittest.main()
