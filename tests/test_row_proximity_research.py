"""Sampling reports row proximity without treating connectors as treatment."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from femto.planning import row_proximity_research as module
from femto.planning.surface_trajectory import ScanSegment


def segment(index,block,row,y,role='scan'):
    points=[dict(xyz_m=[0.,y,0.],row_index=row,role='scan'),
            dict(xyz_m=[.004,y,0.],row_index=row,role=role)]
    return ScanSegment(index,row,points,block)


class RowProximityTests(unittest.TestCase):
    def test_only_different_rows_in_same_block_count(self):
        trajectory=SimpleNamespace(ordered_items=[segment(0,0,0,0.),segment(1,0,1,.0004),
            segment(2,1,2,0.),segment(3,0,3,0.,'connector')])
        result=module.measure(trajectory)
        self.assertAlmostEqual(result['sampled_scan_length_m'],.012)
        self.assertAlmostEqual(result['proximity'][0]['near_fraction'],2/3)
        self.assertEqual(result['proximity'][1]['near_fraction'],result['proximity'][0]['near_fraction'])
        self.assertIn('not_geodesic',result['semantics'])

    def test_same_row_empty_route_and_budget(self):
        trajectory=SimpleNamespace(ordered_items=[segment(0,0,0,0.),segment(1,0,0,.0001)])
        self.assertEqual(module.measure(trajectory)['proximity'][0]['near_fraction'],0.)
        empty=module.measure(SimpleNamespace(ordered_items=[]))
        self.assertIsNone(empty['proximity'][0]['near_fraction'])
        with patch.object(module,'MAX_SAMPLES',2):
            with self.assertRaisesRegex(ValueError,'采样超过预算'):module.measure(trajectory)
        with patch.object(module,'MAX_NEIGHBORS',1):
            with self.assertRaisesRegex(ValueError,'搜索超过预算'):module.measure(trajectory)

    def test_points_exactly_on_radius_keep_existing_boundary_semantics(self):
        trajectory=SimpleNamespace(ordered_items=[segment(0,0,0,0.),segment(1,0,1,.001)])
        result=module.measure(trajectory,radii_m=(.001,.002))
        self.assertEqual(result['proximity'][0]['near_fraction'],0.)
        self.assertEqual(result['proximity'][1]['near_fraction'],1.)

    def test_invalid_parameters_and_cancellation(self):
        trajectory=SimpleNamespace(ordered_items=[segment(0,0,0,0.)])
        for radii in ((.002,.001),(.001,.001),(True,),(.011,)):
            with self.subTest(radii=radii),self.assertRaises(ValueError):module.measure(trajectory,radii_m=radii)
        def stop():raise RuntimeError('cancelled')
        with self.assertRaisesRegex(RuntimeError,'cancelled'):module.measure(trajectory,check=stop)

if __name__=='__main__':unittest.main()
