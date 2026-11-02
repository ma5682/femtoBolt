from types import SimpleNamespace
import unittest
from unittest.mock import patch

from femto.planning import row_overlap_diagnostics as diagnostics,row_proximity_research as reference
from femto.planning.surface_trajectory import ScanSegment


def segment(index,block,row,y,role='scan'):
    return ScanSegment(index,row,[dict(xyz_m=[0.,y,0.],row_index=row,role='scan'),
                                  dict(xyz_m=[.004,y,0.],row_index=row,role=role)],block)


class RowOverlapDiagnosticsTests(unittest.TestCase):
    def test_near_length_matches_reference_and_attributes_only_once(self):
        trajectory=SimpleNamespace(ordered_items=[segment(0,0,0,0.),segment(1,0,1,.0004),
            segment(2,0,2,.0008),segment(3,1,3,0.),segment(4,0,4,0.,'connector')])
        report=diagnostics.inspect(trajectory,supplement_rows=[2],include_edges=True)
        expected=reference.measure(trajectory,radii_m=(.001,))
        self.assertAlmostEqual(report['near_sample_m'],expected['proximity'][0]['near_sample_m'])
        self.assertAlmostEqual(sum(report['groups'].values()),report['near_sample_m'])
        self.assertAlmostEqual(sum(row['near_m'] for row in report['rows']),report['near_sample_m'])
        self.assertAlmostEqual(sum(pair['near_m'] for pair in report['pairs']),report['near_sample_m'])
        self.assertAlmostEqual(sum(edge['near_m'] for edge in report['edges']),report['near_sample_m'])
        self.assertTrue(all(len(edge['endpoints'])==2 for edge in report['edges']))
        self.assertGreater(report['groups']['one_supplement'],0.)
        self.assertTrue(all(pair['block']==0 for pair in report['pairs']))

    def test_empty_invalid_budget_and_cancellation(self):
        empty=diagnostics.inspect(SimpleNamespace(ordered_items=[]))
        self.assertIsNone(empty['near_fraction'])
        trajectory=SimpleNamespace(ordered_items=[segment(0,0,0,0.)])
        for options in ({'radius_m':True},{'supplement_rows':[True]},{'supplement_rows':[1,1]},{'sample_step_m':0},{'include_edges':1}):
            with self.subTest(options=options),self.assertRaises(ValueError):diagnostics.inspect(trajectory,**options)
        with patch.object(reference,'MAX_SAMPLES',2):
            with self.assertRaisesRegex(ValueError,'采样超过预算'):diagnostics.inspect(trajectory)
        with patch.object(reference,'MAX_NEIGHBORS',1):
            with self.assertRaisesRegex(ValueError,'搜索超过预算'):diagnostics.inspect(trajectory)
        def cancel():raise RuntimeError('cancelled')
        with self.assertRaisesRegex(RuntimeError,'cancelled'):diagnostics.inspect(trajectory,check=cancel)


if __name__=='__main__':unittest.main()
