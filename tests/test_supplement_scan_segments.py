"""Extra scans may contain only explicitly selected old scan edges."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from femto.planning import supplement_scan_segments as module
from femto.planning.surface_trajectory import ScanSegment


class SupplementSegmentTests(unittest.TestCase):
    def setUp(self):
        points=[dict(xyz_m=[i*.01,0.,0.],row_index=0,triangle_index=0,role='scan',revisit=True) for i in range(6)]
        points[3]['role']='connector'
        self.source=SimpleNamespace(ordered_items=[ScanSegment(4,0,points,1)])

    def test_segments_coalesce_selected_edges_without_mutating_source(self):
        before=deepcopy(self.source)
        segments,report=module.materialize(self.source,[(4,0),(4,1),(4,3)],row_offset=10)
        self.assertEqual([len(s.points) for s in segments],[3,2])
        self.assertEqual([s.block_index for s in segments],[1,1])
        self.assertTrue(all(p['row_index']==10 and p['role']=='scan' and 'revisit' not in p for s in segments for p in s.points))
        self.assertEqual(report['selected_edges'],3);self.assertEqual(self.source,before)
        segments[0].points[0]['xyz_m'][0]=99.;self.assertEqual(self.source,before)

    def test_connector_duplicate_and_budget_are_rejected(self):
        for edges in ([(4,2)],[(4,0),(4,0)],[(4,9)],[(True,0)]):
            with self.subTest(edges=edges),self.assertRaises(ValueError):module.materialize(self.source,edges,row_offset=1)
        with patch.object(module,'MAX_SEGMENTS',0),self.assertRaisesRegex(ValueError,'超过预算'):
            module.materialize(self.source,[(4,0)],row_offset=1)
        def cancel():raise RuntimeError('cancelled')
        with self.assertRaisesRegex(RuntimeError,'cancelled'):module.materialize(self.source,[(4,0)],row_offset=1,check=cancel)

    def test_adjacent_selected_edges_from_different_rows_are_separate(self):
        self.source.ordered_items[0].points[2]['row_index']=1
        result,_=module.materialize(self.source,[(4,0),(4,1)],row_offset=10)
        self.assertEqual([s.row_index for s in result],[10,11]);self.assertEqual([len(s.points) for s in result],[2,2])


if __name__=='__main__':unittest.main()
