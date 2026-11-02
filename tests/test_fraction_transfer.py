"""Transfer fractional requests only across identical active scan edges."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from femto.planning import fraction_transfer,scan_fraction_seeds
from femto.planning.surface_trajectory import ScanSegment


class FractionTransferTests(unittest.TestCase):
    def setUp(self):
        self.points=[dict(xyz_m=[.01,.01,0.],triangle_index=0,row_index=0,role='scan'),
                     dict(xyz_m=[.08,.01,0.],triangle_index=1,row_index=0,role='scan')]
        self.source=SimpleNamespace(planner={},ordered_items=[ScanSegment(0,0,deepcopy(self.points),0)])
        self.target=SimpleNamespace(planner={},ordered_items=[ScanSegment(5,0,list(reversed(deepcopy(self.points))),0)])
        self.faces=np.array([10,11],dtype='<i8');self.requests=[(0,0,3,8)]

    def test_reverse_edge_rederives_same_positions_and_bounds(self):
        before=deepcopy((self.source,self.target))
        requests,positions,report=fraction_transfer.transfer(self.source,self.target,self.faces,self.requests)
        self.assertEqual(requests,[(5,0,5,8)]);self.assertEqual(report['matched_records'],2)
        self.assertEqual(report['unmatched_records'],0)
        source,_=scan_fraction_seeds.propose(self.source,self.faces,self.requests)
        self.assertEqual({(p['source_face'],*p['xyz_m'],p['midpoint_error_bound_m']) for p in source},
                         {(p['source_face'],*p['xyz_m'],p['midpoint_error_bound_m']) for p in positions})
        self.assertEqual((self.source,self.target),before)

    def test_connectors_changed_faces_and_budget_do_not_become_evidence(self):
        target=deepcopy(self.target);target.ordered_items[0].points[1]['role']='connector'
        requests,_,report=fraction_transfer.transfer(self.source,target,self.faces,self.requests)
        self.assertEqual(requests,[]);self.assertEqual(report['unmatched_records'],2)
        target=deepcopy(self.target);target.ordered_items[0].points[0]['triangle_index']=0
        requests,_,report=fraction_transfer.transfer(self.source,target,self.faces,self.requests)
        self.assertEqual(requests,[]);self.assertEqual(report['unmatched_records'],2)
        with patch.object(fraction_transfer,'MAX_EDGES',0),self.assertRaisesRegex(ValueError,'超过预算'):
            fraction_transfer.transfer(self.source,self.target,self.faces,self.requests)
        def cancel():raise RuntimeError('cancelled')
        with self.assertRaisesRegex(RuntimeError,'cancelled'):
            fraction_transfer.transfer(self.source,self.target,self.faces,self.requests,check=cancel)


if __name__=='__main__':unittest.main()
