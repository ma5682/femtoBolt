from copy import deepcopy
from fractions import Fraction
from types import SimpleNamespace
import unittest
import numpy as np
from femto.planning import scan_midpoint_seeds as module
from femto.planning.surface_trajectory import ScanSegment


def sample(x,y,z,face,role='scan'):
    return dict(xyz_m=[x,y,z],triangle_index=face,row_index=0,role=role)


class MidpointProposalTests(unittest.TestCase):
    def setUp(self):
        points=[sample(.1,.2,.3,0),sample(.13,.24,.37,1),
                sample(.15,.24,.37,1,'connector'),sample(.17,.24,.37,1)]
        self.trajectory=SimpleNamespace(planner={},ordered_items=[ScanSegment(0,0,points,0)])
        self.faces=np.asarray([8,17],dtype=np.int64)

    def test_midpoint_error_bound_and_conservative_radius(self):
        before=deepcopy(self.trajectory)
        endpoints,records,policy=module.propose(self.trajectory,self.faces)
        self.assertEqual(self.trajectory,before)
        self.assertEqual(len(endpoints),4);self.assertEqual(policy['eligible_edges'],2)
        self.assertEqual(len(records),3);self.assertEqual(module.propose(self.trajectory,self.faces),(endpoints,records,policy))
        for record in records:
            a,b=self.trajectory.ordered_items[0].points[record['edge_index']:record['edge_index']+2]
            self.assertNotEqual(b['role'],'connector')
            actual=sum(abs(Fraction(z)-(Fraction(x)+Fraction(y))/2) for x,y,z in zip(a['xyz_m'],b['xyz_m'],record['xyz_m']))
            self.assertGreaterEqual(Fraction(record['midpoint_error_bound_m']),actual)
            radius=.0025;safe=module.conservative_radius(radius,record)
            self.assertGreater(safe,0)
            self.assertGreaterEqual(Fraction(radius)-Fraction(safe),actual)

    def test_budget_cancellation_and_invalid_inputs(self):
        with self.assertRaisesRegex(ValueError,'超过预算'):
            module.propose(self.trajectory,self.faces,max_records=1)
        def cancel():raise RuntimeError('cancelled')
        with self.assertRaisesRegex(RuntimeError,'cancelled'):
            module.propose(self.trajectory,self.faces,check=cancel)
        for kwargs in (dict(min_edge_m=-1),dict(max_records=-1),dict(max_records=True)):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):module.propose(self.trajectory,self.faces,**kwargs)
        with self.assertRaises(ValueError):module.conservative_radius(.0025,dict(midpoint_error_bound_m=.0025))
        with self.assertRaises(ValueError):module.propose(self.trajectory,np.asarray([-1,2]))

    def test_zero_length_short_edges_and_connectors_do_not_create_midpoints(self):
        points=[sample(0.,0.,0.,0),sample(0.,0.,0.,0),sample(.001,0.,0.,0),sample(.1,0.,0.,0,'connector')]
        trajectory=SimpleNamespace(planner={},ordered_items=[ScanSegment(0,0,points,0)])
        _,records,policy=module.propose(trajectory,np.asarray([2]))
        self.assertEqual(records,[]);self.assertEqual(policy['eligible_edges'],0)
