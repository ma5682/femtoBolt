from copy import deepcopy
from fractions import Fraction
from types import SimpleNamespace
import unittest
import numpy as np
from femto.planning import scan_fraction_seeds as module,scan_midpoint_seeds
from femto.planning.surface_trajectory import ScanSegment


def point(x,y,z,face,role='scan'):
    return dict(xyz_m=[x,y,z],triangle_index=face,row_index=0,role=role)


class FractionSeedTests(unittest.TestCase):
    def setUp(self):
        self.points=[point(.1,.2,.3,0),point(.13,.24,.37,1),point(.15,.24,.37,1,'connector'),point(.17,.24,.37,1)]
        self.trajectory=SimpleNamespace(planner={},ordered_items=[ScanSegment(0,0,self.points,0)])
        self.faces=np.asarray([8,17],dtype=np.int64)

    def test_dyadic_points_bound_error_and_are_deterministic(self):
        before=deepcopy(self.trajectory);requests=[(0,0,3,8),(0,0,3,8),(0,2,1,4)]
        records,report=module.propose(self.trajectory,self.faces,requests)
        self.assertEqual(len(records),3);self.assertEqual(report['unique_requests'],2)
        self.assertEqual(self.trajectory,before);self.assertEqual(module.propose(self.trajectory,self.faces,requests),(records,report))
        for p in records:
            a,b=self.points[p['edge_index']:p['edge_index']+2];n=p['fraction_numerator'];d=p['fraction_denominator']
            actual=sum(abs(Fraction(z)-((d-n)*Fraction(x)+n*Fraction(y))/d) for x,y,z in zip(a['xyz_m'],b['xyz_m'],p['xyz_m']))
            self.assertGreaterEqual(Fraction(p['midpoint_error_bound_m']),actual)
            radius=.0025;safe=scan_midpoint_seeds.conservative_radius(radius,p)
            self.assertGreaterEqual(Fraction(radius)-Fraction(safe),actual)

    def test_half_position_matches_existing_midpoint_proposal(self):
        _,mid,_=scan_midpoint_seeds.propose(self.trajectory,self.faces)
        dyadic,_=module.propose(self.trajectory,self.faces,[(0,0,1,2),(0,2,1,2)])
        self.assertEqual([(x['source_face'],x['xyz_m'],x['midpoint_error_bound_m']) for x in mid],
                         [(x['source_face'],x['xyz_m'],x['midpoint_error_bound_m']) for x in dyadic])

    def test_invalid_connector_fraction_budget_and_cancel(self):
        for request in ((0,1,1,2),(0,0,0,2),(0,0,2,2),(0,0,1,3),(0,0,1,512),(9,0,1,2),(0,0,True,2)):
            with self.subTest(request=request),self.assertRaises(ValueError):module.propose(self.trajectory,self.faces,[request])
        with self.assertRaisesRegex(ValueError,'超过预算'):
            module.propose(self.trajectory,self.faces,[(0,0,1,2)],max_records=1)
        def cancel():raise RuntimeError('cancelled')
        with self.assertRaisesRegex(RuntimeError,'cancelled'):module.propose(self.trajectory,self.faces,[(0,0,1,2)],check=cancel)
        bad=SimpleNamespace(planner={'algorithm_version':'nominal-variable-width-surface-v1'},ordered_items=self.trajectory.ordered_items)
        with self.assertRaises(ValueError):module.propose(bad,self.faces,[])
