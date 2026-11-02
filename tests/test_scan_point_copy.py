"""Fast point copying must retain deepcopy semantics and caller isolation."""
from copy import deepcopy
from dataclasses import replace
import unittest

from femto.planning.scan_visit_order import _clone_points, oriented,reconstruct
from femto.planning.bounded_joint_reorder import _full_anchors
from femto.planning.surface_trajectory import ScanSegment


class ScanPointCopyTests(unittest.TestCase):
    def test_shared_points_and_coordinate_lists_stay_shared_only_in_copy(self):
        coordinates=[.1,.2,.3]
        point=dict(xyz_m=coordinates,pixel=[1.,2.],row_index=2,triangle_index=0,role='scan')
        original=[point,point,dict(xyz_m=coordinates,pixel=[3.,4.],row_index=2,triangle_index=0)]
        result=_clone_points(original)
        self.assertEqual(result,deepcopy(original))
        self.assertIs(result[0],result[1]);self.assertIs(result[0]['xyz_m'],result[2]['xyz_m'])
        self.assertIsNot(result[0],point);self.assertIsNot(result[0]['xyz_m'],coordinates)
        result[0]['xyz_m'][0]=99.;self.assertEqual(coordinates[0],.1)

    def test_nested_objects_and_cycles_use_full_deepcopy(self):
        nested={'values':[[1,2],{'third':[3]}]};point={'xyz_m':[0.,0.,0.],'metadata':nested}
        original=[point,point];result=_clone_points(original)
        self.assertEqual(result,deepcopy(original));self.assertIs(result[0],result[1])
        result[0]['metadata']['values'][0][0]=42;self.assertEqual(nested['values'][0][0],1)
        cyclic=[];cyclic.append(cyclic)
        original=[{'cycle':cyclic}];result=_clone_points(original)
        self.assertIs(result[0]['cycle'][0],result[0]['cycle']);self.assertIsNot(result[0]['cycle'],cyclic)

    def test_tuple_containing_mutable_value_uses_full_copy_and_preserves_alias(self):
        shared=[1,2];original=[{'nested':(shared,), 'other':shared}]
        result=_clone_points(original)
        self.assertEqual(result,deepcopy(original));self.assertIs(result[0]['nested'][0],result[0]['other'])
        result[0]['other'][0]=99;self.assertEqual(shared[0],1)

    def test_reverse_preserves_incoming_roles_and_isolates_mutations(self):
        points=[dict(xyz_m=[float(i),0.,0.],pixel=[i,0],row_index=0,triangle_index=0,
                     role=role,revisit=True) for i,role in enumerate(('scan','connector','scan'))]
        segment=ScanSegment(0,0,points,0)
        for reverse in (False,True):
            expected=deepcopy(points)
            if reverse:
                expected.reverse()
                for i in range(1,len(expected)):expected[i]['role']=points[-i].get('role','scan')
                expected[0]['role']='scan'
            for point in expected:point.pop('revisit',None)
            result=oriented(segment,reverse)
            self.assertEqual(result,replace(segment,points=expected,row_index=expected[0]['row_index']))
            result.points[0]['xyz_m'][0]=99.;self.assertNotEqual(points[-1 if reverse else 0]['xyz_m'][0],99.)

    def test_reconstructed_connector_aliases_are_detached_in_both_paths(self):
        scans=[ScanSegment(i,i,[dict(xyz_m=[float(i),0.,0.],row_index=i,triangle_index=0,role='scan'),
                                dict(xyz_m=[float(i),1.,0.],row_index=i,triangle_index=0,role='scan')],0) for i in range(2)]
        for nested in (False,True):
            point=dict(xyz_m=[1.,0.,0.],row_index=1,triangle_index=0,role='connector')
            if nested:point['metadata']={'values':[[1,2]]}
            links=[dict(from_segment=0,to_segment=1,connected=True,added_edges=2,
                        samples=[point,point],reason='verified_surface_path')]
            for build in (lambda:reconstruct(scans,[(0,False),(1,False)],links,preserve_endpoint_faces=True,budget=100),
                          lambda:_full_anchors(scans,[(0,False),(1,False)],links,lambda:None,100)):
                chains,_,_=build();copied=[p for _,chain in chains for p in chain if p.get('role')=='connector']
                self.assertEqual(len(copied),2);self.assertIs(copied[0],copied[1]);self.assertIsNot(copied[0],point)
                copied[0]['xyz_m'][0]=99.;self.assertEqual(point['xyz_m'][0],1.)
                if nested:
                    copied[0]['metadata']['values'][0][0]=42
                    self.assertEqual(point['metadata']['values'][0][0],1)


if __name__ == '__main__':unittest.main()
