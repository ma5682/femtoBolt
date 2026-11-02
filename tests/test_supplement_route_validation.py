"""Independent assembly refuses invalid edges or selection projections."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from test_verified_visit_improvement import Graph,make_segment
from femto.planning import supplement_route_validation as module
from femto.planning.scan_visit_order import connect
from femto.planning.coverage_domain import scan_endpoints


class SupplementRouteValidationTests(unittest.TestCase):
    def setUp(self):
        self.segments=[make_segment(i,float(i)) for i in range(3)]
        for segment in self.segments:
            for point in segment.points:point['triangle_index']=0
        self.visits=list(enumerate([False]*3));self.links=connect(Graph(self.segments),self.segments,self.visits,lambda:None,1000)
        self.base=SimpleNamespace(planner={'parameters':dict(effective_width_mm=5.,quality_evidence={},row_spacing_mm=8.,effective_domain={})},
                                  blocks={'items':[{'component_id':0}]})
        self.mesh=SimpleNamespace(vertices=np.zeros((3,3)),faces=np.array([[0,1,2]]))
        self.region={'view_reference':{'undistorted_intrinsic':np.eye(3).tolist(),'image_size':[100,100]}}

    def run_checked(self,edge_result,projection_error=None):
        with patch.object(module.effective_domain,'validate',return_value=np.array([True])), \
             patch.object(module.quality,'edge_checks',return_value=(np.array([edge_result]),[],np.array([]))), \
             patch.object(module.quality,'checked_evidence',return_value=(np.ones((2,2)),None)), \
             patch.object(module.quality,'_summaries',return_value=({'status':'partial'},{'items':[]})), \
             patch.object(module,'selection_face_assignment',return_value=(None,None,np.ones((2,2)),None)), \
             patch.object(module,'check_projection_inside_selection',side_effect=projection_error) as projection:
            result=module.reconstruct_and_validate(self.base,self.segments,self.visits,self.links,self.mesh,self.region,budget=1000)
            self.assertEqual(projection.call_count,1)
            return result

    def test_valid_route_is_rebuilt_and_recorded(self):
        items,_,_,report=self.run_checked(True)
        self.assertEqual(len(items),1);self.assertTrue(report['all_edges_passed'])
        self.assertTrue(report['selection_projection_passed']);self.assertEqual(len(report['items_sha256']),64)
        self.assertEqual(report['semantics'],'geometry_only_not_coverage_or_physical_motion')

    def test_rejected_edge_or_projection_never_returns_route(self):
        with self.assertRaisesRegex(ValueError,'整边几何检查'):self.run_checked(False)
        with self.assertRaisesRegex(ValueError,'selection'):
            self.run_checked(True,ValueError('selection'))

    def test_relocated_handoffs_preserve_unique_scan_support(self):
        for connector_face in (0,1):
            with self.subTest(connector_face=connector_face):
                links=connect(Graph(self.segments),self.segments,self.visits,lambda:None,1000)
                links[0]['samples'][-1]['triangle_index']=connector_face
                with patch.object(module.effective_domain,'validate',return_value=np.array([True])), \
                     patch.object(module.quality,'edge_checks',return_value=(np.array([True]),[],np.array([]))), \
                     patch.object(module.quality,'checked_evidence',return_value=(np.ones((2,2)),None)), \
                     patch.object(module.quality,'_summaries',return_value=({'status':'partial'},{'items':[]})), \
                     patch.object(module,'selection_face_assignment',return_value=(None,None,np.ones((2,2)),None)), \
                     patch.object(module,'check_projection_inside_selection'):
                    old,_,_,old_report=module.reconstruct_and_validate(self.base,self.segments,self.visits,links,self.mesh,self.region,budget=1000)
                    new,_,_,new_report=module.reconstruct_relocated_and_validate(self.base,self.segments,self.visits,links,self.mesh,self.region,budget=1000)
                self.assertEqual(old_report['version'],module.VERSION)
                self.assertEqual(new_report['version'],module.RELOCATED_VERSION)
                def identities(items):
                    trajectory=SimpleNamespace(planner={},ordered_items=items)
                    return {(s['source_face'],*s['xyz_m']) for s in scan_endpoints(trajectory,np.array([7,8]))}
                self.assertEqual(identities(old),identities(new))
                # Distinct-face handoffs retain both records; equal-face ones
                # remove only the redundant, coincident scan point.
                self.assertEqual(len(old[0].points)-len(new[0].points),2 if connector_face==0 else 1)


if __name__=='__main__':unittest.main()
