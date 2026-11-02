"""Adaptive proposals react to geometry without certifying a treatment area."""
import unittest
import numpy as np
from femto.planning import adaptive_spacing_research as module


class AdaptiveSpacingResearchTests(unittest.TestCase):
    def setUp(self):
        self.vertices=np.array([[-.01,-.01,1.],[.01,-.01,1.],[0.,.01,1.]])
        self.faces=np.array([[0,1,2]])
        self.frame={'origin':[0.,0.,1.],'e2':[0.,1.,0.],'n':[0.,0.,1.]}
        self.k=np.array([[100.,0.,10.],[0.,100.,10.],[0.,0.,1.]])
        self.distance=module.selection_distance(np.ones((20,20),dtype=bool))

    def test_boundary_and_invalid_mask(self):
        self.assertIsNone(module.selection_distance(None))
        self.assertFalse(self.distance.flags.writeable)
        self.assertGreater(self.distance[10,10],self.distance[0,0])
        with self.assertRaises(ValueError):module.selection_distance(np.ones((2,2)))

    def test_flat_wide_and_interior_get_higher_lower_bound(self):
        kwargs=dict(vertices=self.vertices,faces=self.faces,face_ids=[0],frame=self.frame,
                    metric_span_m=.04,width_mm=10.,distance=self.distance,intrinsic=self.k)
        flat=module.choose_fraction(**kwargs)
        self.assertAlmostEqual(flat['minimum_fraction'],.8)
        narrow=module.choose_fraction(**dict(kwargs,metric_span_m=.005))
        self.assertAlmostEqual(narrow['minimum_fraction'],.65)
        unknown=module.choose_fraction(**dict(kwargs,distance=None))
        self.assertAlmostEqual(unknown['minimum_fraction'],.65)
        tilted=module.choose_fraction(**dict(kwargs,frame=dict(self.frame,n=[1.,0.,0.])))
        self.assertAlmostEqual(tilted['minimum_fraction'],.65)
        near=np.ones((20,20),dtype=bool);near[9:12,9:12]=False
        edge=module.choose_fraction(**dict(kwargs,distance=module.selection_distance(near)))
        self.assertLess(edge['minimum_fraction'],flat['minimum_fraction'])
        self.assertIn('not_physical_clearance',flat['semantics'])

    def test_parallel_shared_seam_projects_previous_positions(self):
        patch={'seam_edges':[[1,2]]};previous=[dict(seed_face=7,seam_edges=[[1,2]],
            frame={'origin':[0.,.1,1.],'e2':[0.,1.,0.]},positions_m=[-.05,0.,.05])]
        anchors,sources=module.seam_anchors(patch,self.frame,previous,0.,.2)
        np.testing.assert_allclose(anchors,[.05,.1,.15],atol=1e-14)
        self.assertEqual(sources,[{'seed_face':7,'anchors':3}])
        previous[0]['frame']['e2']=[1.,0.,0.]
        self.assertEqual(module.seam_anchors(patch,self.frame,previous,0.,.2),([],[]))
        previous[0]['frame']['e2']=[0.,1.,0.];previous[0]['seam_edges']=[[5,6]]
        self.assertEqual(module.seam_anchors(patch,self.frame,previous,0.,.2),([],[]))


if __name__=='__main__':unittest.main()
