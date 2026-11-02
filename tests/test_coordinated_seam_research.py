"""Shared-seam direction proposals retain row-count and geometry guards."""
import unittest
import numpy as np
from femto.planning import coordinated_seam_research as module


class CoordinatedSeamResearchTests(unittest.TestCase):
    def setUp(self):
        self.vertices=np.array([[-.05,-.01,.5],[.05,-.01,.5],[.05,.01,.5],[-.05,.01,.5]])
        self.faces=np.array([[0,1,2],[0,2,3]])
        self.patch={'face_ids':[0,1],'seam_edges':[[0,1]]}
        self.frame={'origin':[0.,0.,.5],'n':[0.,0.,1.],'e1':[1.,0.,0.],'e2':[.5,3**.5/2,0.]}
        self.prior=[{'seed_face':8,'seam_edges':[[0,1]],'frame':{'e2':[0.,1.,0.]}}]

    def test_real_shared_seam_can_align_without_increasing_span_proxy(self):
        result=module.propose(self.vertices,self.faces,self.patch,self.frame,self.prior)
        self.assertIsNotNone(result);frame,decision=result
        self.assertAlmostEqual(abs(frame['e2'][1]),1.)
        self.assertLess(decision['before_alignment'],decision['after_alignment'])
        self.assertLessEqual(decision['candidate_span_proxy_m'],decision['original_span_proxy_m'])
        self.assertEqual(decision['neighbor_seed_face'],8)

    def test_incompatible_or_unshared_direction_is_not_proposed(self):
        frame=dict(self.frame,e2=[0.,1.,0.]);prior=[dict(self.prior[0],frame={'e2':[1.,0.,0.]})]
        self.assertIsNone(module.propose(self.vertices,self.faces,self.patch,frame,prior))
        self.assertIsNone(module.propose(self.vertices,self.faces,dict(self.patch,seam_edges=[[2,3]]),self.frame,self.prior))
        self.assertIsNone(module.propose(self.vertices,self.faces,self.patch,dict(self.frame,e2=[0.,1.,0.]),self.prior))

    def test_invalid_input_and_cancellation(self):
        with self.assertRaises(ValueError):module.propose(self.vertices,self.faces,self.patch,dict(self.frame,n=[0.,0.,0.]),self.prior)
        def cancel():raise RuntimeError('cancelled')
        with self.assertRaisesRegex(RuntimeError,'cancelled'):
            module.propose(self.vertices,self.faces,self.patch,self.frame,self.prior,cancel)

    def test_near_normal_direction_is_rejected_even_with_smaller_span(self):
        vertices=np.array([[-.01,-.05,0.5+np.tan(np.radians(60))*.01],
                           [.01,-.05,0.5-np.tan(np.radians(60))*.01],
                           [.01,.05,0.5-np.tan(np.radians(60))*.01],
                           [-.01,.05,0.5+np.tan(np.radians(60))*.01]])
        frame=dict(self.frame,e2=[0.,1.,0.]);prior=[dict(self.prior[0],frame={'e2':[1.,0.,0.]})]
        self.assertIsNone(module.propose(vertices,self.faces,self.patch,frame,prior))


if __name__=='__main__':unittest.main()
