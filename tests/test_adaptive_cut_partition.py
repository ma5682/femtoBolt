from copy import deepcopy
import unittest
from unittest.mock import patch
import numpy as np
from test_mesh_slice_planner import grid_mesh
from femto.planning import adaptive_cut_partition as cut,spatial_patch_split,surface_direction_patches


class AdaptiveCutTests(unittest.TestCase):
    def setUp(self):
        mesh=grid_mesh(width=.2,height=.1,cell=.005)
        self.vertices=np.asarray(mesh.vertices).copy();self.faces=np.asarray(mesh.faces).copy()
        self.part=surface_direction_patches.partition(self.vertices,self.faces,np.arange(len(self.faces)))[0]

    def bump(self):
        vertices=self.vertices.copy()
        vertices[np.argmin(np.linalg.norm(vertices[:,:2]-[.075,0.],axis=1)),2]+=.005
        return vertices

    def test_fraction_cut_preserves_faces_seams_and_validates_input(self):
        for fraction in (.4,.5,.6):
            with self.subTest(fraction=fraction):
                parts=spatial_patch_split.split(self.vertices,self.faces,self.part,[1,0,0],2,cut_fraction=fraction)
                self.assertEqual(sorted(i for p in parts for i in p['face_ids']),self.part['face_ids'])
                self.assertEqual(set(map(tuple,self.part['outer_edges'])),{tuple(e) for p in parts for e in p['outer_edges']})
                self.assertEqual(parts,spatial_patch_split.split(self.vertices,self.faces,self.part,[1,0,0],2,cut_fraction=fraction))
        for invalid in (0,1,-.1,float('nan'),True):
            with self.subTest(invalid=invalid),self.assertRaises(ValueError):
                spatial_patch_split.split(self.vertices,self.faces,self.part,[1,0,0],2,cut_fraction=invalid)
        with self.assertRaises(ValueError):
            spatial_patch_split.split(self.vertices,self.faces,self.part,[1,0,0],3,cut_fraction=.4)

    def test_flat_rejected_local_bump_proposes_best_bounded_score(self):
        unchanged=deepcopy(self.part)
        parts,decision=cut.propose(self.vertices,self.faces,self.part,[1,0,0],[0,1,0],5)
        self.assertFalse(decision['adopted']);self.assertEqual(parts,[unchanged])
        with patch.object(spatial_patch_split,'split',wraps=spatial_patch_split.split) as split:
            parts,decision=cut.propose(self.bump(),self.faces,self.part,[1,0,0],[0,1,0],5)
        self.assertTrue(decision['adopted']);self.assertLessEqual(split.call_count,4)
        self.assertEqual(sorted(i for p in parts for i in p['face_ids']),unchanged['face_ids'])
        scores=[c['predicted_reduction'] for c in decision['candidates'] if c['status']=='evaluated']
        self.assertAlmostEqual(decision['predicted_reduction'],max(scores))
        self.assertEqual(self.part,unchanged)
        again=cut.propose(self.bump(),self.faces[:,[0,2,1]],self.part,[1,0,0],[0,1,0],5)
        self.assertEqual((parts,decision),again)

    def test_cancellation_and_component_budget(self):
        def cancel():raise RuntimeError('cancelled')
        with self.assertRaisesRegex(RuntimeError,'cancelled'):
            cut.propose(self.vertices,self.faces,self.part,[1,0,0],[0,1,0],5,cancel)
        with patch.object(spatial_patch_split,'MAX_PIECES',0):
            parts,record=cut.propose(self.bump(),self.faces,self.part,[1,0,0],[0,1,0],5)
        self.assertFalse(record['adopted']);self.assertEqual(parts,[self.part])

    def test_explicit_half_cut_does_not_follow_other_research_candidates(self):
        with patch.object(cut,'CANDIDATE_FRACTIONS',(.4,)):
            _,record=cut.propose(self.bump(),self.faces,self.part,[1,0,0],[0,1,0],5,fractions=(.5,))
        self.assertTrue(record['adopted'])
        self.assertEqual([item['fraction'] for item in record['candidates']],[.5])
        for fractions in ((),(.5,.5),(0,),([.5],),(True,),(.1,.2,.3,.4)):
            with self.subTest(fractions=fractions),self.assertRaises(ValueError):
                cut.propose(self.vertices,self.faces,self.part,[1,0,0],[0,1,0],5,fractions=fractions)
