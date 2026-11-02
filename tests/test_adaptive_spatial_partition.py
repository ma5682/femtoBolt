"""Geometry-only proposals do not remove faces or change treatment width."""
from copy import deepcopy
import unittest
from unittest.mock import patch
import numpy as np
from test_mesh_slice_planner import grid_mesh
from femto.planning import adaptive_spatial_partition as adaptive,surface_direction_patches,spatial_patch_split


class AdaptiveSpatialPartitionTests(unittest.TestCase):
    def setUp(self):
        mesh=grid_mesh(width=.2,height=.1,cell=.005)
        self.vertices=np.asarray(mesh.vertices).copy();self.faces=np.asarray(mesh.faces).copy()
        patches=surface_direction_patches.partition(self.vertices,self.faces,np.arange(len(self.faces)))
        self.assertEqual(len(patches),1);self.part=patches[0]

    def localized(self):
        vertices=self.vertices.copy()
        target=np.argmin(np.linalg.norm(vertices[:,:2]-np.asarray([.075,0.]),axis=1))
        vertices[target,2]+=.005
        return vertices

    def test_flat_and_small_patches_keep_original(self):
        old=deepcopy(self.part)
        pieces,record=adaptive.propose(self.vertices,self.faces,self.part,[1,0,0],[0,1,0],5)
        self.assertFalse(record['adopted']);self.assertEqual(record['reason'],'no_local_metric_outlier')
        self.assertEqual(pieces,[old]);pieces[0]['face_ids'].pop();self.assertEqual(self.part,old)
        _,record=adaptive.propose(self.vertices,self.faces,self.part,[1,0,0],[0,1,0],30)
        self.assertEqual(record['reason'],'small_patch')

    def test_uniform_inclined_surface_does_not_trigger_local_outlier(self):
        vertices=self.vertices.copy();vertices[:,2]+=.5*vertices[:,1]
        pieces,record=adaptive.propose(vertices,self.faces,self.part,[1,0,0],[0,1,0],5)
        self.assertFalse(record['adopted']);self.assertEqual(record['reason'],'no_local_metric_outlier')
        self.assertEqual(pieces,[self.part])

    def test_local_outlier_proposes_bounded_face_preserving_split(self):
        vertices=self.localized();old=deepcopy(self.part)
        result,record=adaptive.propose(vertices,self.faces,self.part,[1,0,0],[0,1,0],5)
        self.assertTrue(record['adopted']);self.assertGreaterEqual(record['predicted_reduction'],adaptive.MIN_PREDICTED_REDUCTION)
        self.assertGreater(len(result),1)
        self.assertEqual(sorted(i for p in result for i in p['face_ids']),old['face_ids'])
        self.assertEqual(self.part,old)
        again=adaptive.propose(vertices,self.faces,self.part,[1,0,0],[0,1,0],5)
        self.assertEqual((result,record),again)
        reversed_faces=self.faces[:,[0,2,1]]
        reversed_result,reversed_record=adaptive.propose(vertices,reversed_faces,self.part,[1,0,0],[0,1,0],5)
        self.assertEqual((result,record),(reversed_result,reversed_record))
        scaled,scaled_record=adaptive.propose(vertices*2,self.faces,self.part,[1,0,0],[0,1,0],10)
        self.assertTrue(scaled_record['adopted'])
        self.assertEqual([p['face_ids'] for p in scaled],[p['face_ids'] for p in result])
        self.assertAlmostEqual(scaled_record['predicted_reduction'],record['predicted_reduction'])

    def test_component_budget_falls_back_and_cancel_propagates(self):
        with patch.object(spatial_patch_split,'MAX_PIECES',0):
            pieces,record=adaptive.propose(self.localized(),self.faces,self.part,[1,0,0],[0,1,0],5)
        self.assertEqual(record['reason'],'component_budget');self.assertEqual(pieces,[self.part])
        def cancel():raise RuntimeError('cancelled')
        with self.assertRaisesRegex(RuntimeError,'cancelled'):
            adaptive.propose(self.vertices,self.faces,self.part,[1,0,0],[0,1,0],5,cancel)

    def test_invalid_width_axis_and_mesh_rejected(self):
        for along,across,width in (([1,0,0],[1,0,0],5),([0,0,0],[0,1,0],5),([1,0,0],[0,1,0],0)):
            with self.subTest(along=along,width=width),self.assertRaises(ValueError):
                adaptive.propose(self.vertices,self.faces,self.part,along,across,width)
