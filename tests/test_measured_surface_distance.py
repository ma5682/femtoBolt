"""Triangle witnesses preserve measured support and component boundaries."""
import unittest
from unittest.mock import Mock
import numpy as np
import open3d as o3d
from femto.geometry.measured_surface_distance import same_surface_support


class MeasuredSurfaceDistanceTests(unittest.TestCase):
    def surface(self, slope=0., second=False):
        vertices=np.array([[-.1,-.1,1-slope*.1],[.1,-.1,1+slope*.1],
                           [-.1,.1,1-slope*.1],[.1,.1,1+slope*.1]])
        faces=np.array([[0,1,2],[1,3,2]])
        if second:
            vertices=np.r_[vertices,vertices+[0,0,.004]];faces=np.r_[faces,faces+4]
        mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),o3d.utility.Vector3iVector(faces))
        scene=o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        return mesh,scene

    def test_oblique_surface_uses_actual_distance_not_axial_depth_difference(self):
        mesh,scene=self.surface(slope=5.)
        accepted,distance,_=same_surface_support(mesh,scene,[[0,0,1.004]],[[0,0,1]],[0],2.)
        self.assertTrue(accepted[0])
        self.assertAlmostEqual(distance[0],4/np.sqrt(26),places=5)

    def test_flat_surface_shift_and_unrelated_nearby_sheet_remain_rejected(self):
        for second in (False,True):
            with self.subTest(second=second):
                mesh,scene=self.surface(second=second)
                accepted,_,record=same_surface_support(mesh,scene,[[0,0,1.004]],[[0,0,1]],[0],2.)
                self.assertFalse(accepted[0])
                self.assertEqual(record['component_mismatches'],int(second))

    def test_missing_reference_face_requires_an_actual_coincident_triangle(self):
        mesh,scene=self.surface()
        accepted,_,record=same_surface_support(mesh,scene,[[0,0,1.001]],[[0,0,1.01]],[-1],2.)
        self.assertFalse(accepted[0]);self.assertEqual(record['unresolved_reference_faces'],1)
        accepted,_,_=same_surface_support(mesh,scene,[[0,0,1.001]],[[0,0,1]],[-1],2.)
        self.assertTrue(accepted[0])

    def test_approximate_nearest_proposal_cannot_forge_a_distance(self):
        mesh,_=self.surface()
        scene=Mock()
        scene.compute_closest_points.return_value={
            'primitive_ids':o3d.core.Tensor([0],dtype=o3d.core.Dtype.UInt32),
            'primitive_uvs':o3d.core.Tensor([[.5,.5]],dtype=o3d.core.Dtype.Float32),
            'points':o3d.core.Tensor([[0,0,1.004]],dtype=o3d.core.Dtype.Float32)}
        accepted,distance,_=same_surface_support(mesh,scene,[[0,0,1.004]],[[0,0,1]],[0],2.)
        self.assertFalse(accepted[0]);self.assertAlmostEqual(distance[0],4.)
