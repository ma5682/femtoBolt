import unittest
import numpy as np
import open3d as o3d
from femto.annotation.scene_surface_picker import SurfacePicker


class SurfacePickerTests(unittest.TestCase):
    def setUp(self):
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector([[-1,-1,1],[1,-1,1],[0,1,1],[-1,-1,2],[1,-1,2],[0,1,2]])
        mesh.triangles = o3d.utility.Vector3iVector([[0,1,2],[3,4,5]])
        self.picker = SurfacePicker(mesh)

    def test_nearest_surface_and_units(self):
        hit = self.picker.pick([0,0,0],[0,0,3])
        np.testing.assert_allclose(hit['point_world_m'],[0,0,1])
        self.assertEqual(hit['triangle_index'],0)
        self.assertEqual(hit['unit'],'metre')

    def test_background_and_clipping_have_no_hit(self):
        self.assertIsNone(self.picker.pick([2,2,0],[2,2,3]))
        self.assertIsNone(self.picker.pick([0,0,0],[0,0,.5]))
        self.assertIsNone(self.picker.pick([0,0,0],[0,0,-3]))

    def test_invalid_ray(self):
        with self.assertRaises(ValueError): self.picker.pick([0,0,0],[0,0,0])
        with self.assertRaises(ValueError): self.picker.pick([np.nan,0,0],[0,0,1])
