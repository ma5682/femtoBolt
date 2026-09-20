from types import SimpleNamespace
import unittest
import numpy as np
from single_frame_mesh import build_pixel_mesh


class PixelMeshTests(unittest.TestCase):
    def build(self, depth):
        h, w = depth.shape
        intr = SimpleNamespace(width=w, height=h, fx=1000., fy=1000., cx=0., cy=0.)
        dist = dict.fromkeys(('k1','k2','p1','p2','k3','k4','k5','k6'), 0.)
        dist['model'] = 'BROWN_CONRADY_K6'
        color = np.zeros((h,w,3), np.uint8)
        color[0,0] = [10,20,30]
        return build_pixel_mesh(depth, color, intr, dist, np.eye(4))

    def test_planar_geometry_uv_and_winding(self):
        mesh, uv, report = self.build(np.full((4,5),1000,np.uint16))
        self.assertEqual(len(mesh.triangles), 24)
        self.assertEqual(len(mesh.vertices),20)
        np.testing.assert_allclose(np.asarray(mesh.vertices)[0],[0,0,1])
        np.testing.assert_allclose(uv[0],[.1,.875])
        np.testing.assert_allclose(uv[-1],[.9,.125])
        self.assertTrue(np.all(np.asarray(mesh.vertex_normals)[:,2] < -.99))

    def test_missing_pixel_and_foreground_step_not_bridged(self):
        depth = np.full((6,8), 1000, np.uint16)
        depth[:,4:] = 1500
        depth[2,2] = 0
        mesh, _, report = self.build(depth)
        vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
        self.assertTrue(np.all(np.ptp(vertices[faces,2],axis=1) == 0))
        self.assertFalse(np.any(np.all(np.isclose(vertices,[.002,.002,1]),axis=1)))
        self.assertGreater(report['rejected_discontinuity_triangles'],0)
        self.assertGreater(report['rejected_missing_depth_triangles'],0)

    def test_range_inclusive_and_invalid_range_rejected(self):
        depth = np.full((5,8),250,np.uint16)
        depth[:,4:] = 2210
        depth[0,0] = 249
        depth[4,7] = 2211
        mesh, _, _ = self.build(depth)
        self.assertEqual(set(np.asarray(mesh.vertices)[:,2]),{.25,2.21})


if __name__ == '__main__':
    unittest.main()
