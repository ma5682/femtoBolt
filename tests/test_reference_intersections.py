"""A nearby point is never sufficient to recover a missing surface ray."""
import unittest
import numpy as np
import open3d as o3d
from femto.geometry.reference_intersections import retry_missed_rays


class ReferenceIntersectionTests(unittest.TestCase):
    def mesh_scene(self):
        vertices=np.array([[-.1,-.1,.5],[.1,-.1,.5],[-.1,.1,.5],[.1,.1,.5]])
        mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),o3d.utility.Vector3iVector([[0,1,2]]))
        scene=o3d.t.geometry.RaycastingScene();scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        return mesh,scene

    def test_interior_shared_edge_and_vertex_are_actual_intersections(self):
        mesh,scene=self.mesh_scene();rays=np.array([[-.1,-.1,1.],[-.2,-.2,1.],[0.,0.,1.]])
        hits=retry_missed_rays(mesh,scene,rays,rays*.5)
        np.testing.assert_allclose(hits,.5,rtol=0,atol=1e-15)

    def test_positive_gap_near_edge_is_not_closed_by_nearest_point(self):
        mesh,scene=self.mesh_scene();rays=np.array([[1e-9,1e-9,1.],[.1,.1,1.]])
        hits=retry_missed_rays(mesh,scene,rays,rays*.5)
        self.assertTrue(np.isnan(hits).all())

    def test_cancel(self):
        mesh,scene=self.mesh_scene()
        def cancel():raise InterruptedError('cancel')
        with self.assertRaises(InterruptedError):retry_missed_rays(mesh,scene,[[0,0,1]],[[0,0,.5]],cancel)


if __name__=='__main__':unittest.main()
