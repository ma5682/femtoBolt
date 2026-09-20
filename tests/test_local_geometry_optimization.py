import unittest
import open3d as o3d
from local_geometry import filter_small_components


class LocalGeometryOptimizationTests(unittest.TestCase):
    def test_post_retain_component_filter_reports_removed_fragment(self):
        mesh=o3d.geometry.TriangleMesh.create_sphere(radius=.1,resolution=8)
        fragment=o3d.geometry.TriangleMesh()
        fragment.vertices=o3d.utility.Vector3dVector([[1,0,0],[1,.01,0],[1,0,.01]])
        fragment.triangles=o3d.utility.Vector3iVector([[0,1,2]])
        mesh+=fragment
        report=filter_small_components(mesh)
        self.assertEqual(report['removed_components'],1)
        self.assertEqual(report['removed_triangles'],1)
        self.assertGreater(report['triangles_after'],30)


if __name__=='__main__':unittest.main()
