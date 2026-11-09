"""Integer connectivity must match the existing native edge graph exactly."""
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import numpy as np
import open3d as o3d
from femto.geometry import triangle_components as component
from femto.local.local_geometry import filter_small_components


class TriangleComponentTests(unittest.TestCase):
    def mesh(self, faces):
        faces=np.asarray(faces,dtype=np.int32).reshape(-1,3)
        n=int(faces.max())+1 if faces.size else 0
        points=np.random.default_rng(17).normal(size=(n,3))
        return o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(points),o3d.utility.Vector3iVector(faces))

    def check_native(self, mesh):
        labels,counts,_=mesh.cluster_connected_triangles()
        actual=component.triangle_components(mesh)
        self.assertIsNotNone(actual)
        np.testing.assert_array_equal(actual[0],labels)
        np.testing.assert_array_equal(actual[1],counts)

    def test_edges_not_vertex_contacts_and_nonmanifold_duplicate_degenerate_faces(self):
        cases=[[],[[0,1,2],[2,3,4]],[[0,1,2],[1,0,3],[0,1,4]],
               [[0,1,2],[2,1,0],[1,1,2],[1,1,3],[4,4,4]],
               [[0,1,2],[3,4,5],[6,7,8],[2,1,3],[3,1,4]]]
        for faces in cases:
            with self.subTest(faces=faces):self.check_native(self.mesh(faces))

    def test_random_and_reordered_native_labels_counts(self):
        rng=np.random.default_rng(411)
        for n in (1,7,23,150,1000):
            for vertices in (10,100,2000):
                faces=rng.integers(0,vertices,(n,3),dtype=np.int32)
                for _ in range(3):
                    rng.shuffle(faces);self.check_native(self.mesh(faces))

    def test_long_strip_connectivity(self):
        faces=np.column_stack([np.arange(3000),np.arange(1,3001),np.arange(2,3002)])
        self.check_native(self.mesh(faces))

    def test_optional_work_failure_and_bounds_fall_back(self):
        mesh=self.mesh([[0,1,2],[2,1,3]])
        with patch.object(component,'MAX_WORK_BYTES',0):self.assertIsNone(component.triangle_components(mesh))
        with patch.object(component,'MAX_ROUNDS',0):self.assertIsNone(component.triangle_components(mesh))
        with patch.object(component.np,'argsort',side_effect=MemoryError):self.assertIsNone(component.triangle_components(mesh))
        for faces in (np.array([[0.,1.,2.]]),np.array([[-1,0,2]]),np.array([[0,1,2**31]])):
            self.assertIsNone(component.triangle_components(SimpleNamespace(triangles=faces)))

    def test_component_filter_matches_native_including_small_components(self):
        faces=np.column_stack([np.arange(200),np.arange(1,201),np.arange(2,202)])
        faces=np.concatenate([faces,np.array([[203,204,205],[204,205,206]])])
        fast=self.mesh(faces);legacy=o3d.geometry.TriangleMesh(fast)
        actual=filter_small_components(fast)
        with patch.object(component,'triangle_components',return_value=None):expected=filter_small_components(legacy)
        self.assertEqual(actual,expected)
        for field in ('vertices','triangles','vertex_normals'):
            np.testing.assert_array_equal(np.asarray(getattr(fast,field)),np.asarray(getattr(legacy,field)))

if __name__=='__main__':unittest.main()
