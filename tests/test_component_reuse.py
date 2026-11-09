"""Counts may be reused only for the exact mesh returned by the filter."""
from pathlib import Path
from tempfile import TemporaryDirectory
from concurrent.futures import ThreadPoolExecutor
from unittest import TestCase,main
from unittest.mock import patch
import numpy as np
import open3d as o3d
from femto.geometry.component_reuse import ComponentReuse
from femto.geometry.fusion import export_mesh_formats
from femto.local.local_geometry import filter_small_components
from femto.performance_detail import collect


def mesh_with_fragment():
    mesh=o3d.geometry.TriangleMesh.create_sphere(radius=.1,resolution=8)
    fragment=o3d.geometry.TriangleMesh()
    fragment.vertices=o3d.utility.Vector3dVector([[1,0,0],[1,.01,0],[1,0,.01]])
    fragment.triangles=o3d.utility.Vector3iVector([[0,1,2]])
    return mesh+fragment


class ComponentReuseTests(TestCase):
    def test_report_bytes_and_geometry_match_fresh_statistics(self):
        mesh=mesh_with_fragment();old=o3d.geometry.TriangleMesh(mesh)
        receipt=ComponentReuse()
        expected=filter_small_components(old)
        actual=filter_small_components(mesh,component_reuse=receipt)
        self.assertEqual(actual,expected)
        for field in ('vertices','triangles','vertex_normals'):
            self.assertEqual(np.asarray(getattr(mesh,field)).tobytes(),np.asarray(getattr(old,field)).tobytes())
        self.assertEqual(receipt.matching_counts(mesh),(len(mesh.triangles),))
        with TemporaryDirectory() as folder:
            root=Path(folder)
            a=export_mesh_formats(mesh,root/'fresh',write_obj_glb=False)
            with collect() as detail:
                b=export_mesh_formats(mesh,root/'reuse',write_obj_glb=False,component_reuse=receipt)
            self.assertEqual(a['report'],b['report'])
            self.assertEqual(Path(a['ply']).read_bytes(),Path(b['ply']).read_bytes())
            self.assertEqual(detail.snapshot()['rows']['mesh.component_reuse']['counts'],{'hits':1})

    def test_vertex_and_topology_alias_mutations_invalidate(self):
        for kind in ('vertex','face'):
            mesh=mesh_with_fragment();receipt=ComponentReuse()
            filter_small_components(mesh,component_reuse=receipt)
            if kind=='vertex':np.asarray(mesh.vertices)[0,0]+=.001
            else:np.asarray(mesh.triangles)[0]=[0,1,2]
            self.assertIsNone(receipt.matching_counts(mesh))
            self.assertIsNone(receipt.matching_counts(mesh))
            with TemporaryDirectory() as folder:
                root=Path(folder)
                a=export_mesh_formats(mesh,root/'plain',write_obj_glb=False)
                with collect() as details:
                    b=export_mesh_formats(mesh,root/'changed',write_obj_glb=False,component_reuse=receipt)
                self.assertEqual(a['report'],b['report'])
                self.assertEqual(details.snapshot()['rows']['mesh.component_reuse']['counts']['cluster_fallbacks'],1)

    def test_receipt_does_not_trust_object_identity_copy_thread_or_failed_rebind(self):
        mesh=mesh_with_fragment();receipt=ComponentReuse()
        filter_small_components(mesh,component_reuse=receipt)
        self.assertIsNone(receipt.matching_counts(o3d.geometry.TriangleMesh(mesh)))
        with ThreadPoolExecutor(max_workers=1) as executor:
            self.assertIsNone(executor.submit(receipt.matching_counts,mesh).result())
        with patch('femto.geometry.component_reuse._fingerprint',side_effect=MemoryError):
            filter_small_components(mesh,component_reuse=receipt)
        self.assertIsNone(receipt.matching_counts(mesh))
        filter_small_components(mesh,component_reuse=receipt)
        filter_small_components(o3d.geometry.TriangleMesh(),component_reuse=receipt)
        self.assertIsNone(receipt.matching_counts(mesh))

    def test_nonmanifold_and_empty_report_semantics_remain_trimesh(self):
        nonmanifold=o3d.geometry.TriangleMesh()
        nonmanifold.vertices=o3d.utility.Vector3dVector([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[0,-1,0]])
        nonmanifold.triangles=o3d.utility.Vector3iVector([[0,1,2],[0,1,3],[0,1,4]])
        with TemporaryDirectory() as folder:
            report=export_mesh_formats(nonmanifold,Path(folder),write_obj_glb=False)['report']
        self.assertFalse(report['watertight'])
        self.assertTrue(report['winding_consistent'])
        self.assertEqual(report['nonmanifold_edges'],1)
        with TemporaryDirectory() as folder,self.assertRaisesRegex(RuntimeError,'Open3D failed to write'):
            export_mesh_formats(o3d.geometry.TriangleMesh(),Path(folder),write_obj_glb=False,component_reuse=ComponentReuse())


if __name__=='__main__':main()
