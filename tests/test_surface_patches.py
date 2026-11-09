"""Observed representatives must not close holes or merge separate surfaces."""
import unittest
import numpy as np
import open3d as o3d
from femto.geometry.surface_patches import simplify, _edges


def grid(hole=False):
    y,x=np.mgrid[:121,:121];v=np.c_[x.ravel()*.0005,y.ravel()*.0005,np.full(x.size,.5)]
    a=(y[:-1,:-1]*121+x[:-1,:-1]).ravel();f=np.concatenate((np.c_[a,a+1,a+121],np.c_[a+1,a+122,a+121]))
    if hole:
        centre=v[f].mean(axis=1);f=f[~((centre[:,0]>.025)&(centre[:,0]<.035)&(centre[:,1]>.025)&(centre[:,1]<.035))]
    mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),o3d.utility.Vector3iVector(f))
    mesh.remove_unreferenced_vertices();return mesh


def boundary(mesh):
    edges,count=np.unique(_edges(np.asarray(mesh.triangles)),axis=0,return_counts=True)
    points=np.asarray(mesh.vertices)[edges[count==1]]
    return sorted(tuple(sorted(tuple(p) for p in edge)) for edge in points)


class SurfacePatchTests(unittest.TestCase):
    def test_measured_representatives_preserve_outer_and_hole_boundaries(self):
        for hole in (False,True):
            with self.subTest(hole=hole):
                source=grid(hole);vertices=np.asarray(source.vertices).copy()
                result,report=simplify(source)
                self.assertLess(len(result.triangles),len(source.triangles)/3)
                self.assertEqual(boundary(source),boundary(result))
                original={tuple(p) for p in vertices}
                self.assertTrue(all(tuple(p) in original for p in result.vertices))
                self.assertLessEqual(report['max_vertex_displacement_mm'],5.)
                self.assertTrue(result.is_edge_manifold(True))
                np.testing.assert_array_equal(source.vertices,vertices)
                repeated,record=simplify(source)
                np.testing.assert_array_equal(repeated.vertices,result.vertices)
                np.testing.assert_array_equal(repeated.triangles,result.triangles)
                self.assertEqual(record,report)

    def test_nearby_disconnected_layers_are_not_merged(self):
        a=grid();b=grid();b.translate([0,0,.0002]);source=a+b
        result,_=simplify(source)
        _,counts,_=result.cluster_connected_triangles()
        self.assertEqual(len(counts),2)
        self.assertEqual(boundary(source),boundary(result))

    def test_cancel_and_invalid_cell_size(self):
        def cancel():raise InterruptedError('cancel')
        with self.assertRaises(InterruptedError):simplify(grid(),cancel)
        for value in (0,6,float('nan'),True):
            with self.assertRaises(ValueError):simplify(grid(),cell_mm=value)


if __name__=='__main__':unittest.main()
