"""Exact partitions must keep real microscopic gaps and overlap ambiguous."""
import unittest
import numpy as np
from femto.geometry.exact_pixel_partition import PixelPartition


def quad(left=-.5,right=.5,top=-.5,bottom=.5):
    a,b,c,d=([left,top,1.],[right,top,1.],[right,bottom,1.],[left,bottom,1.])
    return np.array([[a,b,c],[a,c,d]])


class ExactPixelPartitionTests(unittest.TestCase):
    def test_shared_diagonal_and_reversed_faces_partition_exactly(self):
        for triangles in (quad(),quad()[:,::-1],quad(left=-1.,right=1.,top=-1.,bottom=1.)):
            partition=PixelPartition(triangles,np.eye(3))
            self.assertEqual(partition.classify([0,1],0,0),1)

    def test_gap_far_smaller_than_old_area_tolerance_remains_partial(self):
        partition=PixelPartition(quad(right=.5-2**-48),np.eye(3))
        self.assertEqual(partition.classify([0,1],0,0),10)

    def test_equal_total_area_with_overlap_and_missing_strip_is_not_full(self):
        triangles=np.concatenate([quad(right=.125),quad(left=-.125,right=.25)])
        self.assertEqual(PixelPartition(triangles,np.eye(3)).classify(list(range(4)),0,0),8)

    def test_nonbinary_calibration_and_perspective_shared_vertices(self):
        intrinsic=np.array([[331.7,0,103.2],[0,329.1,97.4],[0,0,1.]])
        vertices=np.array([[-.2,-.2,.49],[.2,-.2,.51],[.2,.2,.53],[-.2,.2,.51]])
        partition=PixelPartition(vertices[[[0,1,2],[0,2,3]]],intrinsic)
        self.assertEqual(partition.classify([0,1],103,97),1)

    def test_cache_is_bounded_and_detached_from_mutable_input(self):
        triangles=quad();partition=PixelPartition(triangles,np.eye(3))
        triangles[:]=0
        self.assertEqual(partition.classify([0,1],0,0),1)
        self.assertEqual(partition.projected.cache_info().maxsize,4096)
        self.assertEqual(partition.separated.cache_info().maxsize,4096)
        with self.assertRaises(ValueError):partition.triangles[0]=0

    def test_cancel_is_checked_even_when_source_geometry_is_cached(self):
        cancelled=[]
        def check():
            if cancelled:raise RuntimeError('cancelled')
        partition=PixelPartition(quad(),np.eye(3),check)
        self.assertEqual(partition.classify([0,1],0,0),1)
        cancelled.append(True)
        with self.assertRaisesRegex(RuntimeError,'cancelled'):partition.classify([0,1],0,0)

    def test_boundary_shortcut_matches_exact_clipped_areas_on_perspective_patches(self):
        from femto.geometry.exact_pixel_partition import _area,_clip
        from fractions import Fraction as Q
        rng=np.random.default_rng(20261015)
        for _ in range(120):
            left,top=rng.uniform(-2,0,2);right,bottom=rng.uniform(0,2,2)
            triangles=quad(left,right,top,bottom)
            # Shared vertices keep identical depths under a perspective tilt.
            triangles[:,:,2]=1.+.1*triangles[:,:,0]+.05*triangles[:,:,1]
            partition=PixelPartition(triangles,np.eye(3))
            for x,y in ((0,0),(-1,0),(0,1),(1,-1)):
                size=sum((_area(_clip(list(partition.projected(f)),x,y)) for f in (0,1)),Q(0))
                expected=1 if size==1 else 10 if 0<size<1 else 8
                self.assertEqual(partition.classify([0,1],x,y),expected)

    def test_partition_cannot_override_reference_depth_gate_or_real_mesh_hole(self):
        import open3d as o3d
        from test_target_surface import TargetSurfaceTests
        from femto.planning import planning_revision as revisions,target_surface as target
        from femto.planning.quality_constrained import evidence
        case=TargetSurfaceTests();case.setUp();self.addCleanup(case.doCleanups)
        for kind in ('missing','conflict','partial'):
            with self.subTest(kind=kind):
                depth=np.full((100,100),500,np.uint16)
                if kind!='partial':depth[50,50]=0 if kind=='missing' else 480
                mesh=None
                if kind=='partial':
                    mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(
                        [[-.0025,-.0025,.5],[.0025,-.0025,.5],[-.0025,.0025,.5]]),
                        o3d.utility.Vector3iVector([[0,1,2]]))
                model,region=case.fixture(mesh=mesh,depth=depth)
                revision=revisions.capture(model,region,case.head,
                    algorithms=dict(revisions.ALGORITHMS,target=revisions.EXACT_TARGET))
                quality=evidence(model,region,allow_empty=True)
                ledger=target.build(model,region,revision,evidence=quality)
                self.assertEqual(ledger.arrays['pixel_status'][50,50],dict(missing=2,conflict=6,partial=10)[kind])
                self.assertIsNone(ledger.summary['total_target_area_m2'])
                target.validate(ledger,revision,evidence=quality,domain=None)


if __name__=='__main__':unittest.main()
