"""Prepared integer predicates must reproduce the original rational algorithm."""
from fractions import Fraction as Q
import gc
import unittest
from unittest.mock import patch
import weakref

import numpy as np

from femto.geometry import exact_pixel_partition as exact


class RationalPartition(exact.PixelPartition):
    """The original Fraction patch and whole-cell implementation as an oracle."""
    def _patch(self, faces):
        edges={};triangles=[]
        for index,face in enumerate(faces):
            self.check()
            for pair_index,other in enumerate(faces[:index]):
                if pair_index%64==0:self.check()
                if not self.separated(min(face,other),max(face,other)):return None
            triangle=self.projected(face);triangles.append(triangle)
            if exact._cross(*triangle)==0:return None
            for a,b in zip(triangle,triangle[1:]+triangle[:1]):
                edge=(a,b) if a<b else (b,a)
                edges[edge]=edges.get(edge,0)+1
        if any(count>2 for count in edges.values()):return None
        return triangles,tuple(edge for edge,count in edges.items() if count==1)

    def _whole_cell(self, triangles, boundary, x, y):
        centre=(Q(x),Q(y));left,right=Q(2*x-1,2),Q(2*x+1,2)
        top,bottom=Q(2*y-1,2),Q(2*y+1,2)
        for a,b in boundary:
            if (max(a[0],b[0])<=left or min(a[0],b[0])>=right
                    or max(a[1],b[1])<=top or min(a[1],b[1])>=bottom):continue
            radius=(abs(b[0]-a[0])+abs(b[1]-a[1]))/2
            if abs(exact._cross(a,b,centre))<radius:return False
        return any(exact._inside(triangle,centre) for triangle in triangles)


def quad(left=-.5,right=.5,top=-.5,bottom=.5):
    a,b,c,d=([left,top,1.],[right,top,1.],[right,bottom,1.],[left,bottom,1.])
    return np.array([[a,b,c],[a,c,d]])


class PreparedPartitionTests(unittest.TestCase):
    def test_partition_caches_release_owned_geometry_without_cyclic_collection(self):
        enabled = gc.isenabled()
        gc.disable()
        try:
            for _ in range(4):
                partition = exact.PixelPartition(quad(), np.eye(3))
                self.assertEqual(partition.classify((0, 1), 0, 0), 1)
                owner, triangles = weakref.ref(partition), weakref.ref(partition.triangles)
                del partition
                self.assertIsNone(owner(), 'Construction cache must not keep its owner alive')
                self.assertIsNone(triangles(), 'Owned geometry must release with its construction')
        finally:
            if enabled:
                gc.enable()

    def test_retained_cache_does_not_keep_partition_alive(self):
        enabled = gc.isenabled()
        gc.disable()
        try:
            partition = exact.PixelPartition(quad(), np.eye(3))
            partition.classify((0, 1), 0, 0)
            cache = partition.prepared
            owner, triangles = weakref.ref(partition), weakref.ref(partition.triangles)
            del partition
            self.assertIsNone(owner())
            self.assertIsNone(triangles())
            # Existing immutable cached values remain usable without retaining
            # the construction; a miss cannot access released source geometry.
            self.assertEqual(len(cache(0)[0]), 3)
            cache.cache_clear()
            with self.assertRaises(ReferenceError):
                cache(0)
        finally:
            if enabled:
                gc.enable()

    def test_edge_predicates_keep_exact_signs_and_open_boundary_at_all_scales(self):
        rng=np.random.default_rng(20261020)
        edges=[((Q(1,2),Q(-1,2)),(Q(1,2),Q(1,2))),
               ((Q(-1,2),Q(-1,2)),(Q(1,2),Q(1,2))),
               ((Q(1,2)-Q(1,2**1000),Q(-1)),(Q(1,2),Q(1)))]
        for scale in (Q(1,2**1000),Q(1,7),Q(1),Q(10**100)):
            for _ in range(20):
                points=rng.integers(-20,21,size=(2,2))
                edges.append(tuple(tuple(Q(int(v))*scale for v in point) for point in points))
        for a,b in edges:
            right,left,bottom,top,aa,bb,cc=exact._cell_edge(a,b)
            for x,y in ((-2,-1),(-1,0),(0,0),(0,1),(1,0),(2,3)):
                with self.subTest(a=a,b=b,x=x,y=y):
                    cross=exact._cross(a,b,(Q(x),Q(y)))
                    value=aa*x+bb*y+cc
                    self.assertEqual((cross>0)-(cross<0),(value>0)-(value<0))
                    self.assertEqual(abs(cross)<(abs(b[0]-a[0])+abs(b[1]-a[1]))/2,
                                     2*abs(value)<abs(aa)+abs(bb))
                    self.assertEqual(
                        max(a[0],b[0])<=Q(2*x-1,2) or min(a[0],b[0])>=Q(2*x+1,2)
                        or max(a[1],b[1])<=Q(2*y-1,2) or min(a[1],b[1])>=Q(2*y+1,2),
                        x>=right or x<=left or y>=bottom or y<=top)

    def test_partition_and_shortcut_match_oracle_for_gaps_overlap_and_perspective(self):
        rng=np.random.default_rng(20261020)
        cases=[quad(),quad(right=.5-2**-48),quad(right=.5+2**-48),
               quad(left=-2.,right=2.,top=-2.,bottom=2.),
               np.concatenate([quad(right=.125),quad(left=-.125,right=.25)]),
               np.array([[[0.,0.,1.],[0.,0.,1.],[1.,1.,1.]]])]
        for index in range(60):
            left,top=rng.uniform(-2,0,2);right,bottom=rng.uniform(0,2,2)
            triangles=quad(left,right,top,bottom)
            triangles[:,:,2]=1.+.1*triangles[:,:,0]+.05*triangles[:,:,1]
            cases.append(triangles if index%2 else triangles[:,::-1])
        for intrinsic in (np.eye(3),np.array([[1.7,0,.2],[0,1.3,-.4],[0,0,1.]])):
            for triangles in cases:
                actual=exact.PixelPartition(triangles,intrinsic)
                expected=RationalPartition(triangles,intrinsic)
                faces=tuple(range(len(triangles)))
                prepared,original=actual.patch(faces),expected.patch(faces)
                self.assertEqual(prepared is None,original is None)
                for x in range(-2,3):
                    for y in range(-2,3):
                        if prepared is not None:
                            self.assertEqual(actual._whole_cell(*prepared,x,y),
                                             expected._whole_cell(*original,x,y))
                        self.assertEqual(actual.classify(faces,x,y),expected.classify(faces,x,y))

    def test_dense_patch_bypasses_cache_and_cancellation_remains_checked(self):
        triangles=np.concatenate([quad(left=i,right=i+1,top=0,bottom=1) for i in range(9)])
        calls=[];cancelled=[]
        def check():
            calls.append(None)
            if cancelled:raise RuntimeError('cancelled')
        actual=exact.PixelPartition(triangles,np.eye(3),check)
        expected=RationalPartition(triangles,np.eye(3))
        faces=list(range(len(triangles)))
        for x,y in ((0,0),(2,1),(8,0)):
            self.assertEqual(actual.classify(faces,x,y),expected.classify(faces,x,y))
        self.assertEqual(actual.patch.cache_info().currsize,0)
        self.assertEqual(actual.patch.cache_info().maxsize,4096)
        self.assertGreater(len(calls),len(faces))
        actual.classify([0,1],0,0)
        cancelled.append(True)
        with self.assertRaisesRegex(RuntimeError,'cancelled'):actual.classify([0,1],0,0)

    def test_overlapping_patches_share_bounded_face_preparation_without_cross_instance_reuse(self):
        triangles=quad()
        actual=exact.PixelPartition(triangles,np.eye(3))
        with patch.object(exact,'_integer_line',wraps=exact._integer_line) as prepare:
            first=actual.patch((0,))
            combined=actual.patch((0,1))
            reverse=actual.patch((1,0))
        self.assertEqual(prepare.call_count,6)  # Three edges for each source face.
        self.assertIs(first[0][0],combined[0][0])
        self.assertIs(first[0][0],reverse[0][1])
        self.assertEqual(actual.prepared.cache_info().currsize,2)
        self.assertEqual(actual.prepared.cache_info().maxsize,4096)
        for line,boundary in zip(*actual.prepared(0)):
            self.assertTrue(all(a is b for a,b in zip(line,boundary[4:])))
        triangles[:,:,0]*=.5
        # This instance retains its owned immutable geometry; a new one must
        # prepare the changed source and continue reporting the real gap.
        self.assertEqual(actual.classify([0,1],0,0),1)
        changed=exact.PixelPartition(triangles,np.eye(3))
        self.assertEqual(changed.classify([0,1],0,0),10)
        self.assertIsNot(changed.prepared(0),actual.prepared(0))

    def test_target_replay_keeps_metadata_array_bytes_and_budget_frontiers(self):
        import test_target_surface as fixtures
        from femto.planning import planning_revision as revisions,target_surface as target
        from femto.planning.quality_constrained import evidence
        case=fixtures.TargetSurfaceTests();case.setUp();self.addCleanup(case.doCleanups)
        mask=np.zeros((100,100),bool);mask[47:54,47:54]=True;mask[50,50]=False
        model,region=case.fixture(mask=mask)
        revision=revisions.capture(model,region,case.head,
            algorithms=dict(revisions.ALGORITHMS,target=revisions.EXACT_TARGET))
        quality=evidence(model,region,allow_empty=True)
        for budget in (target.TargetBudget(),target.TargetBudget(max_units=7),
                       target.TargetBudget(max_pair_checks=3)):
            with self.subTest(budget=budget):
                actual=target.build(model,region,revision,evidence=quality,budget=budget)
                with patch.object(exact,'PixelPartition',RationalPartition):
                    expected=target.build(model,region,revision,evidence=quality,budget=budget)
                self.assertEqual(actual.metadata,expected.metadata)
                for key in actual.arrays:
                    self.assertEqual(actual.arrays[key].tobytes(),expected.arrays[key].tobytes(),key)
                target.validate(actual,revision,evidence=quality,domain=None)


if __name__=='__main__':unittest.main()
