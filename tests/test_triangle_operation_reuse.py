from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
import unittest
from unittest.mock import patch
import numpy as np
from femto.geometry import triangle_support
from femto.geometry.region_reuse import operation
from femto.planning import quality_constrained


class TriangleOperationReuseTests(unittest.TestCase):
    def setUp(self):
        self.vertices=np.array([[0.,0.,.5],[.02,0.,.5],[0.,.02,.5],[.02,.02,.5]])
        self.faces=np.array([[0,1,2],[1,3,2]])

    def test_reuses_only_geometry_and_keeps_independent_solvers(self):
        original=triangle_support._prepare_one
        with patch.object(triangle_support,'_prepare_one',wraps=original) as make:
            with operation():
                v1,f1,p1=triangle_support.prepare_triangle_faces(self.vertices,self.faces)
                first=p1(0)
                with operation():
                    v2,f2,p2=triangle_support.prepare_triangle_faces(self.vertices,self.faces)
                    second=p2(0)
                self.assertEqual(make.call_count,1)
                self.assertIsNot(first.solver,second.solver)
                first.solver._ready=True;first.solver._inverse=None
                self.assertFalse(second.solver._ready)
                for array in (v1,f1,v2,f2,first.origin,first.ab,first.matrix,second.normal):
                    with self.assertRaises(ValueError):array.setflags(write=True)
                np.testing.assert_array_equal(first.matrix,second.matrix)
            with operation():
                _,_,fresh=triangle_support.prepare_triangle_faces(self.vertices,self.faces);fresh(0)
            self.assertEqual(make.call_count,2)

    def test_input_mutation_and_exception_do_not_reuse_stale_snapshot(self):
        original=triangle_support._prepare_one
        with patch.object(triangle_support,'_prepare_one',wraps=original) as make:
            with self.assertRaisesRegex(RuntimeError,'cancelled'):
                with operation():
                    _,_,old=triangle_support.prepare_triangle_faces(self.vertices,self.faces)
                    old_point=old(0).origin.copy()
                    self.vertices[:,2]+=1.
                    _,_,new=triangle_support.prepare_triangle_faces(self.vertices,self.faces)
                    self.assertEqual(new(0).origin[2],1.5)
                    self.assertEqual(old_point[2],.5)
                    raise RuntimeError('cancelled')
            with operation():
                _,_,after=triangle_support.prepare_triangle_faces(self.vertices,self.faces);after(0)
            self.assertEqual(make.call_count,3)

    def test_cached_and_uncached_geometry_and_chord_decisions_match(self):
        rng=np.random.default_rng(451)
        for dtype in (np.float32,np.float64):
            vertices=self.vertices.astype(dtype)
            v0,f0,p0=triangle_support.prepare_triangle_faces(vertices,self.faces)
            with operation():
                v1,f1,p1=triangle_support.prepare_triangle_faces(vertices,self.faces)
                for index in range(len(self.faces)):
                    a,b=p0(index),p1(index)
                    for name in ('origin','ab','ac','normal','matrix'):
                        np.testing.assert_array_equal(getattr(a,name),getattr(b,name))
                        self.assertEqual(getattr(a,name).tobytes(),getattr(b,name).tobytes())
                    self.assertEqual(a.normal_length,b.normal_length)
                    self.assertEqual(a.minimum_edge_length,b.minimum_edge_length)
                for _ in range(100):
                    endpoints=np.c_[rng.uniform(-.005,.025,(2,2)),np.full(2,.5)]
                    endpoints[:,2]+=rng.choice([0.,1e-6,np.nextafter(1e-6,np.inf)],2)
                    self.assertEqual(quality_constrained._supported_chord(*endpoints,f0,v0,[0,1],_prepared_faces=p0),
                                     quality_constrained._supported_chord(*endpoints,f1,v1,[0,1],_prepared_faces=p1))

    def test_other_thread_does_not_inherit_operation_cache(self):
        original=triangle_support._prepare_one
        with patch.object(triangle_support,'_prepare_one',wraps=original) as make,operation():
            _,_,prepared=triangle_support.prepare_triangle_faces(self.vertices,self.faces);prepared(0)
            def independent():
                _,_,other=triangle_support.prepare_triangle_faces(self.vertices,self.faces);other(0)
            inherited=copy_context()
            with ThreadPoolExecutor(max_workers=1) as pool:pool.submit(inherited.run,independent).result()
            self.assertEqual(make.call_count,2)

    def test_degenerate_and_scaled_face_preparation_is_bit_identical(self):
        rng=np.random.default_rng(20261030)
        triangles=[np.zeros((3,3)),np.array([[0.,0.,0.],[1.,0.,0.],[2.,0.,0.]]),
                   np.array([[0.,0.,0.],[1.,0.,0.],[1.,1e-14,0.]])]
        triangles.extend(rng.normal(size=(3,3))*scale for scale in (1e-9,1e-5,1.,1e5) for _ in range(8))
        for dtype in (np.float32,np.float64):
            for triangle in triangles:
                points=triangle.astype(dtype);_,_,baseline=triangle_support.prepare_triangle_faces(points,[[0,1,2]])
                with operation():
                    _,_,first=triangle_support.prepare_triangle_faces(points,[[0,1,2]])
                    _,_,second=triangle_support.prepare_triangle_faces(points,[[0,1,2]])
                    expected=baseline(0)
                    for prepared in (first(0),second(0)):
                        for field in ('origin','ab','ac','normal','matrix'):
                            self.assertEqual(getattr(expected,field).tobytes(),getattr(prepared,field).tobytes())
                        self.assertEqual(expected.normal_length,prepared.normal_length)
                        self.assertEqual(expected.minimum_edge_length,prepared.minimum_edge_length)

    def test_oversized_or_unsupported_snapshots_fall_back_to_original(self):
        original=triangle_support._prepare_one
        with patch.object(triangle_support,'_prepare_one',wraps=original) as make, \
             patch.object(triangle_support,'_MAX_SNAPSHOT_BYTES',1),operation():
            for _ in range(2):
                _,_,prepared=triangle_support.prepare_triangle_faces(self.vertices,self.faces);prepared(0)
            self.assertEqual(make.call_count,2)
        with operation():
            vertices,_,_=triangle_support.prepare_triangle_faces(self.vertices.astype(object),self.faces)
            self.assertEqual(vertices.dtype,object)
            self.assertEqual(triangle_support._prepared_operation_cache.get()['entries'],{})

    def test_inherited_context_cannot_reuse_after_operation_exits(self):
        original=triangle_support._prepare_one
        with patch.object(triangle_support,'_prepare_one',wraps=original) as make:
            with operation():
                _,_,prepared=triangle_support.prepare_triangle_faces(self.vertices,self.faces);prepared(0)
                inherited=copy_context()
            def outside():
                for _ in range(2):
                    _,_,prepared=triangle_support.prepare_triangle_faces(self.vertices,self.faces);prepared(0)
            inherited.run(outside)
            self.assertEqual(make.call_count,3)


if __name__=='__main__':unittest.main()
