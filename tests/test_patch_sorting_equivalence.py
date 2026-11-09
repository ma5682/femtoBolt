"""Frozen pre-optimization algorithm versus each refinement round and output."""
import hashlib
import inspect
import sys
import unittest
import numpy as np
import reference_surface_patches_v1 as old
from femto.geometry import surface_patches as new
from test_surface_patches import grid

FIELDS=('labels','keys','inverse','count','centre','distance','representatives',
        'mapped','keep','candidate','coarse_edges','incidence','bad','locked',
        'characteristic','reversed_faces','boundary_keys','expected_keys','displacement',
        'source_boundary','protected','original_side')


def traced(function, mesh, check=lambda:None):
    raw=inspect.unwrap(function)
    source,start=inspect.getsourcelines(raw)
    target=start+next(i for i,line in enumerate(source) if 'if not len(bad):break' in line)
    rounds=[]
    def trace(frame,event,arg):
        if frame.f_code is not raw.__code__:return None
        if event=='line' and frame.f_lineno==target:
            row={}
            for key in FIELDS:
                value=np.asarray(frame.f_locals[key])
                row[key]=(value.dtype.str,value.shape,hashlib.sha256(value.tobytes()).hexdigest())
            rounds.append(row)
        return trace
    previous=sys.gettrace();sys.settrace(trace)
    try:
        result=function(mesh,check)
        error=None
    except Exception as failure:
        result=None;error=(type(failure),str(failure))
    finally:sys.settrace(previous)
    return result,error,rounds


class PatchSortingEquivalenceTests(unittest.TestCase):
    def test_integer_face_keys_preserve_first_indices_order_and_fallbacks(self):
        rng=np.random.default_rng(930)
        cases=[np.empty((0,3),np.int64),
               np.array([[0,0,0],[2**21-3,2**21-2,2**21-1],[1,2,3],[3,1,2]]),
               np.array([[-1,0,1],[1,2,3]]),np.array([[0,1,2**21],[0,1,2**21-1]]),
               np.array([[0,1,2**63+1],[0,1,2**63]],np.uint64),
               np.array([[0.,1.,2.],[2.,0.,1.],[np.nan,1.,2.]])]
        for dtype in (np.int16,np.int32,np.int64,np.uint32,np.uint64,np.dtype('>i8')):
            faces=rng.integers(0,1000,(2000,3)).astype(dtype)
            cases.extend((faces,np.concatenate((faces,faces[:30,::-1])),faces[::2,::-1]))
        for faces in cases:
            with self.subTest(dtype=faces.dtype,shape=faces.shape):
                expected=np.unique(np.sort(faces,axis=1),axis=0,return_index=True)[1]
                actual=new._face_first(faces)
                np.testing.assert_array_equal(actual,expected)
                self.assertEqual(faces[np.sort(actual)].tobytes(),faces[np.sort(expected)].tobytes())

    def test_representatives_keep_ties_singletons_and_nonfinite_fallback(self):
        rng=np.random.default_rng(927)
        for n in (1,3,100,10000):
            _,groups=np.unique(rng.integers(0,max(n//4,1),n),return_inverse=True)
            counts=np.bincount(groups)
            for values in (rng.integers(0,4,n).astype(float),np.full(n,np.inf),np.full(n,np.nan)):
                order=np.lexsort((np.arange(n),values,groups))
                _,first=np.unique(groups[order],return_index=True)
                np.testing.assert_array_equal(new._representatives(values,groups,counts),order[first])

    def test_integer_edges_preserve_order_counts_dtype_and_fallback(self):
        rng=np.random.default_rng(928)
        for dtype in (np.int32,np.int64):
            for faces in (np.empty((0,3),dtype),rng.integers(0,500,(3000,3)).astype(dtype),
                          np.array([[0,2**31-1,10],[10,0,2**31-1]],dtype),np.array([[-1,0,2]],dtype)):
                actual=new._unique_edges(faces);expected=np.unique(old._edges(faces),axis=0,return_counts=True)
                for a,b in zip(actual,expected):
                    self.assertEqual(a.dtype,b.dtype);np.testing.assert_array_equal(a,b)
        faces=np.array([[0,2**31,2**31+2]],np.int64)
        for a,b in zip(new._unique_edges(faces),np.unique(old._edges(faces),axis=0,return_counts=True)):
            np.testing.assert_array_equal(a,b)

    def test_every_refinement_round_and_final_mesh_match(self):
        cases=[grid(),grid(True)]
        curve=grid();v=np.asarray(curve.vertices);v[:,2]+=.003*np.sin(v[:,0]*150);cases.append(curve)
        pair=grid();other=grid();other.translate([0,0,.0002]);cases.append(pair+other)
        degenerate=grid();np.asarray(degenerate.vertices)[:,1]=0.;cases.append(degenerate)
        for index,mesh in enumerate(cases):
            with self.subTest(index=index):
                before=traced(old.simplify,mesh);after=traced(new.simplify,mesh)
                self.assertEqual(before[1:],after[1:])
                if before[0] is not None:
                    a,ar=before[0];b,br=after[0];self.assertEqual(ar,br)
                    for name in ('vertices','triangles','vertex_normals'):
                        self.assertEqual(np.asarray(getattr(a,name)).tobytes(),np.asarray(getattr(b,name)).tobytes())

    def test_cancellation_keeps_the_same_callback_prefix(self):
        mesh=grid(True)
        for stop in (1,2,4,6):
            calls=[]
            for function in (old.simplify,new.simplify):
                seen=[]
                def check():
                    seen.append(1)
                    if len(seen)==stop:raise InterruptedError('cancel refinement')
                result,error,rounds=traced(function,mesh,check)
                calls.append((error,len(seen),rounds))
            self.assertEqual(calls[0],calls[1])
            self.assertEqual(calls[0][0],(InterruptedError,'cancel refinement'))


if __name__=='__main__':unittest.main()
