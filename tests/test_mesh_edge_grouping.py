"""Compare packed grouping against NumPy rows and complete boundary contexts."""
import unittest
from unittest.mock import patch
import numpy as np
from femto.geometry.mesh_edges import unique_edge_rows
from femto.planning.quality_constrained import _mesh_edge_context


class MeshEdgeGroupingTests(unittest.TestCase):
    def test_rows_keep_order_multiplicity_dtype_and_range_fallback(self):
        rng=np.random.default_rng(721)
        cases=[np.empty((0,2),np.int64),np.array([[0,0],[0,0],[0,1],[1,0]]),
               np.array([[0,2**31-1],[2**31-1,0],[0,2**31-1]]),
               np.array([[0,2**31],[2**31,2**31+1],[-1,0]]),
               np.array([[0,2**63],[2**63,0]],np.uint64),
               np.array([[np.nan,1.],[0.,-0.],[0.,0.],[np.nan,1.]]),
               np.array([2,1,2]),np.array([[0,1,3],[0,1,2]]),np.empty((0,3),np.int32)]
        for dtype in (np.int16,np.int32,np.int64,np.uint32,np.uint64,np.dtype('>i8')):
            values=rng.integers(0,500,(4000,2)).astype(dtype)
            cases.extend((values,values[::2,::-1],np.sort(values,axis=1)))
        for rows in cases:
            with self.subTest(dtype=rows.dtype,shape=rows.shape):
                expected=np.unique(rows,axis=0,return_counts=True)
                for a,b in zip(unique_edge_rows(rows),expected):
                    self.assertEqual((a.dtype,a.shape,a.tobytes()),(b.dtype,b.shape,b.tobytes()))

    def test_context_keeps_adjacency_boundary_order_and_failure_reasons(self):
        vertices=np.arange(18,dtype=float).reshape(6,3)
        cases=[np.array([[0,1,2],[0,2,3]]),np.array([[0,0,1],[1,2,3]]),
               np.array([[0,1,2],[0,1,3],[0,1,4]]),
               np.array([[0,1,2],[0,3,1],[1,3,2],[0,2,3]])]
        def outcome(faces,allowed):
            try:return ('values',[(v.dtype.str,v.shape,v.tobytes()) for v in _mesh_edge_context(vertices,faces,allowed)])
            except Exception as e:return (type(e).__name__,str(e))
        for faces in cases:
            for allowed in (None,np.ones(len(faces),bool),np.zeros(len(faces),bool),
                            np.arange(len(faces))%2==0,np.ones(len(faces)+1,bool)):
                actual=outcome(faces,allowed)
                with patch('femto.geometry.mesh_edges.unique_edge_rows',
                           side_effect=lambda edges:np.unique(edges,axis=0,return_counts=True)):
                    expected=outcome(faces,allowed)
                self.assertEqual(actual,expected)


if __name__=='__main__':unittest.main()
