import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from femto.geometry.cloud_radius_batches import NativeRadiusCounter
from test_cloud_csr_research import CellCounterTests,counter


class NativeRadiusTests(CellCounterTests):
    def setUp(self):
        self.binding=patch('test_cloud_csr_research.CellRadiusCounter',NativeRadiusCounter)
        self.binding.start();self.addCleanup(self.binding.stop)

    def test_dense_cell_bounded_buffers_and_cancellation(self):
        rng=np.random.default_rng(1271)
        p=rng.uniform(0,.001,(400,3));n=np.tile([0.,0.,1.],(len(p),1))
        fast,old=counter(p,n,max_pairs=37)
        with patch.object(fast,'search') as search:
            self.assertEqual(fast(np.arange(30)),old(np.arange(30)))
            search.fixed_radius_search.assert_not_called()
        fast.check=lambda:(_ for _ in ()).throw(InterruptedError('cancel'))
        with self.assertRaises(InterruptedError):fast(np.arange(30))

    def test_native_buffers_are_bounded_and_failure_keeps_completed_counts(self):
        rng=np.random.default_rng(1281)
        p=rng.uniform(-.1,.1,(1000,3));n=rng.normal(size=p.shape)
        fast,old=counter(p,n,max_pairs=100)
        native=fast.search;calls=[]
        def partial_failure(*args,**kwargs):
            if len(calls)==2:raise RuntimeError('backend failed')
            values=native.fixed_radius_search(*args,**kwargs)
            calls.append(int(values[0].shape[0]));return values
        fast.search=SimpleNamespace(fixed_radius_search=partial_failure)
        self.assertEqual(fast(np.arange(len(p))),old(np.arange(len(p))))
        self.assertTrue(fast.failed)
        self.assertLessEqual(max(calls),100)
        self.assertEqual(fast(np.arange(len(p))),old(np.arange(len(p))))

    def test_native_threshold_ambiguity_uses_legacy_without_memory_fallback(self):
        # Unlike the inherited max_pairs=1 stress case, all candidates fit and
        # the native predicate itself must recognize threshold ambiguity.
        rng=np.random.default_rng(7103)
        cases=[]
        for boundary in (.004,.01):
            for value in (np.nextafter(boundary,0),boundary,np.nextafter(boundary,np.inf)):
                cases.append((np.array([[0.,0.,0.],[0.,0.,value]]),np.tile([0.,0.,1.],(2,1))))
        for origin in (np.zeros(3),np.array([.51,-.25,1.3]),np.array([1e6,-1e6,1e6])):
            for _ in range(40):
                direction=rng.normal(size=3);direction/=np.linalg.norm(direction)
                cases.append((np.array([origin,origin+direction*.009]),np.tile(direction*(.004/.009),(2,1))))
        native_queries=0;boundary_queries=0
        for points,normals in cases:
            fast,old=counter(points,normals,max_pairs=65536)
            self.assertTrue(fast.enabled)
            with patch.object(fast,'legacy',wraps=old) as legacy,patch.object(fast,'_upper_counts',wraps=fast._upper_counts) as preflight:
                self.assertEqual(fast(np.arange(len(points))),old(np.arange(len(points))))
                native_queries+=preflight.call_count
                boundary_queries+=sum(len(call.args[0]) for call in legacy.call_args_list)
        self.assertEqual(native_queries,len(cases))
        self.assertGreaterEqual(boundary_queries,12)

    def test_custom_numpy_error_policy_stays_on_original_arithmetic(self):
        points=np.array([[0.,0.,0.],[0.,0.,1e-200]])
        normals=np.tile([0.,0.,1e-200],(2,1))
        for mode in ('ignore','raise','warn'):
            with np.errstate(all=mode):
                fast,old=counter(points,normals)
                self.assertFalse(fast.enabled)
                def outcome(function):
                    try:return ('value',function(np.arange(2)))
                    except Exception as error:return (type(error).__name__,str(error))
                self.assertEqual(outcome(fast),outcome(old))


if __name__=='__main__':unittest.main()
