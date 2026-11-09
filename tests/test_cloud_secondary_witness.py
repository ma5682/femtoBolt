"""Complete old predicates remain authoritative after both certificate stages."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import open3d as o3d
from femto.geometry import cloud_layer_batches as batches
from femto.performance_detail import collect


def reference(points,normals):
    cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    tree=o3d.geometry.KDTreeFlann(cloud)
    def labels(indices):
        result=[]
        for index in indices:
            count,ids,_=tree.search_radius_vector_3d(points[index],.01)
            result.append(count>1 and np.any(np.abs((points[np.asarray(ids)]-points[index])@normals[index])>.004))
        return np.asarray(result,bool)
    return labels


def fixtures():
    uv=np.mgrid[-.007:.0071:.0007,-.007:.0071:.0007].reshape(2,-1).T
    plane=np.column_stack((uv,np.zeros(len(uv))))
    for height in (0.,np.nextafter(.004,0.),.004,np.nextafter(.004,np.inf),.0045):
        points=np.concatenate((plane,plane+[0,0,height]))
        yield str(height),points,np.tile([0.,0.,1.],(len(points),1))
    theta=np.pi/4
    rotation=np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
    points=np.concatenate((plane,plane+[0,0,.0045]))@rotation.T
    normals=np.tile([0.,0.,1.],(len(points),1))@rotation.T
    for origin in (np.array([.001575,.001925,.0014]),np.array([2**19,-2**19,2**19])):
        yield str(origin),points+origin,normals
    for radius in (np.nextafter(.01,0.),.01,np.nextafter(.01,np.inf)):
        points=np.zeros((130,3));points[65:,2]=radius
        yield str(radius),points,np.tile([0.,0.,1.],(len(points),1))
    rng=np.random.default_rng(718)
    points=rng.uniform(-.012,.012,(600,3));normals=rng.normal(size=points.shape)
    normals[::5]=0.;normals[1::5]*=3.
    yield 'random nonunit/zero normals',points,normals


class SecondaryWitnessTests(unittest.TestCase):
    def test_certificates_and_final_counts_match_old_predicates(self):
        for name,points,normals in fixtures():
            with self.subTest(name=name):
                old=reference(points,normals);expected=old(range(len(points)))
                negative=batches.negative_bounds(points,normals)
                witness=batches._SecondaryWitness(points,normals)
                positive=np.zeros(len(points),bool)
                for start in range(0,len(points),512):
                    ids=np.arange(start,min(start+512,len(points)))
                    positive[ids]=witness.query(ids)
                self.assertFalse(np.any(negative&expected))
                self.assertFalse(np.any(positive&~expected))
                self.assertEqual(batches.count_dense(points,normals,lambda ids:int(old(ids).sum())),int(expected.sum()))

    def test_uncertified_queries_all_reach_complete_fallback(self):
        # Simulate capped discovery returning only the observed first point.
        # Such a list cannot prove that the other 128 queries are negative.
        points=np.zeros((129,3));points[-1,2]=.005
        normals=np.tile([0.,0.,1.],(len(points),1))
        old=reference(points,normals);seen=[]
        class CappedSearch:
            def __init__(self,*args,**kwargs):pass
            def hybrid_index(self):return True
            def hybrid_search(self,query,**kwargs):
                shape=(query.shape[0],64)
                return (o3d.core.Tensor(np.zeros(shape,np.int64)),
                        o3d.core.Tensor(np.zeros(shape)),
                        o3d.core.Tensor(np.full(shape[0],64,np.int64)))
        def complete(ids):
            seen.extend(ids.tolist());return int(old(ids).sum())
        with patch.object(batches,'negative_bounds',return_value=np.zeros(len(points),bool)),\
             patch.object(batches.o3d.core.nns,'NearestNeighborSearch',CappedSearch),\
             patch.object(batches,'_complete_radius_counter',return_value=complete):
            self.assertEqual(batches.count_dense(points,normals,complete),int(old(range(len(points))).sum()))
        self.assertEqual(seen,list(range(128)))

    def test_native_failures_disable_optional_witness_without_certifying_absence(self):
        points=np.array([[0.,0.,0.],[0.,0.,.005]])
        normals=np.tile([0.,0.,1.],(2,1))
        for error in (RuntimeError('backend'),MemoryError('allocation')):
            with self.subTest(error=type(error).__name__):
                witness=batches._SecondaryWitness(points,normals)
                def fail(*args,**kwargs):raise error
                witness.search=SimpleNamespace(hybrid_search=fail)
                self.assertFalse(witness.query([0,1]).any())
                self.assertTrue(witness.failed)
                self.assertIsNone(witness.search)
                self.assertIsNone(witness.observed)
                self.assertFalse(witness.query([0,1]).any())
        with patch.object(batches.np,'unique',side_effect=MemoryError('allocation')):
            witness=batches._SecondaryWitness(points,normals)
        self.assertFalse(witness.enabled)
        self.assertIsNone(witness.observed)

    def test_conversion_failure_and_custom_error_policies_use_no_certificate(self):
        points=np.array([[0.,0.,0.],[0.,0.,.005]])
        normals=np.tile([0.,0.,1.],(2,1))
        witness=batches._SecondaryWitness(points,normals)
        def fail():raise MemoryError('tensor conversion')
        value=SimpleNamespace(numpy=fail)
        witness.search=SimpleNamespace(hybrid_search=lambda *a,**k:(value,value,value))
        self.assertFalse(witness.query([0,1]).any())
        self.assertTrue(witness.failed)
        for mode in ('ignore','warn','raise'):
            witness=batches._SecondaryWitness(points,normals)
            with np.errstate(all=mode):
                with patch.object(batches.o3d.core.nns,'NearestNeighborSearch',side_effect=AssertionError('unexpected index')):
                    disabled=batches._SecondaryWitness(points,normals)
                self.assertFalse(disabled.enabled)
                self.assertFalse(disabled.query([0,1]).any())
                self.assertFalse(witness.query([0,1]).any())

    def test_cancellation_propagates_and_diagnostics_are_observations(self):
        _,points,normals=list(fixtures())[2]
        old=reference(points,normals)
        for stop in (1,2,3,5):
            calls=[]
            def check():
                calls.append(1)
                if len(calls)==stop:raise InterruptedError('cancel cloud')
            with self.assertRaisesRegex(InterruptedError,'cancel cloud'):
                batches.count_dense(points,normals,lambda ids:int(old(ids).sum()),check=check)
            self.assertEqual(len(calls),stop)
        def cancel():raise InterruptedError('cancel witness')
        with self.assertRaisesRegex(InterruptedError,'cancel witness'):
            batches._SecondaryWitness(points,normals,check=cancel)
        with collect() as detail:
            result=batches.count_dense(points,normals,lambda ids:int(old(ids).sum()))
        self.assertEqual(result,int(old(range(len(points))).sum()))
        rows=detail.snapshot()['rows']
        self.assertIn('cloud.secondary_witness_prepare',rows)
        self.assertIn('cloud.secondary_witness_work',rows)


if __name__=='__main__':unittest.main()
