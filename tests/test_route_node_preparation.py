"""Boundary traversal budgets and pixel rounding remain independent of fast paths."""
import unittest
import warnings
from unittest.mock import patch
import numpy as np
from femto.planning import boundary_clearance as boundary, quality_constrained as quality
from test_route_candidate_reuse import original_rectangle


class RouteNodePreparationTests(unittest.TestCase):
    def test_half_integer_projection_bounds_keep_both_sides_and_cached_radius(self):
        k=np.eye(3)
        for base in (-1000.5,-1.5,-.5,.5,1.5,1000.5):
            for x in (np.nextafter(base,-np.inf),base,np.nextafter(base,np.inf)):
                for y in (-0.,x,-x):
                    mid=np.array([x,y,1.]);cache=[]
                    for radius in (0.,np.nextafter(.001,0.),.001,np.nextafter(.001,np.inf)):
                        self.assertEqual(quality._rectangle(mid,radius,k,_projection=cache),
                                         original_rectangle(mid,radius,k))

    def test_unusual_ranges_dtypes_and_error_policies_preserve_numpy_outcome(self):
        def outcome(fn,mid,radius,k):
            with warnings.catch_warnings(record=True) as records:
                warnings.simplefilter('always')
                try:value=('return',fn(mid,radius,k))
                except Exception as error:value=(type(error).__name__,str(error))
            return value,[(r.category.__name__,str(r.message)) for r in records]
        for dtype in (np.float32,np.float64):
            k=np.eye(3,dtype=dtype)
            for xyz,radius in (([.5,-0.,1.],.01),([2**31,0.,1.],0.),
                               ([float('nan'),0.,1.],.01),([float('inf'),0.,1.],0.),
                               ([.01,.02,1.],1.),([1e-25,0.,1.],0.)):
                mid=np.array(xyz,dtype=dtype)
                for policy in ('warn','raise','ignore'):
                    with np.errstate(all=policy):
                        self.assertEqual(outcome(quality._rectangle,mid,radius,k),
                                         outcome(original_rectangle,mid,radius,k))

    def test_extended_scalar_radius_retains_numpy_type_promotion(self):
        if np.longdouble is np.float64:self.skipTest('no distinct extended scalar type')
        mid=np.array([.125,-.25,1.]);k=np.eye(3);radius=np.longdouble('.001')
        expected=original_rectangle(mid,radius,k)
        self.assertEqual(quality._rectangle(mid,radius,k),expected)
        # Windows may expose a distinct scalar class with binary64 precision;
        # only a genuinely wider result must bypass the scalar integer bounds.
        if np.finfo(np.longdouble).nmant > np.finfo(np.float64).nmant:
            with patch.object(quality.math,'floor',side_effect=AssertionError('extended scalar entered binary64 fast path')):
                self.assertEqual(quality._rectangle(mid,radius,k),expected)

    def test_leaf_incremental_pruning_keeps_four_distance_charges_not_six(self):
        edges=np.array([[[-.05,1.,0.],[.05,1.,0.]],
                        [[-100.,.1,0.],[1.,.1,0.]],
                        [[-.1,.5,0.],[.1,.5,0.]]])
        points=np.array([[0.,0.,0.],[.01,0.,0.]])
        query=boundary.prepare_nearest_boundary_distances(edges)
        calls=[];original=boundary.point_segment_distance
        def distance(p,a,b):
            calls.append((len(p),a.tolist(),b.tolist()));return original(p,a,b)
        with patch.object(boundary,'MAX_DISTANCE_CHECKS',4),patch.object(boundary,'point_segment_distance',distance):
            np.testing.assert_array_equal(query(points),[.1,.1])
        self.assertEqual(calls,[(2,*edges[2].tolist()),(2,*edges[1].tolist())])
        with patch.object(boundary,'MAX_DISTANCE_CHECKS',3),self.assertRaisesRegex(ValueError,'超过预算'):
            query(points)

    def test_scalar_singletons_match_numpy_fallback_distances_visit_order_and_budget(self):
        rng=np.random.default_rng(1546)
        for scale in (1e-10,1.,1e6,1e7):
            edges=rng.normal(size=(37,2,3))*scale
            edges[1]=edges[0];edges[2,1]=edges[2,0]
            points=np.vstack((rng.normal(size=(12,3))*scale,edges[:4,0],np.nextafter(edges[:4,0],np.inf)))
            for point in points:
                histories=[];answers=[];checks=[]
                for policy in ('ignore','warn'):
                    trace=[];ticks=[0];original=boundary.point_segment_distance
                    def check():ticks[0]+=1
                    def distance(p,a,b):
                        trace.append((p.tobytes(),a.tobytes(),b.tobytes()));return original(p,a,b)
                    with np.errstate(under=policy),patch.object(boundary,'point_segment_distance',distance):
                        answers.append(boundary.prepare_nearest_boundary_distances(edges,check)([point]))
                    histories.append(trace);checks.append(ticks[0])
                np.testing.assert_array_equal(*answers);self.assertEqual(*histories);self.assertEqual(*checks)
                count=len(histories[0]);query=boundary.prepare_nearest_boundary_distances(edges)
                with patch.object(boundary,'MAX_DISTANCE_CHECKS',count):
                    np.testing.assert_array_equal(query([point]),answers[0])
                with patch.object(boundary,'MAX_DISTANCE_CHECKS',count-1),self.assertRaisesRegex(ValueError,'超过预算'):
                    query([point])

    def test_cancelled_singleton_and_batch_can_retry_without_reusing_distances(self):
        rng=np.random.default_rng(711);edges=rng.normal(size=(67,2,3));points=rng.normal(size=(48,3))
        state=[0,None]
        def check():
            state[0]+=1
            if state[1] is not None and state[0]>=state[1]:raise RuntimeError('cancelled')
        query=boundary.prepare_nearest_boundary_distances(edges,check)
        for values in (points[:1],points):
            state[:]=[0,None];expected=query(values)
            state[:]=[0,3]
            with self.assertRaisesRegex(RuntimeError,'cancelled'):query(values)
            state[:]=[0,None];np.testing.assert_array_equal(query(values),expected)


if __name__=='__main__':unittest.main()
