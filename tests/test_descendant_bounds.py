"""Directed child boxes checked against independent rational midpoint splits."""
from fractions import Fraction as Q
import math
import unittest

import numpy as np

from femto.geometry.exact_surface import descendant_boxes,descendant_vertex_bounds,contains_ball_pairs
from femto.planning.coverage_residual import _outside_pairs


def child_reference(triangle, depth, code):
    for shift in range(depth-1, -1, -1):
        a, b, c = triangle
        ab = tuple((x+y)/2 for x,y in zip(a,b))
        bc = tuple((x+y)/2 for x,y in zip(b,c))
        ca = tuple((x+y)/2 for x,y in zip(c,a))
        triangle = ((a,ab,ca),(ab,b,bc),(ca,bc,c),(ab,bc,ca))[(code>>(2*shift))&3]
    return triangle


class DescendantBoundsTests(unittest.TestCase):
    def check_bounds(self, triangle, addresses):
        rational = tuple(tuple(Q(float(v)) for v in row) for row in triangle)
        roots = np.repeat(np.asarray(triangle, float)[None,:,:],len(addresses),axis=0)
        depth = np.asarray([d for d,_ in addresses],np.int64)
        code = np.asarray([c for _,c in addresses],np.int64)
        low, high = descendant_boxes(roots,roots,depth,code)
        for row, (d,c) in enumerate(addresses):
            for point in child_reference(rational,d,c):
                for axis, value in enumerate(point):
                    if math.isfinite(low[row,axis]):self.assertLessEqual(Q(low[row,axis]),value)
                    if math.isfinite(high[row,axis]):self.assertGreaterEqual(Q(high[row,axis]),value)
        return low, high

    def test_all_addresses_through_depth_four_enclose_exact_vertices(self):
        addresses = [(d,c) for d in range(5) for c in range(4**d)]
        self.check_bounds([[.1,-.3,.5],[1.25,.6,.5],[-.7,1.1,.51]],addresses)

    def test_depth_eight_random_scales_signs_and_cancellation(self):
        random = np.random.default_rng(271828)
        addresses = [(8,int(v)) for v in random.integers(0,4**8,128)]
        for exponent in (-1070,-520,-10,10,500,1000):
            triangle = np.ldexp(random.uniform(-1,1,(3,3)),exponent)
            with self.subTest(exponent=exponent):self.check_bounds(triangle,addresses)

    def test_each_vertex_interval_uses_the_matching_exact_dyadic_vertex(self):
        triangle=np.array([[.13,-.27,.511],[.091,.223,.592],[-.131,.147,.482]])
        addresses=[(d,c) for d in range(5) for c in range(4**d)]
        roots=np.repeat(triangle[None,:,:],len(addresses),axis=0)
        low,high=descendant_vertex_bounds(roots,roots,np.array([d for d,_ in addresses]),np.array([c for _,c in addresses]))
        rational=tuple(tuple(Q(float(x)) for x in p) for p in triangle)
        for index,(depth,code) in enumerate(addresses):
            for vertex,point in enumerate(child_reference(rational,depth,code)):
                for axis,value in enumerate(point):
                    self.assertLessEqual(Q(low[index,vertex,axis]),value)
                    self.assertGreaterEqual(Q(high[index,vertex,axis]),value)

    def test_ambiguous_vertex_containment_materializes_exact_geometry_only_when_needed(self):
        from femto.planning.coverage_residual import _CurrentTriangles
        triangle=np.array([[.125,0,0],[0,0,0],[0,.0625,0]])
        bounds=descendant_vertex_bounds(triangle[None],triangle[None],np.array([0]),np.array([0]))
        built=[]
        def geometry(number):
            built.append(number)
            return tuple(tuple(Q(float(x)) for x in p) for p in triangle),None
        for radius,calls in ((.5,0),(.125,1)):
            built.clear()
            inside,fallbacks=contains_ball_pairs(_CurrentTriangles(geometry,[17]),((Q(0),Q(0),Q(0)),),[radius],
                triangle_bounds=bounds,center_bounds=(np.zeros((1,3)),np.zeros((1,3))))
            self.assertTrue(inside[0]);self.assertEqual(fallbacks,calls);self.assertEqual(len(built),calls)

    def test_nonzero_input_intervals_enclose_the_entire_root_uncertainty(self):
        low = np.asarray([[[-2.,-.1,.2],[.3,-.4,.7],[.8,.9,1.]]])
        high = np.nextafter(low+1.e-9,np.inf)
        result_low, result_high = descendant_boxes(low,high,np.array([7]),np.array([9381]))
        for endpoint in (low[0],high[0]):
            reference = child_reference(tuple(tuple(Q(float(x)) for x in p) for p in endpoint),7,9381)
            for p in reference:
                for axis,v in enumerate(p):
                    self.assertLessEqual(Q(result_low[0,axis]),v)
                    self.assertGreaterEqual(Q(result_high[0,axis]),v)

    def test_ambiguous_infinite_arithmetic_falls_back_to_whole_root(self):
        low = np.full((1,3,3),-np.inf); high=np.full((1,3,3),np.inf)
        actual=descendant_boxes(low,high,np.array([4]),np.array([80]))
        np.testing.assert_array_equal(actual[0],low.min(axis=1))
        np.testing.assert_array_equal(actual[1],high.max(axis=1))
        self.assertFalse(_outside_pairs(actual[0],actual[1],np.zeros((1,3)),np.zeros((1,3)),np.array([1.])).any())

    def test_empty_input_has_no_phantom_box(self):
        low,high=descendant_boxes(np.empty((0,3,3)),np.empty((0,3,3)),np.array([],int),np.array([],int))
        self.assertEqual(low.shape,(0,3));self.assertEqual(high.shape,(0,3))

    def test_bad_addresses_or_bounds_are_rejected(self):
        root=np.zeros((1,3,3))
        for depth,code in ((9,0),(-1,0),(1,-1),(1,4),(1.,0),(1,0.)):
            with self.subTest(depth=depth,code=code),self.assertRaises(ValueError):
                descendant_boxes(root,root,np.array([depth]),np.array([code]))
        for low,high in ((root+1,root),(root+np.nan,root),(root,root+np.nan)):
            with self.assertRaises(ValueError):descendant_boxes(low,high,np.array([0]),np.array([0]))

    def test_tangent_and_one_ulp_neighbors_are_never_falsely_excluded(self):
        for radius in (.125,2.**-500,2.**500):
            for x in (math.nextafter(radius,0),radius,math.nextafter(radius,np.inf),radius*1.01):
                low=np.array([[x,0.,0.]])
                excluded=_outside_pairs(low,low,np.zeros((1,3)),np.zeros((1,3)),np.array([radius]))[0,0]
                if excluded:self.assertGreater(Q(x)**2,Q(radius)**2)
                if x<=radius:self.assertFalse(excluded)

    def test_random_outside_pairs_are_sufficient_exact_distance_bounds(self):
        random=np.random.default_rng(314159)
        low=random.uniform(-2,2,(64,3));high=low+random.uniform(0,.2,(64,3))
        center=random.uniform(-2,2,(23,3));radii=random.uniform(.01,2,23)
        excluded=_outside_pairs(low,high,center,center,radii)
        self.assertTrue(excluded.any());self.assertFalse(excluded.all())
        for row,seed in np.argwhere(excluded):
            squared=sum(max(Q(0),Q(low[row,a])-Q(center[seed,a]),Q(center[seed,a])-Q(high[row,a]))**2 for a in range(3))
            self.assertGreater(squared,Q(radii[seed])**2)


if __name__=='__main__':unittest.main()
