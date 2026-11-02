"""Preparation changes preserve numerics, query budgets and source ownership."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np
import open3d as o3d

from femto.local import local_pose_refinement as refinement
from femto.geometry.triangle_separation import _three_extrema
from femto.planning import boundary_clearance as boundary, target_surface as target


class EfficiencyPreparationTests(unittest.TestCase):
    def test_extrema_keep_signed_zero_nonfinite_and_three_value_reductions(self):
        rng=np.random.default_rng(98)
        cases=list(rng.normal(size=(1000,3)))
        cases += [np.array(v) for v in [(0.,-0.,0.),(-0.,0.,-0.),(np.inf,1.,2.),
                                        (1.,np.nan,2.),(-np.inf,np.inf,1.)]]
        for values in cases:
            actual=_three_extrema(values)
            expected=(values.max(),values.min(),float(np.abs(values).max()))
            np.testing.assert_array_equal(actual,expected)
            np.testing.assert_array_equal(np.signbit(actual),np.signbit(expected))

    def test_prepared_reference_solver_and_kd_tree_match_ordinary_calls(self):
        rng=np.random.default_rng(12);points=rng.uniform(-.1,.1,(1800,3));points[:,2]+=.6
        fixed=o3d.geometry.PointCloud();fixed.points=o3d.utility.Vector3dVector(points)
        fixed.colors=o3d.utility.Vector3dVector(rng.random((len(points),3)))
        moving=deepcopy(fixed);moving.translate([.002,0,0])
        original=np.asarray(fixed.points).copy();scales={}
        for voxel in (.012,.006,.003):
            cloud=fixed.voxel_down_sample(voxel)
            cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*3,max_nn=40))
            scales[voxel]=cloud
        for source,dest,is_target in ((moving,fixed,True),(fixed,moving,False)):
            expected=refinement.solve(source,dest,np.eye(4))
            actual=refinement.solve(source,dest,np.eye(4),_reference_scales=scales,_reference_is_target=is_target)
            np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-12)
            self.assertEqual(refinement.metrics(source,dest,actual),refinement.metrics(
                source,dest,actual,_target_tree=o3d.geometry.KDTreeFlann(dest)))
        np.testing.assert_array_equal(np.asarray(fixed.points),original)

    def test_boundary_singleton_and_empty_queries_match_fresh_index(self):
        rng=np.random.default_rng(93);edges=rng.normal(size=(100,2,3))
        prepared=boundary.prepare_nearest_boundary_distances(edges)
        snapshot=edges.copy()
        for count in (0,1,2,31):
            points=rng.normal(size=(count,3))
            np.testing.assert_array_equal(prepared(points),boundary.nearest_boundary_distances(points,edges))
        edges[:]=1000
        np.testing.assert_array_equal(prepared([[0.,0.,0.]]),boundary.nearest_boundary_distances([[0.,0.,0.]],snapshot))
        with self.assertRaises(ValueError):prepared([[np.nan,0.,0.]])
        with patch.object(boundary,'MAX_DISTANCE_CHECKS',0),self.assertRaisesRegex(ValueError,'超过预算'):
            prepared([[0.,0.,0.]])

    def test_cached_singleton_pruning_matches_batch_and_exact_segments(self):
        rng=np.random.default_rng(592)
        for scale in (1e-8,1.,1e6):
            edges=rng.normal(size=(85,2,3))*scale
            edges[1]=edges[0];edges[2,1]=edges[2,0]
            points=np.vstack((rng.normal(size=(60,3))*scale,edges[:10,0],
                              np.nextafter(edges[:10,0],np.inf)))
            prepared=boundary.prepare_nearest_boundary_distances(edges)
            for point in points:
                singleton=prepared([point]);batch=prepared([point,point])
                exact=np.min([boundary.point_segment_distance(np.asarray([point]),*edge)[0] for edge in edges])
                np.testing.assert_allclose(singleton,batch[:1],rtol=1e-14,atol=0.)
                np.testing.assert_array_equal(singleton,np.asarray([exact]))

    def test_singleton_budget_remains_per_evaluated_segment(self):
        rng=np.random.default_rng(412);edges=rng.normal(size=(70,2,3));point=np.array([[.1,.2,.3]])
        query=boundary.prepare_nearest_boundary_distances(edges)
        with patch.object(boundary,'point_segment_distance',wraps=boundary.point_segment_distance) as probe:
            expected=query(point);count=probe.call_count
        self.assertGreater(count,0)
        with patch.object(boundary,'MAX_DISTANCE_CHECKS',count):
            np.testing.assert_array_equal(query(point),expected)
        with patch.object(boundary,'MAX_DISTANCE_CHECKS',count-1),self.assertRaisesRegex(ValueError,'超过预算'):
            query(point)

    def test_source_preparation_cache_hits_keep_pair_outcomes(self):
        rng=np.random.default_rng(11);tri=rng.normal(size=(25,3,3));tri[:,:,2]=rng.uniform(.3,.8,(25,3))
        k=np.array([[300.,0,200.],[0,300.,100.],[0,0,1.]])
        cold=target._source_separation(tri,k)
        warm=target._source_separation(tri,k)
        for i in range(25):
            for j in range(25):
                expected=cold(i,j)
                self.assertEqual(expected,warm(i,j));self.assertEqual(expected,warm(i,j))


if __name__=='__main__':unittest.main()
