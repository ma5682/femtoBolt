"""Efficiency paths retain geometry, numerical fallbacks and query boundaries."""
from copy import deepcopy
from unittest.mock import patch
import unittest

import numpy as np
import open3d as o3d

from femto.geometry.pixel_clip import EPS, area2, project, pixel_planes, _orient_sign
from femto.geometry.pixel_clip_batch import clip_triangles_planes, project_clipped_polygons
from femto.geometry.triangle_separation import PreparedTriangle
from femto.planning import boundary_clearance as boundary
from femto.planning.region_coverage import point_segment_distance
from femto.local import local_pose_refinement as refinement


class SurfaceEfficiencyTests(unittest.TestCase):
    def test_bounded_polygon_projection_keeps_scalar_bytes_and_error_envelope(self):
        rng = np.random.default_rng(502810)
        k = np.array([[1.7, 0., .2], [0., 1.3, -.4], [0., 0., 1.]])
        for scale in (1e-150, 1e-12, 1., 1e12, 1e150):
            for count in (0, 1, 7, 8, 31, 64):
                triangles = rng.uniform(-2, 2, (count, 3, 3))*scale
                triangles[:, :, 2] = rng.uniform(.5, 2, (count, 3))*scale
                clipped = clip_triangles_planes(triangles, pixel_planes(k, 0, 0))
                actual = project_clipped_polygons(clipped, triangles, k)
                for index, (bary, _) in enumerate(clipped):
                    if len(bary) < 3:
                        self.assertIsNone(actual[index]); continue
                    xyz = bary @ triangles[index]
                    uv = project(xyz, k)
                    edges = np.empty_like(uv)
                    edges[:-1] = uv[1:]-uv[:-1]; edges[-1] = uv[0]-uv[-1]
                    error = EPS*max(1., float(np.abs(uv).max()))*float(np.abs(edges).sum())
                    with self.subTest(scale=scale, count=count, index=index):
                        self.assertEqual(actual[index][0].tobytes(), xyz.tobytes())
                        self.assertEqual(actual[index][1].tobytes(), uv.tobytes())
                        self.assertEqual(actual[index][2], area2(uv))
                        self.assertEqual(actual[index][3], error)
                        self.assertEqual(actual[index][4], bool(np.isfinite(xyz).all() and np.isfinite(uv).all()))
        with self.assertRaises(ValueError):project_clipped_polygons([(np.eye(3), False)]*65, np.ones((65, 3, 3)), k)

    def test_projection_preserves_degenerate_and_signed_zero_polygons(self):
        triangles = np.array([[[-0., 0., 1.], [1., -0., 1.], [-0., 1., 1.]]]*8)
        clipped = [(np.eye(3), True), (np.zeros((0, 3)), False)]*4
        result = project_clipped_polygons(clipped, triangles, np.eye(3))
        for index in range(0, 8, 2):
            xyz = clipped[index][0] @ triangles[index]
            self.assertEqual(result[index][0].tobytes(), xyz.tobytes())
            self.assertEqual(result[index][1].tobytes(), project(xyz, np.eye(3)).tobytes())
            self.assertIsNone(result[index+1])

    def test_prepared_orientation_matches_original_filter_and_exact_cases(self):
        rng = np.random.default_rng(872)
        for dtype in (np.float32, np.float64):
            for scale in (1e-20, 1., 1e20):
                for _ in range(80):
                    points = (rng.normal(size=(3, 2))*scale).astype(dtype)
                    value = PreparedTriangle(points, np.arange(3))
                    for start, end, point in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
                        with np.errstate(all='ignore'):
                            self.assertEqual(value.orientation(start, end, point), _orient_sign(points[start], points[end], points[point]))
        for tiny in (0., 2**-52, -2**-52):
            points = np.array([[0., -0.], [1., 1.], [2., 2.+tiny]])
            value = PreparedTriangle(points, np.arange(3))
            self.assertEqual(value.orientation(0, 1, 2), _orient_sign(*points))

    def test_single_boundary_query_matches_exhaustive_at_thresholds_and_scales(self):
        rng = np.random.default_rng(728)
        for scale in (1e-10, 1., 1e10):
            segments = rng.normal(size=(65, 2, 3))*scale
            segments[0, 1] = segments[0, 0]
            segments[1] = np.array([[-1000., 0., 0.], [1., 0., 0.]])*scale
            query = boundary.prepare_nearest_boundary_distances(segments)
            for xyz in rng.normal(size=(30, 1, 3))*scale:
                expected = np.full(1, np.inf)
                for a, b in segments:expected = np.minimum(expected, point_segment_distance(xyz, a, b))
                np.testing.assert_array_equal(query(xyz), expected)
        segments = np.array([[[-1., 0., 0.], [1., 0., 0.]]])
        query = boundary.prepare_nearest_boundary_distances(segments)
        for y in (.005, .005+1e-9, np.nextafter(.005+1e-9, np.inf)):
            np.testing.assert_array_equal(query([[0., y, 0.]]), point_segment_distance(np.array([[0., y, 0.]]), *segments[0]))

    def test_single_query_budget_counts_seed_and_candidate_and_can_cancel(self):
        segments = np.array([[[0., 0., 0.], [1., 0., 0.]]])
        query = boundary.prepare_nearest_boundary_distances(segments)
        with patch.object(boundary, 'MAX_DISTANCE_CHECKS', 1), self.assertRaises(ValueError):query([[.5, 1., 0.]])
        with patch.object(boundary, 'MAX_DISTANCE_CHECKS', 2):np.testing.assert_array_equal(query([[.5, 1., 0.]]), [1.])
        stop = [False]
        def check():
            if stop[0]:raise RuntimeError('stopped')
        query = boundary.prepare_nearest_boundary_distances(segments, check)
        stop[0] = True
        with self.assertRaisesRegex(RuntimeError, 'stopped'):query([[.5, 1., 0.]])

    def test_pair_scale_reuse_keeps_bidirectional_solver_and_original_clouds(self):
        rng = np.random.default_rng(207)
        fixed = o3d.geometry.PointCloud()
        fixed.points = o3d.utility.Vector3dVector(rng.uniform(-.1, .1, (1800, 3))+[0., 0., .6])
        moving = deepcopy(fixed).translate([.001, -.001, .001])
        before = np.asarray(moving.points).copy()
        reference = {}
        other = {}
        for voxel in (.012, .006, .003):
            reference[voxel] = fixed.voxel_down_sample(voxel)
            reference[voxel].estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*3, max_nn=40))
            other[voxel] = moving.voxel_down_sample(voxel)
        for a, b, reference_target in ((moving, fixed, True), (fixed, moving, False)):
            expected = refinement.solve(a, b, np.eye(4), _reference_scales=reference, _reference_is_target=reference_target)
            actual = refinement.solve(a, b, np.eye(4), _reference_scales=reference, _reference_is_target=reference_target, _other_scales=other)
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(np.asarray(moving.points), before)


if __name__ == '__main__':unittest.main()
