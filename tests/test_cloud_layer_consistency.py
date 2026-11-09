"""Differential coverage for bounded batch reconstruction diagnostics."""
import unittest
from unittest.mock import patch

import numpy as np
import open3d as o3d

from femto.geometry.cloud_layer_consistency import count_second_sheet_points


def cloud(points, normals=None):
    value = o3d.geometry.PointCloud()
    value.points = o3d.utility.Vector3dVector(np.asarray(points, float).reshape(-1, 3))
    if normals is not None:
        value.normals = o3d.utility.Vector3dVector(np.asarray(normals, float).reshape(-1, 3))
    return value


def original_count(value):
    # Frozen R4 predicate, deliberately independent of the optimized helper.
    points = np.asarray(value.points)
    normals = np.asarray(value.normals)
    tree = o3d.geometry.KDTreeFlann(value)
    second = 0
    for index, point in enumerate(points):
        count, ids, _ = tree.search_radius_vector_3d(point, .01)
        if count > 1 and np.any(np.abs((points[np.asarray(ids)] - point) @ normals[index]) > .004):
            second += 1
    return second


class CloudLayerConsistencyTest(unittest.TestCase):
    def same(self, value):
        self.assertEqual(count_second_sheet_points(value), original_count(value))

    def test_dense_complete_batches_keep_original_radius_and_height_predicates(self):
        y,x=np.mgrid[:80,:80]
        base=np.c_[x.ravel()*.001,y.ravel()*.001,np.zeros(x.size)]
        for height in (.003,np.nextafter(.004,0),.004,np.nextafter(.004,np.inf),.005):
            with self.subTest(height=height):
                points=np.r_[base,base+[0,0,height]]
                self.same(cloud(points,np.tile([0.,0.,1.],(len(points),1))))

    def test_dense_cell_bound_never_hides_a_witness_at_cell_boundaries(self):
        from femto.geometry.cloud_layer_batches import negative_bounds
        rng=np.random.default_rng(771)
        points=np.r_[rng.uniform(-.04,.04,(1000,3)),[[0.,0.,0.],[.02,.02,.02],[-.02,-.02,-.02]]]
        normals=rng.normal(size=points.shape)
        negative=negative_bounds(points,normals)
        for i in np.flatnonzero(negative):
            delta=points-points[i];near=np.linalg.norm(delta,axis=1)<.01
            self.assertFalse(np.any(np.abs(delta[near]@normals[i])>.004))

    def test_dense_bounds_and_observed_witnesses_match_at_large_origins(self):
        from femto.geometry.cloud_layer_batches import count_dense
        rng=np.random.default_rng(719)
        points=rng.uniform(-.025,.025,(300,3));normals=rng.normal(size=points.shape)
        for origin in (np.zeros(3),np.array([1e6,-1e6,1e6])):
            value=cloud(points+origin,normals);p=np.asarray(value.points);n=np.asarray(value.normals)
            tree=o3d.geometry.KDTreeFlann(value)
            def legacy(indices):
                return sum(np.any(np.abs((p[np.asarray(tree.search_radius_vector_3d(p[i],.01)[1])]-p[i])@n[i])>.004) for i in indices)
            self.assertEqual(count_dense(p,n,legacy),original_count(value))
            with patch('femto.geometry.cloud_layer_batches.o3d.core.nns.NearestNeighborSearch',side_effect=RuntimeError('unavailable')):
                self.assertEqual(count_dense(p,n,legacy),original_count(value))

    def test_random_clouds_nonunit_normals_and_batch_boundary(self):
        rng = np.random.default_rng(20261026)
        for size in (1, 2, 17, 63, 65, 511, 512, 513, 1503):
            for scale in (.02, .1):
                with self.subTest(size=size, scale=scale):
                    self.same(cloud(rng.uniform(-scale, scale, (size, 3)),
                                    rng.normal(size=(size, 3))))

    def test_strict_radius_and_height_at_adjacent_floats(self):
        for radius in (.01, np.nextafter(.01, 0), np.nextafter(.01, np.inf)):
            for height in (.004, np.nextafter(.004, 0), np.nextafter(.004, np.inf)):
                with self.subTest(radius=radius, height=height):
                    value = cloud([[0, 0, 0], [radius, 0, 0], [0, height, 0]],
                                  [[1, 1, 0]] * 3)
                    self.same(value)

    def test_radius_and_height_ulp_sweeps_rotated_translated(self):
        rng = np.random.default_rng(3101)
        radius_values = [.01]
        height_values = [.004]
        for direction in (0, np.inf):
            r, h = .01, .004
            for _ in range(48):
                r = np.nextafter(r, direction)
                h = np.nextafter(h, direction)
                radius_values.append(r)
                height_values.append(h)
        for origin in (np.zeros(3), np.array([.51, -.25, 1.3]), np.array([1e6, -1e6, 1e6])):
            for radius, height in zip(radius_values, height_values):
                direction = rng.normal(size=3)
                direction /= np.linalg.norm(direction)
                points = np.array([origin, origin + direction * radius])
                normal = direction * (height / radius)
                self.same(cloud(points, [normal, -normal]))

    def test_random_radius_shells_keep_all_original_neighbors(self):
        rng = np.random.default_rng(982031)
        # Separate pairs avoid saturating the bounded candidate list, so this
        # exercises actual candidate completeness rather than dense fallback.
        for gap in (1e-12, 1e-10, 1e-9, 1e-8):
            directions = rng.normal(size=(300, 3))
            directions /= np.linalg.norm(directions, axis=1)[:, None]
            origins = np.column_stack((np.arange(300) * .04,
                                       rng.uniform(-1, 1, 300), np.zeros(300)))
            radii = .01 + np.tile([-gap, gap], 150)
            points = np.stack((origins, origins + directions * radii[:, None]), axis=1)
            normals = np.repeat(directions, 2, axis=0)
            self.same(cloud(points.reshape(-1, 3), normals))

    def test_dense_candidate_overflow_includes_late_witness(self):
        points = np.zeros((200, 3))
        points[-1, 2] = .005
        value = cloud(points, [[0, 0, 1]] * len(points))
        self.assertEqual(original_count(value), 200)
        self.assertEqual(count_second_sheet_points(value), 200)

    def test_duplicates_zero_and_nan_normals_keep_original_statistics(self):
        points = [[0, 0, 0], [0, 0, 0], [0, 0, .006], [1, 0, 0]]
        for normals in ([[0, 0, 0]] * 4,
                        [[0, 0, 1], [0, 0, np.nan], [0, 0, 1], [1, 0, 0]]):
            self.same(cloud(points, normals))

    def test_large_coordinates_and_extreme_normals_fall_back_safely(self):
        self.same(cloud([[1e20, 0, 0], [1e20, 0, .006], [1e20, 0, .01]],
                        [[0, 0, 1]] * 3))
        with np.errstate(over='ignore', invalid='ignore'):
            self.same(cloud([[0, 0, 0], [.003, -.003, .004], [-.003, .003, -.004]],
                            [[1e307, 1e307, 1], [-1e307, -1e307, 1], [1e307, 1e307, 0]]))

    def test_missing_normals_preserves_original_conditional_error(self):
        self.same(cloud([[0, 0, 0]]))
        self.same(cloud([[0, 0, 0], [1, 0, 0]]))
        with self.assertRaises(IndexError):
            count_second_sheet_points(cloud([[0, 0, 0], [0, 0, .005]]))

    def test_native_index_failure_uses_original_full_query(self):
        value = cloud([[0, 0, 0], [0, 0, .006]], [[0, 0, 1]] * 2)
        with patch('femto.geometry.cloud_layer_consistency.o3d.core.nns.NearestNeighborSearch',
                   side_effect=RuntimeError('Float64 unsupported')):
            self.same(value)

    def test_native_search_failure_after_one_batch_does_not_repeat_counts(self):
        origins = np.column_stack((np.arange(600) * .03, np.zeros((600, 2))))
        points = np.stack((origins, origins + [0, 0, .006]), axis=1).reshape(-1, 3)
        value = cloud(points, [[0, 0, 1]] * len(points))
        native = o3d.core.nns.NearestNeighborSearch
        calls = []

        class InterruptedSearch:
            def __init__(self, *args, **kwargs):
                self.search = native(*args, **kwargs)

            def hybrid_index(self):
                return self.search.hybrid_index()

            def hybrid_search(self, *args, **kwargs):
                calls.append(1)
                if len(calls) == 2:
                    raise RuntimeError('Native search interrupted')
                return self.search.hybrid_search(*args, **kwargs)

        with patch('femto.geometry.cloud_layer_consistency.o3d.core.nns.NearestNeighborSearch',
                   InterruptedSearch):
            self.assertEqual(count_second_sheet_points(value), original_count(value))
        self.assertEqual(len(calls), 2)

    def test_empty_cloud_preserves_zero_count(self):
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
            self.same(cloud([]))

    def test_ordinary_case_really_uses_batch_queries(self):
        value = cloud([[0, 0, 0], [0, 0, .006]], [[0, 0, 1]] * 2)
        with patch('femto.geometry.cloud_layer_consistency.o3d.geometry.KDTreeFlann',
                   side_effect=AssertionError('Unexpected per-point fallback')):
            self.assertEqual(count_second_sheet_points(value), 2)

    def test_complete_report_matches_original_count_and_distances(self):
        from femto.pipeline.scene_reconstruction import _cloud_mesh_consistency
        rng = np.random.default_rng(991)
        points = rng.uniform(0, .04, (200, 3))
        value = cloud(points, rng.normal(size=points.shape))
        mesh = o3d.geometry.TriangleMesh.create_box(.04, .04, .04)
        expected_count = original_count(value)
        with patch('femto.pipeline.scene_reconstruction.count_second_sheet_points',
                   return_value=expected_count):
            original = _cloud_mesh_consistency(value, mesh)
        self.assertEqual(_cloud_mesh_consistency(value, mesh), original)


if __name__ == '__main__':
    unittest.main()
