import unittest
import json
from unittest.mock import patch

import numpy as np

from femto.planning.boundary_clearance import boundary_clearance_mask, nearest_boundary_distances, prepare_nearest_boundary_distances
from femto.planning.region_coverage import point_segment_distance


def exhaustive(mids, half, boundary, radius):
    clearance = np.full(len(mids), np.inf)
    for a, b in boundary:
        clearance = np.minimum(clearance, point_segment_distance(mids, a, b))
    return clearance - half >= radius + 1e-9


class BoundaryClearanceIndexTests(unittest.TestCase):
    def test_prepared_boundary_repeated_queries_are_exact_and_ignore_later_input_mutation(self):
        rng=np.random.default_rng(790)
        boundary=rng.normal(size=(91,2,3));original=boundary.copy()
        query=prepare_nearest_boundary_distances(boundary)
        boundary[:]=1000
        for _ in range(5):
            points=rng.normal(size=(137,3))
            np.testing.assert_array_equal(query(points),nearest_boundary_distances(points,original))

    def test_prepared_queries_keep_original_per_query_budget_and_cancel_checks(self):
        boundary=np.array([[[0.,0.,0.],[1.,0.,0.]]])
        query=prepare_nearest_boundary_distances(boundary)
        from femto.planning import boundary_clearance as module
        with patch.object(module,'MAX_DISTANCE_CHECKS',2):
            self.assertEqual(query(np.array([[.5,1.,0.]])).tolist(),[1.])
            self.assertEqual(query(np.array([[.5,1.,0.]])).tolist(),[1.])
            with self.assertRaises(ValueError):query(np.array([[.5,1.,0.],[.5,2.,0.]]))
        stopped=[False]
        def check():
            if stopped[0]:raise RuntimeError('cancelled')
        query=prepare_nearest_boundary_distances(boundary,check);stopped[0]=True
        with self.assertRaisesRegex(RuntimeError,'cancelled'):query(np.array([[.5,1.,0.]]))

    def test_matches_all_boundary_segments_on_folded_3d_geometry(self):
        rng = np.random.default_rng(673)
        for radius in (.001, .01, .05):
            mids = rng.random((1000, 3))
            boundary = rng.random((257, 2, 3))
            half = rng.random(1000) * .02
            np.testing.assert_array_equal(boundary_clearance_mask(mids, half, boundary, radius),
                                          exhaustive(mids, half, boundary, radius))

    def test_threshold_equality_long_segments_and_zero_length_edges(self):
        radius = .005
        distances = np.array([radius - 1e-12, radius, radius + 1e-9,
                              np.nextafter(radius + 1e-9, np.inf), radius + 2e-9])
        mids = np.column_stack((np.zeros(len(distances)), distances, np.zeros(len(distances))))
        boundary = np.array([[[-100., 0., 0.], [100., 0., 0.]],
                             [[0., 1., 0.], [0., 1., 0.]]])
        for half in (np.zeros(len(mids)), np.full(len(mids), .002)):
            np.testing.assert_array_equal(boundary_clearance_mask(mids, half, boundary, radius),
                                          exhaustive(mids, half, boundary, radius))

    def test_counts_only_exact_candidates_and_enforces_original_budget(self):
        mids = np.zeros((10, 3))
        half = np.zeros(10)
        # Many remote boundaries are proven irrelevant without exact queries.
        remote = np.array([[[10. + i, 0., 0.], [10. + i, 1., 0.]] for i in range(100)])
        with patch('femto.planning.boundary_clearance.MAX_DISTANCE_CHECKS', 5):
            self.assertTrue(boundary_clearance_mask(mids, half, remote, .001).all())
            with self.assertRaisesRegex(ValueError, 'Boundary check budget exceeded'):
                boundary_clearance_mask(mids, half, np.zeros((1, 2, 3)), .001)

    def test_cancel_and_nonfinite_geometry_fail_closed(self):
        def cancel(): raise RuntimeError('cancelled')
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):
            boundary_clearance_mask(np.zeros((1, 3)), np.zeros(1), np.zeros((1, 2, 3)), .01, cancel)
        with self.assertRaisesRegex(ValueError, '非有限'):
            boundary_clearance_mask(np.full((1, 3), np.nan), np.zeros(1), np.zeros((1, 2, 3)), .01)

    def test_nearest_distances_equal_exhaustive_including_long_edges(self):
        rng = np.random.default_rng(786)
        points = rng.random((500, 3))
        boundary = rng.random((91, 2, 3))
        # A very long segment has a distant center yet can pass near a query.
        boundary[0] = [[-1000., .5, .5], [1., .5, .5]]
        boundary[1, 1] = boundary[1, 0]
        expected = np.full(len(points), np.inf)
        for a, b in boundary:
            expected = np.minimum(expected, point_segment_distance(points, a, b))
        # BLAS may round the 3-term dot product differently for grouped rows;
        # this is floating-point roundoff, far below any geometric threshold.
        np.testing.assert_allclose(nearest_boundary_distances(points, boundary), expected, rtol=1e-14, atol=1e-15)

    def test_nearest_distance_budget_and_cancellation(self):
        with patch('femto.planning.boundary_clearance.MAX_DISTANCE_CHECKS', 5):
            with self.assertRaisesRegex(ValueError, '覆盖边界采样检查超过预算'):
                nearest_boundary_distances(np.zeros((10, 3)), np.zeros((1, 2, 3)))
        def cancel(): raise RuntimeError('cancelled')
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):
            nearest_boundary_distances(np.zeros((1, 3)), np.zeros((1, 2, 3)), cancel)

    def test_document_budget_matches_compact_utf8_without_dropping_data(self):
        from femto.planning.coverage_inspection import check_document_budget
        plan = {'samples': [{'xyz_m': [.1, .2, .3], 'reason': '边界仍待检查'}] * 100}
        encoded = json.dumps(plan, ensure_ascii=False, indent=None, allow_nan=False).encode('utf-8')
        self.assertGreater(len(json.dumps(plan, ensure_ascii=False, indent=2).encode('utf-8')), len(encoded))
        check_document_budget(plan, limit=len(encoded))
        with self.assertRaisesRegex(ValueError, '可回读大小'):
            check_document_budget(plan, limit=len(encoded) - 1)
        self.assertEqual(json.loads(encoded), plan)


if __name__ == '__main__':
    unittest.main()
