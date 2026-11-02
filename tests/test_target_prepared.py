"""Target acceleration must retain float bits, visibility and budget frontiers."""
from unittest.mock import patch
import unittest

import numpy as np

from femto.geometry import pixel_clip
from femto.planning import target_surface as target
import test_target_surface as fixtures


def original_area(origin, first, second):
    return float(np.linalg.norm(np.cross(first-origin, second-origin)) / 2)


def unprepared_source(triangles, intrinsic, source_vertex_ids=None, *, source_projection=None):
    def separated(left, right):
        a, b = triangles[left], triangles[right]
        return (pixel_clip.shared_edge_separates(a, b, intrinsic)
            or pixel_clip.interior_overlap(pixel_clip.project(a, intrinsic), pixel_clip.project(b, intrinsic)) is False)
    return separated


class PreparedTargetTests(unittest.TestCase):
    def fixture(self):
        case = fixtures.TargetSurfaceTests()
        case.setUp(); self.addCleanup(case.doCleanups)
        return case

    def test_area_keeps_original_float_bits_across_scales_and_degeneracy(self):
        rng = np.random.default_rng(1010)
        for scale in (1e-20, 1e-8, 1., 1e8, 1e20):
            for _ in range(160):
                triangle = (rng.normal(size=(3, 3)) * scale).astype('f4').astype('f8')
                self.assertEqual(target._triangle_area(*triangle), original_area(*triangle))
        for triangle in (np.zeros((3, 3)), np.array([[0., 0., 1.], [1., 0., 1.], [2., 0., 1.]])):
            self.assertEqual(target._triangle_area(*triangle), 0.)

    def test_full_target_arrays_and_metrics_match_original_operations(self):
        case = self.fixture()
        mask = np.zeros((100, 100), bool); mask[46:55, 45:56] = True; mask[49:52, 49:52] = False
        model, region = case.fixture(mask=mask)
        actual, revision, quality = case.build(model, region)
        with patch.object(target, '_source_separation', unprepared_source), patch.object(target, '_triangle_area', original_area):
            expected, _, _ = case.build(model, region)
        self.assertEqual(actual.metadata, expected.metadata)
        for key in actual.arrays:np.testing.assert_array_equal(actual.arrays[key], expected.arrays[key])
        target.validate(actual, revision, evidence=quality, domain=None)

    def test_cached_face_pairs_still_spend_each_pixel_pair_budget(self):
        case = self.fixture()
        mask = np.zeros((100, 100), bool); mask[46:55, 46:55] = True
        model, region = case.fixture(mask=mask)
        for budget in (target.TargetBudget(max_pair_checks=3), target.TargetBudget(max_units=7)):
            with self.subTest(budget=budget):
                actual, revision, quality = case.build(model, region, budget=budget)
                with patch.object(target, '_source_separation', unprepared_source), patch.object(target, '_triangle_area', original_area):
                    expected, _, _ = case.build(model, region, budget=budget)
                self.assertEqual(actual.metadata, expected.metadata)
                self.assertGreater(actual.summary['unknown_by_reason']['budget_unprocessed'], 0)
                target.validate(actual, revision, evidence=quality, domain=None)

    def test_new_construction_rechecks_source_geometry_and_bounds_its_cache(self):
        a = np.array([[0., 0., 1.], [1., 0., 1.], [0., 1., 1.]])
        triangles = np.array([a, a + [2., 0., 0.]])
        first = target._source_separation(triangles, np.eye(3))
        self.assertTrue(first(0, 1)); self.assertTrue(first(0, 1))
        self.assertEqual(first.cache_info().hits, 1)
        self.assertEqual(first.cache_info().maxsize, 4096)
        changed = np.array([a, a])
        self.assertFalse(target._source_separation(changed, np.eye(3))(0, 1))
        self.assertTrue(first(0, 1))


if __name__ == '__main__':unittest.main()
