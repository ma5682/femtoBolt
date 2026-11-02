"""Differential checks for source preparation against untouched pixel predicates."""
from itertools import permutations
import unittest
from unittest.mock import patch

import numpy as np

from femto.geometry import pixel_clip as original
from femto.geometry.triangle_separation import PreparedTriangle, prepared_overlap, _finite_signs
from femto.planning import target_surface as target


def original_source(triangles, intrinsic, source_vertex_ids=None, *, source_projection=None):
    def separated(left, right):
        a, b = triangles[left], triangles[right]
        return (original.shared_edge_separates(a, b, intrinsic)
                or original.interior_overlap(original.project(a, intrinsic),
                                             original.project(b, intrinsic)) is False)
    return separated


def prepared(points):
    return PreparedTriangle(np.array(points, dtype=float), np.arange(3))


class TriangleSeparationTests(unittest.TestCase):
    def test_finite_endpoint_signs_keep_exact_determinants_at_extreme_scales(self):
        # Endpoint identity is exact even when intermediate subtraction or
        # multiplication overflows/underflows in the floating filter.
        for scale in (1e-200, 1., 1e200, 8e307):
            a, b = np.array([-scale, -0.]), np.array([scale, scale])
            points = np.array([a, b, [0., scale], [0., -scale], [0., 0.]])
            with np.errstate(all='ignore'):
                expected = {original._orient_sign(a, b, p) for p in points}
                self.assertEqual(_finite_signs(a, b, points), expected)
                value = prepared([a, b, points[2]])
                self.assertEqual(value.signs(0),
                                 {original._orient_sign(a, b, p) for p in value.points})

    def test_nonfinite_endpoint_orientation_retains_original_error(self):
        # The finite shortcut must not turn invalid coordinates into a valid
        # orientation simply because the caller supplied the same vertex ID.
        for coordinate in (np.inf, np.nan):
            value = prepared([[coordinate, 0.], [1., 0.], [0., 1.]])
            with np.errstate(all='ignore'):
                self.assert_same_outcome(lambda: value.orientation(0, 1, 0),
                    lambda: original._orient_sign(value.points[0], value.points[1], value.points[0]))

    def assert_same_outcome(self, actual, expected):
        try:
            old = expected()
        except (ValueError, OverflowError, StopIteration) as error:
            with self.assertRaises(type(error)):
                actual()
        else:
            self.assertIs(actual(), old)

    def test_overlap_matches_original_across_scales_and_offsets(self):
        rng = np.random.default_rng(20261023)
        for scale in (1e-150, 1e-12, 1., 1e12, 1e150):
            for _ in range(100):
                a = rng.uniform(-2, 2, (3, 2))*scale
                b = rng.uniform(-2, 2, (3, 2))*scale
                pa, pb = prepared(a), prepared(b)
                self.assertIs(prepared_overlap(pa, pb), original.interior_overlap(a, b))
                self.assertIs(prepared_overlap(pb, pa), original.interior_overlap(b, a))

    def test_exact_touch_tiny_overlap_degenerate_and_nonfinite_keep_outcome(self):
        a = np.array([[0., 0.], [1., 0.], [0., 1.]])
        near = np.array([[0., 1.-2**-50], [1., 1.-2**-50], [0., 2.]])
        self.assertIsNone(original.interior_overlap(a, near))
        examples = [(a, near), (a, a+[1., 0.]), (a, a),
                    (np.zeros((3, 2)), a),
                    (np.array([[0., 0.], [0., 0.], [1., 0.]]), a),
                    (a*1e200, (a+[2., 0.])*1e200),
                    (a*1e-200, (a+[2., 0.])*1e-200),
                    (np.array([[np.inf, 0.], [0., 0.], [1., 1.]]), a),
                    (np.array([[np.nan, 0.], [0., 0.], [1., 1.]]), a)]
        with np.errstate(all='ignore'):
            for left, right in examples:
                for p in permutations(range(3)):
                    for q in ((0, 1, 2), (2, 1, 0)):
                        aa, bb = left[list(p)], right[list(q)]
                        self.assert_same_outcome(
                            lambda: prepared_overlap(prepared(aa), prepared(bb)),
                            lambda: original.interior_overlap(aa, bb))

    def test_source_pairs_keep_duplicate_coordinate_identity_and_winding(self):
        a = np.array([[0., -0., 1.], [1., 0., 1.], [0., 1., 1.]])
        adjacent = np.array([[1., 0., 1.], [1., 1., 1.], [-0., 1., 1.]])
        intrinsic = np.array([[117., 0., 49.3], [0., 91., 47.7], [0., 0., 1.]])
        variants = [a, adjacent, a+[.1, .1, 0.], a+[2., 0., 0.],
                    a+[0., 0., .01], a[[0, 0, 2]], np.zeros((3, 3))+[0., 0., 1.]]
        triangles = np.asarray([v[list(order)] for v in variants
                                for order in permutations(range(3))])
        _, ids = np.unique(triangles.reshape(-1, 3), axis=0, return_inverse=True)
        explicit = target._source_separation(triangles, intrinsic, ids.reshape(-1, 3))
        implicit = target._source_separation(triangles, intrinsic)
        reference = original_source(triangles, intrinsic)
        for left in range(len(triangles)):
            for right in range(left, len(triangles)):
                self.assert_same_outcome(lambda: explicit(left, right), lambda: reference(left, right))
                self.assert_same_outcome(lambda: implicit(left, right), lambda: reference(left, right))

    def test_shared_edges_reuse_projection_and_each_construction_is_fresh(self):
        triangles = np.array([[[0., 0., 1.], [1., 0., 1.], [0., 1., 1.]],
                              [[1., 0., 1.], [1., 1., 1.], [0., 1., 1.]],
                              [[1., 0., 1.], [2., 0., 1.], [1., 1., 1.]]])
        with patch.object(target, 'project', wraps=target.project) as projection:
            separation = target._source_separation(triangles, np.eye(3))
            for pair in ((0, 1), (0, 2), (1, 2), (0, 1)):
                self.assertTrue(separation(*pair))
            self.assertEqual(projection.call_count, 3)
            self.assertEqual(separation.cache_info().maxsize, 4096)
            self.assertEqual(separation.cache_info().hits, 1)
        changed = triangles.copy(); changed[1] = changed[0]
        self.assertFalse(target._source_separation(changed, np.eye(3))(0, 1))
        self.assertTrue(separation(0, 1))

    def test_full_target_arrays_metadata_and_fixed_budget_frontiers_match(self):
        import test_target_surface as fixtures
        from test_fixed_rgb_experience import quad
        from femto.planning import planning_revision as revisions
        from femto.planning.quality_constrained import evidence
        case = fixtures.TargetSurfaceTests(); case.setUp(); self.addCleanup(case.doCleanups)
        mask = np.zeros((100, 100), bool); mask[48:53, 48:53] = True; mask[50, 50] = False
        front = quad(left=-.1, right=.1, top=-.1, bottom=.1)
        back = quad(.505, left=-.1, right=.1, top=-.1, bottom=.1)
        for mesh in (front+front, front+back):
            model, region = case.fixture(mesh=mesh, mask=mask)
            quality = evidence(model, region, allow_empty=True)
            for algorithm in (revisions.ALGORITHMS['target'], revisions.EXACT_TARGET):
                revision = revisions.capture(model, region, case.head,
                    algorithms=dict(revisions.ALGORITHMS, target=algorithm))
                for budget in (target.TargetBudget(), target.TargetBudget(max_units=7),
                               target.TargetBudget(max_pair_checks=3),
                               target.TargetBudget(max_candidates=3)):
                    with self.subTest(algorithm=algorithm, budget=budget):
                        actual = target.build(model, region, revision, evidence=quality, budget=budget)
                        with patch.object(target, '_source_separation', original_source):
                            expected = target.build(model, region, revision, evidence=quality, budget=budget)
                        self.assertEqual(actual.metadata, expected.metadata)
                        for name in actual.arrays:
                            self.assertEqual(actual.arrays[name].tobytes(), expected.arrays[name].tobytes(), name)
                        target.validate(actual, revision, evidence=quality, domain=None)


if __name__ == '__main__':
    unittest.main()
