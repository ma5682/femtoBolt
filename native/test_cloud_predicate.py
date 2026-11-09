"""Required direct-native checks; run explicitly after building the extension."""
import unittest
import numpy as np
from femto.geometry import cloud_radius_kernel as kernel
from femto.geometry.cloud_layer_consistency import _RADIUS_SQUARED, _RADIUS_GUARD, _HEIGHT, _ROUNDING_FACTOR


class NativePredicateTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(kernel.available(), 'Build the native kernel before running native/ tests')

    def classify(self, points, normals, queries, ids, splits):
        return np.frombuffer(kernel._cloud_predicate.classify(points, normals, queries, ids, splits,
            _RADIUS_SQUARED, _RADIUS_GUARD, _HEIGHT, _ROUNDING_FACTOR), np.uint8)

    def test_strict_thresholds_empty_queries_and_positive_after_ambiguity(self):
        points = np.array([[0., 0., 0.], [0., 0., .004], [0., 0., .005], [.01, 0., 0.]])
        normals = np.tile([0., 0., 1.], (4, 1))
        queries = np.zeros(6, np.int64)
        ids = np.array([0, 1, 2, 3, 1, 2], np.int64)
        splits = np.array([0, 0, 1, 2, 3, 4, 6], np.int64)
        np.testing.assert_array_equal(self.classify(points, normals, queries, ids, splits), [2, 0, 2, 1, 2, 1])
        self.assertEqual(len(self.classify(points, normals, np.empty(0, np.int64), np.empty(0, np.int64), np.array([0], np.int64))), 0)

    def test_complete_suffix_validated_even_after_a_positive_witness(self):
        points = np.array([[0., 0., 0.], [0., 0., .005]])
        normals = np.tile([0., 0., 1.], (2, 1))
        for invalid in (-1, 2, 2**62):
            with self.assertRaises(ValueError):
                self.classify(points, normals, np.array([0]), np.array([1, invalid]), np.array([0, 2]))

    def test_malformed_shapes_splits_queries_and_budget_raise_without_native_read(self):
        points = np.zeros((2, 3)); normals = np.ones((2, 3))
        cases = [(np.array([2]), np.array([0]), np.array([0, 1])),
                 (np.array([-1]), np.array([0]), np.array([0, 1])),
                 (np.array([0]), np.array([0]), np.array([1, 1])),
                 (np.array([0]), np.array([0]), np.array([0, 2])),
                 (np.array([0, 1]), np.array([0]), np.array([0, 2, 1])),
                 (np.array([0]), np.zeros(65537, np.int64), np.array([0, 65537])),
                 (np.zeros(65537, np.int64), np.empty(0, np.int64), np.zeros(65538, np.int64))]
        for queries, ids, splits in cases:
            with self.assertRaises(ValueError): self.classify(points, normals, queries, ids, splits)
        for p in (np.zeros((2, 2)), points.astype(np.float32), np.zeros((2, 6))[:, ::2],
                  np.ndarray((2, 3), dtype=np.float64, buffer=bytearray(49), offset=1)):
            with self.assertRaises(ValueError):
                self.classify(p, normals, np.array([0]), np.array([0]), np.array([0, 1]))

    def test_random_and_boundary_products_match_binary64_numpy(self):
        rng = np.random.default_rng(41110)
        for origin in (0., 1e6):
            for scale in (0., 1., 1e-310, 1e308):
                points = rng.uniform(-.01, .01, (128, 3)) + origin
                points[0] = origin
                points[1:4] = [[origin, origin, origin + value] for value in (.004, np.nextafter(.004, 0), np.nextafter(.004, np.inf))]
                normals = rng.uniform(-1., 1., (128, 3))*scale
                queries = np.arange(48, dtype=np.int64)
                ids = np.tile(np.arange(128, dtype=np.int64), len(queries))
                splits = np.arange(len(queries) + 1, dtype=np.int64)*128
                delta = points[ids] - points[np.repeat(queries, 128)]
                squared = (delta[:, 0]*delta[:, 0] + delta[:, 1]*delta[:, 1]) + delta[:, 2]*delta[:, 2]
                with np.errstate(over='ignore', invalid='ignore'):
                    product = delta*normals[np.repeat(queries, 128)]
                    heights = np.abs((product[:, 0] + product[:, 1]) + product[:, 2])
                    guard = _ROUNDING_FACTOR*np.maximum(np.abs(product).sum(axis=1), _HEIGHT)
                inside = squared < _RADIUS_SQUARED - _RADIUS_GUARD
                finite = np.isfinite(heights) & np.isfinite(guard)
                positive = (inside & finite & (heights > _HEIGHT + guard)).reshape(-1, 128).any(axis=1)
                ambiguous = ((np.abs(squared - _RADIUS_SQUARED) <= _RADIUS_GUARD) |
                             (inside & (~finite | (np.abs(heights - _HEIGHT) <= guard)))).reshape(-1, 128).any(axis=1)
                expected = np.where(positive, 1, np.where(ambiguous, 2, 0))
                np.testing.assert_array_equal(self.classify(points, normals, queries, ids, splits), expected)


if __name__ == '__main__':
    unittest.main()
