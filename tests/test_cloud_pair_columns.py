"""Column-wise radius arithmetic preserves individual decisions and live data."""
import unittest
import numpy as np
from femto.geometry.cloud_radius_batches import NativeRadiusCounter
from test_cloud_csr_research import counter


class CloudPairColumnTests(unittest.TestCase):
    def test_individual_decisions_for_strided_points_and_nonunit_normals(self):
        rng = np.random.default_rng(41106)
        backing = rng.uniform(-.012, .012, (180, 6))
        points = backing[:, ::2]
        normals = rng.normal(size=(180, 6))[:, ::2]
        normals[::9] = 0.
        _, legacy = counter(points, normals)
        fast = NativeRadiusCounter(points, normals, legacy, max_pairs=65536)
        self.assertTrue(fast.enabled)
        actual = [fast(np.array([i])) for i in range(len(points))]
        expected = [legacy(np.array([i])) for i in range(len(points))]
        self.assertEqual(actual, expected)
        # In-place edits to the normal array must not observe stale copies.
        normals[:] = [0., 0., 2.]
        self.assertEqual([fast(np.array([i])) for i in range(0, len(points), 3)],
                         [legacy(np.array([i])) for i in range(0, len(points), 3)])

    def test_overflow_and_underflow_products_keep_boundary_fallback(self):
        points = np.array([[0., 0., 0.], [.009, .001, .003], [-.004, .002, .006]])
        for scale in (0., 1e-310, 1e308):
            normals = np.tile([scale, -scale, scale], (3, 1))
            _, legacy = counter(points, normals)
            fast = NativeRadiusCounter(points, normals, legacy)
            with np.errstate(over='ignore', invalid='ignore'):
                self.assertEqual([fast(np.array([i])) for i in range(3)],
                                 [legacy(np.array([i])) for i in range(3)])


if __name__ == '__main__':
    unittest.main()
