"""Public cloud decisions are identical with or without the optional kernel."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from femto.geometry import cloud_radius_kernel as kernel
from femto.geometry.cloud_layer_batches import negative_bounds
from femto.geometry.cloud_radius_batches import NativeRadiusCounter
from test_cloud_csr_research import counter


class CloudKernelTests(unittest.TestCase):
    def test_negative_certificates_match_numpy_across_boundaries_and_origins(self):
        rng = np.random.default_rng(41107)
        base = rng.uniform(-.02, .02, (420, 3))
        edge = np.array([0., .004, np.nextafter(.004, 0), np.nextafter(.004, np.inf),
                         .010001, np.nextafter(.010001, 0), -.010001])
        points = np.r_[base, np.c_[edge, np.zeros((len(edge), 2))]]
        normals = rng.normal(size=points.shape)
        normals[::11] = 0.
        for origin in (np.zeros(3), np.array([1e6, -1e6, 1e6])):
            for scale in (1., 1e-310, 1e308):
                with np.errstate(over='ignore'):
                    n = np.clip(normals, -1., 1.)*scale
                p = points + origin
                actual = negative_bounds(p, n)
                with patch.object(kernel, '_cloud_predicate', None):
                    expected = negative_bounds(p, n)
                np.testing.assert_array_equal(actual, expected)

    def test_individual_radius_decisions_match_original_tree(self):
        rng = np.random.default_rng(41108)
        points = rng.uniform(-.015, .015, (192, 3))
        normals = rng.normal(size=points.shape)
        points = np.r_[points, [[0, 0, 0], [.01, 0, 0], [0, 0, .004],
                               [0, 0, np.nextafter(.004, np.inf)]]]
        normals = np.r_[normals, np.tile([0., 0., 1.], (4, 1))]
        _, old = counter(points, normals)
        fast = NativeRadiusCounter(points, normals, old)
        queries = np.arange(len(points), dtype=np.int64)
        self.assertEqual([fast(q.reshape(1)) for q in queries],
                         [old(q.reshape(1)) for q in queries])
        # No normal snapshot may survive a completed query.
        normals[:] = [2., 0., 0.]
        self.assertEqual(fast(queries[::-1]), old(queries[::-1]))

    def test_oversized_query_keeps_original_fallback_and_cancellation(self):
        points = np.zeros((80, 3)); normals = np.tile([0., 0., 1.], (80, 1))
        calls = []
        def old(indices):
            calls.extend(indices.tolist())
            return 0
        fast = NativeRadiusCounter(points, normals, old, max_pairs=37)
        self.assertEqual(fast(np.arange(4)), 0)
        self.assertEqual(calls, list(range(4)))
        fast.check = lambda: (_ for _ in ()).throw(InterruptedError('cancel'))
        with self.assertRaises(InterruptedError):
            fast(np.arange(4))
        with self.assertRaises(InterruptedError):
            negative_bounds(points, normals, check=fast.check)

    def test_unavailable_or_old_extension_uses_numpy(self):
        args = (np.zeros((2, 3)), np.ones((2, 3)), np.array([0]), np.array([0, 1]), np.array([0, 2]))
        for backend in (None, SimpleNamespace(ABI_VERSION=1)):
            with patch.object(kernel, '_cloud_predicate', backend):
                self.assertIsNone(kernel.classify(*args))
                self.assertFalse(kernel.available())

    def test_custom_error_policy_and_unsupported_layouts_do_not_enter_native(self):
        def unexpected(*args):
            raise AssertionError('native code should not run')
        backend = SimpleNamespace(ABI_VERSION=2, classify=unexpected)
        points = np.zeros((2, 3)); normals = np.ones((2, 3))
        queries = np.array([0]); ids = np.array([0, 1]); splits = np.array([0, 2])
        with patch.object(kernel, '_cloud_predicate', backend):
            with np.errstate(under='raise'):
                self.assertIsNone(kernel.classify(points, normals, queries, ids, splits))
            for p in (np.zeros((2, 6))[:, ::2], points.astype('>f8'), points.astype(np.float32),
                      np.ndarray((2, 3), dtype=np.float64, buffer=bytearray(49), offset=1)):
                self.assertIsNone(kernel.classify(p, normals, queries, ids, splits))

    def test_native_resource_failure_retains_original_computation(self):
        rng = np.random.default_rng(41109)
        points = rng.uniform(-.01, .01, (80, 3)); normals = rng.normal(size=points.shape)
        _, old = counter(points, normals)
        def failed(*args):
            raise MemoryError('injected native allocation failure')
        backend = SimpleNamespace(ABI_VERSION=2, classify=failed, negative_boxes=failed)
        with patch.object(kernel, '_cloud_predicate', backend):
            self.assertEqual(NativeRadiusCounter(points, normals, old)(np.arange(80)), old(np.arange(80)))
            actual = negative_bounds(points, normals)
        with patch.object(kernel, '_cloud_predicate', None):
            np.testing.assert_array_equal(actual, negative_bounds(points, normals))


if __name__ == '__main__':
    unittest.main()
