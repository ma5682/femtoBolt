"""Compact preparation preserves geometry, ownership and resource limits."""
import unittest
from unittest.mock import Mock, patch

import numpy as np

from femto.geometry import triangle_support as support
from femto.resource_policy import GiB, MiB, MemorySnapshot


class TriangleCompactCacheTests(unittest.TestCase):
    def test_packing_preserves_bytes_shapes_strides_and_readonly_views(self):
        rng = np.random.default_rng(20261103)
        fields = ('origin', 'ab', 'ac', 'normal', 'normal_length', 'matrix', 'minimum_edge_length')
        for dtype in (np.float64, np.float32, '>f8', np.int64):
            for _ in range(30):
                vertices = (rng.normal(size=(3, 3)) * 10).astype(dtype)
                original = support._prepare_one(vertices, np.array([[0, 1, 2]]), 0)
                values = support._unpack_geometry(support._pack_geometry(original))
                for name, actual in zip(fields, values):
                    expected = getattr(original, name)
                    with self.subTest(dtype=dtype, field=name):
                        if isinstance(expected, np.ndarray):
                            self.assertEqual(expected.dtype.str, actual.dtype.str)
                            self.assertEqual(expected.shape, actual.shape)
                            self.assertEqual(expected.strides, actual.strides)
                            self.assertEqual(expected.tobytes(), actual.tobytes())
                            with self.assertRaises(ValueError):
                                actual.setflags(write=True)
                        else:
                            self.assertEqual(np.asarray(expected).tobytes(), np.asarray(actual).tobytes())

    def test_packed_solver_matrix_keeps_column_order(self):
        vertices = np.array([[0., 0., 0.], [1., 2., 3.], [4., 5., 6.]])
        with support._prepared_operation():
            _, _, prepared = support.prepare_triangle_faces(vertices, [[0, 1, 2]])
            matrix = prepared(0).matrix
        np.testing.assert_array_equal(matrix, [[1., 4.], [2., 5.], [3., 6.]])
        self.assertEqual(matrix.strides, (16, 8))

    def test_multiple_live_mesh_caches_share_one_byte_budget(self):
        pool = {'budget': 7000, 'owners': [], 'peak_bytes': 0}
        caches = [support._SharedGeometry(lambda i: (b'x' * 144, float(i)), 100, 7000, pool=pool)
                  for _ in range(3)]
        for index in range(100):
            for cache in caches:
                self.assertEqual(cache(index), (b'x' * 144, float(index)))
                self.assertLessEqual(sum(item.retained_bytes for item in caches), 7000)
        self.assertLessEqual(pool['peak_bytes'], 7000)
        disabled = support._SharedGeometry(lambda i: (i,), 0, 0)
        self.assertEqual(disabled(1), (1,))
        self.assertEqual(disabled.cache_info().currsize, 0)
        oversized = support._SharedGeometry(lambda i: b'x' * 8000, 100, 7000)
        self.assertEqual(oversized(1), b'x' * 8000)
        self.assertEqual(oversized.cache_info().currsize, 0)

    def test_eviction_recomputes_and_exceptions_are_not_cached(self):
        build = Mock(side_effect=lambda index: (index,))
        cache = support._SharedGeometry(build, 2, MiB)
        for index in (0, 1, 0, 2, 1):
            self.assertEqual(cache(index), (index,))
        self.assertEqual([call.args[0] for call in build.call_args_list], [0, 1, 2, 1])
        self.assertEqual(cache.cache_info().currsize, 2)
        build.side_effect = ValueError('invalid face')
        for _ in range(2):
            with self.assertRaisesRegex(ValueError, 'invalid face'):
                cache(3)
        self.assertEqual(build.call_count, 6)

    def test_resource_decision_is_frozen_and_unknown_or_low_memory_keeps_legacy_cap(self):
        cases = ((MemorySnapshot(None, None, 'unknown'), 4096, 4 * MiB),
                 (MemorySnapshot(16 * GiB, GiB, 'low'), 4096, 4 * MiB),
                 (MemorySnapshot(32 * GiB, 24 * GiB, 'high'), 32768, 16 * MiB))
        for memory, entries, budget in cases:
            with self.subTest(memory=memory.source), \
                    patch('femto.resource_policy.memory_snapshot', return_value=memory) as observe:
                state = {}
                self.assertEqual(support._shared_policy(state, 100000), (entries, budget))
                self.assertEqual(support._shared_policy(state, 200000), (entries, budget))
                self.assertEqual(support._shared_policy(state, 3), (3, budget))
                self.assertEqual(observe.call_count, 1)


if __name__ == '__main__':
    unittest.main()
