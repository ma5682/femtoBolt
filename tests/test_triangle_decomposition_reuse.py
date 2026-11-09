"""Shared QR data stays immutable, bounded and confined to one mesh operation."""
from contextvars import copy_context
import threading
import unittest
from unittest.mock import patch

import numpy as np

from femto.geometry import triangle_support as support
from femto.geometry.triangle_solve import TriangleSolve


class TriangleDecompositionReuseTests(unittest.TestCase):
    def setUp(self):
        self.vertices = np.array([[0., 0., .5], [.01, 0., .5], [0., .02, .5], [.02, .02, .5]])
        self.faces = np.array([[0, 1, 2], [1, 3, 2]])

    def fresh(self):
        return support.prepare_triangle_faces(self.vertices, self.faces)[2]

    def test_same_scope_shares_only_decomposition_and_keeps_original_bits(self):
        expected = TriangleSolve(np.column_stack((self.vertices[1]-self.vertices[0], self.vertices[2]-self.vertices[0])))
        expected._prepare()
        original = support._solver_record
        with patch.object(support, '_solver_record', wraps=original) as build:
            with support._prepared_operation():
                first, second = self.fresh()(0), self.fresh()(0)
                first.solver._prepare();second.solver._prepare()
                self.assertEqual(build.call_count, 1)
                self.assertIsNot(first.solver, second.solver)
                self.assertIsNot(first.solver._inverse, second.solver._inverse)
                self.assertEqual(first.solver._inverse.tobytes(), expected._inverse.tobytes())
                self.assertEqual(second.solver._inverse.tobytes(), expected._inverse.tobytes())
                self.assertEqual((second.solver._guard, second.solver._inverse_norm, second.solver._matrix_norm),
                                 (expected._guard, expected._inverse_norm, expected._matrix_norm))
                with self.assertRaises(ValueError):
                    first.solver._inverse.setflags(write=True)
                first.solver._inverse = np.zeros((2, 3));first.solver._guard = 999.
                third = self.fresh()(0);third.solver._prepare()
                self.assertEqual(build.call_count, 1)
                self.assertEqual(third.solver._inverse.tobytes(), expected._inverse.tobytes())

    def test_changed_mesh_and_new_operation_do_not_reuse_old_map(self):
        original = support._solver_record
        with patch.object(support, '_solver_record', wraps=original) as build:
            with support._prepared_operation():
                first = self.fresh()(0);first.solver._prepare()
                self.vertices[1, 0] *= 2
                second = self.fresh()(0);second.solver._prepare()
                self.assertEqual(build.call_count, 2)
                self.assertNotEqual(first.solver._inverse.tobytes(), second.solver._inverse.tobytes())
            with support._prepared_operation():
                self.fresh()(0).solver._prepare()
                self.assertEqual(build.call_count, 3)

    def test_replaced_matrix_keeps_local_preparation(self):
        with support._prepared_operation():
            first = self.fresh()(0)
            changed = first.matrix.copy();changed[:, 0] *= 3
            first.solver.matrix = changed
            with patch.object(support, '_solver_record', side_effect=AssertionError('stale matrix binding')):
                first.solver._prepare()
            expected = TriangleSolve(changed);expected._prepare()
            self.assertEqual(first.solver._inverse.tobytes(), expected._inverse.tobytes())

    def test_changed_face_indices_and_solver_method_cannot_reuse_old_record(self):
        with support._prepared_operation():
            first = self.fresh()(0);first.solver._prepare()
            self.faces[0] = [1, 3, 2]
            second = self.fresh()(0);second.solver._prepare()
            self.assertNotEqual(first.solver._inverse.tobytes(), second.solver._inverse.tobytes())
            third = self.fresh()(0)
            original = TriangleSolve._prepare
            with patch.object(TriangleSolve, '_prepare', lambda self: original(self)), \
                    patch.object(support, '_solver_record', side_effect=AssertionError('changed method')):
                third.solver._prepare()
            self.assertEqual(third.solver._inverse.tobytes(), second.solver._inverse.tobytes())

    def test_reused_map_rechecks_every_query_coordinate(self):
        with support._prepared_operation():
            first = self.fresh()(0);second = self.fresh()(0)
            inside = np.array([[.002, .003, .5]])
            outside = np.array([[.02, .03, .5]])
            self.assertTrue(first.solver.point_inside(inside, first, 1e-6)[0])
            self.assertFalse(second.solver.point_inside(outside, second, 1e-6)[0])
            self.assertTrue(second.solver.point_inside(inside, second, 1e-6)[0])

    def test_scope_exit_and_copied_context_on_other_thread_do_not_share(self):
        results = []
        with support._prepared_operation():
            escaped = self.fresh()(0)
            threaded = self.fresh()(0)
            context = copy_context()
            with patch.object(support, '_solver_record', side_effect=AssertionError('cross-thread reuse')):
                def run():
                    try:
                        threaded.solver._prepare();results.append(threaded.solver._inverse is not None)
                    except BaseException as error:
                        results.append(error)
                thread = threading.Thread(target=lambda: context.run(run))
                thread.start();thread.join()
        self.assertEqual(results, [True])
        with patch.object(support, '_solver_record', side_effect=AssertionError('expired scope')):
            escaped.solver._prepare()
        self.assertIsNotNone(escaped.solver._inverse)

    def test_shared_geometry_and_decompositions_use_one_byte_budget(self):
        with patch.object(support, '_shared_policy', return_value=(100, 3200)), support._prepared_operation():
            for _ in range(8):
                prepare = self.fresh()
                for index in range(2):
                    prepare(index).solver._prepare()
                state = support._prepared_operation_cache.get()
                pool = state['shared_pool']
                owners = [ref() for ref in pool['owners'] if ref() is not None]
                self.assertLessEqual(sum(owner.retained_bytes for owner in owners), 3200)
                self.assertLessEqual(pool['peak_bytes'], 3200)

    def test_memory_failure_and_nondefault_error_policy_keep_local_solver(self):
        with support._prepared_operation():
            first = self.fresh()(0)
            with patch.object(support, '_solver_record', side_effect=MemoryError('optional shared storage')):
                first.solver._prepare()
            self.assertIsNotNone(first.solver._inverse)
            second = self.fresh()(1)
            with np.errstate(over='raise'), patch.object(support, '_solver_record', side_effect=AssertionError('changed policy')):
                second.solver._prepare()
            self.assertIsNotNone(second.solver._inverse)

    def test_unusable_decompositions_retain_fallback(self):
        original = support._solver_record
        for dtype in (np.float32, np.float64):
            vertices = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1e-8, 0.]], dtype=dtype)
            with patch.object(support, '_solver_record', wraps=original) as build, support._prepared_operation():
                one = support.prepare_triangle_faces(vertices, [[0, 1, 2]])[2](0)
                two = support.prepare_triangle_faces(vertices, [[0, 1, 2]])[2](0)
                one.solver._prepare();two.solver._prepare()
                self.assertIsNone(one.solver._inverse);self.assertIsNone(two.solver._inverse)
                self.assertEqual(build.call_count, 1)


if __name__ == '__main__':
    unittest.main()
