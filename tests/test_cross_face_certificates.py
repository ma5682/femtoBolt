"""Cross-face certificates match original guarded/scalar geometry decisions."""
import unittest
from unittest.mock import patch

import numpy as np

from femto.geometry import triangle_batch
from femto.geometry.triangle_solve import TriangleSolve
from femto.geometry.triangle_support import prepare_triangle_faces
from femto.planning import triangle_certificates as certificates
from femto.planning import surface_trajectory, quality_constrained


def fixture(count=96):
    rng = np.random.default_rng(51106)
    vertices, points, ends = [], [], []
    for index in range(count):
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        scale = (1e-5, .001, .1, 10., 1000.)[index % 5]
        sine = (1., .3, .02, .00101, 1e-5, 0.)[index % 6]
        origin = rng.uniform(-10., 10., 3)
        ab = rotation[:, 0]*scale
        ac = (rotation[:, 0]+rotation[:, 1]*sine)*scale
        normal = rotation[:, 2]
        vertices.extend((origin, origin+ab, origin+ac))
        # Interior, outside, shared-edge, near-plane, and translated samples.
        for u, v, offset in ((.3, .3, 0.), (-.2, .3, 0.), (0., .5, 0.),
                             (.3, .3, 1e-6*(1-2e-6)), (.3, .3, 1e-6*(1+2e-6))):
            points.append(origin+u*ab+v*ac+normal*offset)
            ends.append(origin+.2*ab+.6*ac+normal*offset)
    return (np.asarray(vertices), np.arange(3*count).reshape(-1, 3),
            np.asarray(points), np.asarray(ends), np.repeat(np.arange(count), 5))


def originals(vertices, faces, points, ends, ids):
    _, _, prepare = prepare_triangle_faces(vertices, faces)
    point_flags, chord_flags = np.zeros(len(points), bool), np.zeros(len(points), bool)
    for rows, face in certificates._groups(ids, vertices, faces, prepare):
        point_flags[rows] = face.solver.point_inside(points[rows], face, 1e-6)
        chord_flags[rows] = face.solver.chord_inside(points[rows], ends[rows], face)
    return point_flags, chord_flags


class CrossFaceCertificateTests(unittest.TestCase):
    def test_mixed_condition_scale_and_boundary_rows_match_original_certificates(self):
        vertices, faces, points, ends, ids = fixture()
        expected_points, expected_chords = originals(vertices, faces, points, ends, ids)
        observed = []
        original = triangle_batch.points

        def inspect(*args):
            result = original(*args)
            observed.append(int(result.sum()))
            return result

        with patch.object(triangle_batch, 'points', inspect):
            actual_points = certificates.points_on_triangles(points, vertices, faces, ids)
        actual_chords = certificates.chords_on_triangles(points, ends, vertices, faces, ids)
        self.assertGreater(observed[0], 0)
        np.testing.assert_array_equal(actual_points, expected_points)
        np.testing.assert_array_equal(actual_chords, expected_chords)
        for i in np.flatnonzero(actual_points):
            self.assertTrue(surface_trajectory._point_on_triangle(points[i], vertices, faces[ids[i]]), i)
        for i in np.flatnonzero(actual_chords):
            self.assertTrue(quality_constrained._supported_chord(points[i], ends[i], faces, vertices, [int(ids[i])]), i)

    def test_random_order_strides_duplicates_and_bounded_chunks_match(self):
        vertices, faces, points, ends, ids = fixture(120)
        order = np.random.default_rng(51).permutation(len(points))
        points, ends, ids = points[order], ends[order], ids[order]
        p = np.zeros((len(points), 6));q = p.copy()
        p[:, ::2], q[:, ::2] = points, ends
        points, ends = p[:, ::2], q[:, ::2]
        expected = originals(vertices, faces, points, ends, ids)
        with patch.object(triangle_batch, '_BLOCK', 11), patch.object(certificates, '_FACE_BLOCK', 13):
            np.testing.assert_array_equal(certificates.points_on_triangles(points, vertices, faces, ids), expected[0])
            np.testing.assert_array_equal(certificates.chords_on_triangles(points, ends, vertices, faces, ids), expected[1])

    def test_nonstandard_dtype_error_policy_and_nonfinite_keep_old_decline(self):
        vertices, faces, points, ends, ids = fixture()
        for values in (points.astype(np.float32), points.copy()):
            if values.dtype == np.float64:
                values[0, 0] = np.nan
            self.assertFalse(certificates.points_on_triangles(values, vertices, faces, ids).any())
        with np.errstate(over='raise'):
            self.assertFalse(certificates.points_on_triangles(points, vertices, faces, ids).any())
            self.assertFalse(certificates.chords_on_triangles(points, ends, vertices, faces, ids).any())

    def test_changed_mesh_has_fresh_preparation_and_matches_original(self):
        vertices, faces, points, ends, ids = fixture()
        certificates.points_on_triangles(points, vertices, faces, ids)
        vertices[:, 2] += .002
        expected = originals(vertices, faces, points, ends, ids)
        np.testing.assert_array_equal(certificates.points_on_triangles(points, vertices, faces, ids), expected[0])
        np.testing.assert_array_equal(certificates.chords_on_triangles(points, ends, vertices, faces, ids), expected[1])

    def test_cancel_during_preparation_and_during_cross_face_chunks(self):
        vertices, faces, points, ends, ids = fixture()
        for stop in (3, len(faces)+3):
            calls = 0

            def check():
                nonlocal calls
                calls += 1
                if calls == stop:
                    raise RuntimeError('cancel certificate')

            with patch.object(triangle_batch, '_BLOCK', 11):
                with self.assertRaisesRegex(RuntimeError, 'cancel certificate'):
                    certificates.points_on_triangles(points, vertices, faces, ids, check=check)

    def test_patched_solver_remains_authoritative(self):
        vertices, faces, points, ends, ids = fixture()
        with patch.object(TriangleSolve, 'point_inside', lambda self, xyz, face, tolerance: np.zeros(len(xyz), bool)):
            self.assertFalse(certificates.points_on_triangles(points, vertices, faces, ids).any())
        with patch.object(TriangleSolve, 'chord_inside', lambda self, a, b, face: np.zeros(len(a), bool)):
            self.assertFalse(certificates.chords_on_triangles(points, ends, vertices, faces, ids).any())
        with patch.object(TriangleSolve, 'coordinates', return_value=None):
            self.assertFalse(certificates.points_on_triangles(points, vertices, faces, ids).any())
            self.assertFalse(certificates.chords_on_triangles(points, ends, vertices, faces, ids).any())

    def test_large_query_budget_retains_original_unbatched_result(self):
        vertices, faces, points, ends, ids = fixture()
        expected = originals(vertices, faces, points, ends, ids)
        with patch.object(certificates, '_MAX_QUERY_ROWS', 100), \
                patch.object(triangle_batch, '_table', side_effect=AssertionError('oversized batch')):
            np.testing.assert_array_equal(certificates.points_on_triangles(points, vertices, faces, ids), expected[0])
            np.testing.assert_array_equal(certificates.chords_on_triangles(points, ends, vertices, faces, ids), expected[1])

    def test_optional_table_and_partial_batch_memory_failure_use_original_predicates(self):
        vertices, faces, points, ends, ids = fixture()
        expected = originals(vertices, faces, points, ends, ids)
        with patch.object(triangle_batch.np, 'full', side_effect=MemoryError('optional table')):
            np.testing.assert_array_equal(certificates.points_on_triangles(points, vertices, faces, ids), expected[0])
            np.testing.assert_array_equal(certificates.chords_on_triangles(points, ends, vertices, faces, ids), expected[1])
        original = triangle_batch._point_block
        calls = 0

        def fail_after_first(*args):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise MemoryError('optional later block')
            return original(*args)

        with patch.object(triangle_batch, '_BLOCK', 11), patch.object(triangle_batch, '_point_block', fail_after_first):
            np.testing.assert_array_equal(certificates.points_on_triangles(points, vertices, faces, ids), expected[0])
        self.assertGreaterEqual(calls, 2)

    def test_callback_memory_errors_are_not_swallowed_by_optional_fallback(self):
        vertices, faces, points, ends, ids = fixture()
        calls = 0

        def check():
            nonlocal calls
            calls += 1
            if calls > len(faces):
                raise MemoryError('caller cancellation or resource failure')

        with self.assertRaisesRegex(MemoryError, 'caller cancellation'):
            certificates.points_on_triangles(points, vertices, faces, ids, check=check)


if __name__ == '__main__':
    unittest.main()
