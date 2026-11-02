"""Operation-local face preparation keeps the existing numerical predicates."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from femto.geometry.triangle_support import prepare_triangle_faces
from femto.planning import quality_constrained as quality
from femto.planning.surface_trajectory import _point_on_triangle, _validate_scan_points, ScanSegment
from test_refinement_candidates_v2 import evidence


class PreparedTriangleSupportTests(unittest.TestCase):
    def test_point_decisions_equal_unprepared_at_edges_planes_and_degeneracies(self):
        rng = np.random.default_rng(20261024)
        triangles = [np.zeros((3, 3)),
                     np.array([[0., 0., .5], [1., 0., .5], [2., 0., .5]]),
                     np.array([[0., 0., .5], [1., 0., .5], [1., 1e-14, .5]])]
        for scale in (1e-9, 1e-5, 1., 1e5):
            triangles.extend(rng.normal(size=(20, 3, 3))*scale)
        for triangle in triangles:
            vertices, faces, prepared = prepare_triangle_faces(triangle, [[0, 1, 2]])
            normal = np.cross(triangle[1]-triangle[0], triangle[2]-triangle[0])
            length = np.linalg.norm(normal)
            direction = normal/length if length else np.array([0., 0., 1.])
            for tolerance in (1e-9, 1e-6, 1e-3):
                for weights in ([1., 0., 0.], [0., .5, .5], [1/3]*3,
                                [-1e-8, .5, .50000001], [-.1, .5, .6]):
                    point = np.asarray(weights) @ triangle
                    for offset in (0., np.nextafter(tolerance, 0.), tolerance,
                                   np.nextafter(tolerance, np.inf), 2*tolerance):
                        sample = point+direction*offset
                        self.assertEqual(
                            _point_on_triangle(sample, vertices, faces[0], tolerance),
                            _point_on_triangle(sample, vertices, faces[0], tolerance,
                                               _prepared=prepared(0)))

    def test_chord_intervals_equal_for_folded_missing_and_repeated_faces(self):
        rng = np.random.default_rng(753)
        vertices = np.array([[-.02, -.02, .5], [.02, -.02, .5], [.02, .02, .5],
                             [-.02, .02, .5], [0., 0., .52], [.04, .02, .5]])
        faces = np.array([[0, 1, 2], [0, 2, 3], [1, 2, 4], [1, 2, 5], [0, 0, 1]])
        vertices, faces, prepared = prepare_triangle_faces(vertices, faces)
        for _ in range(400):
            endpoints = np.c_[rng.uniform(-.03, .03, (2, 2)), np.full(2, .5)]
            endpoints[:, 2] += rng.choice([0., 1e-6, np.nextafter(1e-6, np.inf), .005], 2)
            ids = rng.choice(len(faces), rng.integers(0, 8)).tolist()
            self.assertEqual(quality._supported_chord(*endpoints, faces, vertices, ids),
                             quality._supported_chord(*endpoints, faces, vertices, ids,
                                                      _prepared_faces=prepared))

    def test_snapshot_is_frozen_bounded_and_fresh_for_each_operation(self):
        original = np.array([[0., 0., .5], [.01, 0., .5], [0., .01, .5]])
        original_faces = np.tile([0, 1, 2], (4100, 1))
        vertices, faces, prepared = prepare_triangle_faces(original, original_faces)
        for index in range(len(faces)):
            face = prepared(index)
            self.assertFalse(face.matrix.flags.writeable)
        self.assertEqual(prepared.cache_info().currsize, 4096)
        self.assertFalse(vertices.flags.writeable)
        self.assertFalse(faces.flags.writeable)
        original[:, 2] += 1.
        original_faces[:] = [0, 0, 0]
        np.testing.assert_array_equal(prepared(0).origin, [0., 0., .5])
        _, _, fresh = prepare_triangle_faces(original, original_faces)
        self.assertEqual(fresh(0).normal_length, 0.)
        self.assertEqual(fresh.cache_info().currsize, 1)

    def test_preparation_never_caches_point_or_chord_acceptance(self):
        vertices, faces, prepared = prepare_triangle_faces(
            [[0., 0., .5], [.01, 0., .5], [0., .01, .5]], [[0, 1, 2]])
        point = np.array([.002, .002, .5])
        with patch.object(np.linalg, 'lstsq', wraps=np.linalg.lstsq) as solve:
            for _ in range(2):
                self.assertTrue(_point_on_triangle(point, vertices, faces[0], _prepared=prepared(0)))
            # A shared thin QR map handles these interior points. Acceptance
            # is still recalculated from the current coordinates on each call.
            self.assertEqual(solve.call_count, 0)
        self.assertFalse(_point_on_triangle(point+[0., 0., .001], vertices, faces[0],
                                           _prepared=prepared(0)))
        self.assertTrue(quality._supported_chord(point, point, faces, vertices, [0],
                                               _prepared_faces=prepared))
        self.assertFalse(quality._supported_chord(point, point+[0., 0., .001], faces, vertices, [0],
                                                _prepared_faces=prepared))

    def test_invalid_triangle_indices_still_fail_before_preparation(self):
        vertices, faces, prepared = prepare_triangle_faces(
            [[0., 0., .5], [.01, 0., .5], [0., .01, .5]], [[0, 1, 2]])
        for index in (-1, 1, True, 0.0):
            point = dict(xyz_m=[.002, .002, .5], geometry_support='measured_mesh',
                         row_index=0, route_index=0, triangle_index=index)
            with self.assertRaises(ValueError):
                _validate_scan_points(ScanSegment(0, 0, [point]), 'mesh-slice-serpentine-v1',
                    False, False, 0, .001, vertices, faces, _prepared_faces=prepared)
            self.assertEqual(prepared.cache_info().currsize, 0)

    def test_edge_results_and_diagnostics_match_unprepared_chord_path(self):
        mesh = SimpleNamespace(
            vertices=np.array([[-.05, -.05, .5], [.05, -.05, .5], [.05, .05, .5], [-.05, .05, .5]]),
            faces=np.array([[0, 1, 2], [0, 2, 3]]))
        record = evidence(np.ones((100, 100), np.uint8))
        segments = []
        for z in (.5, .500001, .501):
            points = [dict(xyz_m=[x, -.005, z], triangle_index=0, role='scan', row_index=0)
                      for x in (-.001, .001)]
            segments.append(ScanSegment(0, 0, points, 0))
        diagnostics, expected_diagnostics = [], []
        actual = quality.edge_checks(segments, mesh, record, 1., diagnostics=diagnostics)
        old = quality._supported_chord
        def unprepared(*args, **kwargs):
            kwargs.pop('_prepared_faces', None)
            return old(*args, **kwargs)
        with patch.object(quality, '_supported_chord', side_effect=unprepared):
            expected = quality.edge_checks(segments, mesh, record, 1., diagnostics=expected_diagnostics)
        np.testing.assert_array_equal(actual[0], expected[0])
        self.assertEqual(actual[1], expected[1])
        np.testing.assert_array_equal(actual[2], expected[2])
        self.assertEqual(diagnostics, expected_diagnostics)
        frozen = quality._prepare_edge_checks(mesh, record, 1., lambda: None, None)
        before = frozen(segments)
        mesh.vertices[:, 2] += 1.
        after = frozen(segments)
        for index in (0, 2): np.testing.assert_array_equal(before[index], after[index])
        self.assertEqual(before[1], after[1])
        self.assertFalse(quality.edge_checks(segments, mesh, record, 1.)[0].any())

    def test_identical_rectangle_reuse_preserves_residual_work_and_limit(self):
        record = evidence(np.ones((100, 100), np.uint8))
        gate = quality.ReferenceEnvelope(record, 5.)
        mid = np.array([0., 0., .5])
        with patch.object(quality, '_rectangle', wraps=quality._rectangle) as rectangle:
            actual = gate.probe(mid, .001, details=True)
            self.assertEqual(rectangle.call_count, 1)
        self.assertTrue(actual[0])
        self.assertGreater(gate.sampled_pixels, 0)
        gate.sampled_pixels = 100000000
        with self.assertRaisesRegex(ValueError, '超过预算'):
            gate.probe(mid, .001)
        different = quality.ReferenceEnvelope(deepcopy(record), 5.)
        different.global_residual = .01
        with patch.object(quality, '_rectangle', wraps=quality._rectangle) as rectangle:
            self.assertTrue(different.probe(mid, .001)[0])
            self.assertEqual(rectangle.call_count, 2)


if __name__ == '__main__':
    unittest.main()
