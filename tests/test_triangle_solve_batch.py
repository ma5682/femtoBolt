"""Grouped interior checks retain scalar decisions and boundary fallbacks."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from femto.geometry.triangle_support import prepare_triangle_faces
from femto.geometry.triangle_solve import TriangleSolve
from femto.planning import quality_constrained as quality
from femto.planning.surface_trajectory import _point_on_triangle, _validate_scan_points, ScanSegment
from test_refinement_candidates_v2 import evidence


class TriangleSolveBatchTests(unittest.TestCase):
    def test_points_match_scalar_for_random_scale_translation_and_threshold_ulps(self):
        rng = np.random.default_rng(20261026)
        fast_count = 0
        for scale in (1e-8, 1e-4, .01, 1., 1e4):
            for translation in (np.zeros(3), np.array([1., -.7, .5]), np.array([1e6, -1e6, 1e6])):
                for _ in range(12):
                    triangle = rng.normal(size=(3, 3))*scale + translation
                    vertices, faces, prepared = prepare_triangle_faces(triangle, [[0, 1, 2]])
                    pr = prepared(0)
                    normal = pr.normal/pr.normal_length
                    for tolerance in (1e-9, 1e-6, 1e-3):
                        bary_tolerance = tolerance/max(pr.minimum_edge_length, tolerance)
                        weights = rng.uniform(-.2, .8, (40, 2))
                        boundaries = np.array([-bary_tolerance, 1+bary_tolerance])
                        near = np.r_[np.nextafter(boundaries, -np.inf), boundaries,
                                     np.nextafter(boundaries, np.inf)]
                        weights = np.vstack((weights, np.c_[near, np.zeros(len(near))]))
                        points = (pr.origin + weights[:, 0, None]*pr.ab + weights[:, 1, None]*pr.ac)
                        offsets = np.array([0., tolerance, np.nextafter(tolerance, 0.),
                                            np.nextafter(tolerance, np.inf), 2*tolerance])
                        points += normal * np.resize(offsets, len(points))[:, None]
                        expected = [_point_on_triangle(p, vertices, faces[0], tolerance) for p in points]
                        actual = [_point_on_triangle(p, vertices, faces[0], tolerance, _prepared=pr)
                                  for p in points]
                        self.assertEqual(actual, expected)
                        fast = pr.solver.point_inside(points, pr, tolerance)
                        self.assertTrue(np.asarray(expected)[fast].all())
                        fast_count += int(fast.sum())
        self.assertGreater(fast_count, 200)

    def test_degenerate_and_ill_conditioned_faces_retain_original_solves(self):
        for triangle in (np.zeros((3, 3)),
                         np.array([[0., 0., .5], [1., 0., .5], [2., 0., .5]]),
                         np.array([[0., 0., .5], [1., 0., .5], [1., 1e-10, .5]])):
            vertices, faces, prepared = prepare_triangle_faces(triangle, [[0, 1, 2]])
            pr = prepared(0)
            points = np.array([triangle.mean(axis=0), triangle[0], triangle[1]])
            self.assertFalse(pr.solver.point_inside(points, pr, 1e-6).any())
            self.assertFalse(pr.solver.chord_inside(points, points[::-1], pr).any())
            for point in points:
                self.assertEqual(_point_on_triangle(point, vertices, faces[0]),
                                 _point_on_triangle(point, vertices, faces[0], _prepared=pr))

    def test_chord_fast_path_implies_full_original_interval_support(self):
        rng = np.random.default_rng(9207)
        count = 0
        for scale in (1e-5, .005, 1., 1e3):
            for _ in range(40):
                triangle = rng.normal(size=(3, 3))*scale + [0., 0., .5]
                vertices, faces, prepared = prepare_triangle_faces(triangle, [[0, 1, 2]])
                pr = prepared(0)
                weights = rng.uniform(-.05, .8, (100, 2, 2))
                weights[:6, 0, 0] = [-1e-8, np.nextafter(-1e-8, -np.inf),
                                     np.nextafter(-1e-8, 0.), 1e-8, 0., 1.]
                points = pr.origin + weights[..., 0, None]*pr.ab + weights[..., 1, None]*pr.ac
                direction = pr.normal/pr.normal_length
                points[:10] += direction*np.resize([1e-6, np.nextafter(1e-6, 0.),
                                                    np.nextafter(1e-6, np.inf), 0.], (10, 2))[:, :, None]
                fast = pr.solver.chord_inside(points[:, 0], points[:, 1], pr)
                count += int(fast.sum())
                for a, b in points[fast]:
                    self.assertTrue(quality._supported_chord(a, b, faces, vertices, [0]))
        self.assertGreater(count, 1000)

    def test_factorization_is_shared_but_queries_are_not_retained(self):
        vertices, faces, prepared = prepare_triangle_faces(
            [[0., 0., .5], [.01, 0., .5], [0., .01, .5]], [[0, 1, 2]])
        pr = prepared(0)
        points = np.tile([.002, .002, .5], (20, 1))
        with patch.object(np.linalg, 'lstsq', wraps=np.linalg.lstsq) as solve:
            self.assertTrue(pr.solver.point_inside(points, pr, 1e-6).all())
            self.assertTrue(pr.solver.chord_inside(points, points, pr).all())
            self.assertEqual(solve.call_count, 0)
            points[:, 2] += .01
            self.assertFalse(pr.solver.point_inside(points, pr, 1e-6).any())
            self.assertFalse(pr.solver.chord_inside(points, points, pr).any())
            self.assertEqual(solve.call_count, 0)
        self.assertFalse(pr.solver._inverse.flags.writeable)

    def test_condition_cutoff_scale_and_float32_keep_scalar_decisions(self):
        for condition in (1., 100., 500., 1000., 1100., 1e6, 1e12):
            for scale in (1e-12, 1e-6, .01, 1., 1e6, 1e12):
                triangle = np.array([[0., 0., .5], [scale, 0., .5], [0., scale/condition, .5]])
                vertices, faces, prepared = prepare_triangle_faces(triangle, [[0, 1, 2]])
                pr = prepared(0)
                weights = np.array([[.2, .2], [-1e-8, .5], [0., 0.], [.5, .5], [.2, .7]])
                points = pr.origin + weights[:, 0, None]*pr.ab + weights[:, 1, None]*pr.ac
                fast = pr.solver.point_inside(points, pr, 1e-6)
                expected = [_point_on_triangle(p, vertices, faces[0]) for p in points]
                self.assertTrue(np.asarray(expected)[fast].all())
                self.assertEqual(expected, [_point_on_triangle(p, vertices, faces[0], _prepared=pr)
                                             for p in points])
                for a in points:
                    for b in points:
                        self.assertEqual(quality._supported_chord(a, b, faces, vertices, [0]),
                                         quality._supported_chord(a, b, faces, vertices, [0],
                                                                  _prepared_faces=prepared))
                if condition >= 1100:
                    self.assertIsNone(pr.solver._inverse)
        vertices, faces, prepared = prepare_triangle_faces(
            np.array([[0., 0., .5], [.01, 0., .5], [0., .01, .5]], np.float32), [[0, 1, 2]])
        pr = prepared(0)
        point = np.array([.002, .002, .5])
        self.assertFalse(pr.solver.point_inside(point[None, :], pr, 1e-6).any())
        self.assertEqual(_point_on_triangle(point, vertices, faces[0]),
                         _point_on_triangle(point, vertices, faces[0], _prepared=pr))

    def test_unsupported_linalg_dtype_is_not_silently_accepted(self):
        # Windows longdouble aliases binary64; wider builds must retain the
        # original NumPy error instead of narrowing inputs in the fast path.
        if np.dtype(np.longdouble) == np.dtype(np.float64):
            return
        vertices, faces, prepared = prepare_triangle_faces(
            [[0., 0., .5], [.01, 0., .5], [0., .01, .5]], [[0, 1, 2]])
        point = np.array([.002, .002, .5], dtype=np.longdouble)
        with self.assertRaises(TypeError):
            quality._supported_chord(point, point, faces, vertices, [0])
        with self.assertRaises(TypeError):
            quality._supported_chord(point, point, faces, vertices, [0], _prepared_faces=prepared)

    def test_qr_overflow_underflow_and_unbalanced_columns_use_legacy_decisions(self):
        triangles = [np.array([[0., 0., 0.], [scale, 0., 0.], [0., scale, 0.]])
                     for scale in (1e-150, 1e150, 1e308)]
        triangles += [np.array([[0., 0., 0.], [1e200, 0., 0.], [0., 1e-200, 0.]]),
                      np.array([[0., 0., 0.], [1., 0., 0.], [1., np.nextafter(0., 1.), 0.]])]
        def outcome(fn):
            try:
                return ('result', fn())
            except (ValueError, TypeError, np.linalg.LinAlgError) as error:
                return ('error', type(error), str(error))
        with np.errstate(all='ignore'):
            for triangle in triangles:
                vertices, faces, prepared = prepare_triangle_faces(triangle, [[0, 1, 2]])
                pr = prepared(0)
                points = np.array([triangle[0], triangle[1]/4+triangle[2]/4,
                                   triangle[1]/2+triangle[2]/4])
                for point in points:
                    self.assertEqual(outcome(lambda: _point_on_triangle(point, vertices, faces[0])),
                                     outcome(lambda: _point_on_triangle(point, vertices, faces[0], _prepared=pr)))
                for a, b in zip(points, points[::-1]):
                    self.assertEqual(outcome(lambda: quality._supported_chord(a, b, faces, vertices, [0])),
                                     outcome(lambda: quality._supported_chord(a, b, faces, vertices, [0],
                                                                              _prepared_faces=prepared)))
                self.assertFalse(pr.solver.chord_inside(points, points[::-1], pr).any())

    def test_quality_batches_match_old_chords_with_holes_thresholds_and_diagnostics(self):
        mesh = SimpleNamespace(vertices=np.array([[-.05, -.05, .5], [.05, -.05, .5],
                                                  [.05, .05, .5], [-.05, .05, .5]]),
                               faces=np.array([[0, 1, 2], [0, 2, 3]]))
        record = evidence(np.ones((100, 100), np.uint8))
        rng = np.random.default_rng(125)
        segments = []
        for z in (.5, .500001, np.nextafter(.500001, np.inf), .501):
            for _ in range(30):
                xy = rng.uniform(-.02, .02, (2, 2))
                points = [dict(xyz_m=[*v, z], triangle_index=int(v[0] < v[1]),
                               role='scan', row_index=0) for v in xy]
                segments.append(ScanSegment(0, 0, points, 0))
        for width in (1., 5., 15.):
            diagnostics, old_diagnostics = [], []
            actual = quality.edge_checks(segments, mesh, record, width, diagnostics=diagnostics)
            original_chord = quality._supported_chord
            def unprepared_chord(*args, **kwargs):
                kwargs.pop('_prepared_faces', None)
                return original_chord(*args, **kwargs)
            with patch.object(TriangleSolve, 'chord_inside',
                              lambda self, starts, ends, prepared: np.zeros(len(starts), bool)), \
                    patch.object(quality, '_supported_chord', unprepared_chord):
                old = quality.edge_checks(segments, mesh, record, width, diagnostics=old_diagnostics)
            np.testing.assert_array_equal(actual[0], old[0])
            self.assertEqual(actual[1], old[1])
            np.testing.assert_array_equal(actual[2], old[2])
            self.assertEqual(diagnostics, old_diagnostics)

    def test_grouped_validation_retains_structural_failures_before_geometry(self):
        vertices, faces, prepared = prepare_triangle_faces(
            [[0., 0., .5], [.01, 0., .5], [0., .01, .5]], [[0, 1, 2]])
        mutations = [lambda p: p[0].update(geometry_support='invented'),
                     lambda p: p[0].update(route_index=3),
                     lambda p: p[0].update(triangle_index=-1),
                     lambda p: p[0].update(xyz_m=[float('nan'), 0., .5]),
                     lambda p: p[0].update(xyz_m=[10**1000, 0., .5]),
                     lambda p: p[0].update(pixel=[0., float('nan')]),
                     lambda p: p[0].update(role='robot_pose')]
        for change in mutations:
            points = [dict(xyz_m=[.002+i*.00001, .002, .5], triangle_index=0,
                           geometry_support='measured_mesh', row_index=0, route_index=i)
                      for i in range(16)]
            change(points)
            # A later invalid coordinate must not override an earlier error.
            points[-1]['xyz_m'] = [0., 0., .6]
            item = ScanSegment(0, 0, points)
            errors = []
            for enabled in (False, True):
                try:
                    _validate_scan_points(item, 'mesh-slice-serpentine-v1', False, False,
                                          0, .001, vertices, faces,
                                          _prepared_faces=prepared if enabled else None)
                except Exception as error:
                    errors.append((type(error), str(error)))
            self.assertEqual(len(errors), 2)
            self.assertEqual(errors[0], errors[1])


if __name__ == '__main__':
    unittest.main()
