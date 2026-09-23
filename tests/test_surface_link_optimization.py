"""Independent geometry checks for the versioned, measured-face connector shortening.

These tests concern a polyline on existing triangles, not execution safety or treatment
coverage. Endpoints and the allowed-domain constraints remain authoritative.
"""
from __future__ import annotations

import unittest

import numpy as np

from femto.planning.surface_link_optimization import optimize_surface_path
from femto.planning.surface_trajectory import _point_on_triangle


def length(points):
    return float(np.linalg.norm(np.diff(np.asarray(points), axis=0), axis=1).sum())


def planar_pair():
    vertices = np.array([[0., 0., 1.], [2., 0., 1.], [0., 2., 1.], [2., 2., 1.]])
    faces = np.array([[0, 1, 2], [1, 3, 2]])
    points = np.array([[.2, .2, 1.], [2., 0., 1.], [1.8, 1.8, 1.]])
    return vertices, faces, points, [0, 1]


def avoids_rectangle(bounds):
    """Closed rectangle obstruction, checked over the entire projected XY edge."""
    low = np.array(bounds[:2], dtype=float)
    high = np.array(bounds[2:], dtype=float)

    def allowed(p, q):
        start = np.asarray(p)[:2]
        delta = np.asarray(q)[:2] - start
        enter, leave = 0., 1.
        for axis in range(2):
            if delta[axis] == 0:
                if start[axis] < low[axis] or start[axis] > high[axis]:
                    return True
                continue
            interval = sorted(((low[axis] - start[axis]) / delta[axis],
                               (high[axis] - start[axis]) / delta[axis]))
            enter = max(enter, interval[0])
            leave = min(leave, interval[1])
            if enter > leave:
                return True
        return False

    return allowed


class SurfaceLinkOptimizationTests(unittest.TestCase):
    def assert_supported(self, vertices, faces, points, edge_faces, allowed=lambda p, q: True):
        self.assertEqual(len(edge_faces), len(points) - 1)
        self.assertTrue(np.isfinite(points).all())
        for p, q, face in zip(points[:-1], points[1:], edge_faces):
            self.assertTrue(_point_on_triangle(p, vertices, faces[face]))
            self.assertTrue(_point_on_triangle(q, vertices, faces[face]))
            self.assertTrue(allowed(p, q))

    def test_planar_pair_uses_the_analytic_shared_edge_crossing(self):
        vertices, faces, points, edge_faces = planar_pair()
        old_vertices, old_faces, old_points = vertices.copy(), faces.copy(), points.copy()
        old_edge_faces = list(edge_faces)
        result, supports = optimize_surface_path(vertices, faces, points, edge_faces, lambda p, q: True)
        self.assertLess(length(result), length(points))
        self.assertAlmostEqual(length(result), np.linalg.norm(points[-1] - points[0]), places=12)
        np.testing.assert_allclose(result[1], [1., 1., 1.], atol=1e-12)
        self.assert_supported(vertices, faces, result, supports)
        np.testing.assert_array_equal(result[0], points[0])
        np.testing.assert_array_equal(result[-1], points[-1])
        np.testing.assert_array_equal(vertices, old_vertices)
        np.testing.assert_array_equal(faces, old_faces)
        np.testing.assert_array_equal(points, old_points)
        self.assertEqual(edge_faces, old_edge_faces)

    def test_folded_pair_preserves_the_surface_corner_not_a_space_chord(self):
        vertices = np.array([[0., 0., 1.], [1., 0., 1.], [0., 1., 1.], [1., 0., 2.]])
        faces = np.array([[0, 1, 2], [0, 1, 3]])
        points = np.array([[.1, .6, 1.], [0., 0., 1.], [.9, 0., 1.6]])
        result, supports = optimize_surface_path(vertices, faces, points, [0, 1], lambda p, q: True)
        self.assertLess(length(result), length(points))
        self.assertGreater(length(result), np.linalg.norm(points[-1] - points[0]))
        self.assertEqual(len(result), 3)
        np.testing.assert_allclose(result[1], [.5, 0., 1.], atol=1e-12)
        self.assert_supported(vertices, faces, result, supports)

    def test_portal_minimum_is_clipped_to_the_shared_edge_endpoint(self):
        vertices = np.array([[0., 0., 1.], [1., 0., 1.], [-1., 1., 1.], [-1., 0., 2.]])
        faces = np.array([[0, 1, 2], [0, 1, 3]])
        points = np.array([[-.5, .6, 1.], [1., 0., 1.], [-.5, 0., 1.6]])
        result, supports = optimize_surface_path(vertices, faces, points, [0, 1], lambda p, q: True)
        self.assertLess(length(result), length(points))
        np.testing.assert_array_equal(result[1], vertices[0])
        self.assert_supported(vertices, faces, result, supports)

    def test_endpoint_on_the_portal_does_not_make_the_formula_singular(self):
        vertices = np.array([[0., 0., 1.], [1., 0., 1.], [0., 1., 1.], [1., 0., 2.]])
        faces = np.array([[0, 1, 2], [0, 1, 3]])
        points = np.array([[.8, 0., 1.], [0., 0., 1.], [.9, 0., 1.6]])
        result, supports = optimize_surface_path(vertices, faces, points, [0, 1], lambda p, q: True)
        self.assertLess(length(result), length(points))
        np.testing.assert_allclose(result[1], points[0], atol=1e-12)
        self.assert_supported(vertices, faces, result, supports)

        # Both endpoints on the portal: the minimum is a flat interval, not a
        # unique crossing. Keep the old supported point instead of dividing by zero.
        flat = np.array([[.2, 0., 1.], [.5, 0., 1.], [.8, 0., 1.]])
        result, supports = optimize_surface_path(vertices, faces, flat, [0, 1], lambda p, q: True)
        np.testing.assert_array_equal(result, flat)
        self.assert_supported(vertices, faces, result, supports)

    def test_a_same_face_shortcut_cannot_cross_a_forbidden_rectangle(self):
        vertices = np.array([[0., 0., 1.], [4., 0., 1.], [0., 4., 1.]])
        faces = np.array([[0, 1, 2]])
        points = np.array([[.2, .2, 1.], [2., .2, 1.], [.2, 2., 1.]])
        allowed = avoids_rectangle((.1, .8, .3, 1.2))
        self.assertTrue(allowed(points[0], points[1]))
        self.assertTrue(allowed(points[1], points[2]))
        self.assertFalse(allowed(points[0], points[2]))
        result, supports = optimize_surface_path(vertices, faces, points, [0, 0], allowed)
        np.testing.assert_array_equal(result, points)
        self.assert_supported(vertices, faces, result, supports, allowed)

    def test_an_illegal_shared_edge_minimum_keeps_the_old_legal_route(self):
        vertices, faces, points, edge_faces = planar_pair()
        allowed = avoids_rectangle((.8, .8, 1.2, 1.2))
        self.assertTrue(allowed(points[0], points[1]))
        self.assertTrue(allowed(points[1], points[2]))
        result, supports = optimize_surface_path(vertices, faces, points, edge_faces, allowed)
        np.testing.assert_array_equal(result, points)
        self.assert_supported(vertices, faces, result, supports, allowed)

    def test_same_face_compression_keeps_endpoints_and_support(self):
        vertices = np.array([[0., 0., 1.], [4., 0., 1.], [0., 4., 1.]])
        faces = np.array([[0, 1, 2]])
        points = np.array([[.2, .2, 1.], [2., .2, 1.], [.2, 2., 1.]])
        result, supports = optimize_surface_path(vertices, faces, points, [0, 0], lambda p, q: True)
        np.testing.assert_array_equal(result, points[[0, -1]])
        self.assertEqual(supports, [0])
        self.assertLess(length(result), length(points))
        self.assert_supported(vertices, faces, result, supports)

    def test_repeated_points_and_zero_edges_do_not_corrupt_face_assignment(self):
        vertices, faces, points, _ = planar_pair()
        repeated = points[[0, 1, 1, 2]]
        result, supports = optimize_surface_path(vertices, faces, repeated, [0, 0, 1], lambda p, q: True)
        self.assertLessEqual(length(result), length(repeated))
        np.testing.assert_array_equal(result[0], repeated[0])
        np.testing.assert_array_equal(result[-1], repeated[-1])
        self.assert_supported(vertices, faces, result, supports)
        all_zero = np.repeat(points[:1], 3, axis=0)
        result, supports = optimize_surface_path(vertices, faces, all_zero, [0, 0], lambda p, q: True)
        self.assertEqual(length(result), 0.)
        self.assert_supported(vertices, faces, result, supports)

    def test_nonmanifold_shared_edge_is_not_optimized(self):
        vertices, faces, points, edge_faces = planar_pair()
        vertices = np.vstack([vertices, [1., 1., 2.]])
        faces = np.vstack([faces, [1, 2, 4]])
        result, supports = optimize_surface_path(vertices, faces, points, edge_faces, lambda p, q: True)
        np.testing.assert_array_equal(result, points)
        self.assertEqual(supports, edge_faces)
        self.assert_supported(vertices, faces, result, supports)

    def test_faces_touching_at_one_vertex_are_not_shortcut(self):
        vertices = np.array([[0., 0., 1.], [-1., 0., 1.], [0., -1., 1.],
                             [1., 0., 1.], [0., 1., 1.]])
        faces = np.array([[0, 1, 2], [0, 3, 4]])
        points = np.array([[-.2, -.2, 1.], [0., 0., 1.], [.2, .2, 1.]])
        result, supports = optimize_surface_path(vertices, faces, points, [0, 1], lambda p, q: True)
        np.testing.assert_array_equal(result, points)
        self.assert_supported(vertices, faces, result, supports)

    def test_degenerate_support_does_not_trigger_a_shared_edge_move(self):
        vertices = np.array([[0., 0., 1.], [2., 0., 1.], [1., 0., 1.], [1., 1., 1.]])
        faces = np.array([[0, 1, 2], [1, 2, 3]])
        points = np.array([[0., 0., 1.], [2., 0., 1.], [1.3, .5, 1.]])
        result, supports = optimize_surface_path(vertices, faces, points, [0, 1], lambda p, q: True)
        np.testing.assert_array_equal(result, points)
        self.assertEqual(supports, [0, 1])

    def test_determinism_length_monotonicity_and_cancellation(self):
        vertices, faces, points, edge_faces = planar_pair()
        first, supports = optimize_surface_path(vertices, faces, points, edge_faces, lambda p, q: True)
        second, supports_again = optimize_surface_path(vertices, faces, points, edge_faces, lambda p, q: True)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(supports, supports_again)
        self.assertLessEqual(length(first), length(points))

        def cancelled():
            raise RuntimeError("cancel requested")

        with self.assertRaisesRegex(RuntimeError, "cancel requested"):
            optimize_surface_path(vertices, faces, points, edge_faces, lambda p, q: True, cancelled)


if __name__ == "__main__":
    unittest.main()
