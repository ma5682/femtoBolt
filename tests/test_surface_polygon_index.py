from types import SimpleNamespace
import unittest

import numpy as np

from femto.planning.mesh_slice_planner import _point_in_polygon, _project
from femto.planning.surface_polygon_index import PreparedPolygonIndex
from femto.planning.surface_row_turns import BoundaryTurns, SurfaceLinks, _intersects, edge_pixels_allowed


def scalar_blocks(polygons, a, b):
    a, b = np.asarray(a), np.asarray(b)
    return any(_point_in_polygon(a, polygon) or _point_in_polygon(b, polygon) or
               any(_intersects(a, b, c, d) for c, d in zip(polygon, np.roll(polygon, -1, axis=0)))
               for polygon in polygons)


class SurfacePolygonIndexTests(unittest.TestCase):
    def polygons(self):
        return [np.asarray(points, dtype=float) for points in (
            [(0, 0), (3, 0), (3, 3), (0, 3)],
            [(0, 0), (3, 0), (3, 1), (1, 1), (1, 3), (0, 3)],
            [(0, 0), (3, 3), (0, 3), (3, 0)],
            # Two loops with the doubled contour bridge used by legacy masks.
            [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0), (3, 0),
             (4, 0), (4, 1), (3, 1), (3, 0), (0, 0)],
            [(0, 0), (0, 0), (1, 0), (2, 0), (2, 2), (0, 2), (0, 2)],
            [(0, 0), (1, 0), (2, 0)],
        )]

    def assert_equivalent(self, polygons, points, segments):
        index = PreparedPolygonIndex(polygons)
        for point in points:
            self.assertEqual(index.contains(point), any(_point_in_polygon(point, p) for p in polygons),
                             msg=f'point {point}')
        for a, b in segments:
            self.assertEqual(index.blocks_segment(a, b), scalar_blocks(polygons, a, b),
                             msg=f'segment {a} -> {b}')

    def test_deterministic_random_concave_self_crossing_and_repeated_edges(self):
        rng = np.random.default_rng(23092026)
        polygons = self.polygons() + [rng.uniform(-1, 4, (count, 2)) for count in (3, 7, 32, 100)]
        points = rng.uniform(-2, 5, (160, 2))
        segments = rng.uniform(-2, 5, (240, 2, 2))
        for polygon in polygons:
            with self.subTest(vertices=len(polygon)):
                self.assert_equivalent([polygon], points, segments)
        self.assert_equivalent(polygons, points, segments)

    def test_vertices_edge_points_zero_length_and_epsilon_neighbors(self):
        for scale in (1e-6, 1., 1e6):
            for base in self.polygons():
                polygon = base * scale
                edge_midpoints = (polygon + np.roll(polygon, -1, axis=0)) / 2
                points = np.concatenate((polygon, edge_midpoints))
                points = np.concatenate([points + [offset, 0] for offset in
                                         (0., -2e-9, -.5e-9, -.5e-12, .5e-12, .5e-9, 2e-9)])
                segments = [(p, p) for p in points]
                segments += list(zip(polygon, np.roll(polygon, -1, axis=0)))
                segments += [(p, q) for p, q in zip(points, np.roll(points, 1, axis=0))]
                with self.subTest(scale=scale, vertices=len(polygon)):
                    self.assert_equivalent([polygon], points, segments)

    def test_empty_degenerate_reversed_and_outside_segments(self):
        for polygons in ([], [np.empty((0, 2))], [np.array([[0., 0.]])],
                         [np.array([[0., 0.], [1., 0.]])], self.polygons()):
            points = np.array([[0., 0.], [-1., 0.], [1., 0.], [100., 100.]])
            self.assert_equivalent(polygons, points, [(p, q) for p in points for q in points])

    def make_routers(self):
        vertices = np.array([[-.02, -.02, 1.], [.02, -.02, 1.], [.02, .02, 1.], [-.02, .02, 1.]])
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        mesh = SimpleNamespace(vertices=vertices, faces=faces)
        region = {'view_reference': {'undistorted_intrinsic': [[100., 0., 50.], [0., 100., 50.], [0., 0., 1.]],
                                     'image_size': [100, 100]},
                  'forbidden_screen_polygon': [[49., 49.], [51., 49.], [51., 51.], [49., 51.]]}
        boundary = BoundaryTurns(mesh, region, None, lambda: None)
        links = SurfaceLinks(mesh, region, None, lambda: None, [0, 1], np.ones((100, 100), dtype=bool))
        return boundary, links

    def test_router_polygon_replacement_and_diagnostic_restoration_are_not_stale(self):
        boundary, links = self.make_routers()
        a, b = np.array([-.02, 0., 1.]), np.array([.02, 0., 1.])
        self.assertFalse(boundary.allowed(a, b))
        self.assertEqual(links._state(a, b), 2)
        for router in (boundary, links):
            original = router.polygons
            router.polygons = []
            if router is boundary:
                self.assertTrue(router.allowed(a, b))
            else:
                self.assertEqual(router._state(a, b), 0)
            router.polygons = original
        self.assertFalse(boundary.allowed(a, b))
        self.assertEqual(links._state(a, b), 2)

    def test_router_states_match_scalar_oracle_including_selection_and_image(self):
        boundary, links = self.make_routers()
        links.allowed[52:54, :] = False
        rng = np.random.default_rng(20260923)
        segments = rng.uniform(-.7, .7, (220, 2, 3))
        segments[:, :, 2] = 1.
        segments[:5, 0, 2] = -1.
        for a, b in segments:
            uv = [_project(p, links.k, links.size) for p in (a, b)]
            image_ok = all(p is not None for p in uv)
            blocked = image_ok and scalar_blocks(links.polygons, *uv)
            expected = (1 if not image_ok else 2 if blocked else
                        0 if edge_pixels_allowed(a, b, links.k, links.size, links.allowed) else 3)
            self.assertEqual(links._state(a, b), expected)
            self.assertEqual(boundary.allowed(a, b), image_ok and not blocked)

    def test_cancellation_is_still_checked_during_routing(self):
        boundary, links = self.make_routers()
        def cancel():
            raise RuntimeError('cancelled')
        boundary.check = links.check = cancel
        links.boundary.check = cancel
        left = {'xyz_m': [-.02, -.02, 1.], 'triangle_index': 0, 'row_index': 0}
        right = {'xyz_m': [.02, .02, 1.], 'triangle_index': 1, 'row_index': 1}
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):
            boundary.route(left, right, .001, np.array([0., 1., 0.]))
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):
            links.route(left, right, .001, np.array([0., 1., 0.]), adjacent_rows=True)


if __name__ == '__main__':
    unittest.main()
