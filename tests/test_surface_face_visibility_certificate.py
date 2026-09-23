"""Batch visibility is an exact certificate over already admitted fixed nodes."""
from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from femto.planning.surface_face_routing import _PortalSearch
from femto.planning.surface_polygon_index import PreparedPolygonIndex
from femto.planning.surface_trajectory import _point_on_triangle
from test_surface_face_routing import LEFT, RIGHT, pair, point, router
from test_surface_face_search_acceleration import EagerPortalSearch


def prepared_search(route=None, left=LEFT, right=RIGHT):
    route = route or router(allowed=np.ones((200, 200), dtype=bool))
    route._prepare_face_portals()
    return _PortalSearch(route, left, right)


class SurfaceFaceVisibilityCertificateTests(unittest.TestCase):
    def assert_same_path(self, actual, expected):
        if expected is None:
            self.assertIsNone(actual)
        else:
            self.assertIsNotNone(actual)
            np.testing.assert_array_equal(actual[0], expected[0])
            self.assertEqual(actual[1], expected[1])

    def assert_certified_pairs_are_originally_visible(self, search):
        certified = 0
        for face, nodes in search.face_nodes.items():
            if not search._face_visibility_certified(face, nodes):
                continue
            certified += 1
            for a in nodes:
                for b in nodes:
                    self.assertEqual(search.router._state(search.points[a], search.points[b]), 0)
        self.assertGreater(certified, 0)

    def test_all_ordered_pairs_and_full_result_match_original_search(self):
        route = router(allowed=np.ones((200, 200), dtype=bool))
        route._prepare_face_portals()
        eager = EagerPortalSearch(route, LEFT, RIGHT)
        fast = _PortalSearch(route, LEFT, RIGHT)
        self.assert_same_path(fast.run(), eager.run())
        np.testing.assert_array_equal(np.asarray(fast.points), np.asarray(eager.points))
        self.assertEqual(fast.face_nodes, eager.face_nodes)
        self.assert_certified_pairs_are_originally_visible(fast)
        self.assertLess(fast.budget.counts['visibility checks'], eager.budget.counts['visibility checks'])

    def test_perspective_nodes_and_distant_self_intersecting_polygon_are_exact(self):
        vertices, faces = pair()
        vertices[:, 2] = [.8, 1.1, 1.3, .9]
        endpoints = [dict(xyz_m=(np.array([.2, .5, .3]) @ vertices[face]).tolist(),
                          triangle_index=index, row_index=index) for index, face in enumerate(faces)]
        route = router(vertices, faces, allowed=np.ones((200, 200), dtype=bool),
                       polygons=[[[5., 5.], [15., 15.], [5., 15.], [15., 5.]]])
        search = prepared_search(route, *endpoints)
        self.assert_same_path(search.run(), EagerPortalSearch(route, *endpoints).run())
        self.assert_certified_pairs_are_originally_visible(search)

    def test_mask_hole_and_nearby_forbidden_boundary_require_original_checks(self):
        allowed = np.ones((200, 200), dtype=bool)
        allowed[100, 100] = False
        routes = [router(allowed=allowed),
                  router(allowed=np.ones((200, 200), dtype=bool),
                         polygons=[[[99., 95.], [101., 95.], [101., 105.], [99., 105.]]])]
        for route in routes:
            with self.subTest(forbidden=bool(route.polygons)):
                search = prepared_search(route)
                expected = EagerPortalSearch(route, LEFT, RIGHT).run()
                self.assert_same_path(search.run(), expected)
                self.assertTrue(search._face_certificates)
                self.assertFalse(any(search._face_certificates.values()))
                self.assertGreater(search.budget.counts['visibility checks'], 0)

    def test_direct_neighbors_without_endpoint_admission_use_original_predicate(self):
        search = prepared_search(right=point(-.005, 0., 0))
        search.face_nodes = {0: [0, 1]}
        search.node_faces = [(0,), (0,)]
        with mock.patch.object(search.router, '_state', return_value=3) as state:
            self.assertEqual(list(search.neighbors(0, 0., {0: 0.})), [])
        state.assert_called_once()
        self.assertEqual(search._face_certificates, {})

    def test_certificate_uses_actual_nodes_not_triangle_vertex_bounds(self):
        route = router(allowed=np.ones((500, 500), dtype=bool))
        route.vertices = np.array([[-1e-6, -1e-6, 1.], [1e-6, -1e-6, 1.],
                                   [-1e-6, 1e-6, 1.], [1e-6, 1e-6, 1.]])
        route.k = np.array([[1e8, 0., 250.], [0., 1e8, 250.], [0., 0., 1.]])
        route.size = np.array([500, 500])
        route.allowed[150, 375] = False
        search = _PortalSearch(route, point(0., -1e-6, 0), point(1.5e-6, -1e-6, 0))
        for xyz in search.points:
            self.assertTrue(_point_on_triangle(xyz, route.vertices, route.faces[0]))
            self.assertEqual(route._state(xyz, xyz), 0)
        search._endpoints_checked = True
        self.assertFalse(search._face_visibility_certified(0, [0, 1]))
        self.assertEqual(search.budget.counts['visibility checks'], 0)
        self.assertEqual(route._state(*search.points), 3)

    def test_image_border_and_original_pixel_walk_limit_are_not_bypassed(self):
        for first, last, size in [([.1, 3., 1.], [2., 3., 1.], [10, 10]),
                                  ([2., 3., 1.], [150002., 3., 1.], [200000, 10])]:
            with self.subTest(size=size):
                route = SimpleNamespace(check=lambda: None, k=np.eye(3), size=np.array(size),
                                        allowed=None, _polygon_index=PreparedPolygonIndex([]))
                search = _PortalSearch(route, dict(xyz_m=first, triangle_index=0),
                                       dict(xyz_m=last, triangle_index=0))
                search._endpoints_checked = True
                self.assertFalse(search._face_visibility_certified(0, [0, 1]))
                self.assertEqual(search.budget.counts['visibility checks'], 0)

    def test_boundary_box_test_retains_original_epsilon_and_is_not_containment(self):
        minimum, maximum = np.array([2., 2.]), np.array([3., 3.])
        for delta, expected in [(5e-10, False), (2e-9, True)]:
            index = PreparedPolygonIndex([np.array([[3. + delta, 2.], [4., 2.], [4., 3.]])])
            self.assertEqual(index.boundary_outside_box(minimum, maximum), expected)
        enclosing = PreparedPolygonIndex([np.array([[0., 0.], [5., 0.], [5., 5.], [0., 5.]])])
        self.assertTrue(enclosing.boundary_outside_box(minimum, maximum))
        self.assertTrue(enclosing.contains(minimum))
        self.assertFalse(enclosing.boundary_outside_box([np.nan, 0.], maximum))

    def test_success_is_charged_once_but_never_skips_final_sample_validation(self):
        search = prepared_search()
        path = search.run()
        self.assertIsNotNone(path)
        before = search.budget.counts['visibility checks']
        for face, nodes in search.face_nodes.items():
            self.assertTrue(search._face_visibility_certified(face, nodes))
            self.assertTrue(search._face_visibility_certified(face, nodes))
        self.assertEqual(search.budget.counts['visibility checks'], before)
        with mock.patch.object(search.router, '_state', return_value=3) as state:
            self.assertIsNone(search.router._face_samples(*path, RIGHT, .001))
        state.assert_called_once()

    def test_certificate_obeys_cancellation_and_visibility_budget(self):
        search = prepared_search()
        nodes = search.nodes_on_face(0)
        for xyz in search.points:
            self.assertEqual(search.router._state(xyz, xyz), 0)
        search._endpoints_checked = True
        with mock.patch('femto.planning.surface_face_routing.POINT_BUDGET', 0):
            with self.assertRaisesRegex(ValueError, '0 visibility checks'):
                search._face_visibility_certified(0, nodes)
        self.assertNotIn(0, search._face_certificates)
        search.budget.check = mock.Mock(side_effect=RuntimeError('cancel requested'))
        with self.assertRaisesRegex(RuntimeError, 'cancel requested'):
            search._face_visibility_certified(0, nodes)


if __name__ == '__main__':
    unittest.main()
