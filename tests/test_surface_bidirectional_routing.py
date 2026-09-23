"""Directed bidirectional portal search against an independent Dijkstra oracle."""
from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
import heapq
import math
import unittest
from unittest import mock

import numpy as np

from femto.planning.surface_bidirectional_routing import (
    BidirectionalFacePortalSurfaceLinks, _BidirectionalPortalSearch,
)
from femto.planning.surface_link_optimization import OptimizedSurfaceLinks
from femto.planning.surface_trajectory import POINT_BUDGET
import test_surface_face_routing as face_fixtures
from test_surface_face_routing import FRAME, LEFT, RIGHT, router


class FixedDirectedGraph(_BidirectionalPortalSearch):
    """Finite face-node fixture; only visibility is synthetic, edge lengths are XYZ."""

    def __init__(self, points, edges):
        points = np.asarray(points, float)
        self.edges = set(edges)
        self.checked_edges = []
        self.expansions = []
        indices = {tuple(point): index for index, point in enumerate(points)}

        def state(a, b):
            edge = indices[tuple(a)], indices[tuple(b)]
            self.checked_edges.append(edge)
            return 0 if edge[0] == edge[1] or edge in self.edges else 3

        route = SimpleNamespace(check=lambda: None, member={0}, _state=state,
                                vertices=points, faces=np.array([[0, 1, 2]]))
        endpoints = [dict(xyz_m=point.tolist(), triangle_index=0) for point in points[:2]]
        super().__init__(route, *endpoints)
        self.points = [point.copy() for point in points]
        self.node_faces = [(0,)] * len(points)
        self.face_nodes = {0: list(range(len(points)))}
        self.budget.consume('graph nodes', len(points) - 2)

    def _face_visibility_certified(self, face, nodes):
        return False  # Synthetic predicates have no image/polygon certificate.

    def neighbors(self, node, distance, distances, *, reverse=False):
        self.expansions.append((reverse, node))
        yield from super().neighbors(node, distance, distances, reverse=reverse)

    def checked_run(self):
        # This isolated graph fixture has no real supporting surface. All actual
        # mesh integration tests below retain the real triangle validator.
        with mock.patch('femto.planning.surface_bidirectional_routing._point_on_triangle', return_value=True):
            return self.run()


def dijkstra_cost(points, edges):
    """Independent one-way oracle, without potentials, portals or shared search code."""
    adjacency = {node: [] for node in range(len(points))}
    for a, b in edges:
        adjacency[a].append((b, float(np.linalg.norm(points[b] - points[a]))))
    distances, queue = {0: 0.}, [(0., 0)]
    while queue:
        cost, node = heapq.heappop(queue)
        if cost != distances.get(node):
            continue
        if node == 1:
            return cost
        for other, length in adjacency[node]:
            candidate = cost + length
            if candidate < distances.get(other, math.inf):
                distances[other] = candidate
                heapq.heappush(queue, (candidate, other))
    return math.inf


class SurfaceBidirectionalRoutingTests(unittest.TestCase):
    def assert_oracle(self, search, path):
        expected = dijkstra_cost(np.asarray(search.points), search.edges)
        if not math.isfinite(expected):
            self.assertIsNone(path)
            return
        self.assertIsNotNone(path)
        points, faces = path
        np.testing.assert_array_equal(points[0], search.points[0])
        np.testing.assert_array_equal(points[-1], search.points[1])
        indices = {tuple(point): index for index, point in enumerate(search.points)}
        for a, b in zip(points, points[1:]):
            self.assertIn((indices[tuple(a)], indices[tuple(b)]), search.edges)
        self.assertEqual(faces, [0] * (len(points) - 1))
        self.assertAlmostEqual(float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()), expected, places=12)

    def test_random_directed_graphs_match_independent_dijkstra(self):
        rng = np.random.default_rng(9324)
        for index in range(60):
            points = np.column_stack((rng.uniform(-1., 1., (8, 2)), np.ones(8)))
            edges = {(a, b) for a in range(8) for b in range(8)
                     if a != b and rng.random() < .27}
            with self.subTest(index=index):
                search = FixedDirectedGraph(points, edges)
                self.assert_oracle(search, search.checked_run())

    def test_first_contact_can_be_longer_than_the_final_shortest_path(self):
        points = np.array([[0., 0., 1.], [10., 0., 1.], [5., 5., 1.], [5., 1., 1.]])
        search = FixedDirectedGraph(points, {(0, 2), (2, 1), (0, 3), (3, 1)})
        path = search.checked_run()
        self.assert_oracle(search, path)
        np.testing.assert_array_equal(path[0], points[[0, 3, 1]])
        # The backward target expansion tests and finds the longer meeting at 2
        # before discovering 3. Stopping at that first contact would be wrong.
        self.assertLess(search.checked_edges.index((2, 1)), search.checked_edges.index((3, 1)))
        self.assertGreater(2 * np.linalg.norm(points[2] - points[0]),
                           2 * np.linalg.norm(points[3] - points[0]))

    def test_reverse_expansion_checks_incoming_forward_direction_and_separate_cache_keys(self):
        search = FixedDirectedGraph([[0., 0., 1.], [1., 0., 1.], [2., 1., 1.]], {(0, 1)})
        backward = list(search.neighbors(1, 0., {1: 0.}, reverse=True))
        forward = list(search.neighbors(1, 0., {1: 0.}, reverse=False))
        self.assertEqual([node for node, *_ in backward], [0])
        self.assertEqual(forward, [])
        self.assertTrue(search.visibility[(0, 1, 0)])
        self.assertFalse(search.visibility[(1, 0, 0)])
        self.assertEqual(search.budget.counts['visibility checks'], 4)
        before = len(search.checked_edges)
        list(search.neighbors(1, 0., {1: 0.}, reverse=True))
        self.assertEqual(len(search.checked_edges), before)

    def test_small_unreachable_goal_component_exhausts_with_shared_small_budget(self):
        points = [[0., 0., 1.], [10., 0., 1.], [10., .001, 1.]]
        points.extend([[.001 * index, 3. + .01 * index, 1.] for index in range(117)])
        source = [0, *range(3, len(points))]
        edges = {(a, b) for a in source for b in source if a != b} | {(1, 2), (2, 1)}
        with mock.patch('femto.planning.surface_face_routing.POINT_BUDGET', 500):
            search = FixedDirectedGraph(points, edges)
            self.assertIsNone(search.checked_run())
        self.assertEqual([node for reverse, node in search.expansions if not reverse], [0])
        self.assertEqual([node for reverse, node in search.expansions if reverse], [1, 2])
        self.assertLess(search.budget.counts['visibility checks'], 500)

    def test_actual_interior_portal_samples_remain_supported_and_deterministic(self):
        route = router(cls=BidirectionalFacePortalSurfaceLinks)
        before = [route.vertices.copy(), route.faces.copy(), route.allowed.copy(), deepcopy(LEFT), deepcopy(RIGHT)]
        first = route.route(LEFT, RIGHT, .001, FRAME['e2'], adjacent_rows=False)
        self.assertEqual(first[1], 'surface_face_portals_bidirectional')
        face_fixtures.SurfaceFaceRoutingTests.assert_supported(self, route, first[0])
        for _ in range(2):
            self.assertEqual(route.route(LEFT, RIGHT, .001, FRAME['e2'], adjacent_rows=False), first)
        for actual, expected in zip([route.vertices, route.faces, route.allowed], before[:3]):
            np.testing.assert_array_equal(actual, expected)
        self.assertEqual(LEFT, before[3])
        self.assertEqual(RIGHT, before[4])

    def test_v41_success_is_returned_unchanged_without_preparing_portals(self):
        allowed = np.ones((200, 200), dtype=bool)
        expected = router(cls=OptimizedSurfaceLinks, allowed=allowed).route(
            LEFT, RIGHT, .001, FRAME['e2'], adjacent_rows=False)
        self.assertTrue(expected[0])
        route = router(cls=BidirectionalFacePortalSurfaceLinks, allowed=allowed)
        with mock.patch.object(route, '_prepare_face_portals', side_effect=AssertionError('unneeded fallback')):
            self.assertEqual(route.route(LEFT, RIGHT, .001, FRAME['e2'], adjacent_rows=False), expected)

    def test_final_forward_sampling_rejects_a_newly_invalid_segment(self):
        route = router(cls=BidirectionalFacePortalSurfaceLinks)
        route._prepare_face_portals()
        path = _BidirectionalPortalSearch(route, LEFT, RIGHT).run()
        self.assertIsNotNone(path)
        with mock.patch.object(OptimizedSurfaceLinks, 'route', return_value=(None, None)), \
                mock.patch.object(_BidirectionalPortalSearch, 'run', return_value=path), \
                mock.patch.object(route, '_state', return_value=3) as state:
            self.assertEqual(route.route(LEFT, RIGHT, .001, FRAME['e2'], adjacent_rows=False), (None, None))
        state.assert_called_once()

    def test_both_directions_share_and_enforce_visibility_cap(self):
        search = FixedDirectedGraph([[0., 0., 1.], [1., 0., 1.], [2., 1., 1.]], {(0, 1)})
        with mock.patch('femto.planning.surface_face_routing.POINT_BUDGET', 2):
            list(search.neighbors(0, 0., {0: 0.}))
            with self.assertRaisesRegex(ValueError, '2 visibility checks'):
                list(search.neighbors(1, 0., {1: 0.}, reverse=True))
        self.assertEqual(search.budget.counts['visibility checks'], 3)

    def test_existing_candidate_subdivision_and_graph_node_caps_are_preserved(self):
        for kind in ('portal candidates', 'pixel subdivisions', 'graph nodes'):
            with self.subTest(kind=kind):
                route = router(cls=BidirectionalFacePortalSurfaceLinks)
                route._prepare_face_portals()
                search = _BidirectionalPortalSearch(route, LEFT, RIGHT)
                search.budget.counts[kind] = POINT_BUDGET
                with self.assertRaisesRegex(ValueError, kind):
                    search.run()

    def test_existing_final_sample_cap_is_preserved(self):
        route = router(cls=BidirectionalFacePortalSurfaceLinks)
        route._prepare_face_portals()
        path = _BidirectionalPortalSearch(route, LEFT, RIGHT).run()
        with mock.patch('femto.planning.surface_face_routing.POINT_BUDGET', 2):
            with self.assertRaisesRegex(ValueError, '2 samples'):
                route._face_samples(*path, RIGHT, .001)

    def test_cancellation_propagates_in_search_and_final_path_reconstruction(self):
        search = FixedDirectedGraph([[0., 0., 1.], [1., 0., 1.], [2., 1., 1.]], {(0, 1)})
        search.budget.check = mock.Mock(side_effect=RuntimeError('cancel requested'))
        with self.assertRaisesRegex(RuntimeError, 'cancel requested'):
            search.checked_run()
        with self.assertRaisesRegex(RuntimeError, 'cancel requested'):
            search._path(1, [{1: (0, 0)}, {}])


if __name__ == '__main__':
    unittest.main()
