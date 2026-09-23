"""Distance pruning preserves the frozen v4.2 portal graph and predicate order."""
from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from femto.planning.surface_face_routing import _PortalSearch
from tests.test_surface_face_routing import LEFT, RIGHT, pair, point, router, stripe


class EagerPortalSearch(_PortalSearch):
    """Pre-optimization neighbor enumeration, including first predicate direction."""

    def neighbors(self, node, distance, distances):
        for face in self.node_faces[node]:
            for other in self.nodes_on_face(face):
                self.budget.check()
                if other == node:
                    continue
                key = (min(node, other), max(node, other), face)
                if key not in self.visibility:
                    self.budget.consume('visibility checks')
                    self.visibility[key] = self.router._state(self.points[node], self.points[other]) == 0
                if self.visibility[key]:
                    yield other, float(np.linalg.norm(self.points[other] - self.points[node])), face


def dense_router(*, barrier=False):
    vertices = np.array([[-.08, -.08, 1.], [.08, -.08, 1.],
                         [-.08, .08, 1.], [.08, .08, 1.]])
    faces = np.array([[0, 1, 2], [1, 3, 2]])
    allowed = np.ones((200, 200), dtype=bool)
    if barrier:
        allowed[:104, 99:102] = False
    result = router(vertices, faces, allowed=allowed)
    result._prepare_face_portals()
    return result


def two_node_search(cls=_PortalSearch):
    # Explicitly materialize one face so these unit tests isolate relaxation and
    # cache direction without changing candidate generation or geometry rules.
    result = cls(router(), LEFT, point(-.005, 0., 0))
    result.face_nodes = {0: [0, 1]}
    result.node_faces = [(0,), (0,)]
    return result


class SurfaceFaceSearchAccelerationTests(unittest.TestCase):
    def assert_same_path(self, actual, expected):
        if expected is None:
            self.assertIsNone(actual)
        else:
            self.assertIsNotNone(actual)
            np.testing.assert_array_equal(actual[0], expected[0])
            self.assertEqual(actual[1], expected[1])

    def test_exact_points_and_faces_match_eager_for_masks_forbidden_and_same_face(self):
        unselected = stripe()
        unselected[:, 99:102] = False
        cases = [
            ('stripe', router(), LEFT, RIGHT),
            ('full', router(allowed=np.ones((200, 200), dtype=bool)), LEFT, RIGHT),
            ('unselected', router(allowed=unselected), LEFT, RIGHT),
            ('forbidden', router(polygons=[[[99., 0.], [101., 0.],
                                           [101., 199.], [99., 199.]]]), LEFT, RIGHT),
            ('same_face', router(), LEFT, point(-.005, .001, 0)),
            ('detour', dense_router(barrier=True), LEFT, RIGHT),
        ]
        for name, route, left, right in cases:
            with self.subTest(name=name):
                route._prepare_face_portals()
                eager, fast = EagerPortalSearch(route, left, right), _PortalSearch(route, left, right)
                expected, actual = eager.run(), fast.run()
                self.assert_same_path(actual, expected)
                # Search-order equivalence also keeps lazy candidate numbering.
                np.testing.assert_array_equal(np.asarray(fast.points), np.asarray(eager.points))
                self.assertEqual(fast.face_nodes, eager.face_nodes)
                self.assertLessEqual(fast.budget.counts['visibility checks'],
                                     eager.budget.counts['visibility checks'])

    def test_perspective_surface_repeats_exact_eager_result(self):
        vertices, faces = pair()
        vertices[:, 2] = [1., 1.01, 1.04, 1.03]
        weights = np.array([.3, .4, .3])
        endpoints = [dict(xyz_m=(weights @ vertices[face]).tolist(), triangle_index=index,
                          row_index=index) for index, face in enumerate(faces)]
        route = router(vertices, faces, allowed=np.ones((200, 200), dtype=bool))
        route._prepare_face_portals()
        expected = EagerPortalSearch(route, *endpoints).run()
        self.assertIsNotNone(expected)
        for _ in range(3):
            self.assert_same_path(_PortalSearch(route, *endpoints).run(), expected)

    def test_dense_graph_fits_same_budget_by_skipping_only_nonimproving_checks(self):
        route = dense_router()
        reference = EagerPortalSearch(route, LEFT, RIGHT)
        expected = reference.run()
        self.assertIsNotNone(expected)
        # Every cap remains the same for both searches, including candidates,
        # graph nodes and subdivisions. Only genuinely executed checks count.
        with mock.patch('femto.planning.surface_face_routing.POINT_BUDGET', 400):
            with self.assertRaisesRegex(ValueError, '400 visibility checks'):
                EagerPortalSearch(route, LEFT, RIGHT).run()
            fast = _PortalSearch(route, LEFT, RIGHT)
            self.assert_same_path(fast.run(), expected)
        self.assertGreater(reference.budget.counts['visibility checks'], 400)
        self.assertLess(fast.budget.counts['visibility checks'], 400)

    def test_nonimproving_pair_does_not_evaluate_or_charge_visibility(self):
        search = two_node_search()
        with mock.patch.object(search.router, '_state', side_effect=AssertionError('unneeded check')):
            self.assertEqual(list(search.neighbors(0, 10., {0: 10., 1: 0.})), [])
        self.assertEqual(search.budget.counts['visibility checks'], 0)
        self.assertEqual(search.visibility, {})

    def test_deferred_reverse_check_preserves_original_first_direction(self):
        for cls in (EagerPortalSearch, _PortalSearch):
            with self.subTest(search=cls.__name__):
                search = two_node_search(cls)
                # Deliberately asymmetric: rounded pixel walks and boundary
                # arithmetic must not be assumed bitwise direction-invariant.
                def directed_state(a, b):
                    return 0 if a[0] < b[0] else 3
                with mock.patch.object(search.router, '_state', side_effect=directed_state) as state:
                    list(search.neighbors(0, 10., {0: 10., 1: 0.}))
                    result = list(search.neighbors(1, 0., {0: 10., 1: 0.}))
                self.assertEqual([entry[0] for entry in result], [0])
                self.assertEqual(state.call_count, 1)
                np.testing.assert_array_equal(state.call_args.args[0], search.points[0])
                np.testing.assert_array_equal(state.call_args.args[1], search.points[1])

    def test_reopening_keeps_first_expansion_direction(self):
        search = two_node_search()
        with mock.patch.object(search.router, '_state', return_value=0) as state:
            # Simulate changed tentative labels / reopening directly. Neither
            # initial expansion needs the pair, but the later relaxation does.
            list(search.neighbors(0, 10., {0: 10., 1: 0.}))
            list(search.neighbors(1, 10., {0: 0., 1: 10.}))
            self.assertEqual(state.call_count, 0)
            result = list(search.neighbors(0, 0., {0: 0., 1: 10.}))
        self.assertEqual([entry[0] for entry in result], [1])
        self.assertEqual(state.call_count, 1)
        np.testing.assert_array_equal(state.call_args.args[0], search.points[0])
        np.testing.assert_array_equal(state.call_args.args[1], search.points[1])

    def test_required_visibility_still_obeys_the_hard_cap(self):
        search = two_node_search()
        # Patch after construction: this test targets visibility, not the
        # independently enforced two-endpoint graph-node reservation.
        with mock.patch('femto.planning.surface_face_routing.POINT_BUDGET', 0):
            with self.assertRaisesRegex(ValueError, '0 visibility checks'):
                list(search.neighbors(0, 0., {0: 0.}))

    def test_cancellation_is_checked_even_when_every_pair_would_be_pruned(self):
        search = two_node_search()
        search.budget.check = mock.Mock(side_effect=RuntimeError('cancel requested'))
        with mock.patch.object(search.router, '_state') as state:
            with self.assertRaisesRegex(RuntimeError, 'cancel requested'):
                list(search.neighbors(0, 10., {0: 10., 1: 0.}))
        state.assert_not_called()
        self.assertEqual(search.budget.counts['visibility checks'], 0)


if __name__ == '__main__':
    unittest.main()
