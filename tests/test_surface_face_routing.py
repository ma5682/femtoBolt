"""Interior shared-edge routing is measured support, not fabricated geometry."""
from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from femto.planning.surface_face_routing import (
    FacePortalSurfaceLinks, _PortalSearch, _SearchBudget, _portal_midpoints,
)
from femto.planning.surface_link_optimization import OptimizedSurfaceLinks
from femto.planning.surface_trajectory import _point_on_triangle


K = np.array([[1000., 0., 100.], [0., 1000., 100.], [0., 0., 1.]])
SIZE = np.array([200, 200])
FRAME = {'e2': [0., 1., 0.]}


def pair():
    vertices = np.array([[-.01, -.01, 1.], [.01, -.01, 1.],
                         [-.01, .01, 1.], [.01, .01, 1.]])
    return vertices, np.array([[0, 1, 2], [1, 3, 2]])


def point(x, y, face, row=0):
    return dict(xyz_m=[x, y, 1.], triangle_index=face, row_index=row)


def stripe():
    allowed = np.zeros((200, 200), dtype=bool)
    allowed[98:103, 89:112] = True
    return allowed


def router(vertices=None, faces=None, allowed=None, cls=FacePortalSurfaceLinks, ids=None, polygons=None):
    if vertices is None:
        vertices, faces = pair()
    mesh = SimpleNamespace(vertices=vertices, faces=faces)
    region = dict(view_reference=dict(undistorted_intrinsic=K.tolist(), image_size=SIZE.tolist()))
    if polygons:
        region['exclusion_polygons'] = polygons
    return cls(mesh, region, FRAME, lambda: None, range(len(faces)) if ids is None else ids,
               allowed=stripe() if allowed is None else allowed)


LEFT = point(-.007, 0., 0)
RIGHT = point(.007, 0., 1, 1)


class SurfaceFaceRoutingTests(unittest.TestCase):
    def assert_supported(self, route, samples, left=LEFT, right=RIGHT):
        self.assertTrue(samples)
        previous = np.array(left['xyz_m'])
        for sample in samples:
            xyz = np.array(sample['xyz_m'])
            face = route.faces[sample['triangle_index']]
            self.assertTrue(_point_on_triangle(previous, route.vertices, face))
            self.assertTrue(_point_on_triangle(xyz, route.vertices, face))
            self.assertEqual(route._state(previous, xyz), 0)
            self.assertLessEqual(float(np.linalg.norm(xyz - previous)), .001 + 1e-12)
            self.assertEqual(sample['geometry_support'], 'measured_mesh')
            self.assertEqual(sample['role'], 'connector')
            self.assertEqual(sample['row_index'], right['row_index'])
            previous = xyz
        np.testing.assert_array_equal(previous, right['xyz_m'])

    def test_interior_portal_connects_when_every_triangle_vertex_is_unselected(self):
        old = router(cls=OptimizedSurfaceLinks)
        self.assertEqual(old.route(LEFT, RIGHT, .001, FRAME['e2'], adjacent_rows=False), (None, None))
        route = router()
        samples, method = route.route(LEFT, RIGHT, .001, FRAME['e2'], adjacent_rows=False)
        self.assertEqual(method, 'surface_face_portals')
        self.assert_supported(route, samples)
        self.assertEqual({sample['triangle_index'] for sample in samples}, {0, 1})

    def test_old_success_is_exactly_preserved_and_does_not_build_the_fallback(self):
        allowed = np.ones((200, 200), dtype=bool)
        old = router(cls=OptimizedSurfaceLinks, allowed=allowed)
        fresh = router(allowed=allowed)
        expected = old.route(LEFT, RIGHT, .001, FRAME['e2'], adjacent_rows=False)
        self.assertTrue(expected[0])
        with mock.patch.object(fresh, '_prepare_face_portals', side_effect=AssertionError('unneeded fallback')):
            self.assertEqual(fresh.route(LEFT, RIGHT, .001, FRAME['e2'], adjacent_rows=False), expected)
        # The same rule applies to successful boundary-band links.
        sentinel = ([{'kept': 'byte-identical'}], 'boundary_band')
        with mock.patch.object(OptimizedSurfaceLinks, 'route', return_value=sentinel):
            self.assertEqual(fresh.route(LEFT, RIGHT, .001, FRAME['e2'], adjacent_rows=True), sentinel)

    def test_true_unselected_barrier_cannot_be_crossed(self):
        allowed = stripe()
        allowed[:, 99:102] = False
        route = router(allowed=allowed)
        self.assertEqual(route.route(LEFT, RIGHT, .001, FRAME['e2'], adjacent_rows=False), (None, None))

    def test_painted_forbidden_barrier_cannot_be_crossed(self):
        route = router(polygons=[[[99., 0.], [101., 0.], [101., 199.], [99., 199.]]])
        self.assertEqual(route.route(LEFT, RIGHT, .001, FRAME['e2'], adjacent_rows=False), (None, None))

    def test_missing_surface_and_coincident_unwelded_edges_are_not_bridged(self):
        vertices, faces = pair()
        duplicated = np.vstack([vertices, vertices[faces[1]]])
        separate = np.array([faces[0], [4, 5, 6]])
        for shift in (0., .002):
            with self.subTest(shift=shift):
                changed = duplicated.copy()
                changed[4:, 0] += shift
                route = router(changed, separate)
                right = point(.007 + shift, 0., 1)
                self.assertEqual(route.route(LEFT, right, .001, FRAME['e2'], adjacent_rows=False), (None, None))

    def test_only_shared_vertex_is_not_a_face_portal(self):
        vertices = np.array([[0., 0., 1.], [-.02, -.02, 1.], [-.02, .02, 1.],
                             [.02, -.02, 1.], [.02, .02, 1.]])
        faces = np.array([[0, 1, 2], [0, 3, 4]])
        route = router(vertices, faces, allowed=np.ones((200, 200), dtype=bool))
        route._prepare_face_portals()
        # Inspect the fallback separately: preserving a pre-existing v4.1 success
        # is intentionally a different requirement from admitting a new portal.
        self.assertIsNone(_PortalSearch(route, LEFT, RIGHT).run())

    def test_nonmanifold_edge_is_rejected_even_if_third_face_is_outside_subset(self):
        vertices, faces = pair()
        vertices = np.vstack([vertices, [.02, .02, 1.02]])
        faces = np.vstack([faces, [1, 2, 4]])
        for ids in (range(3), [0, 1]):
            with self.subTest(ids=list(ids)):
                route = router(vertices, faces, ids=ids)
                self.assertEqual(route.route(LEFT, RIGHT, .001, FRAME['e2'], adjacent_rows=False), (None, None))

    def test_midpoints_use_perspective_correct_xyz_and_keep_pixel_subintervals(self):
        a, b = np.array([0., 0., 1.]), np.array([8., 0., 2.])
        points = list(_portal_midpoints(a, b, np.eye(3), np.array([10, 10]), _SearchBudget(lambda: None)))
        np.testing.assert_allclose([p[0] / p[2] for p in points], [.25, 1., 2., 3., 3.75])
        np.testing.assert_allclose(points[2], [8 / 3, 0., 4 / 3])
        self.assertFalse(np.allclose(points[2], (a + b) / 2))
        for p in points:
            self.assertAlmostEqual(p[0] / 8, p[2] - 1)

    def test_visible_edge_interior_is_not_discarded_for_outside_image_vertices(self):
        a, b = np.array([-2., 1., 1.]), np.array([12., 1., 1.])
        points = list(_portal_midpoints(a, b, np.eye(3), np.array([10, 10]), _SearchBudget(lambda: None)))
        visible = [p for p in points if 0 <= p[0] < 10]
        self.assertTrue(visible)
        self.assertTrue(any(0 < p[0] < .5 for p in visible))
        self.assertTrue(any(9.5 < p[0] < 10 for p in visible))

    def test_determinism_and_no_input_mutation(self):
        route = router()
        vertices, faces = route.vertices.copy(), route.faces.copy()
        first = route.route(LEFT, RIGHT, .001, FRAME['e2'], adjacent_rows=False)
        self.assertEqual(route.route(LEFT, RIGHT, .001, FRAME['e2'], adjacent_rows=False), first)
        np.testing.assert_array_equal(route.vertices, vertices)
        np.testing.assert_array_equal(route.faces, faces)
        self.assertEqual(LEFT, point(-.007, 0., 0))
        self.assertEqual(RIGHT, point(.007, 0., 1, 1))

    def test_cancellation_propagates_during_portal_search(self):
        route = router()
        def cancelled():
            raise RuntimeError('cancel requested')
        route.check = cancelled
        with mock.patch.object(OptimizedSurfaceLinks, 'route', return_value=(None, None)):
            with self.assertRaisesRegex(RuntimeError, 'cancel requested'):
                route.route(LEFT, RIGHT, .001, FRAME['e2'], adjacent_rows=False)

    def test_candidate_and_visibility_budgets_raise_instead_of_relaxing_constraints(self):
        route = router()
        with mock.patch('femto.planning.surface_face_routing.POINT_BUDGET', 3):
            with self.assertRaisesRegex(ValueError, 'Surface face routing exceeds'):
                route.route(LEFT, RIGHT, .001, FRAME['e2'], adjacent_rows=False)
        budget = _SearchBudget(lambda: None)
        with mock.patch('femto.planning.surface_face_routing.POINT_BUDGET', 2):
            budget.consume('visibility checks', 2)
            with self.assertRaisesRegex(ValueError, 'visibility checks'):
                budget.consume('visibility checks')

    def test_sample_budget_is_not_bypassed(self):
        route = router()
        points = np.array([LEFT['xyz_m'], [-.001, 0., 1.]])
        with mock.patch('femto.planning.surface_face_routing.POINT_BUDGET', 2):
            with self.assertRaisesRegex(ValueError, 'samples'):
                route._face_samples(points, [0], RIGHT, .001)


if __name__ == '__main__':
    unittest.main()
