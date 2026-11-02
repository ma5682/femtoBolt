"""Differential checks against the persisted v1 grouping and clipping rules."""
import unittest
from unittest.mock import patch
import numpy as np
from femto.geometry.pixel_clip import EPS, _rational_clip, clip_triangle, pixel_planes
from femto.planning import target_surface as target

def legacy_clip(triangle: np.ndarray, k: np.ndarray, u: int, v: int) -> tuple[np.ndarray, bool]:
    """Barycentric polygon on the source face; half-open ownership is by pixel_owner.

    Closed polygon boundaries are stored to keep mesh edges representable. Their
    measure is zero; queries on a shared pixel edge belong to the higher pixel.
    """
    planes = np.array([[k[0, 0], 0, k[0, 2]-(u-.5)],
                       [-k[0, 0], 0, u+.5-k[0, 2]],
                       [0, k[1, 1], k[1, 2]-(v-.5)],
                       [0, -k[1, 1], v+.5-k[1, 2]]], float)
    points = np.eye(3)
    for plane in planes:
        distances = triangle @ plane
        errors = EPS * np.sum(np.abs(triangle * plane), axis=1)
        signs = points @ distances
        bounds = points @ errors + EPS * (np.abs(points) @ np.abs(distances))
        if np.any(np.abs(signs) <= bounds):
            return _rational_clip(triangle, planes), True
        out = []
        for index, a in enumerate(points):
            b = points[(index+1) % len(points)]
            da, db = signs[index], signs[(index+1) % len(points)]
            if da > 0:
                out.append(a)
            if (da < 0) != (db < 0):
                out.append(a + (da/(da-db))*(b-a))
        if len(out) < 3:
            return np.empty((0, 3), float), False
        points = np.asarray(out)
    return points, False

class TargetGroupingTests(unittest.TestCase):
    def test_stable_groups_keep_candidate_order_and_duplicates(self):
        rng = np.random.default_rng(1510)
        for count in (1, 7, 10000):
            pixels = rng.integers(0, 91, count, dtype=np.int64)
            faces = rng.integers(0, 100, count, dtype=np.int64)
            order = np.argsort(pixels, kind='stable')
            sorted_pixels = pixels[order]
            groups, starts = np.unique(sorted_pixels, return_index=True)
            expected = (groups, faces[order], starts, np.r_[starts[1:], count])
            actual = target._candidate_groups([pixels], [faces])
            for a,b in zip(actual,expected): np.testing.assert_array_equal(a,b)

    def test_clipping_matches_legacy_bits_and_exact_fallback(self):
        rng = np.random.default_rng(1510)
        k = np.array([[100.,0.,50.],[0.,100.,50.],[0.,0.,1.]])
        triangles = [rng.uniform([-.1,-.1,.4],[.1,.1,.6],(3,3)) for _ in range(1200)]
        triangles += [np.array([[x,-.0025,.5],[x,.0025,.5],[x+.005,0.,.5]])
                      for x in (-.0025, np.nextafter(-.0025,0.), np.nextafter(-.0025,-1.))]
        fallbacks = 0
        for triangle in triangles:
            expected, exact = legacy_clip(triangle,k,50,50)
            actual, actual_exact = clip_triangle(triangle,k,50,50)
            np.testing.assert_array_equal(actual,expected)
            self.assertEqual(actual_exact,exact)
            fallbacks += exact
        self.assertGreater(fallbacks,0)

    def test_projection_is_reused_across_different_face_pairs(self):
        a = np.array([[0.,0.,1.],[1.,0.,1.],[0.,1.,1.]])
        triangles = np.array([a,a+[3.,0.,0.],a+[6.,0.,0.]])
        with patch.object(target,'project',wraps=target.project) as project:
            separated = target._source_separation(triangles,np.eye(3))
            self.assertTrue(separated(0,1))
            self.assertTrue(separated(0,2))
            self.assertTrue(separated(1,2))
            self.assertEqual(project.call_count,3)
        # Each construction owns fresh projections and geometry.
        self.assertFalse(target._source_separation(np.array([a,a]),np.eye(3))(0,1))

if __name__ == '__main__': unittest.main()
