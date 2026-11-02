"""Interval scratch buffers must not modify authoritative bounds or views."""
import unittest

import numpy as np

from femto.geometry import exact_surface as exact


class BallIntervalBufferTests(unittest.TestCase):
    def test_readonly_aliased_bounds_and_radii_stay_unchanged(self):
        storage = np.array([[-1., -1., -1.], [1., 1., 1.], [0., 0., 0.]])
        storage.setflags(write=False)
        radii = storage[1, :2]
        low = storage[:1, None, :]
        high = storage[1:2, None, :]
        centers = np.broadcast_to(storage[2], (2, 3))
        before = storage.tobytes()
        inside, outside = exact._ball_intervals(low, high, centers, centers, radii)
        np.testing.assert_array_equal(inside, [False, False])
        np.testing.assert_array_equal(outside, [False, False])
        self.assertEqual(storage.tobytes(), before)

    def test_overflow_and_nan_never_optimistically_confirm(self):
        for value in (1e200, np.inf, np.nan):
            with self.subTest(value=value):
                bounds = np.full((1, 3, 3), value)
                zeros = np.zeros((1, 3))
                original = np.geterr()
                with np.errstate(all='raise'):
                    inside, _ = exact._ball_intervals(bounds, bounds, zeros, zeros, np.array([1.]))
                    self.assertTrue(all(v == 'raise' for v in np.geterr().values()))
                self.assertEqual(np.geterr(), original)
                self.assertFalse(inside[0])

    def test_exact_boundary_fallback_and_zero_radius(self):
        triangle = tuple(exact.rational(p) for p in ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)))
        center = exact.rational((0., 0., 0.))
        inside, fallbacks = exact.contains_balls(triangle, (center,), (1.,))
        self.assertTrue(inside[0])
        self.assertEqual(fallbacks, 1)
        inside, _ = exact.contains_balls(triangle, (center,), (np.nextafter(1., 0.),))
        self.assertFalse(inside[0])
        collapsed = (center, center, center)
        inside, fallbacks = exact.contains_balls(collapsed, (center,), (0.,))
        self.assertTrue(inside[0])
        self.assertEqual(fallbacks, 1)


if __name__ == '__main__':
    unittest.main()
