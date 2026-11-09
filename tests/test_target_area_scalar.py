"""The scalar area path retains the original numerical and error behavior."""
import unittest
import warnings

import numpy as np

from femto.planning.target_surface import _triangle_area


def numpy_area(origin, first, second):
    """Reference arithmetic used before the scalar optimization."""
    a, b = first - origin, second - origin
    cross = np.array((a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]))
    return float(np.sqrt(cross.dot(cross)) / 2)


def outcome(function, triangle, mode):
    callbacks = []
    with warnings.catch_warnings(record=True) as seen, np.errstate(all=mode):
        warnings.simplefilter('always')
        previous = np.geterrcall()
        np.seterrcall(lambda message, flag: callbacks.append((message, flag)))
        try:
            try:
                result = ('value', float(function(*triangle)).hex())
            except Exception as exc:
                result = (type(exc).__name__, str(exc))
        finally:
            np.seterrcall(previous)
        return result, [str(warning.message) for warning in seen], callbacks


class TargetAreaScalarTests(unittest.TestCase):
    def test_error_modes_match_for_degenerate_large_tiny_and_nonfinite_inputs(self):
        rng = np.random.default_rng(624)
        cases = [np.zeros((3, 3)),
                 np.array([[0., 0., 0.], [1., 1e-200, 0.], [0., 1., 1e-200]]),
                 np.array([[0., 0., 0.], [1e200, 1e200, 0.], [0., 1e200, 1e200]]),
                 np.array([[0., 0., 0.], [np.inf, 0., 0.], [0., 1., 0.]]),
                 np.array([[np.nan, 0., 0.], [1., 0., 0.], [0., 1., 0.]])]
        cases.extend(rng.normal(size=(3, 3)) * scale
                     for scale in (1e-200, 1e-150, 1e-80, 1e-6, 1., 1e2, 1e70, 1e100)
                     for _ in range(20))
        for mode in ('ignore', 'warn', 'raise', 'call'):
            for index, triangle in enumerate(cases):
                with self.subTest(mode=mode, case=index):
                    self.assertEqual(outcome(numpy_area, triangle, mode),
                                     outcome(_triangle_area, triangle, mode))

    def test_normal_float64_areas_remain_bit_identical(self):
        rng = np.random.default_rng(20261103)
        for _ in range(10000):
            triangle = rng.normal(size=(3, 3)) * .01 + np.array([0., 0., .5])
            self.assertEqual(numpy_area(*triangle).hex(), _triangle_area(*triangle).hex())

    def test_large_final_area_does_not_hide_intermediate_underflow(self):
        triangle = np.array([[0., 0., 0.], [1., 1e-200, 0.], [0., 1., 1e-200]])
        with np.errstate(under='raise'):
            for function in (numpy_area, _triangle_area):
                with self.assertRaises(FloatingPointError):
                    function(*triangle)

    def test_other_dtypes_and_noncontiguous_views_keep_original_arithmetic(self):
        rng = np.random.default_rng(42)
        for dtype in (np.float32, '>f8', np.int64):
            triangle = (rng.normal(size=(3, 6)) * 10).astype(dtype)[:, ::2]
            for mode in ('ignore', 'warn', 'raise', 'call'):
                with self.subTest(dtype=dtype, mode=mode):
                    self.assertEqual(outcome(numpy_area, triangle, mode),
                                     outcome(_triangle_area, triangle, mode))


if __name__ == '__main__':
    unittest.main()
