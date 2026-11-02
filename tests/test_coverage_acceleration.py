"""Independent rational construction checks for the equivalent fast path."""
from fractions import Fraction as Q
import math
import unittest

import numpy as np

from femto.geometry import exact_surface as exact
from femto.planning.coverage_domain import CoverageDomain


def fractional_clip(triangle, intrinsic, u, v):
    """Reference Sutherland-Hodgman construction in normalized Fractions."""
    fx, fy, cx, cy = map(Q, (float(intrinsic[0, 0]), float(intrinsic[1, 1]),
                            float(intrinsic[0, 2]), float(intrinsic[1, 2])))
    planes = ((fx, 0, cx-u+Q(1, 2)), (-fx, 0, u+Q(1, 2)-cx),
              (0, fy, cy-v+Q(1, 2)), (0, -fy, v+Q(1, 2)-cy))
    polygon = list(exact.IDENTITY)
    for plane in planes:
        coefficients = tuple(sum(x*y for x, y in zip(vertex, plane)) for vertex in triangle)
        output = []
        for left, right in zip(polygon, polygon[1:]+polygon[:1]):
            a = sum(x*y for x, y in zip(left, coefficients))
            b = sum(x*y for x, y in zip(right, coefficients))
            if a >= 0:output.append(left)
            if (a < 0) != (b < 0):
                t = a/(a-b)
                output.append(tuple(x+t*(y-x) for x, y in zip(left, right)))
        polygon = []
        for point in output:
            if not polygon or point != polygon[-1]:polygon.append(point)
        if len(polygon)>1 and polygon[0] == polygon[-1]:polygon.pop()
        if len(polygon)<3:return ()
    return tuple(polygon)


class CoverageAccelerationTests(unittest.TestCase):
    def test_integer_clipping_matches_independent_fractional_construction(self):
        rng = np.random.default_rng(902)
        intrinsic = np.array([[7.25, 0, 3.5], [0, 8.75, 3.25], [0, 0, 1]])
        for scale in (1e-30, 1e-6, 1., 1e20):
            for _ in range(75):
                xyz = rng.uniform(-1, 1, (3, 3));xyz[:, 2] = rng.uniform(.2, 2, 3)
                triangle = tuple(exact.rational(row) for row in xyz*scale)
                u, v = (int(value) for value in rng.integers(-3, 10, 2))
                self.assertEqual(exact.clip_pixel(triangle, intrinsic, u, v),
                                 fractional_clip(triangle, intrinsic, u, v))

    def test_half_pixel_boundaries_and_vertex_contacts_keep_original_order(self):
        intrinsic = np.eye(3)
        for rows in (((-.5, -.5, 1), (.5, -.5, 1), (.5, .5, 1)),
                     ((-.5, -.5, 1), (.5, .5, 1), (-.5, .5, 1)),
                     ((.5, .5, 1), (1.5, .5, 1), (.5, 1.5, 1)),
                     ((0, 0, 1), (0, 0, 1), (0, 1, 1)),
                     ((0, 0, 0), (1, 0, 0), (0, 1, 0))):
            triangle = tuple(exact.rational(row) for row in rows)
            for u, v in ((0, 0), (1, 0), (0, 1)):
                with self.subTest(rows=rows, pixel=(u, v)):
                    self.assertEqual(exact.clip_pixel(triangle, intrinsic, u, v),
                                     fractional_clip(triangle, intrinsic, u, v))

    def test_outward_bounds_are_tight_at_rounding_and_underflow(self):
        tiny = Q(1, 2**1100)
        values = (Q(1, 3), -Q(1, 3), Q(1), Q(0), tiny, -tiny,
                  Q(2**1020)+Q(1, 7), Q(1)+Q(1, 2**53))
        low, high = exact.bounds(values)
        for value, lower, upper in zip(values, low, high):
            self.assertLessEqual(Q(float(lower)), value)
            self.assertGreaterEqual(Q(float(upper)), value)
            if lower != upper:
                self.assertEqual(math.nextafter(float(lower), math.inf), float(upper))

    def test_point_cache_is_local_bounded_and_keeps_exact_geometry(self):
        vertices = np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1]], float)
        first = CoverageDomain.mesh(vertices, np.array([[0, 1, 2]]))
        bary = exact.subdivide(exact.IDENTITY)[3]
        triangle, bounds = first.cell(0, bary)
        expected = tuple(exact.xyz(point, first.triangle(0)) for point in bary)
        self.assertEqual(triangle, expected)
        for a, b in zip(bounds, exact.bounds(expected)):np.testing.assert_array_equal(a, b)
        first.cell(0, bary)
        self.assertGreater(first._point.cache_info().hits, 0)
        self.assertEqual(first._point.cache_info().maxsize, 8192)
        vertices[:, 2] = 2
        second = CoverageDomain.mesh(vertices, np.array([[0, 1, 2]]))
        self.assertNotEqual(second.cell(0, bary)[0], triangle)
        self.assertEqual(first.cell(0, bary)[0], triangle)
        bounds[0][:] = 0
        np.testing.assert_array_equal(first.cell(0, bary)[1][0], exact.bounds(expected)[0])


if __name__ == '__main__':unittest.main()
