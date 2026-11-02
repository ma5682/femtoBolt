"""Independent Fraction oracles for the reusable exact geometry fast paths."""
from fractions import Fraction as Q
import math
import unittest
from weakref import ref
from types import SimpleNamespace

import numpy as np

from femto.geometry import exact_surface as exact
from femto.planning.coverage_domain import CoverageDomain
from test_coverage_acceleration import fractional_clip


def orientation(a, b, c):
    return (b[1]-a[1])*(c[2]-a[2])-(b[2]-a[2])*(c[1]-a[1])


def edge_reference(face, ids, a, b):
    if a == b: return None
    for missing in range(3):
        if a[missing] == b[missing] == 0:
            vertices = sorted(ids[i] for i in range(3) if i != missing)
            axis = ids.index(vertices[1])
            return ('source_edge', *vertices), min(a[axis], b[axis]), max(a[axis], b[axis])
    x, y = a[1:]; u, v = b[1:]
    coefficients = (y-v, u-x, x*v-u*y)
    scale = coefficients[0] or coefficients[1]
    axis = 1 if x != u else 2
    return ('face_line', face, tuple(value/scale for value in coefficients)), min(a[axis], b[axis]), max(a[axis], b[axis])


class ExactRecomputationTests(unittest.TestCase):
    def test_construction_caches_release_domain_without_cyclic_collection(self):
        domain = CoverageDomain.mesh([[0,0,0],[1,0,0],[0,1,0]], [[0,1,2]])
        domain.cell(0, exact.IDENTITY)
        domain.integer_cell(0, ((1,0,0),(0,1,0),(0,0,1)))
        for name in ('_triangle', '_basis', '_point', '_integer_point'):
            self.assertGreater(getattr(domain, name).cache_info().currsize, 0)
        weak = ref(domain)
        del domain
        self.assertIsNone(weak())

    def test_reused_projection_matches_independent_clipping_at_all_pixels(self):
        rng = np.random.default_rng(31010)
        intrinsic = np.array([[7.1, 0, 3.2], [0, 8.3, -1.7], [0, 0, 1.]])
        for scale in (1e-60, 1., 1e60):
            for _ in range(9):
                xyz = rng.uniform(-1, 1, (3, 3)); xyz[:, 2] = rng.uniform(.1, 2, 3)
                triangle = tuple(exact.rational(row) for row in xyz*scale)
                projector = exact.PixelClipper(triangle, intrinsic)
                for u, v in ((0, 0), (1, 2), (-2, -3), (20, 20)):
                    self.assertEqual(projector.clip(u, v), fractional_clip(triangle, intrinsic, u, v))

    def test_integer_affine_coordinates_equal_fraction_oracle(self):
        rng = np.random.default_rng(71010)
        for exponent in (-1000, -60, 0, 500):
            triangle = tuple(tuple(Q(int(v), 13)*Q(2)**exponent for v in row)
                             for row in rng.integers(-100, 100, (3, 3)))
            basis = exact.AffineTriangle(triangle)
            for _ in range(40):
                u, v = (Q(int(value), 37) for value in rng.integers(-80, 80, 2))
                bary = (1-u-v, u, v)
                expected = tuple(sum(bary[j]*triangle[j][axis] for j in range(3)) for axis in range(3))
                self.assertEqual(basis.xyz(bary), expected)

    def test_orientation_and_point_membership_at_exact_boundaries(self):
        triangles = [exact.IDENTITY, *exact.subdivide(exact.IDENTITY)]
        tiny = Q(1, 2**2048)
        triangles += [((Q(1), Q(0), Q(0)), (1-tiny, tiny, Q(0)), (1-tiny, Q(0), tiny)),
                      (exact.IDENTITY[0], exact.IDENTITY[0], exact.IDENTITY[1])]
        points = [*exact.IDENTITY, (Q(1, 3),)*3, (1-tiny, tiny, Q(0)),
                  (1+tiny, -tiny, Q(0)), (Q(1, 7), Q(2, 7), Q(4, 7))]
        for triangle in triangles:
            for corners in (triangle, tuple(reversed(triangle))):
                area = orientation(*corners)
                self.assertEqual(exact.orientation_sign(*corners), (area > 0)-(area < 0))
                for point in points:
                    signs = [orientation(a, b, point) for a, b in zip(corners, corners[1:]+corners[:1])]
                    expected = area != 0 and (all(v >= 0 for v in signs) or all(v <= 0 for v in signs))
                    self.assertEqual(exact.contains_point(corners, point), expected)

    def test_edge_keys_keep_exact_old_normalization_and_positive_overlap(self):
        points = [*exact.IDENTITY, (Q(1, 7), Q(2, 7), Q(4, 7)),
                  (Q(1, 3), Q(1, 3), Q(1, 3)), (Q(0), Q(2, 3), Q(1, 3)),
                  (Q(1, 5), Q(1, 3), Q(7, 15))]
        for ids in ((9, 2, 7), (2, 7, 9)):
            for a in points:
                for b in points:
                    self.assertEqual(exact.edge_interval(5, ids, a, b), edge_reference(5, ids, a, b))

    def test_specialized_bounds_keep_tight_intervals_under_roundoff(self):
        for point in ((Q(1, 3), Q(-1, 3), Q(0)), (Q(1, 2**1100), -Q(1, 2**1100), Q(1)),
                      (Q(2**1020)+Q(1, 7), Q(1)+Q(1, 2**53), Q(-3, 7))):
            lo, hi = exact.point_bounds(point)
            for actual, expected in zip((lo, hi), exact.bounds(point)):
                np.testing.assert_array_equal(actual, expected)
            for q, a, b in zip(point, lo, hi):
                self.assertLessEqual(Q(float(a)), q); self.assertGreaterEqual(Q(float(b)), q)
                if a != b: self.assertEqual(math.nextafter(float(a), math.inf), float(b))

    def test_paired_ball_batches_match_independent_fraction_truth(self):
        triangles, centers, radii = [], [], []
        for radius in (0., 2.**-540, .125, math.nextafter(.125, 0), 2.**500):
            r = Q(radius)
            triangle = ((r, Q(0), Q(0)), (Q(0), r, Q(0)), (Q(0), Q(0), r))
            for center in ((Q(0),)*3, (r, r, r)):
                triangles.append(triangle); centers.append(center); radii.append(radius)
        intervals = [exact.bounds(triangle) for triangle in triangles]
        inside, fallbacks = exact.contains_ball_pairs(triangles, centers, radii,
            triangle_bounds=(np.asarray([p[0] for p in intervals]), np.asarray([p[1] for p in intervals])),
            center_bounds=exact.bounds(centers))
        expected = [all(sum((x-y)**2 for x,y in zip(p,c)) <= Q(r)**2 for p in tri)
                    for tri,c,r in zip(triangles,centers,radii)]
        np.testing.assert_array_equal(inside, expected)
        self.assertGreater(fallbacks, 0)

    def test_direct_dyadic_coordinates_equal_recursive_fraction_subdivision(self):
        triangle = ((Q(1, 7), Q(2, 7), Q(4, 7)), (Q(1, 5), Q(1, 3), Q(7, 15)), exact.IDENTITY[1])
        rng = np.random.default_rng(1010)
        for depth in range(9):
            for code in rng.integers(0, 4**depth, 12):
                expected = triangle
                for shift in range(depth-1, -1, -1):
                    a, b, c = expected
                    ab, bc, ca = (tuple((x+y)/2 for x,y in zip(p,q)) for p,q in ((a,b),(b,c),(c,a)))
                    expected = ((a,ab,ca),(ab,b,bc),(ca,bc,c),(ab,bc,ca))[(int(code) >> (2*shift)) & 3]
                self.assertEqual(exact.descend(triangle, depth, int(code)), expected)

    def test_integer_subedges_match_fraction_overlap_with_gaps_and_reversed_faces(self):
        cells = [(2, (9, 2, 7), exact.descend(exact.IDENTITY, depth, code))
                 for depth, code in ((0,0),(1,0),(1,1),(1,2),(1,3),(3,7),(3,8))]
        cells += [(3, (2, 9, 11), exact.IDENTITY), (4, (19, 12, 17), exact.IDENTITY)]
        tiny = Q(1, 2**1024)
        cells.append((2, (9,2,7), ((Q(0),Q(1),Q(0)), (Q(0),Q(1,2)+tiny,Q(1,2)-tiny), exact.IDENTITY[0])))
        for fa, ia, a in cells:
            aa = [edge_reference(fa, ia, x, y) for x,y in zip(a,a[1:]+a[:1])]
            ints_a = exact.integer_edge_intervals(fa, ia, a)
            for fb, ib, b in cells:
                bb = [edge_reference(fb, ib, x, y) for x,y in zip(b,b[1:]+b[:1])]
                expected = any(x and y and x[0] == y[0] and min(x[2],y[2]) > max(x[1],y[1]) for x in aa for y in bb)
                ints_b = exact.integer_edge_intervals(fb, ib, b)
                self.assertEqual(any(exact.integer_subedges_overlap(x,y) for x in ints_a for y in ints_b), expected)

    def test_reference_subedges_match_transformed_nonorthogonal_root(self):
        root = ((Q(1,7),Q(2,7),Q(4,7)), (Q(1,5),Q(1,3),Q(7,15)), exact.IDENTITY[1])
        descendants = [(0,0),(1,0),(1,1),(1,2),(1,3),(3,7),(3,8),(8,1487)]
        for ld, lc in descendants:
            a = exact.descend(root, ld, lc)
            aa = [edge_reference(2,(9,2,7),x,y) for x,y in zip(a,a[1:]+a[:1])]
            for rd, rc in descendants:
                b = exact.descend(root, rd, rc)
                bb = [edge_reference(2,(9,2,7),x,y) for x,y in zip(b,b[1:]+b[:1])]
                expected = any(x and y and x[0] == y[0] and min(x[2],y[2]) > max(x[1],y[1]) for x in aa for y in bb)
                self.assertEqual(exact.descendants_share_subedge(ld,lc,rd,rc), expected)

    def test_lazy_integer_geometry_and_bounds_equal_fraction_construction(self):
        triangle = tuple(exact.rational(row) for row in ((-.37,.21,.53),(.19,-.31,.67),(.41,.51,.79)))
        projector = exact.PixelClipper(triangle, np.array([[4.,0,1.],[0,4.,1.],[0,0,1.]]))
        polygon = projector.clip_integer(1,1)
        self.assertGreaterEqual(len(polygon), 3)
        root = polygon[:3]; basis = exact.AffineTriangle(triangle)
        for depth, code in ((0,0),(1,3),(4,27),(8,14287)):
            cell = exact.descend_integer(root, depth, code)
            bary = exact.descend(exact.rational_weights(root), depth, code)
            self.assertEqual(exact.rational_weights(cell), bary)
            rows = [exact.integer_xyz_bounds(basis, point) for point in cell]
            lazy = exact.IntegerXYZTriangle(tuple(row[0] for row in rows))
            expected = tuple(exact.xyz(point, triangle) for point in bary)
            self.assertEqual(tuple(lazy), expected)
            for value, correct in zip((np.asarray([r[1] for r in rows]), np.asarray([r[2] for r in rows])), exact.bounds(expected)):
                np.testing.assert_array_equal(value, correct)
            for point in (*bary, (Q(1,3),)*3):
                self.assertEqual(exact.integer_contains_point(cell, point), exact.contains_point(bary, point))

    def test_integer_pixel_fans_preserve_source_topology_or_stay_unconfirmed(self):
        vertices = np.array([[-2.,-2.,1.],[2.,-2.,1.],[0.,2.,1.]])
        revision = {'inputs':{'calibration':{'intrinsic':np.eye(3).tolist()}}}
        for children in ((0,1),(0,1,2)):
            ledger = SimpleNamespace(arrays=dict(mask=np.ones((1,1),'u1'), source_face=np.zeros(len(children),'i8'),
                pixel_index=np.zeros(len(children),'i8'), child_index=np.array(children,'i8')))
            domain = CoverageDomain(vertices, [[0,1,2]], ledger.arrays['source_face'], np.ones(len(children)),
                                    np.zeros(len(children),'u1'), {}, ledger=ledger, revision=revision)
            for index in range(len(children)):
                root = domain.root(index); integer = domain.integer_root(index)
                if root is None:self.assertIsNone(integer)
                else:
                    self.assertEqual(root, exact.rational_weights(integer))
                    cell, bounds = domain.integer_cell(0, integer)
                    expected, expected_bounds = domain.cell(0, root)
                    self.assertEqual(tuple(cell), expected)
                    for a,b in zip(bounds, expected_bounds):np.testing.assert_array_equal(a,b)

    def test_pixel_prisms_contain_independent_exact_clip_and_shrink_source_box(self):
        rng = np.random.default_rng(131010)
        pixels = np.array([[0,0],[1,1],[-1,0],[2,-1]], dtype='i8')
        checked = 0
        for scale in (1e-310, 1e-90, 1., 1e90, 1e300):
            for crossing in (False, True):
                vertices = rng.uniform(-2,2,(3,3))*scale
                vertices[:,2] = np.array([.5,1.,2.])*scale
                if crossing:vertices[0,2] *= -1
                intrinsic = np.array([[2.3,0,-.4],[0,1.7,.2],[0,0,1.]])
                for tighten in (False,True):
                    lo, hi = exact.pixel_prism_bounds(vertices, intrinsic, pixels,plane_tightening=tighten)
                    triangle = tuple(exact.rational(row) for row in vertices)
                    for index, (u,v) in enumerate(pixels):
                        polygon = fractional_clip(triangle, intrinsic, int(u), int(v))
                        for bary in polygon:
                            point = tuple(sum(bary[j]*triangle[j][axis] for j in range(3)) for axis in range(3))
                            for axis in range(3):
                                self.assertLessEqual(Q(float(lo[index,axis])), point[axis])
                                self.assertGreaterEqual(Q(float(hi[index,axis])), point[axis])
                            checked += 1
        self.assertGreater(checked, 20)
        triangle = np.array([[-2.,-2.,1.],[2.,-2.,1.],[0.,2.,1.]])
        lo, hi = exact.pixel_prism_bounds(triangle, np.eye(3), np.array([[0,0]]))
        self.assertGreater(lo[0,0], -1.)
        self.assertLess(hi[0,0], 1.)

    def test_pixel_prism_invalid_focal_length_keeps_whole_face_box(self):
        triangle = np.array([[-2.,-2.,1.],[2.,-2.,1.],[0.,2.,1.]])
        for focal in (0., -1., float('nan')):
            intrinsic = np.eye(3); intrinsic[1,1] = focal
            lo, hi = exact.pixel_prism_bounds(triangle, intrinsic, np.array([[0,0]]))
            np.testing.assert_array_equal(lo[0], triangle.min(axis=0))
            np.testing.assert_array_equal(hi[0], triangle.max(axis=0))

    def test_plane_prism_tightens_sloped_depth_and_keeps_ambiguous_plane_box(self):
        triangle = np.array([[-2.,-2.,.5],[2.,-2.,2.],[0.,2.,1.]])
        intrinsic = np.diag([20.,20.,1.]); pixels = np.array([[0,0],[1,1]])
        old_low, old_high = exact.pixel_prism_bounds(triangle,intrinsic,pixels)
        low, high = exact.pixel_prism_bounds(triangle,intrinsic,pixels,plane_tightening=True)
        self.assertTrue(np.all(high[:,2]-low[:,2] < (old_high[:,2]-old_low[:,2])*.1))
        triangles = np.asarray([triangle,triangle*2])
        batched = exact.pixel_prism_bounds(triangles,intrinsic,pixels,plane_tightening=True)
        for i in range(2):
            separate = exact.pixel_prism_bounds(triangles[i],intrinsic,pixels[i:i+1],plane_tightening=True)
            for values, expected in zip(batched,separate):np.testing.assert_array_equal(values[i],expected[0])
        # The ray cone crosses a zero plane denominator; this cannot tighten Z.
        vertical = np.array([[0.,-2.,1.],[0.,2.,1.],[0.,0.,2.]])
        old = exact.pixel_prism_bounds(vertical,np.eye(3),np.array([[0,0]]))
        new = exact.pixel_prism_bounds(vertical,np.eye(3),np.array([[0,0]]),plane_tightening=True)
        for a,b in zip(old,new):np.testing.assert_array_equal(a,b)

    def test_box_ball_filter_never_accepts_an_outside_exact_corner(self):
        rng = np.random.default_rng(181010)
        low = rng.uniform(-.2,.2,(64,3)); high = low+rng.uniform(0,.1,(64,3))
        centres = rng.uniform(-.2,.2,(64,3)); radii = rng.uniform(.05,.5,64)
        low = np.r_[low, [[.125,0,0], [0,0,0]]]
        high = np.r_[high, [[.125,0,0], [.01,.01,.01]]]
        centres = np.r_[centres, [[0,0,0], [0,0,0]]]; radii = np.r_[radii, [.125,.1]]
        actual = exact.boxes_in_balls(low, high, radii, center_bounds=(centres,centres))
        self.assertTrue(actual[-1]); self.assertTrue(actual.any())
        for i in np.flatnonzero(actual):
            for bits in range(8):
                point = [Q(float((high if bits & (1<<axis) else low)[i,axis])) for axis in range(3)]
                self.assertLessEqual(sum((p-Q(float(c)))**2 for p,c in zip(point,centres[i])), Q(float(radii[i]))**2)

    def test_multiple_source_face_prisms_match_separate_construction(self):
        triangle = np.array([[-2.,-2.,1.],[2.,-2.,1.],[0.,2.,1.]])
        triangles = np.asarray([triangle, triangle*2, triangle*[1,1,-1]])
        pixels = np.asarray([[0,0],[1,1],[0,0]])
        low, high = exact.pixel_prism_bounds(triangles, np.eye(3), pixels)
        for i in range(3):
            a,b = exact.pixel_prism_bounds(triangles[i],np.eye(3),pixels[i:i+1])
            np.testing.assert_array_equal(low[i],a[0]); np.testing.assert_array_equal(high[i],b[0])


if __name__ == '__main__': unittest.main()
