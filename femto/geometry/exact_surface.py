"""Exact constructions and conservative predicates on a binary-input mesh.

Fractions describe a nominal numeric model, never physical measurement accuracy.
The float interval filter only accepts proved containment; ambiguous comparisons
fall back to exact arithmetic. Topology uses source IDs and positive subedges.
"""
from __future__ import annotations

from fractions import Fraction as Q
from functools import lru_cache
import math
from typing import Iterable

import numpy as np

Bary = tuple[Q, Q, Q]
Triangle = tuple[Bary, Bary, Bary]
IDENTITY: Triangle = tuple(tuple(Q(int(i == j)) for j in range(3)) for i in range(3))


def rational(value: Iterable) -> Bary:
    return tuple(x if isinstance(x, Q) else Q(float(x)) for x in value)


def xyz(bary: Bary, triangle: Triangle) -> Bary:
    return tuple(sum(bary[j]*triangle[j][axis] for j in range(3)) for axis in range(3))


def dot(a: Bary, b: Bary) -> Q:
    return sum(x*y for x, y in zip(a, b))


def subtract(a: Bary, b: Bary) -> Bary:
    return tuple(x-y for x, y in zip(a, b))


def _homogeneous_weights(values: tuple[int, int, int]) -> tuple[int, int, int]:
    """Normalize a barycentric numerator triple with positive implicit sum."""
    x, y, z = values
    divisor = math.gcd(x, y, z)
    if x+y+z < 0:divisor = -divisor
    return x//divisor, y//divisor, z//divisor


def _clip_coefficients(planes):
    points = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    for coefficients in planes:
        c0, c1, c2 = coefficients
        distances = [a[0]*c0+a[1]*c1+a[2]*c2 for a in points]
        output = []
        for i, a in enumerate(points):
            following = i+1 if i+1 < len(points) else 0
            b, da, db = points[following], distances[i], distances[following]
            if da >= 0 and (not output or a != output[-1]):output.append(a)
            if (da < 0) != (db < 0):
                point = _homogeneous_weights((da*b[0]-db*a[0], da*b[1]-db*a[1], da*b[2]-db*a[2]))
                if not output or point != output[-1]:output.append(point)
        points = output
        if len(points) > 1 and points[0] == points[-1]:points.pop()
        if len(points) < 3:return ()
    return tuple(points)


def rational_weights(points):
    return tuple((Q(x, x+y+z), Q(y, x+y+z), Q(z, x+y+z)) for x,y,z in points)


class PixelClipper:
    """Immutable exact projection coefficients for one face and calibration.

    Pixel indices only multiply integer Z coefficients. This factors out the
    repeated Fraction dot products without changing clipping or boundary order.
    It stores geometry, never a coverage or admission decision.
    """
    __slots__ = ('horizontal', 'vertical', 'depth')

    def __init__(self, triangle: Triangle, intrinsic: np.ndarray):
        k = intrinsic
        fx, fy, cx, cy = (Q(float(x)) for x in (k[0, 0], k[1, 1], k[0, 2], k[1, 2]))
        values = (tuple(fx*x+(cx+Q(1, 2))*z for x, y, z in triangle),
                  tuple(fy*y+(cy+Q(1, 2))*z for x, y, z in triangle),
                  tuple(vertex[2] for vertex in triangle))
        scale = math.lcm(*(value.denominator for row in values for value in row))
        rows = tuple(tuple(value.numerator*(scale//value.denominator) for value in row) for row in values)
        divisor = math.gcd(*(value for row in rows for value in row)) or 1
        self.horizontal, self.vertical, self.depth = (tuple(value//divisor for value in row) for row in rows)

    def clip(self, u: int, v: int) -> tuple[Bary, ...]:
        return rational_weights(self.clip_integer(u, v))

    def clip_integer(self, u: int, v: int):
        x, y, z = self.horizontal, self.vertical, self.depth
        left = x[0]-u*z[0], x[1]-u*z[1], x[2]-u*z[2]
        top = y[0]-v*z[0], y[1]-v*z[1], y[2]-v*z[2]
        return _clip_coefficients((left, (z[0]-left[0], z[1]-left[1], z[2]-left[2]),
                                   top, (z[0]-top[0], z[1]-top[1], z[2]-top[2])))


def clip_pixel(triangle: Triangle, intrinsic: np.ndarray, u: int, v: int) -> tuple[Bary, ...]:
    """The same exact half-pixel construction, with a reusable face projector."""
    return PixelClipper(triangle, intrinsic).clip(u, v)


def pixel_prism_bounds(triangle: np.ndarray, intrinsic: np.ndarray, pixels: np.ndarray, *, plane_tightening=False):
    """Outward AABBs from pixel half-spaces and the source face's Z range.

    Used only for exclusion. Every clipped fan and its descendants are inside
    this box; no float-clipped corner or previous classification is trusted.
    Unbounded/ambiguous arithmetic falls back to the whole source-face box.
    """
    source_low, source_high = triangle.min(axis=-2), triangle.max(axis=-2)
    low = np.broadcast_to(source_low, (len(pixels), 3)).copy()
    high = np.broadcast_to(source_high, (len(pixels), 3)).copy()
    original_low, original_high = low.copy(), high.copy()
    rays_low = np.ones((len(pixels),3)); rays_high = rays_low.copy()
    safe_rays = np.ones(len(pixels), bool)
    # The two horizontal clipping inequalities imply Z >= 0.
    zlo, zhi = np.maximum(0., low[:,2]), high[:,2]
    for axis in (0, 1):
        focal, centre = intrinsic[axis, axis], intrinsic[axis, 2]
        if not np.isfinite([focal, centre]).all() or focal <= 0:return original_low, original_high
        with np.errstate(over='ignore', under='ignore', invalid='ignore', divide='ignore'):
            coordinate = pixels[:,axis].astype(float)
            ray_low = np.nextafter(np.nextafter(np.nextafter(coordinate-.5, -np.inf)-centre, -np.inf)/focal, -np.inf)
            ray_high = np.nextafter(np.nextafter(np.nextafter(coordinate+.5, np.inf)-centre, np.inf)/focal, np.inf)
            products = np.asarray([ray_low*zlo, ray_low*zhi, ray_high*zlo, ray_high*zhi])
            lower, upper = np.nextafter(products.min(axis=0), -np.inf), np.nextafter(products.max(axis=0), np.inf)
        valid = np.isfinite(lower) & np.isfinite(upper) & (np.abs(coordinate) <= 2.**52-1)
        rays_low[:,axis], rays_high[:,axis] = ray_low, ray_high
        safe_rays &= valid
        low[valid,axis] = np.maximum(low[valid,axis], lower[valid])
        high[valid,axis] = np.minimum(high[valid,axis], upper[valid])
    low[:,2] = zlo
    empty = np.any(low > high, axis=1)
    low[empty], high[empty] = original_low[empty], original_high[empty]
    if plane_tightening:
        low, high = _plane_prism_bounds(triangle, rays_low, rays_high, safe_rays, low, high)
    return low, high


def _interval_product(al, ah, bl, bh):
    products = np.asarray((al*bl, al*bh, ah*bl, ah*bh))
    return np.nextafter(products.min(axis=0), -np.inf), np.nextafter(products.max(axis=0), np.inf)


def _interval_sum3(low, high):
    return (np.nextafter(np.nextafter(low[:,0]+low[:,1],-np.inf)+low[:,2],-np.inf),
            np.nextafter(np.nextafter(high[:,0]+high[:,1],np.inf)+high[:,2],np.inf))


def _plane_prism_bounds(triangle, rays_low, rays_high, safe_rays, low, high):
    """Intersect ray intervals with the current source plane, outwards only.

    n dot (z*ray) = n dot a bounds depth without constructing clipped points.
    An interval crossing a zero denominator, overflow, or an empty intersection
    keeps the previous conservative box. Correlated terms are deliberately
    treated independently so this filter can never manufacture precision.
    """
    t = np.broadcast_to(triangle,(len(low),3,3))
    with np.errstate(over='ignore',under='ignore',invalid='ignore',divide='ignore'):
        v, w = t[:,1]-t[:,0], t[:,2]-t[:,0]
        vl, vh = np.nextafter(v,-np.inf), np.nextafter(v,np.inf)
        wl, wh = np.nextafter(w,-np.inf), np.nextafter(w,np.inf)
        left, right = [1,2,0], [2,0,1]
        al, ah = _interval_product(vl[:,left],vh[:,left],wl[:,right],wh[:,right])
        bl, bh = _interval_product(vl[:,right],vh[:,right],wl[:,left],wh[:,left])
        nl, nh = np.nextafter(al-bh,-np.inf), np.nextafter(ah-bl,np.inf)
        dl, dh = _interval_sum3(*_interval_product(nl,nh,t[:,0],t[:,0]))
        denom_low, denom_high = _interval_sum3(*_interval_product(nl,nh,rays_low,rays_high))
        ratios = np.asarray((dl/denom_low,dl/denom_high,dh/denom_low,dh/denom_high))
        zl, zh = np.nextafter(ratios.min(axis=0),-np.inf), np.nextafter(ratios.max(axis=0),np.inf)
        zl, zh = np.maximum(np.maximum(low[:,2],0.),zl), np.minimum(high[:,2],zh)
        candidate_low, candidate_high = _interval_product(rays_low,rays_high,zl[:,None],zh[:,None])
        candidate_low = np.maximum(low,candidate_low); candidate_high = np.minimum(high,candidate_high)
        # Z is an identity coordinate, so keep its already rounded interval.
        candidate_low[:,2], candidate_high[:,2] = zl, zh
    valid = (safe_rays & ((denom_low > 0) | (denom_high < 0))
             & np.isfinite(candidate_low).all(axis=1) & np.isfinite(candidate_high).all(axis=1)
             & np.all(candidate_low <= candidate_high,axis=1))
    low[valid], high[valid] = candidate_low[valid], candidate_high[valid]
    return low, high


class AffineTriangle:
    """Integer basis for exactly evaluating barycentrics on a fixed face."""
    __slots__ = ('rows', 'denominator')

    def __init__(self, triangle: Triangle):
        self.denominator = math.lcm(*(value.denominator for row in triangle for value in row))
        self.rows = tuple(tuple(value.numerator*(self.denominator//value.denominator) for value in row)
                          for row in triangle)

    def xyz(self, bary: Bary) -> Bary:
        scale = math.lcm(*(value.denominator for value in bary))
        weights = tuple(value.numerator*(scale//value.denominator) for value in bary)
        return tuple(Q(sum(weights[j]*self.rows[j][axis] for j in range(3)), scale*self.denominator)
                     for axis in range(3))


def _planar_integer(point):
    x, y = point[1:]
    scale = math.lcm(x.denominator, y.denominator)
    return x.numerator*(scale//x.denominator), y.numerator*(scale//y.denominator), scale


def _orientation_numerator(a, b, c):
    ax, ay, ad = a; bx, by, bd = b; cx, cy, cd = c
    return (bx*ad-ax*bd)*(cy*ad-ay*cd)-(by*ad-ay*bd)*(cx*ad-ax*cd)


def orientation_sign(a: Bary, b: Bary, c: Bary) -> int:
    value = _orientation_numerator(*map(_planar_integer, (a, b, c)))
    return (value > 0)-(value < 0)


def integer_orientation(a, b, c):
    return _orientation_numerator((a[1], a[2], sum(a)), (b[1], b[2], sum(b)), (c[1], c[2], sum(c)))


def integer_contains_point(triangle, point: Bary):
    corners = tuple((v[1], v[2], sum(v)) for v in triangle)
    p = _planar_integer(point)
    signs = [_orientation_numerator(a, b, p) for a, b in zip(corners, corners[1:]+corners[:1])]
    return _orientation_numerator(*corners) != 0 and (all(v >= 0 for v in signs) or all(v <= 0 for v in signs))


def orient(a: Bary, b: Bary, c: Bary) -> Q:
    return (b[1]-a[1])*(c[2]-a[2])-(b[2]-a[2])*(c[1]-a[1])


def contains_point(triangle: Triangle, point: Bary) -> bool:
    corners = tuple(map(_planar_integer, triangle)); p = _planar_integer(point)
    signs = [_orientation_numerator(a, b, p) for a, b in zip(corners, corners[1:]+corners[:1])]
    return _orientation_numerator(*corners) != 0 and (all(s >= 0 for s in signs) or all(s <= 0 for s in signs))


def subdivide(triangle: Triangle) -> tuple[Triangle, ...]:
    a, b, c = triangle
    ab, bc, ca = (tuple((x+y)/2 for x, y in zip(p, q)) for p, q in ((a, b), (b, c), (c, a)))
    return ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))


def descend(triangle: Triangle, depth: int, code: int) -> Triangle:
    if not depth:return triangle
    rows = _dyadic_weights(depth, code)
    basis = AffineTriangle(triangle)
    denominator = (2**depth)*basis.denominator
    return tuple(tuple(Q(sum(weights[j]*basis.rows[j][axis] for j in range(3)), denominator)
                       for axis in range(3)) for weights in rows)


@lru_cache(maxsize=4096)
def _dyadic_weights(depth: int, code: int):
    """Geometry-independent integer coefficients; no source/model cache."""
    a, b, c = (1, 0, 0), (0, 1, 0), (0, 0, 1)
    for shift in range(depth-1, -1, -1):
        ab, bc, ca = (tuple(x+y for x,y in zip(p,q)) for p,q in ((a,b),(b,c),(c,a)))
        child = (code >> (2*shift)) & 3
        if child == 0:a, b, c = tuple(2*x for x in a), ab, ca
        elif child == 1:a, b, c = ab, tuple(2*x for x in b), bc
        elif child == 2:a, b, c = ca, bc, tuple(2*x for x in c)
        else:a, b, c = ab, bc, ca
    return a, b, c


@lru_cache(maxsize=4096)
def _dyadic_coefficients(depth: int, code: int):
    """Read-only coefficients depend only on the subdivision address."""
    values = np.asarray(_dyadic_weights(depth, code), float) / (2**depth)
    values.setflags(write=False)
    return values


def descendant_vertex_bounds(root_low, root_high, depths, codes):
    """Enclose dyadic descendant vertices of current root vertex intervals.

    The nonnegative dyadic coefficients are exact float64 values (depth <= 8).
    Each product and sum is rounded outwards. The bounds do not establish root
    validity or topology. Ambiguity encloses each vertex in the whole root box.
    """
    low, high = np.asarray(root_low, float), np.asarray(root_high, float)
    depths, codes = np.asarray(depths), np.asarray(codes)
    if (low.shape != (len(depths), 3, 3) or high.shape != low.shape
            or depths.ndim != 1 or codes.shape != depths.shape
            or depths.dtype.kind not in 'iu' or codes.dtype.kind not in 'iu'
            or np.any(depths > 8) or np.any(depths < 0)
            or np.any(codes < 0) or np.any(codes >= np.left_shift(1, 2*depths.astype('i8')))
            or np.any(np.isnan(low)) or np.any(np.isnan(high)) or np.any(low > high)):
        raise ValueError('Dyadic root bounds or addresses are invalid')
    keys = np.left_shift(depths.astype('u8'), 32) | codes.astype('u8')
    unique, inverse = np.unique(keys, return_inverse=True)
    coefficients = np.asarray([_dyadic_coefficients(int(k)>>32, int(k)&0xffffffff)
                               for k in unique]).reshape(-1, 3, 3)[inverse]
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        products_low = np.nextafter(low[:,None,:,:]*coefficients[:,:,:,None], -np.inf)
        products_high = np.nextafter(high[:,None,:,:]*coefficients[:,:,:,None], np.inf)
        vertices_low = np.nextafter(np.nextafter(products_low[:,:,0]+products_low[:,:,1], -np.inf)
                                    + products_low[:,:,2], -np.inf)
        vertices_high = np.nextafter(np.nextafter(products_high[:,:,0]+products_high[:,:,1], np.inf)
                                     + products_high[:,:,2], np.inf)
    ambiguous = (~np.isfinite(vertices_low).all(axis=(1,2)) | ~np.isfinite(vertices_high).all(axis=(1,2))
                 | np.any(vertices_low > vertices_high, axis=(1,2)))
    vertices_low[ambiguous] = low[ambiguous].min(axis=1)[:,None,:]
    vertices_high[ambiguous] = high[ambiguous].max(axis=1)[:,None,:]
    return vertices_low, vertices_high


def descendant_boxes(root_low, root_high, depths, codes):
    """Sufficient exclusion boxes of the directed descendant vertex bounds."""
    low, high = descendant_vertex_bounds(root_low, root_high, depths, codes)
    return low.min(axis=1), high.max(axis=1)


class IntegerTriangle:
    """Reusable common-denominator basis for descendants of one exact root."""
    __slots__ = ('triangle', 'rows')

    def __init__(self, triangle):
        self.triangle = triangle
        self.rows = None

    def descend(self, depth: int, code: int):
        if not depth:return self.triangle
        if self.rows is None:
            denominators = tuple(map(sum, self.triangle))
            scale = math.lcm(*denominators)
            self.rows = tuple(tuple(v*(scale//d) for v in point) for point, d in zip(self.triangle, denominators))
        a, b, c = self.rows
        result = []
        for x, y, z in _dyadic_weights(depth, code):
            if not y and not z:point = self.triangle[0]
            elif not x and not z:point = self.triangle[1]
            elif not x and not y:point = self.triangle[2]
            else:point = _homogeneous_weights((x*a[0]+y*b[0]+z*c[0],
                x*a[1]+y*b[1]+z*c[1], x*a[2]+y*b[2]+z*c[2]))
            result.append(point)
        return tuple(result)


def descend_integer(triangle, depth: int, code: int):
    return IntegerTriangle(triangle).descend(depth, code)


@lru_cache(maxsize=4096)
def _unit_edges(depth: int, code: int):
    return homogeneous_edge_intervals(0, (0, 1, 2), _dyadic_weights(depth, code))


def descendants_share_subedge(left_depth: int, left_code: int, right_depth: int, right_code: int) -> bool:
    """Positive overlap in an invertible root's affine reference triangle."""
    return any(integer_subedges_overlap(a, b) for a in _unit_edges(left_depth, left_code)
               for b in _unit_edges(right_depth, right_code))


class IntegerXYZTriangle:
    """Lazy Fractions for the rare ambiguous containment fallback only."""
    __slots__ = ('points',)

    def __init__(self, points):self.points = points

    def __iter__(self):
        return (tuple(Q(v, denominator) for v in xyz) for xyz, denominator in self.points)


def integer_xyz_bounds(basis: AffineTriangle, weights):
    denominator = sum(weights)*basis.denominator
    a, b, c = basis.rows; x, y, z = weights
    numerators = (x*a[0]+y*b[0]+z*c[0], x*a[1]+y*b[1]+z*c[1], x*a[2]+y*b[2]+z*c[2])
    low, high = [], []
    for numerator in numerators:
        near = numerator/denominator
        if not math.isfinite(near):
            low.append(-math.inf); high.append(math.inf); continue
        n, d = near.as_integer_ratio()
        difference = n*denominator-numerator*d
        low.append(math.nextafter(near, -math.inf) if difference > 0 else near)
        high.append(math.nextafter(near, math.inf) if difference < 0 else near)
    return (numerators, denominator), np.asarray(low), np.asarray(high)


def anchor_ball(center: Iterable, triangle: Triangle, radius: float) -> tuple[Bary, Bary, float] | None:
    """A surface-anchored inner ball contained in the unchanged path-point ball.

    Orthogonal projection is only an anchor construction, not a moved route.
    The exact displacement bound is deducted from the inner radius.
    """
    p = rational(center)
    a, b, c = triangle
    ab, ac, ap = subtract(b, a), subtract(c, a), subtract(p, a)
    aa, bb, cc = dot(ab, ab), dot(ab, ac), dot(ac, ac)
    determinant = aa*cc-bb*bb
    if determinant <= 0:return None
    d, e = dot(ap, ab), dot(ap, ac)
    u, v = (d*cc-e*bb)/determinant, (e*aa-d*bb)/determinant
    bary = (1-u-v, u, v)
    if min(bary) < 0:return None
    anchor = xyz(bary, triangle)
    distance2 = dot(subtract(anchor, p), subtract(anchor, p))
    outer = Q(radius)
    if distance2 >= outer*outer:return None
    if distance2 == 0:return bary, anchor, radius
    # Scale before sqrt: converting a tiny squared displacement directly to
    # float can underflow to zero and make an ulp-by-ulp correction unbounded.
    exponent = (distance2.numerator.bit_length()-distance2.denominator.bit_length())//2
    factor = Q(2)**exponent
    upper = math.nextafter(math.ldexp(math.sqrt(float(distance2/factor**2)), exponent), math.inf)
    while Q(upper)**2 < distance2:upper = math.nextafter(upper, math.inf)
    inner = math.nextafter(radius-upper, -math.inf)
    while inner > 0 and (outer-Q(inner))**2 < distance2:
        inner = math.nextafter(inner, -math.inf)
    return (bary, anchor, inner) if inner > 0 else None


def bounds(points: Iterable) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(points, dtype=object)
    near = values.astype(float)
    low, high = near.copy(), near.copy()
    for index in np.ndindex(values.shape):
        if not math.isfinite(near[index]):
            low[index], high[index] = -math.inf, math.inf
            continue
        value = values[index]
        if isinstance(value, Q):
            numerator, denominator = float(near[index]).as_integer_ratio()
            difference = numerator*value.denominator-value.numerator*denominator
        else:
            represented = Q(float(near[index]))
            difference = represented-value
        if difference > 0:low[index] = math.nextafter(near[index], -math.inf)
        if difference < 0:high[index] = math.nextafter(near[index], math.inf)
    return low, high


def point_bounds(point: Bary) -> tuple[np.ndarray, np.ndarray]:
    """The same tight directed bounds, specialized for one exact XYZ point."""
    low, high = [], []
    for value in point:
        near = float(value)
        if not math.isfinite(near):
            low.append(-math.inf); high.append(math.inf); continue
        numerator, denominator = near.as_integer_ratio()
        difference = numerator*value.denominator-value.numerator*denominator
        low.append(math.nextafter(near, -math.inf) if difference > 0 else near)
        high.append(math.nextafter(near, math.inf) if difference < 0 else near)
    return np.asarray(low), np.asarray(high)


def contains_ball(triangle: Triangle, center: Bary, radius: float, *,
                  triangle_bounds=None, center_bounds=None) -> tuple[bool, bool]:
    """Return (all three vertices in one ball, used_exact_fallback)."""
    cb = None if center_bounds is None else tuple(v[None, :] for v in center_bounds)
    inside, fallbacks = contains_balls(triangle, (center,), (radius,), triangle_bounds=triangle_bounds, center_bounds=cb)
    return bool(inside[0]), bool(fallbacks)


def _single_ball_bounds(low, high, clow, chigh, radii):
    """Same directed float64 operations for one finite triangle/ball pair.

    Plain arrays only: subclasses and other dtypes retain their NumPy
    semantics. Nonfinite inputs/intermediates also use the vector path.
    Python float arithmetic avoids tiny temporary arrays and does not alter
    the caller's NumPy error state. No geometric decision is cached.
    """
    arrays=(low,high,clow,chigh,radii)
    if any(type(a) is not np.ndarray or a.dtype!=np.dtype('float64') for a in arrays):return None
    if (low.shape!=(1,3,3) or high.shape!=(1,3,3) or clow.shape!=(1,3)
            or chigh.shape!=(1,3) or radii.shape!=(1,)):return None
    radius=float(radii[0]);lows=[];highs=[]
    if not math.isfinite(radius):return None
    radius_squared=radius*radius
    radius_low=max(0.,math.nextafter(radius_squared,-math.inf))
    radius_high=math.nextafter(radius_squared,math.inf)
    if not math.isfinite(radius_high):return None
    for vertex in range(3):
        total_low=total_high=0.
        for axis in range(3):
            values=(float(low[0,vertex,axis]),float(high[0,vertex,axis]),float(clow[0,axis]),float(chigh[0,axis]))
            if not all(math.isfinite(x) for x in values):return None
            dl=math.nextafter(values[0]-values[3],-math.inf)
            dh=math.nextafter(values[1]-values[2],math.inf)
            al,ah=abs(dl),abs(dh)
            nearest=0. if dl<=0<=dh else min(al,ah);farthest=max(al,ah)
            lower=max(0.,math.nextafter(nearest*nearest,-math.inf))
            upper=math.nextafter(farthest*farthest,math.inf)
            total_low=max(0.,math.nextafter(total_low+lower,-math.inf))
            total_high=math.nextafter(total_high+upper,math.inf)
            if not math.isfinite(total_low) or not math.isfinite(total_high):return None
        lows.append(total_low);highs.append(total_high)
    return lows,highs,radius_low,radius_high


def _ball_intervals(low, high, clow, chigh, radii):
    """Shared directed predicate for one triangle/many balls or paired batches."""
    if type(radii) is np.ndarray and radii.shape==(1,):
        scalar=_single_ball_bounds(low,high,clow,chigh,radii)
        if scalar is not None:
            lows,highs,radius_low,radius_high=scalar
            return np.asarray([all(v<=radius_low for v in highs)]),np.asarray([any(v>radius_high for v in lows)])
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        dlo = np.nextafter(low-chigh[:, None, :], -np.inf)
        dhi = np.nextafter(high-clow[:, None, :], np.inf)
        abslo, abshi = np.abs(dlo), np.abs(dhi)
        minabs = np.where((dlo <= 0) & (dhi >= 0), 0., np.minimum(abslo, abshi))
        maxabs = np.maximum(abslo, abshi)
        np.multiply(minabs, minabs, out=minabs)
        np.nextafter(minabs, -np.inf, out=minabs)
        np.maximum(0., minabs, out=minabs)
        sqlo = minabs
        np.multiply(maxabs, maxabs, out=maxabs)
        np.nextafter(maxabs, np.inf, out=maxabs)
        sqhi = maxabs
        total_low, total_high = np.zeros((len(radii), low.shape[-2])), np.zeros((len(radii), low.shape[-2]))
        for axis in range(3):
            # Private accumulators only: retain each addition and directed
            # rounding in axis order without allocating replacement arrays.
            np.add(total_low, sqlo[:, :, axis], out=total_low)
            np.nextafter(total_low, -np.inf, out=total_low)
            np.maximum(0., total_low, out=total_low)
            np.add(total_high, sqhi[:, :, axis], out=total_high)
            np.nextafter(total_high, np.inf, out=total_high)
        radius_squared = radii*radii
        radius_low = np.nextafter(radius_squared, -np.inf)
        np.maximum(0., radius_low, out=radius_low)
        radius_high = np.nextafter(radius_squared, np.inf)
        inside = np.isfinite(total_high).all(axis=1) & (total_high <= radius_low[:, None]).all(axis=1)
        outside = (total_low > radius_high[:, None]).any(axis=1)
    return inside, outside


def boxes_in_balls(low, high, radii, *, center_bounds):
    """Sufficient containment of each whole AABB, with no optimistic fallback."""
    return _ball_intervals(low[:,None,:], high[:,None,:], *center_bounds, np.asarray(radii, float))[0]


def contains_balls(triangle: Triangle, centers: tuple[Bary, ...], radii: Iterable[float], *,
                   triangle_bounds=None, center_bounds=None) -> tuple[np.ndarray, int]:
    """Batch the same outward interval predicate, with exact ambiguous fallbacks."""
    low, high = bounds(triangle) if triangle_bounds is None else triangle_bounds
    clow, chigh = bounds(centers) if center_bounds is None else center_bounds
    radii = np.asarray(radii, float)
    if not len(radii):return np.zeros(0, bool), 0
    inside, outside = _ball_intervals(low[None, :, :], high[None, :, :], clow, chigh, radii)
    ambiguous = np.flatnonzero(~inside & ~outside)
    for index in ambiguous:
        center, r2 = centers[index], Q(float(radii[index]))**2
        inside[index] = all(dot(subtract(p, center), subtract(p, center)) <= r2 for p in triangle)
    return inside, len(ambiguous)


def contains_ball_pairs(triangles, centers, radii, *, triangle_bounds, center_bounds):
    """Verify each triangle/ball pair independently using the same predicate."""
    radii = np.asarray(radii, float)
    if not len(radii):return np.zeros(0, bool), 0
    inside, outside = _ball_intervals(*triangle_bounds, *center_bounds, radii)
    ambiguous = np.flatnonzero(~inside & ~outside)
    for index in ambiguous:
        center, r2 = centers[index], Q(float(radii[index]))**2
        inside[index] = all(dot(subtract(p, center), subtract(p, center)) <= r2 for p in triangles[index])
    return inside, len(ambiguous)


def edge_interval(source_face: int, source_vertices: tuple[int, int, int], a: Bary, b: Bary) -> tuple | None:
    """Canonical line and exact positive subinterval; no XYZ welding."""
    if a == b:return None
    for absent in range(3):
        if a[absent] == b[absent] == 0:
            axes = [i for i in range(3) if i != absent]
            lo_id, hi_id = sorted(source_vertices[i] for i in axes)
            axis = source_vertices.index(hi_id)
            return ('source_edge', lo_id, hi_id), min(a[axis], b[axis]), max(a[axis], b[axis])
    x1, y1, d1 = _planar_integer(a)
    x2, y2, d2 = _planar_integer(b)
    aa, bb, cc = y1*d2-y2*d1, x2*d1-x1*d2, x1*y2-x2*y1
    scale = aa if aa else bb
    line = (Q(aa, scale), Q(bb, scale), Q(cc, scale))
    axis = 1 if a[1] != b[1] else 2
    return ('face_line', source_face, line), min(a[axis], b[axis]), max(a[axis], b[axis])


def integer_edge_intervals(source_face: int, source_vertices: tuple[int, int, int], triangle: Triangle):
    """Equivalent canonical lines and rational intervals without Fraction keys.

    This is an in-memory representation for direct proof-path rechecking.
    Source IDs still separate unrelated/coincident faces; strict overlap is
    tested by integer cross multiplication and cannot close a positive gap.
    """
    weights = []
    for point in triangle:
        denominator = math.lcm(*(v.denominator for v in point))
        weights.append(tuple(v.numerator*(denominator//v.denominator) for v in point))
    return homogeneous_edge_intervals(source_face, source_vertices, tuple(weights))


def homogeneous_edge_intervals(source_face, source_vertices, triangle):
    planar = tuple((v[1], v[2], sum(v)) for v in triangle)
    result = []
    for i, j in ((0, 1), (1, 2), (2, 0)):
        a, b = triangle[i], triangle[j]
        x1, y1, d1 = planar[i]; x2, y2, d2 = planar[j]
        aa, bb = y1*d2-y2*d1, x2*d1-x1*d2
        if not aa and not bb:continue
        for absent in range(3):
            if a[absent] == b[absent] == 0:
                ids = sorted(source_vertices[k] for k in range(3) if k != absent)
                axis = source_vertices.index(ids[1])
                key = ('source_edge', *ids)
                n1, d1, n2, d2 = a[axis], sum(a), b[axis], sum(b)
                break
        else:
            cc = x1*y2-x2*y1
            divisor = math.gcd(aa, bb, cc)
            if (aa or bb) < 0:divisor = -divisor
            key = ('face_line', source_face, (aa//divisor, bb//divisor, cc//divisor))
            n1, n2 = (x1, x2) if bb else (y1, y2)
        if n1*d2 > n2*d1:n1, d1, n2, d2 = n2, d2, n1, d1
        result.append((key, n1, d1, n2, d2))
    return tuple(result)


def integer_subedges_overlap(left, right):
    ka, al, ad, ah, ae = left; kb, bl, bd, bh, be = right
    return ka == kb and ah*bd > bl*ae and bh*ad > al*be
