"""Exact pixel partition tests over the canonical binary mesh and calibration.

This proves a projected pixel partition, not physical accuracy or treatment
coverage. No tolerance fills a gap: both source separation and clipped area are
rational. Cached source geometry is immutable and local to one construction.
"""
from fractions import Fraction as Q
from functools import lru_cache
from math import lcm
from weakref import ref

import numpy as np


ALGORITHM = 'reference-pixel-exact-partition-v2'


def _construction_cache(method):
    """Bound each cache to one construction without retaining its owner."""
    owner, function = ref(method.__self__), method.__func__

    @lru_cache(maxsize=4096)
    def cached(*args):
        current = owner()
        if current is None:
            raise ReferenceError('Pixel partition cache owner was released')
        return function(current, *args)

    return cached


def _clip(points, x, y):
    for axis, bound, direction in ((0, Q(2*x-1, 2), 1), (0, Q(2*x+1, 2), -1),
                                   (1, Q(2*y-1, 2), 1), (1, Q(2*y+1, 2), -1)):
        output = []
        for a, b in zip(points, points[1:]+points[:1]):
            da, db = direction*(a[axis]-bound), direction*(b[axis]-bound)
            if da >= 0:
                output.append(a)
            if (da < 0) != (db < 0):
                ratio = da/(da-db)
                output.append(tuple(i+ratio*(j-i) for i, j in zip(a, b)))
        points = output
        if len(points) < 3:
            return []
    return points


def _area(points):
    if len(points) < 3:
        return Q(0)
    return abs(sum(a[0]*b[1]-a[1]*b[0] for a, b in zip(points, points[1:]+points[:1])))/2


def _separated(a, b):
    """Closed convex triangles may touch; only positive interior overlap matters."""
    for polygon in (a, b):
        for start, end in zip(polygon, polygon[1:]+polygon[:1]):
            dx, dy = end[0]-start[0], end[1]-start[1]
            if dx == dy == 0:
                continue
            pa = tuple((v[0]-start[0])*dy-(v[1]-start[1])*dx for v in a)
            pb = tuple((v[0]-start[0])*dy-(v[1]-start[1])*dx for v in b)
            if min(max(pa), max(pb)) <= max(min(pa), min(pb)):
                return True
    return False


def _cross(a, b, point):
    return (b[0]-a[0])*(point[1]-a[1])-(b[1]-a[1])*(point[0]-a[0])


def _inside(triangle, point):
    signs=tuple(_cross(a,b,point) for a,b in zip(triangle,triangle[1:]+triangle[:1]))
    return min(signs)>=0 or max(signs)<=0


def _integer_line(a, b):
    """Scale an exact oriented edge equation by one positive denominator.

    The resulting integer equation has the same sign at every rational point.
    Pixel centres are integers, so repeated cell tests need no Fraction work.
    """
    dx,dy=b[0]-a[0],b[1]-a[1]
    values=(-dy,dx,dy*a[0]-dx*a[1])
    denominator=lcm(*(value.denominator for value in values))
    return tuple(value.numerator*(denominator//value.denominator) for value in values)


def _cell_edge(a, b, line=None):
    """Prepare the exact integer-centre exclusions for an open pixel cell."""
    maximum_x,minimum_x=max(a[0],b[0])+Q(1,2),min(a[0],b[0])-Q(1,2)
    maximum_y,minimum_y=max(a[1],b[1])+Q(1,2),min(a[1],b[1])-Q(1,2)
    # ceil(max + 1/2), floor(min - 1/2); equality stays outside the open cell.
    limits=(-((-maximum_x.numerator)//maximum_x.denominator),
            minimum_x.numerator//minimum_x.denominator,
            -((-maximum_y.numerator)//maximum_y.denominator),
            minimum_y.numerator//minimum_y.denominator)
    return (*limits,*(_integer_line(a,b) if line is None else line))


class PixelPartition:
    def __init__(self, triangles, intrinsic, check=lambda: None):
        self.triangles = np.array(triangles, dtype=float, copy=True)
        self.triangles.setflags(write=False)
        self.fx, self.fy = Q(float(intrinsic[0, 0])), Q(float(intrinsic[1, 1]))
        self.cx, self.cy = Q(float(intrinsic[0, 2])), Q(float(intrinsic[1, 2]))
        self.check = check
        self.projected = _construction_cache(self._projected)
        self.separated = _construction_cache(self._separated)
        self.prepared = _construction_cache(self._prepared)
        self.patch = _construction_cache(self._patch)

    def _projected(self, face):
        return tuple((Q(float(x))/Q(float(z))*self.fx+self.cx,
                      Q(float(y))/Q(float(z))*self.fy+self.cy)
                     for x, y, z in self.triangles[face])

    def _separated(self, left, right):
        return _separated(self.projected(left), self.projected(right))

    def _prepared(self, face):
        """Share exact source-edge integers across overlapping patch keys."""
        triangle=self.projected(face)
        inside=[];boundary=[]
        for a,b in zip(triangle,triangle[1:]+triangle[:1]):
            line=_integer_line(a,b)
            inside.append(line)
            # Boundary tests use abs(line), so directed signs do not matter.
            # Reuse these same integer objects in both prepared predicates.
            boundary.append(_cell_edge(a,b,line))
        return tuple(inside),tuple(boundary)

    def _patch(self, faces):
        edges={}
        for index,face in enumerate(faces):
            self.check()
            for pair_index,other in enumerate(faces[:index]):
                if pair_index%64==0:self.check()
                if not self.separated(min(face,other),max(face,other)):
                    return None
            triangle=self.projected(face)
            if _cross(*triangle)==0:return None
            for edge_index,(a,b) in enumerate(zip(triangle,triangle[1:]+triangle[:1])):
                edge=(a,b) if a<b else (b,a)
                if edge in edges:edges[edge][0]+=1
                else:edges[edge]=[1,face,edge_index]
        if any(record[0]>2 for record in edges.values()):return None
        # Cache only immutable exact geometry, with the same bounded patch keys.
        # Source disjointness and the exact area fallback remain unchanged.
        inside=tuple(self.prepared(face)[0] for face in faces)
        boundary=tuple(self.prepared(face)[1][edge_index]
                       for count,face,edge_index in edges.values() if count==1)
        return inside,boundary

    def _whole_cell(self, triangles, boundary, x, y):
        """No patch boundary meets the open cell, and its centre is inside.

        With disjoint triangles, membership is constant within that open cell.
        The line/bbox exclusions below use exact comparisons, never a tolerance.
        A failed exclusion falls back to exact clipped area.
        """
        for right,left,bottom,top,a,b,c in boundary:
            if x>=right or x<=left or y>=bottom or y<=top:continue
            # Both sides are scaled by the same positive exact denominator.
            if 2*abs(a*x+b*y+c)<abs(a)+abs(b):return False
        for triangle in triangles:
            signs=tuple(a*x+b*y+c for a,b,c in triangle)
            if min(signs)>=0 or max(signs)<=0:return True
        return False

    def classify(self, faces, x, y):
        """Unknown on any unproved disjointness, including overlap plus a gap."""
        self.check()
        # Dense pixels bypass the patch cache; a cached patch has <=16 faces.
        patch=self.patch(tuple(faces)) if len(faces)<=16 else self._patch(tuple(faces))
        if patch is None:return 8
        if self._whole_cell(*patch,x,y):return 1
        area = Q(0)
        for face in faces:
            self.check()
            area += _area(_clip(list(self.projected(face)), x, y))
        return 1 if area == 1 else 10 if 0 < area < 1 else 8
