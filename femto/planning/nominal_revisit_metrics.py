"""Operation-local arithmetic for the retained nominal revisit score.

Only scan proximity and scalar lengths are reused. Nothing here verifies a
surface edge or a treatment footprint. Final route reconstruction and geometry
validation remain independent, with the retained full scorer as a fallback.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from threading import get_ident
import math
import numpy as np

MAX_SCAN_POINTS = 50_000
MAX_ROUTE_POINTS = 200_000
MAX_QUERIES = 20_000
MAX_NEIGHBOURS = 100_000
MAX_LENGTHS = 20_000
_active = ContextVar('nominal_revisit_metrics', default=None)


@contextmanager
def operation():
    current=_active.get()
    if current is not None and current.active and current.thread==get_ident():
        yield
        return
    state=_Cache();token=_active.set(state)
    try:yield
    finally:
        state.active=False;state.clear();_active.reset(token)


class _Cache:
    def __init__(self):
        self.thread=get_ident();self.active=True
        self.clear()

    def clear(self):
        self.current=frozenset();self.points=frozenset();self.tree=None;self.keys=();self.spacing=None
        self.queries={};self.lengths={};self.neighbours=0

    def prepare(self,scan,spacing,check):
        # Query results are sets of row identities: repeated identical scan
        # points and permutations cannot change the original membership test.
        current=frozenset(scan);check()
        if not current.issubset(self.points):
            combined=self.points|current
            if len(combined)>MAX_SCAN_POINTS:combined=current
            import open3d as o3d
            keys=tuple(sorted(combined))
            cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray([p[:3] for p in keys],dtype=np.float64)))
            self.tree=o3d.geometry.KDTreeFlann(cloud);self.keys=keys;self.points=combined
            self.queries.clear();self.neighbours=0
        if spacing!=self.spacing:self.queries.clear();self.neighbours=0
        # A moved scan head can disappear from a particular candidate. Always
        # filter the union index by this complete candidate's actual scan set.
        self.current=current;self.spacing=spacing;check()

    def nearby(self,xyz,spacing):
        result=self.queries.get(xyz)
        if result is None:
            _,found,_=self.tree.search_radius_vector_3d(np.asarray(xyz,dtype=np.float64),spacing/2)
            result=tuple(self.keys[i] for i in found)
            if len(self.queries)<MAX_QUERIES and self.neighbours+len(result)<=MAX_NEIGHBOURS:
                self.queries[xyz]=result;self.neighbours+=len(result)
        return result

    def length(self,left,right):
        key=(left,right);value=self.lengths.get(key)
        if value is None:
            # Same scalar NumPy expression as _walk_connectors, not a batch sum.
            value=float(np.linalg.norm(np.asarray(right)-np.asarray(left)))
            if len(self.lengths)<MAX_LENGTHS:self.lengths[key]=value
        return value


def measure(chains,spacing,check):
    """Return exact legacy fields, or None for the retained implementation."""
    state=_active.get()
    if state is None or not state.active or state.thread!=get_ident():return None
    if type(chains) not in (list,tuple):return None
    if type(spacing) not in (int,float) or not math.isfinite(spacing) or spacing<=0:return None
    scan=[];parsed=[];total=0
    for chain in chains:
        if type(chain) not in (list,tuple) or len(chain)!=2:return None
        _,points=chain
        if type(points) is not list or not points or type(points[0]) is not dict or points[0].get('role')=='connector':return None
        row=[]
        for point in points:
            total+=1
            if total>MAX_ROUTE_POINTS:return None
            if total%256==0:check()
            if type(point) is not dict:return None
            xyz=point.get('xyz_m');identity=point.get('row_index');role=point.get('role')
            if (type(xyz) not in (list,tuple) or len(xyz)!=3
                    or any(type(x) is not float or not math.isfinite(x) or abs(x)>1e100 for x in xyz)
                    or type(identity) is not int or not -(2**63)<=identity<2**63
                    or role not in (None,'scan','connector')):return None
            xyz=tuple(xyz);row.append((xyz,identity,role,point.get('revisit') is True))
            if role!='connector':
                scan.append((*xyz,identity))
                if len(scan)>MAX_SCAN_POINTS:return None
        parsed.append(row)
    if not scan:return None
    check();state.prepare(scan,spacing,check)
    detours=0;revisit=0.;queries=0
    for points in parsed:
        index=0;last_row=None
        while index<len(points):
            xyz,identity,role,marked=points[index]
            if role!='connector':last_row=identity;index+=1;continue
            start=index
            while index<len(points) and points[index][2]=='connector':index+=1
            next_row=points[index][1] if index<len(points) else points[start][1]
            detours+=int(next_row==last_row)
            joined={last_row,next_row}
            for offset in range(start,index):
                queries+=1
                if queries%256==0:check()
                xyz,_,_,marked=points[offset]
                if marked or any(p in state.current and p[3] not in joined for p in state.nearby(xyz,spacing)):
                    # Preserve the original sequential addition across the full
                    # route; summing per-link subtotals would change rounding.
                    revisit+=state.length(points[offset-1][0],xyz)
    check()
    return dict(detours=detours,revisit_connector_m=revisit)
