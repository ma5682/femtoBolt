"""Research-only scan-edge midpoint proposals with bounded rounding error.

These are not registered coverage seeds and do not alter an accepted route or
its footprint model. A caller must first validate the entire route, select and
version a compatible nominal model, bind the proposed records, and replay all
coverage and geometry checks before any evidence can be published.
"""
from __future__ import annotations

from fractions import Fraction
import math
import numpy as np

from femto.planning.coverage_domain import scan_endpoints
from femto.planning.surface_trajectory import ScanSegment

VERSION='scan-edge-midpoint-rounding-bound-proposal-v1'
DEFAULT_MIN_EDGE_M=.002
MAX_RECORDS=20_000


def conservative_radius(radius_m,record):
    """Deduct an upward-bounded midpoint rounding error from the ball radius."""
    if isinstance(radius_m,bool) or not isinstance(radius_m,(int,float)) or not math.isfinite(radius_m) or radius_m<=0:
        raise ValueError('中点名义半径无效')
    bound=record.get('midpoint_error_bound_m') if isinstance(record,dict) else None
    if isinstance(bound,bool) or not isinstance(bound,(int,float)) or not math.isfinite(bound) or bound<0 or bound>=radius_m:
        raise ValueError('中点舍入误差无效')
    result=math.nextafter(radius_m-bound,-math.inf) if bound else float(radius_m)
    if result<=0:raise ValueError('中点保守半径无效')
    return result


def propose(trajectory,original_faces,*,min_edge_m=DEFAULT_MIN_EDGE_M,max_records=MAX_RECORDS,check=lambda:None):
    """Return existing endpoints and new bounded midpoint records separately.

    Only nonzero scan edges longer than ``min_edge_m`` are considered; no
    connector creates a seed. Candidate source faces are the two edge endpoint
    faces; downstream exact anchor construction must still reject unsupported
    faces. The L1 rounding bound encloses the Euclidean midpoint error.
    """
    if (isinstance(min_edge_m,bool) or not isinstance(min_edge_m,(int,float))
            or not math.isfinite(min_edge_m) or min_edge_m<0
            or type(max_records) is not int or not 0<=max_records<=MAX_RECORDS):
        raise ValueError('中点生成参数无效')
    faces=np.asarray(original_faces)
    if faces.ndim!=1 or not np.issubdtype(faces.dtype,np.integer) or np.any(faces<0):
        raise ValueError('中点原始面映射无效')
    endpoints=scan_endpoints(trajectory,faces)
    seen={(p['source_face'],*p['xyz_m']) for p in endpoints};new=[];examined=eligible=0
    for item in trajectory.ordered_items:
        if not isinstance(item,ScanSegment):continue
        for i,(a,b) in enumerate(zip(item.points,item.points[1:])):
            examined+=1
            if examined%256==0:check()
            role=b.get('role')
            if role not in (None,'scan','connector'):raise ValueError('扫描边角色无效')
            if role=='connector' or a['xyz_m']==b['xyz_m']:continue
            left=np.asarray(a['xyz_m'],float);right=np.asarray(b['xyz_m'],float)
            if left.shape!=(3,) or right.shape!=(3,) or not np.isfinite(left).all() or not np.isfinite(right).all():
                raise ValueError('扫描边坐标无效')
            length=float(np.linalg.norm(left-right))
            if not math.isfinite(length):raise ValueError('扫描边长度无效')
            if length<=min_edge_m:continue
            eligible+=1
            mid=[(float(x)+float(y))/2 for x,y in zip(left,right)]
            if not all(math.isfinite(v) for v in mid):raise ValueError('中点坐标无效')
            error=sum(abs(Fraction(z)-(Fraction(float(x))+Fraction(float(y)))/2) for x,y,z in zip(left,right,mid))
            bound=math.nextafter(float(error),math.inf) if error else 0.
            if not math.isfinite(bound) or Fraction(bound)<error:raise ValueError('中点舍入误差无法保守表示')
            local_faces={a['triangle_index'],b['triangle_index']}
            if any(type(face) is not int or not 0<=face<len(faces) for face in local_faces):raise ValueError('中点来源面无效')
            for local in sorted(local_faces):
                source=int(faces[local]);key=(source,*mid)
                if key in seen:continue
                if len(new)>=max_records:raise ValueError('中点候选数量超过预算')
                seen.add(key);new.append(dict(source_face=source,xyz_m=mid,role='scan',segment_index=item.segment_index,
                    edge_index=i,endpoint=2,provenance='rounded_midpoint_with_l1_error_deduction',midpoint_error_bound_m=bound))
    check()
    return endpoints,new,dict(version=VERSION,min_edge_m=float(min_edge_m),max_records=max_records,
        examined_edges=examined,eligible_edges=eligible,midpoint_records=len(new),
        max_rounding_bound_m=max((p['midpoint_error_bound_m'] for p in new),default=0.))
