"""Research-only dyadic scan-edge positions with rounding-error bounds.

This does not validate the route, admit coverage, or register a footprint
policy. A caller must validate every source edge and independently bind and
replay a versioned model before any coverage evidence can be published.
"""
from __future__ import annotations

from fractions import Fraction
import math
import numpy as np
from femto.planning.coverage_domain import scan_endpoints
from femto.planning.surface_trajectory import ScanSegment

VERSION='scan-edge-dyadic-position-rounding-bound-proposal-v1'
MAX_REQUESTS=20_000
MAX_DENOMINATOR=256


def propose(trajectory,original_faces,requests,*,max_records=20_000,check=lambda:None):
    """Propose bounded dyadic positions on nonzero, active scan edges.

    Requests are ``(segment_index, edge_index, numerator, denominator)``;
    denominators must be powers of two. Endpoints and duplicate source/XYZ
    records are omitted. Candidate source faces are the edge's endpoint faces;
    exact downstream anchor construction must still reject unsupported faces.
    """
    if (not isinstance(requests,list) or len(requests)>MAX_REQUESTS
            or type(max_records) is not int or not 0<=max_records<=MAX_REQUESTS):
        raise ValueError('扫描边位置请求超过预算或无效')
    planner=getattr(trajectory,'planner',{})
    if planner.get('algorithm_version') in ('nominal-variable-width-surface-v1','nominal-variable-width-refined-surface-v1'):
        raise ValueError('研究位置仅支持固定名义宽度路线')
    faces=np.asarray(original_faces)
    if faces.ndim!=1 or not np.issubdtype(faces.dtype,np.integer) or np.any(faces<0):raise ValueError('扫描边原始面映射无效')
    segments={}
    for item in trajectory.ordered_items:
        if not isinstance(item,ScanSegment):continue
        if type(item.segment_index) is not int or item.segment_index in segments:raise ValueError('扫描段索引无效或重复')
        segments[item.segment_index]=item
    endpoints=scan_endpoints(trajectory,faces)
    seen={(p['source_face'],*p['xyz_m']) for p in endpoints};output=[];seen_requests=set()
    for index,request in enumerate(requests):
        if index%128==0:check()
        if (not isinstance(request,(tuple,list)) or len(request)!=4 or any(type(x) is not int for x in request)):
            raise ValueError('扫描边位置请求无效')
        segment_id,edge,numerator,denominator=request
        if (segment_id not in segments or not 0<=edge<len(segments[segment_id].points)-1
                or denominator<2 or denominator>MAX_DENOMINATOR or denominator&(denominator-1)
                or not 0<numerator<denominator):raise ValueError('扫描边位置或分数无效')
        key=tuple(request)
        if key in seen_requests:continue
        seen_requests.add(key)
        a,b=segments[segment_id].points[edge:edge+2]
        if b.get('role') not in (None,'scan') or a['xyz_m']==b['xyz_m']:
            raise ValueError('扫描边位置不能使用连接或零长度边')
        left=np.asarray(a['xyz_m'],dtype=float);right=np.asarray(b['xyz_m'],dtype=float)
        if left.shape!=(3,) or right.shape!=(3,) or not np.isfinite(left).all() or not np.isfinite(right).all():raise ValueError('扫描边坐标无效')
        fraction=numerator/denominator
        point=[(1-fraction)*float(x)+fraction*float(y) for x,y in zip(left,right)]
        if not all(math.isfinite(x) for x in point):raise ValueError('扫描边位置坐标溢出')
        error=sum(abs(Fraction(z)-((denominator-numerator)*Fraction(float(x))+numerator*Fraction(float(y)))/denominator)
                  for x,y,z in zip(left,right,point))
        bound=math.nextafter(float(error),math.inf) if error else 0.
        if not math.isfinite(bound) or Fraction(bound)<error:raise ValueError('扫描边位置误差无法保守表示')
        local_faces={a['triangle_index'],b['triangle_index']}
        if any(type(face) is not int or not 0<=face<len(faces) for face in local_faces):raise ValueError('扫描边位置来源面无效')
        for local in sorted(local_faces):
            source=int(faces[local]);identity=(source,*point)
            if identity in seen:continue
            if len(output)>=max_records:raise ValueError('扫描边位置候选超过预算')
            seen.add(identity);output.append(dict(source_face=source,xyz_m=point,role='scan',segment_index=segment_id,
                edge_index=edge,endpoint=3,provenance='dyadic_edge_position_with_l1_error_deduction',
                fraction_numerator=numerator,fraction_denominator=denominator,midpoint_error_bound_m=bound))
    check()
    return output,dict(version=VERSION,request_count=len(requests),unique_requests=len(seen_requests),
        proposed_records=len(output),max_rounding_bound_m=max((p['midpoint_error_bound_m'] for p in output),default=0.),
        semantics='candidate_only_not_coverage_or_physical_motion')
