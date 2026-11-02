"""Transfer dyadic work requests to identical verified scan edges.

Floating-point positions and conservative error bounds are regenerated, not
copied. This is a candidate correspondence, never a coverage or motion proof.
Both routes and the original face map must be validated by the caller.
"""
from __future__ import annotations

import numpy as np
from femto.planning import scan_fraction_seeds
from femto.planning.coverage_domain import scan_endpoints
from femto.planning.surface_trajectory import ScanSegment

VERSION='research-identical-edge-fraction-transfer-v1'
MAX_EDGES=200_000
MAX_COMPARISONS=1_000_000


def transfer(source,target,original_faces,requests,*,check=lambda:None):
    """Match whole edges by exact endpoints and source faces, then rederive seeds."""
    check()
    original,_=scan_fraction_seeds.propose(source,original_faces,requests,check=check)
    faces=np.asarray(original_faces);source_segments={};target_segments={};lookup={};count=0
    for trajectory,segments,is_target in ((source,source_segments,False),(target,target_segments,True)):
        for segment in trajectory.ordered_items:
            if not isinstance(segment,ScanSegment):continue
            if type(segment.segment_index) is not int or segment.segment_index in segments:
                raise ValueError('边内位置转移的扫描段索引无效或重复')
            segments[segment.segment_index]=segment
            for edge,(a,b) in enumerate(zip(segment.points,segment.points[1:])):
                count+=1
                if count%256==0:check()
                if count>MAX_EDGES:raise ValueError('边内位置转移扫描边超过预算')
                role=b.get('role')
                if role not in (None,'scan','connector'):raise ValueError('扫描边角色无效')
                if not is_target or role=='connector' or a['xyz_m']==b['xyz_m']:continue
                ids=(a.get('triangle_index'),b.get('triangle_index'))
                if any(type(i) is not int or not 0<=i<len(faces) for i in ids):raise ValueError('扫描边来源面无效')
                left,right=tuple(a['xyz_m']),tuple(b['xyz_m'])
                if len(left)!=3 or len(right)!=3 or not np.isfinite([left,right]).all():raise ValueError('扫描边坐标无效')
                key=tuple(sorted((left,right)));support=frozenset(int(faces[i]) for i in ids)
                lookup.setdefault(key,[]).append((segment.segment_index,edge,left,right,support))
    new_requests=[];seen=set();unmapped=0;comparisons=0
    for idx,request in enumerate(requests):
        if idx%128==0:check()
        segment,edge,numerator,denominator=request
        a,b=source_segments[segment].points[edge:edge+2]
        left,right=tuple(a['xyz_m']),tuple(b['xyz_m']);key=tuple(sorted((left,right)))
        support={int(faces[a['triangle_index']]),int(faces[b['triangle_index']])}
        matches=lookup.get(key,());comparisons+=len(matches)
        if comparisons>MAX_COMPARISONS:raise ValueError('边内位置转移匹配超过预算')
        compatible=[entry for entry in matches if support<=entry[4]]
        if not compatible:unmapped+=1;continue
        segment2,edge2,start,end,_=min(compatible,key=lambda v:(v[0],v[1]))
        transferred=numerator if (start,end)==(left,right) else denominator-numerator
        candidate=(segment2,edge2,transferred,denominator)
        if candidate not in seen:new_requests.append(candidate);seen.add(candidate)
    positions,proposal=scan_fraction_seeds.propose(target,original_faces,new_requests,check=check)
    supported={}
    for point in scan_endpoints(target,faces)+positions:
        key=(point['source_face'],*point['xyz_m']);bound=point.get('midpoint_error_bound_m',0.)
        supported[key]=min(supported.get(key,float('inf')),bound)
    matched=sum(supported.get((p['source_face'],*p['xyz_m']),float('inf'))<=p['midpoint_error_bound_m'] for p in original)
    report=dict(version=VERSION,source_requests=len(requests),transferred_requests=len(new_requests),
        unmapped_requests=unmapped,source_records=len(original),matched_records=matched,unmatched_records=len(original)-matched,
        target_proposal=proposal,semantics='candidate_correspondence_only_not_coverage_or_physical_motion')
    check()
    return new_requests,positions,report
