"""Propose scan-owned support faces for verified connector endpoints.

Both incident edges must pass the existing whole-edge predicate;
unaccepted endpoint changes retain the original connector unchanged.
"""
from __future__ import annotations

from copy import deepcopy
from femto.planning.scan_visit_order import checked_visits,reconstruct
from femto.planning.surface_trajectory import ScanSegment

VERSION='scan-owned-connector-end-support-v1'


def _oriented_point(segment, reverse, index):
    """Copy one point with exactly the roles used by scan_visit_order.oriented."""
    size=len(segment.points);position=index if index>=0 else size+index
    if not 0<=position<size:raise IndexError('扫描点索引无效')
    point=deepcopy(segment.points[-1-position] if reverse else segment.points[position])
    if reverse:point['role']='scan' if position==0 else segment.points[-position].get('role','scan')
    point.pop('revisit',None)
    return point


def propose(segments,visits,links,mesh,record,width,allowed,check=lambda:None):
    """Return copied links and a report; callers must validate the full route."""
    visits=checked_visits(segments,visits)
    reconstruct(segments,visits,links,check)
    result=deepcopy(links);candidates=[];edges=[]
    for index,link in enumerate(result):
        check()
        if not link['connected']:continue
        previous,previous_reverse=segments[visits[index][0]],visits[index][1]
        following,following_reverse=segments[visits[index+1][0]],visits[index+1][1]
        end=link['samples'][-1];first=_oriented_point(following,following_reverse,0)
        if end['xyz_m']!=first['xyz_m']:raise ValueError('连接终点与扫描起点不一致')
        if end['triangle_index']==first['triangle_index']:continue
        candidate=deepcopy(end);candidate['triangle_index']=first['triangle_index']
        prior=link['samples'][-2] if len(link['samples'])>1 else _oriented_point(previous,previous_reverse,-1)
        # Check the incoming connector and outgoing original scan, separately.
        for a,b in ((prior,candidate),(candidate,_oriented_point(following,following_reverse,1))):
            edges.append(ScanSegment(len(edges),b['row_index'],[deepcopy(a),deepcopy(b)],following.block_index))
        candidates.append((index,candidate))
    if edges:
        from femto.planning.quality_constrained import edge_checks
        flags,_,_=edge_checks(edges,mesh,record,width,check,allowed_faces=allowed)
        if len(flags)!=len(edges):raise ValueError('连接端点支持面的整边检查数量不一致')
    else:flags=[]
    adopted=[];rejected=[]
    for position,(index,candidate) in enumerate(candidates):
        check()
        if bool(flags[2*position]) and bool(flags[2*position+1]):
            result[index]['samples'][-1]=candidate;adopted.append(index)
        else:rejected.append(index)
    reconstruct(segments,visits,result,check)
    return result,dict(version=VERSION,candidate_count=len(candidates),adopted_links=adopted,rejected_links=rejected,
                       semantics='whole_edge_rechecked_proposal_not_coverage_certificate')


def validate_record(decision,segments,visits,original,final,mesh,record,width,allowed,check=lambda:None):
    """Validate the saved change and recompute both incident checks when possible."""
    if (not isinstance(decision,dict) or set(decision)!={'version','candidate_count','adopted_links','rejected_links','semantics'}
            or decision.get('version')!=VERSION
            or decision.get('semantics')!='whole_edge_rechecked_proposal_not_coverage_certificate'
            or type(decision.get('candidate_count')) is not int or decision['candidate_count']<0
            or not all(isinstance(decision.get(k),list) for k in ('adopted_links','rejected_links'))):
        raise ValueError('连接终点支持面记录无效')
    visits=checked_visits(segments,visits);reconstruct(segments,visits,original,check)
    candidates={}
    for index,link in enumerate(original):
        check()
        if not link['connected']:continue
        first=_oriented_point(segments[visits[index+1][0]],visits[index+1][1],0)
        end=link['samples'][-1]
        if end['triangle_index']!=first['triangle_index']:candidates[index]=first['triangle_index']
    adopted=decision['adopted_links'];rejected=decision['rejected_links']
    if (any(type(i) is not int for i in adopted+rejected) or adopted!=sorted(set(adopted))
            or rejected!=sorted(set(rejected)) or set(adopted)&set(rejected)
            or set(adopted)|set(rejected)!=set(candidates) or decision['candidate_count']!=len(candidates)):
        raise ValueError('连接终点支持面候选或结果不一致')
    expected=deepcopy(original)
    for index in adopted:expected[index]['samples'][-1]['triangle_index']=candidates[index]
    if final!=expected:raise ValueError('连接终点支持面变更与记录不一致')
    if mesh is not None:
        fresh,actual=propose(segments,visits,original,mesh,record,width,allowed,check)
        if fresh!=final or actual!=decision:raise ValueError('连接终点支持面检查不能独立重算')
    reconstruct(segments,visits,final,check)
