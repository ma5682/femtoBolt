"""Replayable single-budget routing stage for the explicitly selected floor mode."""
from copy import deepcopy
import math
from numbers import Real

import numpy as np
from femto.planning import budgeted_relocation_research as budgeted


def _supported(segments):
    """The proxy index only handles ordinary single-row scan-only segments."""
    for segment in segments:
        if len(segment.points)<2:return False
        rows=set()
        for point in segment.points:
            xyz=point.get('xyz_m');row=point.get('row_index')
            if (point.get('role') not in (None,'scan') or type(row) is not int or not 0<=row<2**63
                    or not isinstance(xyz,(list,tuple,np.ndarray)) or len(xyz)!=3
                    or any(isinstance(x,(bool,np.bool_)) or not isinstance(x,Real) or not math.isfinite(x) for x in xyz)):
                return False
            rows.add(row)
        if len(rows)!=1:return False
    return True


def global_motion_limits(segments,visits,links,spacing,check,budget):
    """Limits against the checked route before the detour and joint stages."""
    from femto.planning.scan_visit_order import reconstruct
    from femto.planning.balanced_visit_improvement import _metrics
    baseline=reconstruct(segments,visits,links,check,preserve_endpoint_faces=True,budget=budget)
    return dict(budgeted._limits_from(baseline,_metrics(baseline[0],spacing,check)),internal_gaps=baseline[2][0])


def check_global_motion(segments,visits,links,spacing,limits,check,budget):
    if limits is None:return
    from femto.planning.scan_visit_order import reconstruct
    from femto.planning.balanced_visit_improvement import _metrics
    route=reconstruct(segments,visits,links,check,preserve_endpoint_faces=True,budget=budget)
    measured=_metrics(route[0],spacing,check)
    if (route[2][0]>limits['internal_gaps'] or route[2][1]>limits['connector_length_m']+budgeted.EPS
            or measured['detours']>limits['detours']
            or measured['revisit_connector_m']>limits['revisit_connector_m']+budgeted.EPS):
        raise ValueError('全流程路线超过共同运动代价限制')


def _context(segments,visits,links,chains,blocks,classes,spacing,check,*,strict=False,budget=200_000,global_limits=None):
    from femto.planning.continuous_quality import _relocation_context
    skipped,baseline=_relocation_context(segments,visits,links,chains,blocks,classes,spacing,check)
    if skipped:return skipped,None
    if not _supported(segments):return 'unsupported_scan_structure',None
    length=baseline['connector_length_m'];revisit=baseline['revisit_connector_m']
    limits=dict(detours=baseline['detours'],
        connector_length_m=length+min(budgeted.MAX_CONNECTOR_INCREASE_M,length*budgeted.MAX_RELATIVE_INCREASE),
        revisit_connector_m=revisit+min(budgeted.MAX_REVISIT_INCREASE_M,revisit*budgeted.MAX_RELATIVE_INCREASE))
    if strict:
        measured=budgeted.limits_for(segments,visits,links,spacing,check=check,budget=budget)
        limits={key:min(limits[key],measured[key]) for key in limits}
    if global_limits is not None:
        limits={key:min(limits[key],global_limits[key]) for key in limits}
    return None,limits


def propose(graph,segments,visits,links,chains,added,blocks,classes,spacing,check,budget,*,search=None,global_limits=None):
    prior=deepcopy(links)
    skipped,limits=_context(segments,visits,prior,chains,blocks,classes,spacing,check,
                            strict=search is not None,budget=budget,global_limits=global_limits)
    if skipped:
        decision=dict(version=budgeted.VERSION,adopted=False,skipped=skipped)
    else:
        try:
            candidate,new_links,new_added,decision=budgeted.propose(graph,segments,visits,prior,spacing,
                motion_limits=limits,check=check,budget=budget,
                **({'search':search} if search is not None else {}))
        except ValueError as error:
            if str(error)!='连续路线整边搜索超过预算，请缩小选区':raise
            decision=dict(version=budgeted.VERSION,adopted=False,skipped='geometry_budget')
        if decision['adopted']:chains,links,added=candidate,new_links,new_added
    return chains,links,added,dict(decision,baseline_visits=[list(v) for v in visits],baseline_connections=prior)


def validate(record,segments,visits,links,chains,added,blocks,classes,params,mesh,region,allowed,check,budget,*,global_limits=None):
    from femto.planning import quality_constrained as quality
    from femto.planning.surface_trajectory import ScanSegment
    from femto.planning.selection_blocks import selection_face_assignment,check_projection_inside_selection
    if not isinstance(record,dict) or record.get('baseline_visits')!=[list(v) for v in visits]:
        raise ValueError('统一预算换序基准顺序不一致')
    prior=record.get('baseline_connections')
    prior_segments=[s for s in quality._items(chains) if isinstance(s,ScanSegment)]
    if mesh is not None:
        flags,_,_=quality.edge_checks(prior_segments,mesh,params['quality_evidence'],params['effective_width_mm'],check,allowed_faces=allowed)
        if not flags.all():raise ValueError('统一预算换序基准包含未核验连接')
        if region is not None:
            _,_,labels,_=selection_face_assignment(region,np.asarray(mesh.vertices),np.asarray(mesh.faces))
            check_projection_inside_selection(prior_segments,blocks,labels,
                np.asarray(region['view_reference']['undistorted_intrinsic']),np.asarray(region['view_reference']['image_size']))
    spacing=params['row_spacing_mm']/1000
    from femto.planning.continuous_quality import GLOBAL_FLOOR_ROUTING
    skipped,limits=_context(segments,visits,prior,chains,blocks,classes,spacing,check,
                            strict=params.get('continuous_routing')==GLOBAL_FLOOR_ROUTING,budget=budget,global_limits=global_limits)
    decision={k:v for k,v in record.items() if k not in ('baseline_visits','baseline_connections')}
    if skipped or decision.get('skipped')=='geometry_budget':
        reason=skipped or 'geometry_budget'
        if decision!=dict(version=budgeted.VERSION,adopted=False,skipped=reason) or links!=prior:
            raise ValueError('统一预算换序回退记录无效')
    else:
        proposal,new_added=budgeted.validate_record(decision,segments,visits,prior,links,spacing,
            motion_limits=limits,check=check,budget=budget)
        if decision['adopted']:chains,added=proposal,new_added
    return chains,added
