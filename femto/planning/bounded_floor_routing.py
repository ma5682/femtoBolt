"""Small, replayable detour-priority stage for the explicit spacing-floor mode.

The 12% allowance is a numerical search tradeoff against this stage's verified
input, not a physical tolerance or a guarantee against another layout. Legacy
ordering remains unchanged. Final surface/width/selection checks still apply.
"""
from copy import deepcopy
import numpy as np
from femto.planning import verified_relocation_research as relocation

ATTEMPTS=128
ROUNDS=4
LENGTH_FACTOR=1.12
POLICY='detour_first_with_limits'


def check_layout(params,*,allow_adaptive=False):
    from femto.planning.fixed_width_layout import FLOOR_PATCH_VERSION,ADAPTIVE_FLOOR_PATCH_VERSION,COORDINATED_FLOOR_PATCH_VERSION
    request=(params.get('fixed_width_layout') or {}).get('request') or {}
    fixed=request.get('version')==FLOOR_PATCH_VERSION and request.get('minimum_metric_fraction')==.65
    adaptive=(allow_adaptive and request.get('version') in (ADAPTIVE_FLOOR_PATCH_VERSION,COORDINATED_FLOOR_PATCH_VERSION)
              and 'minimum_metric_fraction' not in request)
    if not (fixed or adaptive):
        raise ValueError('疏密绕行优化需要 65% 尺度行距下限排布')


def _context(segments,visits,links,chains,blocks,classes,spacing,check):
    from femto.planning.continuous_quality import _relocation_context
    skipped,limits=_relocation_context(segments,visits,links,chains,blocks,classes,spacing,check)
    if limits is not None:
        limits=dict(limits,connector_length_m=limits['connector_length_m']*LENGTH_FACTOR,
                    revisit_connector_m=limits['revisit_connector_m']*LENGTH_FACTOR)
    return skipped,limits


def propose(graph,segments,visits,links,chains,added,blocks,classes,spacing,check,budget):
    prior=deepcopy(links)
    skipped,limits=_context(segments,visits,prior,chains,blocks,classes,spacing,check)
    if skipped:
        decision=dict(version=relocation.LIMITED_DETOUR_VERSION,adopted=False,skipped=skipped)
    else:
        try:
            candidate,new_links,new_added,decision=relocation.propose(graph,segments,visits,prior,spacing,
                motion_limits=limits,attempts=ATTEMPTS,rounds=ROUNDS,objective_policy=POLICY,check=check,budget=budget)
        except ValueError as error:
            if str(error)!='连续路线整边搜索超过预算，请缩小选区':raise
            decision=dict(version=relocation.LIMITED_DETOUR_VERSION,adopted=False,skipped='geometry_budget')
        if decision['adopted']:chains,links,added=candidate,new_links,new_added
    return chains,links,added,dict(decision,baseline_visits=[list(v) for v in visits],baseline_connections=prior)


def validate(record,segments,visits,links,chains,added,blocks,classes,params,mesh,region,allowed,check,budget):
    from femto.planning import quality_constrained as quality
    from femto.planning.surface_trajectory import ScanSegment
    from femto.planning.selection_blocks import selection_face_assignment,check_projection_inside_selection
    if not isinstance(record,dict) or record.get('baseline_visits')!=[list(v) for v in visits]:
        raise ValueError('疏密绕行优化基准顺序不一致')
    prior=record.get('baseline_connections')
    prior_segments=[s for s in quality._items(chains) if isinstance(s,ScanSegment)]
    if mesh is not None:
        flags,_,_=quality.edge_checks(prior_segments,mesh,params['quality_evidence'],params['effective_width_mm'],check,allowed_faces=allowed)
        if not flags.all():raise ValueError('疏密绕行优化基准包含未核验连接')
        if region is not None:
            _,_,labels,_=selection_face_assignment(region,np.asarray(mesh.vertices),np.asarray(mesh.faces))
            check_projection_inside_selection(prior_segments,blocks,labels,
                np.asarray(region['view_reference']['undistorted_intrinsic']),np.asarray(region['view_reference']['image_size']))
    spacing=params['row_spacing_mm']/1000
    skipped,limits=_context(segments,visits,prior,chains,blocks,classes,spacing,check)
    decision={k:v for k,v in record.items() if k not in ('baseline_visits','baseline_connections')}
    if skipped or decision.get('skipped')=='geometry_budget':
        reason=skipped or 'geometry_budget'
        if decision!=dict(version=relocation.LIMITED_DETOUR_VERSION,adopted=False,skipped=reason) or links!=prior:
            raise ValueError('疏密绕行优化回退记录无效')
    else:
        proposal,new_added=relocation.validate_record(decision,segments,visits,prior,links,spacing,
            motion_limits=limits,attempts=ATTEMPTS,rounds=ROUNDS,objective_policy=POLICY,check=check,budget=budget)
        if decision['adopted']:chains,added=proposal,new_added
    return chains,added
