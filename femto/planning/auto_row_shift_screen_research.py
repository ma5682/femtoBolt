"""Bounded geometric screening of automatically proposed row shifts.

The returned route is an isolated research candidate, not installed, proven
to retain coverage, suitable for treatment, or ready for equipment execution.
"""
from __future__ import annotations

from copy import deepcopy
from femto.geometry.region_reuse import operation
from femto.planning import (planning_revision, planning_state, row_proximity_research,
    row_shift_proposal_research, surface_workspace, trim_subset_research)
from femto.planning.surface_trajectory import SurfaceTrajectory

VERSION='research-auto-row-shift-motion-screen-v1'
MAX_CANDIDATES=16
EPS=1e-12


def _acceptable(original, candidate, old_density, new_density):
    old=original.statistics;new=candidate.statistics
    return (trim_subset_research._motion_ok(new,old)
        and new['total_length_m']<=old['total_length_m']+EPS
        and new['gap_count']-new['inter_block_gap_count']<=old['gap_count']-old['inter_block_gap_count']
        and trim_subset_research._density_not_worse(new_density,old_density)
        # A longer scan must not dilute the fraction while leaving more
        # absolute near-row length. Require improvement in both measures.
        and all(a['near_sample_m']<=b['near_sample_m']+EPS
                for a,b in zip(new_density['proximity'],old_density['proximity']))
        and any(a['near_sample_m']<b['near_sample_m']-EPS
                for a,b in zip(new_density['proximity'],old_density['proximity']))
        and any(a['near_fraction']<b['near_fraction']-EPS
                for a,b in zip(new_density['proximity'],old_density['proximity'])))


def screen(model,region,plan,*,max_rows=8,max_candidates=8,fraction=.03,rank_strategy='original',max_rank=16,check=lambda:None,
           progress=lambda text:None):
    """Replan bounded proposals and return the best motion/density candidate.

    Candidate whole-edge validation is repeated before comparison. No coverage
    is evaluated here; the caller must independently replay source and final
    evidence, and must not treat this research candidate as installable.
    """
    if type(max_candidates) is not int or not 1<=max_candidates<=MAX_CANDIDATES:
        raise ValueError('自动行偏移筛选候选预算无效')
    if rank_strategy not in ('original','preconnection','local','joint_preview','joint_partial'):
        raise ValueError('自动行偏移候选排序方式无效')
    check(); snapshot=planning_revision.digest(plan)
    head=plan['treatment_head'];revision=plan['planning_context']['revision']
    planning_revision.require_current(model,region,head,revision)
    from femto.planning.screen_source_reuse import _ScreenSource
    with operation(), _ScreenSource(model,region,plan,check) as source:
        proposal=row_shift_proposal_research.propose(model,region,plan,max_rows=max_rows,fraction=fraction,check=check,_source=source)
        original=source.read(model,region,plan)
        old_density=row_proximity_research.measure(original,check=check)
        ranking=None;fallback_reason=None
        if rank_strategy in ('local','joint_preview','joint_partial'):
            from femto.planning import local_row_rank_research
            try:
                kwargs=({'motion_priority':True,'allow_partial':True,'prefer_known_motion':True} if rank_strategy=='joint_partial'
                        else {'motion_priority':True} if rank_strategy=='joint_preview' else {})
                ranking=local_row_rank_research.rank(model,region,plan,proposal,max_rank=max_rank,check=check,progress=progress,_source=source,**kwargs)
            except local_row_rank_research.PreviewUnavailable as error:
                check();source.verify();fallback_reason=str(error)
        if rank_strategy=='preconnection' or (rank_strategy in ('local','joint_preview','joint_partial') and ranking is None):
            from femto.planning import preconnection_rank_research
            ranking=preconnection_rank_research.rank(model,region,plan,proposal,max_rank=max_rank,check=check,progress=progress,_source=source)
        if ranking is not None:
            order=ranking['order']
        else:order=list(range(len(proposal['proposals'])))
        trials=[];best=None
        for sequence,index in enumerate(order[:max_candidates]):
            item=proposal['proposals'][index]
            check();progress(f'核验行偏移候选 {sequence+1}/{min(max_candidates,len(order))}')
            source.verify()
            candidate=surface_workspace.generate(model,region,head,check=check,
                layout_algorithm=proposal['layout_algorithm'],routing_algorithm=proposal['routing_algorithm'],
                research_row_shifts=[item['row_shift']])
            planning_state.validate(candidate,model.store.mesh,region,check=check)
            trajectory=SurfaceTrajectory.from_dict(candidate['surface_trajectory'],check=check)
            density=row_proximity_research.measure(trajectory,check=check)
            accepted=_acceptable(original,trajectory,old_density,density)
            statistics=trajectory.statistics
            trials.append(dict(proposal_index=index,proposal=deepcopy(item),accepted=accepted,statistics=deepcopy(statistics),density=density))
            if accepted:
                rank=(statistics['connected_detours'],statistics['revisit_connector_length_m'],
                      statistics['connector_length_m'],statistics['total_length_m'],
                      *[v['near_fraction'] for v in density['proximity']],index)
                if best is None or rank<best[0]:best=(rank,index,candidate)
            if planning_revision.digest(plan)!=snapshot:raise ValueError('自动行偏移筛选期间输入路线已改变')
            planning_revision.require_current(model,region,head,revision)
            source.verify()
        check()
        if planning_revision.digest(plan)!=snapshot:raise ValueError('自动行偏移筛选期间输入路线已改变')
        planning_revision.require_current(model,region,head,revision)
        source.verify()
    report=dict(schema=VERSION,proposals=len(proposal['proposals']),tested=len(trials),trials=trials,
        rank_strategy=rank_strategy,ranking=ranking,ranking_fallback_reason=fallback_reason,
        source_route_sha256=snapshot,source_density=old_density,
        selected_index=None if best is None else best[1],candidate_found=best is not None,
        candidate_budget_exhausted=len(proposal['proposals'])>max_candidates,
        installed_route=False,coverage_verified=False,full_target_coverage_claim=False,
        execution_ready=False,physical_accuracy_verified=False,
        semantics='motion_and_sampled_density_only_requires_independent_coverage_replay')
    return report,None if best is None else best[2]
