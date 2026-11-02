"""Rank, never accept or reject, row-shift work by preconnection scan density.

Each prospective scan is sliced and constrained on the actual reference mesh;
no connector or final route is generated here. Preconnection samples may not
equal final samples on every input. A caller must validate any selected route
and independently reprove its coverage.
"""
from __future__ import annotations

from copy import deepcopy
from femto.geometry.region_reuse import operation
from femto.planning import (continuous_quality, planning_revision, planning_state,
    quality_constrained, row_proximity_research, surface_workspace)
from femto.planning.fixed_width_layout import SHIFT_PATCH_VERSION, request

VERSION='research-preconnection-scan-density-ranking-v1'
MAX_RANK=32


def rank(model,region,plan,proposal,*,max_rank=16,check=lambda:None,progress=lambda text:None,_source=None):
    """Return original proposal indices ordered by sampled scan proximity."""
    if type(max_rank) is not int or not 1<=max_rank<=MAX_RANK:
        raise ValueError('连接前排序预算无效')
    if (not isinstance(proposal,dict) or not isinstance(proposal.get('proposals'),list)
            or len(proposal['proposals'])>MAX_RANK or proposal.get('layout_algorithm')!=SHIFT_PATCH_VERSION):
        raise ValueError('连接前排序候选无效或超过预算')
    check();snapshot=planning_revision.digest(plan)
    if proposal.get('source_route_sha256')!=snapshot:raise ValueError('连接前排序的来源路线不一致')
    head=plan['treatment_head'];revision=plan['planning_context']['revision']
    planning_revision.require_current(model,region,head,revision)
    with operation():
        if _source is None:
            original=planning_state.validated_surface(plan,model.store.mesh,region,check=check)
        else:
            from femto.planning.screen_source_reuse import _ScreenSource
            if type(_source) is not _ScreenSource or _source.check is not check:
                raise ValueError('连接前排序源路线复用上下文无效')
            original=_source.read(model,region,plan)
        params=original.planner['parameters'];record=params['quality_evidence'];domain=params['effective_domain']
        ranked=[];count=min(max_rank,len(proposal['proposals']))
        if count:
            manifest=surface_workspace.snapshot(model,region,check,include_texture=False)
            local=surface_workspace.canonical_region_mesh(model.store.mesh,region)
            spec=request(head['effective_treatment_width_mm'],head['overlap_percent'],version=SHIFT_PATCH_VERSION)
        for index,item in enumerate(proposal['proposals'][:count]):
            check();progress(f'估算扫描行密度 {index+1}/{count}')
            if not isinstance(item,dict) or not isinstance(item.get('row_shift'),dict):
                raise ValueError('连接前排序的偏移记录无效')
            raw=continuous_quality.candidate(manifest,row_spacing_mm=head['row_spacing_mm'],check=check,
                fixed_width_layout=dict(request=spec,row_shifts=[deepcopy(item['row_shift'])]),layout_evidence=record)
            constrained=quality_constrained.constrain(raw,local,region,record,head['effective_treatment_width_mm'],
                                                     check,domain=domain)
            density=row_proximity_research.measure(constrained,check=check)
            fractions=[value['near_fraction'] for value in density['proximity']]
            if any(value is None for value in fractions):raise ValueError('连接前排序没有可采样扫描边')
            ranked.append(dict(index=index,density=density))
            if planning_revision.digest(plan)!=snapshot:raise ValueError('连接前排序期间来源路线已改变')
            planning_revision.require_current(model,region,head,revision)
        ranked.sort(key=lambda value:(*[item['near_fraction'] for item in value['density']['proximity']],value['index']))
        order=[value['index'] for value in ranked]+list(range(count,len(proposal['proposals'])))
        check()
        if planning_revision.digest(plan)!=snapshot:raise ValueError('连接前排序期间来源路线已改变')
        planning_revision.require_current(model,region,head,revision)
        if _source is not None:_source.verify()
    return dict(version=VERSION,order=order,ranked=ranked,ranked_count=count,
        unranked_count=len(proposal['proposals'])-count,source_route_sha256=snapshot,
        candidate_only=True,coverage_claim=False,execution_ready=False,
        semantics='ranking_only_final_route_and_coverage_require_independent_verification')
