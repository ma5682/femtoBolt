"""Bounded proposal/replan workflow for isolated short scan rows.

An unsuccessful graph search is evidence to try a row phase, not proof of an
obstacle. Each trial is regenerated and passes the unchanged quality and graph
predicates. No proposal changes the original target or claims coverage.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

SCHEMA='femto-isolated-row-repair-v1'
MAX_TRIALS=2


def _internal_gaps(trajectory):
    stats=trajectory.statistics
    return stats['gap_count']-stats['inter_block_gap_count']


def _accept(initial,trial,width):
    a,b=initial.statistics,trial.statistics
    # The upper length allowance accounts for new verified turns. It is a
    # numerical proposal gate, never a coverage or physical safety claim.
    return (_internal_gaps(trial)<_internal_gaps(initial)
            and b['unplannable_face_count']<=a['unplannable_face_count']
            and b['total_length_m']<=a['total_length_m']+10*width/1000)


def generate(source,mesh,region,quality,width,domain,layout_request,*,check=lambda:None,routing=None,**values):
    from femto.planning.continuous_quality import candidate,replan,ORDERED_ROUTING,GLOBAL_FLOOR_ROUTING,REFINED_CONNECTOR_ROUTING
    from femto.planning.isolated_row_repair import requests
    if routing is None:routing=ORDERED_ROUTING
    if routing in (GLOBAL_FLOOR_ROUTING,REFINED_CONNECTOR_ROUTING):
        from femto.planning.global_search_budget_research import SearchBudget
        search=SearchBudget()
    else:search=None
    initial_candidate=candidate(source,check=check,fixed_width_layout=layout_request,layout_evidence=quality,**values)
    initial=replan(initial_candidate,mesh,region,quality,width,domain=domain,check=check,routing=routing,
                   **({'search_budget':search} if search is not None else {}))
    proposals=requests(initial,check)
    attempts=[];selected=None;result=initial
    for proposal in proposals:
        for shift in proposal['trial_shifts']:
            if len(attempts)>=MAX_TRIALS:break
            check()
            try:
                trial_candidate=candidate(source,check=check,fixed_width_layout=dict(request=layout_request,row_shifts=[shift]),
                                          layout_evidence=quality,**values)
                trial=replan(trial_candidate,mesh,region,quality,width,domain=domain,check=check,routing=routing,
                             **({'search_budget':search} if search is not None else {}))
            except ValueError as error:
                if not str(error).startswith(('局部行位置调整','局部行调整的原位置','局部行调整缺少相邻扫描行')):raise
                attempts.append(dict(shift=deepcopy(shift),status='infeasible_spacing'))
                continue
            accepted=_accept(initial,trial,width)
            attempts.append(dict(shift=deepcopy(shift),status='accepted' if accepted else 'not_improved',
                                 internal_gaps=_internal_gaps(trial),total_length_m=trial.statistics['total_length_m']))
            if accepted:selected=deepcopy(shift);result=trial;break
        if selected is not None or len(attempts)>=MAX_TRIALS:break
    report=dict(schema=SCHEMA,selected=selected,attempts=attempts,
                original_internal_gaps=_internal_gaps(initial),
                failed_search_is_proof_of_disconnection=False,
                target_coverage_claim='none',initial_trajectory=initial.to_dict() if selected is not None else None)
    if search is not None:report['search_budget']=search.report()
    return replace(result,gap_report=dict(result.gap_report,row_repair=report))


def validate(trajectory,mesh=None,region=None,check=lambda:None):
    """Recheck input binding, initial evidence and the admitted decision."""
    from femto.planning.surface_trajectory import SurfaceTrajectory
    from femto.planning.isolated_row_repair import requests
    params=trajectory.planner['parameters'];layout=params['fixed_width_layout'];report=trajectory.gap_report.get('row_repair')
    if (not isinstance(report,dict) or report.get('schema')!=SCHEMA
            or report.get('target_coverage_claim')!='none'
            or report.get('failed_search_is_proof_of_disconnection') is not False
            or not isinstance(report.get('attempts'),list) or len(report['attempts'])>MAX_TRIALS):
        raise ValueError('局部短行调整报告无效')
    selected=report.get('selected');shifts=layout.get('row_shifts')
    from femto.planning.continuous_quality import GLOBAL_FLOOR_ROUTING,REFINED_CONNECTOR_ROUTING
    if params.get('continuous_routing') in (GLOBAL_FLOOR_ROUTING,REFINED_CONNECTOR_ROUTING):
        from femto.planning.global_search_budget_research import validate_report
        total=validate_report(report.get('search_budget'))
        route_budget=trajectory.gap_report.get('reconnection',{}).get('ordering',{}).get('search_budget')
        used=validate_report(route_budget)
        if any(used[key]>total[key] for key in total) or (selected is not None and used!=total):
            raise ValueError('局部调整与全流程搜索预算不一致')
    if selected is None:
        if shifts!=[] or report.get('initial_trajectory') is not None:
            raise ValueError('未采用的局部行调整包含实际偏移')
        if report.get('original_internal_gaps')!=_internal_gaps(trajectory):
            raise ValueError('局部行调整前断点计数不一致')
        return
    if shifts!=[selected] or not report['attempts'] or report['attempts'][-1].get('shift')!=selected or report['attempts'][-1].get('status')!='accepted':
        raise ValueError('局部行调整与采用记录不一致')
    initial=SurfaceTrajectory.from_dict(report.get('initial_trajectory'),mesh=mesh,region=region,check=check)
    if initial.planning_context is not None:raise ValueError('局部行调整基准上下文无效')
    left={k:v for k,v in initial.planner['parameters'].items() if k!='fixed_width_layout'}
    right={k:v for k,v in params.items() if k!='fixed_width_layout'}
    # Export creates a fresh input snapshot. Its path and manifest checksum can
    # change while the selected geometry and intent remain exactly the same.
    binding=('mesh_file_sha256','geometry_sha256','region_id','region_signature','selection_intent_sha256')
    if (left!=right or any(initial.source.get(k)!=trajectory.source.get(k) for k in binding)
            or initial.planner['parameters']['fixed_width_layout']['request']!=layout['request']
            or initial.planner['parameters']['fixed_width_layout'].get('row_shifts')!=[]
            or report.get('original_internal_gaps')!=_internal_gaps(initial)):
        raise ValueError('局部行调整基准输入与路线不一致')
    proposals=requests(initial,check)
    if not any(selected in item['trial_shifts'] for item in proposals):
        raise ValueError('局部行调整不是已记录短行的有界候选')
    if not _accept(initial,trajectory,params['effective_width_mm']):
        raise ValueError('局部行调整未改善内部连接')
