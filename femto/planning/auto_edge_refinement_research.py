"""Bounded automatic scan-row refinement with independent nominal evidence.

All results are detached research outputs. No route, coverage certificate or
equipment command is installed; complete physical coverage is not asserted.
"""
from __future__ import annotations

from copy import deepcopy
from femto.planning import (auto_fraction_supplement_research,auto_row_shift_screen_research,
    coverage_certificate as coverage,coverage_partition,coverage_workflow,planning_revision,
    planning_state,row_proximity_research,surface_workspace,witness_bend_workflow_research)
from femto.planning.continuous_quality import RELOCATION_ROUTING, DEEP_RELOCATION_ROUTING
from femto.planning.coverage_domain import from_plan
from femto.planning.coverage_retention_union import _same_target
from femto.planning.fixed_width_layout import SHIFT_PATCH_VERSION
from femto.planning.surface_trajectory import SurfaceTrajectory

VERSION='research-auto-edge-refinement-v1'
MAX_VERIFIED_CANDIDATES=4


def _rank(trial):
    s=trial['statistics'];d=trial['density']['proximity']
    return (s['connected_detours'],s['revisit_connector_length_m'],s['connector_length_m'],
            s['total_length_m'],*[v['near_fraction'] for v in d],trial['proposal_index'])


def run(model,region,plan,*,old_certificate=None,max_rows=8,max_candidates=8,max_rank=8,
        max_verified_candidates=2,proof_budget=None,check=lambda:None,progress=lambda text:None,rank_strategy='local',
        evidence_strategy='intermediate',old_guide=None):
    """Screen proposals and reprove a bounded number; never install a result."""
    if (type(max_verified_candidates) is not int or not 1<=max_verified_candidates<=MAX_VERIFIED_CANDIDATES
            or evidence_strategy not in ('intermediate','direct_witness','single_final','ranked_replace','paired_replace')):
        raise ValueError('自动细化覆盖核验候选预算无效')
    check();snapshot=planning_revision.digest(plan);region_snapshot=planning_revision.digest(region)
    head=plan['treatment_head'];revision=plan['planning_context']['revision']
    certificate_snapshot=None
    guide_snapshot=None if old_guide is None else planning_revision.digest(old_guide.metadata)
    planning_revision.require_current(model,region,head,revision)
    def guard():
        check()
        if planning_revision.digest(plan)!=snapshot or planning_revision.digest(region)!=region_snapshot:
            raise ValueError('自动细化期间原路线或选区发生变化')
        if certificate_snapshot is not None and planning_revision.digest(old_certificate.metadata)!=certificate_snapshot:
            raise ValueError('自动细化期间原证据发生变化')
        if old_guide is not None:
            coverage._checked_assets(old_guide,check)
            if planning_revision.digest(old_guide.metadata)!=guide_snapshot:raise ValueError('自动细化期间原引导发生变化')
        planning_revision.require_current(model,region,head,revision);check()
    progress('筛选局部扫描行候选')
    screen,first=auto_row_shift_screen_research.screen(model,region,plan,max_rows=max_rows,max_candidates=max_candidates,
        rank_strategy=rank_strategy,max_rank=max_rank,check=check,progress=progress)
    ranked=sorted((trial for trial in screen['trials'] if trial['accepted']),key=_rank)
    report=dict(schema=VERSION,adopted=False,reason='no_screen_candidate',screen=screen,trials=[],
        installed_route=False,installed_certificate=False,full_target_coverage_claim=False,
        execution_ready=False,physical_accuracy_verified=False)
    if not ranked:
        if first is not None:raise ValueError('候选筛选结果不一致')
        guard();return report,None
    if first is None or ranked[0]['proposal_index']!=screen['selected_index']:
        raise ValueError('候选筛选排序结果不一致')
    if plan.get('coverage_evidence') is not None:
        existing=coverage_workflow.cached(model,plan)
        if old_certificate is None:old_certificate=existing
        elif old_certificate.metadata!=existing.metadata:
            raise ValueError('自动细化必须保留原方案已有覆盖证据')
    if old_certificate is None:
        progress('建立原路线名义覆盖证据')
        old_certificate,_=coverage_workflow.calculate(model,region,plan,check=check,progress=progress,budget=proof_budget)
    certificate_snapshot=planning_revision.digest(old_certificate.metadata)
    old_domain,old_seeds,old_width,_=from_plan(model,region,plan,check)
    old_route=coverage_workflow.route_binding(SurfaceTrajectory.from_dict(plan['surface_trajectory'],check=check))
    from femto.planning.old_coverage_session_research import _OldCoverageSession
    with _OldCoverageSession(model,region,plan,old_certificate,old_domain,old_seeds,old_width,old_route,check,guide=old_guide) as session:
        report,selected=_reprove_candidates(model,region,plan,head,old_certificate,old_domain,old_width,ranked,first,
            report,max_verified_candidates,proof_budget,check,progress,guard,session,evidence_strategy)
        if selected is not None and old_guide is not None:selected['old_guide']=old_guide
        return report,selected


def _reprove_candidates(model,region,plan,head,old_certificate,old_domain,old_width,ranked,first,
                         report,max_verified_candidates,proof_budget,check,progress,guard,session,evidence_strategy):
    selected=None
    for position,trial in enumerate(ranked[:max_verified_candidates]):
        check();progress(f'核验候选覆盖 {position+1}/{min(len(ranked),max_verified_candidates)}')
        if position==0:candidate=first
        else:
            routing=(DEEP_RELOCATION_ROUTING if plan['planning_context']['revision'].get('inputs',{}).get('algorithms',{}).get('routing')==DEEP_RELOCATION_ROUTING
                     else RELOCATION_ROUTING)
            candidate=surface_workspace.generate(model,region,head,check=check,layout_algorithm=SHIFT_PATCH_VERSION,
                routing_algorithm=routing,research_row_shifts=[deepcopy(trial['proposal']['row_shift'])])
            planning_state.validate(candidate,model.store.mesh,region,check=check)
            trajectory=SurfaceTrajectory.from_dict(candidate['surface_trajectory'],check=check)
            if trajectory.statistics!=trial['statistics'] or row_proximity_research.measure(trajectory,check=check)!=trial['density']:
                raise ValueError('候选重规划结果与筛选记录不一致')
        domain,seeds,width,_=from_plan(model,region,candidate,check)
        _same_target(old_domain,domain,check)
        if old_width!=width:raise ValueError('候选作用宽度与原路线不一致')
        route=coverage_workflow.route_binding(SurfaceTrajectory.from_dict(candidate['surface_trajectory'],check=check))
        single_attempt=None;single_fallback=None
        if evidence_strategy in ('single_final','ranked_replace'):
            if evidence_strategy=='single_final':
                from femto.planning import single_final_witness_research as candidate_research
            else:
                from femto.planning import ranked_point_witness_research as candidate_research
            progress('直接构造见证候选并核验最终证据')
            try:
                kwargs={'progress':progress} if evidence_strategy=='ranked_replace' else {}
                single_attempt,final=candidate_research.build(model,region,plan,old_certificate,candidate,
                    proof_budget=proof_budget,check=check,_old_proof_session=session,**kwargs)
            except ValueError as error:
                budget_errors=('分数位置分批待确认单元超过预算','分数位置分批距离筛选超过预算',
                    '分数位置分批次数超过预算','分数位置分批请求超过预算','扫描边位置候选超过预算',
                    '旧见证候选超过预算','见证点局部调整比较超过预算','见证点候选复制超过预算',
                    '见证叶排序超过预算')
                if str(error) not in budget_errors:raise
                check();guard();single_fallback=str(error)
            if single_attempt is not None and single_attempt['adopted']:
                selected=dict(candidate_plan=candidate,old_certificate=old_certificate,final_certificate=final,
                    replay_strategy='edge_insertion_direct' if evidence_strategy=='single_final' else 'point_replacement_direct',
                    direct_recipe=deepcopy(single_attempt['direct_recipe']))
                entry=dict(proposal_index=trial['proposal_index'],fraction_comparison=None,edge_result=single_attempt,
                           evidence_strategy=evidence_strategy,fallback_reason=None,reason=single_attempt['reason'])
                report['trials'].append(entry);guard()
                coverage._checked_assets(selected['old_certificate'],check);coverage._checked_assets(selected['final_certificate'],check)
                report.update(adopted=True,reason=entry['reason'],selected_index=trial['proposal_index']);break
            if single_fallback is None:single_fallback=single_attempt['reason']
            progress('单份最终证据未形成结果，回退候选覆盖核验')
        mode='direct_witness' if evidence_strategy in ('single_final','ranked_replace','paired_replace') else evidence_strategy
        fallback=None;fraction=prior=None
        if mode=='direct_witness':
            from femto.planning import direct_fraction_witness_research
            progress('从原见证点推导候选扫描边核验位置')
            try:
                fraction,prior=direct_fraction_witness_research.propose(model,region,plan,old_certificate,candidate,
                    budget=proof_budget,check=check,_old_proof_session=session)
            except ValueError as error:
                budget_errors=('分数位置分批待确认单元超过预算','分数位置分批距离筛选超过预算',
                               '分数位置分批次数超过预算','分数位置分批请求超过预算',
                               '扫描边位置候选超过预算')
                if str(error) not in budget_errors:raise
                check();guard();fallback=str(error)
        def intermediate():
            algorithm=coverage.REPROOF_ALGORITHM if coverage.checked_page_binding(domain.binding) else coverage.INDEXED_PARTIAL_REPROOF_ALGORITHM
            progress('核验候选路线的名义覆盖')
            normal=coverage.evaluate(domain,seeds,width,route_binding=route,guide=old_certificate,
                                     algorithm=algorithm,budget=proof_budget,check=check)
            progress('检查候选扫描边上的覆盖见证')
            return auto_fraction_supplement_research.propose(model,region,plan,old_certificate,candidate,normal,
                budget=proof_budget,check=check,_old_proof_session=session)
        def assess(fraction,prior):
            single=coverage_partition.compare_union(old_certificate,[prior],domain.areas,check,include_pending_leaves=True)
            entry=dict(proposal_index=trial['proposal_index'],fraction_comparison=single,edge_result=None)
            if single['retained_subset']:
                entry['reason']='fraction_proof_retained'
                result=dict(candidate_plan=candidate,old_certificate=old_certificate,final_certificate=prior,
                            replay_strategy='fraction_reproof',fraction_requests=deepcopy(fraction['requests']))
            else:
                progress('重建见证点并核验最终路线')
                if evidence_strategy=='paired_replace':
                    from femto.planning import paired_witness_workflow_research
                    edge,final=paired_witness_workflow_research.build(model,region,plan,old_certificate,candidate,
                        fraction['requests'],prior,proof_budget=proof_budget,check=check,progress=progress,
                        _old_proof_session=session)
                else:
                    edge,final=witness_bend_workflow_research.build(model,region,plan,old_certificate,candidate,
                        fraction['requests'],prior,placement='edge_insertion',proof_budget=proof_budget,check=check,
                        _old_proof_session=session)
                entry['edge_result']=edge;entry['reason']=edge['reason'];result=None
                if edge['adopted']:
                    result=dict(candidate_plan=candidate,old_certificate=old_certificate,final_certificate=final)
                    if edge.get('route_unchanged'):
                        result.update(replay_strategy='fraction_reproof',fraction_requests=deepcopy(edge['fraction_requests']))
                    else:
                        result.update(replay_strategy='paired_replacement_direct' if evidence_strategy=='paired_replace' else 'edge_insertion_direct',
                                      direct_recipe=deepcopy(edge['direct_recipe']))
            return entry,result
        if fraction is None:fraction,prior=intermediate()
        entry,selected=assess(fraction,prior)
        direct_attempt=None
        if mode=='direct_witness' and fallback is None and selected is None:
            direct_attempt=entry;fallback='direct_no_qualifying_candidate';check();guard()
            progress('直接见证未形成结果，回退原有候选覆盖核验')
            fraction,prior=intermediate();entry,selected=assess(fraction,prior)
        if mode=='direct_witness':
            entry.update(evidence_strategy='intermediate_fallback' if fallback else 'direct_witness',fallback_reason=fallback)
            if direct_attempt is not None:entry['direct_attempt']=direct_attempt
        if evidence_strategy in ('single_final','ranked_replace'):
            entry['single_attempt']=single_attempt;entry['single_fallback_reason']=single_fallback
        report['trials'].append(entry);guard()
        if selected is not None:
            coverage._checked_assets(selected['old_certificate'],check);coverage._checked_assets(selected['final_certificate'],check)
            report.update(adopted=True,reason=entry['reason'],selected_index=trial['proposal_index']);break
    if selected is None:
        report['reason']='no_coverage_retained_candidate'
    report['coverage_candidate_budget_exhausted']=len(ranked)>max_verified_candidates
    session.require(model,region,plan,old_certificate,old_domain,session.seeds,old_width,session.route,check)
    guard();return report,selected
