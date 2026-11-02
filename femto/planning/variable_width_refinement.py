"""Independent P5 refinement: rebind every width and reprove every old claim.

P4 supplies conservative candidate construction, not coverage or variable-width
admission. One cumulative meter covers construction plus whole-route checks.
The caller verifies source assets/current inputs; archive replay is date-neutral.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
from types import SimpleNamespace

from femto.planning import coverage_refinement as fixed, coverage_certificate as coverage
from femto.planning import variable_width_route as variable, variable_footprint as footprint
from femto.planning import planning_revision as revisions, strict_path, strict_refinement
from femto.planning import quality_constrained as quality
from femto.planning.surface_trajectory import SurfaceTrajectory, ScanSegment, canonical_json_bytes, validate_trajectory

ALGORITHM = 'nominal-variable-width-refined-surface-v1'
SCHEMA = 'femto-variable-width-refinement-v1'
EDGE_SCHEMA = 'femto-variable-width-refined-edges-v1'


def algorithms_for(base, certificate, budget, candidate_policy, coverage_policy):
    from femto.planning.coverage_workflow import route_binding
    return dict(route=ALGORITHM, target=base.planning_context['revision']['inputs']['algorithms']['target'], refinement=dict(
        schema=SCHEMA, base_computation_sha256=base.planning_context['revision']['sha256'],
        base_route_sha256=revisions.digest(route_binding(base)), base_coverage_sha256=certificate.metadata['sha256'],
        model_sha256=base.planner['parameters']['variable_footprint']['sha256'], budget=asdict(budget),
        candidate_policy=deepcopy(candidate_policy), coverage_policy=deepcopy(coverage_policy),
        width_binding='recompute_all_chain_intervals_including_connectors',
        admission='common_partition_retention_and_positive_new_scan_witness'))


def checked_algorithms(algorithms):
    if not isinstance(algorithms,dict) or set(algorithms)!={'route','target','refinement'}:return False
    if algorithms['route']!=ALGORITHM or algorithms['target'] not in revisions.TARGET_ALGORITHMS:return False
    v=algorithms['refinement']
    fields={'schema','base_computation_sha256','base_route_sha256','base_coverage_sha256','model_sha256','budget',
            'candidate_policy','coverage_policy','width_binding','admission'}
    if not isinstance(v,dict) or set(v)!=fields or v['schema']!=SCHEMA:return False
    if (v['width_binding']!='recompute_all_chain_intervals_including_connectors'
            or v['admission']!='common_partition_retention_and_positive_new_scan_witness'):return False
    for key in ('base_computation_sha256','base_route_sha256','base_coverage_sha256','model_sha256'):
        if not isinstance(v[key],str) or len(v[key])!=64 or any(c not in '0123456789abcdef' for c in v[key]):return False
    try:
        from femto.planning.refinement_candidates_v2 import checked_policy
        checked_policy(v['candidate_policy']);fixed.RefinementBudget(**v['budget']).checked()
        if coverage.reproof_algorithm_for_policy(v['coverage_policy'])!=coverage.INTERLEAVED_ALGORITHM:return False
    except (TypeError,ValueError):return False
    return True


def context_for(base, certificate, budget, candidate_policy, coverage_policy):
    inputs=deepcopy(base.planning_context['revision']['inputs'])
    inputs['algorithms']=algorithms_for(base,certificate,budget,candidate_policy,coverage_policy)
    revision=revisions.create(inputs)
    target=deepcopy(base.planning_context['target']);target['computation_sha256']=revision['sha256']
    if target.get('construction_pages') is not None:
        from femto.planning.target_paging import rebind_header_records
        target['construction_pages']=rebind_header_records(target,revision)
    target['sha256']=revisions.digest({k:v for k,v in target.items() if k!='sha256'})
    return dict(schema=revisions.CONTEXT_SCHEMA,revision=revision,target=target)


def bounds(trajectory, model, check=lambda:None):
    rows=footprint.edge_bounds(trajectory,model,check)
    width=trajectory.planner['parameters']['effective_width_mm']
    if any(r['status']!='bounded_nominal' for r in rows):raise ValueError('变宽补充需要完整的分段线性上下界')
    if any(r['upper_half_width_m']>width/2000 for r in rows):raise ValueError('变宽段内上界超过原生成包络')
    for row in rows:row['outer_check_width_mm']=variable._diameter(row['upper_half_width_m'],width)
    return rows


def edge_report(rows, model, width):
    return dict(schema=EDGE_SCHEMA,records=rows,model_sha256=model['sha256'],checked_edges=len(rows),
        accepted_edges=sum(r.get('accepted') is True for r in rows),connector_edges=sum(r['role']=='connector' for r in rows),
        zero_inner_edges=sum(r['role']!='connector' and r['lower_half_width_m']==0 for r in rows),
        generation_envelope_width_mm=width,rebound_all_chain_intervals=True,preserved_connection_search_envelope=True,
        scope='nominal_model_surface_geometry_only',offset_bound_applied=False,physical_accuracy_verified=False,execution_ready=False)


def make(base, chains, context, report, rows, *, strict_base=None):
    strict_base=strict_base or SurfaceTrajectory.from_dict(base.gap_report['variable_width']['base_trajectory'])
    raw=fixed._route(strict_base,chains,context,report)
    model=base.planner['parameters']['variable_footprint']
    return replace(raw,planner=dict(raw.planner,kind='variable_width_refined_surface',algorithm_version=ALGORITHM,
        parameters=dict(raw.planner['parameters'],variable_footprint=deepcopy(model))),
        gap_report=dict(raw.gap_report,variable_width=edge_report(rows,model,raw.planner['parameters']['effective_width_mm'])))


def checked_rows(trajectory, mesh, allowed, meter, check):
    params=trajectory.planner['parameters'];model=params['variable_footprint']
    rows=bounds(trajectory,model,check);meter.reserve(len(rows))
    checker=strict_path.Checks(meter.support);diagnostics=[]
    flags,reasons,margins=quality.edge_checks([s for s in trajectory.ordered_items if isinstance(s,ScanSegment)],
        mesh,params['quality_evidence'],params['effective_width_mm'],check,allowed_faces=allowed,
        support_checker=checker,diagnostics=diagnostics,edge_widths_mm=[r['outer_check_width_mm'] for r in rows])
    strict_refinement._budget_exit((),checker.decisions)
    for row,flag,reason,margin,diag,support in zip(rows,flags,reasons,margins,diagnostics,checker.decisions):
        row.update(accepted=bool(flag),reason=reason,residual_margin_mm=float(margin),quality_diagnostic=diag,support_check=support)
    if len(flags)!=len(rows) or not flags.all():
        raise quality.PlanningConstraintBlocked('变宽候选全路线仍有未通过边，原路线保留。',edge_report(rows,model,params['effective_width_mm']))
    return rows


def scan_endpoints(trajectory, original_faces):
    from femto.planning.coverage_domain import scan_endpoints as geometric_endpoints
    shadow=replace(trajectory,planner=dict(trajectory.planner,algorithm_version=strict_path.REFINED_ALGORITHM))
    seeds=geometric_endpoints(shadow,original_faces)
    rows={(r['segment_index'],r['edge_index']):r for r in trajectory.gap_report['variable_width']['records']}
    for seed in seeds:
        row=rows[seed['segment_index'],seed['edge_index']]
        seed.update(width_interval=deepcopy(row['interval']),lower_half_width_m=row['lower_half_width_m'])
    return seeds


def refine(base,ledger,certificate,mesh,region,*,budget=None,check=lambda:None,progress=lambda text:None,
           frontiers=None,candidate_policy=None,coverage_policy=None):
    from femto.planning.coverage_domain import CoverageDomain
    from femto.planning.coverage_workflow import route_binding
    from femto.planning.coverage_partition import compare
    from femto.planning.effective_domain import validate as validate_domain
    from femto.planning.refinement_candidates_v2 import POLICY,checked_policy
    from femto.planning.coverage_interleaved import POLICY as COVERAGE_POLICY
    budget=(budget or fixed.RefinementBudget()).checked()
    if base.planner['algorithm_version']!=variable.ALGORITHM:raise ValueError('请先恢复补充前变宽路线')
    if certificate.metadata['algorithm'] not in coverage.P2_ALGORITHMS:raise ValueError('变宽补充需要当前完整覆盖资产')
    candidate_policy=deepcopy(checked_policy(POLICY if candidate_policy is None else candidate_policy))
    coverage_policy=deepcopy(COVERAGE_POLICY if coverage_policy is None else coverage_policy)
    if coverage.reproof_algorithm_for_policy(coverage_policy)!=coverage.INTERLEAVED_ALGORITHM:raise ValueError('变宽补充覆盖策略不一致')
    profile=footprint.validate(base.planner['parameters']['variable_footprint'])
    if certificate.metadata['inputs']['footprint']!=profile:raise ValueError('原覆盖与当前变宽模型不一致')
    params=base.planner['parameters'];allowed=validate_domain(params['effective_domain'],params['quality_evidence'],params['effective_width_mm'],mesh,region,check)
    context=context_for(base,certificate,budget,candidate_policy,coverage_policy)
    target=fixed.rebind_target(ledger,context['revision'])
    if target.metadata!=context['target']:raise ValueError('变宽补充改变了原目标')
    domain=CoverageDomain._checked_target(target,context['revision'])
    report=dict(schema=SCHEMA,base_trajectory=base.to_dict(),base_coverage=deepcopy(certificate.metadata),
        budget=asdict(budget),candidate_policy=candidate_policy,coverage_policy=coverage_policy,rounds=[],stop_reason='round_limit',
        edge_checks=0,coverage_work={},max_total_length_m=base.statistics['total_length_m']+budget.max_added_length_m,
        width_rebinding='every_trial_all_scan_and_connector_edges',failure_is_proof_of_infeasibility=False,
        registration_uncertainty_mm=None,physical_accuracy_verified=False,execution_ready=False)
    meter=fixed._Meter(budget);meter.candidate_policy=candidate_policy
    meter.support_budget=strict_path.SupportBudget(**params['surface_support']['budget']).checked()
    meter.support=strict_path.engine_for(mesh,allowed,meter.support_budget,check,region=region)
    strict_base=SurfaceTrajectory.from_dict(base.gap_report['variable_width']['base_trajectory'])
    current=make(base,fixed._chains(base),context,report,[],strict_base=strict_base)
    initial=checked_rows(current,mesh,allowed,meter,check)
    current=make(base,fixed._chains(base),context,report,initial,strict_base=strict_base)
    previous=certificate;history=[certificate];attempted=set();low=0;width_checks=len(initial)
    for index in range(budget.max_rounds):
        check();progress(f'变宽补充第 {index+1} 轮：构造候选并重新绑定全部边的宽度')
        try:chains,candidates,status=fixed._propose(current,domain,previous,mesh,region,allowed,attempted,meter,check)
        except fixed._Stop as error:chains,candidates,status=fixed._chains(current),[],str(error)
        record=dict(index=index,candidates=candidates,status=status,accepted=False,certificate_index=None,comparison=None,
            candidate_search=deepcopy(getattr(meter,'search_info',{})),variable_check=None)
        report['rounds'].append(record)
        if status.endswith('_budget'):report['stop_reason']=status;break
        if status!='proposed':
            record['status']='no_admitted_candidates';low+=1
            if not candidates:report['stop_reason']='no_eligible_candidates';break
        else:
            trial=make(base,chains,context,report,[],strict_base=strict_base)
            if trial.statistics['point_count']>budget.max_output_points:record['status']=report['stop_reason']='point_budget';break
            if trial.statistics['total_length_m']>report['max_total_length_m']:record['status']=report['stop_reason']='length_budget';break
            try:
                rows=checked_rows(trial,mesh,allowed,meter,check);width_checks+=len(rows)
            except fixed._Stop as error:record['status']=report['stop_reason']=str(error);break
            except quality.PlanningConstraintBlocked as error:
                record.update(status='variable_outer_unconfirmed',variable_check=error.report);low+=1
            else:
                record['variable_check']=edge_report(rows,profile,params['effective_width_mm'])
                trial=make(base,chains,context,report,rows,strict_base=strict_base)
                seeds=scan_endpoints(trial,target.arrays['region_faces'])
                if frontiers is not None and len(history)-1>=len(frontiers):raise ValueError('变宽补充缺少覆盖停止边界')
                fresh=coverage.evaluate(domain,seeds,profile,route_binding=route_binding(trial),
                    algorithm=coverage.INTERLEAVED_ALGORITHM,budget=meter.coverage_budget(),check=check,guide=previous,
                    _frontier=None if frontiers is None else frontiers[len(history)-1])
                for key in meter.counts:meter.counts[key]+=fresh.metadata['metrics'][key]
                record.update(certificate_index=len(history),coverage=deepcopy(fresh.metadata));history.append(fresh)
                comparison=compare(previous,fresh,domain.areas,check,new_witnesses=fixed._new_witnesses(current,fresh,seeds,target))
                gain=(comparison['retained_subset'] and comparison['new_scan_witness_weight_units']>0
                      and comparison['new_scan_witness_gain_m2']*1e6>=budget.min_gain_mm2)
                record.update(comparison=comparison,accepted=bool(gain),status='accepted' if gain else
                    'coverage_regression' if not comparison['retained_subset'] else 'insufficient_gain')
                if gain:current=trial;previous=fresh;low=0
                else:low+=1
        if low>=budget.low_gain_rounds:report['stop_reason']='low_gain_limit';break
        if any(meter.counts[k]>=getattr(budget,'max_coverage_'+k) for k in meter.counts):report['stop_reason']='coverage_work_budget';break
    if frontiers is not None and len(frontiers)!=len(history)-1:raise ValueError('变宽补充多出未使用的停止边界')
    check();report.update(edge_checks=meter.edge_checks,width_edge_checks=width_checks,coverage_work=meter.counts,
        strict_support=dict(face_checks=meter.support.face_checks,filter_comparisons=meter.support.filter_comparisons,
            construction_fast_paths=meter.support.fast_paths,support_queries=meter.support.queries,
            interval_queries=meter.support.interval_queries,aabb_rejected_faces=meter.support.aabb_rejected_faces,
            scope='exact_constructed_numeric_surface_only',preserved_base_anchors=True))
    current=make(base,fixed._chains(current),context,report,current.gap_report['variable_width']['records'],strict_base=strict_base)
    if len(canonical_json_bytes(current.to_dict()))>64*1024*1024:raise ValueError('变宽补充路线超过 64 MiB 预算')
    final=fixed.rebind_coverage(previous,current,target)
    return current,target,fixed.RefinedCertificate(final.metadata,final.arrays,tuple(history))


def validate(trajectory,mesh=None,region=None,check=lambda:None):
    from femto.planning.refinement_validation import semantic_payload,_append_record
    report=trajectory.gap_report.get('refinement') or {}
    if (report.get('schema')!=SCHEMA or report.get('width_rebinding')!='every_trial_all_scan_and_connector_edges'
            or report.get('failure_is_proof_of_infeasibility') is not False or report.get('execution_ready') is not False
            or report.get('physical_accuracy_verified') is not False or report.get('registration_uncertainty_mm',0) is not None):
        raise ValueError('变宽补充证据语义无效')
    budget=fixed.RefinementBudget(**report['budget']).checked()
    base=SurfaceTrajectory.from_dict(report['base_trajectory'])
    if base.planner['algorithm_version']!=variable.ALGORITHM:raise ValueError('变宽补充原版本不一致')
    validate_trajectory(base,mesh=mesh,region=region,check=check)
    coverage.validate_metadata(report['base_coverage'])
    context=context_for(base,SimpleNamespace(metadata=report['base_coverage']),budget,report['candidate_policy'],report['coverage_policy'])
    if trajectory.planning_context!=context:raise ValueError('变宽补充改变了原目标或模型修订')
    if trajectory.planner['parameters'].get('coverage_refinement')!=context['revision']['inputs']['algorithms']['refinement']:
        raise ValueError('变宽补充策略与修订不一致')
    rounds=report['rounds'];chains=fixed._chains(base);accepted=0
    if not isinstance(rounds,list) or len(rounds)>budget.max_rounds:raise ValueError('变宽补充轮数超过预算')
    for index,record in enumerate(rounds):
        check()
        if record['index']!=index or type(record['accepted']) is not bool or len(record['candidates'])>budget.candidates_per_round:
            raise ValueError('变宽补充候选或轮次不一致')
        for candidate in record['candidates']:
            if candidate['status']=='checked':strict_refinement.validate_candidate(candidate)
        if record['accepted']:
            gain=record['comparison']
            if (record['status']!='accepted' or gain.get('retained_subset') is not True or gain['lost_weight_units']!=0
                    or gain['new_scan_witness_weight_units']<=0 or gain['new_scan_witness_gain_m2']*1e6<budget.min_gain_mm2):
                raise ValueError('变宽补充缺少保留原集合的新增扫描见证')
            for candidate in record['candidates']:
                if candidate['status']=='checked':_append_record(chains,candidate)
            accepted+=1
    if not accepted:raise ValueError('变宽补充没有可接纳增益，必须保留原方案')
    rows=trajectory.gap_report['variable_width']['records']
    expected=bounds(trajectory,base.planner['parameters']['variable_footprint'],check)
    if len(rows)!=len(expected):raise ValueError('变宽补充逐边宽度记录缺失')
    for row,bound in zip(rows,expected):
        if (set(row)!=set(bound)|{'accepted','reason','residual_margin_mm','quality_diagnostic','support_check'}
                or canonical_json_bytes({k:row.get(k) for k in bound})!=canonical_json_bytes(bound)
                or row['accepted'] is not True or row['reason']!='accepted' or row['support_check'].get('status')!='passed'
                or row['quality_diagnostic'].get('accepted') is not True):raise ValueError('变宽补充区间、宽度或逐边证据不一致')
    if semantic_payload(make(base,chains,context,report,rows))!=semantic_payload(trajectory):
        raise ValueError('变宽补充改变了原路线或新增边身份')
    if (type(report['edge_checks']) is not int or not 0<=report['edge_checks']<=budget.max_edge_checks
            or type(report['width_edge_checks']) is not int or not 0<=report['width_edge_checks']<=report['edge_checks']
            or report['max_total_length_m']!=base.statistics['total_length_m']+budget.max_added_length_m
            or trajectory.statistics['total_length_m']>report['max_total_length_m']
            or trajectory.statistics['point_count']>budget.max_output_points
            or len(canonical_json_bytes(trajectory.to_dict()))>64*1024*1024):raise ValueError('变宽补充超过累计预算')
    support_budget=strict_path.SupportBudget(**base.planner['parameters']['surface_support']['budget']).checked()
    support=report['strict_support']
    for key,cap in (('face_checks',support_budget.max_face_checks),('filter_comparisons',support_budget.max_filter_comparisons)):
        if type(support.get(key)) is not int or not 0<=support[key]<=cap:raise ValueError('变宽补充支持预算不一致')
    if support.get('scope')!='exact_constructed_numeric_surface_only' or support.get('preserved_base_anchors') is not True:
        raise ValueError('变宽补充未保留原锚点')
    if mesh is not None:
        from femto.planning.effective_domain import validate as validate_domain
        region=region or base.planning_context['revision']['inputs']['selection']
        params=trajectory.planner['parameters']
        allowed=validate_domain(params['effective_domain'],params['quality_evidence'],params['effective_width_mm'],mesh,region,check)
        meter=fixed._Meter(budget);meter.support=strict_path.engine_for(mesh,allowed,support_budget,check,region=region)
        if canonical_json_bytes(checked_rows(trajectory,mesh,allowed,meter,check))!=canonical_json_bytes(rows):
            raise ValueError('变宽补充逐边证据不能从当前几何重现')


def summary_text(trajectory):
    values=trajectory.to_dict() if isinstance(trajectory,SurfaceTrajectory) else trajectory
    report=values['gap_report']['refinement'];accepted=[r for r in report['rounds'] if r['accepted']]
    gain=sum(r['comparison']['new_scan_witness_gain_m2'] for r in accepted)*1e6
    return f'变宽补充规划：采用 {len(accepted)} 轮，新增扫描见证名义面积 {gain:.3f} mm²；原确认范围无损，全部边已重绑宽度并核验。'
