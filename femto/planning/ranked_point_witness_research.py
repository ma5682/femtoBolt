"""Bounded ranked point-replacement research with independent final evidence.

Distances rank work only. Every candidate is geometrically rebuilt and a
single final proof must independently retain the old confirmed nominal set.
Nothing here installs a route or represents equipment execution readiness.
"""
from femto.planning import (auto_supplement_route_research,coverage_certificate as coverage,
    coverage_partition,coverage_workflow,footprint_model,fraction_footprint,fraction_priority,
    planning_revision,row_proximity_research,scan_fraction_seeds,surface_workspace,target_surface,
    witness_bend_research,witness_bend_workflow_research as bend,
    witness_edge_direct_replay_research as direct,witness_leaf_rank_research)
from femto.planning.coverage_domain import from_plan,scan_endpoints
from femto.planning.coverage_retention_union import _matched_witnesses,_same_target
from femto.planning.surface_trajectory import ScanSegment,SurfaceTrajectory

VERSION='research-ranked-point-witness-v1'
MAX_RANKED_LEAVES=256
MAX_PROOFS=4


def build(model,region,old_plan,old_certificate,base_plan,*,max_ranked_leaves=16,max_proofs=2,
          proof_budget=None,check=lambda:None,progress=lambda text:None,_old_proof_session=None):
    if (type(max_ranked_leaves) is not int or not 1<=max_ranked_leaves<=MAX_RANKED_LEAVES
            or type(max_proofs) is not int or not 1<=max_proofs<=MAX_PROOFS):
        raise ValueError('见证点排序研究预算无效')
    check();snapshot=planning_revision.digest(dict(old=old_plan,base=base_plan,region=region,proof=old_certificate.metadata))
    old_domain,old_seeds,old_width,_=from_plan(model,region,old_plan,check)
    old=SurfaceTrajectory.from_dict(old_plan['surface_trajectory'],check=check)
    old_route=coverage_workflow.route_binding(old)
    if _old_proof_session is None:
        coverage.validate(old_certificate,old_domain,old_seeds,old_width,route_binding=old_route,check=check)
    else:
        from femto.planning.old_coverage_session_research import _OldCoverageSession
        if type(_old_proof_session) is not _OldCoverageSession:raise ValueError('原证据复用上下文无效')
        _old_proof_session.require(model,region,old_plan,old_certificate,old_domain,old_seeds,old_width,old_route,check)
    domain,endpoints,width,_=from_plan(model,region,base_plan,check)
    base=SurfaceTrajectory.from_dict(base_plan['surface_trajectory'],check=check);_same_target(old_domain,domain,check)
    if old_width!=width or width.get('schema')!=footprint_model.SCHEMA:raise ValueError('见证点排序需要相同固定名义作用宽度')
    _,_,missing=_matched_witnesses(old_certificate,domain,endpoints,check)
    reconnect=base.gap_report.get('reconnection') or {}
    segments=[ScanSegment.from_dict(item) for item in reconnect.get('base_items',[]) if item['kind']=='scan_segment']
    if not segments or not isinstance(reconnect.get('connections'),list):raise ValueError('见证点排序缺少扫描与连接记录')
    report=dict(schema=VERSION,adopted=False,reason='no_qualifying_candidate',unmatched_old_leaves=len(missing),
        trials=[],installed_route=False,installed_certificate=False,full_target_coverage_claim=False,
        execution_ready=False,physical_accuracy_verified=False)
    accepted=None;proposals=[];requests=[]
    if len(missing)>witness_leaf_rank_research.MAX_LEAVES:
        report['reason']='witness_budget_exhausted'
    elif len(missing):
        progress('排序原见证单元并构造扫描点候选')
        requests,positions,fraction_ranking=fraction_priority.propose_batched(base,old_domain.ledger,old_certificate,missing,check=check)
        ranking=witness_leaf_rank_research.rank(old_domain.ledger,old_certificate,missing,endpoints+positions,check=check)
        chosen=ranking['order'][:max_ranked_leaves];priority={leaf:i for i,leaf in enumerate(chosen)}
        proposals,proposal_report=witness_bend_research.propose(old,segments,old_domain.ledger.arrays['region_faces'],
            old_certificate,chosen,representative_faces=old_domain.representative_faces,edge_insert=False,check=check)
        proposals.sort(key=lambda p:(min(priority[i] for i in p['pending_leaves']),p['seed'],p['segment'],p['point']))
        report.update(fraction_ranking=fraction_ranking,leaf_ranking=ranking,proposal_report=proposal_report,
                      ranked_leaves_examined=len(chosen),ranked_leaf_budget_exhausted=len(missing)>len(chosen))
    state=dict(old_domain=old_domain,domain=domain,old=old,base=base,segments=segments,
        visits=auto_supplement_route_research._source_visits(base,segments),links=reconnect['connections'],width=width,
        nominal=fraction_footprint.nominal(width['effective_width_mm']),mesh=surface_workspace.canonical_region_mesh(model.store.mesh,region),
        old_density=row_proximity_research.measure(old,check=check),base_density=row_proximity_research.measure(base,check=check))
    algorithm=(coverage.RESEARCH_PAGED_FRACTION_REPROOF_ALGORITHM if coverage.checked_page_binding(domain.binding)
               else coverage.RESEARCH_FRACTION_REPROOF_ALGORITHM)
    evaluated=0
    for proposal in proposals:
        check()
        if evaluated>=max_proofs:report['reason']='proof_budget_exhausted';break
        identity={k:proposal[k] for k in ('seed','segment','point','distance_m','pending_leaves')}
        try:final,statistics,geometry,density=bend._candidate(state,region,proposal,check,200000)
        except ValueError as error:
            check();report['trials'].append(dict(proposal=identity,reason='geometry_rejected',detail=str(error)));continue
        if not bend._acceptable(state,statistics,density):
            report['trials'].append(dict(proposal=identity,reason='motion_or_density_rejected'));continue
        final_requests,seeds,route,transfer=bend._proof_inputs(state,final,geometry,requests,check)
        for attempt in range(2):
            if evaluated>=max_proofs:report['reason']='proof_budget_exhausted';break
            evaluated+=1
            progress(f'核验扫描点调整覆盖证据 {evaluated}/{max_proofs}')
            certificate=coverage.evaluate(domain,seeds,state['nominal'],route_binding=route,guide=old_certificate,
                                          algorithm=algorithm,budget=proof_budget,check=check)
            coverage.validate(certificate,domain,seeds,state['nominal'],route_binding=route,guide=old_certificate,check=check)
            comparison=coverage_partition.compare_union(old_certificate,[certificate],domain.areas,check,include_pending_leaves=True)
            report['trials'].append(dict(proposal=identity,attempt=attempt,comparison=comparison,
                reason='accepted' if comparison['retained_subset'] else 'not_reconfirmed'))
            if comparison['retained_subset']:
                replayed,_=witness_bend_research.propose(old,segments,old_domain.ledger.arrays['region_faces'],old_certificate,
                    identity['pending_leaves'],representative_faces=old_domain.representative_faces,edge_insert=False,check=check)
                matches=[item for item in replayed if all(item[k]==identity[k] for k in ('seed','segment','point'))]
                expected=planning_revision.digest([item.to_dict() for item in proposal['segments']])
                if len(matches)!=1 or planning_revision.digest([item.to_dict() for item in matches[0]['segments']])!=expected:
                    report['trials'][-1]['reason']='recipe_not_rederivable';break
                accepted=certificate
                report.update(adopted=True,reason='research_candidate_validated',geometry=geometry,statistics=statistics,
                    density=density,comparison=comparison,coverage_summary=certificate.summary,transfer=transfer,
                    direct_recipe=direct.point_recipe(old_plan,old_certificate,base_plan,identity,[list(r) for r in final_requests]))
                break
            if attempt or not comparison['pending_old_leaf_indices']:break
            extra,_,_=fraction_priority.propose_batched(final,old_domain.ledger,old_certificate,
                comparison['pending_old_leaf_indices'],nearest_edges_per_leaf=3,radius_m=.004,denominator=16,check=check)
            merged=sorted(set(tuple(r) for r in final_requests)|set(extra))
            if len(merged)==len(final_requests):break
            final_requests=merged
            positions,_=scan_fraction_seeds.propose(final,domain.ledger.arrays['region_faces'],final_requests,check=check)
            seeds=scan_endpoints(final,domain.ledger.arrays['region_faces'])+positions
        if accepted is not None or report['reason']=='proof_budget_exhausted':break
    for plan,checked in ((old_plan,old_domain),(base_plan,domain)):
        planning_revision.require_current(model,region,plan['treatment_head'],plan['planning_context']['revision'])
        params=plan['surface_trajectory']['planner']['parameters']
        target_surface.validate(checked.ledger,plan['planning_context']['revision'],
            evidence=params['quality_evidence'],domain=params['effective_domain'],check=check,reproduce=False)
    for item in (old_certificate,*([accepted] if accepted is not None else [])):coverage._checked_assets(item,check)
    check()
    if snapshot!=planning_revision.digest(dict(old=old_plan,base=base_plan,region=region,proof=old_certificate.metadata)):
        raise ValueError('见证点排序研究期间输入发生变化')
    return report,accepted
