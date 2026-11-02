"""Build a witness-bend research candidate with one final coverage proof.

Unmatched witnesses only propose work. The accepted output still requires a
fresh checked route and a single independently replayed final proof retaining
the entire old confirmed set. No route or certificate is installed here.
"""
from femto.planning import (auto_supplement_route_research,coverage_certificate as coverage,
    coverage_partition,coverage_workflow,footprint_model,fraction_footprint,fraction_priority,
    planning_revision,row_proximity_research,surface_workspace,target_surface,witness_bend_research,
    witness_bend_workflow_research as bend,witness_edge_direct_replay_research as direct)
from femto.planning.coverage_domain import from_plan
from femto.planning.coverage_retention_union import _matched_witnesses,_same_target
from femto.planning.surface_trajectory import ScanSegment,SurfaceTrajectory

VERSION='research-single-final-witness-v1'
MAX_PROOFS=4


def build(model,region,old_plan,old_certificate,base_plan,*,max_proofs=2,proof_budget=None,
          check=lambda:None,_old_proof_session=None):
    if type(max_proofs) is not int or not 1<=max_proofs<=MAX_PROOFS:raise ValueError('单份最终证据预算无效')
    check();snapshot=planning_revision.digest(dict(old=old_plan,base=base_plan,region=region,proof=old_certificate.metadata))
    old_domain,old_seeds,old_width,_=from_plan(model,region,old_plan,check)
    old=SurfaceTrajectory.from_dict(old_plan['surface_trajectory'],check=check);old_route=coverage_workflow.route_binding(old)
    if _old_proof_session is None:
        coverage.validate(old_certificate,old_domain,old_seeds,old_width,route_binding=old_route,check=check)
    else:
        from femto.planning.old_coverage_session_research import _OldCoverageSession
        if type(_old_proof_session) is not _OldCoverageSession:raise ValueError('原证据复用上下文无效')
        _old_proof_session.require(model,region,old_plan,old_certificate,old_domain,old_seeds,old_width,old_route,check)
    domain,endpoints,width,_=from_plan(model,region,base_plan,check)
    base=SurfaceTrajectory.from_dict(base_plan['surface_trajectory'],check=check);_same_target(old_domain,domain,check)
    if old_width!=width or width.get('schema')!=footprint_model.SCHEMA:raise ValueError('单份最终证据需要相同固定名义作用宽度')
    _,_,missing=_matched_witnesses(old_certificate,domain,endpoints,check)
    reconnect=base.gap_report.get('reconnection') or {}
    segments=[ScanSegment.from_dict(item) for item in reconnect.get('base_items',[]) if item['kind']=='scan_segment']
    if not segments or not isinstance(reconnect.get('connections'),list):raise ValueError('单份最终证据缺少扫描与连接记录')
    over_budget=len(missing)>witness_bend_research.MAX_PENDING
    if over_budget:
        requests=[];ranking=None;proposals=[];proposal_report=None
    else:
        requests,_,ranking=fraction_priority.propose_batched(base,old_domain.ledger,old_certificate,missing,check=check)
        proposals,proposal_report=witness_bend_research.propose(old,segments,old_domain.ledger.arrays['region_faces'],
            old_certificate,missing,representative_faces=old_domain.representative_faces,edge_insert=True,check=check)
    state=dict(old_domain=old_domain,domain=domain,old=old,base=base,segments=segments,
        visits=auto_supplement_route_research._source_visits(base,segments),links=reconnect['connections'],
        width=width,nominal=fraction_footprint.nominal(width['effective_width_mm']),
        mesh=surface_workspace.canonical_region_mesh(model.store.mesh,region),
        old_density=row_proximity_research.measure(old,check=check),base_density=row_proximity_research.measure(base,check=check))
    algorithm=(coverage.RESEARCH_PAGED_FRACTION_REPROOF_ALGORITHM if coverage.checked_page_binding(domain.binding)
               else coverage.RESEARCH_FRACTION_REPROOF_ALGORITHM)
    report=dict(schema=VERSION,adopted=False,reason='witness_budget_exhausted' if over_budget else 'no_qualifying_candidate',unmatched_old_leaves=len(missing),
        fraction_ranking=ranking,proposal_report=proposal_report,trials=[],installed_route=False,
        installed_certificate=False,full_target_coverage_claim=False,execution_ready=False,physical_accuracy_verified=False)
    accepted=None;evaluated=0
    for proposal in proposals:
        check()
        if evaluated>=max_proofs:report['reason']='proof_budget_exhausted';break
        identity={k:proposal[k] for k in ('seed','segment','point','edge','distance_m','pending_leaves')}
        try:final,statistics,geometry,density=bend._candidate(state,region,proposal,check,200000)
        except ValueError as error:
            check();report['trials'].append(dict(proposal=identity,reason='geometry_rejected',detail=str(error)));continue
        if not bend._acceptable(state,statistics,density):
            report['trials'].append(dict(proposal=identity,reason='motion_or_density_rejected'));continue
        if evaluated>=max_proofs:report['reason']='proof_budget_exhausted';break
        evaluated+=1
        final_requests,seeds,route,transfer=bend._proof_inputs(state,final,geometry,requests,check)
        certificate=coverage.evaluate(domain,seeds,state['nominal'],route_binding=route,guide=old_certificate,
                                      algorithm=algorithm,budget=proof_budget,check=check)
        coverage.validate(certificate,domain,seeds,state['nominal'],route_binding=route,guide=old_certificate,check=check)
        comparison=coverage_partition.compare_union(old_certificate,[certificate],domain.areas,check,include_pending_leaves=True)
        if comparison['retained_subset']:
            replayed,_=witness_bend_research.propose(old,segments,old_domain.ledger.arrays['region_faces'],
                old_certificate,identity['pending_leaves'],representative_faces=old_domain.representative_faces,
                edge_insert=True,check=check)
            matches=[item for item in replayed if all(item[key]==identity[key] for key in ('seed','segment','point','edge'))]
            expected=planning_revision.digest([item.to_dict() for item in proposal['segments']])
            if len(matches)!=1 or planning_revision.digest([item.to_dict() for item in matches[0]['segments']])!=expected:
                report['trials'].append(dict(proposal=identity,reason='recipe_not_rederivable',comparison=comparison));continue
        report['trials'].append(dict(proposal=identity,reason='accepted' if comparison['retained_subset'] else 'not_reconfirmed',comparison=comparison))
        if comparison['retained_subset']:
            accepted=certificate
            report.update(adopted=True,reason='research_candidate_validated',geometry=geometry,statistics=statistics,
                density=density,comparison=comparison,coverage_summary=certificate.summary,transfer=transfer,
                direct_recipe=direct.recipe(old_plan,old_certificate,base_plan,identity,[list(r) for r in final_requests]))
            break
    for plan,checked in ((old_plan,old_domain),(base_plan,domain)):
        planning_revision.require_current(model,region,plan['treatment_head'],plan['planning_context']['revision'])
        params=plan['surface_trajectory']['planner']['parameters']
        target_surface.validate(checked.ledger,plan['planning_context']['revision'],
            evidence=params['quality_evidence'],domain=params['effective_domain'],check=check,reproduce=False)
    for item in (old_certificate,*([accepted] if accepted is not None else [])):coverage._checked_assets(item,check)
    check()
    if snapshot!=planning_revision.digest(dict(old=old_plan,base=base_plan,region=region,proof=old_certificate.metadata)):
        raise ValueError('单份最终证据研究期间输入发生变化')
    return report,accepted
