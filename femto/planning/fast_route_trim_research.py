"""One-call, proof-checked relocation and protected end-trim research.

Freshly validated state is reused only within this synchronous operation.
The result is not an installed plan, clinical coverage or robot trajectory.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

from femto.planning import (auto_supplement_route_research, continuous_quality, coverage_certificate as coverage,
    coverage_partition, coverage_workflow, effective_domain, fraction_footprint, planning_revision,
    supplement_route_validation, surface_workspace, trim_subset_research, verified_relocation_research)
from femto.planning.coverage_domain import from_plan,scan_endpoints
from femto.planning.surface_trajectory import ScanSegment,SurfaceTrajectory

VERSION='research-fast-relocation-protected-trim-v1'


def build(model,region,plan,old_proof,*,relocation_attempts=32,max_trim_candidates=2,max_proofs=3,
          prior_proof_budget=None,final_proof_budget=None,check=lambda:None,budget=200_000):
    """Recheck the source, optimize scan visits, then try protected trims.

    The final proof and the preceding route proof are both returned for
    independent replay. No route or evidence is installed into the model.
    """
    if type(relocation_attempts) is not int or not 1<=relocation_attempts<=verified_relocation_research.MAX_TRIALS:
        raise ValueError('换序研究预算无效')
    trim_subset_research._check_budget(max_trim_candidates,max_proofs,True,budget)
    check();source_snapshot=planning_revision.digest(plan)
    domain,seeds,footprint,_=from_plan(model,region,plan,check)
    if footprint.get('schema')!='femto-nominal-fixed-ball-v1':raise ValueError('组合研究需要固定名义作用宽度')
    base=SurfaceTrajectory.from_dict(plan['surface_trajectory'],check=check)
    binding=coverage_workflow.route_binding(base)
    coverage.validate(old_proof,domain,seeds,footprint,route_binding=binding,check=check)
    reconnection=base.gap_report.get('reconnection') or {}
    segments=[ScanSegment.from_dict(item) for item in reconnection.get('base_items',[]) if item['kind']=='scan_segment']
    if not segments:raise ValueError('组合研究缺少原扫描段')
    visits=auto_supplement_route_research._source_visits(base,segments);links=reconnection.get('connections')
    params=base.planner['parameters'];width=params['effective_width_mm'];evidence=params['quality_evidence']
    mesh=surface_workspace.canonical_region_mesh(model.store.mesh,region)
    allowed=effective_domain.validate(params['effective_domain'],evidence,width,mesh,region,check)
    class Graph(continuous_quality.LazyVerifiedSurfaceGraph):quarter_portals=True
    graph=Graph(mesh,region,evidence,width,allowed,replace(base,ordered_items=segments),check)
    limits=dict(detours=base.statistics['connected_detours'],connector_length_m=base.statistics['connector_length_m'],
                revisit_connector_m=base.statistics['revisit_connector_length_m'])
    _,candidate_links,_,decision=verified_relocation_research.propose(graph,segments,visits,links,
        params['row_spacing_mm']/1000,motion_limits=limits,attempts=relocation_attempts,check=check,budget=budget)
    verified_relocation_research.validate_record(decision,segments,visits,links,candidate_links,
        params['row_spacing_mm']/1000,motion_limits=limits,attempts=relocation_attempts,check=check,budget=budget)
    candidate_visits=decision['visits']
    items,stats,blocks,geometry=supplement_route_validation.reconstruct_relocated_and_validate(
        base,segments,candidate_visits,candidate_links,mesh,region,check=check,budget=budget)
    if not trim_subset_research._motion_ok(stats,base.statistics):raise ValueError('组合研究换序不满足原路线运动上限')
    prior=replace(base,ordered_items=items,statistics=stats,blocks=blocks)
    nominal=fraction_footprint.nominal(footprint['effective_width_mm'])
    prior_seeds=scan_endpoints(prior,domain.ledger.arrays['region_faces'])
    prior_route=trim_subset_research._route(domain,binding,geometry)
    algorithm=(coverage.RESEARCH_PAGED_FRACTION_REPROOF_ALGORITHM if coverage.checked_page_binding(domain.binding)
               else coverage.RESEARCH_FRACTION_REPROOF_ALGORITHM)
    prior_proof=coverage.evaluate(domain,prior_seeds,nominal,route_binding=prior_route,guide=old_proof,
                                  algorithm=algorithm,budget=prior_proof_budget,check=check)
    coverage.validate(prior_proof,domain,prior_seeds,nominal,route_binding=prior_route,guide=old_proof,check=check)
    retained=coverage_partition.compare_union(old_proof,[prior_proof],domain.areas,check)
    relocation=dict(decision=decision,geometry=geometry,statistics=stats,baseline_comparison=retained,
                    recipe=dict(visits=deepcopy(candidate_visits),links=deepcopy(candidate_links)),
                    coverage_summary=prior_proof.summary)
    result=dict(schema=VERSION,relocation=relocation,trim=None,installed_route=False,installed_certificate=False,
                full_target_coverage_claim=False,execution_ready=False,physical_accuracy_verified=False)
    if not retained['retained_subset']:
        if planning_revision.digest(plan)!=source_snapshot:raise ValueError('组合研究期间源路线发生变化')
        relocation.pop('recipe',None)
        snapshot=planning_revision.digest(dict(plan=plan,visits=candidate_visits,links=candidate_links))
        trim_subset_research._check_end(model,region,plan,candidate_visits,candidate_links,domain,
                                        (old_proof,prior_proof),snapshot,check)
        result['reason']='relocation_retention_not_confirmed';check();return result,None,None
    # This state was constructed and both source proofs were replayed above in
    # the same call. The private helper rechecks inputs/assets before returning.
    state=dict(domain=domain,base=base,binding=binding,segments=segments,
               visits=[tuple(v) for v in candidate_visits],links=candidate_links,mesh=mesh,
               prior=prior,prior_stats=stats,footprint=nominal)
    snapshot=planning_revision.digest(dict(plan=plan,visits=candidate_visits,links=candidate_links))
    trim_result,final_proof=trim_subset_research._build_prepared(model,region,plan,old_proof,candidate_visits,
        candidate_links,prior_proof,state,snapshot,max_candidates=max_trim_candidates,max_proofs=max_proofs,
        proof_budget=final_proof_budget,motion_prefilter=True,check=check,budget=budget)
    if planning_revision.digest(plan)!=source_snapshot:raise ValueError('组合研究期间源路线发生变化')
    result.update(reason='trim_candidate_validated' if trim_result['adopted'] else 'no_trim_candidate',trim=trim_result)
    check();return result,prior_proof,final_proof
