"""Build and reprove a bounded automatically selected supplement research route.

The returned candidate is not an installed plan, treatment coverage or robot
trajectory. A caller must inspect retention and independently replay it.
"""
from __future__ import annotations

from dataclasses import replace
from femto.planning import (auto_fraction_supplement_research,balanced_visit_improvement,
    continuous_quality,coverage_certificate as coverage,coverage_partition,coverage_workflow,
    effective_domain,fraction_footprint,fraction_transfer,planning_revision,scan_visit_order,
    supplement_insertion_research,supplement_route_validation,supplement_scan_segments,
    supplement_two_opt_research,surface_workspace,target_surface,witness_scan_supplement)
from femto.planning.coverage_domain import scan_endpoints
from femto.planning.surface_trajectory import ScanSegment

VERSION='research-auto-supplement-route-proposal-v1'


def _source_visits(base,segments):
    ordering=(base.gap_report.get('reconnection') or {}).get('ordering') or {}
    improvement=ordering.get('improvement') or {}
    visits=improvement.get('visits') if improvement.get('adopted') is True else ordering.get('visits')
    for key in ('preserved_detour','joint_reorder','relocation','relocation_deep','relocation_floor','relocation_unified'):
        decision=ordering.get(key) or {}
        if decision.get('adopted') is True:visits=decision.get('visits')
    return scan_visit_order.checked_visits(segments,visits)


def build(model,region,old_plan,old_certificate,base_plan,normal_certificate,*,
          selection_budget=None,proof_budget=None,gap_m=.008,correction_rounds=1,check=lambda:None):
    """Generate checked candidates, with bounded old-witness correction."""
    if type(correction_rounds) is not int or not 0<=correction_rounds<=3:
        raise ValueError('自动补线校正轮数无效或超过预算')
    check();before=(planning_revision.digest(old_plan),planning_revision.digest(base_plan))
    selected,selection_proof,state=auto_fraction_supplement_research._propose_with_state(
        model,region,old_plan,old_certificate,base_plan,normal_certificate,
        budget=selection_budget,gap_m=gap_m,check=check)
    old,base=state['old'],state['base'];domain=state['domain'];params=base.planner['parameters']
    reconnection=base.gap_report.get('reconnection') or {}
    original=[ScanSegment.from_dict(item) for item in reconnection.get('base_items',[]) if item['kind']=='scan_segment']
    if not original:raise ValueError('自动补线缺少原扫描段')
    visits=_source_visits(base,original)
    offset=max(point['row_index'] for segment in original for point in segment.points)+1
    mesh=surface_workspace.canonical_region_mesh(model.store.mesh,region)
    record=params['quality_evidence'];width=params['effective_width_mm'];spacing=params['row_spacing_mm']/1000
    allowed=effective_domain.validate(params['effective_domain'],record,width,mesh,region,check)
    class Graph(continuous_quality.LazyVerifiedSurfaceGraph):quarter_portals=True
    footprint=fraction_footprint.nominal(state['current_model']['effective_width_mm'])
    algorithm=(coverage.RESEARCH_PAGED_FRACTION_REPROOF_ALGORITHM if coverage.checked_page_binding(domain.binding)
               else coverage.RESEARCH_FRACTION_REPROOF_ALGORITHM)
    edges=[list(edge) for edge in selected['selected_edges']];corrections=[]
    for round_index in range(correction_rounds+1):
        check()
        extra,materialized=supplement_scan_segments.materialize(old,edges,row_offset=offset,check=check)
        segments=sorted(original+extra,key=lambda s:s.block_index)
        graph=Graph(mesh,region,record,width,allowed,replace(base,ordered_items=segments),check)
        chains,links,_,insertion=supplement_insertion_research.propose(graph,segments,original,extra,visits,spacing,
            check=check,budget=continuous_quality.BUDGET)
        measured=balanced_visit_improvement._metrics(chains,spacing,check)
        _,links,_,improvement=supplement_two_opt_research.propose(graph,segments,insertion['visits'],links,spacing,
            max_detours=measured['detours'],max_revisit_m=measured['revisit_connector_m'],check=check,
            budget=continuous_quality.BUDGET)
        items,statistics,blocks,geometry=supplement_route_validation.reconstruct_and_validate(base,segments,
            improvement['visits'],links,mesh,region,check=check,budget=continuous_quality.BUDGET)
        final=replace(base,ordered_items=items,statistics=statistics,blocks=blocks)
        requests,positions,transfer=fraction_transfer.transfer(base,final,domain.ledger.arrays['region_faces'],
            selected['requests'],check=check)
        seeds=scan_endpoints(final,domain.ledger.arrays['region_faces'])+positions
        route=dict(schema='research-checked-supplement-route-v1',computation_sha256=domain.revision['sha256'],
            source_route=coverage_workflow.route_binding(base),items_sha256=planning_revision.digest([x.to_dict() for x in items]),
            execution_ready=False)
        final_proof=coverage.evaluate(domain,seeds,footprint,route_binding=route,guide=old_certificate,
            algorithm=algorithm,budget=proof_budget,check=check)
        coverage.validate(final_proof,domain,seeds,footprint,route_binding=route,guide=old_certificate,check=check)
        comparison=coverage_partition.compare_union(old_certificate,[final_proof],domain.areas,check,include_pending_leaves=True)
        pending=comparison['pending_old_leaf_indices']
        if not pending or round_index==correction_rounds:break
        old_domain=state['old_domain']
        added,correction=witness_scan_supplement.propose(old,old_domain.ledger.arrays['region_faces'],old_certificate,
            old_domain.areas,pending,representative_faces=old_domain.representative_faces,gap_m=gap_m,check=check)
        combined=sorted(set(map(tuple,edges))|set(added))
        if len(combined)>supplement_scan_segments.MAX_EDGES:raise ValueError('自动补线候选超过预算')
        corrections.append(dict(selection=correction,added_edges=[list(edge) for edge in added],
                                pending_before=len(pending)))
        if len(combined)==len(edges):break
        edges=[list(edge) for edge in combined]
    if before!=(planning_revision.digest(old_plan),planning_revision.digest(base_plan)):
        raise ValueError('自动补线路线生成期间输入发生变化')
    for plan,checked_domain in ((old_plan,state['old_domain']),(base_plan,domain)):
        planning_revision.require_current(model,region,plan['treatment_head'],plan['planning_context']['revision'])
        inputs=plan['surface_trajectory']['planner']['parameters']
        target_surface.validate(checked_domain.ledger,plan['planning_context']['revision'],
            evidence=inputs['quality_evidence'],domain=inputs['effective_domain'],check=check,reproduce=False)
    for proof in (old_certificate,normal_certificate,selection_proof,final_proof):coverage._checked_assets(proof,check)
    check()
    return dict(schema=VERSION,selection=selected,materialized=materialized,insertion=insertion,
        improvement=improvement,geometry=geometry,statistics=statistics,transfer=transfer,
        recipe=dict(edges=edges,visits=improvement['visits'],links=links),corrections=corrections,
        requests=[list(x) for x in requests],route=route,comparison=comparison,coverage_summary=final_proof.summary,
        installed_route=False,installed_certificate=False,full_target_coverage_claim=False,
        execution_ready=False,physical_accuracy_verified=False),final_proof
