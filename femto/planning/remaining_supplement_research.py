"""Derive remaining old witnesses after a complete supplement proof replay."""
from __future__ import annotations

from femto.planning import (coverage_certificate as coverage,planning_revision,
    supplement_coverage_research,supplement_scan_segments,target_surface,witness_scan_supplement)

VERSION='research-replayed-remaining-supplement-proposal-v1'


def propose(model,region,old_plan,old_certificate,base_plan,edges,visits,links,requests,certificate,*,
            gap_m=.008,check=lambda:None,budget=200_000):
    """Replay the candidate, then return additional original-edge proposals."""
    def snapshot():
        return planning_revision.digest(dict(old=old_plan,base=base_plan,edges=edges,visits=visits,
                                             links=links,requests=requests))
    check();before=snapshot()
    report,state=supplement_coverage_research._replay_with_state(model,region,old_plan,old_certificate,
        base_plan,edges,visits,links,requests,certificate,check=check,budget=budget)
    domain=state['old_domain']
    added,selection=witness_scan_supplement.propose(state['old_trajectory'],domain.ledger.arrays['region_faces'],
        old_certificate,domain.areas,report['comparison']['pending_old_leaf_indices'],
        representative_faces=domain.representative_faces,gap_m=gap_m,check=check)
    combined=sorted(set(map(tuple,edges))|set(added))
    if len(combined)>supplement_scan_segments.MAX_EDGES:raise ValueError('补线候选超过预算')
    if snapshot()!=before:raise ValueError('补线候选计算期间输入发生变化')
    for plan,checked_domain in ((old_plan,domain),(base_plan,state['domain'])):
        planning_revision.require_current(model,region,plan['treatment_head'],plan['planning_context']['revision'])
        params=plan['surface_trajectory']['planner']['parameters']
        target_surface.validate(checked_domain.ledger,plan['planning_context']['revision'],
            evidence=params['quality_evidence'],domain=params['effective_domain'],check=check,reproduce=False)
    for proof in (old_certificate,certificate):coverage._checked_assets(proof,check)
    check()
    return dict(schema=VERSION,prior=report,selection=selection,added_edges=[list(e) for e in added],
        combined_edges=[list(e) for e in combined],installed_route=False,installed_certificate=False,
        full_target_coverage_claim=False,execution_ready=False,physical_accuracy_verified=False)
