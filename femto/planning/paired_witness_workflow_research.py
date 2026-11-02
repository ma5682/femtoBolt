"""Bounded two-point restoration from independently replayed residual cells.

Detached research results only. Neither pairing nor a favorable distance is
acceptance; the combined geometry, motion and final proof must all pass.
"""
from copy import deepcopy
from itertools import combinations
from femto.planning import (auto_row_shift_screen_research as screening,
    coverage_certificate as coverage,coverage_partition,coverage_workflow,
    fraction_priority,planning_revision,scan_fraction_seeds,
    witness_bend_research as proposals,witness_bend_workflow_research as bend,
    witness_edge_direct_replay_research as direct)
from femto.planning.coverage_domain import scan_endpoints

VERSION='research-paired-witness-workflow-v1'
MAX_PAIRS=4
MAX_PROOFS=4


def _pairs(values):
    """Prioritize more residual cells, then smaller movement; no acceptance."""
    pairs=[list(pair) for pair in combinations(values,2)
           if pair[0]['seed']!=pair[1]['seed']
           and (pair[0]['segment'],pair[0]['point'])!=(pair[1]['segment'],pair[1]['point'])]
    return sorted(pairs,key=lambda pair:(-len(set(pair[0]['pending_leaves'])|set(pair[1]['pending_leaves'])),
        sum(member['distance_m'] for member in pair),
        tuple((member['seed'],member['segment'],member['point']) for member in pair)))


def build(model,region,old_plan,old_proof,base_plan,prior_requests,prior_proof,*,
          max_pairs=2,max_proofs=2,proof_budget=None,check=lambda:None,progress=lambda text:None,
          _old_proof_session=None):
    if (type(max_pairs) is not int or not 1<=max_pairs<=MAX_PAIRS
            or type(max_proofs) is not int or not 1<=max_proofs<=MAX_PROOFS):
        raise ValueError('联合见证点候选或证明预算无效')
    check();snapshot=bend._snapshot(old_plan,base_plan,prior_requests,old_proof,prior_proof)
    region_snapshot=planning_revision.digest(region)
    state=bend._prepare(model,region,old_plan,old_proof,base_plan,prior_requests,prior_proof,check,
                        _old_proof_session=_old_proof_session)
    domain=state['domain'];accepted=None;requests=deepcopy(prior_requests);evaluated=0
    report=dict(schema=VERSION,adopted=False,reason='no_qualifying_pair',trials=[],
        prior_comparison=state['previous'],installed_route=False,installed_certificate=False,
        full_target_coverage_claim=False,execution_ready=False,physical_accuracy_verified=False)
    algorithm=(coverage.RESEARCH_PAGED_FRACTION_REPROOF_ALGORITHM if coverage.checked_page_binding(domain.binding)
               else coverage.RESEARCH_FRACTION_REPROOF_ALGORITHM)
    def compare(proof,seeds,route):
        coverage.validate(proof,domain,seeds,state['nominal'],route_binding=route,guide=old_proof,check=check)
        return coverage_partition.compare_union(old_proof,[proof],domain.areas,check,include_pending_leaves=True)
    # One refinement of evidence positions before changing any route geometry.
    if not state['previous']['retained_subset']:
        progress('补充剩余单元附近的沿线核验位置')
        extra,_,_=fraction_priority.propose_batched(state['base'],domain.ledger,old_proof,
            state['previous']['pending_old_leaf_indices'],nearest_edges_per_leaf=3,radius_m=.004,denominator=16,check=check)
        merged=sorted(set(tuple(r) for r in requests)|set(extra))
        if len(merged)>len(requests):
            requests=[list(r) for r in merged]
            positions,_=scan_fraction_seeds.propose(state['base'],domain.ledger.arrays['region_faces'],requests,check=check)
            seeds=scan_endpoints(state['base'],domain.ledger.arrays['region_faces'])+positions
            route=coverage_workflow.route_binding(state['base'])
            refined=coverage.evaluate(domain,seeds,state['nominal'],route_binding=route,guide=old_proof,
                                      algorithm=algorithm,budget=proof_budget,check=check)
            state['previous']=compare(refined,seeds,route)
            report['position_refinement']=state['previous']
            if state['previous']['retained_subset']:
                accepted=refined
                report.update(adopted=True,reason='fraction_proof_retained',route_unchanged=True,
                              fraction_requests=requests,coverage_summary=refined.summary)
    if accepted is None and not state['previous']['retained_subset']:
        candidates,proposal_report=proposals.propose(state['old'],state['segments'],
            state['old_domain'].ledger.arrays['region_faces'],old_proof,state['previous']['pending_old_leaf_indices'],
            representative_faces=state['old_domain'].representative_faces,check=check)
        pairs=_pairs(candidates);report.update(proposals=proposal_report,pair_count=len(pairs),
                                             pair_budget_exhausted=len(pairs)>max_pairs)
        for index,pair in enumerate(pairs[:max_pairs]):
            check()
            if evaluated>=max_proofs:report['reason']='proof_budget_exhausted';break
            progress(f'核验联合见证点 {index+1}/{min(len(pairs),max_pairs)}')
            combined=proposals.combine_replacements(state['segments'],pair,check=check)
            entry=dict(members=deepcopy(combined['members']))
            try:final,statistics,geometry,density=bend._candidate(state,region,combined,check,200_000)
            except ValueError as error:
                check();entry.update(reason='geometry_rejected',detail=str(error));report['trials'].append(entry);continue
            if (not bend._acceptable(state,statistics,density)
                    or not screening._acceptable(state['old'],final,state['old_density'],density)):
                entry.update(reason='motion_or_density_rejected',statistics=statistics,density=density)
                report['trials'].append(entry);continue
            current,seeds,route,transfer=bend._proof_inputs(state,final,geometry,requests,check)
            for attempt in range(2):
                if evaluated>=max_proofs:report['reason']='proof_budget_exhausted';break
                evaluated+=1
                proof=coverage.evaluate(domain,seeds,state['nominal'],route_binding=route,guide=old_proof,
                                        algorithm=algorithm,budget=proof_budget,check=check)
                comparison=compare(proof,seeds,route)
                report['trials'].append(dict(entry,attempt=attempt,comparison=comparison,
                    reason='accepted' if comparison['retained_subset'] else 'not_reconfirmed'))
                if comparison['retained_subset']:
                    accepted=proof
                    report.update(adopted=True,reason='research_candidate_validated',route_unchanged=False,
                        statistics=statistics,geometry=geometry,density=density,transfer=transfer,
                        comparison=comparison,coverage_summary=proof.summary,
                        direct_recipe=direct.pair_recipe(old_plan,old_proof,base_plan,combined['members'],[list(r) for r in current]))
                    break
                if attempt:break
                extra,_,_=fraction_priority.propose_batched(final,domain.ledger,old_proof,
                    comparison['pending_old_leaf_indices'],nearest_edges_per_leaf=3,radius_m=.004,denominator=16,check=check)
                merged=sorted(set(tuple(r) for r in current)|set(extra))
                if len(merged)==len(current):break
                current=merged;positions,_=scan_fraction_seeds.propose(final,domain.ledger.arrays['region_faces'],current,check=check)
                seeds=scan_endpoints(final,domain.ledger.arrays['region_faces'])+positions
            if accepted is not None:break
    elif accepted is None:
        # The supplied prior was already independently checked in _prepare.
        accepted=prior_proof
        report.update(adopted=True,reason='fraction_proof_retained',route_unchanged=True,
                      fraction_requests=requests,coverage_summary=prior_proof.summary)
    report['combined_proofs_evaluated']=evaluated
    bend._finish(model,region,old_plan,base_plan,old_proof,prior_proof,accepted,state,snapshot,prior_requests,check)
    if _old_proof_session is not None:
        _old_proof_session.require(model,region,old_plan,old_proof,state['old_domain'],_old_proof_session.seeds,
                                  state['width'],_old_proof_session.route,check)
    check()
    if planning_revision.digest(region)!=region_snapshot:raise ValueError('联合见证点期间选区发生变化')
    return report,accepted
