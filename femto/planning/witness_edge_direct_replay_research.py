"""Re-derive a research edge candidate without replaying its search heuristic.

The old proof and final proof are still independently replayed. No prior proof
status is inherited, and the output is neither installed nor execution-ready.
"""
from __future__ import annotations

from copy import deepcopy
from femto.planning import (auto_supplement_route_research, coverage_certificate as coverage,
    coverage_partition, coverage_workflow, fraction_footprint, planning_revision,
    row_proximity_research, scan_fraction_seeds, surface_workspace, target_surface,
    witness_bend_research, witness_bend_workflow_research as bend)
from femto.planning.coverage_domain import from_plan, scan_endpoints
from femto.planning.coverage_retention_union import _same_target
from femto.planning.surface_trajectory import ScanSegment, SurfaceTrajectory

RECIPE_VERSION='research-witness-edge-direct-recipe-v1'
VERSION='research-witness-edge-direct-replay-v1'
POINT_RECIPE_VERSION='research-witness-point-direct-recipe-v1'
POINT_VERSION='research-witness-point-direct-replay-v1'
PAIR_RECIPE_VERSION='research-witness-pair-direct-recipe-v1'
PAIR_VERSION='research-witness-pair-direct-replay-v1'


def _checked_pair_members(members):
    if not isinstance(members,list) or len(members)!=2:
        raise ValueError('直接重放需要两个见证点成员')
    for member in members:
        if (not isinstance(member,dict) or set(member)!={'seed','segment','point','pending_leaves'}
                or any(type(member.get(k)) is not int or member[k]<0 for k in ('seed','segment','point'))
                or not isinstance(member['pending_leaves'],list) or not member['pending_leaves']
                or len(member['pending_leaves'])>witness_bend_research.MAX_PENDING
                or any(type(i) is not int or i<0 for i in member['pending_leaves'])
                or len(set(member['pending_leaves']))!=len(member['pending_leaves'])):
            raise ValueError('直接重放的联合见证点成员无效')
    if (members[0]['seed']==members[1]['seed']
            or (members[0]['segment'],members[0]['point'])==(members[1]['segment'],members[1]['point'])):
        raise ValueError('直接重放的联合见证点冲突')


def pair_recipe(old_plan,old_proof,base_plan,members,requests):
    if (not isinstance(members,list) or len(members)!=2
            or any(not isinstance(member,dict) or not {'seed','segment','point','pending_leaves'}<=set(member) for member in members)):
        raise ValueError('直接重放的联合见证点成员无效')
    identities=[{k:deepcopy(member[k]) for k in ('seed','segment','point','pending_leaves')} for member in members]
    _checked_pair_members(identities)
    if not isinstance(requests,list) or len(requests)>scan_fraction_seeds.MAX_REQUESTS:
        raise ValueError('直接重放的联合见证位置无效')
    return dict(version=PAIR_RECIPE_VERSION,members=identities,requests=deepcopy(requests),
        old_route=coverage_workflow.route_binding(SurfaceTrajectory.from_dict(old_plan['surface_trajectory'])),
        base_route=coverage_workflow.route_binding(SurfaceTrajectory.from_dict(base_plan['surface_trajectory'])),
        old_proof_sha256=old_proof.metadata['sha256'])


def recipe(old_plan, old_proof, base_plan, selected, requests):
    """Create a detached re-derivation request, never an acceptance decision."""
    if (not isinstance(selected,dict) or not isinstance(requests,list)
            or any(type(selected.get(k)) is not int or selected[k]<0 for k in ('seed','segment','point','edge'))
            or selected['point']!=selected['edge']+1
            or not isinstance(selected.get('pending_leaves'),list)
            or len(selected['pending_leaves'])>witness_bend_research.MAX_PENDING):
        raise ValueError('直接重放的见证提案无效')
    return dict(version=RECIPE_VERSION,seed=selected['seed'],segment=selected['segment'],
        point=selected['point'],edge=selected['edge'],pending_leaves=deepcopy(selected['pending_leaves']),
        requests=deepcopy(requests),old_route=coverage_workflow.route_binding(SurfaceTrajectory.from_dict(old_plan['surface_trajectory'])),
        base_route=coverage_workflow.route_binding(SurfaceTrajectory.from_dict(base_plan['surface_trajectory'])),
        old_proof_sha256=old_proof.metadata['sha256'])


def point_recipe(old_plan, old_proof, base_plan, selected, requests):
    """Bind a point replacement separately from the existing edge format."""
    if (not isinstance(selected,dict) or not isinstance(requests,list)
            or any(type(selected.get(k)) is not int or selected[k]<0 for k in ('seed','segment','point'))
            or not isinstance(selected.get('pending_leaves'),list)
            or len(selected['pending_leaves'])>witness_bend_research.MAX_PENDING):
        raise ValueError('直接重放的见证点提案无效')
    return dict(version=POINT_RECIPE_VERSION,seed=selected['seed'],segment=selected['segment'],point=selected['point'],
        pending_leaves=deepcopy(selected['pending_leaves']),requests=deepcopy(requests),
        old_route=coverage_workflow.route_binding(SurfaceTrajectory.from_dict(old_plan['surface_trajectory'])),
        base_route=coverage_workflow.route_binding(SurfaceTrajectory.from_dict(base_plan['surface_trajectory'])),
        old_proof_sha256=old_proof.metadata['sha256'])


def replay(model,region,old_plan,old_proof,base_plan,request,proof,*,check=lambda:None,budget=200_000,old_model=None,
           include_research_route=False,old_guide=None):
    """Replay both proofs, optionally using a separately opened old source model."""
    common={'version','seed','segment','point','pending_leaves','requests','old_route','base_route','old_proof_sha256'}
    pair=isinstance(request,dict) and request.get('version')==PAIR_RECIPE_VERSION
    point=isinstance(request,dict) and request.get('version')==POINT_RECIPE_VERSION
    keys={'version','members','requests','old_route','base_route','old_proof_sha256'} if pair else common if point else common|{'edge'}
    if (not isinstance(request,dict) or set(request)!=keys or request.get('version') not in (RECIPE_VERSION,POINT_RECIPE_VERSION,PAIR_RECIPE_VERSION)
            or (not pair and (any(type(request.get(k)) is not int or request[k]<0 for k in (('seed','segment','point') if point else ('seed','segment','point','edge')))
                or (not point and request['point']!=request['edge']+1)
                or not isinstance(request['pending_leaves'],list) or len(request['pending_leaves'])>witness_bend_research.MAX_PENDING
                or any(type(i) is not int or i<0 for i in request['pending_leaves'])
                or len(set(request['pending_leaves']))!=len(request['pending_leaves'])))
            or not isinstance(request['requests'],list) or len(request['requests'])>scan_fraction_seeds.MAX_REQUESTS
            or type(budget) is not int or not 1<=budget<=200_000 or type(include_research_route) is not bool):
        raise ValueError('直接重放的见证配方无效')
    if pair:_checked_pair_members(request['members'])
    check();snapshot=planning_revision.digest(dict(old=old_plan,base=base_plan,region=region,request=request,
                                                    old_proof=old_proof.metadata,proof=proof.metadata,
                                                    old_guide=None if old_guide is None else old_guide.metadata))
    source_model=model if old_model is None else old_model
    old_domain,old_seeds,old_width,_=from_plan(source_model,region,old_plan,check)
    old=SurfaceTrajectory.from_dict(old_plan['surface_trajectory'],check=check)
    old_route=coverage_workflow.route_binding(old)
    if old_route!=request['old_route'] or old_proof.metadata.get('sha256')!=request['old_proof_sha256']:
        raise ValueError('直接重放的原路线或证据不一致')
    coverage.validate(old_proof,old_domain,old_seeds,old_width,route_binding=old_route,check=check,guide=old_guide)
    domain,_,width,_=from_plan(model,region,base_plan,check)
    base=SurfaceTrajectory.from_dict(base_plan['surface_trajectory'],check=check)
    base_route=coverage_workflow.route_binding(base)
    if base_route!=request['base_route']:raise ValueError('直接重放的基础路线不一致')
    _same_target(old_domain,domain,check)
    if old_width!=width or width.get('schema')!='femto-nominal-fixed-ball-v1':
        raise ValueError('直接重放需要相同固定名义作用宽度')
    reconnect=base.gap_report.get('reconnection') or {}
    segments=[ScanSegment.from_dict(item) for item in reconnect.get('base_items',[]) if item['kind']=='scan_segment']
    if not segments or not isinstance(reconnect.get('connections'),list):raise ValueError('直接重放缺少扫描与连接记录')
    members=request['members'] if pair else [request];chosen=[]
    for member in members:
        proposals,_=witness_bend_research.propose(old,segments,old_domain.ledger.arrays['region_faces'],old_proof,
            member['pending_leaves'],representative_faces=old_domain.representative_faces,edge_insert=not (point or pair),check=check)
        matches=[p for p in proposals if all(p[k]==member[k] for k in (('seed','segment','point') if point or pair else ('seed','segment','point','edge')))]
        if len(matches)!=1:raise ValueError('直接重放的候选不能从原证据重建')
        chosen.append(matches[0])
    selected=witness_bend_research.combine_replacements(segments,chosen,check=check) if pair else chosen[0]
    state=dict(old_domain=old_domain,domain=domain,old=old,base=base,segments=segments,
        visits=auto_supplement_route_research._source_visits(base,segments),links=reconnect['connections'],
        width=width,nominal=fraction_footprint.nominal(width['effective_width_mm']),
        mesh=surface_workspace.canonical_region_mesh(model.store.mesh,region),
        old_density=row_proximity_research.measure(old,check=check),base_density=row_proximity_research.measure(base,check=check))
    final,statistics,geometry,density=bend._candidate(state,region,selected,check,budget)
    if not bend._acceptable(state,statistics,density):raise ValueError('直接重放未满足运动与密度约束')
    if pair:
        from femto.planning.auto_row_shift_screen_research import _acceptable
        if not _acceptable(old,final,state['old_density'],density):raise ValueError('联合见证点未改善绝对近邻长度')
    positions,_=scan_fraction_seeds.propose(final,domain.ledger.arrays['region_faces'],request['requests'],check=check)
    seeds=scan_endpoints(final,domain.ledger.arrays['region_faces'])+positions
    route=dict(schema='research-checked-supplement-route-v1',computation_sha256=domain.revision['sha256'],
        source_route=base_route,items_sha256=geometry['items_sha256'],execution_ready=False)
    coverage.validate(proof,domain,seeds,state['nominal'],route_binding=route,guide=old_proof,check=check)
    comparison=coverage_partition.compare_union(old_proof,[proof],domain.areas,check,include_pending_leaves=True)
    if not comparison['retained_subset']:raise ValueError('直接重放未重新确认原名义集合')
    for plan,verified,source in ((old_plan,old_domain,source_model),(base_plan,domain,model)):
        planning_revision.require_current(source,region,plan['treatment_head'],plan['planning_context']['revision'])
        params=plan['surface_trajectory']['planner']['parameters']
        target_surface.validate(verified.ledger,plan['planning_context']['revision'],
            evidence=params['quality_evidence'],domain=params['effective_domain'],check=check,reproduce=False)
    for cert in (old_proof,proof,*(() if old_guide is None else (old_guide,))):coverage._checked_assets(cert,check)
    check()
    if planning_revision.digest(dict(old=old_plan,base=base_plan,region=region,request=request,
                                   old_proof=old_proof.metadata,proof=proof.metadata,
                                   old_guide=None if old_guide is None else old_guide.metadata))!=snapshot:
        raise ValueError('直接重放期间输入发生变化')
    result=dict(schema=PAIR_VERSION if pair else POINT_VERSION if point else VERSION,geometry=geometry,statistics=statistics,density=density,
        comparison=comparison,coverage_summary=proof.summary,intermediate_search_proof_replayed=False,
        installed_route=False,installed_certificate=False,full_target_coverage_claim=False,
        execution_ready=False,physical_accuracy_verified=False)
    if include_research_route:
        # This is a detached, review-only geometry snapshot, not a product
        # SurfaceTrajectory or an executable instruction. The same route was
        # rebuilt and checked above and is bound to the final proof.
        items=deepcopy([item.to_dict() for item in final.ordered_items])
        if planning_revision.digest(items)!=geometry['items_sha256']:
            raise ValueError('研究路线快照与已核验几何不一致')
        result['research_route']=dict(schema='research-validated-route-snapshot-v1',ordered_items=items,
            items_sha256=geometry['items_sha256'],source_route=base_route,computation_sha256=domain.revision['sha256'],
            installed_route=False,full_target_coverage_claim=False,execution_ready=False,physical_accuracy_verified=False)
        check()
        for cert in (old_proof,proof,*(() if old_guide is None else (old_guide,))):coverage._checked_assets(cert,check)
        if planning_revision.digest(dict(old=old_plan,base=base_plan,region=region,request=request,
                                       old_proof=old_proof.metadata,proof=proof.metadata,
                                       old_guide=None if old_guide is None else old_guide.metadata))!=snapshot:
            raise ValueError('研究路线快照期间输入发生变化')
    return result
