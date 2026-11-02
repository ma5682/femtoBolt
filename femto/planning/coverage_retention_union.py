"""Experimental replayed comparison of an old proof to current-route proofs.

This function does not change a route, install a certificate, or assert
physical coverage. Every proof must be replayed from the bound accepted route.
Different computation revisions are allowed only for an identical target.
"""
from __future__ import annotations

import math
import numpy as np

from femto.planning import coverage_certificate as coverage, footprint_model
from femto.planning.coverage_domain import from_plan
from femto.planning.coverage_partition import MAX_UNION_CERTIFICATES, compare_union
from femto.planning.coverage_workflow import route_binding
from femto.planning.surface_trajectory import SurfaceTrajectory


def _same_target(before, after, check):
    def identity(binding):
        return {k:v for k,v in binding.items() if k not in ('sha256','target_sha256','computation_sha256')}
    if (before.ledger is None or after.ledger is None or before.revision is None or after.revision is None
            or identity(before.binding) != identity(after.binding)
            or before.revision['target_id'] != after.revision['target_id']):
        raise ValueError('覆盖比较必须绑定同一个原目标')
    first, second = before.ledger.arrays, after.ledger.arrays
    if set(first) != set(second):raise ValueError('覆盖比较的原目标数组不一致')
    for key in sorted(first):
        check()
        a,b=np.asarray(first[key]),np.asarray(second[key])
        if (a.shape != b.shape or a.dtype != b.dtype or a.dtype.hasobject
                or not a.flags.c_contiguous or not b.flags.c_contiguous):
            raise ValueError('覆盖比较的原目标数组不一致：'+key)
        # Check actual bytes rather than archive metadata; bound cancellation
        # latency and temporary memory even for large target arrays.
        left=memoryview(a.reshape(-1)).cast('B');right=memoryview(b.reshape(-1)).cast('B')
        for offset in range(0,len(left),1024*1024):
            check()
            if left[offset:offset+1024*1024] != right[offset:offset+1024*1024]:
                raise ValueError('覆盖比较的原目标数组不一致：'+key)


def _inputs(model,region,previous_plan,current_plan,check):
    old_domain,old_seeds,old_model,_ = from_plan(model,region,previous_plan,check)
    new_domain,new_seeds,new_model,_ = from_plan(model,region,current_plan,check)
    _same_target(old_domain,new_domain,check)
    if old_model != new_model:raise ValueError('覆盖比较不能混用治疗头作用模型')
    old_route=route_binding(SurfaceTrajectory.from_dict(previous_plan['surface_trajectory'],check=check))
    new_route=route_binding(SurfaceTrajectory.from_dict(current_plan['surface_trajectory'],check=check))
    return old_domain,old_seeds,old_model,old_route,new_domain,new_seeds,new_model,new_route


def compare_replayed(model, region, previous_plan, previous_certificate, current_plan, current_proofs, *,
                     previous_guide=None, include_existing_witnesses=False, include_pending_leaves=False, check=lambda:None):
    """Replay each (certificate, guide) against its plan, then compare sets.

    Each plan goes through the retained route and target validators first.
    The previous and current target arrays must be byte-identical, even when
    their computation revision identifiers differ. A guide is an input to
    replay, never a source of inherited proof status.
    """
    if (type(include_existing_witnesses) is not bool or type(include_pending_leaves) is not bool
            or not isinstance(current_proofs, (list,tuple))
            or not 1 <= len(current_proofs) <= MAX_UNION_CERTIFICATES-int(include_existing_witnesses)
            or any(not isinstance(item,(list,tuple)) or len(item)!=2 for item in current_proofs)):
        raise ValueError('覆盖比较的当前证据数量或格式无效')
    old_domain,old_seeds,old_model,old_route,new_domain,new_seeds,new_model,new_route = _inputs(
        model,region,previous_plan,current_plan,check)
    coverage.validate(previous_certificate,old_domain,old_seeds,old_model,
                      route_binding=old_route,guide=previous_guide,check=check)
    for certificate,guide in current_proofs:
        check()
        coverage.validate(certificate,new_domain,new_seeds,new_model,
                          route_binding=new_route,guide=guide,check=check)
    union=[item[0] for item in current_proofs]
    matched_count=0
    if include_existing_witnesses:
        if old_model.get('schema') != footprint_model.SCHEMA:
            raise ValueError('扫描点见证匹配仅支持固定名义作用模型')
        leaves,_,_=_matched_witnesses(previous_certificate,new_domain,new_seeds,check)
        matched_count=len(leaves)
        statuses=previous_certificate.arrays['status'].copy()
        statuses[statuses==1]=7
        statuses[leaves]=1
        # In-memory partition view only. The original proof has been replayed
        # and every marked witness is on an accepted current-route scan edge.
        # It is not a newly generated or installable certificate.
        union.append(coverage.Certificate({},dict(previous_certificate.arrays,status=statuses)))
    comparison=compare_union(previous_certificate,union,new_domain.areas,check,include_pending_leaves=include_pending_leaves)
    return dict(schema='experimental-replayed-coverage-retention-comparison-v1',
                target_id=new_domain.revision['target_id'],
                previous_certificate_sha256=previous_certificate.metadata['sha256'],
                current_certificate_sha256=[item[0].metadata['sha256'] for item in current_proofs],
                previous_route=old_route,current_route=new_route,comparison=comparison,
                existing_witness_matching=include_existing_witnesses,matched_previous_leaves=matched_count,
                installed_certificate=False,full_target_coverage_claim=False,
                execution_ready=False,physical_accuracy_verified=False)


def _matched_witnesses(previous_certificate,new_domain,new_seeds,check):
    """Match against inputs that have already passed authoritative replay."""
    keys=set()
    for i,seed in enumerate(new_seeds):
        if i%1024==0:check()
        if seed.get('role')!='scan':continue
        face=int(seed['source_face'])
        if not 0<=face<len(new_domain.representative_faces):raise ValueError('新路线扫描点来源面无效')
        point=tuple(seed['xyz_m'])
        if len(point)!=3 or not all(math.isfinite(v) for v in point):raise ValueError('新路线扫描点坐标无效')
        keys.add((int(new_domain.representative_faces[face]),*point))
    arrays=previous_certificate.arrays
    present=np.empty(len(arrays['seed_face']),dtype=bool)
    for i,(face,point) in enumerate(zip(arrays['seed_face'],arrays['seed_xyz'])):
        if i%1024==0:check()
        present[i]=(int(face),*point) in keys
    check()
    proof_ids=np.flatnonzero(present[arrays['proof_seed']])
    # A replayed alternative proof on the same leaf is as valid as its chosen
    # display witness. Choose the earliest proof deterministically.
    leaves,first=np.unique(arrays['proof_leaf'][proof_ids],return_index=True)
    selected=proof_ids[first]
    confirmed=arrays['status'][leaves]==1
    leaves,selected=leaves[confirmed],selected[confirmed]
    keep=np.zeros(len(arrays['root']),dtype=bool);keep[leaves]=True
    missing=np.flatnonzero((arrays['status']==1)&~keep)
    return leaves,selected,missing


def retained_by_existing_scans(model, region, previous_plan, previous_certificate, current_plan, *,
                               previous_guide=None, check=lambda:None):
    """Identify old proof leaves whose same fixed-ball witness is on the new route.

    Only a diagnostic result is returned. It depends on replaying the old proof,
    validating both routes, exact target identity and matching source face and
    XYZ of an accepted scan endpoint. Parent proof paths were replayed in full.
    Missing witnesses mean unconfirmed, not physically uncovered. No result is
    installed as a coverage certificate and no route is changed.
    """
    old_domain,old_seeds,old_model,old_route,new_domain,new_seeds,_,new_route = _inputs(
        model,region,previous_plan,current_plan,check)
    if old_model.get('schema') != footprint_model.SCHEMA:
        raise ValueError('扫描点见证匹配仅支持固定名义作用模型')
    coverage.validate(previous_certificate,old_domain,old_seeds,old_model,
                      route_binding=old_route,guide=previous_guide,check=check)
    leaves,selected,missing=_matched_witnesses(previous_certificate,new_domain,new_seeds,check)
    arrays=previous_certificate.arrays
    def weight(indices):
        def values():
            for j,i in enumerate(indices):
                if j%1024==0:check()
                yield float(new_domain.areas[int(arrays['root'][i])])/(4**int(arrays['depth'][i]))
        return math.fsum(values())
    check()
    return dict(schema='experimental-existing-scan-witness-retention-v1',
                target_id=new_domain.revision['target_id'],
                previous_certificate_sha256=previous_certificate.metadata['sha256'],
                previous_route=old_route,current_route=new_route,
                retained_leaf_indices=leaves.tolist(),retained_proof_indices=selected.tolist(),
                not_reconfirmed_leaf_indices=missing.tolist(),
                previous_confirmed_leaf_count=len(leaves)+len(missing),
                retained_nominal_area_m2=weight(leaves),
                previously_confirmed_not_reconfirmed_nominal_area_m2=weight(missing),
                retained_subset=len(missing)==0,installed_certificate=False,
                full_target_coverage_claim=False,execution_ready=False,physical_accuracy_verified=False)


def propose_replayed(model, region, previous_plan, previous_certificate, current_plan, current_proofs, *,
                     previous_guide=None, max_selected=512, max_updates=1_000_000, check=lambda:None):
    """Produce only old-edge proposals after independently replaying evidence.

    The proposal is not a modified route or coverage certificate. Its source
    edges must still be connected, checked against current geometry and bound
    to freshly validated evidence before they can be used.
    """
    comparison=compare_replayed(model,region,previous_plan,previous_certificate,current_plan,current_proofs,
        previous_guide=previous_guide,include_existing_witnesses=True,include_pending_leaves=True,check=check)
    domain,seeds,model_record,_=from_plan(model,region,previous_plan,check)
    trajectory=SurfaceTrajectory.from_dict(previous_plan['surface_trajectory'],check=check)
    if (domain.revision['target_id']!=comparison['target_id']
            or route_binding(trajectory)!=comparison['previous_route']
            or previous_certificate.metadata['sha256']!=comparison['previous_certificate_sha256']):
        raise ValueError('补段候选的路线或原目标在核验过程中发生变化')
    from femto.planning.retention_edge_candidates import propose
    proposal=propose(previous_certificate,seeds,trajectory,domain,comparison['comparison']['pending_old_leaf_indices'],
                     model_record['effective_width_mm'],max_selected=max_selected,max_updates=max_updates,check=check)
    if route_binding(SurfaceTrajectory.from_dict(current_plan['surface_trajectory'],check=check))!=comparison['current_route']:
        raise ValueError('补段候选的当前路线在核验过程中发生变化')
    return dict(schema='experimental-replayed-retention-edge-proposal-v1',comparison=comparison,proposal=proposal,
                installed_route=False,installed_certificate=False,execution_ready=False,physical_accuracy_verified=False)
