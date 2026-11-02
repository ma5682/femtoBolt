"""Propose bounded v8 row shifts from a validated route's nearby scan rows.

This module ranks work only. Its 3-D sampled proximity is not a surface
distance, coverage proof, or physical treatment assessment. Every proposal
must be regenerated, geometrically checked and independently compared with
the original evidence before it could be used.
"""
from __future__ import annotations

from copy import deepcopy
import math
import numpy as np

from femto.planning import planning_revision, planning_state, row_overlap_diagnostics, surface_workspace
from femto.planning.continuous_quality import RELOCATION_ROUTING, DEEP_RELOCATION_ROUTING
from femto.planning.fixed_width_layout import OPTIMIZED_PATCH_VERSION, SHIFT_PATCH_VERSION
from femto.planning.local_metric_alignment import shifted
from femto.planning.surface_trajectory import ScanSegment

VERSION='research-cross-patch-row-shift-proposals-v1'
MAX_ROWS=16
MAX_PROPOSALS=32
MAX_SOURCE_POINTS=200_000
LEVEL_TOLERANCE_M=1e-8


def _derive(trajectory, mesh, diagnostics, *, width_mm, overlap_percent, max_rows, fraction, check):
    layout=trajectory.planner['parameters']['fixed_width_layout']
    if layout.get('request',{}).get('version')!=OPTIMIZED_PATCH_VERSION or layout.get('row_shifts')!=[]:
        raise ValueError('行偏移研究需要未调整的 v4 局部分区路线')
    lookup={}
    for fragment in layout['fragments']:
        check()
        for patch in fragment['patches']:
            if patch.get('frame') is None:continue
            for row in range(patch['first_row_index'],patch['first_row_index']+patch['retained_row_count']):
                key=(fragment['block_index'],row)
                if key in lookup:raise ValueError('局部分区扫描行归属不唯一')
                lookup[key]=(fragment,patch)
    source={}; examined=0
    reconnect=trajectory.gap_report.get('reconnection') or {}
    for raw in reconnect.get('base_items',[]):
        check()
        if raw.get('kind')!='scan_segment':continue
        segment=ScanSegment.from_dict(raw)
        for point in segment.points:
            examined+=1
            if examined%1024==0:check()
            if examined>MAX_SOURCE_POINTS:raise ValueError('行偏移源扫描点超过预算')
            if point.get('role') not in (None,'scan'):continue
            key=(segment.block_index,point.get('row_index'))
            source.setdefault(key,point)
    vertices=np.asarray(mesh.vertices);faces=np.asarray(mesh.faces)
    maximum=width_mm*(1-overlap_percent/100)/1000
    amount=width_mm*fraction/1000
    proposals=[]; skipped=[]; matched=0
    for row in diagnostics['rows']:
        if matched>=max_rows:break
        check(); key=(row['block'],row['row'])
        owner=lookup.get(key); point=source.get(key)
        if owner is None or point is None:
            skipped.append(dict(block=key[0],row=key[1],reason='missing_patch_or_source_row'));continue
        fragment,patch=owner
        positions=np.asarray(patch['positions_m'],float);frame=patch['frame']
        origin=np.asarray(frame['origin'],float);axis=np.asarray(frame['e2'],float)
        xyz=np.asarray(point['xyz_m'],float)
        if (positions.ndim!=1 or len(positions)<3 or not np.isfinite(positions).all()
                or np.any(np.diff(positions)<=0) or origin.shape!=(3,) or axis.shape!=(3,)
                or xyz.shape!=(3,) or not np.isfinite([origin,axis,xyz]).all()):
            skipped.append(dict(block=key[0],row=key[1],reason='invalid_patch_positions'));continue
        level=float((xyz-origin)@axis);idx=int(np.argmin(np.abs(positions-level)))
        if abs(float(positions[idx])-level)>LEVEL_TOLERANCE_M or idx in (0,len(positions)-1):
            skipped.append(dict(block=key[0],row=key[1],reason='unmatched_or_boundary_level'));continue
        ids=np.asarray(patch['face_ids'],dtype=int)
        if ids.ndim!=1 or not len(ids) or ids.min()<0 or ids.max()>=len(faces):
            raise ValueError('局部分区来源面无效')
        triangles=vertices[faces[ids]]
        normals=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]);norms=np.linalg.norm(normals,axis=1)
        valid=norms>0
        if not valid.any():
            skipped.append(dict(block=key[0],row=key[1],reason='degenerate_patch'));continue
        triangles=triangles[valid];normals=normals[valid]/norms[valid,None]
        factors=np.sqrt(np.maximum(0.,1-(normals@axis)**2));levels=(triangles-origin)@axis
        intervals=np.column_stack((levels.min(axis=1),levels.max(axis=1)))
        matched+=1
        for direction in (-1,1):
            check();delta=direction*amount
            try: shifted(positions,idx,delta,intervals,factors,maximum,check)
            except ValueError:
                check();skipped.append(dict(block=key[0],row=key[1],direction=direction,reason='spacing_infeasible'));continue
            record=dict(block_index=key[0],fragment_face_sha256=fragment['face_indices_sha256'],
                        patch_seed_face=patch['seed_face'],position_m=float(positions[idx]),shift_m=float(delta))
            proposals.append(dict(row=key[1],near_sample_m=float(row['near_m']),row_shift=record))
            if len(proposals)>=MAX_PROPOSALS:break
        if len(proposals)>=MAX_PROPOSALS:break
    return proposals,skipped,matched


def propose(model,region,plan,*,max_rows=8,fraction=.03,check=lambda:None,_source=None):
    """Return deterministic v8 shift requests for the caller to verify."""
    if (type(max_rows) is not int or not 1<=max_rows<=MAX_ROWS
            or isinstance(fraction,bool) or not isinstance(fraction,(int,float))
            or not math.isfinite(fraction) or not 0<fraction<=.05):
        raise ValueError('行偏移候选预算或幅度无效')
    check(); snapshot=planning_revision.digest(plan)
    head=plan['treatment_head']; context=plan['planning_context']
    planning_revision.require_current(model,region,head,context['revision'])
    if _source is None:
        trajectory=planning_state.validated_surface(plan,model.store.mesh,region,check=check)
    else:
        from femto.planning.screen_source_reuse import _ScreenSource
        if type(_source) is not _ScreenSource or _source.check is not check:
            raise ValueError('行偏移源路线复用上下文无效')
        trajectory=_source.read(model,region,plan)
    mesh=surface_workspace.canonical_region_mesh(model.store.mesh,region)
    diagnostics=row_overlap_diagnostics.inspect(trajectory,check=check)
    proposals,skipped,matched=_derive(trajectory,mesh,diagnostics,width_mm=head['effective_treatment_width_mm'],
        overlap_percent=head['overlap_percent'],max_rows=max_rows,fraction=fraction,check=check)
    if planning_revision.digest(plan)!=snapshot:raise ValueError('行偏移候选期间输入路线已改变')
    planning_revision.require_current(model,region,head,context['revision']);check()
    if _source is not None:_source.verify()
    # Keep the source route's additional search when comparing shifted rows;
    # reverting v12 to v11 would discard the baseline's ordering improvement.
    routing=(DEEP_RELOCATION_ROUTING if context['revision']['inputs']['algorithms'].get('routing')==DEEP_RELOCATION_ROUTING
             else RELOCATION_ROUTING)
    return dict(version=VERSION,source_route_sha256=snapshot,diagnostics=diagnostics,
        proposals=deepcopy(proposals),skipped=skipped,matched_rows=matched,max_rows=max_rows,
        shift_fraction=float(fraction),layout_algorithm=SHIFT_PATCH_VERSION,routing_algorithm=routing,
        candidate_only=True,coverage_claim=False,execution_ready=False,
        semantics='sampled_proximity_ranking_requires_route_geometry_and_coverage_replay')
