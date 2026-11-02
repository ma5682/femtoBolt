"""Research-only local-row preview for ranking v8 shift candidates.

The preview does not certify route geometry or coverage. Any unsupported case
causes the caller to use the full preconnection rank. All chosen candidates
still go through independent full planning and validation.
"""
from __future__ import annotations

from types import SimpleNamespace
import numpy as np
import trimesh

from femto.geometry.region_reuse import operation
from femto.planning import (effective_domain,planning_revision,planning_state,quality_constrained,
    row_proximity_research,surface_workspace)
from femto.planning.fixed_width_layout import OPTIMIZED_PATCH_VERSION,SHIFT_PATCH_VERSION
from femto.planning.local_metric_alignment import shifted
from femto.planning.mesh_slice_planner import _slice_rows
from femto.planning.selection_blocks import selection_face_assignment
from femto.planning.surface_trajectory import ScanSegment,face_components

VERSION='research-local-row-density-ranking-v1'
MAX_RANK=32
MAX_SOURCE_POINTS=200_000
LEVEL_TOLERANCE_M=1e-8


class PreviewUnavailable(ValueError):
    """The local-only preview is inconclusive and must not be used."""


def _motion_priority(value,original,threshold,*,prefer_known=False):
    fractions=[v['near_fraction'] for v in value['density']['proximity']]
    nonworse=all(a<=b+1e-12 for a,b in zip(fractions,original)) and any(a<b-1e-12 for a,b in zip(fractions,original))
    predicted=value['motion_proxy']
    significant=predicted is not None and predicted['predicted_connector_delta_m'] < -threshold
    if prefer_known:
        return (not nonworse,not significant,predicted is None,*fractions,value['index'])
    return (not nonworse,not significant,*fractions,value['index'])


def _prepare(trajectory,mesh,region,check):
    params=trajectory.planner['parameters'];layout=params.get('fixed_width_layout') or {}
    if layout.get('request',{}).get('version')!=OPTIMIZED_PATCH_VERSION or layout.get('row_shifts')!=[]:
        raise PreviewUnavailable('局部预览需要未调整的 v4 分区')
    vertices=np.asarray(mesh.vertices);faces=np.asarray(mesh.faces)
    components=face_components(faces)
    _,_,labels,_=selection_face_assignment(region,vertices,faces)
    width=params['effective_width_mm'];record=params['quality_evidence'];domain=params['effective_domain']
    allowed_faces=effective_domain.validate(domain,record,width,mesh,region,check)
    patches={};rows={}
    for fragment in layout['fragments']:
        check()
        for part in fragment['patches']:
            if part.get('frame') is None:continue
            key=(fragment['block_index'],fragment['face_indices_sha256'],part['seed_face'])
            if key in patches:raise PreviewUnavailable('局部预览分区索引不唯一')
            patches[key]=part
            for row in range(part['first_row_index'],part['first_row_index']+part['retained_row_count']):
                row_key=(fragment['block_index'],row)
                if row_key in rows:raise PreviewUnavailable('局部预览扫描行归属不唯一')
                rows[row_key]=key
    base=[];count=0
    for raw in (trajectory.gap_report.get('reconnection') or {}).get('base_items',[]):
        check()
        if raw.get('kind')!='scan_segment':continue
        segment=ScanSegment.from_dict(raw);count+=len(segment.points)
        if count>MAX_SOURCE_POINTS:raise PreviewUnavailable('局部预览扫描点超过预算')
        if (len({point.get('row_index') for point in segment.points})!=1
                or any(point.get('role')=='connector' for point in segment.points)):
            raise PreviewUnavailable('局部预览源扫描段包含混合行或连接')
        base.append(segment)
    if not base:raise PreviewUnavailable('局部预览缺少原扫描段')
    intrinsic=np.asarray(region['view_reference']['undistorted_intrinsic'])
    image_size=np.asarray(region['view_reference']['image_size'])
    return dict(vertices=vertices,faces=faces,components=components,labels=labels,allowed_faces=allowed_faces,
                params=params,width=width,record=record,patches=patches,rows=rows,base=base,
                blocks=trajectory.blocks['items'],intrinsic=intrinsic,image_size=image_size,mesh=mesh)


def _preview(state,item,head,check,*,return_segments=False):
    if not isinstance(item,dict) or not isinstance(item.get('row_shift'),dict):
        raise ValueError('局部预览偏移记录无效')
    shift=item['row_shift'];row=item.get('row');block=shift.get('block_index')
    if type(row) is not int or type(block) is not int:raise ValueError('局部预览行编号无效')
    key=(block,shift.get('fragment_face_sha256'),shift.get('patch_seed_face'))
    if state['rows'].get((block,row))!=key or key not in state['patches']:
        raise PreviewUnavailable('局部预览行与分区不匹配')
    patch=state['patches'][key];frame=patch['frame'];origin=np.asarray(frame['origin'],float);axis=np.asarray(frame['e2'],float)
    positions=np.asarray(patch['positions_m'],float);ids=np.asarray(patch['face_ids'],dtype=int)
    vertices,faces=state['vertices'],state['faces']
    if (origin.shape!=(3,) or axis.shape!=(3,) or positions.ndim!=1 or len(positions)<3
            or not np.isfinite([origin,axis]).all() or not np.isfinite(positions).all()
            or ids.ndim!=1 or not len(ids) or ids.min()<0 or ids.max()>=len(faces)):
        raise PreviewUnavailable('局部预览几何无效')
    where=np.flatnonzero(np.abs(positions-shift['position_m'])<=1e-12)
    if len(where)!=1 or where[0] in (0,len(positions)-1):raise PreviewUnavailable('局部预览层位置不唯一')
    source=[segment for segment in state['base'] if segment.block_index==block and segment.points[0]['row_index']==row]
    if not source:raise PreviewUnavailable('局部预览缺少源扫描行')
    if any(abs(float((np.asarray(point['xyz_m'])-origin)@axis)-float(positions[where[0]]))>LEVEL_TOLERANCE_M
           for segment in source for point in segment.points):
        raise PreviewUnavailable('局部预览源扫描行不位于原切片')
    triangles=vertices[faces[ids]];cross=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]);norm=np.linalg.norm(cross,axis=1)
    valid=norm>0
    if not valid.any():raise PreviewUnavailable('局部预览分区退化')
    triangles=triangles[valid];normal=cross[valid]/norm[valid,None]
    factors=np.sqrt(np.maximum(0.,1-(normal@axis)**2));levels=(triangles-origin)@axis
    intervals=np.column_stack((levels.min(axis=1),levels.max(axis=1)))
    maximum=state['width']*(1-head['overlap_percent']/100)/1000
    current=shifted(positions,int(where[0]),shift['shift_m'],intervals,factors,maximum,check)
    changed=np.flatnonzero(np.abs(current-positions)>1e-12)
    if len(changed)!=1 or int(changed[0])!=int(where[0]):
        raise PreviewUnavailable('局部预览不能忽略相邻行的再平衡')
    if not 0<=block<len(state['blocks']):raise PreviewUnavailable('局部预览工作块无效')
    component=state['blocks'][block]['component_id'];allowed=state['labels']==component+1
    raw,_=_slice_rows(trimesh.Trimesh(vertices=vertices,faces=faces[ids],process=False),ids,state['components'],
        origin,axis,np.asarray([current[where[0]]]),state['params'],state['intrinsic'],state['image_size'],
        check,{'points':0},block,allowed)
    replacement=[]
    for entry in raw:
        for candidate in entry['candidates']:
            points=[dict(point,row_index=row,role='scan') for point in candidate['points']]
            replacement.append(ScanSegment(len(replacement),row,points,block))
    if not replacement:raise PreviewUnavailable('局部预览的新行为空')
    accepted,_,_=quality_constrained.edge_checks(replacement,state['mesh'],state['record'],state['width'],
                                                  check,allowed_faces=state['allowed_faces'])
    changed_segments=[];offset=0
    for segment in replacement:
        points=[]
        for a,b in zip(segment.points,segment.points[1:]):
            if accepted[offset]:
                if not points:points=[a]
                points.append(b)
            elif points:
                changed_segments.append(ScanSegment(len(changed_segments),row,points,block));points=[]
            offset+=1
        if points:changed_segments.append(ScanSegment(len(changed_segments),row,points,block))
    untouched=[s for s in state['base'] if not(s.block_index==block and s.points[0]['row_index']==row)]
    density=row_proximity_research.measure(SimpleNamespace(ordered_items=untouched+changed_segments),check=check)
    return (density,changed_segments) if return_segments else density


def rank(model,region,plan,proposal,*,max_rank=16,check=lambda:None,progress=lambda text:None,_source=None,motion_priority=False,
         allow_partial=False,prefer_known_motion=False):
    if type(max_rank) is not int or not 1<=max_rank<=MAX_RANK:raise ValueError('局部预览排序预算无效')
    if (type(motion_priority) is not bool or type(allow_partial) is not bool or type(prefer_known_motion) is not bool
            or (prefer_known_motion and not motion_priority)):raise ValueError('局部预览排序方式无效')
    if (not isinstance(proposal,dict) or not isinstance(proposal.get('proposals'),list)
            or len(proposal['proposals'])>MAX_RANK or proposal.get('layout_algorithm')!=SHIFT_PATCH_VERSION):
        raise ValueError('局部预览候选无效')
    check();snapshot=planning_revision.digest(plan)
    if proposal.get('source_route_sha256')!=snapshot:raise ValueError('局部预览来源路线不一致')
    head=plan['treatment_head'];revision=plan['planning_context']['revision']
    planning_revision.require_current(model,region,head,revision)
    with operation():
        if _source is None:trajectory=planning_state.validated_surface(plan,model.store.mesh,region,check=check)
        else:
            from femto.planning.screen_source_reuse import _ScreenSource
            if type(_source) is not _ScreenSource or _source.check is not check:
                raise ValueError('局部预览源路线复用上下文无效')
            trajectory=_source.read(model,region,plan)
        count=min(max_rank,len(proposal['proposals']));ranked=[];skipped=[]
        if count:
            mesh=surface_workspace.canonical_region_mesh(model.store.mesh,region)
            state=_prepare(trajectory,mesh,region,check)
            baseline=row_proximity_research.measure(trajectory,check=check) if motion_priority else None
        for index,item in enumerate(proposal['proposals'][:count]):
            check();progress(f'局部预览扫描行 {index+1}/{count}')
            try:
                if motion_priority:
                    from femto.planning import local_motion_proxy_research
                    density,changed=_preview(state,item,head,check,return_segments=True)
                    motion=local_motion_proxy_research.estimate(state,trajectory,item['row_shift']['block_index'],item['row'],changed,check)
                else:density=_preview(state,item,head,check)
            except PreviewUnavailable as error:
                if not allow_partial:raise
                check()
                if planning_revision.digest(plan)!=snapshot:raise ValueError('局部预览期间来源路线已改变')
                planning_revision.require_current(model,region,head,revision)
                if _source is not None:_source.verify()
                skipped.append(dict(index=index,reason=str(error)));continue
            if any(x['near_fraction'] is None for x in density['proximity']):raise PreviewUnavailable('局部预览没有扫描边')
            entry=dict(index=index,density=density)
            if motion_priority:entry['motion_proxy']=motion
            ranked.append(entry)
            if planning_revision.digest(plan)!=snapshot:raise ValueError('局部预览期间来源路线已改变')
            planning_revision.require_current(model,region,head,revision)
        if motion_priority and count:
            original=[v['near_fraction'] for v in baseline['proximity']]
            if any(v is None for v in original):raise PreviewUnavailable('局部预览原路线没有扫描边')
            threshold=state['width']*.1/1000
            ranked.sort(key=lambda value:_motion_priority(value,original,threshold,prefer_known=prefer_known_motion))
        else:ranked.sort(key=lambda value:(*[x['near_fraction'] for x in value['density']['proximity']],value['index']))
        if count and not ranked:raise PreviewUnavailable('全部局部候选无法预览')
        order=[value['index'] for value in ranked]+[v['index'] for v in skipped]+list(range(count,len(proposal['proposals'])))
        check()
        if planning_revision.digest(plan)!=snapshot:raise ValueError('局部预览期间来源路线已改变')
        planning_revision.require_current(model,region,head,revision)
        if _source is not None:_source.verify()
    result=dict(version=VERSION,order=order,ranked=ranked,ranked_count=count,
        unranked_count=len(proposal['proposals'])-count,source_route_sha256=snapshot,
        candidate_only=True,coverage_claim=False,execution_ready=False,
        semantics='local_row_preview_ranking_only_requires_full_route_and_coverage_verification')
    if motion_priority:result['ordering_policy']='density_nonworse_then_predicted_connector_gain_above_tenth_width'
    if allow_partial:
        result['ranked_count']=len(ranked);result['unranked_count']=len(proposal['proposals'])-len(ranked)
        result['preview_skipped']=skipped
    if prefer_known_motion:result['ordering_policy']='density_nonworse_then_predicted_gain_then_known_motion_v2'
    return result
