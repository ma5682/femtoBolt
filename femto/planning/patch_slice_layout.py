"""Local-direction slices with original mesh/selection support and global IDs."""
from __future__ import annotations

import numpy as np
import trimesh

PRIOR_ALIGNMENT_MIN_AREA_WIDTHS_SQUARED=40.
PRIOR_ALIGNMENT_MIN_METRIC=.3
OPTIMIZED_EXTRA_FRACTION=.2


def prepare(layout,vertices,faces,ids,component,components,params,region,intrinsic,image_size,
            check,budget,row_offset,block_index,allowed,global_frame,gap_context):
    from femto.planning.surface_direction_patches import partition,rows,VERSION
    from femto.planning.mesh_slice_planner import (_slice_rows,_order_rows_by_frame,_build_items,
                                                   _segment_lengths,Unplannable)
    from femto.planning.surface_scan_order import order_scan_records
    from femto.planning.surface_trajectory import face_list_sha256
    from femto.planning.fixed_width_layout import MAX_ROWS
    patches=partition(vertices,faces,ids,check)
    adaptive_records=[]
    if layout.is_adaptive:
        from femto.planning.adaptive_cut_partition import propose as propose_cut
        pieces=[]
        for patch in patches:
            divided,decision=propose_cut(vertices,faces,patch,global_frame['e1'],global_frame['e2'],
                layout.spec['effective_width_mm'],check,fractions=(.5,))
            adaptive_records.append(dict(source_seed_face=patch['seed_face'],**decision))
            pieces.extend(divided)
        patches=sorted(pieces,key=lambda part:part['face_ids'][0])
    if layout.is_foldback:
        from femto.planning.foldback_partition import split
        patches=split(vertices,faces,patches,check)
    fragment_hash=face_list_sha256(ids)
    prior=[];prior_status=None
    if layout.is_stable:
        from femto.planning.fixed_width_layout import Layout,request,RESIDUAL_VERSION
        previous=Layout(request(layout.spec['effective_width_mm'],layout.spec['overlap_percent'],version=RESIDUAL_VERSION),layout.evidence)
        try:
            prior=previous.positions(vertices,faces,ids,np.asarray(global_frame['origin']),np.asarray(global_frame['e2']),
                                     block_index,row_offset,check).tolist()
            prior_status='computed' if prior else 'no_prior_rows'
        except ValueError as error:
            if str(error)!='局部表面需要另一扫描方向':raise
            # A vertical face can make the legacy global frame unusable. Local
            # frames remain eligible; this never substitutes a fake position.
            prior_status='global_frame_metric_degenerate'
    saved=[];records=[];dropped=[];offset=row_offset;slice_offset=0;all_steps=[]
    if layout.is_adaptive_floor:
        from femto.planning.adaptive_spacing_research import selection_distance
        boundary_distance=selection_distance(allowed,check)
    for patch_index,patch in enumerate(patches):
        check()
        try:
            local_frame=None
            triangle=vertices[faces[np.asarray(patch['face_ids'],dtype=int)]]
            if layout.is_stable and patch['area_m2'] >= PRIOR_ALIGNMENT_MIN_AREA_WIDTHS_SQUARED*(layout.spec['effective_width_mm']/1000)**2:
                normals=np.cross(triangle[:,1]-triangle[:,0],triangle[:,2]-triangle[:,0]);sizes=np.linalg.norm(normals,axis=1)
                valid=sizes>0
                if valid.any():
                    normals=normals[valid]/sizes[valid,None];axis=np.asarray(global_frame['e2'])
                    factor=float(np.sqrt(np.maximum(0.,1-(normals@axis)**2)).min())
                    if factor>=PRIOR_ALIGNMENT_MIN_METRIC:
                        local_frame=dict(global_frame,minimum_transverse_metric_factor=factor,
                                         direction_source='prior_fragment_frame_local_metric')
            decision=rows(vertices,faces,patch,layout.spec['effective_width_mm'],layout.spec['overlap_percent'],layout.reference,check,local_frame=local_frame)
            if layout.is_coordinated_floor and len(decision['positions_m'])>1:
                from femto.planning.coordinated_seam_research import propose as propose_seam_direction,VERSION as SEAM_DIRECTION_VERSION
                candidate=propose_seam_direction(vertices,faces,patch,decision['frame'],[item for item in saved if item.get('frame')],check)
                direction=dict(version=SEAM_DIRECTION_VERSION,adopted=False,reason='no_eligible_direction')
                if candidate is not None:
                    candidate_frame,detail=candidate
                    try:
                        trial=rows(vertices,faces,patch,layout.spec['effective_width_mm'],layout.spec['overlap_percent'],
                                   layout.reference,check,local_frame=candidate_frame)
                    except ValueError as error:
                        if str(error) not in ('局部表面需要另一扫描方向','自动排布横向范围存在无表面区间'):raise
                        trial=None;direction=dict(detail,adopted=False,reason='infeasible_metric')
                    if trial is not None:
                        if trial['positions_m'] and len(trial['positions_m'])<=len(decision['positions_m']):
                            baseline_rows=len(decision['positions_m']);trial_rows=len(trial['positions_m'])
                            decision=trial;local_frame=candidate_frame
                            direction=dict(detail,adopted=True,reason='no_extra_rows',baseline_rows=baseline_rows,trial_rows=trial_rows)
                        else:direction=dict(detail,adopted=False,reason='extra_or_missing_rows')
                decision['seam_direction']=direction
            if (local_frame is not None or layout.is_adaptive_floor) and len(decision['positions_m'])>1:
                from femto.planning.local_metric_alignment import positions as aligned_positions,VERSION as ALIGNMENT_VERSION
                if local_frame is None:
                    local_frame=decision['frame']
                normals=np.cross(triangle[:,1]-triangle[:,0],triangle[:,2]-triangle[:,0]);sizes=np.linalg.norm(normals,axis=1)
                valid=sizes>0;triangle=triangle[valid];normals=np.cross(triangle[:,1]-triangle[:,0],triangle[:,2]-triangle[:,0])
                normals=normals/np.linalg.norm(normals,axis=1)[:,None]
                axis=np.asarray(local_frame['e2']);origin=np.asarray(local_frame['origin'])
                factors=np.sqrt(np.maximum(0.,1-(normals@axis)**2));levels=(triangle-origin)@axis
                intervals=np.column_stack((levels.min(axis=1),levels.max(axis=1)))
                first,last=decision['positions_m'][0],decision['positions_m'][-1]
                step=layout.spec['effective_width_mm']*(1-layout.spec['overlap_percent']/100)/1000
                position_prior=prior if local_frame.get('direction_source')=='prior_fragment_frame_local_metric' else decision['positions_m']
                minimum_fraction=layout.minimum_metric_fraction
                if layout.is_adaptive_floor:
                    from femto.planning.adaptive_spacing_research import choose_fraction,seam_anchors
                    from femto.planning.local_metric_alignment import schedule
                    metric_span=float(schedule(first,last,intervals,factors,check)[1][-1])
                    adaptive=choose_fraction(vertices,faces,patch['face_ids'],local_frame,metric_span,
                        layout.spec['effective_width_mm'],boundary_distance,intrinsic,check)
                    minimum_fraction=adaptive['minimum_fraction']
                    anchors,sources=seam_anchors(patch,local_frame,[item for item in saved if item.get('frame')],first,last,check)
                    position_prior=sorted(set([*position_prior,*anchors]))
                    decision['adaptive_spacing']=adaptive
                    decision['adaptive_seams']=dict(anchors_m=anchors,sources=sources,
                        semantics='parallel_neighbor_phase_proposal_not_coverage')
                if layout.is_optimized:
                    from femto.planning.metric_row_optimizer import positions_with_slack,SLACK_VERSION,FLOOR_VERSION
                    direction_adopted=layout.is_coordinated_floor and decision['seam_direction']['adopted']
                    extra_fraction=0. if layout.is_minimal or direction_adopted else OPTIMIZED_EXTRA_FRACTION
                    floor_options=dict(minimum_fraction=minimum_fraction) if layout.is_spacing_floor else {}
                    try:
                        proposed,optimization=positions_with_slack(first,last,intervals,factors,step,position_prior,check,
                                                                   extra_fraction=extra_fraction,**floor_options)
                        decision['alignment_optimization']=dict(strategy='bounded_global',**optimization)
                    except ValueError as error:
                        budget_errors=('局部优化候选行超过预算','局部优化候选比较超过预算',
                                       '局部优化行数超过预算','局部优化状态转移超过预算')
                        if str(error) not in budget_errors:raise
                        if layout.is_spacing_floor:
                            # Uniform metric rows preserve the original maximum
                            # without the old prior-greedy half-spacing fallback.
                            from femto.planning.fixed_width_layout import metric_positions
                            proposed,metric_step=metric_positions(first,last,intervals,factors,step)
                            decision['alignment_optimization']=dict(strategy='uniform_metric_budget_fallback',
                                version=FLOOR_VERSION,minimum_fraction=minimum_fraction,reason=str(error),metric_step_m=metric_step)
                        else:
                            proposed=aligned_positions(first,last,intervals,factors,step,prior,check)
                            decision['alignment_optimization']=dict(strategy='prior_greedy_budget_fallback',
                                version=SLACK_VERSION,extra_fraction=extra_fraction,reason=str(error))
                else:proposed=aligned_positions(first,last,intervals,factors,step,prior,check)
                if layout.is_coordinated_floor and decision['seam_direction']['adopted'] and len(proposed)>decision['seam_direction']['baseline_rows']:
                    raise ValueError('接缝方向协调增加了原候选行数')
                decision['positions_m']=proposed.tolist()
                decision['alignment_rule']=(FLOOR_VERSION if layout.is_spacing_floor else SLACK_VERSION) if layout.is_optimized else ALIGNMENT_VERSION
                decision['retained_prior_level_count']=sum(float(x) in prior for x in proposed)
            if layout.is_repair:
                matches=[(i,item) for i,item in enumerate(layout.row_shifts) if item['block_index']==block_index
                         and item['fragment_face_sha256']==fragment_hash and item['patch_seed_face']==patch['seed_face']]
                if matches:
                    from femto.planning.local_metric_alignment import shifted,SHIFT_VERSION
                    chosen=np.asarray(decision['positions_m'],dtype=float)
                    if len(chosen)<3:raise ValueError('局部行调整缺少相邻扫描行')
                    axis=np.asarray(decision['frame']['e2']);origin=np.asarray(decision['frame']['origin'])
                    raw_triangles=vertices[faces[np.asarray(patch['face_ids'],dtype=int)]]
                    normals=np.cross(raw_triangles[:,1]-raw_triangles[:,0],raw_triangles[:,2]-raw_triangles[:,0]);sizes=np.linalg.norm(normals,axis=1)
                    valid=sizes>0;raw_triangles=raw_triangles[valid];normals=normals[valid]/sizes[valid,None]
                    factors=np.sqrt(np.maximum(0.,1-(normals@axis)**2));levels=(raw_triangles-origin)@axis
                    intervals=np.column_stack((levels.min(axis=1),levels.max(axis=1)))
                    step=layout.spec['effective_width_mm']*(1-layout.spec['overlap_percent']/100)/1000
                    for shift_index,item in matches:
                        locations=np.flatnonzero(np.abs(chosen-item['position_m'])<=1e-12)
                        if len(locations)!=1:raise ValueError('局部行调整的原位置不唯一')
                        chosen=shifted(chosen,int(locations[0]),item['shift_m'],intervals,factors,step,check)
                        layout.applied_shifts.append(shift_index)
                    if layout.is_spacing_floor:
                        from femto.planning.local_metric_alignment import schedule
                        import math
                        levels,distances=schedule(float(chosen[0]),float(chosen[-1]),intervals,factors,check)
                        span=float(distances[-1]);minimum=max(1,math.ceil((span-1e-12)/step))
                        floor_fraction=(decision['adaptive_spacing']['minimum_fraction'] if layout.is_adaptive_floor else layout.minimum_metric_fraction)
                        floor=min(floor_fraction*step,span/minimum)
                        actual=np.diff(np.interp(chosen,levels,distances))
                        if np.any(actual<floor-2e-12):raise ValueError('局部行位置调整违反尺度行距下限')
                    decision['positions_m']=chosen.tolist();decision['row_shift_rule']=SHIFT_VERSION
        except ValueError as error:
            # Degenerate faces remain explicit; no other geometry/budget error
            # can silently turn into a missing patch.
            if str(error)!='表面规划分区没有正面积三角面':raise
            saved.append(dict(**patch,status='no_positive_area',first_row_index=offset,retained_row_count=0))
            continue
        positions=np.asarray(decision['positions_m']);local=decision['frame']
        if sum(len(p.get('positions_m',[])) for p in saved)+len(positions)>MAX_ROWS:
            raise ValueError('局部表面分区超过扫描行预算，请缩小区域')
        patch_ids=np.asarray(patch['face_ids'],dtype=int)
        origin=np.asarray(local['origin']);e1=np.asarray(local['e1']);e2=np.asarray(local['e2'])
        raw,removed=_slice_rows(trimesh.Trimesh(vertices=vertices,faces=faces[patch_ids],process=False),patch_ids,
            components,origin,e2,positions,params,intrinsic,image_size,check,budget,block_index,allowed)
        for row in raw:
            for candidate in row['candidates']:candidate['level_m']=row['position_m']
        ordered=_order_rows_by_frame(raw,params,offset,origin,e1,e2,intrinsic,image_size)
        ordered=order_scan_records(ordered,local,check)
        for record in ordered:record['slice_index']+=slice_offset
        count=len({r['row_index'] for r in ordered})
        saved.append(dict(**patch,**decision,first_row_index=offset,retained_row_count=count))
        offset+=count;slice_offset+=len(positions)+1
        records.extend(ordered)
        dropped.extend(dict(d,patch_index=patch_index) for d in removed)
        all_steps.extend((np.diff(positions)*1000).tolist())
    fragment=dict(block_index=int(block_index),first_row_index=int(row_offset),
        face_indices_sha256=fragment_hash,face_count=len(ids),partition_version=VERSION,patches=saved,
        row_count=sum(len(p.get('positions_m',[])) for p in saved),retained_row_count=offset-row_offset,
        actual_plane_step_mm=max(all_steps) if all_steps else None,
        minimum_plane_step_mm=min(all_steps) if all_steps else None,
        status='proposed' if records else 'no_supported_segment')
    if layout.is_foldback:
        from femto.planning.foldback_partition import VERSION as FOLDBACK_VERSION
        fragment['foldback_split_version']=FOLDBACK_VERSION
    if layout.is_adaptive:
        fragment['adaptive_cut_proposals']=adaptive_records
    if layout.is_stable:
        fragment.update(prior_positions_m=prior,prior_status=prior_status,
            prior_alignment_policy=dict(min_area_widths_squared=PRIOR_ALIGNMENT_MIN_AREA_WIDTHS_SQUARED,
                                        minimum_transverse_metric=PRIOR_ALIGNMENT_MIN_METRIC))
    layout.fragments.append(fragment)
    if not records:
        raise Unplannable('no_supported_segment','局部表面分区没有通过参考选择的扫描段',details=dict(dropped_fragments=dropped))
    segments,gaps=_build_items(records,region,intrinsic,image_size,params['sample_step_mm']/1000,*gap_context)
    return dict(frame=global_frame,e2=np.asarray(global_frame['e2']),ordered_records=records,segments=segments,gaps=gaps,
                dropped_fragments=dropped,lengths=_segment_lengths(segments),gap_context=gap_context,local_patch_order=True)


def validate(record,mesh,region,parameters,check):
    """Rebuild partition, local frames, seam policy, slices and row IDs."""
    from femto.planning.fixed_width_layout import Layout
    from femto.planning.mesh_slice_planner import _prepare_block,Unplannable
    from femto.planning.selection_blocks import selection_face_assignment,_reading_key,_fragment_descriptors
    from femto.planning.surface_trajectory import face_components
    vertices,faces=np.asarray(mesh.vertices),np.asarray(mesh.faces)
    intent,_,labels,owners=selection_face_assignment(region,vertices,faces)
    components=face_components(faces);fresh=Layout(record['request'],parameters.get('quality_evidence'),row_shifts=record.get('row_shifts'))
    intrinsic=np.asarray(region['view_reference']['undistorted_intrinsic']);size=np.asarray(region['view_reference']['image_size'])
    offset=0;budget={'points':0}
    for block_index,block in enumerate(sorted(intent['components'],key=lambda b:_reading_key(b['bbox'],b['component_id'],parameters['start_corner']))):
        cid=block['component_id'];ids=np.flatnonzero(owners==cid)
        for descriptor in _fragment_descriptors(vertices,faces,ids,components,intrinsic,size,parameters['start_corner']):
            check()
            try:
                prepared=_prepare_block(vertices,faces,descriptor['face_ids'],descriptor['mesh_component_id'],components,
                    parameters,region,intrinsic,size,check,budget,offset,block_index,labels==cid+1,row_layout=fresh)
            except Unplannable:continue
            offset+=len({s.row_index for s in prepared['segments']})
    if fresh.record()!=record:raise ValueError('局部表面分区、扫描方向或切片记录不能独立重算')
