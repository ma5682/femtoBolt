"""Identify bounded row-shift candidates from checked connection failures.

The result is only a proposal. A failed graph search does not prove physical
disconnection, and neither a small segment nor a candidate authorizes dropping
geometry or changing the original target. Re-slicing, all original predicates,
continuity and coverage review remain required before a result can be used.
"""
from __future__ import annotations

import math
import numpy as np

VERSION='isolated-micro-row-shift-candidates-v1'
MAX_CANDIDATES=4
WIDTH_FRACTION=.05
ABSOLUTE_MAX_M=.0005


def candidates(segments,visits,links,width_mm,check=lambda:None):
    from femto.planning.scan_visit_order import checked_visits
    visits=checked_visits(segments,visits)
    if (isinstance(width_mm,bool) or not isinstance(width_mm,(float,int)) or not math.isfinite(width_mm)
            or not .5<=width_mm<=100 or not isinstance(links,list) or len(links)!=len(visits)-1):
        raise ValueError('孤立短行候选输入无效')
    for (left,_),(right,_),link in zip(visits,visits[1:],links):
        check()
        if (not isinstance(link,dict) or link.get('from_segment')!=left or link.get('to_segment')!=right
                or type(link.get('connected')) is not bool):
            raise ValueError('孤立短行候选连接记录无效')
    row_segments={}
    for i,segment in enumerate(segments):
        check()
        rows={p['row_index'] for p in segment.points if p.get('role','scan')!='connector'}
        for row in rows:row_segments.setdefault((segment.block_index,row),set()).add(i)
    threshold=min(ABSOLUTE_MAX_M,width_mm*WIDTH_FRACTION/1000)
    result=[]
    for position in range(1,len(visits)-1):
        check()
        left,_=visits[position-1];index,_=visits[position];right,_=visits[position+1]
        segment=segments[index]
        if (links[position-1]['connected'] or links[position]['connected']
                or segments[left].block_index!=segment.block_index
                or segments[right].block_index!=segment.block_index):continue
        rows={p['row_index'] for p in segment.points}
        if len(rows)!=1 or any(p.get('role','scan')=='connector' for p in segment.points):continue
        row=next(iter(rows))
        if row_segments[(segment.block_index,row)]!={index}:continue
        xyz=np.asarray([p['xyz_m'] for p in segment.points],float)
        if xyz.ndim!=2 or xyz.shape[1]!=3 or not np.isfinite(xyz).all():raise ValueError('孤立短行坐标无效')
        length=float(np.linalg.norm(np.diff(xyz,axis=0),axis=1).sum())
        if 0<length<=threshold:
            result.append(dict(segment_index=index,block_index=segment.block_index,row_index=row,
                               length_m=length,shift_m=width_mm*WIDTH_FRACTION/1000))
    result.sort(key=lambda x:(x['length_m'],x['block_index'],x['row_index'],x['segment_index']))
    return result[:MAX_CANDIDATES]


def requests(trajectory,check=lambda:None):
    """Map isolated rows to their recorded source patch and trial offsets."""
    from femto.planning.surface_trajectory import ScanSegment
    params=trajectory.planner['parameters'];layout=params.get('fixed_width_layout') or {}
    fragments=layout.get('fragments');report=trajectory.gap_report.get('reconnection') or {}
    if not isinstance(fragments,list):raise ValueError('孤立短行缺少局部分区记录')
    segments=[ScanSegment.from_dict(item) for item in report.get('base_items',[]) if item.get('kind')=='scan_segment']
    ordering=report.get('ordering') or {}
    improvement=ordering.get('improvement') or {}
    visits=improvement.get('visits') if improvement.get('adopted') is True else ordering.get('visits')
    detour=ordering.get('preserved_detour') or {}
    if detour.get('adopted') is True:visits=detour.get('visits')
    joint=ordering.get('joint_reorder') or {}
    if joint.get('adopted') is True:visits=joint.get('visits')
    relocation=ordering.get('relocation') or {}
    if relocation.get('adopted') is True:visits=relocation.get('visits')
    deep=ordering.get('relocation_deep') or {}
    if deep.get('adopted') is True:visits=deep.get('visits')
    floor=ordering.get('relocation_floor') or {}
    if floor.get('adopted') is True:visits=floor.get('visits')
    unified=ordering.get('relocation_unified') or {}
    if unified.get('adopted') is True:visits=unified.get('visits')
    found=candidates(segments,visits,report.get('connections'),params['effective_width_mm'],check)
    proposals=[]
    for item in found:
        check();matching=[]
        for fragment in fragments:
            if fragment['block_index']!=item['block_index']:continue
            for part in fragment.get('patches',[]):
                first=part['first_row_index'];count=part['retained_row_count']
                if first<=item['row_index']<first+count:matching.append((fragment,part))
        if len(matching)!=1:raise ValueError('孤立短行无法唯一绑定局部分区')
        fragment,part=matching[0];frame=part['frame']
        xyz=np.asarray(segments[item['segment_index']].points[0]['xyz_m'],float)
        level=float((xyz-np.asarray(frame['origin']))@np.asarray(frame['e2']))
        positions=np.asarray(part['positions_m'],float)
        if not len(positions):raise ValueError('孤立短行缺少原扫描位置')
        index=int(np.argmin(np.abs(positions-level)))
        if abs(float(positions[index])-level)>1e-8:raise ValueError('孤立短行与分区位置不一致')
        identity=dict(block_index=item['block_index'],fragment_face_sha256=fragment['face_indices_sha256'],
                      patch_seed_face=part['seed_face'],position_m=float(positions[index]))
        proposals.append(dict(candidate=item,trial_shifts=[dict(identity,shift_m=item['shift_m']),
                                                            dict(identity,shift_m=-item['shift_m'])]))
    return proposals
