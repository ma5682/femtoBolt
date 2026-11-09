"""Versioned sampling choice from the selected canonical mesh, never row spacing.

This is a planning workload heuristic, not a physical accuracy certificate.
Triangle crossings and the existing whole-edge predicates remain authoritative.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

POLICY = 'selection-area-boundary-sampling-v1'
CURVATURE_POLICY = 'selection-shape-tangent-boundary-sampling-v2'
STEPS = (1., 2., 3., 5.)


def _record(area: float, perimeter: float, policy=POLICY) -> dict:
    if policy not in (POLICY, CURVATURE_POLICY):raise ValueError('未知的沿线取点策略')
    if (type(area) not in (int, float) or type(perimeter) not in (int, float)
            or not math.isfinite(area) or not math.isfinite(perimeter)
            or area <= 0 or perimeter < 0):
        raise ValueError('自动取点的选区面积或边界长度无效')
    # 2A/P penalizes thin, perforated and fragmented selections. A closed
    # surface has no boundary; its area scale alone supplies the upper bound.
    width = 2*area/perimeter if perimeter else math.sqrt(area)
    limit = min(math.sqrt(area)/12, width/4, STEPS[-1])
    step = max(s for s in STEPS if s <= max(STEPS[0], limit))
    result = dict(policy=policy, area_mm2=area, boundary_length_mm=perimeter,
                characteristic_width_mm=width, sample_step_mm=step,
                metric_scope='selected_canonical_mesh_not_certified_target_area',
                intra_face_sampling='equal_intervals_per_mesh_edge_v1',
                preserve_face_crossings=True, changes_row_spacing=False)
    if policy == CURVATURE_POLICY:
        result.update(shape_step_mm=step, sample_step_mm=max(s for s in STEPS if s<=min(5.,2*step)),
                      intra_face_sampling='shape_tangent_boundary_equal_intervals_v2',
                      minimum_step_mm=1., bend_thresholds_deg=[5.,15.],
                      bend_step_mm=[2.,1.], endpoint_band_mm=2., endpoint_step_mm=1.,
                      curvature_scope='discrete_scan_tangent_turn_not_physical_curvature')
    return result


def decision(mesh: Any, policy=POLICY) -> dict:
    """Use the selected region mesh in original face order, at PLY precision."""
    vertices = np.asarray(mesh.vertices).astype('<f4').astype('<f8')
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if (vertices.ndim != 2 or vertices.shape[1:] != (3,) or not np.isfinite(vertices).all()
            or faces.ndim != 2 or faces.shape[1:] != (3,) or not len(faces)
            or faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError('自动取点需要有效的选区网格')
    triangles = vertices[faces]
    area = float(np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0],
                                       triangles[:, 2]-triangles[:, 0]), axis=1).sum()/2*1e6)
    edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
    edges, counts = np.unique(edges, axis=0, return_counts=True)
    if np.any(counts > 2):raise ValueError('自动取点不接受非流形网格')
    boundary = edges[counts == 1]
    perimeter = float(np.linalg.norm(vertices[boundary[:, 1]]-vertices[boundary[:, 0]], axis=1).sum()*1000)
    return _record(area, perimeter, policy)


def validate(record: dict, mesh: Any = None) -> dict:
    if not isinstance(record, dict):raise ValueError('自动取点决策记录缺失')
    if (type(record.get('sample_step_mm')) not in (int, float)
            or type(record.get('characteristic_width_mm')) not in (int, float)
            or record.get('preserve_face_crossings') is not True
            or record.get('changes_row_spacing') is not False):
        raise ValueError('自动取点决策字段无效')
    expected = _record(record.get('area_mm2'), record.get('boundary_length_mm'),record.get('policy'))
    if record != expected:
        raise ValueError('自动取点决策或采样间隔不一致')
    if mesh is not None and record != decision(mesh,record['policy']):
        raise ValueError('自动取点决策与当前选区网格不一致')
    return record


def tangent_samples(points, faces, step_m, record):
    """Keep measured face crossings; densify bends and the two row endpoints.

    Each output edge remains a subedge of the original face-local scan line.
    The existing quality planner subsequently rechecks the whole edges.
    """
    validate(record)
    points=np.asarray(points,float)
    lengths=np.linalg.norm(np.diff(points,axis=0),axis=1)
    directions=np.diff(points,axis=0)/lengths[:,None]
    steps=np.full(len(lengths),step_m)
    if len(lengths)>1:
        angles=np.degrees(np.arccos(np.clip(np.einsum('ij,ij->i',directions[:-1],directions[1:]),-1.,1.)))
        # A small angular guard keeps threshold uncertainty on
        # the denser side, rather than silently treating a bend as flat.
        for threshold, mm in zip(record['bend_thresholds_deg'],record['bend_step_mm']):
            indices=np.flatnonzero(angles>=threshold-1e-10)
            steps[indices]=np.minimum(steps[indices],mm/1000)
            steps[indices+1]=np.minimum(steps[indices+1],mm/1000)
    cumulative=np.r_[0.,np.cumsum(lengths)];total=float(cumulative[-1]);band=record['endpoint_band_mm']/1000
    result=[]
    for i,length in enumerate(lengths):
        # Break only at physical endpoint-band limits, then sample each part.
        breaks=np.unique(np.clip([0.,length,band-cumulative[i],total-band-cumulative[i]],0.,length))
        for left,right in zip(breaks[:-1],breaks[1:]):
            midpoint=cumulative[i]+(left+right)/2
            step=min(steps[i],record['endpoint_step_mm']/1000) if midpoint<band or midpoint>total-band else steps[i]
            count=max(1,math.ceil(float(right-left)/step))
            for offset in left+(right-left)*np.arange(count,dtype=float)/count:
                fraction=offset/length
                result.append((points[i]*(1-fraction)+points[i+1]*fraction,int(faces[i])))
    result.append((points[-1],int(faces[-1])))
    return result
