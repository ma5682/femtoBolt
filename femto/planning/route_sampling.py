"""Versioned sampling choice from the selected canonical mesh, never row spacing.

This is a planning workload heuristic, not a physical accuracy certificate.
Triangle crossings and the existing whole-edge predicates remain authoritative.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

POLICY = 'selection-area-boundary-sampling-v1'
STEPS = (1., 2., 3., 5.)


def _record(area: float, perimeter: float) -> dict:
    if (type(area) not in (int, float) or type(perimeter) not in (int, float)
            or not math.isfinite(area) or not math.isfinite(perimeter)
            or area <= 0 or perimeter < 0):
        raise ValueError('自动取点的选区面积或边界长度无效')
    # 2A/P penalizes thin, perforated and fragmented selections. A closed
    # surface has no boundary; its area scale alone supplies the upper bound.
    width = 2*area/perimeter if perimeter else math.sqrt(area)
    limit = min(math.sqrt(area)/12, width/4, STEPS[-1])
    step = max(s for s in STEPS if s <= max(STEPS[0], limit))
    return dict(policy=POLICY, area_mm2=area, boundary_length_mm=perimeter,
                characteristic_width_mm=width, sample_step_mm=step,
                metric_scope='selected_canonical_mesh_not_certified_target_area',
                intra_face_sampling='equal_intervals_per_mesh_edge_v1',
                preserve_face_crossings=True, changes_row_spacing=False)


def decision(mesh: Any) -> dict:
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
    return _record(area, perimeter)


def validate(record: dict, mesh: Any = None) -> dict:
    if not isinstance(record, dict):raise ValueError('自动取点决策记录缺失')
    if (type(record.get('sample_step_mm')) not in (int, float)
            or type(record.get('characteristic_width_mm')) not in (int, float)
            or record.get('preserve_face_crossings') is not True
            or record.get('changes_row_spacing') is not False):
        raise ValueError('自动取点决策字段无效')
    expected = _record(record.get('area_mm2'), record.get('boundary_length_mm'))
    if record != expected:
        raise ValueError('自动取点决策或采样间隔不一致')
    if mesh is not None and record != decision(mesh):
        raise ValueError('自动取点决策与当前选区网格不一致')
    return record
