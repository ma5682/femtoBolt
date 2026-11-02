"""Nearest forward ray/surface intersection with explicit observation evidence.

Float64 Möller–Trumbore: 1e-9 m distance tolerance, 1e-10 barycentric tolerance.
Back faces block rather than reveal a deeper surface. Shared edges/vertices
coalesce; any unsupported coincident/closer hit makes the result unavailable.
Degenerate triangles are skipped and counted. This is not a collision model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SurfaceEvidence:
    vertices: tuple
    faces: tuple
    confidence: tuple
    view_support: tuple
    source: str
    frame: str
    unit: str = 'metre'
    known_occlusion: bool | None = None

    def __post_init__(self):
        vertices = np.asarray(self.vertices, dtype=np.float64)
        faces = np.asarray(self.faces)
        if (vertices.ndim != 2 or vertices.shape[1:] != (3,) or not np.isfinite(vertices).all()
                or faces.ndim != 2 or faces.shape[1:] != (3,) or faces.dtype.kind not in 'iu'
                or (faces.size and (faces.min() < 0 or faces.max() >= len(vertices)))
                or len(self.confidence) != len(faces) or len(self.view_support) != len(faces)
                or self.unit != 'metre' or not self.frame or not self.source):
            raise ValueError('Invalid local surface frame/unit/support data')
        if any(type(v) not in (int, float) or v not in (0, .3, 1) for v in self.confidence):
            raise ValueError('Confidence must retain original 1, repaired .3 or unknown 0 semantics')
        if self.known_occlusion is not None and type(self.known_occlusion) is not bool:
            raise ValueError('Occlusion must be explicit or unknown')
        if (not isinstance(self.frame, str) or not isinstance(self.source, str)
                or any(isinstance(row, (str, bytes)) or any(not isinstance(v, str) or not v.strip() for v in row)
                       for row in self.view_support)):
            raise ValueError('View support requires explicit observation IDs')
        object.__setattr__(self, 'vertices', tuple(tuple(float(v) for v in row) for row in vertices))
        object.__setattr__(self, 'faces', tuple(tuple(int(v) for v in row) for row in faces))
        object.__setattr__(self, 'confidence', tuple(self.confidence))
        object.__setattr__(self, 'view_support', tuple(tuple(v) for v in self.view_support))


def unavailable(reason, *, source=None, frame=None, mode='historical_offline', **details):
    return dict(schema='femto-axial-gap-v1', available=False, reason=reason, distance_m=None,
                intersection_m=None, frame=frame, unit='metre', source=source, mode=mode,
                execution_ready=False, physical_accuracy_verified=False, **details)


def axial_gap(surface, origin, axis, *, frame, mode='historical_offline',
              calibration_applicable=False, surface_fresh=False, state_applicable=False):
    if mode not in ('historical_offline', 'synthetic_test', 'realtime'):
        raise ValueError('Explicit measurement mode is required')
    if not isinstance(surface, SurfaceEvidence):
        return unavailable('surface_missing', frame=frame, mode=mode)
    fail = lambda reason, **extra: unavailable(reason, source=surface.source, frame=frame, mode=mode, **extra)
    if frame != surface.frame or surface.unit != 'metre':
        return fail('frame_or_unit_mismatch')
    if calibration_applicable is not True:
        return fail('calibration_unavailable')
    if mode == 'realtime' and (surface_fresh is not True or state_applicable is not True):
        return fail('surface_stale_or_state_invalid')
    if surface.known_occlusion is not False:
        return fail('known_occlusion' if surface.known_occlusion else 'occlusion_evidence_missing')
    try:
        origin, axis = np.asarray(origin, dtype=np.float64), np.asarray(axis, dtype=np.float64)
    except (TypeError, ValueError):
        return fail('invalid_tool_axis_or_origin')
    if origin.shape != (3,) or axis.shape != (3,) or not np.isfinite([origin, axis]).all():
        return fail('invalid_tool_axis_or_origin')
    length = float(np.linalg.norm(axis))
    if not np.isfinite(length) or length <= 1e-15:
        return fail('zero_axis')
    axis = axis/length
    vertices = np.asarray(surface.vertices)
    hits, degenerate = [], 0
    for i, face in enumerate(surface.faces):
        a, b, c = vertices[list(face)]
        e1, e2 = b-a, c-a
        normal = np.cross(e1, e2)
        area2 = float(np.linalg.norm(normal))
        if area2 <= 1e-18:
            degenerate += 1
            continue
        p = np.cross(axis, e2); determinant = float(e1 @ p)
        if abs(determinant) <= area2*1e-12:
            continue
        s = origin-a; u = float(s @ p)/determinant
        q = np.cross(s, e1); v = float(axis @ q)/determinant
        distance = float(e2 @ q)/determinant
        if u < -1e-10 or v < -1e-10 or u+v > 1+1e-10 or distance < -1e-9:
            continue
        hits.append((max(0., distance), i, determinant > 0))
    if not hits:
        return fail('no_forward_intersection', degenerate_faces=degenerate)
    hits.sort()
    distance = hits[0][0]
    nearest = [hit for hit in hits if hit[0] <= distance+1e-9]
    if any(surface.confidence[i] != 1 or not surface.view_support[i] for _, i, _ in nearest):
        return fail('nearest_surface_not_measured_supported', degenerate_faces=degenerate)
    if any(not front for _, _, front in nearest):
        return fail('backface_hit', degenerate_faces=degenerate)
    return dict(schema='femto-axial-gap-v1', available=True, reason=None, distance_m=distance,
                intersection_m=(origin+axis*distance).tolist(), frame=frame, unit='metre',
                source=surface.source, mode=mode, axis=axis.tolist(),
                face_indices=[i for _, i, _ in nearest], degenerate_faces=degenerate,
                intersection_count=len(hits), distance_tolerance_m=1e-9,
                execution_ready=False, physical_accuracy_verified=False)
