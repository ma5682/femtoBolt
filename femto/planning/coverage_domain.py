"""Bound nominal surface cells and accepted scan endpoints for the P2 prototype."""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from typing import Any, Callable
from weakref import ref

import numpy as np

from femto.geometry import exact_surface as exact
from femto.planning.planning_revision import array_record, digest


def construction_cache(method: Callable, *, maxsize: int) -> Callable:
    """Bounded instance cache without retaining its owner in a reference cycle."""
    owner, function = ref(method.__self__), method.__func__

    @lru_cache(maxsize=maxsize)
    def cached(*args):
        current = owner()
        if current is None:raise ReferenceError('Coverage construction cache owner was released')
        return function(current, *args)

    return cached


class CoverageDomain:
    """Original cells are immutable; coverage subdivisions never replace intent."""
    def __init__(self, vertices: Any, faces: Any, source_faces: Any, areas: Any, quality: Any, binding: dict, *,
                 ledger: Any = None, revision: dict | None = None, representative_faces: Any = None) -> None:
        self.vertices = np.array(vertices, dtype='<f8', copy=True)
        self.faces = np.array(faces, dtype='<i8', copy=True)
        self.source_faces = np.array(source_faces, dtype='<i8', copy=True)
        self.areas = np.array(areas, dtype='<f8', copy=True)
        self.quality = np.array(quality, dtype='u1', copy=True)
        self.representative_faces = (np.arange(len(faces), dtype='<i8') if representative_faces is None
                                     else np.array(representative_faces, dtype='<i8', copy=True))
        self.binding = deepcopy(binding)
        self.ledger, self.revision = ledger, revision
        self._triangle = construction_cache(self._make_triangle, maxsize=4096)
        self._basis = construction_cache(self._make_basis, maxsize=4096)
        self._projector = construction_cache(self._make_projector, maxsize=4096)
        self._pixel = construction_cache(self._make_pixel, maxsize=2048)
        self._integer_pixel = construction_cache(self._make_integer_pixel, maxsize=2048)
        self._integer_point = construction_cache(self._make_integer_point, maxsize=8192)
        # Bounded, per-domain construction cache. Shared pixel/fan vertices
        # retain their exact values; no certificate or cross-revision reuse.
        self._point = construction_cache(self._make_point, maxsize=8192)
        self._parent_children = {}
        if ledger is not None:
            for face, pixel, child in zip(ledger.arrays['source_face'], ledger.arrays['pixel_index'], ledger.arrays['child_index']):
                self._parent_children.setdefault((int(face), int(pixel)), set()).add(int(child))
        for array in (self.vertices, self.faces, self.source_faces, self.areas, self.quality, self.representative_faces):
            array.setflags(write=False)

    def _make_triangle(self, face: int) -> exact.Triangle:
        return tuple(exact.rational(row) for row in self.vertices[self.faces[face]])

    def triangle(self, face: int) -> exact.Triangle:
        return self._triangle(int(face))

    def _make_basis(self, face: int):
        return exact.AffineTriangle(self.triangle(face))

    def _make_point(self, face: int, bary: exact.Bary):
        point = self._basis(face).xyz(bary)
        low, high = exact.point_bounds(point)
        low.setflags(write=False); high.setflags(write=False)
        return point, low, high

    def cell(self, face: int, bary: exact.Triangle):
        values = [self._point(int(face), point) for point in bary]
        return (tuple(value[0] for value in values),
                (np.asarray([value[1] for value in values]), np.asarray([value[2] for value in values])))

    def _make_integer_point(self, face, weights):
        value = exact.integer_xyz_bounds(self._basis(face), weights)
        value[1].setflags(write=False); value[2].setflags(write=False)
        return value

    def integer_cell(self, face, triangle):
        values = [self._integer_point(int(face), point) for point in triangle]
        return (exact.IntegerXYZTriangle(tuple(value[0] for value in values)),
                (np.asarray([value[1] for value in values]), np.asarray([value[2] for value in values])))

    def _make_projector(self, face: int):
        k = np.asarray(self.revision['inputs']['calibration']['intrinsic'], float)
        return exact.PixelClipper(self.triangle(face), k)

    def _make_pixel(self, face: int, pixel: int):
        width = self.ledger.arrays['mask'].shape[1]
        v, u = divmod(pixel, width)
        polygon = self._projector(face).clip(u, v)
        fans = {child-1:(polygon[0], polygon[child], polygon[child+1])
                for child in range(1, len(polygon)-1)
                if exact.orientation_sign(polygon[0], polygon[child], polygon[child+1]) != 0}
        # Different exact/float clipping topology is not silently re-associated.
        return fans if set(fans) == self._parent_children[(face, pixel)] else {}

    def root(self, index: int) -> exact.Triangle | None:
        if self.ledger is None:return exact.IDENTITY
        face, pixel = int(self.source_faces[index]), int(self.ledger.arrays['pixel_index'][index])
        return self._pixel(face, pixel).get(int(self.ledger.arrays['child_index'][index]))

    def _make_integer_pixel(self, face, pixel):
        v, u = divmod(pixel, self.ledger.arrays['mask'].shape[1])
        polygon = self._projector(face).clip_integer(u, v)
        fans = {i-1:(polygon[0], polygon[i], polygon[i+1]) for i in range(1, len(polygon)-1)
                if exact.integer_orientation(polygon[0], polygon[i], polygon[i+1]) != 0}
        return fans if set(fans) == self._parent_children[(face, pixel)] else {}

    def integer_root(self, index):
        if self.ledger is None:return ((1,0,0), (0,1,0), (0,0,1))
        face, pixel = int(self.source_faces[index]), int(self.ledger.arrays['pixel_index'][index])
        return self._integer_pixel(face, pixel).get(int(self.ledger.arrays['child_index'][index]))

    @classmethod
    def mesh(cls, vertices: Any, faces: Any) -> CoverageDomain:
        """An explicitly declared mesh-only numerical experiment, without RGB-D claims."""
        v = np.asarray(vertices, dtype=float)
        f = np.asarray(faces)
        if (v.ndim != 2 or v.shape[1:] != (3,) or not np.isfinite(v).all()
                or f.ndim != 2 or f.shape[1:] != (3,) or f.dtype.kind not in 'iu'
                or len(f) > 200_000 or (f.size and (f.min() < 0 or f.max() >= len(v)))):
            raise ValueError('覆盖实验网格形状、坐标或索引无效')
        v, f = v.astype('<f4').astype('<f8'), f.astype('<i8')
        if not np.isfinite(v).all():raise ValueError('规范网格坐标溢出')
        _, ids = np.unique(v, axis=0, return_inverse=True)
        _, first, inverse = np.unique(np.sort(ids[f], axis=1), axis=0, return_index=True, return_inverse=True)
        representative = first[inverse]
        first = np.sort(first)
        triangles = v[f[first]]
        areas = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]), axis=1)/2
        if not np.isfinite(areas).all() or np.any(areas <= 0):raise ValueError('覆盖实验拒绝退化或溢出的面片')
        binding = dict(scope='declared_nominal_mesh_only', vertices=array_record(v), faces=array_record(f),
                       source_faces=array_record(first.astype('<i8')), area=array_record(areas),
                       coordinate_frame='world_rdf', unit='metre',
                       unknown_pixel_count=0, original_target_complete=True,
                       source_duplicate_faces=len(f)-len(first), acquisition_geometry_verified=False)
        binding['sha256'] = digest(binding)
        return cls(v, f, first, areas, np.zeros(len(first), 'u1'), binding, representative_faces=representative)

    @classmethod
    def target(cls, ledger: Any, revision: dict, *, evidence: dict, effective_domain: dict | None,
               check: Callable[[], None] = lambda: None) -> CoverageDomain:
        from femto.planning import target_surface
        target_surface.validate(ledger, revision, evidence=evidence, domain=effective_domain, check=check, reproduce=True)
        return cls._checked_target(ledger, revision)

    @classmethod
    def _checked_target(cls, ledger, revision) -> CoverageDomain:
        arrays = ledger.arrays
        binding = dict(scope='p1_original_reference_selection', target_sha256=ledger.metadata['sha256'],
                       computation_sha256=revision['sha256'], target_id=revision['target_id'],
                       unknown_pixel_count=ledger.summary['unlocated_pixel_count'],
                       original_target_complete=ledger.metadata['metrics']['stop_reason'] == 'complete',
                       construction='exact_rational_reconstruction_of_source_pixel_fan',
                       area='p1_nominal_root_area_times_exact_dyadic_leaf_weight',
                       coordinate_frame='world_rdf', unit='metre')
        if ledger.metadata['schema']=='femto-original-target-surface-paged-v1':
            binding.update(target_construction_schema=ledger.metadata['schema'],
                target_page_units=[header['arrays']['area_m2']['shape'][0]
                                   for header in ledger.metadata['construction_pages']])
        binding['sha256'] = digest(binding)
        representative = np.arange(len(arrays['faces']), dtype='<i8')
        duplicates = arrays['duplicate_faces']
        if len(duplicates):representative[duplicates[:, 0]] = duplicates[:, 1]
        return cls(arrays['vertices'], arrays['faces'], arrays['source_face'], arrays['area_m2'],
                   arrays['quality_status'], binding, ledger=ledger, revision=deepcopy(revision),
                   representative_faces=representative)


def scan_endpoints(trajectory: Any, original_faces: np.ndarray) -> list[dict]:
    if getattr(trajectory, 'planner', {}).get('algorithm_version') == 'nominal-variable-width-refined-surface-v1':
        from femto.planning.variable_width_refinement import scan_endpoints as refined_endpoints
        return refined_endpoints(trajectory,original_faces)
    if getattr(trajectory, 'planner', {}).get('algorithm_version') == 'nominal-variable-width-surface-v1':
        from femto.planning.variable_width_route import scan_endpoints as variable_endpoints
        return variable_endpoints(trajectory, original_faces)
    from femto.planning.surface_trajectory import ScanSegment
    records = []
    seen = set()
    for item in trajectory.ordered_items:
        if not isinstance(item, ScanSegment):continue
        for index, (left, right) in enumerate(zip(item.points, item.points[1:])):
            role = right.get('role')
            if role not in (None, 'scan', 'connector'):raise ValueError('扫描边角色无效')
            # v3 omits role on ordinary scan samples; only an incoming connector
            # label disables action (same rule as surface_trajectory.role_lengths).
            if role == 'connector' or np.array_equal(left['xyz_m'], right['xyz_m']):continue
            for end, point in enumerate((left, right)):
                local = point['triangle_index']
                if type(local) is not int or not 0 <= local < len(original_faces):
                    raise ValueError('扫描点没有有效的原网格面映射')
                face = int(original_faces[local])
                key = (face, *point['xyz_m'])
                if key in seen:continue
                seen.add(key)
                records.append(dict(source_face=face, xyz_m=list(point['xyz_m']), role='scan',
                                    segment_index=item.segment_index, edge_index=index, endpoint=end))
    return records


def from_plan(model: Any, region: dict, plan: dict,
              check: Callable[[], None] = lambda: None) -> tuple[CoverageDomain, list[dict], dict, dict]:
    """Run the retained P1/v3 validators before an independent coverage experiment."""
    from femto.planning import planning_revision, planning_state, surface_workspace, footprint_model
    context = plan.get('planning_context')
    if context is None:raise ValueError('覆盖原型需要已验证的 P1 原目标台账，请生成新版本')
    planning_revision.require_current(model, region, plan['treatment_head'], context['revision'])
    trajectory = planning_state.validated_surface(plan, model.store.mesh, region, check=check)
    if trajectory.planner['algorithm_version'] not in ('quality-constrained-surface-v3', 'source-anchored-surface-v1', 'coverage-refined-surface-v1', 'constructed-surface-route-v1', 'constructed-coverage-refined-surface-v1', 'nominal-variable-width-surface-v1', 'nominal-variable-width-refined-surface-v1'):
        raise ValueError('独立覆盖原型当前只接受原 v3 路线')
    ledger = surface_workspace.target_for(model, plan, check, reproduce=True)
    domain = CoverageDomain._checked_target(ledger, context['revision'])
    seeds = scan_endpoints(trajectory, ledger.arrays['region_faces'])
    model_record = footprint_model.for_trajectory(trajectory)
    route = dict(schema='femto-accepted-v3-coverage-source-v1', route_record_sha256=digest(plan['surface_trajectory']),
                 result_sha256=trajectory.to_dict()['result_sha256'], computation_sha256=context['revision']['sha256'],
                 algorithm=trajectory.planner['algorithm_version'], accepted_scan_edges_only=True)
    if trajectory.planner['algorithm_version'] in ('source-anchored-surface-v1', 'coverage-refined-surface-v1', 'constructed-surface-route-v1', 'constructed-coverage-refined-surface-v1', 'nominal-variable-width-surface-v1', 'nominal-variable-width-refined-surface-v1'):
        from femto.planning.coverage_workflow import route_binding
        route = route_binding(trajectory)
    return domain, seeds, model_record, route
