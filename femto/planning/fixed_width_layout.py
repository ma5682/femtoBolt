"""Versioned 3-D row placement for a fixed nominal footprint.

Rows are proposals. The existing whole-edge width, residual, reference and mesh
predicates remain authoritative for both scans and links. A spacing bound is
not a coverage certificate, and a failed row is never silently widened.
"""
from __future__ import annotations

from copy import deepcopy
import math
import heapq
from numbers import Real
import numpy as np

VERSION = 'fixed-width-auto-layout-v1'
RESIDUAL_VERSION = 'fixed-width-auto-layout-residual-v2'
PATCH_VERSION = 'fixed-width-local-direction-patches-v1'
STABLE_PATCH_VERSION = 'fixed-width-local-direction-stable-v2'
REPAIR_PATCH_VERSION = 'fixed-width-local-direction-repair-v3'
OPTIMIZED_PATCH_VERSION = 'fixed-width-local-direction-budgeted-v4'
FOLDBACK_PATCH_VERSION = 'fixed-width-local-direction-foldback-v5'
ADAPTIVE_PATCH_VERSION = 'research-fixed-width-local-direction-adaptive-half-v6'
MINIMAL_PATCH_VERSION = 'research-fixed-width-local-direction-minimal-v7'
SHIFT_PATCH_VERSION = 'research-fixed-width-local-direction-bounded-shift-v8'
FLOOR_PATCH_VERSION = 'fixed-width-local-direction-spacing-floor-v9'
ADAPTIVE_FLOOR_PATCH_VERSION = 'research-fixed-width-local-direction-adaptive-metric-v10'
COORDINATED_FLOOR_PATCH_VERSION = 'research-fixed-width-local-direction-coordinated-seam-v11'
MODE = 'fixed_width_auto'
MAX_ROWS = 2000


def metric_positions(first, last, intervals, factors, maximum_m):
    """Fit rows in the integral of the local worst-face transverse metric.

    A steep face constrains only its transverse extent, not the entire fragment.
    This bounds the within-face gradient spacing, not global coverage or routes
    across folds; the existing support and coverage predicates remain required.
    """
    if last <= first:
        return np.array([first]), 0.
    intervals = np.asarray(intervals, dtype=float)
    factors = np.asarray(factors, dtype=float)
    if (intervals.shape != (len(factors), 2) or not np.isfinite(intervals).all()
            or not np.isfinite(factors).all() or np.any(factors < 0)
            or np.any(factors > 1) or maximum_m <= 0):
        raise ValueError('自动排布局部表面尺度无效')
    events = {}
    for index, ((low, high), factor) in enumerate(zip(intervals, factors)):
        low, high = max(first, low), min(last, high)
        if high <= low:
            continue
        if factor <= 1e-8:
            raise ValueError('局部表面需要另一扫描方向')
        events.setdefault(low, []).append((True, index))
        events.setdefault(high, []).append((False, index))
    levels = sorted({first, last, *events})
    active, queue, cumulative = set(), [], [0.]
    for left, right in zip(levels, levels[1:]):
        for entering, index in events.get(left, ()):
            if entering:
                active.add(index)
                heapq.heappush(queue, (factors[index], index))
            else:
                active.discard(index)
        while queue and queue[0][1] not in active:
            heapq.heappop(queue)
        if not queue:
            raise ValueError('自动排布横向范围存在无表面区间')
        cumulative.append(cumulative[-1]+(right-left)/queue[0][0])
    distance = cumulative[-1]
    count = max(1, math.ceil(distance/maximum_m-1e-12))
    if count+1 > MAX_ROWS:
        raise ValueError('固定宽度自动排布超过扫描行预算，请拆分区域')
    return np.interp(np.linspace(0., distance, count+1), cumulative, levels), distance/count


def request(width_mm, overlap_percent, *, version=VERSION, minimum_metric_fraction=None):
    from femto.planning.treatment_head import _number
    width = _number('effective_treatment_width_mm', width_mm)
    overlap = _number('overlap_percent', overlap_percent)
    if version not in (VERSION,RESIDUAL_VERSION,PATCH_VERSION,STABLE_PATCH_VERSION,REPAIR_PATCH_VERSION,OPTIMIZED_PATCH_VERSION,FOLDBACK_PATCH_VERSION,ADAPTIVE_PATCH_VERSION,MINIMAL_PATCH_VERSION,SHIFT_PATCH_VERSION,FLOOR_PATCH_VERSION,ADAPTIVE_FLOOR_PATCH_VERSION,COORDINATED_FLOOR_PATCH_VERSION):raise ValueError('固定宽度排布版本无效')
    result=dict(version=version, effective_width_mm=width, overlap_percent=overlap)
    if minimum_metric_fraction is not None:
        if (version!=FLOOR_PATCH_VERSION or isinstance(minimum_metric_fraction,bool)
                or not isinstance(minimum_metric_fraction,Real) or not np.isfinite(minimum_metric_fraction)
                or not .5<=minimum_metric_fraction<=1.):raise ValueError('尺度行距下限仅适用于对应排布且须在 0.5～1 之间')
        result['minimum_metric_fraction']=float(minimum_metric_fraction)
    return result


def row_positions(low_m, high_m, width_mm, overlap_percent, *, margin_mm=0., metric_factor=1.):
    """Fit both outer row centers, with the smallest count meeting the step bound."""
    spec = request(width_mm, overlap_percent)
    values = (low_m, high_m, margin_mm, metric_factor)
    if not all(np.isfinite(v) for v in values) or high_m < low_m or margin_mm < 0 or not 0 < metric_factor <= 1:
        raise ValueError('自动排布的三维范围、余量或表面尺度无效')
    inset = (spec['effective_width_mm']/2+margin_mm)/1000
    first, last = low_m+inset, high_m-inset
    maximum = spec['effective_width_mm']*(1-spec['overlap_percent']/100)*metric_factor/1000
    span = last-first
    if span < -1e-12:
        return np.empty(0), dict(status='narrower_than_fixed_width', nominal_inset_mm=inset*1000,
                                maximum_plane_step_mm=maximum*1000, row_count=0, actual_plane_step_mm=None)
    if span <= 1e-12:
        positions = np.array([(low_m+high_m)/2])
    else:
        intervals = max(1, math.ceil(span/maximum-1e-12))
        if intervals+1 > MAX_ROWS:
            raise ValueError('固定宽度自动排布超过扫描行预算，请拆分区域')
        positions = np.linspace(first, last, intervals+1)
    return positions, dict(status='proposed', nominal_inset_mm=inset*1000,
        maximum_plane_step_mm=maximum*1000, row_count=len(positions),
        actual_plane_step_mm=float(positions[1]-positions[0])*1000 if len(positions)>1 else None)


class Layout:
    def __init__(self, spec, evidence=None, *, row_shifts=None):
        self.spec = request(spec['effective_width_mm'], spec['overlap_percent'],version=spec['version'],
                            minimum_metric_fraction=spec.get('minimum_metric_fraction'))
        if spec != self.spec:raise ValueError('固定宽度排布版本或参数无效')
        self.fragments = []
        self.is_patch=spec['version'] in (PATCH_VERSION,STABLE_PATCH_VERSION,REPAIR_PATCH_VERSION,OPTIMIZED_PATCH_VERSION,FOLDBACK_PATCH_VERSION,ADAPTIVE_PATCH_VERSION,MINIMAL_PATCH_VERSION,SHIFT_PATCH_VERSION,FLOOR_PATCH_VERSION,ADAPTIVE_FLOOR_PATCH_VERSION,COORDINATED_FLOOR_PATCH_VERSION)
        self.is_stable=spec['version'] in (STABLE_PATCH_VERSION,REPAIR_PATCH_VERSION,OPTIMIZED_PATCH_VERSION,FOLDBACK_PATCH_VERSION,ADAPTIVE_PATCH_VERSION,MINIMAL_PATCH_VERSION,SHIFT_PATCH_VERSION,FLOOR_PATCH_VERSION,ADAPTIVE_FLOOR_PATCH_VERSION,COORDINATED_FLOOR_PATCH_VERSION)
        self.is_repair=spec['version'] in (REPAIR_PATCH_VERSION,OPTIMIZED_PATCH_VERSION,FOLDBACK_PATCH_VERSION,ADAPTIVE_PATCH_VERSION,MINIMAL_PATCH_VERSION,SHIFT_PATCH_VERSION,FLOOR_PATCH_VERSION,ADAPTIVE_FLOOR_PATCH_VERSION,COORDINATED_FLOOR_PATCH_VERSION)
        self.is_optimized=spec['version'] in (OPTIMIZED_PATCH_VERSION,FOLDBACK_PATCH_VERSION,ADAPTIVE_PATCH_VERSION,MINIMAL_PATCH_VERSION,SHIFT_PATCH_VERSION,FLOOR_PATCH_VERSION,ADAPTIVE_FLOOR_PATCH_VERSION,COORDINATED_FLOOR_PATCH_VERSION)
        self.is_foldback=spec['version']==FOLDBACK_PATCH_VERSION
        self.is_adaptive=spec['version'] in (ADAPTIVE_PATCH_VERSION,MINIMAL_PATCH_VERSION)
        self.is_minimal=spec['version']==MINIMAL_PATCH_VERSION
        self.is_spacing_floor=spec['version'] in (FLOOR_PATCH_VERSION,ADAPTIVE_FLOOR_PATCH_VERSION,COORDINATED_FLOOR_PATCH_VERSION)
        self.is_adaptive_floor=spec['version'] in (ADAPTIVE_FLOOR_PATCH_VERSION,COORDINATED_FLOOR_PATCH_VERSION)
        self.is_coordinated_floor=spec['version']==COORDINATED_FLOOR_PATCH_VERSION
        self.minimum_metric_fraction=self.spec.get('minimum_metric_fraction',.75)
        if row_shifts is None:row_shifts=[]
        if not isinstance(row_shifts,list) or (row_shifts and not self.is_repair) or len(row_shifts)>4:
            raise ValueError('局部行调整版本或数量无效')
        keys={'block_index','fragment_face_sha256','patch_seed_face','position_m','shift_m'}
        seen=set()
        for item in row_shifts:
            if (not isinstance(item,dict) or set(item)!=keys or type(item['block_index']) is not int or item['block_index']<0
                    or type(item['patch_seed_face']) is not int or item['patch_seed_face']<0
                    or not isinstance(item['fragment_face_sha256'],str) or len(item['fragment_face_sha256'])!=64
                    or any(ch not in '0123456789abcdef' for ch in item['fragment_face_sha256'])
                    or any(isinstance(item[k],bool) or not isinstance(item[k],Real)
                           for k in ('position_m','shift_m'))
                    or not np.isfinite([item['position_m'],item['shift_m']]).all()
                    or (not 0 < abs(item['shift_m']) <= self.spec['effective_width_mm']*.05/1000 + 1e-12
                        if spec['version']==SHIFT_PATCH_VERSION else
                        abs(abs(item['shift_m'])-self.spec['effective_width_mm']*.05/1000)>1e-12)):
                raise ValueError('局部行调整记录无效')
            key=(item['block_index'],item['fragment_face_sha256'],item['patch_seed_face'],item['position_m'])
            if key in seen:raise ValueError('局部行调整重复')
            seen.add(key)
        self.row_shifts=deepcopy(row_shifts);self.applied_shifts=[]
        self.evidence=deepcopy(evidence) if self.is_stable else None
        self.reference=None
        if spec['version'] in (RESIDUAL_VERSION,PATCH_VERSION,STABLE_PATCH_VERSION,REPAIR_PATCH_VERSION,OPTIMIZED_PATCH_VERSION,FOLDBACK_PATCH_VERSION,ADAPTIVE_PATCH_VERSION,MINIMAL_PATCH_VERSION,SHIFT_PATCH_VERSION,FLOOR_PATCH_VERSION,ADAPTIVE_FLOOR_PATCH_VERSION,COORDINATED_FLOOR_PATCH_VERSION):
            if evidence is None:raise ValueError('自动排布缺少局部残差证据')
            from femto.planning.quality_constrained import ReferenceEnvelope
            self.reference=ReferenceEnvelope(evidence,spec['effective_width_mm'])

    def positions(self, vertices, faces, ids, origin, axis, block_index, row_offset, check):
        from femto.planning.surface_trajectory import face_list_sha256
        check()
        triangle = vertices[faces[ids]]
        normals = np.cross(triangle[:,1]-triangle[:,0], triangle[:,2]-triangle[:,0])
        norm = np.linalg.norm(normals, axis=1)
        valid = norm > 0
        if not valid.any():raise ValueError('自动排布没有有效三角面')
        normals = normals[valid]/norm[valid,None]
        # |grad_surface(x dot axis)| gives the within-face conversion from a
        # slice-plane interval to distance on that measured triangle.
        factors = np.sqrt(np.maximum(0., 1-(normals@axis)**2))
        factor = float(factors.min())
        coordinates = (vertices[np.unique(faces[ids])]-origin)@axis
        low, high = float(coordinates.min()), float(coordinates.max())
        margins=[0.,0.]
        if self.reference is not None:
            points=vertices[np.unique(faces[ids])]
            strip=self.spec['effective_width_mm']/2000+self.reference.global_residual
            # Probe the original local residual rule near each transverse edge.
            # These are placement proposals; whole-edge acceptance still checks
            # the actual scan and connector locations independently afterwards.
            for side,mask in enumerate((coordinates<=low+strip,coordinates>=high-strip)):
                for point in points[mask]:
                    check()
                    _,margin,_=self.reference.probe(point,0.)
                    margins[side]=max(margins[side],margin)
        shifted_low,shifted_high=low+margins[0],high-margins[1]
        if shifted_high<shifted_low:
            shifted_low=shifted_high=(shifted_low+shifted_high)/2
        positions, decision = row_positions(shifted_low, shifted_high, self.spec['effective_width_mm'], self.spec['overlap_percent'])
        if self.reference is not None:
            decision.update(boundary_residual_margins_mm=[m*1000 for m in margins],
                            center_insets_mm=[self.spec['effective_width_mm']/2+m*1000 for m in margins])
        if len(positions)>1:
            triangle_levels = (triangle[valid]-origin)@axis
            intervals = np.column_stack((triangle_levels.min(axis=1), triangle_levels.max(axis=1)))
            positions, metric_step = metric_positions(positions[0], positions[-1], intervals, factors,
                                                     decision['maximum_plane_step_mm']/1000)
            steps = np.diff(positions)*1000
            decision.update(row_count=len(positions), actual_plane_step_mm=float(steps.max()),
                            minimum_plane_step_mm=float(steps.min()), transverse_metric_step_mm=metric_step*1000)
        self.fragments.append(dict(block_index=int(block_index), first_row_index=int(row_offset),
            face_indices_sha256=face_list_sha256(ids), face_count=len(ids),
            origin_m=origin.tolist(), transverse_axis=axis.tolist(), bounds_m=[low, high],
            minimum_surface_metric_factor=factor, positions_m=positions.tolist(),
            local_residual_rule='original_whole_edge_width_plus_local_reference_residual', **decision))
        return positions

    def record(self):
        if self.is_repair and sorted(self.applied_shifts)!=list(range(len(self.row_shifts))):
            raise ValueError('局部行调整与选区片段不匹配')
        result=dict(request=deepcopy(self.spec), fragments=deepcopy(self.fragments),
            placement_rule=('local_direction_patches_with_artificial_seam_centers' if self.is_patch else
                            'three_dimensional_local_metric_fixed_width_inset_endpoint_fitted_rows'),
            coverage_claim='none_from_row_placement', width_is_fixed=True)
        if self.is_repair:result['row_shifts']=deepcopy(self.row_shifts)
        return result


def retained_steps(record, trajectory):
    """Measured slice-plane distances of retained rows, not coverage proofs."""
    groups={}
    fragments=record['fragments']
    if record.get('request',{}).get('version') in (PATCH_VERSION,STABLE_PATCH_VERSION,REPAIR_PATCH_VERSION,OPTIMIZED_PATCH_VERSION,FOLDBACK_PATCH_VERSION,ADAPTIVE_PATCH_VERSION,MINIMAL_PATCH_VERSION,SHIFT_PATCH_VERSION,FLOOR_PATCH_VERSION,ADAPTIVE_FLOOR_PATCH_VERSION,COORDINATED_FLOOR_PATCH_VERSION):
        fragments=[dict(block_index=f['block_index'],first_row_index=p['first_row_index'],
            row_count=p['retained_row_count'],origin_m=p['frame']['origin'],transverse_axis=p['frame']['e2'])
            for f in fragments for p in f['patches'] if p.get('frame') is not None]
    for item in trajectory['ordered_items']:
        if item['kind']!='scan_segment':continue
        for point in item['points']:
            if point.get('role','scan')!='scan':continue
            row=point['row_index']
            candidates=[(i,r) for i,r in enumerate(fragments) if r['row_count'] and
                        r['block_index']==item['block_index'] and r['first_row_index']<=row]
            if not candidates:continue
            index,fragment=max(candidates,key=lambda pair:pair[1]['first_row_index'])
            position=float((np.asarray(point['xyz_m'])-fragment['origin_m'])@np.asarray(fragment['transverse_axis']))
            groups.setdefault(index,{}).setdefault(row,[]).append(position)
    return [float(value)*1000 for rows in groups.values()
            for value in np.diff(sorted(float(np.mean(points)) for points in rows.values()))]


def summary(record, actual_rows=None, trajectory=None):
    rows = record['fragments']
    steps = [r['actual_plane_step_mm'] for r in rows if r['actual_plane_step_mm'] is not None]
    minima = [r.get('minimum_plane_step_mm',r['actual_plane_step_mm']) for r in rows if r['actual_plane_step_mm'] is not None]
    spacing = ('单行' if not steps else f"{min(minima):.2f}～{max(steps):.2f} mm")
    proposed=sum(r['row_count'] for r in rows)
    retained='' if actual_rows is None else f'，核验后保留 {actual_rows} 行'
    actual=retained_steps(record,trajectory) if trajectory is not None else []
    measured=(f'保留行横向切片距离 {min(actual):.2f}～{max(actual):.2f} mm；' if actual else '')
    text=(f"固定作用宽度 {record['request']['effective_width_mm']:g} mm；候选排布 {proposed} 行{retained}，"
          f"候选切片间距 {spacing}。"+measured+'曲面距离及覆盖仍需核验；裁剪或缺行处不保证覆盖。')
    if record['request'].get('version') in (OPTIMIZED_PATCH_VERSION,FOLDBACK_PATCH_VERSION,ADAPTIVE_PATCH_VERSION,MINIMAL_PATCH_VERSION,SHIFT_PATCH_VERSION,FLOOR_PATCH_VERSION,ADAPTIVE_FLOOR_PATCH_VERSION,COORDINATED_FLOOR_PATCH_VERSION):
        fallback=sum(p.get('alignment_optimization',{}).get('strategy')=='prior_greedy_budget_fallback'
                     for fragment in rows for p in fragment.get('patches',[]))
        if fallback:text+=f' {fallback} 个分区因计算预算采用原行位置选择。'
    return text


def validate(record, mesh, region, parameters, check):
    """Independently recompute saved placement from the measured fragment faces."""
    if record['request']['version'] in (PATCH_VERSION,STABLE_PATCH_VERSION,REPAIR_PATCH_VERSION,OPTIMIZED_PATCH_VERSION,FOLDBACK_PATCH_VERSION,ADAPTIVE_PATCH_VERSION,MINIMAL_PATCH_VERSION,SHIFT_PATCH_VERSION,FLOOR_PATCH_VERSION,ADAPTIVE_FLOOR_PATCH_VERSION,COORDINATED_FLOOR_PATCH_VERSION):
        from femto.planning.patch_slice_layout import validate as validate_patches
        return validate_patches(record,mesh,region,parameters,check)
    import trimesh
    from femto.planning.selection_blocks import selection_face_assignment, _reading_key
    from femto.planning.surface_trajectory import face_components, face_list_sha256
    from femto.planning.mesh_slice_planner import _pca_frame, _outer_loop, Unplannable
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    intent, _, _, owners = selection_face_assignment(region, vertices, faces)
    components = face_components(faces)
    blocks = sorted(intent['components'], key=lambda item:_reading_key(
        item['bbox'],item['component_id'],parameters['start_corner']))
    lookup={}
    for block_index, block in enumerate(blocks):
        ids=np.flatnonzero(owners==block['component_id'])
        for component in np.unique(components[ids]):
            fragment=ids[components[ids]==component]
            lookup[(block_index,face_list_sha256(fragment))]=fragment
    fresh=Layout(record['request'],parameters.get('quality_evidence'));seen=set()
    for saved in record['fragments']:
        check()
        key=(saved['block_index'],saved['face_indices_sha256'])
        if key not in lookup or key in seen:
            raise ValueError('固定宽度排布表面片段不一致')
        seen.add(key);ids=lookup[key]
        origin, _, axis, _, _ = _pca_frame(trimesh.Trimesh(vertices=vertices,faces=faces[ids],process=False),parameters)
        fresh.positions(vertices,faces,ids,origin,axis,saved['block_index'],saved['first_row_index'],check)
    for key in lookup.keys()-seen:
        check();ids=lookup[key]
        # The candidate may reject a degenerate fragment before layout begins.
        # Reproduce those exact preparatory checks, rather than accepting an
        # arbitrary omitted fragment merely because it has no saved record.
        try:
            origin,e1,e2,_,_=_pca_frame(trimesh.Trimesh(vertices=vertices,faces=faces[ids],process=False),parameters)
            _outer_loop(vertices,faces[ids],origin,e1,e2)
        except Unplannable:
            continue
        raise ValueError('固定宽度排布缺少表面片段')
    if fresh.record()!=record:
        raise ValueError('固定宽度排布决策与实测表面不一致')
