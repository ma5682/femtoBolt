"""Explicit measured surface domain and missing-intent inventory; no new XYZ."""
from __future__ import annotations

import hashlib
from typing import Any, Callable

import numpy as np

from femto.annotation.selection_intent import decode_runs, encode_runs
from femto.planning.surface_trajectory import canonical_json_bytes

SCHEMA = 'femto-effective-surface-domain-v1'
RULE = 'whole_projected_face_bounding_rectangle_has_consistent_selected_reference_pixels'
FACE_CLASSES = {0:'quality_supported', 1:'outside_reference_image',
                2:'touches_unselected_or_quality_pending_pixels', 3:'invalid_face_geometry'}


def mesh_hash(mesh: Any) -> str:
    digest=hashlib.sha256()
    for array,dtype in ((mesh.vertices,'<f8'),(mesh.faces,'<i8')):
        value=np.asarray(array,dtype=dtype)
        digest.update(np.asarray(value.shape,dtype='<i8').tobytes());digest.update(value.tobytes())
    return digest.hexdigest()


def missing_regions(classes: np.ndarray, check: Callable = lambda: None) -> list[dict]:
    """Separate each reason spatially; missing intent never becomes unselected."""
    from femto.annotation.reference_quality import LABELS
    from femto.planning.quality_components import component_rows
    records=[]
    with component_rows(classes, check) as batches:
        for code,count,stats,components in batches:
            check()
            # The same aggregate budget and raster numbering are evaluated in
            # class order, regardless of which worker finishes first.
            if len(records)+count>4096:raise ValueError('缺失范围过于碎片化，超过报告预算；请缩小选区')
            for index,component in enumerate(components):
                if index%64==0:check()
                x,y,w,h,pixels=map(int,stats[component])
                records.append(dict(id=len(records), quality_class=code, reason=LABELS[code],
                    pixel_count=pixels, pixel_bounds=[x,y,x+w-1,y+h-1],
                    action='inspect_reference_and_registration_then_rescan' if code==5 else 'inspect_or_rescan',
                    xyz_available=False, keeps_coverage_requirement=True))
    return records


def build(mesh: Any, region: dict, evidence: dict, width_mm: float,
          check: Callable = lambda: None, *, _classes: np.ndarray | None = None,
          _derived: dict | None = None) -> dict:
    from femto.planning.quality_constrained import checked_evidence, _hash
    from femto.planning.planning_state import region_signature
    # _classes: this exact evidence/region pair, already checked by the same call.
    # _derived: _class_records of that same _classes array, from the same call.
    classes=checked_evidence(evidence,region)[0] if _classes is None else _classes
    if isinstance(width_mm,bool) or not np.isfinite(width_mm) or not .5<=width_mm<=100:
        raise ValueError('有效作用宽度无效')
    vertices=np.asarray(mesh.vertices,float);faces=np.asarray(mesh.faces,np.int64)
    if not len(faces) or len(faces)!=len(region['triangle_indices']):
        raise ValueError('有效区域必须绑定完整原选区的网格面索引')
    check()
    k=np.asarray(evidence['view_reference']['undistorted_intrinsic'],float)
    corners=vertices[faces]
    area=.5*np.linalg.norm(np.cross(corners[:,1]-corners[:,0],corners[:,2]-corners[:,0]),axis=1)
    valid=np.isfinite(corners).all(axis=(1,2)) & (corners[:,:,2]>0).all(axis=1) & (area>0)
    status=np.full(len(faces),3,np.uint8)
    from femto.planning.task_preparation import integral_image
    integral=integral_image(classes)
    h,w=classes.shape
    # All projected points of a positive-depth triangle lie within the bbox of
    # its projected vertices. Passing this conservative bbox certifies its
    # reference-pixel support; it does not assert physical surface accuracy.
    for start in range(0,len(faces),16384):
        check()
        ids=np.flatnonzero(valid[start:start+16384])+start
        xyz=corners[ids]
        uv=xyz[:,:,:2]/xyz[:,:,2:3]*[k[0,0],k[1,1]]+k[:2,2]
        lo=np.floor(uv.min(axis=1)+.5).astype(np.int64)
        hi=np.floor(uv.max(axis=1)+.5).astype(np.int64)
        inside=(lo[:,0]>=0)&(lo[:,1]>=0)&(hi[:,0]<w)&(hi[:,1]<h)
        status[ids]=1
        ids=ids[inside];lo=lo[inside];hi=hi[inside]
        bad=integral[hi[:,1]+1,hi[:,0]+1]-integral[lo[:,1],hi[:,0]+1]-integral[hi[:,1]+1,lo[:,0]]+integral[lo[:,1],lo[:,0]]
        status[ids]=np.where(bad==0,0,2)
    eligible=status==0
    selected=classes>0;pending=selected&(classes!=1)
    derived=_class_records(classes,check) if _derived is None else _derived
    record=dict(schema=SCHEMA, rule=RULE, source_region_signature=region_signature(region),
        source_mesh_sha256=mesh_hash(mesh), quality_evidence_sha256=evidence['sha256'],
        image_size=evidence['image_size'], coordinate_frame='world_rdf', unit='metre',
        face_index_space='canonical_original_region_mesh', face_count=len(faces),
        source_triangle_indices=list(region['triangle_indices']), face_class_runs=encode_runs(status),
        face_class_names={str(k):v for k,v in FACE_CLASSES.items()},
        eligible_face_count=int(eligible.sum()), excluded_face_count=int((~eligible).sum()),
        eligible_area_m2=float(area[eligible].sum()), excluded_area_m2=float(area[~eligible & valid].sum()),
        quality_supported_mask_runs=derived['quality_supported_mask_runs'], pending_mask_runs=derived['pending_mask_runs'],
        selected_pixel_count=int(selected.sum()), quality_supported_pixel_count=int((classes==1).sum()),
        pending_pixel_count=int(pending.sum()), missing_regions=derived['missing_regions'],
        effective_width_mm=float(width_mm),
        center_domain_definition='eligible_mesh_intersect_whole_chord_width_plus_local_residual_clearance_predicate',
        center_domain_representation='implicit_predicate_not_a_pre_eroded_mesh',
        state='available_for_constraint_checks' if eligible.any() else 'blocked_no_supported_faces',
        target_completion='unverified', execution_ready=False, physical_accuracy_verified=False)
    record['sha256']=_hash(record)
    return record


def validate(domain: dict, evidence: dict, width_mm: float, mesh: Any = None,
             region: dict | None = None, check: Callable = lambda: None, *,
             _classes: np.ndarray | None = None) -> np.ndarray:
    from femto.planning.quality_constrained import checked_evidence, _hash
    classes=checked_evidence(evidence,region)[0] if _classes is None else _classes
    if (domain.get('schema')!=SCHEMA or domain.get('sha256')!=_hash(domain)
            or domain.get('quality_evidence_sha256')!=evidence['sha256']
            or domain.get('effective_width_mm')!=width_mm or domain.get('rule')!=RULE
            or domain.get('target_completion')!='unverified'
            or domain.get('execution_ready') is not False or domain.get('physical_accuracy_verified') is not False):
        raise ValueError('有效区域的来源、参数或状态不一致')
    status=decode_runs(domain['face_class_runs'],[domain['face_count'],1],values=FACE_CLASSES).ravel()
    eligible=status==0
    # Same comparisons in the original short-circuit order; the records are
    # kept so the mesh rebuild below does not recompute them from classes.
    derived={}
    if (domain['eligible_face_count']!=int(eligible.sum()) or domain['excluded_face_count']!=int((~eligible).sum())
            or domain['quality_supported_mask_runs']!=derived.setdefault('quality_supported_mask_runs',encode_runs(classes==1))
            or domain['pending_mask_runs']!=derived.setdefault('pending_mask_runs',encode_runs((classes>0)&(classes!=1)))
            or domain['missing_regions']!=derived.setdefault('missing_regions',missing_regions(classes,check))):
        raise ValueError('有效区域或缺失范围不能由质量证据重现')
    if mesh is not None:
        if region is None:raise ValueError('核验有效区域网格时需要原选择意图')
        if build(mesh,region,evidence,width_mm,check,_classes=classes,_derived=derived)!=domain:
            raise ValueError('有效区域不能由原网格和质量证据重建')
    return eligible


def _class_records(classes: np.ndarray, check: Callable) -> dict:
    """Records that depend only on the checked class image.

    Computed directly: build() runs once per operation without _derived, and
    validate() passes its own records, so a cache here would never be hit.
    """
    return dict(quality_supported_mask_runs=encode_runs(classes==1),
                pending_mask_runs=encode_runs((classes>0)&(classes!=1)),
                missing_regions=missing_regions(classes,check))
