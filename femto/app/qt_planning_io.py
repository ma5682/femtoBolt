"""Qt-facing planning adapters. Geometry and saved-path checks stay in the backend."""
from __future__ import annotations

from copy import copy, deepcopy
import json
from pathlib import Path

import numpy as np


def clone_model(model, *, freeze_inputs=False):
    candidate=copy(model)
    candidate.store=copy(model.store)
    for name in ('points','regions','_history'):
        setattr(candidate.store,name,deepcopy(getattr(model.store,name)))
    candidate._overlay_cache=dict(model._overlay_cache)
    if freeze_inputs:
        # Workers receive a stable input snapshot. In-place changes in the live
        # model cannot mix old RGB/depth with new geometry halfway through work.
        candidate.mesh=deepcopy(model.mesh);candidate.store.mesh=candidate.mesh
        candidate.projection=copy(model.projection);candidate.projection.mesh=candidate.mesh
        for name in ('depth','image_valid','ids','z','reasons','valid','new_support','k'):
            if hasattr(model.projection,name):
                value=np.array(getattr(model.projection,name),copy=True);value.setflags(write=False)
                setattr(candidate.projection,name,value)
        candidate.rgb=np.array(model.rgb,copy=True);candidate.rgb.setflags(write=False)
        candidate.view_reference=deepcopy(model.view_reference)
        candidate.store.origin_reference=deepcopy(model.store.origin_reference)
        candidate.store.local_context=deepcopy(model.store.local_context)
        for name in ('task_context','planning_quality_config'):
            if hasattr(model,name):setattr(candidate,name,deepcopy(getattr(model,name)))
    return candidate


def masks_from_region(region, shape):
    from femto.annotation.selection_intent import intent_mask, decode_runs
    from femto.annotation.square_brush import polygon_mask
    size=[shape[1],shape[0]]
    if region.get('selection_intent'):
        intent=region['selection_intent']
        selected=intent_mask(intent).astype(np.uint8)
        runs=intent.get('painted_forbidden_runs')
        forbidden=np.zeros(shape,np.uint8) if runs is None else decode_runs(runs,size,values=(0,1)).astype(np.uint8)
        # Retain the two masks independently.  ``selected_mask_runs`` is the
        # physician's requested coverage after exclusions; forbidden pixels
        # are recorded separately so reopening a project never paints an
        # exclusion back into the work area.  For full-minus-forbidden the
        # selected mask is already the complement of the painted exclusion.
        work=selected
        return work,forbidden
    work=polygon_mask(region.get('work_screen_polygon') or region.get('screen_polygon',[]),size)
    forbidden=polygon_mask(region.get('forbidden_screen_polygon',[]),size)
    return work,forbidden


def extract_region(model, work, forbidden, name, note, cancel):
    from femto.annotation.fixed_rgb_geometry import MAX_POLYGON_CORNERS, GeometryBudget
    from femto.annotation.square_brush import mask_polygon
    check=GeometryBudget(cancel).check
    check()
    work=work.astype(bool);forbidden=forbidden.astype(bool)
    if not work.any() and not forbidden.any():raise ValueError('请先涂抹工作区域或禁止区域。')
    method=('work_minus_forbidden' if forbidden.any() else 'work_only') if work.any() else 'full_minus_forbidden'
    selected=(work.copy() if work.any() else np.ones(work.shape,bool)) & ~forbidden
    if not selected.any():raise ValueError('禁止区域覆盖了全部范围，请调整标记。')
    polygon=mask_polygon(selected,MAX_POLYGON_CORNERS)
    candidate=clone_model(model)
    item=candidate.region(polygon,name,note,cancel=cancel,replace_existing=True,
        selection=dict(mask=selected,forbidden_mask=forbidden if forbidden.any() else None,method=method))
    if item is None:raise ValueError('当前选择没有有效三维表面，请调整区域或返回补采。')
    item['selection_method']=method
    check()
    return candidate,item


from femto.planning.saved_source import load_source, verified_capture_source
