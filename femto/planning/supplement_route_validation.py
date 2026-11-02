"""Independently reconstruct and check a research supplement-route proposal.

This verifies route geometry and selection only; it does not produce a coverage
certificate or claim suitability for human treatment or physical execution.
"""
from __future__ import annotations

from copy import deepcopy
import numpy as np
from femto.planning import effective_domain,quality_constrained as quality
from femto.planning.endpoint_preserving_detour import _signature
from femto.planning.planning_revision import digest
from femto.planning.selection_blocks import (_mark_revisits,selection_face_assignment,
                                              check_projection_inside_selection)
from femto.planning.supplement_two_opt_research import _anchored
from femto.planning.surface_trajectory import ScanSegment

VERSION='research-supplement-geometry-replay-v1'
RELOCATED_VERSION='research-relocated-geometry-replay-v1'
MAX_POINTS=200_000


def reconstruct_and_validate(base,segments,visits,links,mesh,region,*,check=lambda:None,budget=200_000):
    """Rebuild proposed items; check original scans, whole edges and selection.

    Caller must separately validate the base route, target and source scans.
    All results are detached from inputs. No coverage claim is generated.
    """
    return _reconstruct_and_validate(base,segments,visits,links,mesh,region,check=check,budget=budget,distinct_faces=False)


def reconstruct_relocated_and_validate(base,segments,visits,links,mesh,region,*,check=lambda:None,budget=200_000):
    """Replay a relocation using the product's distinct-face handoff rule.

    The caller must still validate the source, reprove coverage and compare
    against the original route. Existing supplement replay semantics are not
    changed by this research-only entry.
    """
    return _reconstruct_and_validate(base,segments,visits,links,mesh,region,check=check,budget=budget,distinct_faces=True)


def _reconstruct_and_validate(base,segments,visits,links,mesh,region,*,check,budget,distinct_faces):
    check()
    if not isinstance(segments,list) or not segments or any(not isinstance(s,ScanSegment) for s in segments):
        raise ValueError('补线路线缺少扫描段')
    if type(budget) is not int or not 1<=budget<=MAX_POINTS:raise ValueError('补线路线点数预算无效')
    params=base.planner['parameters'];width=params['effective_width_mm'];record=params['quality_evidence']
    spacing=params['row_spacing_mm']/1000
    if distinct_faces:
        from femto.planning.scan_visit_order import reconstruct
        chains,added,objective=reconstruct(segments,visits,links,check,preserve_endpoint_faces=True,budget=budget)
    else:chains,added,objective=_anchored(segments,visits,links,check,budget)
    if objective[0]:raise ValueError('补线路线存在工作块内部断点')
    expected=_signature([(s.block_index,s.points) for s in segments],check)
    if _signature(chains,check)!=expected:raise ValueError('补线路线改变原扫描边或来源面')
    _mark_revisits([p for _,p in chains],spacing,check);items=quality._items(chains)
    scan=[item for item in items if isinstance(item,ScanSegment)]
    allowed=effective_domain.validate(params['effective_domain'],record,width,mesh,region,check)
    accepted,_,_=quality.edge_checks(scan,mesh,record,width,check,allowed_faces=allowed)
    if not accepted.all():raise ValueError('补线路线未通过原整边几何检查')
    _,_,labels,_=selection_face_assignment(region,np.asarray(mesh.vertices),np.asarray(mesh.faces))
    check_projection_inside_selection(scan,base.blocks['items'],labels,
        np.asarray(region['view_reference']['undistorted_intrinsic']),np.asarray(region['view_reference']['image_size']))
    check()
    classes,_=quality.checked_evidence(record,region)
    statistics,blocks=quality._summaries(items,base.blocks['items'],classes)
    report=dict(version=RELOCATED_VERSION if distinct_faces else VERSION,all_edges_passed=True,selection_projection_passed=True,
        preserved_scan_edges_and_faces=True,internal_gaps=objective[0],connector_length_m=objective[1],
        added_connector_points=added,items_sha256=digest([item.to_dict() for item in items]),
        semantics='geometry_only_not_coverage_or_physical_motion')
    return deepcopy(items),statistics,blocks,report
