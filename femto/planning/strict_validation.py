"""Validate the new P4 construction, preserving all legacy replay semantics."""
from __future__ import annotations

from collections import Counter
from dataclasses import replace

from femto.geometry.surface_support import fraction
from femto.planning import strict_path, planning_revision as revisions
from femto.planning.refinement_validation import semantic_payload, _check_final
from femto.planning.surface_trajectory import SurfaceTrajectory, ScanSegment, validate_trajectory, canonical_json_bytes


def validate(trajectory, mesh=None, region=None, check=lambda: None):
    external_region=region is not None
    report=trajectory.gap_report.get('surface_support') or {}
    if (report.get('schema')!=strict_path.SCHEMA or report.get('projection_creates_new_candidate') is not True
            or report.get('support_scope')!='exact_constructed_numeric_surface_only'
            or report.get('physical_accuracy_verified') is not False or report.get('execution_ready') is not False
            or report.get('registration_uncertainty_mm',0) is not None):
        raise ValueError('严格路径支持证据语义无效')
    budget=strict_path.SupportBudget(**report['budget']).checked()
    base=SurfaceTrajectory.from_dict(report['base_trajectory'])
    if base.planner['algorithm_version']!=revisions.ALGORITHMS['route'] or base.planning_context is None:
        raise ValueError('严格重建原路线必须是带台账的常规 v3 版本')
    if region is None:region=base.planning_context['revision']['inputs']['selection']
    # from_dict has already performed the complete structural validation.
    # Geometry or an external selection still requires its independent check.
    if mesh is not None or external_region:validate_trajectory(base,mesh=mesh,region=region,check=check)
    if trajectory.planner['parameters'].get('surface_support')!=strict_path.algorithms_for(base,budget)['support']:
        raise ValueError('严格构造修订与原路线或预算不一致')
    expected_context=strict_path.context_for(base,budget)
    if trajectory.planning_context!=expected_context:
        raise ValueError('严格重建改变了原目标或计算修订')
    original=sum(len(s.points)-1 for s in base.ordered_items if isinstance(s,ScanSegment))
    records=report['records']
    if (not isinstance(records,list) or len(records)>original
            or report['original_edge_count']!=original or report['processed_original_edges']!=len(records)
            or report['counts']!=dict(sorted(Counter(r['status'] for r in records).items()))
            or report['edge_outcomes']!=strict_path.outcome_counts(records)):
        raise ValueError('严格重建原边台账不一致')
    expected_sources=[(s.segment_index,i,s.block_index) for s in base.ordered_items if isinstance(s,ScanSegment)
                      for i in range(len(s.points)-1)]
    if (report['stop_reason']=='completed' and len(records)!=original
            or any((r['source_segment'],r['source_edge'],r['block_index'])!=source
                   or r['status'] not in ('checked','rejected','unconfirmed') for r,source in zip(records,expected_sources))):
        raise ValueError('严格重建原边顺序、工作块或停止状态不一致')
    for name,limit in (('face_checks',budget.max_face_checks),('filter_comparisons',budget.max_filter_comparisons),
                       ('edge_checks',budget.max_edge_checks)):
        if type(report[name]) is not int or not 0<=report[name]<=limit:raise ValueError('严格构造超过检查预算')
    for name in ('support_queries','interval_queries','aabb_rejected_faces','construction_fast_paths'):
        if type(report[name]) is not int or report[name]<0:raise ValueError('严格构造计算计数无效')
    if report['interval_queries']+report['construction_fast_paths']>report['support_queries']:
        raise ValueError('严格构造快速证明与精确区间计数无效')
    if trajectory.statistics['point_count']>budget.max_output_points:raise ValueError('严格构造超过输出预算')
    if report['edge_checks']!=sum(len(r['edge_checks']) for r in records):raise ValueError('严格构造边检查计数不一致')
    if len(canonical_json_bytes(trajectory.to_dict()))>64*1024*1024:raise ValueError('严格路线超过 64 MiB 限制')
    for item in trajectory.ordered_items:
        if not isinstance(item,ScanSegment):continue
        for point in item.points:
            check()
            record=point.get(strict_path.ANCHOR) or {}
            if record.get('face')!=point['triangle_index'] or len(record.get('barycentric',[]))!=3:
                raise ValueError('严格路径缺少逐点构造记录')
            weights=[fraction(v) for v in record['barycentric']]
            if min(weights)<0 or sum(weights)!=1:raise ValueError('严格路径重心坐标无效')
    expected=strict_path.route_from_report(base,report,check)
    expected=replace(expected,planning_context=trajectory.planning_context)
    if semantic_payload(expected)!=semantic_payload(trajectory):raise ValueError('严格路径与保存的逐边结果不一致')
    _check_final(trajectory,mesh,region,check)
    if mesh is not None:
        rebuilt=strict_path.reconstruct(base,mesh,region,budget=budget,check=check)
        rebuilt=replace(rebuilt,planning_context=trajectory.planning_context)
        if semantic_payload(rebuilt)!=semantic_payload(trajectory):
            raise ValueError('严格路径构造、逐边结果或预算无法由原输入重现')


def reproduce_base(trajectory, manifest_path, mesh, region, check=lambda: None):
    from femto.planning.refinement_validation import reproduce_v3
    base=SurfaceTrajectory.from_dict(trajectory.gap_report['surface_support']['base_trajectory'])
    reproduce_v3(base,manifest_path,mesh,region,check)
