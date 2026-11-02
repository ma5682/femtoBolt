"""Bridge the workbench to the existing selection-block mesh planner.

Input snapshots are derived assets, not user exports. The live annotations are
never marked saved during generation. Display, playback and final export share
the same ordered trajectory; saving never runs a second planner.
"""
from __future__ import annotations

from copy import copy, deepcopy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import trimesh
from femto.operation_diagnostics import diagnosed
from femto.geometry.region_reuse import operation as geometry_operation
from femto.planning.nominal_revisit_metrics import operation as metric_operation

from femto.planning.surface_trajectory import (
    ORDERED_SELECTION_ALGORITHM, SurfaceTrajectory, to_player_plan, validate_trajectory,
)

SCHEMA = 'femto-workspace-surface-plan-v1'
KIND = 'mesh_slice_selection_blocks_ordered'


def continuity_text(plan):
    if plan['surface_trajectory'].get('planner', {}).get('algorithm_version') == 'source-anchored-surface-v1':
        stats=plan['surface_trajectory']['statistics']
        from femto.planning.coverage_workflow import summary_text
        coverage_note=(summary_text(plan['coverage_evidence']) if plan.get('coverage_evidence') is not None
                       else '新路线覆盖尚未核验。')
        return (f"{stats['block_count']} 个工作块，{stats['segment_count']} 条连续路线；"
                f"块内断点 {stats['gap_count']-stats['inter_block_gap_count']} 处；"
                '来源点修正已重建并核对几何，块间不连接；'+coverage_note)
    if plan['surface_trajectory'].get('planner', {}).get('algorithm_version') == 'nominal-variable-width-refined-surface-v1':
        from femto.planning.variable_width_refinement import summary_text
        return summary_text(plan['surface_trajectory'])
    if plan['surface_trajectory'].get('planner', {}).get('algorithm_version') == 'nominal-variable-width-surface-v1':
        from femto.planning.variable_width_route import summary_text
        return summary_text(plan['surface_trajectory'])
    if plan['surface_trajectory'].get('planner', {}).get('algorithm_version') in ('coverage-refined-surface-v1', 'constructed-coverage-refined-surface-v1'):
        from femto.planning.coverage_refinement import summary_text
        return summary_text(plan['surface_trajectory'])
    if plan['surface_trajectory'].get('planner', {}).get('algorithm_version') == 'constructed-surface-route-v1':
        from femto.planning.strict_path import summary_text
        return summary_text(plan['surface_trajectory'])
    if plan['surface_trajectory'].get('planner',{}).get('algorithm_version')=='quality-constrained-surface-v3':
        trajectory=plan['surface_trajectory'];stats=trajectory['statistics']
        report=trajectory['gap_report']['quality_constraint'];routing=trajectory['gap_report']['reconnection']
        connected=sum(item['connected'] for item in routing['connections'])
        internal=stats['gap_count']-stats['inter_block_gap_count']
        coverage_note='目标覆盖尚未完成核验。'
        if plan.get('coverage_evidence') is not None:
            from femto.planning.coverage_workflow import summary_text
            coverage_note=summary_text(plan['coverage_evidence'])
        sampling=trajectory['planner']['parameters'].get('sampling_decision')
        sampling_text=(f"沿线自动取点：最大 {sampling['sample_step_mm']:g} mm，保留跨面点；" if sampling else '')
        layout=trajectory['planner']['parameters'].get('fixed_width_layout')
        if layout is not None:
            from femto.planning.fixed_width_layout import summary
            sampling_text=summary(layout, stats['row_count'],trajectory)+sampling_text
        return (sampling_text+f"{stats['block_count']} 个工作块，{stats['segment_count']} 条连续路线；"
                f"质量与宽度约束后重新连接 {connected} 处（{routing['before_chains']} 段合并为 {routing['after_chains']} 段）。"
                +(f'块内仍有 {internal} 处未找到可核验连接；' if internal else '块内已连续连接；')
                +f"{report['pending_pixels']} 个选区像素待检查/补扫；"+coverage_note)
    if plan['surface_trajectory'].get('planner', {}).get('algorithm_version') in ('quality-constrained-surface-v1', 'quality-constrained-surface-v2', 'quality-constrained-surface-v3'):
        trajectory = plan['surface_trajectory']
        gaps = trajectory['gap_report']
        report = gaps['quality_constraint']
        domain = trajectory['planner']['parameters'].get('effective_domain')
        prefix = ('' if domain is None else
                  f"质量合格表面域 {domain['eligible_face_count']} 面，缺失范围 {len(domain['missing_regions'])} 处；")
        original = gaps.get('candidate_gaps') or []
        original_breaks = ''
        if original:
            causes = {'disconnected_surface': '表面不连通', 'selection_break': '工作块分离',
                      'unsupported_geometry': '网格不支持'}
            counts = {}
            for gap in original:
                reason = causes.get(gap.get('reason'), '其他断点')
                counts[reason] = counts.get(reason, 0) + 1
            detail = '、'.join(f'{name} {count}' for name, count in counts.items())
            original_breaks = f'候选路线原有 {len(original)} 处断点（{detail}）；'
        causes = {'quality_or_width_boundary': '质量/作用宽度',
                  'outside_effective_domain': '有效表面域外',
                  'chord_not_certified_on_mesh': '网格连接无法核验',
                  'mesh_boundary_clearance': '网格边界余量不足'}
        rejected = report.get('rejection_counts') or {}
        rejection_detail = '、'.join(
            f'{causes.get(reason, reason)} {count}' for reason, count in rejected.items() if count)
        rejected_text = (f"排除 {report['rejected_edges']} 段边"
                         + (f'（{rejection_detail}）' if rejection_detail else ''))
        chains = trajectory['statistics']['segment_count']
        return (prefix + original_breaks
                + f"质量与宽度约束后剩余 {chains} 条连续段、{report['retained_edges']} 段边；"
                + rejected_text + f"；{report['pending_pixels']} 个选区像素待检查/补扫。"
                + '目标覆盖尚未完成核验，断点不连接')
    stats = plan['surface_trajectory']['statistics']
    internal = sum(not gap.get('inter_block', False) for gap in plan['chain_breaks'])
    text = f"{stats['block_count']} 个工作块，{stats['planned_block_count']} 块已连续连接"
    if internal:
        text += f'；块内仍有 {internal} 处断点'
    if stats.get('unplannable_block_count'):
        text += f"；{stats['unplannable_block_count']} 块无法规划"
    if stats.get('coverage_status') != 'complete':
        text += '；覆盖未完整，请核对'
    if stats['block_count'] > 1:
        text += '；分离工作块之间未规划移动'
    return text


def display_runs(plan):
    """Group edges by incoming motion role; never bridge a gap/missing pixel."""
    for chain in plan['segments']:
        run, role = [], None
        for left, right in zip(chain, chain[1:]):
            current = right.get('motion_kind', 'scan')
            if left.get('pixel') is None or right.get('pixel') is None:
                if run:yield role, run
                run, role = [], None
                continue
            if current != role:
                if run:yield role, run
                run, role = [left['pixel']], current
            elif not run:
                run = [left['pixel']]
            run.append(right['pixel'])
        if run:yield role, run


def checked_head(head):
    from femto.planning.treatment_head import head_parameters, _inputs
    fresh = head_parameters(head['mode'], **_inputs(head))
    if fresh != head:
        raise ValueError('治疗头参数记录不一致，请重新设置')
    return fresh


@diagnosed('route.snapshot')
def snapshot(model, region, check, *, include_texture=True):
    if not region.get('selection_intent'):
        raise ValueError('此区域缺少选择意图，请返回工作区域重新涂抹；不能用 RGB 路线替代三维连接')
    store = copy(model.store)
    store.points = []
    store.regions = [deepcopy(region)]
    # Retained as provenance for this preview and any eventual native export.
    path = store.save(Path(model.reconstruction)/'planning_inputs', check=check,
                      include_texture=include_texture)
    return path.parent/'region_1.json'


@diagnosed('route.generate')
@geometry_operation()
@metric_operation()
def generate(model, region, head, check=lambda: None, progress=lambda text: None, *, layout_algorithm=None, routing_algorithm=None,
             target_algorithm=None, research_row_shifts=None, minimum_metric_fraction=None):
    from femto.planning.continuous_quality import candidate, replan
    from femto.planning.planning_state import binding, region_signature
    head = checked_head(head)
    from femto.planning import planning_revision, target_surface
    from femto.planning.fixed_width_layout import SHIFT_PATCH_VERSION,FLOOR_PATCH_VERSION,ADAPTIVE_FLOOR_PATCH_VERSION,COORDINATED_FLOOR_PATCH_VERSION
    from femto.planning.continuous_quality import FLOOR_DETOUR_ROUTING,UNIFIED_FLOOR_ROUTING,GLOBAL_FLOOR_ROUTING,REFINED_CONNECTOR_ROUTING
    if routing_algorithm==REFINED_CONNECTOR_ROUTING and layout_algorithm!=COORDINATED_FLOOR_PATCH_VERSION:
        raise ValueError('贴面连接优化需要自适应接缝协调排布')
    if routing_algorithm in (UNIFIED_FLOOR_ROUTING,GLOBAL_FLOOR_ROUTING,REFINED_CONNECTOR_ROUTING) and layout_algorithm in (ADAPTIVE_FLOOR_PATCH_VERSION,COORDINATED_FLOOR_PATCH_VERSION):
        if minimum_metric_fraction is not None:raise ValueError('自适应尺度下限不能指定固定比例')
    elif routing_algorithm in (FLOOR_DETOUR_ROUTING,UNIFIED_FLOOR_ROUTING,GLOBAL_FLOOR_ROUTING,REFINED_CONNECTOR_ROUTING):
        if layout_algorithm!=FLOOR_PATCH_VERSION or minimum_metric_fraction not in (None,.65):
            raise ValueError('疏密绕行优化需要 65% 尺度行距下限排布')
        minimum_metric_fraction=.65
    if minimum_metric_fraction is not None:
        from femto.planning.fixed_width_layout import request as layout_request
        minimum_metric_fraction=layout_request(head['effective_treatment_width_mm'],head['overlap_percent'],
            version=layout_algorithm,minimum_metric_fraction=minimum_metric_fraction)['minimum_metric_fraction']
    if layout_algorithm==SHIFT_PATCH_VERSION:
        if not isinstance(research_row_shifts,list) or not 1<=len(research_row_shifts)<=4:
            raise ValueError('研究局部行偏移需要一至四条显式记录')
        research_row_shifts=deepcopy(research_row_shifts)
    elif research_row_shifts is not None:
        raise ValueError('局部行偏移仅可用于指定研究排布')
    if layout_algorithm is None and routing_algorithm is None:
        if target_algorithm is not None:raise ValueError('直接精确目标目前仅用于指定的研究排布')
        revision = planning_revision.capture(model, region, head)
    else:
        from femto.planning.fixed_width_layout import OPTIMIZED_PATCH_VERSION,FOLDBACK_PATCH_VERSION,ADAPTIVE_PATCH_VERSION,MINIMAL_PATCH_VERSION
        if layout_algorithm not in (OPTIMIZED_PATCH_VERSION,FOLDBACK_PATCH_VERSION,ADAPTIVE_PATCH_VERSION,MINIMAL_PATCH_VERSION,SHIFT_PATCH_VERSION,FLOOR_PATCH_VERSION,ADAPTIVE_FLOOR_PATCH_VERSION,COORDINATED_FLOOR_PATCH_VERSION) or head.get('mode')!='fixed_width_auto':
            raise ValueError('当前治疗头模式不支持指定的局部分区排布版本')
        from femto.planning.continuous_quality import IMPROVED_ROUTING,SUPPORT_ROUTING,BALANCED_ROUTING,SEAM_ROUTING,SEAM_BALANCED_ROUTING,PRESERVED_DETOUR_ROUTING,SHORT_DETOUR_ROUTING,JOINT_REORDER_ROUTING,BRIDGE_REORDER_ROUTING,RELOCATION_ROUTING,DEEP_RELOCATION_ROUTING,FLOOR_DETOUR_ROUTING,UNIFIED_FLOOR_ROUTING
        if layout_algorithm==FOLDBACK_PATCH_VERSION and routing_algorithm!=SEAM_BALANCED_ROUTING:
            raise ValueError('折返分区需要接缝联合排序版本')
        if layout_algorithm==ADAPTIVE_PATCH_VERSION and routing_algorithm not in (BRIDGE_REORDER_ROUTING,RELOCATION_ROUTING):
            raise ValueError('研究自适应分区需要两步联合排序或移动换序版本')
        if layout_algorithm==MINIMAL_PATCH_VERSION and routing_algorithm!=RELOCATION_ROUTING:
            raise ValueError('研究最少行分区需要移动换序版本')
        if layout_algorithm==SHIFT_PATCH_VERSION and routing_algorithm not in (RELOCATION_ROUTING,DEEP_RELOCATION_ROUTING,FLOOR_DETOUR_ROUTING):
            raise ValueError('研究行偏移需要移动换序版本')
        if layout_algorithm==FLOOR_PATCH_VERSION and routing_algorithm not in (DEEP_RELOCATION_ROUTING,FLOOR_DETOUR_ROUTING,UNIFIED_FLOOR_ROUTING,GLOBAL_FLOOR_ROUTING,REFINED_CONNECTOR_ROUTING):
            raise ValueError('行距下限排布需要深度换序版本')
        if layout_algorithm in (ADAPTIVE_FLOOR_PATCH_VERSION,COORDINATED_FLOOR_PATCH_VERSION) and routing_algorithm not in (UNIFIED_FLOOR_ROUTING,GLOBAL_FLOOR_ROUTING,REFINED_CONNECTOR_ROUTING):
            raise ValueError('自适应尺度下限需要统一预算换序版本')
        if routing_algorithm==DEEP_RELOCATION_ROUTING and layout_algorithm not in (OPTIMIZED_PATCH_VERSION,SHIFT_PATCH_VERSION,FLOOR_PATCH_VERSION):
            raise ValueError('额外扫描段移动需要固定宽度局部分区')
        if routing_algorithm not in (None,IMPROVED_ROUTING,SUPPORT_ROUTING,BALANCED_ROUTING,SEAM_ROUTING,SEAM_BALANCED_ROUTING,PRESERVED_DETOUR_ROUTING,SHORT_DETOUR_ROUTING,JOINT_REORDER_ROUTING,BRIDGE_REORDER_ROUTING,RELOCATION_ROUTING,DEEP_RELOCATION_ROUTING,FLOOR_DETOUR_ROUTING,UNIFIED_FLOOR_ROUTING,GLOBAL_FLOOR_ROUTING,REFINED_CONNECTOR_ROUTING):raise ValueError('当前路线连接算法版本不受支持')
        algorithms=(dict(planning_revision.RESEARCH_SHIFT_DEEP_RELOCATION_ALGORITHMS if routing_algorithm==DEEP_RELOCATION_ROUTING
                         else planning_revision.RESEARCH_SHIFT_RELOCATION_ALGORITHMS,
                         row_shifts_sha256=planning_revision.digest(research_row_shifts)) if layout_algorithm==SHIFT_PATCH_VERSION else
                    planning_revision.REFINED_COORDINATED_FLOOR_ALGORITHMS if routing_algorithm==REFINED_CONNECTOR_ROUTING else
                    planning_revision.GLOBAL_COORDINATED_FLOOR_ALGORITHMS if layout_algorithm==COORDINATED_FLOOR_PATCH_VERSION and routing_algorithm==GLOBAL_FLOOR_ROUTING else
                    planning_revision.GLOBAL_ADAPTIVE_FLOOR_ALGORITHMS if layout_algorithm==ADAPTIVE_FLOOR_PATCH_VERSION and routing_algorithm==GLOBAL_FLOOR_ROUTING else
                    planning_revision.GLOBAL_FLOOR_ALGORITHMS if routing_algorithm==GLOBAL_FLOOR_ROUTING else
                    planning_revision.COORDINATED_UNIFIED_FLOOR_ALGORITHMS if layout_algorithm==COORDINATED_FLOOR_PATCH_VERSION else
                    planning_revision.ADAPTIVE_UNIFIED_FLOOR_ALGORITHMS if layout_algorithm==ADAPTIVE_FLOOR_PATCH_VERSION else
                    planning_revision.UNIFIED_FLOOR_ALGORITHMS if routing_algorithm==UNIFIED_FLOOR_ROUTING else
                    planning_revision.BOUNDED_FLOOR_ALGORITHMS if routing_algorithm==FLOOR_DETOUR_ROUTING else
                    planning_revision.FLOOR_DEEP_RELOCATION_ALGORITHMS if layout_algorithm==FLOOR_PATCH_VERSION else
                    planning_revision.RESEARCH_MINIMAL_RELOCATION_ALGORITHMS if layout_algorithm==MINIMAL_PATCH_VERSION else
                    planning_revision.RESEARCH_ADAPTIVE_RELOCATION_ALGORITHMS if layout_algorithm==ADAPTIVE_PATCH_VERSION and routing_algorithm==RELOCATION_ROUTING else
                    planning_revision.RESEARCH_ADAPTIVE_PATCH_ALGORITHMS if layout_algorithm==ADAPTIVE_PATCH_VERSION else
                    planning_revision.FOLDBACK_PATCH_LAYOUT_ALGORITHMS if layout_algorithm==FOLDBACK_PATCH_VERSION else
                    planning_revision.DEEP_RELOCATION_PATCH_LAYOUT_ALGORITHMS if routing_algorithm==DEEP_RELOCATION_ROUTING else
                    planning_revision.RELOCATION_PATCH_LAYOUT_ALGORITHMS if routing_algorithm==RELOCATION_ROUTING else
                    planning_revision.BRIDGE_REORDER_PATCH_LAYOUT_ALGORITHMS if routing_algorithm==BRIDGE_REORDER_ROUTING else
                    planning_revision.JOINT_REORDER_PATCH_LAYOUT_ALGORITHMS if routing_algorithm==JOINT_REORDER_ROUTING else
                    planning_revision.SHORT_DETOUR_PATCH_LAYOUT_ALGORITHMS if routing_algorithm==SHORT_DETOUR_ROUTING else
                    planning_revision.PRESERVED_DETOUR_PATCH_LAYOUT_ALGORITHMS if routing_algorithm==PRESERVED_DETOUR_ROUTING else
                    planning_revision.SEAM_BALANCED_PATCH_LAYOUT_ALGORITHMS if routing_algorithm==SEAM_BALANCED_ROUTING else
                    planning_revision.SEAM_PATCH_LAYOUT_ALGORITHMS if routing_algorithm==SEAM_ROUTING else
                    planning_revision.BALANCED_PATCH_LAYOUT_ALGORITHMS if routing_algorithm==BALANCED_ROUTING else
                    planning_revision.SUPPORT_PATCH_LAYOUT_ALGORITHMS if routing_algorithm==SUPPORT_ROUTING else
                    planning_revision.IMPROVED_PATCH_LAYOUT_ALGORITHMS if routing_algorithm==IMPROVED_ROUTING
                    else planning_revision.OPTIMIZED_PATCH_LAYOUT_ALGORITHMS)
        if minimum_metric_fraction is not None:
            algorithms=dict(algorithms,minimum_metric_fraction=minimum_metric_fraction)
        if target_algorithm is not None:
            allowed = ((layout_algorithm in (ADAPTIVE_PATCH_VERSION,MINIMAL_PATCH_VERSION,SHIFT_PATCH_VERSION)
                        and routing_algorithm!=DEEP_RELOCATION_ROUTING)
                       or (layout_algorithm == OPTIMIZED_PATCH_VERSION and routing_algorithm == RELOCATION_ROUTING))
            if not allowed or target_algorithm!=planning_revision.EXACT_TARGET:
                raise ValueError('直接精确目标目前仅用于指定的局部分区排布')
            algorithms=dict(algorithms,target=target_algorithm)
        revision = planning_revision.capture(model, region, head, algorithms=algorithms)
    if revision['inputs']['quality'] != planning_revision.QUALITY:
        raise ValueError('当前路线算法不支持该质量配置，请恢复已验证配置后重新计算')
    from femto.planning.quality_constrained import evidence
    progress('核验真实参考深度与网格的一致性')
    quality = evidence(model, region, check, allow_empty=True)
    progress('按原选区像素建立目标表面台账，保留未定位范围')
    target = target_surface.build(model, region, revision, evidence=quality, check=check)
    from femto.planning.effective_domain import build as build_domain
    from femto.planning.surface_trajectory import canonical_json_bytes
    progress('构建质量合格的表面域并识别缺失范围')
    local = canonical_region_mesh(model.store.mesh, region)
    domain = build_domain(local,region,quality,head['effective_treatment_width_mm'],check)
    target = target_surface.with_quality(target, revision, region, quality, domain, check)
    if target_algorithm==planning_revision.EXACT_TARGET:
        # This page was just built from the current inputs and quality domain;
        # keep the same paged provenance as explicit exact reconstruction.
        from femto.planning import target_paging
        check();target=target_paging.materialize(target_paging._seal((target,),revision),revision);check()
    context = dict(schema=planning_revision.CONTEXT_SCHEMA, revision=revision, target=target.metadata)
    progress('准备当前选区的三维规划输入')
    # The planner reads the canonical PLY, not the presentation texture atlas.
    # Full textured output is still produced by the final export snapshot.
    manifest = snapshot(model, region, check, include_texture=False)
    target_surface.write(manifest.parent, target, check)
    assessment_path=manifest.parent/'planning_assessment.json'
    assessment=dict(schema='femto-planning-assessment-v1',quality_evidence=quality,effective_domain=domain,planning_context=context,
        status='assessed', target_completion='unverified',execution_ready=False,physical_accuracy_verified=False)
    assessment_path.write_bytes(canonical_json_bytes(assessment))
    if not domain['eligible_face_count']:
        assessment['status']='blocked_no_supported_faces'
        assessment_path.write_bytes(canonical_json_bytes(assessment))
        raise ValueError('没有通过质量检查的有效表面域；原选区保留，请检查或补扫。报告：'+str(assessment_path))
    progress('三维网格切片、换行折返与块内连接')
    layout = None
    if head['mode']=='fixed_width_auto':
        from femto.planning.fixed_width_layout import request,RESIDUAL_VERSION
        layout=request(head['effective_treatment_width_mm'], head['overlap_percent'],version=revision['inputs']['algorithms'].get('layout',RESIDUAL_VERSION),
                       minimum_metric_fraction=revision['inputs']['algorithms'].get('minimum_metric_fraction'))
    from femto.planning.fixed_width_layout import REPAIR_PATCH_VERSION,OPTIMIZED_PATCH_VERSION,FOLDBACK_PATCH_VERSION,ADAPTIVE_PATCH_VERSION,MINIMAL_PATCH_VERSION
    if layout is not None and layout['version'] in (REPAIR_PATCH_VERSION,OPTIMIZED_PATCH_VERSION,FOLDBACK_PATCH_VERSION,ADAPTIVE_PATCH_VERSION,MINIMAL_PATCH_VERSION,FLOOR_PATCH_VERSION,ADAPTIVE_FLOOR_PATCH_VERSION,COORDINATED_FLOOR_PATCH_VERSION):
        from femto.planning.repair_workflow import generate as repair_generate
        progress('检查局部短行并核验有界调整候选')
        try:
            trajectory=repair_generate(manifest,local,region,quality,head['effective_treatment_width_mm'],domain,layout,
                                       check=check,routing=revision['inputs']['algorithms']['routing'],row_spacing_mm=head['row_spacing_mm'])
        except ValueError as error:
            assessment.update(status='blocked_by_route_constraints',reason=str(error),route_report=getattr(error,'report',None))
            assessment_path.write_bytes(canonical_json_bytes(assessment))
            raise ValueError(str(error)+' 检查报告：'+str(assessment_path)) from error
    else:
        try:
            request_layout=(dict(request=layout,row_shifts=research_row_shifts)
                            if layout_algorithm==SHIFT_PATCH_VERSION else layout)
            trajectory = candidate(manifest, row_spacing_mm=head['row_spacing_mm'],
                                   check=check, fixed_width_layout=request_layout,layout_evidence=quality)
        except ValueError as error:
            assessment.update(status='blocked_by_candidate_planner', reason=str(error),
                candidate_reason=getattr(error, 'reason', None))
            assessment_path.write_bytes(canonical_json_bytes(assessment))
            raise ValueError(str(error)+' 检查报告：'+str(assessment_path)) from error
        progress('检查作用宽度并在有效表面内重新连接换行与绕行')
        try:
            trajectory = replan(trajectory, local, region, quality, head['effective_treatment_width_mm'], check=check, domain=domain,
                                routing=revision['inputs']['algorithms']['routing'])
        except ValueError as error:
            assessment.update(status='blocked_by_route_constraints',reason=str(error),route_report=getattr(error,'report',None))
            assessment_path.write_bytes(canonical_json_bytes(assessment))
            raise ValueError(str(error)+' 检查报告：'+str(assessment_path)) from error
    planning_revision.require_current(model, region, head, revision)
    trajectory = replace(trajectory, planning_context=context)
    assessment.update(status='research_preview_generated',route_report=trajectory.gap_report)
    assessment_path.write_bytes(canonical_json_bytes(assessment))
    plan = to_player_plan(trajectory)
    plan.update(schema=SCHEMA, region_id=region['annotation_id'],
        region_signature=region_signature(region), coordinate_frame='world_rdf', unit='metre',
        treatment_head=deepcopy(head), surface_trajectory=trajectory.to_dict())
    plan['plan_binding'] = binding(model.store.mesh, region, plan)
    return plan


def validate(plan, mesh, region, check=lambda: None):
    """Validate local face support and the exact player projection, offline too."""
    from femto.annotation.scene_annotations import mesh_digest
    from femto.planning.planning_state import region_signature
    head = checked_head(plan['treatment_head'])
    if plan.get('planning_context') is not None:
        from femto.planning.planning_revision import validate_context
        revision = validate_context(plan['planning_context'], mesh=mesh, region=region)
        if revision['inputs']['effect_model']['treatment_head'] != head:
            raise ValueError('覆盖宽度或治疗头参数与计算修订不一致，请重新生成路线')
    local = canonical_region_mesh(mesh, region)
    trajectory = SurfaceTrajectory.from_dict(plan['surface_trajectory'], mesh=local, region=region, check=check)
    layout=trajectory.planner['parameters'].get('fixed_width_layout')
    if (head['mode']=='fixed_width_auto') != (layout is not None):
        raise ValueError('固定宽度排布模式与保存路线不一致')
    if layout is not None:
        from femto.planning.fixed_width_layout import request
        if layout['request'] != request(head['effective_treatment_width_mm'], head['overlap_percent'],version=layout['request']['version'],
                                       minimum_metric_fraction=layout['request'].get('minimum_metric_fraction')):
            raise ValueError('固定宽度排布参数与治疗头不一致')
    if trajectory.planner['algorithm_version'] not in (ORDERED_SELECTION_ALGORITHM, 'quality-constrained-surface-v1', 'quality-constrained-surface-v2', 'quality-constrained-surface-v3', 'source-anchored-surface-v1', 'coverage-refined-surface-v1', 'constructed-surface-route-v1', 'constructed-coverage-refined-surface-v1', 'nominal-variable-width-surface-v1', 'nominal-variable-width-refined-surface-v1'):
        raise ValueError('工作台三维路线算法版本不匹配')
    if trajectory.planner['algorithm_version'] in ('coverage-refined-surface-v1', 'constructed-coverage-refined-surface-v1', 'nominal-variable-width-refined-surface-v1') and plan.get('coverage_evidence') is None:
        raise ValueError('补充规划必须保留完整覆盖证据')
    if trajectory.planner['algorithm_version'] in ('quality-constrained-surface-v1', 'quality-constrained-surface-v2', 'quality-constrained-surface-v3', 'source-anchored-surface-v1', 'coverage-refined-surface-v1', 'constructed-surface-route-v1', 'constructed-coverage-refined-surface-v1', 'nominal-variable-width-surface-v1', 'nominal-variable-width-refined-surface-v1'):
        if trajectory.planner['parameters']['effective_width_mm'] != head['effective_treatment_width_mm']:
            raise ValueError('覆盖宽度已改变，请重新生成路线')
    if (trajectory.source['geometry_sha256'] != mesh_digest(mesh)
            or trajectory.source['region_signature'] != region_signature(region)
            or trajectory.source['region_id'] != region['annotation_id']
            or trajectory.planner['parameters']['row_spacing_mm'] != head['row_spacing_mm']):
        raise ValueError('三维路线与当前模型、选区或行距不匹配')
    from femto.planning.surface_trajectory import _player_plan_view
    check()
    expected = _player_plan_view(trajectory)
    for key, value in expected.items():
        if key != 'schema' and plan.get(key) != value:
            raise ValueError('三维路线显示与原始轨迹不一致：'+key)
    check()
    return trajectory


def canonical_region_mesh(mesh, region):
    from femto.annotation.scene_annotations import region_mesh
    from femto.geometry.region_reuse import canonical
    local = region_mesh(mesh, region['triangle_indices'])
    # The existing region exporter writes PLY vertices as float32. Validate
    # against that same canonical representation, without relaxing tolerance
    # or changing the original mesh. Float64 input can differ at shared edges.
    return canonical(local, lambda: trimesh.Trimesh(
        vertices=np.asarray(local.vertices).astype(np.float32).astype(np.float64),
        faces=np.asarray(local.triangles), process=False))


@geometry_operation()
def report_text(plan, mesh, region):
    from femto.planning.trajectory_diagram import report_text as trajectory_report
    from femto.planning.treatment_head import _payload, report_lines
    trajectory = validate(plan, mesh, region)
    head = _payload(trajectory, plan['treatment_head'], canonical_region_mesh(mesh, region), lambda: None)
    from femto.planning.target_surface import summary_text
    target = (plan.get('planning_context') or {}).get('target')
    from femto.planning.coverage_workflow import summaries, summary_text as coverage_text
    target_text = summary_text(target) if plan.get('coverage_evidence') is None else summaries(plan)[1]
    return (trajectory_report(trajectory)+'\n'+'\n'.join(report_lines(head))+'\n\n'+target_text
            +'\n'+coverage_text(plan.get('coverage_evidence')))


@diagnosed('target.prepare')
def target_for(model, plan, check=lambda: None, *, reproduce=False):
    """Portable models hold their own verified assets, independent of old paths."""
    from femto.planning import target_surface
    context = plan.get('planning_context')
    if context is None:
        return None
    params = plan['surface_trajectory']['planner']['parameters']
    values = dict(evidence=params['quality_evidence'], domain=params['effective_domain'], check=check, reproduce=reproduce)
    cached = getattr(model, 'target_ledger', None)
    if cached is not None and cached.metadata == context['target']:
        return target_surface.validate(cached, context['revision'], **values)
    folder = Path(plan['surface_trajectory']['source']['region_manifest']).parent
    return target_surface.read(folder, context['target'], context['revision'], **values)


@geometry_operation()
@metric_operation()
def export(model, region, plan, check=lambda: None, progress=lambda text: None, commit=None, *, verified_sink=None, target_session=None):
    from femto.planning.trajectory_export import export_trajectory
    from femto.planning.mesh_slice_planner import _source_record
    from femto.planning.trajectory_diagram import write_visual_outputs
    from femto.planning.treatment_head import write_record
    from femto.planning.planning_state import validated_surface
    target = None
    if target_session is not None:
        from femto.planning.save_validation import SaveValidationSession
        if type(target_session) is not SaveValidationSession or not plan.get('planning_context'):
            raise ValueError('当前方案不能使用原目标保存验证会话')
        target = target_for(model, plan, check)
        if target is None:raise ValueError('当前方案不能使用原目标保存验证会话')
        target_session._begin_route_validation(model, region, plan, target, check)
    progress('核对当前三维路线与选区')
    # This is the route just validated against the current mesh, region and
    # selection. Reusing it avoids reparsing/revalidating the same plan; the
    # helper rechecks the plan binding before returning and export_trajectory
    # still checks the relocated source and output contract independently.
    trajectory = validated_surface(plan, model.store.mesh, region, check=check)
    from femto.planning.coverage_workflow import cached
    certificate = cached(model, plan)
    if trajectory.planner['algorithm_version'] in ('quality-constrained-surface-v1', 'quality-constrained-surface-v2', 'quality-constrained-surface-v3', 'source-anchored-surface-v1', 'coverage-refined-surface-v1', 'constructed-surface-route-v1', 'constructed-coverage-refined-surface-v1', 'nominal-variable-width-surface-v1', 'nominal-variable-width-refined-surface-v1'):
        from femto.planning.quality_constrained import evidence
        current = evidence(model, region, check)
        if current != trajectory.planner['parameters']['quality_evidence']:
            raise ValueError('当前几何质量证据已变化，请重新生成路线')
    if target_session is None:target = target_for(model, plan, check)
    if target is not None:
        from femto.planning import planning_revision, target_surface
        planning_revision.require_current(model, region, plan['treatment_head'], plan['planning_context']['revision'])
    if target_session is not None:
        target_session._prepare_native(model, region, plan, target, check)
    # A fresh immutable snapshot also captures current names/notes, and makes
    # portable projects independent of the original cached input directory.
    progress('保存本次输入快照')
    manifest_path = snapshot(model, region, check)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if target is not None:
        target_surface.write(manifest_path.parent, target, check)
    # A portable project may have moved since this route was generated. Export
    # the new snapshot's role declaration too, so strict v3 reproduction sees
    # the same source path without changing the saved route or its geometry.
    trajectory = replace(trajectory, source=dict(trajectory.source, **_source_record(manifest_path, manifest)),
                         data_layers=manifest.get('data_layers'))
    def augment(folder, prepared, check):
        write_visual_outputs(folder, prepared, check)
        write_record(folder, prepared, plan['treatment_head'], check)
        domain=prepared.planner['parameters'].get('effective_domain')
        if domain is not None:
            from femto.planning.surface_trajectory import canonical_json_bytes
            (folder/'effective_domain.json').write_bytes(canonical_json_bytes(domain))
    progress('保存同一条三维路线、参数与报告')
    result = export_trajectory(trajectory, Path(model.reconstruction)/'surface_paths',
        check=check, commit=commit, augment=augment, coverage_certificate=certificate, progress=progress,
        **({'verified_sink':verified_sink} if verified_sink is not None else {}))
    # Publication may have succeeded immediately before cancellation/input
    # rechecking fails. Keep its location in the worker's failure diagnostics.
    from femto.operation_diagnostics import mark_retained
    mark_retained(result, '三维路线包')
    if target_session is not None:
        target_session._native_published(model, region, plan, target, check)
    return result
