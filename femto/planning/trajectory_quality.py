"""Read-only checks of saved polylines; no smoothing, acceptance gate or new file schema.

Positions are W0/RDF metres. Turns are discrete tangent changes, not curvature or robot
joint motion. Measurements never include the unplanned transitions between chains.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Callable

import numpy as np

from femto.planning.surface_trajectory import ScanSegment, SurfaceTrajectory, SELECTION_BLOCKS_SCHEMA


def _distribution(values: np.ndarray) -> dict[str, float] | None:
    if not len(values):
        return None
    return dict(min=float(values.min()), p50=float(np.median(values)),
                p95=float(np.percentile(values, 95)), max=float(values.max()))


def polyline_quality(trajectory: SurfaceTrajectory, check: Callable[[], None] = lambda: None) -> dict[str, Any]:
    """Recompute sample/turn diagnostics from ordered_items, without trusting saved totals.

    Exact duplicate neighbours count as zero edges. For tangent diagnostics only, they
    are collapsed to the last point at that position, preserving its route index. An
    edge uses its destination's role, matching the existing length accounting contract.
    The returned dict is transient; it is not added to trajectory.json or hashed inputs.
    """
    steps, turns, locations = [], [], []
    scan_length = connector_length = 0.0
    chain_count = zero_edges = 0
    for segment in trajectory.ordered_items:
        check()
        if not isinstance(segment, ScanSegment):
            continue
        chain_count += 1
        points = segment.points
        xyz = np.asarray([p['xyz_m'] for p in points], dtype=float)
        edges = np.diff(xyz, axis=0)
        lengths = np.linalg.norm(edges, axis=1)
        connector = np.asarray([p.get('role') == 'connector' for p in points[1:]], dtype=bool)
        scan_length += float(lengths[~connector].sum())
        connector_length += float(lengths[connector].sum())
        zero_edges += int((lengths == 0).sum())
        steps.append(lengths * 1000)
        # Retain the last index in each consecutive run of identical XYZ positions.
        keep = np.r_[np.flatnonzero(lengths > 0), len(points) - 1]
        vectors = np.diff(xyz[keep], axis=0)
        if len(vectors) < 2:
            continue
        directions = vectors / np.linalg.norm(vectors, axis=1)[:, None]
        angles = np.degrees(np.arctan2(np.linalg.norm(np.cross(directions[:-1], directions[1:]), axis=1),
                                      np.einsum('ij,ij->i', directions[:-1], directions[1:])))
        turns.append(angles)
        # Top five per chain suffice for the global top five. Stable ordering resolves ties.
        for index in np.argsort(-angles, kind='stable')[:5]:
            point_index = int(keep[index + 1])
            point = points[point_index]
            incoming = points[int(keep[index]) + 1]
            outgoing = points[int(keep[index + 1]) + 1]
            locations.append(dict(angle_deg=float(angles[index]), block_index=segment.block_index,
                                  segment_index=segment.segment_index, route_index=point['route_index'],
                                  row_index=point['row_index'], xyz_m=list(point['xyz_m']),
                                  kind='connector' if incoming.get('role') == 'connector' or outgoing.get('role') == 'connector'
                                  else 'scan'))
    step_values = np.concatenate(steps) if steps else np.empty(0)
    angle_values = np.concatenate(turns) if turns else np.empty(0)
    total = scan_length + connector_length
    locations.sort(key=lambda item: (-item['angle_deg'], item['segment_index'], item['route_index']))
    return dict(chain_count=chain_count, edge_count=len(step_values), zero_length_edge_count=zero_edges,
                step_mm=_distribution(step_values), turn_count=len(angle_values), turn_deg=_distribution(angle_values),
                largest_turns=locations[:5], scan_length_m=scan_length, connector_length_m=connector_length,
                total_length_m=total, connector_fraction=connector_length / total if total > 0 else None)


def block_pending_reasons(block: dict[str, Any], chain_count: int) -> list[str]:
    """Concrete v4 work still needed; continuity and geometric support remain independent."""
    reasons = []
    if chain_count == 0:
        reasons.append('尚无可用路线：' + str(block.get('message') or block.get('reason') or '缺少路线'))
    elif chain_count != 1 or block.get('status') != 'planned':
        reasons.append(f'块内存在 {chain_count} 条链，需要沿可信表面完成连接')
    links = block.get('link_failure_counts') or {}
    if links.get('explicit_unselected_blocked') or links.get('forbidden_blocked'):
        reasons.append('未选区域限制连接：保留原选区，检查选区内是否存在可行绕行')
    if links.get('disconnected_measured_surface') or links.get('no_surface_path'):
        reasons.append('实测表面不连通：核查缺面并评估有依据的重建或补采')
    if links.get('outside_reference_image'):
        reasons.append('路线超出参考图像：核查选区与参考帧绑定')
    if links.get('budget_exceeded'):
        reasons.append('计算预算不足：检查问题规模，当前未完成连接')
    support = (block.get('selection') or {}).get('support')
    if support is None or support.get('geometry_unconfirmed_pixels') is None:
        reasons.append('几何支持未知：需要带逐像素支持记录的区域导出')
    elif support['geometry_unconfirmed_pixels'] > 0:
        reasons.append(f"{support['geometry_unconfirmed_pixels']} 个选中像素的 XYZ 待确认，覆盖需求保留")
        by_reason = support.get('unconfirmed_by_reason') or {}
        if by_reason.get('no_surface_reference_depth_present'):
            reasons.append(f"有参考深度但缺面 {by_reason['no_surface_reference_depth_present']} 像素：可评估候选表面，准入仍待确认")
        needs_capture = sum(by_reason.get(key, 0) for key in
                            ('no_surface_no_reference_depth', 'outside_depth_range', 'no_source_pixel'))
        if needs_capture:
            reasons.append(f'缺少有效采集支持 {needs_capture} 像素：核查数据并补采')
        if by_reason.get('depth_conflict'):
            reasons.append(f"深度冲突 {by_reason['depth_conflict']} 像素：核查遮挡、配准和原始观测，必要时补采")
    coverage = block.get('coverage') or {}
    if coverage.get('uncovered_fragment_count', 0):
        reasons.append(f"未规划网格碎片 {coverage['uncovered_fragment_count']} 个：检查碎片尺度与采样条件")
    if coverage.get('status') != 'complete':
        reasons.append('该块几何覆盖记录尚未完整')
    if block.get('dropped_fragment_count', 0):
        reasons.append(f"存在 {block['dropped_fragment_count']} 个丢弃的切片片段，需复核边缘与窄区是否遗漏")
    return reasons


def quality_report_lines(trajectory: SurfaceTrajectory, check: Callable[[], None] = lambda: None) -> list[str]:
    """Human-readable diagnostics only; thresholds, trajectories and completion policy stay fixed."""
    quality = polyline_quality(trajectory, check)
    lines = ['路线规划检查（仅诊断；无新增通过门限）',
             '连续性、几何覆盖、曲面行距、折返平顺性分别检查。',
             f"可播放连续链 {quality['chain_count']} 条；长度只累计链内边，块间/链间未规划移动不计入。",
             f"扫描 {quality['scan_length_m']*1000:.3f} mm；连接 {quality['connector_length_m']*1000:.3f} mm；"
             f"总长 {quality['total_length_m']*1000:.3f} mm；连接占比 "
             + ('未知' if quality['connector_fraction'] is None else f"{quality['connector_fraction']*100:.2f}%") + '。']
    steps = quality['step_mm']
    lines.append('链内采样步长（含零长度边）min / P50 / P95 / max（mm）：' +
                 ('无样本' if steps is None else ' / '.join(f'{steps[k]:.6f}' for k in ('min', 'p50', 'p95', 'max'))))
    lines.append(f"相邻重合点形成的零长度边：{quality['zero_length_edge_count']}；转角诊断临时合并重合点，原轨迹不改写。")
    angles = quality['turn_deg']
    lines.append('离散转角 P50 / P95 / max（度）：' +
                 ('无可计算转角' if angles is None else ' / '.join(f'{angles[k]:.3f}' for k in ('p50', 'p95', 'max'))))
    lines.append('单链连续不等于平滑；转角依赖采样尺度，不是曲率、姿态、加速度或机械臂可执行性判定。')
    for item in quality['largest_turns']:
        block = '未分块' if item['block_index'] is None else f"块 {item['block_index']+1}"
        xyz = ', '.join(f'{v:.6f}' for v in item['xyz_m'])
        lines.append(f"  最大转角定位：{block}，链 {item['segment_index']+1}，route_index={item['route_index']}，"
                     f"row_index={item['row_index']}，{item['angle_deg']:.3f}°，"
                     + ('连接处' if item['kind'] == 'connector' else '扫描段') + f'，W0 XYZ（m）=({xyz})')
    spacing = trajectory.spacing_report or {}
    samples = spacing.get('measurements') or []
    counts = Counter(sample.get('status', 'unknown') for sample in samples)
    labels = dict(within_bounds='界内', outside_bounds='超差', uncertain_bounds='界限不确定', unmeasurable='不可测')
    lines.append(f"完整行距检查：目标 {spacing.get('target_spacing_mm', '未知')} mm，"
                 f"容差 ±{spacing.get('tolerance_mm', '未知')} mm；{len(samples)} 个样本；"
                 + '，'.join(f'{label} {counts[key]}' for key, label in labels.items()))
    if not samples or counts['within_bounds'] != len(samples):
        lines.append('行距待复核：保留全部超差、不确定、不可测样本；不能用过滤后的有效行距代替完整检查。')
    else:
        lines.append('现有行距采样界内；每链最多 32 个索引采样，不能推定全表面覆盖。')
    lines.append('行距使用网格路径上界及欧氏下界；平面等距不保证曲面等距；未给治疗头有效作用宽度与重叠要求，治疗覆盖尚未判定。')
    blocks = trajectory.blocks or {}
    if blocks.get('schema') == SELECTION_BLOCKS_SCHEMA:
        chain_counts = Counter(s.block_index for s in trajectory.ordered_items if isinstance(s, ScanSegment))
        lines.append('按医师选区逐块列出待处理项（不修改既有结果完成/旧文件清理规则）：')
        for block in blocks['items']:
            check()
            reasons = block_pending_reasons(block, chain_counts[block['block_index']])
            lines.append(f"  工作块 {block['block_index']+1}：" +
                         ('待处理；' + '；'.join(reasons) if reasons else '已形成连续单链且几何覆盖记录完整；仍需结合行距与转角检查'))
    else:
        lines.append('旧版轨迹缺少医师选择组件的完整业务记录，本次不推定其符合 v4 工作块完成规则。')
    return lines
