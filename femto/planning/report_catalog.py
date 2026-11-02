"""Pure planning report formatters shared by Qt and headless callers.

This module deliberately imports no GUI toolkit.  Keep report text and other
serializable summaries here so headless planners and Qt code can import the
planning package without initializing a GUI.
"""
from __future__ import annotations

import json
from typing import Any


def quality_text(plan: Any) -> str:
    """Format the route quality comparison stored on *plan*."""
    quality = plan.get("route_quality")
    if not quality:
        return "旧版无路线质量对照；重新生成新版可获得质量与急转角报告。"
    before = quality["before"]
    after = quality["after"]
    lines = [
        "路线质量对照（同一数据和参数，基线 serpentine-v3）",
        "结果：" + (
            "量化改善并通过约束/覆盖核验"
            if quality["status"] == "improved"
            else "未获得通过全部门禁的改善，保留原折线"
        ),
    ]
    for label, key in [
        ("扫描长度", "scan_length_m"),
        ("连接长度", "connector_length_m"),
        ("总长", "total_length_m"),
        ("精确边重复长度", "repeated_edge_length_m"),
    ]:
        lines.append(f"{label}：{before[key] * 1000:.3f} → {after[key] * 1000:.3f} mm")
    for label, key in [
        ("断链数", "disconnected_break_count"),
        ("急转点数", "sharp_turn_count"),
        ("采样覆盖比例%", "sampled_coverage_percent"),
        ("20mm/s已规划时间s", "known_motion_time_s"),
    ]:
        lines.append(f"{label}：{before[key]} → {after[key]}")
    lines += [
        "重复长度只计精确相同无向XYZ边的额外经过，不是扫描带重叠面积。",
        "急转点阈值：相邻三维弦段方向变化≥60°，用于研发标注，不是速度/加速度或接触姿态结论。",
        "每个原已覆盖采样点都必须保留，分母和链数不变；没有以覆盖下降换取更短连接。",
        "算法限原认证边图内三种确定性候选、累计200万展开、每链512扫描段；不新增穿越表面的捷径。",
        "平滑：" + quality["smoothing"]["reason"],
        "墙钟生成耗时在同源对照记录中测量，不写入可复现结果指纹。",
    ]
    labels = {
        "not_improved": "未满足长度/重复非退化门禁",
        "coverage_regression_rejected": "覆盖退化，已拒绝",
        "validated": "核验通过",
        "budget_exceeded": "预算超限，停止搜索",
        "validation_failed": "核验失败",
        "footprint_rejected": "模拟包络未通过",
    }
    for attempt in quality["attempts"]:
        lines.append(
            attempt["method"]
            + "："
            + labels.get(attempt["status"], attempt["status"])
            + " "
            + attempt.get("reason", "")
            + (
                "；丢失原覆盖采样 "
                + str(attempt["lost_covered_samples"])
                + " 个"
                if "lost_covered_samples" in attempt
                else ""
            )
        )
    return "\n".join(lines)


def spacing_text(plan: Any) -> str:
    """Format the surface spacing report stored on *plan*."""
    report = plan["spacing_report"]
    lines = [
        "三维表面间距检查（研发）",
        f"目标 {report['target_spacing_mm']:g} mm；判断容差 ±{report['tolerance_mm']:g} mm。",
        "状态："
        + (
            "采样上下界均在容差内"
            if report["status"] == "within_sampled_bounds"
            else "未完全满足/无法证实：请检查偏差、缺层或上下界不确定，未切换模式"
        ),
        str(report["counts"]),
        "测得网格距离分布（mm）：" + str(report["distribution"]),
        report["method"],
        report["sampling"],
        *report["limitations"],
    ]
    return "\n".join(lines)


def coverage_report_text(plan: Any) -> str:
    """Format the complete coverage report without constructing a GUI."""
    # Import lazily to keep this catalog usable by planning code that only needs
    # the two small text formatters above and to avoid an import cycle at module
    # initialization time.
    from femto.planning.region_path import path_summary

    summary = path_summary(plan)
    coverage = plan.get("coverage", {})
    guard = plan.get("boundary_check")
    support = plan.get("image_support", {})
    lines = [
        "区域覆盖规划检查报告（研发演示）",
        f"路径 {summary['segment_count']} 段 / {summary['sample_count']} 点；段内总长 {summary['total_segment_length_m'] * 1000:.1f} mm。",
        "各段独立；段间动作未规划。此文件不能直接执行机械臂运动。",
    ]
    if plan.get("coverage_parameters", {}).get("pattern") == "serpentine_v1":
        lines[2] = "水平蛇形往返；链内含换行与绕行连接，不同链之间无过渡规划。非机械臂执行轨迹。"
        lines.append("行距在参考深度换算为像素，不保证曲面等距；跨面连接仅用端点估计覆盖。")
    lines += [
        f"扫描长度 {summary['scan_length_m'] * 1000:.3f} mm；连接长度 {summary['connector_length_m'] * 1000:.3f} mm；连续链 {summary['chain_count']}。",
        f"20 mm/s 恒速模拟：已规划链内扫描和连接 {summary['known_motion_time_s']:.3f} s；不含暂停和加减速。",
        "跨链动作未规划，跨链耗时未知。" if summary["chain_count"] > 1 else "单链，无跨链移动。",
    ]
    if plan.get("start_selection"):
        lines.append("方向与起点：" + json.dumps(plan["coverage_parameters"], ensure_ascii=False))
        lines.append("起点选择：" + json.dumps(plan["start_selection"], ensure_ascii=False))
        for gap in plan.get("chain_breaks", []):
            lines.append(
                f"断点：链 {gap['from_chain'] + 1} → 链 {gap['to_chain'] + 1}：{gap['explanation']}；起止 XYZ（米）{gap['from_xyz_m']} → {gap['to_xyz_m']}。"
            )
    if guard:
        lines += [
            f"有效作用宽度（规划占位值） {guard['width_mm']:g} mm；额外边界预留 {guard['margin_mm']:g} mm。",
            f"候选线段 {guard['candidate_edges']}；保留 {guard['retained_edges']}；边界/可见性筛除 {guard['rejected_edges']}。",
            "边界约束针对模拟包络，包含选区外沿及孔洞边界；不是接触力或真实治疗头碰撞验证。",
        ]
    else:
        lines.append("本路径未执行扫描宽度边界约束；建议重新生成区域覆盖规划。")
    if coverage:
        lines += [
            f"有效表面采样分母 {coverage['target_mesh_area_m2'] * 1e6:.1f} mm²；采样覆盖估计 {coverage['sampled_coverage_percent']:.2f}%。",
            f"估计未覆盖面积 {max(0.0, coverage['target_mesh_area_m2'] - coverage['sampled_covered_area_m2']) * 1e6:.1f} mm²。",
            "每面3个采样点的面积加权估计；不能保证全表面覆盖。约束收缩后可能留下边缘遗漏，不自动越界补齐。",
        ]
    if support:
        lines += [
            f"圈选图像像素 {support.get('requested_pixels')}；有几何支持 {support.get('geometry_supported_pixels')}；无支持 {support.get('unsupported_pixels')}。",
            "无几何区域不计入曲面覆盖率分母；像素数不能换算成未知曲面面积。",
        ]
    else:
        lines.append("原始圈选图像支持统计不可用；不应把网格覆盖率理解为整个圈选范围的覆盖率。")
    if plan.get("spacing_report"):
        lines.append(spacing_text(plan))
    if plan.get("route_quality"):
        lines.append(quality_text(plan))
    inspection = plan.get("coverage_inspection")
    if inspection:
        lines += [
            "新版分母：允许、圈内且可信可见的采样代表面积（每面面积/3），含边界预留；排除禁区和无支持采样。并非精确裁剪曲面面积。",
            "无几何支持的三维面积未知，不编造面积，不计入有效表面覆盖率。",
            "方法：" + inspection["method"],
            "采样密度：" + json.dumps(inspection["sampling"], ensure_ascii=False),
            "报告版本：" + inspection["schema"],
            *inspection["limitations"],
            "遗漏/排除列表（同类共边面分组；可在报告图层页点击定位）：",
        ]
        for item in inspection["omissions"]:
            amount = (
                f"采样代表面积 {item['sampled_area_m2'] * 1e6:.3f} mm²"
                if item["sampled_area_m2"] is not None
                else f"{item['pixel_count']}像素，三维面积未知"
            )
            lines.append(f"{item['id']} · {item['label']} · {amount} · 像素{item['bbox_pixel']} · {item['evidence']}")
    else:
        lines.append("旧版报告没有可定位的采样明细和原因分类；未验证新版模拟包络。请重新生成新版覆盖路线以使用遗漏定位。")
    lines.append("W0 / RDF / 米；几何误差、人体运动、接触形变均未独立验证。")
    if plan.get("plan_binding"):
        lines.append("方案版本关联：" + json.dumps(plan["plan_binding"], ensure_ascii=False, sort_keys=True))
    if plan.get("exclusions"):
        lines.append(
            f"禁止区域 {len(plan['exclusions']['polygons'])} 个；剔除相交网格面 {len(plan['exclusions']['removed_triangle_indices'])} 个。覆盖分母仅为允许表面。"
        )
    return "\n".join(lines)


__all__ = ["quality_text", "spacing_text", "coverage_report_text"]
