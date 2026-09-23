"""Static research visualizations for a measured-mesh SurfaceTrajectory."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Callable
import hashlib
import json
import math

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont

from femto.planning.surface_trajectory import Gap, ScanSegment, SurfaceTrajectory, to_player_plan


GAP_COLORS = {
    "forbidden_region": [235, 55, 65, 255],
    "disconnected_surface": [170, 75, 220, 255],
    "surface_gap": [80, 145, 230, 255],
    "unsupported_geometry": [245, 145, 45, 255],
    "row_turn": [135, 145, 155, 255],
    "selection_break": [60, 210, 200, 255],
}
REVISIT_COLOR = "#ff4fd8"


def _font(size: int = 16):
    path = Path("C:/Windows/Fonts/msyh.ttc")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def _manifest(trajectory: SurfaceTrajectory) -> tuple[Path, dict[str, Any]]:
    path = Path(trajectory.source["region_manifest"]).resolve()
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    return path, data


def _region(trajectory: SurfaceTrajectory) -> dict[str, Any] | None:
    try:
        return _manifest(trajectory)[1].get("region")
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def draw_selection_layers(draw: Any, rgb: np.ndarray, region: Any, scale: float, header: int,
                          font: Any) -> tuple[np.ndarray, list[str]]:
    """Tint explicit-unselected / geometry-unconfirmed pixels and outline selection components.

    Returns the tinted image (to paste before drawing paths) and legend lines. Display only."""
    from femto.annotation.selection_intent import selection_contours, selection_layers, tint_selection
    layers = selection_layers(region)
    if layers is None or tuple(layers["mask"].shape) != tuple(np.asarray(rgb).shape[:2]):
        return np.asarray(rgb), []
    tinted = tint_selection(rgb, layers)
    contours = selection_contours(layers)
    lines = ["青色轮廓：医师选择组件（工作块边界）；红色：明确未选（孔洞/禁止）"]
    if layers["classes"] is not None:
        lines[0] += "；紫色：有参考深度但缺面（待确认）；橙色：无可靠几何（需要补采）"
    return tinted, lines


def _draw_selection_outlines(draw: Any, region: Any, rgb_shape: tuple[int, int], scale: float, header: int,
                             font: Any) -> None:
    from femto.annotation.selection_intent import selection_contours, selection_layers
    layers = selection_layers(region)
    if layers is None or tuple(layers["mask"].shape) != tuple(rgb_shape):
        return
    contours = selection_contours(layers)
    for component_id, polyline in contours["components"]:
        points = polyline * scale + [0, header]
        draw.line([tuple(p) for p in np.vstack([points, points[:1]])], fill="#38e6d8", width=2)
    for _, polyline in contours["holes"]:
        points = polyline * scale + [0, header]
        draw.line([tuple(p) for p in np.vstack([points, points[:1]])], fill="#ff5a6e", width=2)
    for component in layers["components"]:
        x0, y0, _, _ = component["bbox"]
        draw.text((x0 * scale + 4, y0 * scale + header + 4), f"选区 {component['component_id'] + 1}", font=font, fill="#38e6d8")


def reference_rgb(trajectory: SurfaceTrajectory) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Return the verified undistorted reference RGB, or an explicit non-fatal status."""
    try:
        _, manifest = _manifest(trajectory)
        origin = manifest["origin_reference"]
        view = manifest["region"]["view_reference"]
        dataset = Path(origin["source_dataset"]).resolve()
        rows = [json.loads(line) for line in (dataset / "frames.jsonl").read_text(encoding="utf-8-sig").splitlines()
                if line.strip()]
        frame = next(row for row in rows if row["frame"] == origin["reference_frame_id"])
        rgb_path = (dataset / frame["rgb"]).resolve()
        if not rgb_path.is_file() or hashlib.sha256(rgb_path.read_bytes()).hexdigest() != view["rgb_sha256"]:
            raise ValueError("reference RGB file/hash mismatch")
        import cv2
        from femto.io_utils import imread_unicode
        from femto.geometry.rgbd_frames import distortion_maps
        map_x, map_y, _, width, height = distortion_maps(dataset)
        bgr = imread_unicode(rgb_path, cv2.IMREAD_COLOR)
        if bgr.shape != (height, width, 3):
            raise ValueError("reference RGB dimensions mismatch")
        rgb = cv2.remap(bgr, map_x, map_y, cv2.INTER_LINEAR)[:, :, ::-1].copy()
        return rgb, {"status": "available", "source": str(rgb_path), "sha256": view["rgb_sha256"]}
    except (OSError, ValueError, TypeError, KeyError, IndexError, StopIteration, json.JSONDecodeError) as error:
        return None, {"status": "unavailable", "reason": str(error)}


def prepare_export_trajectory(trajectory: SurfaceTrajectory) -> SurfaceTrajectory:
    _, status = reference_rgb(trajectory)
    return replace(trajectory, route_diagram_status=status)


def _block_labels(trajectory: SurfaceTrajectory) -> list[tuple[int, str, str]]:
    """(first chain index, label, status) per block, in visit order; display only."""
    if trajectory.blocks is None:
        return []
    from femto.planning.surface_trajectory import block_view
    view = block_view(trajectory)
    labels = []
    for item in view["items"]:
        if item["chain_indices"]:
            labels.append((item["chain_indices"][0], f"块 {item['block_index'] + 1}", item["status"]))
    return labels


def _block_header(trajectory: SurfaceTrajectory) -> str | None:
    stats = trajectory.statistics
    if trajectory.blocks is None or "block_count" not in stats:
        return None
    if trajectory.blocks.get("schema") == "femto-surface-trajectory-blocks-v2":
        return (f"工作块（按医师选择组件）：{stats['block_count']} 块 | 单链完成 {stats['planned_block_count']} | 未完成 "
                f"{stats['incomplete_block_count']} | 无法规划 {stats['unplannable_block_count']} | 覆盖 "
                + ("完整" if stats["coverage_status"] == "complete" else f"部分（未覆盖碎片 {stats['uncovered_fragment_count']}）")
                + (f" | 几何待确认像素 {stats['geometry_unconfirmed_pixel_count']}" if stats.get("geometry_unconfirmed_pixel_count") else "")
                + "；块间只停不连线")
    return (f"分块：{stats['block_count']} 块 | 可规划 {stats['plannable_block_count']} | 未规划 "
            f"{stats['unplannable_block_count']}（{stats['unplannable_face_count']} 面） | "
            + ("全部块已规划" if stats["coverage_status"] == "complete" else "部分覆盖，未规划块见报告")
            + "；块间只标 gap，不画跨块连线")


def _revisit_line(trajectory: SurfaceTrajectory) -> str | None:
    stats = trajectory.statistics
    if trajectory.blocks is None or trajectory.blocks.get("schema") != "femto-surface-trajectory-blocks-v2":
        return None
    count = stats.get("revisit_connector_sample_count", 0)
    if not count:
        return "洋红：无重复经过已扫描行的连接段；连接段不推定治疗能量开启"
    return (f"洋红连接段：{count} 个采样（{stats['revisit_connector_length_m']*1000:.1f} mm）重复经过其他已扫描行"
            "（按名义行距一半判定，非治疗头宽度）；不得据此推定治疗能量持续开启")


def _skipped_label_line(skipped: list[str]) -> str | None:
    if not skipped:
        return None
    shown = "、".join(skipped[:12]) + ("…" if len(skipped) > 12 else "")
    return f"{len(skipped)} 个块编号标签因重叠未显示（{shown}）；编号与状态见 blocks.csv 和报告"


def render_route_diagram(rgb: np.ndarray, trajectory: SurfaceTrajectory, region: Any = None) -> Image.Image:
    from femto.planning.surface_trajectory import non_overlapping_labels
    height, width = rgb.shape[:2]
    scale = min(1.0, 1400 / width, 850 / height)
    display_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    segments = [item for item in trajectory.ordered_items if isinstance(item, ScanSegment)]
    boxes = []
    for chain_index, label, _status in _block_labels(trajectory):
        xy = np.asarray(segments[chain_index].points[0]["pixel"]) * scale
        boxes.append(((chain_index, label), (xy[0] - 2, xy[1] - 20, xy[0] + 12 * len(label) + 4, xy[1] - 2)))
    kept_labels, skipped = non_overlapping_labels(boxes)
    block_line = _block_header(trajectory)
    skipped_line = _skipped_label_line([label for _, label in skipped])
    revisit_line = _revisit_line(trajectory)
    if region is None and trajectory.blocks is not None and trajectory.blocks.get("schema") == "femto-surface-trajectory-blocks-v2":
        region = _region(trajectory)
    base, selection_lines = draw_selection_layers(None, rgb, region, scale, 0, None)
    header = 112 + (28 if block_line else 0) + (28 if skipped_line else 0) + (28 if revisit_line else 0) + 28 * len(selection_lines)
    image = Image.new("RGB", (display_size[0], display_size[1] + header), (18, 28, 38))
    image.paste(Image.fromarray(base).resize(display_size, Image.Resampling.LANCZOS), (0, header))
    draw = ImageDraw.Draw(image)
    font = _font(16)
    stats = trajectory.statistics
    lines = ["三维表面蛇形扫描轨迹 · 固定初始 RGB 视角",
             f"{stats['row_count']} 行 | {stats['segment_count']} 段 | {stats['point_count']} 点 | 段内长度 {stats['scan_length_m']*1000:.1f} mm",
             "黄色：有 Mesh 支持的扫描段；橙色：贴面连接；gap 不连线。研发预览，非设备执行轨迹。"]
    if block_line:
        lines.append(block_line)
    if skipped_line:
        lines.append(skipped_line)
    if revisit_line:
        lines.append(revisit_line)
    lines.extend(selection_lines)
    for index, line in enumerate(lines):
        draw.text((12, 8 + index * 28), line, font=font, fill="#e4edf4")
    _draw_selection_outlines(draw, region, rgb.shape[:2], scale, header, _font(14))
    for segment in segments:
        points = np.asarray([point["pixel"] for point in segment.points], dtype=float) * scale + [0, header]
        draw.line([tuple(point) for point in points], fill="#ffd52b", width=2)
        for index, point in enumerate(segment.points[1:], start=1):
            if point.get("role") == "connector":
                draw.line([tuple(points[index - 1]), tuple(points[index])],
                          fill=REVISIT_COLOR if point.get("revisit") else "#ff9138", width=2)
        for point, color in ((points[0], "#35ea87"), (points[-1], "#ff5765")):
            draw.ellipse((point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3), fill=color)
    for chain_index, label in kept_labels:
        xy = np.asarray(segments[chain_index].points[0]["pixel"]) * scale + [0, header]
        draw.rectangle((xy[0] - 2, xy[1] - 20, xy[0] + 12 * len(label) + 4, xy[1] - 2), fill="#20344a")
        draw.text((xy[0], xy[1] - 20), label, font=font, fill="#7fd8ff")
    if segments:
        for point, label, color in ((segments[0].points[0]["pixel"], "START", "#35ea87"),
                                    (segments[-1].points[-1]["pixel"], "END", "#ff5765")):
            xy = np.asarray(point) * scale + [0, header]
            draw.text(tuple(xy + [7, 0]), label, font=font, fill=color)
    return image


def _depth_color(value: float, low: float, high: float) -> tuple[int, int, int]:
    ratio = 0.5 if high <= low else float(np.clip((value - low) / (high - low), 0, 1))
    stops = np.asarray([[35, 80, 210], [30, 200, 185], [245, 220, 65], [235, 70, 45]], dtype=float)
    position = ratio * (len(stops) - 1)
    index = min(len(stops) - 2, int(position))
    fraction = position - index
    return tuple(int(v) for v in stops[index] * (1 - fraction) + stops[index + 1] * fraction)


def render_depth_diagram(trajectory: SurfaceTrajectory) -> Image.Image:
    frame = trajectory.planner["sweep_frame"]
    origin = np.asarray(frame["origin"])
    e1, e2 = np.asarray(frame["e1"]), np.asarray(frame["e2"])
    segments = [item for item in trajectory.ordered_items if isinstance(item, ScanSegment)]
    xyz = np.asarray([point["xyz_m"] for segment in segments for point in segment.points])
    plane = np.c_[(xyz - origin) @ e1, (xyz - origin) @ e2]
    low, high = plane.min(axis=0), plane.max(axis=0)
    span = np.maximum(high - low, 1e-6)
    canvas = np.array([1200, 760])
    margin = np.array([80, 100])
    scale = float(min((canvas - 2 * margin) / span))
    from femto.planning.surface_trajectory import non_overlapping_labels

    def label_xy(chain_index: int) -> np.ndarray:
        point = np.asarray(segments[chain_index].points[0]["xyz_m"]) - origin
        return np.array([np.dot(point, e1), np.dot(point, e2)]) * scale + margin - low * scale

    boxes = []
    for chain_index, label, _status in _block_labels(trajectory):
        xy = label_xy(chain_index)
        boxes.append(((chain_index, label), (xy[0] + 4, xy[1] - 20, xy[0] + 4 + 12 * len(label), xy[1] - 2)))
    kept_labels, skipped = non_overlapping_labels(boxes)
    block_line = _block_header(trajectory)
    skipped_line = _skipped_label_line([label for _, label in skipped])
    revisit_line = _revisit_line(trajectory)
    header = 110 + (28 if skipped_line else 0) + (28 if revisit_line else 0)
    image = Image.new("RGB", (int(canvas[0]), int(canvas[1] + header)), (18, 28, 38))
    draw = ImageDraw.Draw(image)
    font = _font(16)
    stats = trajectory.statistics
    draw.text((15, 10), "三维表面轨迹 · 扫描面俯视图（e1/e2）", font=font, fill="#e4edf4")
    draw.text((15, 40), f"深度 {stats['depth_min_m']*1000:.1f}–{stats['depth_max_m']*1000:.1f} mm；颜色仅表示 W0 Z",
              font=font, fill="#e4edf4")
    draw.text((15, 70), f"方向 {frame['direction_source']}；法线 {frame['normal_source']}；gap 仅画端点、不连线",
              font=font, fill="#e4edf4")
    outside = {row["chain_index"] for row in trajectory.spacing_report.get("measurements", [])
               if row.get("status") == "outside_bounds"}
    scan_rows=trajectory.spacing_report.get('scan_segment_rows')
    outside_rows={scan_rows[i] for i in outside if i<len(scan_rows)} if scan_rows is not None else set()
    offset = margin - low * scale + [0, header]
    for segment in segments:
        points3 = np.asarray([point["xyz_m"] for point in segment.points])
        points2 = np.c_[(points3 - origin) @ e1, (points3 - origin) @ e2] * scale + offset
        for index in range(len(points2) - 1):
            color = _depth_color(float((points3[index, 2] + points3[index + 1, 2]) / 2),
                                 stats["depth_min_m"], stats["depth_max_m"])
            highlight=(segment.points[index]['row_index'] in outside_rows if scan_rows is not None
                       else segment.segment_index in outside)
            width = 4 if highlight else 2
            if segment.points[index + 1].get("revisit"):
                draw.line([tuple(points2[index]), tuple(points2[index + 1])], fill=REVISIT_COLOR, width=3)
            elif highlight:
                draw.line([tuple(points2[index]), tuple(points2[index + 1])], fill="#ffe34d", width=width)
            else:
                draw.line([tuple(points2[index]), tuple(points2[index + 1])], fill=color, width=width)
    for item in trajectory.ordered_items:
        if not isinstance(item, Gap):
            continue
        for endpoint in (item.previous_point, item.next_point):
            point = np.asarray(endpoint["xyz_m"]) - origin
            xy = np.array([np.dot(point, e1), np.dot(point, e2)]) * scale + offset
            color = tuple(GAP_COLORS[item.reason][:3])
            draw.ellipse((xy[0] - 3, xy[1] - 3, xy[0] + 3, xy[1] + 3), fill=color)
    for chain_index, label in kept_labels:
        xy = label_xy(chain_index) + [0, header]
        draw.text((xy[0] + 4, xy[1] - 20), label, font=font, fill="#7fd8ff")
    if block_line:
        draw.text((15, header - 22), block_line, font=_font(14), fill="#7fd8ff")
    if skipped_line:
        draw.text((15, header - 50), skipped_line, font=_font(14), fill="#7fd8ff")
    if revisit_line:
        draw.text((15, header - (78 if skipped_line else 50)), revisit_line, font=_font(14), fill=REVISIT_COLOR)
    bar_x = int(canvas[0] - 45)
    for y in range(120, int(canvas[1] + header - 50)):
        ratio = (y - 120) / max(1, canvas[1] + header - 170)
        draw.line([(bar_x, y), (bar_x + 18, y)], fill=_depth_color(1 - ratio, 0, 1), width=1)
    draw.text((bar_x - 40, 95), f"{stats['depth_max_m']*1000:.0f} mm", font=font, fill="#e4edf4")
    draw.text((bar_x - 40, int(canvas[1] + header - 45)), f"{stats['depth_min_m']*1000:.0f} mm", font=font, fill="#e4edf4")
    return image


def _gap_markers(trajectory: SurfaceTrajectory) -> dict[str, trimesh.Trimesh]:
    groups: dict[str, list[trimesh.Trimesh]] = {reason: [] for reason in GAP_COLORS}
    for item in trajectory.ordered_items:
        if not isinstance(item, Gap):
            continue
        for endpoint in (item.previous_point, item.next_point):
            marker = trimesh.creation.icosphere(subdivisions=0, radius=0.00045)
            marker.apply_translation(endpoint["xyz_m"])
            groups[item.reason].append(marker)
    result = {}
    for reason, meshes in groups.items():
        if meshes:
            combined = trimesh.util.concatenate(meshes)
            combined.visual.vertex_colors = GAP_COLORS[reason]
            result[reason] = combined
    return result


def export_glb(path: str | Path, trajectory: SurfaceTrajectory) -> None:
    manifest_path, manifest = _manifest(trajectory)
    mesh_path = manifest_path.parent / manifest["files"]["mesh_file"]
    surface = trimesh.load(mesh_path, force="mesh", process=False)
    transform = np.diag([1.0, -1.0, -1.0, 1.0])
    scene = trimesh.Scene()
    scene.add_geometry(surface, node_name="selected_region_W0", geom_name="selected_region", transform=transform)
    player = to_player_plan(trajectory)
    from femto.planning.region_path_glb import direction_markers, path_tubes
    blocks = player.get("blocks")
    if blocks:
        # One display node per planned block in visit order; tubes never join two blocks.
        for item in blocks["items"]:
            if not item["chain_indices"]:
                continue
            subset = {"segments": [player["segments"][index] for index in item["chain_indices"]]}
            name = f"surface_trajectory_block_{item['block_index'] + 1:03d}"
            scene.add_geometry(path_tubes(subset), node_name=name + "_W0", geom_name=name, transform=transform)
    else:
        scene.add_geometry(path_tubes(player), node_name="surface_trajectory_W0",
                           geom_name="surface_trajectory", transform=transform)
    for name, marker in direction_markers(player).items():
        scene.add_geometry(marker, node_name=name + "_W0", geom_name=name, transform=transform)
    for reason, marker in _gap_markers(trajectory).items():
        scene.add_geometry(marker, node_name="gap_" + reason + "_W0",
                           geom_name="gap_" + reason, transform=transform)
    scene.metadata.update(schema="femto-surface-trajectory-glb-v1", result_sha256=trajectory.to_dict()["result_sha256"],
                          coordinate_frame="world_rdf", unit="metre", execution_ready=False,
                          gap_display="colored_endpoint_markers_only_no_tubes",
                          glb_scene_from_W0=transform.tolist())
    if blocks:
        scene.metadata.update(blocks={"summary": blocks["summary"], "order_rule": blocks["order_rule"],
                                      "items": [{key: item.get(key) for key in ("block_index", "component_id", "status",
                                                                                "reason", "face_count", "chain_indices")}
                                                for item in blocks["items"]],
                                      "display": "one_tube_node_per_planned_block_no_inter_block_tubes"})
        if blocks.get("selection_blocks"):
            scene.metadata.update(selection_blocks={
                "block_definition": blocks.get("block_definition"),
                "revisit_display": "magenta_connector_tubes_energy_not_assumed_on",
                "selection_break_display": "cyan_endpoint_markers_no_tube",
                "items": [{key: item.get(key) for key in ("block_index", "status", "continuity", "coverage",
                                                          "mesh_fragment_count", "planned_fragment_count")}
                          for item in blocks["items"]]})
    Path(path).write_bytes(scene.export(file_type="glb"))


def _spacing_outlier_lines(trajectory: SurfaceTrajectory) -> list[str]:
    report = trajectory.spacing_report
    if "effective_spacing_statistics" not in report:
        return []
    effective = report["effective_spacing_statistics"]
    return [f"有效行距（排除 > {report['spacing_outlier_threshold_mm']:g} mm 的跨孔洞/缺面上界，共 "
            f"{report['spacing_outlier_count']} 个异常样本，占 "
            f"{(report['spacing_outlier_ratio'] or 0)*100:.1f}%）：" +
            (json.dumps(effective, ensure_ascii=False, sort_keys=True) if effective else "无有效样本"),
            "异常样本是图最短路上界，通常跨越孔洞或缺失表面，不代表真实排行间距；上一行为含异常值的原始分布。"]


def _selection_block_report_lines(trajectory: SurfaceTrajectory) -> list[str]:
    stats = trajectory.statistics
    report = trajectory.gap_report
    unconfirmed_count = stats.get("geometry_unconfirmed_pixel_count")
    unconfirmed_text = "未知（导出无逐像素支持记录）" if unconfirmed_count is None else str(unconfirmed_count)
    lines = [
        f"工作块规划（{trajectory.planner['algorithm_version'].rsplit('-', 1)[-1]}，块 = 医师选择组件；网格连通性只作碎片诊断）：{stats['block_count']} 块，单链完成 {stats['planned_block_count']}，"
        f"未完成 {stats['incomplete_block_count']}，无法规划 {stats['unplannable_block_count']}；"
        f"路线连续状态：{'全部块为连续单链' if stats['continuity_status']=='complete' else '存在未完成块（旧结果保留，不自动替换）'}",
        f"覆盖状态（与连续性分开检查）：{ {'complete': '完整', 'partial': '部分'}.get(stats['coverage_status'], '碎片完整但几何支持未知') }；"
        f"未覆盖网格碎片 {stats['uncovered_fragment_count']} 个 / {stats['uncovered_face_count']} 面 / {stats['unplannable_area_m2']*1e6:.1f} mm²；"
        f"几何待确认像素 {unconfirmed_text}（无治疗头有效宽度依据，不宣称完整治疗覆盖）",
        f"块间断开（选择分离）：{stats['inter_block_gap_count']} 处 selection_break；块内换行折返 {stats['connected_row_turns']} 处，"
        f"同行绕行 {stats['connected_detours']} 处；块内未连接 gap 原因（仅诊断）："
        + json.dumps(report.get("link_failure_counts", {}), ensure_ascii=False, sort_keys=True),
        f"重复经过：{stats['revisit_connector_sample_count']} 个连接采样 / {stats['revisit_connector_length_m']*1000:.3f} mm "
        "落在其他已扫描行的名义半行距内（非治疗头宽度）；连接段不得推定治疗能量持续开启",
        "gap 原因×推断：" + json.dumps(report.get("gap_cause_counts", {}), ensure_ascii=False, sort_keys=True),
    ]
    from femto.planning.surface_trajectory import FACE_PORTAL_SELECTION_ALGORITHM, OPTIMIZED_SELECTION_ALGORITHM
    version = trajectory.planner['algorithm_version']
    if version in (OPTIMIZED_SELECTION_ALGORITHM, FACE_PORTAL_SELECTION_ALGORITHM):
        shortened = sum(fragment.get('link_methods', {}).get('surface_corridor_shortened', 0)
                        for block in trajectory.blocks['items'] for fragment in block.get('fragments', []))
        lines.append(f"可信三角面内缩短连接 {shortened} 处：扫描行、连接端点和成功边界折返保持不变；"
                     "不是全局最短路或运动平滑，不补面、不消除缺失几何的覆盖需求。")
    if version == FACE_PORTAL_SELECTION_ALGORITHM:
        portals = sum(fragment.get('link_methods', {}).get('surface_face_portals', 0)
                      for block in trajectory.blocks['items'] for fragment in block.get('fragments', []))
        lines.append(f"面内共享边补充连接 {portals} 处：仅在旧方法失败时，经真实共边内部找路；"
                     "每段有单面支持且整边不穿未选区域/禁区。有限候选搜索不保证找到所有可行路线；"
                     "仍无法连接或缺少几何支持的位置继续待处理，不补面、不跨块。")
    unconfirmed = report.get("geometry_unconfirmed")
    if unconfirmed:
        lines.append("几何待确认像素按原因：" + json.dumps(unconfirmed["by_reason"], ensure_ascii=False, sort_keys=True)
                     + f"；派生表面状态 {unconfirmed['derived_surface']['status']}（{unconfirmed['derived_surface']['reason']}），"
                     "未生成任何伪实测 XYZ")
    else:
        lines.append("区域导出没有逐像素几何支持记录；几何待确认像素未知（请重新选择并导出以获得记录）")
    for block in trajectory.blocks["items"]:
        selection = block.get("selection") or {}
        support = selection.get("support") or {}
        head = (f"  块 {block['block_index']+1}（选择组件 {block['component_id']+1}，选中像素 {selection.get('selected_pixels')}，"
                f"内部未选孔洞 {selection.get('hole_count')}，几何待确认像素 {support.get('geometry_unconfirmed_pixels')}，"
                f"网格碎片 {block['mesh_fragment_count']}）：")
        if block["status"] == "planned":
            lines.append(head + f"已完成单链，{block['face_count']} 面，{block['row_count']} 行 / {block['point_count']} 点，扫描 "
                         f"{block['scan_length_m']*1000:.3f} mm，折返 {block['connected_row_turns']} 处 / 绕行 {block['connected_detours']} 处 / "
                         f"连接 {block['connector_length_m']*1000:.3f} mm，重复经过 {block['revisit']['connector_sample_count']} 采样，"
                         f"覆盖 {block['coverage']['status']}（未覆盖碎片 {block['coverage']['uncovered_fragment_count']}）")
        elif block["status"] == "incomplete":
            lines.append(head + f"未完成：{block['continuity']['chain_count']} 段路线 / {block['gap_count']} 处块内断开，原因 "
                         + json.dumps(block["link_failure_counts"], ensure_ascii=False, sort_keys=True)
                         + f"；{block['face_count']} 面，扫描 {block['scan_length_m']*1000:.3f} mm，覆盖 {block['coverage']['status']}")
        else:
            lines.append(head + f"无法规划，原因 {block['reason']}：{block['message']}")
        for fragment in block.get("fragments", []):
            if fragment["status"] == "planned":
                lines.append(f"    碎片 {fragment['fragment_index']+1}：{fragment['face_count']} 面，{fragment['row_count']} 行，"
                             f"{fragment['chain_count']} 链，块内 gap {fragment['internal_gap_count']}，连接方式 "
                             + json.dumps(fragment['link_methods'], ensure_ascii=False, sort_keys=True))
            else:
                lines.append(f"    碎片 {fragment['fragment_index']+1}：{fragment['face_count']} 面，未规划（{fragment['reason']}）→ 覆盖缺口，选中表面未被轨迹到达")
    return lines


def _block_report_lines(trajectory: SurfaceTrajectory) -> list[str]:
    stats = trajectory.statistics
    if trajectory.blocks is None or "block_count" not in stats:
        return []
    if trajectory.blocks.get("schema") == "femto-surface-trajectory-blocks-v2":
        return _selection_block_report_lines(trajectory)
    lines = [
        f"分块规划：{stats['block_count']} 块（按网格面连通关系，不按外观或距离合并）；可规划 {stats['plannable_block_count']} 块，"
        f"未规划 {stats['unplannable_block_count']} 块（{stats['unplannable_face_count']} 面 / {stats['unplannable_area_m2']*1e6:.1f} mm²）；"
        f"覆盖状态：{'全部块已规划' if stats['coverage_status']=='complete' else '部分覆盖，未规划块不在轨迹内'}",
        f"块间断开 gap：{stats['inter_block_gap_count']} 处（disconnected_surface）；块内扫描先完成再进入下一块；"
        "块访问顺序：参考图像从起始角开始的阅读顺序（不可投影块排最后）",
        f"块内换行：应连接 {stats['row_count'] - stats['plannable_block_count']} 处，已贴面折返 {stats.get('connected_row_turns', 0)} 处，"
        f"仍为 row_turn gap {stats['gap_counts'].get('row_turn', 0)} 处；未连接原因（仅诊断）："
        + json.dumps(trajectory.gap_report.get("row_turn_failure_counts", {}), ensure_ascii=False, sort_keys=True),
        "上述扫描方向/法线来源仅为整体展示坐标系；每块扫描方向按块单独计算，见 blocks 记录。",
    ]
    for block in trajectory.blocks["items"]:
        if block["status"] == "planned":
            lines.append(
                f"  块 {block['block_index']+1}：已规划，{block['face_count']} 面，{block['row_count']} 行 / "
                f"{block['segment_count']} 段 / {block['gap_count']} gap / {block['point_count']} 点，扫描 "
                f"{block['scan_length_m']*1000:.3f} mm，折返 {block['connected_row_turns']} 处 / "
                f"{block['connector_length_m']*1000:.3f} mm，方向 {block['sweep_frame']['direction_source']}，"
                f"法线 {block['sweep_frame']['normal_source']}，未连换行原因 "
                + json.dumps(block.get('row_turn_failure_counts', {}), ensure_ascii=False, sort_keys=True))
        else:
            lines.append(f"  块 {block['block_index']+1}：未规划，{block['face_count']} 面，原因 {block['reason']}：{block['message']}")
    return lines


def report_text(trajectory: SurfaceTrajectory) -> str:
    from femto.planning.trajectory_quality import quality_report_lines
    stats = trajectory.statistics
    frame = trajectory.planner["sweep_frame"]
    spacing = trajectory.spacing_report.get("distribution")
    route_status = trajectory.route_diagram_status or {"status": "unknown"}
    return "\n".join([
        "三维表面蛇形扫描轨迹报告（研发）",
        "算法版本：" + trajectory.planner['algorithm_version'],
        f"行 / 扫描段 / gap / 点：{stats['row_count']} / {stats['segment_count']} / {stats['gap_count']} / {stats['point_count']}",
        f"段内扫描长度：{stats['scan_length_m']*1000:.3f} mm（不含任何 gap 弦长）",
        f"贴面折返连接：{stats.get('connected_row_turns',0)} 处 / {stats.get('connector_length_m',0)*1000:.3f} mm",
        f"W0 深度范围：{stats['depth_min_m']*1000:.3f}–{stats['depth_max_m']*1000:.3f} mm",
        "gap 分类：" + json.dumps(stats["gap_counts"], ensure_ascii=False, sort_keys=True),
        f"扫描方向来源：{frame['direction_source']}；法线来源：{frame['normal_source']}；特征值比："
        + json.dumps(frame['ratios'], ensure_ascii=False, sort_keys=True),
        "实际网格间距分布（mm）：" + (json.dumps(spacing, ensure_ascii=False, sort_keys=True) if spacing else "无可测样本"),
        *_spacing_outlier_lines(trajectory),
        "参考 RGB 路线图：" + json.dumps(route_status, ensure_ascii=False, sort_keys=True),
        *_block_report_lines(trajectory),
        *quality_report_lines(trajectory),
        "平面等距不等于曲面等距；黄色行表示采样间距报告超出容差。",
        "所有点仅证明受当前 TSDF Mesh 支持；gap 处没有规划运动。块内连接保持已有三角面支持并避让未选区域，单链不代表平滑。",
        "没有姿态、速度、避障、自由空间过渡或机械臂接口。",
        "execution_ready=false；physical_accuracy_verified=false；不得用于临床精度、接触安全或设备执行声明。",
    ]) + "\n"


def write_visual_outputs(folder: Path, trajectory: SurfaceTrajectory,
                         check: Callable[[], None] = lambda: None) -> None:
    check()
    export_glb(folder / "region_with_trajectory.glb", trajectory)
    check()
    render_depth_diagram(trajectory).save(folder / "depth_diagram.png")
    rgb, status = reference_rgb(trajectory)
    if status != trajectory.route_diagram_status:
        raise ValueError("Reference RGB availability changed during trajectory export")
    if rgb is not None:
        check()
        render_route_diagram(rgb, trajectory, _region(trajectory)).save(folder / "route_diagram.png")
    check()
    (folder / "trajectory_report.txt").write_text(report_text(trajectory), encoding="utf-8-sig")
