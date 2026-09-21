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
}


def _font(size: int = 16):
    path = Path("C:/Windows/Fonts/msyh.ttc")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def _manifest(trajectory: SurfaceTrajectory) -> tuple[Path, dict[str, Any]]:
    path = Path(trajectory.source["region_manifest"]).resolve()
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    return path, data


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


def render_route_diagram(rgb: np.ndarray, trajectory: SurfaceTrajectory) -> Image.Image:
    height, width = rgb.shape[:2]
    scale = min(1.0, 1400 / width, 850 / height)
    display_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    header = 112
    image = Image.new("RGB", (display_size[0], display_size[1] + header), (18, 28, 38))
    image.paste(Image.fromarray(rgb).resize(display_size, Image.Resampling.LANCZOS), (0, header))
    draw = ImageDraw.Draw(image)
    font = _font(16)
    stats = trajectory.statistics
    lines = ["三维表面蛇形扫描轨迹 · 固定初始 RGB 视角",
             f"{stats['row_count']} 行 | {stats['segment_count']} 段 | {stats['point_count']} 点 | 段内长度 {stats['scan_length_m']*1000:.1f} mm",
             "黄色：有 Mesh 支持的扫描段；gap 不连线。研发预览，非设备执行轨迹。"]
    for index, line in enumerate(lines):
        draw.text((12, 8 + index * 28), line, font=font, fill="#e4edf4")
    segments = [item for item in trajectory.ordered_items if isinstance(item, ScanSegment)]
    for segment in segments:
        points = np.asarray([point["pixel"] for point in segment.points], dtype=float) * scale + [0, header]
        draw.line([tuple(point) for point in points], fill="#ffd52b", width=2)
        for point, color in ((points[0], "#35ea87"), (points[-1], "#ff5765")):
            draw.ellipse((point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3), fill=color)
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
    header = 110
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
            if highlight:
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
    Path(path).write_bytes(scene.export(file_type="glb"))


def report_text(trajectory: SurfaceTrajectory) -> str:
    stats = trajectory.statistics
    frame = trajectory.planner["sweep_frame"]
    spacing = trajectory.spacing_report.get("distribution")
    route_status = trajectory.route_diagram_status or {"status": "unknown"}
    return "\n".join([
        "三维表面蛇形扫描轨迹报告（研发）",
        f"行 / 扫描段 / gap / 点：{stats['row_count']} / {stats['segment_count']} / {stats['gap_count']} / {stats['point_count']}",
        f"段内扫描长度：{stats['scan_length_m']*1000:.3f} mm（不含任何 gap 弦长）",
        f"贴面折返连接：{stats.get('connected_row_turns',0)} 处 / {stats.get('connector_length_m',0)*1000:.3f} mm",
        f"W0 深度范围：{stats['depth_min_m']*1000:.3f}–{stats['depth_max_m']*1000:.3f} mm",
        "gap 分类：" + json.dumps(stats["gap_counts"], ensure_ascii=False, sort_keys=True),
        f"扫描方向来源：{frame['direction_source']}；法线来源：{frame['normal_source']}；特征值比：{frame['ratios']}",
        "实际网格间距分布（mm）：" + (json.dumps(spacing, ensure_ascii=False, sort_keys=True) if spacing else "无可测样本"),
        "参考 RGB 路线图：" + json.dumps(route_status, ensure_ascii=False, sort_keys=True),
        "平面等距不等于曲面等距；黄色行表示采样间距报告超出容差。",
        "所有点仅证明受当前 TSDF Mesh 支持；gap 处没有规划运动。贴面折返仅连接已验证连续网格边界。",
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
        render_route_diagram(rgb, trajectory).save(folder / "route_diagram.png")
    check()
    (folder / "trajectory_report.txt").write_text(report_text(trajectory), encoding="utf-8-sig")
