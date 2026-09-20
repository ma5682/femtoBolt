"""One-command end-to-end Femto scene reconstruction demonstration."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np
import open3d as o3d

from fusion import read_geometry_safe
from scene_data_kind import classify_dataset_data_kind
from scene_dataset_validator import load_json, load_jsonl, resolve_dataset, validate_dataset
from scene_pose_estimator import estimate_poses
from scene_reconstruction import reconstruct_scene
from scene_to_nerfstudio import convert
from synthetic_scene_demo import generate_dataset


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASETS = ROOT / "datasets"
DEFAULT_OUTPUTS = ROOT / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run capture-format validation through TSDF and 3DGS initialization")
    parser.add_argument("--mode", choices=("synthetic", "latest"), default="synthetic")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--datasets-root", type=Path, default=DEFAULT_DATASETS)
    parser.add_argument("--outputs-root", type=Path, default=DEFAULT_OUTPUTS)
    parser.add_argument("--synthetic-frames", type=int, default=14)
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--no-publish-latest", dest="publish_latest", action="store_false")
    parser.add_argument("--no-source-report", action="store_true")
    parser.add_argument("--force-nerfstudio", action="store_true",
                        help="Create full Nerfstudio assets even when a local capture has too few views")
    return parser.parse_args()


def _pose_complete(dataset: Path) -> bool:
    frames = load_jsonl(dataset / "frames.jsonl")
    if not frames:
        return False
    for frame in frames:
        record = load_json(dataset / frame["pose"])
        if not record.get("pose_valid", False) or record.get("T_WC") is None:
            return False
    return True


def _depth_range_mm(dataset: Path) -> tuple[float, float]:
    manifest = load_json(dataset / "manifest.json")
    parameters = manifest.get("parameters", {})
    range_m = manifest.get("camera", {}).get("depth_range_m")
    minimum = float(parameters.get("depth_min_mm", 1000.0 * range_m[0] if range_m else 500.0))
    maximum = float(parameters.get("depth_max_mm", 1000.0 * range_m[1] if range_m else 3860.0))
    return minimum, maximum


def _classify_data_kind(dataset: Path) -> str:
    """Classify the input lineage without treating a mode label as proof.

    Synthetic demos carry explicit generator markers in their manifest.  A
    real capture must have the canonical Femto Bolt device value written by
    the capture adapter.  If neither is present, keep the result ``unknown``
    rather than manufacturing a real-data claim.
    """
    return classify_dataset_data_kind(dataset)


def _validator_args(dataset: Path) -> argparse.Namespace:
    minimum_mm, maximum_mm = _depth_range_mm(dataset)
    return argparse.Namespace(
        dataset=dataset,
        datasets_root=DEFAULT_DATASETS,
        require_pose=True,
        max_sync_delta_ms=20.0,
        min_valid_depth_ratio=0.01,
        min_depth_mm=minimum_mm,
        max_depth_mm=maximum_mm,
        no_write_report=False,
    )


def _read(path: Path, flags: int) -> np.ndarray:
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, flags) if encoded.size else None
    if image is None:
        raise ValueError(f"Cannot decode {path}")
    return image


def _write(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError(f"Cannot encode {path}")
    path.write_bytes(encoded.tobytes())


def _contact_sheet(dataset: Path, output: Path) -> None:
    minimum_mm, maximum_mm = _depth_range_mm(dataset)
    frames = load_jsonl(dataset / "frames.jsonl")
    indexes = sorted(set((0, len(frames) // 2, len(frames) - 1)))
    tiles: list[np.ndarray] = []
    for index in indexes:
        frame = frames[index]
        color = _read(dataset / frame["rgb"], cv2.IMREAD_COLOR)
        depth = _read(dataset / frame["depth"], cv2.IMREAD_UNCHANGED)
        valid = depth > 0
        normalized = np.zeros_like(depth, dtype=np.uint8)
        normalized[valid] = np.clip(
            (depth[valid].astype(np.float32) - minimum_mm)
            / (maximum_mm - minimum_mm)
            * 255.0,
            0,
            255,
        ).astype(np.uint8)
        coloured_depth = cv2.applyColorMap(255 - normalized, cv2.COLORMAP_TURBO)
        coloured_depth[~valid] = 0
        target_width = 480
        target_height = max(1, int(round(color.shape[0] * target_width / color.shape[1])))
        color = cv2.resize(color, (target_width, target_height), interpolation=cv2.INTER_AREA)
        coloured_depth = cv2.resize(coloured_depth, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
        cv2.putText(color, f"RGB frame {frame['frame']}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(coloured_depth, "Depth mm", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(np.vstack((color, coloured_depth)))
    _write(output, np.hstack(tiles))


def _write_html_report(output: Path, summary: dict[str, Any]) -> None:
    metrics = summary["reconstruction_metrics"]
    if summary["mode"] == "synthetic":
        mode_text = "内置已知位姿合成场景"
    elif summary.get("fixed_camera", False):
        mode_text = "Femto 固定机位单视角场景"
    else:
        mode_text = "Femto 移动多视角实拍场景"
    gaussian_status = "初始化 PLY（未训练，不能当作写实 3DGS 成品）"
    low_coverage = bool(summary.get("low_depth_coverage_warning", False))
    quality_text = "LOW DEPTH" if low_coverage else "流程完成"
    quality_class = "warn" if low_coverage else "ok"
    accuracy = summary.get("synthetic_ground_truth_error")
    single_quality = summary.get("single_frame_quality")
    single_card = ""
    if single_quality:
        clean = single_quality["cleaning"]
        comparisons = single_quality["surface_consistency"]
        single_card = (
            '<h2>单帧建模质量对照</h2>'
            f'<p>范围内点数 {clean["in_range_pixels"]:,}；去除孤立点 {clean["isolated_pixels_removed"]:,}；'
            f'保留 {clean["retained_pixels"]:,}。此清理步骤仅去除孤立点；上游融合情况见数据来源报告。</p>'
            f'<p>源点云到网格 P95：{comparisons["raw_to_mesh"]["p95_mm"]:.3f} mm；'
            f'清理后点云到网格 P95：{comparisons["cleaned_to_mesh"]["p95_mm"]:.3f} mm。</p>'
            f'<p>源点云到网格采样最大偏差：{comparisons["raw_to_mesh"]["max_mm"]:.3f} mm；'
            f'偏差大于 10 mm 的采样点：{comparisons["raw_to_mesh"]["fraction_over_10mm"]:.2%}。'
            '此阈值仅用于定位缺面与边缘差异，不是验收容差。</p>'
            '<p>以上为同源几何一致性，真实尺寸精度与 2.21 米端点效果待实测。</p>'
            '<p>相邻像素网格直接使用源点坐标，因此同源距离可为零；这不是测量精度为零误差。带纹理模型默认显示原始 RGB，缺深度与不连续边界保持空缺。</p>'
            '<p>模型查看器可同时勾选“重建 Mesh”和“未清理点云（对照）”。</p>'
            '<p><a href="reconstruction/single_frame_quality.json">详细质量报告</a> · '
            '<a href="reconstruction/source_evidence.json">原始数据校验清单</a></p>'
        )
    accuracy_card = (
        f'<div class="card"><div class="value">{accuracy["p95_m"] * 1000.0:.1f} mm</div><div class="label">合成真值 P95 误差</div></div>'
        if accuracy
        else '<div class="card"><div class="value">REAL</div><div class="label">实拍输入</div></div>'
    )
    report = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Femto Complete Demo Report</title>
<style>
body{{margin:0;background:#0b1118;color:#e8f0f8;font-family:Segoe UI,Microsoft YaHei,sans-serif}}
main{{max-width:1180px;margin:auto;padding:38px}} h1{{font-size:32px;margin:0 0 8px}} .sub{{color:#9cb1c7}}
.grid{{display:grid;grid-template-columns:repeat(5,1fr);gap:14px;margin:26px 0}} .card{{background:#151f2b;border:1px solid #263649;border-radius:14px;padding:18px}}
.value{{font-size:28px;color:#70e1a1;font-weight:700}} .label{{color:#9cb1c7;margin-top:6px}} img{{width:100%;border-radius:12px;border:1px solid #263649}}
.ok{{color:#70e1a1}} .warn{{color:#ffc76a}} code{{color:#b8d7ff;word-break:break-all}} li{{margin:10px 0}} @media(max-width:800px){{.grid{{grid-template-columns:1fr 1fr}}}}
</style></head><body><main>
<h1>Femto 全场景三维重建测试 Demo</h1><p class="sub">{html.escape(mode_text)} · {html.escape(summary['created_local'])}</p>
<div class="grid">
<div class="card"><div class="value">{summary['frame_count']}</div><div class="label">RGB-D 帧</div></div>
<div class="card"><div class="value">{metrics['fused_point_count']:,}</div><div class="label">处理后点云</div></div>
<div class="card"><div class="value">{metrics['mesh_triangles']:,}</div><div class="label">Mesh 三角面</div></div>
<div class="card"><div class="value {quality_class}">{quality_text}</div><div class="label">数据、Pose 与深度覆盖</div></div>
{accuracy_card}
</div>
<img src="rgbd_contact_sheet.jpg" alt="RGB-D contact sheet">
{single_card}
<h2>已跑通</h2><ul>
<li class="ok">RGB + 对齐 Depth + Intrinsic + Timestamp + 每帧 T_WC</li>
<li class="ok">完整性/同步/深度/Pose 校验</li><li class="ok">世界坐标 Point Cloud PLY</li>
<li class="ok">可见表面 Mesh：PLY / OBJ / GLB</li><li class="ok">Nerfstudio/gsplat 输入：去畸变 RGB-D + OpenGL 相机位姿</li>
<li class="warn">Gaussian：{gaussian_status}</li></ul>
{('<p class="warn">有效深度覆盖率低于 35%；结果只表示少量实测表面，不能作为完整场景模型。</p>' if low_coverage else '')}
<h2><a href="result_quality.html">实拍验收与结果质量报告</a></h2>
<p>查看关键帧筛选、配准质量、待实测项目与验收模板；重建完成不代表实物精度通过。</p>
<h2>结果位置</h2><ul>
<li>源数据：<code>{html.escape(summary['source_dataset'])}</code></li>
<li>重建结果：<code>{html.escape(summary['reconstruction'])}</code></li>
<li>Nerfstudio 输入：<code>{html.escape(summary['nerfstudio'] or '未生成 / not generated')}</code></li></ul>
<p class="warn">真实实拍模式使用顺序 RGB-D 里程计时会累积漂移，正式 eye-in-hand 产品仍应接入同步机器人 T_BF 与手眼 T_FC。</p>
</main></body></html>"""
    (output / "demo_report.html").write_text(report, encoding="utf-8")


def run_demo(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    if args.mode == "synthetic":
        dataset = generate_dataset(args.datasets_root, frames=args.synthetic_frames)
        posed_dataset = dataset
        pose_stage = "synthetic_ground_truth"
    else:
        dataset = resolve_dataset(args.dataset, args.datasets_root)
        if _pose_complete(dataset):
            posed_dataset = dataset
            pose_stage = "existing_valid_pose"
        else:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            posed_dataset, _ = estimate_poses(
                dataset, dataset / "derived" / f"posed_rgbd_odometry_{stamp}"
            )
            pose_stage = "open3d_sequential_hybrid_rgbd_odometry"

    validator_args = _validator_args(posed_dataset)
    if getattr(args, 'no_source_report', False):
        validator_args.no_write_report = True
    validation, code = validate_dataset(validator_args)
    if code != 0 or not validation.get("ready_for_multiview_reconstruction", False):
        raise ValueError(f"Dataset validation did not pass: {validation.get('errors', [])}")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    result_root = args.outputs_root.resolve() / f"complete_demo_{stamp}"
    suffix = 1
    while result_root.exists():
        result_root = args.outputs_root.resolve() / f"complete_demo_{stamp}_{suffix:02d}"
        suffix += 1
    result_root.mkdir(parents=True, exist_ok=False)
    publication_id = getattr(args, 'publication_id', None)
    if publication_id:
        from local_processing import write_json
        write_json(result_root / 'publication.json', dict(schema='femto-local-publication-v1',
            status='pending', session_id=publication_id, result=str(result_root)))
        if getattr(args, 'created_result_file', None):
            write_json(Path(args.created_result_file), dict(result=str(result_root), session_id=publication_id))
    if getattr(args, 'no_source_report', False):
        (result_root / 'validation_report.json').write_text(
            json.dumps(validation, ensure_ascii=False, indent=2), encoding='utf-8')
    reconstruction_dir = result_root / "reconstruction"
    nerfstudio_dir = result_root / "nerfstudio"
    minimum_mm, maximum_mm = _depth_range_mm(posed_dataset)
    posed_manifest = load_json(posed_dataset / "manifest.json")
    fixed_camera = bool(posed_manifest.get("parameters", {}).get("fixed_camera", False))
    frames = load_jsonl(posed_dataset / "frames.jsonl")
    single_fixed_view = fixed_camera and len(frames) == 1
    reconstruction = reconstruct_scene(
        posed_dataset,
        reconstruction_dir,
        image_scale=1.0,
        voxel_m=0.005,
        sdf_trunc_m=0.02,
        depth_min_m=minimum_mm / 1000.0,
        depth_max_m=maximum_mm / 1000.0,
        point_stride=1,
        single_frame_quality=bool(single_fixed_view and minimum_mm == 250 and maximum_mm == 2210),
    )
    local_measured=bool(posed_manifest.get('local_capture',{}).get('measured_only'))
    skip_nerfstudio=local_measured and len(frames)<12 and not getattr(args,'force_nerfstudio',False)
    nerfstudio = None if skip_nerfstudio else convert(posed_dataset, nerfstudio_dir, allow_static=fixed_camera)
    _contact_sheet(posed_dataset, result_root / "rgbd_contact_sheet.jpg")
    low_depth_coverage = bool(
        fixed_camera and reconstruction["metrics"]["valid_depth_ratio_mean"] < 0.35
    )
    summary = {
        "schema": "femto-complete-demo-v1",
        "created_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "status": (
            "warning_low_depth_coverage_geometry_gaussian_init_untrained"
            if low_depth_coverage
            else "pass_geometry_gaussian_init_untrained"
        ),
        "mode": args.mode,
        # Keep an explicit lineage label in every demo summary.  ``mode`` is
        # retained for backwards compatibility, while downstream acceptance
        # and scale reports can unambiguously distinguish synthetic rehearsal
        # from a real Femto capture.
        "data_kind": _classify_data_kind(posed_dataset),
        "raw_dataset": str(dataset),
        "source_dataset": str(posed_dataset),
        "pose_stage": pose_stage,
        "fixed_camera": fixed_camera,
        "low_depth_coverage_warning": low_depth_coverage,
        "frame_count": len(frames),
        "validation": {
            "ready_for_multiview_reconstruction": validation["ready_for_multiview_reconstruction"],
            "metrics": validation.get("metrics", {}),
        },
        "reconstruction": str(reconstruction_dir),
        "reconstruction_metrics": reconstruction["metrics"],
        "mesh_method": reconstruction["parameters"]["mesh_method"],
        "nerfstudio": None if skip_nerfstudio else str(nerfstudio_dir),
        "nerfstudio_conversion": nerfstudio,
        "gaussian_status": "initialization_only_untrained",
    }
    if skip_nerfstudio:
        summary.update(nerfstudio_skipped_reason='local_capture_has_fewer_than_12_views',
                       reconstruction_source_mode='moving', local_measured=True)
    if publication_id:
        summary.update(publication_required=True, publication_id=publication_id)
    if posed_manifest.get('local_capture', {}).get('capture_scope') == 'interrupted_prefix':
        summary.update(status='interrupted_prefix_geometry_gaussian_init_untrained',
                       capture_scope='interrupted_prefix',
                       capture_label='中断前数据建模 / Interrupted-prefix reconstruction',
                       interrupted_prefix=posed_manifest['local_capture']['interrupted_prefix'])
    if args.mode == "synthetic":
        cloud = read_geometry_safe(reconstruction_dir / "scene_fused_points_m.ply", mesh=False)
        ground_truth = read_geometry_safe(
            posed_dataset / "ground_truth" / "synthetic_scene_ground_truth_m.ply", mesh=True
        )
        ray_scene = o3d.t.geometry.RaycastingScene()
        ray_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(ground_truth))
        distances = ray_scene.compute_distance(
            o3d.core.Tensor(np.asarray(cloud.points), dtype=o3d.core.Dtype.Float32)
        ).numpy()
        summary["synthetic_ground_truth_error"] = {
            "median_m": float(np.median(distances)),
            "p95_m": float(np.quantile(distances, 0.95)),
            "max_m": float(np.max(distances)),
        }
    if reconstruction.get("outputs", {}).get("quality_report"):
        summary["single_frame_quality"] = load_json(Path(reconstruction["outputs"]["quality_report"]))
    processing_path = getattr(args, 'local_processing_report', None)
    if processing_path:
        from local_processing import read_json, write_json, write_html
        processing = read_json(processing_path)
        if Path(processing['actual_dataset']).resolve() != posed_dataset.resolve():
            raise ValueError('Processing report does not match reconstruction input')
        processing['state'] = 'complete'
        processing['reconstruction'] = dict(status='complete', error=None,
            depth_quality=reconstruction.get('depth_quality'),metrics=reconstruction.get('metrics'))
        processing['result'] = str(result_root)
        write_json(result_root / 'processing_report.json', processing)
        write_html(result_root / 'processing_report.html', processing)
        summary['local_processing_report'] = str(result_root / 'processing_report.json')
    (result_root / "demo_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    from scene_quality_report import write_report
    write_report(result_root)
    _write_html_report(result_root, summary)
    args.outputs_root.resolve().mkdir(parents=True, exist_ok=True)
    if getattr(args, 'publish_latest', True):
        (args.outputs_root.resolve() / "latest_complete_demo.txt").write_text(
            str(reconstruction_dir), encoding="utf-8"
        )
    return result_root, summary


def main() -> int:
    args = parse_args()
    try:
        result_root, summary = run_demo(args)
    except (OSError, ValueError, KeyError, json.JSONDecodeError, RuntimeError) as error:
        print(json.dumps({"status": "blocked", "error": str(error)}, ensure_ascii=False, indent=2))
        return 2
    print(
        json.dumps(
            {
                "status": summary["status"],
                "result": str(result_root),
                "report": str(result_root / "demo_report.html"),
                "metrics": summary["reconstruction_metrics"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    if not args.no_viewer:
        from scene_demo_viewer import run_viewer

        run_viewer(Path(summary["reconstruction"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
