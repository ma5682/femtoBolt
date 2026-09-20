"""Fixed-W0 comparison projections and separate internal geometry/depth metrics.

This is an offline diagnostic renderer, not another measurement or annotation
surface. Existing W0 vertices are read as stored and are never transformed again.
"""
from __future__ import annotations

import html
import json
import math
from pathlib import Path


def _summary(values):
    import numpy as np
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return dict(sample_count=len(values), median_m=float(np.median(values)) if len(values) else None,
                p95_m=float(np.quantile(values, .95)) if len(values) else None,
                rmse_m=float(np.sqrt(np.mean(values**2))) if len(values) else None,
                max_m=float(values.max()) if len(values) else None)


def _sample_indices(valid, budget=4096):
    import numpy as np
    ids = np.flatnonzero(valid)
    if len(ids) > budget:
        ids = ids[np.linspace(0, len(ids)-1, budget, dtype=np.int64)]
    return ids


def _mesh_scene(mesh):
    import open3d as o3d
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene


def sampled_depth_metrics(scene, depth_mm, k, pose, minimum=.25, maximum=2.21, budget=4096):
    """Sample valid raw pixel ordinals deterministically; absent hits stay in the denominator.

    A ray direction has camera Z=1, so its hit parameter is camera depth in metres.
    The pose converts source rays to W0. It is never applied to model vertices.
    """
    import numpy as np
    import open3d as o3d
    depth = np.asarray(depth_mm)
    valid = (depth >= minimum*1000) & (depth <= maximum*1000)
    ids = _sample_indices(valid, budget)
    v, u = np.unravel_index(ids, depth.shape)
    observed = depth[v, u].astype(float)/1000
    camera = np.column_stack(((u-k[0, 2])/k[0, 0], (v-k[1, 2])/k[1, 1], np.ones(len(ids))))
    direction = camera @ pose[:3, :3].T
    origins = np.broadcast_to(pose[:3, 3], direction.shape)
    if not len(ids):
        return dict(raw_valid_pixels=0, sampled_pixels=0, hit_pixels=0, missing_hit_pixels=0,
                    consistent_within_10mm_ratio=None, error_on_hits=_summary([]),
                    error_missing_as_zero=_summary([]), point_to_mesh=_summary([]))
    result = scene.cast_rays(o3d.core.Tensor(np.c_[origins, direction].astype(np.float32)))
    prediction = result['t_hit'].numpy().astype(float)
    hit = np.isfinite(prediction) & (prediction > 0)
    signed = np.where(hit, prediction, 0)-observed
    points = camera*observed[:, None] @ pose[:3, :3].T + pose[:3, 3]
    nearest = scene.compute_distance(o3d.core.Tensor(points.astype(np.float32))).numpy()
    return dict(raw_valid_pixels=int(valid.sum()), sampled_pixels=len(ids), hit_pixels=int(hit.sum()),
        missing_hit_pixels=int((~hit).sum()), consistent_within_10mm_ratio=float(np.mean(hit & (np.abs(signed)<=.01))),
        error_on_hits=_summary(np.abs(signed[hit])), error_missing_as_zero=_summary(np.abs(signed)),
        point_to_mesh=_summary(nearest))


def render_fixed_surface(mesh, k, reference_depth_mm, width=640, image_valid=None):
    """First-hit W0 projection with the same initial raw-depth occlusion gate as fixed RGB."""
    import numpy as np
    import cv2
    from femto.annotation.fixed_rgb_geometry import FixedProjection
    projection = FixedProjection(mesh, k, reference_depth_mm)
    normals = np.asarray(mesh.triangle_normals)
    if len(normals) != len(mesh.triangles):
        mesh.compute_triangle_normals()
        normals = np.asarray(mesh.triangle_normals)
    valid = projection.valid
    if image_valid is not None:
        valid = valid & np.asarray(image_valid, dtype=bool)
    image = np.full((*valid.shape, 3), [24, 29, 36], np.uint8)
    if valid.any():
        n = normals[projection.ids[valid]]
        light = np.array([.25, -.45, -.8573214])
        shade = .35+.65*np.abs(n@light)
        color = np.array([69, 186, 214.])
        image[valid] = np.rint(shade[:, None]*color).astype(np.uint8)
    scale = width / image.shape[1]
    resized = cv2.resize(image, (width, max(1, round(image.shape[0]*scale))), interpolation=cv2.INTER_AREA)
    return resized, dict(width=reference_depth_mm.shape[1], height=reference_depth_mm.shape[0],
        visible_supported_pixels=int(valid.sum()), total_reference_pixels=int(valid.size),
        unsupported_pixel_ratio=float(1-valid.mean()),
        definition='Fixed initial RGB camera; nearest supported mesh hit with original 10 mm reference-depth consistency gate. Not ROI hole-filling coverage.')


def render_raw_cloud(cloud, k, height, width, display_width=640):
    """Display stored W0 points from the original reference only; no transform or point repair."""
    import numpy as np
    import cv2
    points = np.asarray(cloud.points)
    colors = np.asarray(cloud.colors)
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 0)
    indices = np.flatnonzero(valid)
    xyz = points[indices]
    u = np.rint(xyz[:, 0]*k[0, 0]/xyz[:, 2]+k[0, 2]).astype(int)
    v = np.rint(xyz[:, 1]*k[1, 1]/xyz[:, 2]+k[1, 2]).astype(int)
    good = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v, xyz, indices = u[good], v[good], xyz[good], indices[good]
    pixels = v*width+u
    order = np.lexsort((xyz[:, 2], pixels))
    ordered = pixels[order]
    choose = order[np.r_[True, ordered[1:] != ordered[:-1]]] if len(order) else order
    image = np.full((height, width, 3), [24, 29, 36], np.uint8)
    image[v[choose], u[choose]] = (np.rint(np.clip(colors[indices[choose]], 0, 1)*255).astype(np.uint8)
        if len(colors) == len(points) else [69, 186, 214])
    return cv2.resize(image, (display_width, max(1, round(height*display_width/width))), interpolation=cv2.INTER_AREA)


def evaluate_and_render(baseline, dataset, reconstruction, *, width=640):
    import numpy as np
    import cv2
    import open3d as o3d
    from femto.geometry.fusion import read_geometry_safe
    from femto.io_utils import imread_unicode, imwrite_unicode, load_json_object, write_json_strict
    from femto.geometry.rgbd_frames import distortion_maps, prepare_rgbd
    from femto.local.local_temporal_support import filtered_depth
    from femto.pipeline.local_comparison import read_rows, sha256

    dataset, reconstruction = Path(dataset), Path(reconstruction)
    frames = read_rows(dataset / 'frames.jsonl')
    mesh_path = reconstruction / 'scene_tsdf_mesh_m.ply'
    mesh = read_geometry_safe(mesh_path, mesh=True)
    vertices = np.asarray(mesh.vertices)
    if not len(mesh.triangles) or not np.isfinite(vertices).all():
        raise ValueError('Empty or nonfinite comparison mesh')
    scene = _mesh_scene(mesh)
    mx, my, k, original_width, original_height = distortion_maps(Path(baseline['dataset']))
    rows = []
    for frame in frames:
        pose = np.asarray(load_json_object(dataset/frame['pose'])['T_WC'], dtype=float)
        rgb = imread_unicode(dataset/frame['rgb'], cv2.IMREAD_COLOR)
        depth = imread_unicode(dataset/frame['depth'], cv2.IMREAD_UNCHANGED)
        actual = filtered_depth(dataset, frame, depth)
        _, _, rectified_rgb, rectified_depth = prepare_rgbd(rgb, depth, mx, my, k, 1., .25, 2.21)
        metrics = sampled_depth_metrics(scene, rectified_depth, k, pose)
        original_valid = (depth>=250) & (depth<=2210)
        retained_valid = (actual>=250) & (actual<=2210)
        rows.append(dict(frame=frame['frame'], depth_consistency=metrics,
            raw_valid_pixel_ratio=float(original_valid.mean()), actual_valid_pixel_ratio=float(retained_valid.mean()),
            raw_valid_pixels=int(original_valid.sum()), actual_valid_pixels=int(retained_valid.sum()),
            masked_pixels=int((original_valid & ~retained_valid).sum())))
        if frame['frame'] == baseline['reference_frame_id']:
            initial_rgb, initial_depth = rectified_rgb[:, :, ::-1], rectified_depth
            initial_rgb_path = dataset / frame['rgb']
            initial_depth_path = dataset / frame['depth']
    image_valid = (mx>=0) & (mx<=original_width-1) & (my>=0) & (my<=original_height-1)
    image, viewport = render_fixed_surface(mesh, k, initial_depth, width, image_valid=image_valid)
    imwrite_unicode(reconstruction / 'comparison_surface.png', image[:, :, ::-1])
    resized_rgb = cv2.resize(initial_rgb, (width, image.shape[0]), interpolation=cv2.INTER_AREA)
    imwrite_unicode(reconstruction / 'comparison_reference_rgb.png', resized_rgb[:, :, ::-1])
    cloud = read_geometry_safe(reconstruction / 'initial_observed_points_m.ply', mesh=False)
    raw_image = render_raw_cloud(cloud, k, original_height, original_width, width)
    imwrite_unicode(reconstruction / 'comparison_initial_raw.png', raw_image[:, :, ::-1])
    reference = dict(reference_frame_id=baseline['reference_frame_id'], session_id=baseline['session_id'],
        coordinate_frame='world_rdf', unit='metre', view='initial_color_camera_locked',
        camera_to_world=np.eye(4).tolist(), undistorted_intrinsic=k.tolist(),
        source_image_size=[original_width, original_height], display_image_size=[width, image.shape[0]],
        rgb_sha256=sha256(initial_rgb_path), depth_sha256=sha256(initial_depth_path),
        near_m=.25, far_m=2.21, same_lighting=True)
    write_json_strict(reconstruction / 'comparison_view.json', reference)
    triangles = np.asarray(mesh.triangles)
    edges = np.sort(np.concatenate((triangles[:, :2], triangles[:, 1:], triangles[:, [2, 0]])), axis=1)
    _, multiplicities = np.unique(edges, axis=0, return_counts=True)
    bbox = vertices.max(axis=0)-vertices.min(axis=0)
    result = dict(schema='femto-local-comparison-metrics-v1', coordinate_frame='world_rdf', unit='metre',
        fixed_view=reference, fixed_projection=viewport,
        internal_geometric_residual=dict(definition='Distance of <=4096 deterministic valid raw-depth samples per source frame to the reconstructed mesh. Actual version poses place the samples in W0. Not physical accuracy.',
            frames=[dict(frame=r['frame'], **r['depth_consistency']['point_to_mesh']) for r in rows]),
        depth_consistency=dict(definition='Same raw-pixel ordinal samples in each A/B/C version. Nearest mesh ray hit vs source depth; missing hits are reported and also included as zero depth in a separate residual.',
            frames=[dict(frame=r['frame'], **{k:v for k,v in r['depth_consistency'].items() if k!='point_to_mesh'}) for r in rows]),
        valid_pixels=dict(definition='Full aligned raw image pixels before and after actual binary mask; no interpolated depth.',
            frames=[{k:v for k,v in r.items() if k!='depth_consistency'} for r in rows]),
        topology=dict(boundary_edges=int(np.count_nonzero(multiplicities==1)),
                      nonmanifold_edges=int(np.count_nonzero(multiplicities>2)),
                      note='Topological boundary edges are not a count of physically missing holes'),
        axis_aligned_extent_m=bbox.tolist(), mesh_surface_area_m2=float(mesh.get_surface_area()),
        size_definition='W0 axis-aligned geometry extent and mesh triangle area; not an independently measured object dimension',
        noise=dict(status='not_evaluated', reason='No predefined stationary planar ROI or independent surface reference'),
        ghosting=dict(status='not_evaluated', reason='Depth inconsistency is reported separately; no labelled double-surface reference'),
        roi_hole_filling=dict(status='not_evaluated', reason='No predetermined ROI with new-view depth-support annotation'),
        physical_accuracy=dict(status='not_evaluated', accepted=None, reason='No independent physical reference measurement'))
    truth = baseline.get('synthetic_ground_truth')
    if truth:
        if baseline['data_kind'] != 'synthetic' or sha256(truth['mesh']) != truth['sha256']:
            raise ValueError('Ground truth cannot be applied to real or changed input')
        target = _mesh_scene(read_geometry_safe(Path(truth['mesh']), mesh=True))
        sample = vertices[_sample_indices(np.ones(len(vertices), bool), 50000)]
        distances = target.compute_distance(o3d.core.Tensor(sample.astype(np.float32))).numpy()
        result['synthetic_ground_truth'] = dict(**_summary(distances), source=truth,
            definition='Sampled reconstructed mesh vertices to known synthetic W0 surface; software fixture only',
            physical_accuracy_verified=False)
    write_json_strict(reconstruction / 'comparison_metrics.json', result)
    return result


def _value(value, scale=1, suffix=''):
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return '<span data-zh="未记录" data-en="Not recorded">未记录</span>'
    return f'{value*scale:.3f}{suffix}'


def _dual(chinese, english):
    return f'<span data-zh="{html.escape(chinese, quote=True)}" data-en="{html.escape(english, quote=True)}">{html.escape(chinese)}</span>'


def _recorded(value):
    if value is None:
        return _dual('未记录', 'Not recorded')
    return html.escape(str(value))


def _flag(value):
    if value is True:
        return _dual('是', 'Yes')
    if value is False:
        return _dual('否', 'No')
    return _recorded(None)


def _label(value):
    labels = {
        'baseline': ('原始基线', 'Original baseline'),
        'adopted': ('已采用', 'Adopted'),
        'not_adopted': ('未采用', 'Not adopted'),
        'not_run': ('未执行', 'Not run'),
        'not_recorded': ('未记录', 'Not recorded'),
        'running': ('执行中', 'Running'),
        'complete': ('已完成', 'Complete'),
        'failed': ('失败', 'Failed'),
        'cancelled': ('已取消', 'Cancelled'),
        'real_femto': ('Femto 实际采集数据回放', 'Recorded Femto data replay'),
        'synthetic': ('合成数据验证', 'Synthetic data validation'),
        'synthetic_ground_truth': ('合成真值位姿', 'Synthetic ground-truth poses'),
        'online_rgbd_research_grade': ('在线 RGB-D 研发位姿', 'Online RGB-D research poses'),
        'offline_robust_icp_research_grade': ('离线鲁棒 ICP 研发位姿', 'Offline robust ICP research poses'),
    }
    return _dual(*labels[value]) if value in labels else _recorded(value)


def write_comparison_html(output, report):
    """Self-contained offline controls; matching projections retain equal scales in both layouts."""
    output = Path(output)
    cards, options = [], []
    for variant in report.get('variants', []):
        name = variant['name']
        safe_name = html.escape(name)
        metric = variant.get('metrics', {})
        fixed = metric.get('fixed_view', {})
        display_width, display_height = fixed.get('display_image_size', [640, 360])
        options.append(f'<option value="{safe_name}">{safe_name}</option>')
        count = variant.get('reconstruction_metrics', {})
        same = variant.get('same_as')
        decision = _label(variant.get('optimization', {}).get('status'))
        reused = f'<p class="notice">{_dual("与 " + same + " 相同，几何逐字节复用", "Same as " + same + "; geometry reused byte-for-byte")}</p>' if same else ''
        rows = []
        for kind, key, zh, en in [('internal_geometric_residual', 'median_m', '几何距离中位数', 'Median geometric distance')]:
            values = metric.get(kind, {}).get('frames', [])
            rows.extend(f'<tr><td>{_dual(zh, en)} · #{r["frame"]}</td><td>{_value(r.get(key),1000," mm")}</td></tr>' for r in values)
        for r in metric.get('depth_consistency', {}).get('frames', []):
            rows.append(f'<tr><td>{_dual("深度RMSE（缺失按0）", "Depth RMSE (missing=0)")} · #{r["frame"]}</td><td>{_value(r.get("error_missing_as_zero",{}).get("rmse_m"),1000," mm")}</td></tr>')
            rows.append(f'<tr><td>{_dual("无表面命中 / 采样数", "Missing hits / samples")} · #{r["frame"]}</td><td>{_recorded(r.get("missing_hit_pixels"))} / {_recorded(r.get("sampled_pixels"))}</td></tr>')
        for r in metric.get('valid_pixels', {}).get('frames', []):
            rows.append(f'<tr><td>{_dual("有效像素 原始 → 实用", "Valid pixels raw → used")} · #{r["frame"]}</td><td>{_value(r.get("raw_valid_pixel_ratio"),100,"%")} → {_value(r.get("actual_valid_pixel_ratio"),100,"%")}</td></tr>')
        geo = json.dumps(variant['optimization'].get('details', {}), ensure_ascii=False, indent=2)
        from femto.local.local_processing import REASONS
        detail = variant['optimization'].get('details', {})
        reasons = set([detail['reason']]) if detail.get('reason') else set()
        for row in detail.get('frames', []):
            reasons.update(row.get('reasons', []))
            if row.get('reason'):
                reasons.add(row['reason'])
        reason_text = '；'.join(_dual(*REASONS.get(str(reason), (str(reason), str(reason)))) for reason in sorted(reasons))
        resource = variant.get('resource_usage', {})
        memory = resource.get('peak_job_committed_mb')
        memory_title = _dual('进程组峰值提交内存', 'Peak job committed memory')
        if memory is None:
            memory = resource.get('peak_rss_sampled_mb')
            memory_title = _dual('采样峰值内存', 'Sampled peak process RSS')
        cards.append(f'''<article data-version="{safe_name}"><h2>{safe_name}</h2><p>{decision}</p>{reused}
<div class="viewport" style="aspect-ratio:{display_width}/{display_height}" tabindex="0"><img class="projection" src="{safe_name}/reconstruction/comparison_surface.png" width="{display_width}" height="{display_height}" alt="Fixed W0 geometric projection"></div>
<p>{_dual('所有图像使用相同初始相机视角、尺度、投影与照明。', 'All images use the same initial camera, scale, projection and lighting.')}</p>
<table><tr><td>{_dual('源帧 / 采用视角', 'Source frames / used views')}</td><td>{_recorded(variant.get('source_frame_count'))} / {_recorded(variant.get('accepted_frame_count'))}</td></tr>
<tr><td>{_dual('点 / 顶点 / 三角面', 'Points / vertices / triangles')}</td><td>{_recorded(count.get('fused_point_count'))} / {_recorded(count.get('mesh_vertices'))} / {_recorded(count.get('mesh_triangles'))}</td></tr>
<tr><td>{_dual('重建和评估耗时', 'Reconstruction and evaluation time')}</td><td>{_value(variant.get('resource_usage',{}).get('elapsed_s'),1,' s')}</td></tr>
<tr><td>{memory_title}</td><td>{_value(memory,1,' MiB')}</td></tr>{''.join(rows)}</table>
<p>{_dual('实际位姿来源', 'Actual pose source')}: {_label(variant.get('actual_pose_source'))}</p><p>{reason_text}</p>
<p><a href="{safe_name}/variant.json">{_dual('版本来源与全部指标', 'Version lineage and full metrics')}</a> · <a href="{safe_name}/reconstruction/reconstruction_report.json">{_dual('重建报告', 'Reconstruction report')}</a></p>
<details><summary>{_dual('优化决定、未采用原因与候选内部指标', 'Optimization decision, rejection reasons and candidate internal metrics')}</summary><pre>{html.escape(geo)}</pre></details></article>''')
    parameters = report.get('parameters', {})
    first = report.get('variants', [{}])[0].get('name') if report.get('variants') else None
    reference = ''
    if first:
        reference = f'''<section class="references"><div><h2>{_dual('初始 RGB 照片', 'Initial RGB photograph')}</h2><img src="{first}/reconstruction/comparison_reference_rgb.png" alt="Original reference RGB"></div>
<div><h2>{_dual('初始原始观测点云', 'Initial original observed cloud')}</h2><img src="{first}/reconstruction/comparison_initial_raw.png" alt="Original W0 point cloud"></div></section>'''
    content = f'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Femto A/B/C comparison</title>
<style>body{{font:15px/1.5 "Microsoft YaHei",Arial,sans-serif;background:#10151d;color:#e3eaf4;margin:0;padding:20px}}a{{color:#85caff}}header,.controls,.warning{{max-width:1600px;margin:0 auto 16px}}h1{{font-size:24px}}h2{{font-size:19px}}button,select{{font:inherit;padding:8px 12px;background:#263c52;color:white;border:1px solid #66809b;border-radius:5px}}.controls{{display:flex;gap:12px;align-items:center;flex-wrap:wrap}}.warning,.notice{{background:#473b1b;padding:12px;border-left:4px solid #ebc368}}.grid{{display:grid;grid-template-columns:repeat(3,minmax(260px,1fr));gap:16px}}article{{min-width:0;background:#1a2431;border:1px solid #3a4b61;border-radius:8px;padding:16px}}article[hidden]{{display:none}}.grid.single{{grid-template-columns:minmax(280px,1000px);justify-content:center}}.viewport{{overflow:auto;background:#181d24;aspect-ratio:16/9}}.projection{{display:block;width:100%;height:auto;max-width:none}}table{{border-collapse:collapse;width:100%;font-size:13px}}td{{padding:5px;border-bottom:1px solid #3a4b61}}td:last-child{{text-align:right}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;max-height:400px;overflow:auto}}.references{{display:grid;grid-template-columns:1fr 1fr;gap:20px;max-width:1200px;margin:28px auto}}.references img{{width:100%;height:auto}}code{{overflow-wrap:anywhere}}@media(max-width:900px){{.grid{{grid-template-columns:1fr}}.references{{grid-template-columns:1fr}}}}</style>
<header><h1>{_dual('同源局部几何 A / B / C 对照', 'Same-source local geometry A / B / C')}</h1><p>{_dual('任务状态', 'Task state')}: {_label(report.get('state'))} · {_label(report.get('data_kind'))} · W0 / RDF / m</p>
<p>{_dual('体素 / 截断距离', 'Voxel / truncation')}: {_value(parameters.get('voxel_m'),1000,' mm')} / {_value(parameters.get('sdf_trunc_m'),1000,' mm')} · {_dual('默认参数未被本工具修改', 'This tool does not change product defaults')}</p>
<p>{_dual('会话', 'Session')}: <code>{_recorded(report.get('session_id'))}</code></p></header>
<div class="warning">{_dual('内部一致性不是物理误差。生成模型不等于精度达标，RGB完整不等于几何完整。预览点数不代表补洞或覆盖率。未指定ROI、实物尺寸及独立参考时，补洞、噪声、重影和物理验收均未评估。', 'Internal consistency is not physical error. Model generation does not establish accuracy, and complete RGB does not imply complete geometry. Point count is not hole filling or coverage. Without a predefined ROI and independent reference, hole filling, noise, ghosting and physical acceptance are not evaluated.')}</div>
<div class="controls"><button id="language">English</button><label>{_dual('布局', 'Layout')} <select id="layout"><option value="side" data-zh="并排" data-en="Side by side">并排</option><option value="single" data-zh="切换" data-en="Switch">切换</option></select></label>
<label>{_dual('版本', 'Version')} <select id="version">{''.join(options)}</select></label><label>{_dual('共同缩放', 'Shared zoom')} <select id="zoom"><option value="100">100%</option><option value="150">150%</option><option value="200">200%</option></select></label>
<a href="comparison_report.json">{_dual('详细任务报告', 'Full task report')}</a> <a href="source_evidence.json">{_dual('来源与原始哈希', 'Source and original hashes')}</a></div>
<p>{_dual('共同缩放后可滚动图像平移；并排、切换与窗口调整保持相同的图像位置，不改变 W0 几何。', 'After shared zoom, scroll an image to pan; layout switches and resizing preserve the shared image position and do not change W0 geometry.')}</p>
<div class="grid" id="grid">{''.join(cards)}</div>{reference}
<p>{html.escape(str(report.get('error','')))}</p><p>{_dual('原始文件哈希保持', 'Original file hashes unchanged')}: {_flag(report.get('source_unchanged'))} · {_dual('主业务最新索引保持', 'Business latest indices unchanged')}: {_flag(report.get('latest_unchanged'))}</p>
<script>
let language='zh',syncing=false;
const pan={{x:0,y:0}},layout=document.getElementById('layout'),version=document.getElementById('version'),zoom=document.getElementById('zoom');
function applyPan(){{
  syncing=true;
  document.querySelectorAll('.viewport').forEach(v=>{{
    const width=v.querySelector('.projection').getBoundingClientRect().width;
    if(v.clientWidth>0&&width>0){{v.scrollLeft=pan.x*width;v.scrollTop=pan.y*width;}}
  }});
  requestAnimationFrame(()=>syncing=false);
}}
function update(){{
  const single=layout.value==='single';
  document.getElementById('grid').classList.toggle('single',single);
  document.querySelectorAll('article').forEach(x=>x.hidden=single&&x.dataset.version!==version.value);
  document.querySelectorAll('.projection').forEach(x=>x.style.width=zoom.value+'%');
  applyPan();
}}
layout.onchange=version.onchange=zoom.onchange=update;
document.getElementById('language').onclick=()=>{{language=language==='zh'?'en':'zh';document.documentElement.lang=language;document.querySelectorAll('[data-zh]').forEach(x=>x.textContent=x.dataset[language]);document.getElementById('language').textContent=language==='zh'?'English':'中文';}};
document.querySelectorAll('.viewport').forEach(v=>v.onscroll=()=>{{
  if(syncing||v.clientWidth===0)return;
  const width=v.querySelector('.projection').getBoundingClientRect().width;
  if(width<=0)return;
  pan.x=v.scrollLeft/width;pan.y=v.scrollTop/width;applyPan();
}});
window.addEventListener('resize',applyPan);
document.querySelectorAll('.projection').forEach(x=>x.addEventListener('load',applyPan));
update();
</script></html>'''
    (output / 'comparison_report.html').write_text(content, encoding='utf-8')
    return output / 'comparison_report.html'


if __name__ == '__main__':
    import argparse
    import webbrowser
    parser = argparse.ArgumentParser(description='Open an existing independent A/B/C report without recomputing or changing it')
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    page = args.output.resolve() / 'comparison_report.html'
    if not page.is_file():
        parser.error('comparison_report.html is missing')
    webbrowser.open(page.as_uri())
