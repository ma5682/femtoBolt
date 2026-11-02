"""Isolated reconstruction replay and source-bound gap evidence."""
from __future__ import annotations

import numpy as np
from femto.local.reconstruction_gap_audit import GapProbe, STAGES


def audit(model, mask, output, *, check=lambda: None, progress=lambda text: None):
    """Rebuild into a fresh derived directory, never publish or replace a model."""
    import hashlib
    from pathlib import Path
    from femto.io_utils import load_json, load_jsonl, write_json_strict
    from femto.pipeline.scene_reconstruction import reconstruct_scene
    from femto.planning.planning_revision import array_record
    check()
    output = Path(output).resolve()
    source = Path(model.reconstruction).resolve()
    recorded = load_json(source/'reconstruction_report.json')
    dataset = Path(recorded['source_dataset']).resolve()
    if output.exists():raise FileExistsError('Choose a new audit output directory')
    if output == source or source in output.parents or output == dataset or dataset in output.parents:
        raise ValueError('Audit output must be separate from source evidence')
    mask = np.asarray(mask)
    if mask.dtype != np.bool_ or mask.shape != model.projection.depth.shape:
        raise ValueError('Audit mask must match the bound reference image')
    raw = model.projection.depth / 1000.
    measured = (raw >= model.projection.minimum) & (raw <= model.projection.maximum)
    yy, xx = np.nonzero(mask & measured & (model.projection.reasons == 1))
    if not len(xx):raise ValueError('当前定位范围没有可按阶段排查的“有深度但无网格”像素。')
    probe = GapProbe(np.c_[xx, yy], model.projection.k, model.projection.depth, check=check)
    frames = load_jsonl(dataset/'frames.jsonl')
    if not 1 <= len(frames) <= 128:raise ValueError('Audit frame budget exceeded')
    paths = {dataset/'manifest.json', dataset/'frames.jsonl', source/'scene_tsdf_mesh_m.ply',
             source/'reconstruction_report.json'}
    for frame in frames:
        paths.update((dataset/frame[key]).resolve() for key in ('rgb', 'depth', 'pose'))
    paths.update((dataset/'calibration').glob('*.json'))
    def hashes():
        result = {}
        for path in sorted(paths):
            value = hashlib.sha256()
            with path.open('rb') as stream:
                while True:
                    check(); block = stream.read(1024*1024)
                    if not block:break
                    value.update(block)
            result[str(path)] = value.hexdigest()
        return result
    before = hashes()
    output.mkdir(parents=True, exist_ok=False)
    write_json_strict(output/'source_start.json', before)
    progress('以原参数重建，并分别检查提取、实测支持和连通分量过滤')
    parameters = recorded['parameters']
    reconstruct_scene(dataset, output/'reconstruction', geometry_observer=probe,
        **{key: parameters[key] for key in ('image_scale', 'voxel_m', 'sdf_trunc_m',
            'depth_min_m', 'depth_max_m', 'point_stride', 'single_frame_quality')})
    check()
    report, cause = probe.report()
    old_depth = model.projection.z[yy, xx];new_depth = probe.depths[STAGES[-1]]
    equal = np.array_equal(np.isfinite(old_depth), np.isfinite(new_depth)) and np.allclose(
        old_depth[np.isfinite(old_depth)], new_depth[np.isfinite(new_depth)], rtol=0, atol=1e-6)
    report.update(source=str(source), source_hashes=before, source_unchanged=before == hashes(),
        view_reference=model.view_reference, requested_mask=array_record(mask),
        geometry_sha256=model.store.digest, final_probe_matches_original=bool(equal))
    np.savez_compressed(output/'gap_pixels.npz', pixels_xy=probe.pixels, cause=cause,
        reference_depth_m=probe.depth, **probe.depths)
    write_json_strict(output/'report.json', report)
    if not report['source_unchanged'] or not equal:raise ValueError('重建复现与当前来源不同，诊断结果不可直接归因。')
    return report
