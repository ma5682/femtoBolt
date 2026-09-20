"""Bounded TSDF parameter studies using the existing isolated A/B/C replay.

No SDK, geometry mutation, default-parameter change or latest publication occurs
here. The synchronous APIs belong on a worker thread or in the CLI, never on Tk.
Existing reports are reusable only while their source and implementation hashes
still agree. Numerical consistency is never an independent accuracy certificate.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import struct
import time
import uuid


ROOT = Path(__file__).resolve().parent
SCHEMA = 'femto-local-parameter-study-v1'
VARIANTS = ('A_online_raw', 'B_refined_raw', 'C_refined_filtered')
BASELINE = (.005, .02)
DEFAULT_CANDIDATES = (BASELINE, (.003, .012), (.003, .02), (.002, .008), (.002, .02))
# Must match the comparison producer's implementation snapshot. Missing files
# cannot silently weaken reuse when an old report predates a required dependency.
IMPLEMENTATION_FILES = (
    'local_comparison.py', 'local_comparison_viewer.py', 'scene_reconstruction.py',
    'local_pose_refinement.py', 'local_temporal_support.py', 'local_geometry.py',
    'local_reference.py', 'scene_dataset_validator.py', 'fixed_rgb_geometry.py', 'local_stability.py',
)
IDENTITY_KEYS = ('dataset', 'raw_source', 'data_kind', 'session_id', 'reference_frame_id',
                 'reference_provenance', 'accepted_frame_ids', 'source_frame_count',
                 'accepted_frame_count', 'pose_source', 'workspace_radius_m', 'workspace_policy',
                 'synthetic_ground_truth', 'source_sha256', 'implementation_sha256')


class StudyStopped(RuntimeError):
    def __init__(self, state, reason):
        super().__init__(reason)
        self.state = state


class AuditBudget:
    """Cooperative limits for JSON, file hashing and copied report images."""
    def __init__(self, timeout_s=300., *, cancelled=None, cancel_file=None,
                 json_limit=32 * 1024**2, hash_limit=16 * 1024**3, json_total_limit=64 * 1024**2):
        if not _finite(timeout_s) or not 1 <= timeout_s <= 21600:
            raise ValueError('Invalid finite total time budget')
        self.started = time.monotonic()
        self.timeout_s, self.cancelled = timeout_s, cancelled
        self.cancel_file = Path(cancel_file) if cancel_file else None
        self.json_limit, self.hash_limit, self.hashed_bytes = json_limit, hash_limit, 0
        self.json_total_limit, self.json_read_bytes = json_total_limit, 0

    def check(self):
        if ((self.cancel_file and self.cancel_file.exists()) or
                (self.cancelled and self.cancelled())):
            raise StudyStopped('cancelled', 'Parameter study cancelled')
        if time.monotonic() - self.started > self.timeout_s:
            raise StudyStopped('failed', 'Parameter study total time budget exceeded')

    def digest(self, path):
        self.check()
        path = Path(path)
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            while block := stream.read(1024 * 1024):
                self.check()
                self.hashed_bytes += len(block)
                if self.hashed_bytes > self.hash_limit:
                    raise StudyStopped('failed', 'Parameter study hash-read budget exceeded')
                digest.update(block)
        return digest.hexdigest()

    def read(self, path):
        self.check()
        path = Path(path)
        if path.stat().st_size > self.json_limit:
            raise ValueError('JSON file exceeds the report memory budget')
        with path.open('rb') as stream:
            data = stream.read(self.json_limit + 1)
        if len(data) > self.json_limit:
            raise ValueError('JSON file exceeds the report memory budget')
        self.json_read_bytes += len(data)
        if self.json_read_bytes > self.json_total_limit:
            raise ValueError('Cumulative JSON-read budget exceeded')
        value = json.loads(data.decode('utf-8-sig'))
        if not isinstance(value, dict):
            raise ValueError('Expected a JSON object')
        # Reject both literal NaN/Infinity and overflowing exponents such as 1e999.
        json.dumps(value, allow_nan=False)
        return value

    def read_frames(self, path):
        self.check()
        path = Path(path)
        if path.stat().st_size > self.json_limit:
            raise ValueError('Frame index exceeds the report memory budget')
        with path.open('rb') as stream:
            data = stream.read(self.json_limit + 1)
        if len(data) > self.json_limit:
            raise ValueError('Frame index exceeds the report memory budget')
        self.json_read_bytes += len(data)
        if self.json_read_bytes > self.json_total_limit:
            raise ValueError('Cumulative JSON-read budget exceeded')
        rows = [json.loads(line) for line in data.decode('utf-8-sig').splitlines() if line.strip()]
        if len(rows) > 20 or any(not isinstance(row, dict) for row in rows):
            raise ValueError('Accepted-frame index exceeds the view budget')
        json.dumps(rows, allow_nan=False)
        return rows


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _count(value):
    return type(value) is int and value >= 0


def parameter_key(voxel_m, sdf_trunc_m):
    if (not _finite(voxel_m) or not _finite(sdf_trunc_m) or
            not .001 <= voxel_m <= .02 or not voxel_m < sdf_trunc_m <= .1):
        raise ValueError('Invalid finite voxel / truncation parameters')
    return f'voxel_{voxel_m * 1000:.6g}mm_trunc_{sdf_trunc_m * 1000:.6g}mm'


def _write(path, value):
    """Only called inside a newly reserved task directory."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def _hash_map(value, *, missing_allowed=False):
    if not isinstance(value, dict) or len(value) > 50000:
        raise ValueError('Invalid or over-budget fingerprint map')
    result = {}
    for name, digest in value.items():
        path = Path(name)
        if not path.is_absolute() or (digest is None and not missing_allowed):
            raise ValueError('Missing absolute source or implementation fingerprint')
        key = str(path.resolve())
        if key in result or not (digest is None or re.fullmatch('[a-f0-9]{64}', str(digest))):
            raise ValueError('Duplicate path or invalid SHA-256')
        result[key] = digest
    return result


def _merge_hashes(target, incoming):
    for path, digest in incoming.items():
        if path in target and target[path] != digest:
            raise ValueError('Conflicting fingerprints for the same file: ' + path)
        target[path] = digest


def _verify_hashes(snapshot, budget):
    for name, expected in snapshot.items():
        budget.check()
        path = Path(name)
        actual = budget.digest(path) if path.is_file() else None
        if actual != expected:
            raise ValueError('File SHA-256 changed or evidence is missing: ' + name)


def _report_path(path):
    path = Path(path).resolve()
    return path / 'comparison_report.json' if path.is_dir() else path


def _source_contract(evidence, budget):
    """Check the recorded identity against its hashed manifests, without loading RGB-D."""
    from scene_data_kind import classify_manifest_data_kind
    dataset, raw = Path(evidence['dataset']).resolve(), Path(evidence['raw_source']).resolve()
    manifest = budget.read(dataset / 'manifest.json')
    raw_manifest = budget.read(raw / 'manifest.json')
    raw_kind = classify_manifest_data_kind(raw_manifest, dataset=raw)
    kind = 'synthetic' if classify_manifest_data_kind(manifest, dataset=dataset) == 'synthetic' else raw_kind
    if kind != evidence.get('data_kind') or kind == 'unknown':
        raise ValueError('Comparison data kind differs from source camera metadata')
    local = manifest.get('local_capture', {})
    proof = evidence.get('reference_provenance', {})
    if (manifest.get('schema') != 'femto-full-scene-rgbd-v1' or not local.get('measured_only') or
            manifest.get('pose', {}).get('source') != evidence.get('pose_source') or
            evidence.get('pose_source') != ('synthetic_ground_truth' if kind == 'synthetic' else 'online_rgbd_research_grade') or
            manifest.get('refinement_source_dataset') or manifest.get('temporal_fusion') or
            local.get('refinement_report') or local.get('temporal_support_report') or
            Path(manifest.get('source_dataset', '')).resolve() != raw or
            local.get('session_id') != evidence.get('session_id') or
            local.get('reference_frame_id') != evidence.get('reference_frame_id') or
            local.get('reference_provenance') != proof or
            proof.get('schema') != 'femto-local-reference-v1' or
            proof.get('origin') != 'first_valid_color_camera' or
            proof.get('coordinate_frame') != 'world_rdf' or proof.get('unit') != 'metre' or
            proof.get('transform_direction') != 'T_W0_Ct' or
            proof.get('session_id') != evidence.get('session_id') or
            proof.get('reference_frame_id') != evidence.get('reference_frame_id')):
        raise ValueError('Source manifest / fixed W0 reference proof mismatch')
    ids = evidence.get('accepted_frame_ids')
    if (not isinstance(ids, list) or not 3 <= len(ids) <= 20 or any(not _count(x) for x in ids) or
            len(set(ids)) != len(ids) or ids[0] != evidence['reference_frame_id'] or
            evidence.get('accepted_frame_count') != len(ids) or
            not _count(evidence.get('source_frame_count')) or evidence['source_frame_count'] < len(ids)):
        raise ValueError('Invalid accepted frame identities or counts')
    frames = budget.read_frames(dataset / 'frames.jsonl')
    if [row.get('frame') for row in frames] != ids:
        raise ValueError('Original frame index differs from accepted frame IDs')
    required = {dataset / 'manifest.json', dataset / 'frames.jsonl', dataset / 'pose_quality_report.json',
                raw / 'manifest.json', raw / 'frames.jsonl', raw / 'derived/local_capture_report.json',
                raw / 'derived/live_tracking.jsonl'}
    for parent, names in ((raw, ('rgb', 'depth', 'pose', 'calibration')), (dataset, ('pose', 'calibration'))):
        for name in names:
            for path in (parent / name).rglob('*'):
                budget.check()
                if path.is_file():
                    required.add(path)
                if len(required) > 50000:
                    raise ValueError('Source file-count budget exceeded')
    source = _hash_map(evidence.get('source_sha256'))
    if not required.issubset({Path(name) for name in source}):
        raise ValueError('Source fingerprint set omits original evidence files')
    truth = evidence.get('synthetic_ground_truth')
    truth_path = Path(truth['mesh']).resolve() if isinstance(truth, dict) else None
    if any(not (Path(name).is_relative_to(raw) or Path(name).is_relative_to(dataset) or
                Path(name) == truth_path) for name in source):
        raise ValueError('Source fingerprint points outside the bound source data')
    impl = _hash_map(evidence.get('implementation_sha256'))
    if set(impl) != {str((ROOT / name).resolve()) for name in IMPLEMENTATION_FILES}:
        raise ValueError('Implementation fingerprint set is missing or incompatible')
    evidence['source_sha256'], evidence['implementation_sha256'] = source, impl


def _geometry_input_signature(variant, evidence, output, hashes, budget):
    dataset = Path(variant.get('actual_dataset', '')).resolve()
    if dataset != Path(evidence['dataset']).resolve() and not dataset.is_relative_to(output / 'derived'):
        raise ValueError('Actual geometry input is outside this comparison or original data')
    frame_path = dataset / 'frames.jsonl'
    frames = budget.read_frames(frame_path)
    if [f.get('frame') for f in frames] != evidence['accepted_frame_ids']:
        raise ValueError('Actual geometry input substituted accepted frames')
    _merge_hashes(hashes, {str(frame_path): budget.digest(frame_path)})
    result = []
    for frame in frames:
        values = {}
        for kind in ('rgb', 'depth', 'pose'):
            path = (dataset / frame[kind]).resolve()
            if not (path.is_relative_to(Path(evidence['raw_source'])) or path.is_relative_to(output / 'derived')):
                raise ValueError('Actual input file escaped the source/comparison boundary')
            digest = budget.digest(path)
            _merge_hashes(hashes, {str(path): digest})
            values[kind] = digest
            if kind == 'pose':
                pose = budget.read(path)
                matrix = pose.get('T_WC')
                if (not isinstance(matrix, list) or len(matrix) != 4 or
                        any(not isinstance(row, list) or len(row) != 4 or any(not _finite(x) for x in row) for row in matrix) or
                        matrix[3] != [0., 0., 0., 1.] or pose.get('session_id') != evidence['session_id'] or
                        pose.get('reference_frame_id') != evidence['reference_frame_id'] or pose.get('frame') != frame['frame']):
                    raise ValueError('Invalid finite W0 pose or pose provenance')
                values['T_W0_Ct'] = matrix
        spec = frame.get('local_temporal_support')
        values['mask_sha256'] = None
        if spec:
            path = Path(spec['mask']).resolve()
            if not path.is_relative_to(output / 'derived'):
                raise ValueError('Temporal mask belongs to another comparison')
            expected = _hash_map({str(path): spec.get('sha256')})
            _verify_hashes(expected, budget)
            _merge_hashes(hashes, expected)
            values['mask_sha256'] = spec['sha256']
        result.append(dict(frame=frame['frame'], rgb_sha256=values['rgb'], depth_sha256=values['depth'],
                           T_W0_Ct=values['T_W0_Ct'], mask_sha256=values['mask_sha256']))
    if result[0]['T_W0_Ct'] != [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]:
        raise ValueError('First reference pose is not fixed W0')
    if sum(row['mask_sha256'] is not None for row in result) != variant.get('masks_applied'):
        raise ValueError('Actual mask count differs from the optimization result')
    return result


def _load_case(path, budget, *, allow_synthetic=False):
    path = _report_path(path)
    report = budget.read(path)
    output = path.parent
    if (report.get('schema') != 'femto-local-comparison-v1' or
            Path(report.get('output', '')).resolve() != output or
            (report.get('coordinate_frame'), report.get('unit'), report.get('transform_direction')) !=
            ('world_rdf', 'metre', 'T_W0_Ct')):
        raise ValueError('Invalid comparison report or coordinate system')
    params = report.get('parameters', {})
    key = parameter_key(params.get('voxel_m'), params.get('sdf_trunc_m'))
    if report.get('state') not in ('complete', 'failed', 'cancelled'):
        raise ValueError('Running or unrecorded comparison cannot be reused')
    if report.get('physical_accuracy_verified') is not False:
        raise ValueError('An internal comparison cannot claim physical accuracy')
    evidence_path = output / 'source_evidence.json'
    if Path(report.get('source_evidence', '')).resolve() != evidence_path or not evidence_path.is_file():
        raise ValueError('Comparison lacks verified source evidence')
    evidence = budget.read(evidence_path)
    _source_contract(evidence, budget)
    for key_name in ('session_id', 'reference_frame_id', 'data_kind', 'source_frame_count', 'accepted_frame_ids'):
        if report.get(key_name) != evidence.get(key_name):
            raise ValueError('Comparison report source/session/frame mismatch: ' + key_name)
    if Path(report.get('source_dataset', '')).resolve() != Path(evidence['dataset']).resolve():
        raise ValueError('Comparison dataset mismatch')
    if evidence.get('data_kind') not in ('real_femto', 'synthetic') or (evidence['data_kind'] == 'synthetic' and not allow_synthetic):
        raise ValueError('Synthetic replay requires explicit allow_synthetic; unknown source is rejected')
    if report.get('source_unchanged') is not True or report.get('latest_unchanged') is not True:
        raise ValueError('Comparison did not preserve its source or business latest')
    if report.get('source_changed') != [] or report.get('latest_changed') != []:
        raise ValueError('Comparison recorded source/latest changes')
    hashes = {str(path): budget.digest(path), str(evidence_path): budget.digest(evidence_path)}
    _merge_hashes(hashes, evidence['source_sha256'])
    _merge_hashes(hashes, evidence['implementation_sha256'])
    variants = report.get('variants', [])
    if report['state'] == 'complete' and [v.get('name') for v in variants] != list(VARIANTS):
        raise ValueError('Complete comparison must contain A, B and C')
    if len(variants) > 3:
        raise ValueError('Unexpected comparison variants')
    # Failed/cancelled work is listed, but partial models never enter the parameter ranking.
    if report['state'] != 'complete':
        return dict(key=key, path=path, report=report, evidence=evidence, hashes=hashes)
    view, previous, signatures = None, None, []
    for index, variant in enumerate(variants):
        name = VARIANTS[index]
        reconstruction = output / name / 'reconstruction'
        if (variant.get('state') != 'complete' or variant.get('frames') != evidence['accepted_frame_ids'] or
                variant.get('accepted_frame_count') != len(evidence['accepted_frame_ids']) or
                variant.get('source_frame_count') != evidence['source_frame_count'] or
                variant.get('physical_accuracy_verified') is not False or
                Path(variant.get('reconstruction', '')).resolve() != reconstruction):
            raise ValueError('Variant frame/source/geometry binding mismatch')
        for filename, expected in (('variant.json', variant), ('comparison_metrics.json', variant.get('metrics')),
                                   ('comparison_view.json', variant.get('metrics', {}).get('fixed_view'))):
            artifact = output / name / filename if filename == 'variant.json' else reconstruction / filename
            if budget.read(artifact) != expected:
                raise ValueError('Variant or metric file differs from the recorded comparison')
            hashes[str(artifact)] = budget.digest(artifact)
        for filename, field in (('scene_tsdf_mesh_m.ply', 'mesh_sha256'), ('initial_observed_points_m.ply', 'initial_cloud_sha256')):
            digest = variant.get(field)
            if not isinstance(digest, str) or not re.fullmatch('[a-f0-9]{64}', digest):
                raise ValueError('Missing geometry fingerprint')
            hashes[str(reconstruction / filename)] = digest
        metric = variant.get('metrics', {})
        current_view = metric.get('fixed_view', {})
        if (metric.get('coordinate_frame') != 'world_rdf' or metric.get('unit') != 'metre' or
                current_view.get('session_id') != evidence['session_id'] or
                current_view.get('reference_frame_id') != evidence['reference_frame_id'] or
                current_view.get('view') != 'initial_color_camera_locked' or
                current_view.get('camera_to_world') != [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]):
            raise ValueError('Fixed initial W0 projection mismatch')
        if view is not None and current_view != view:
            raise ValueError('A/B/C fixed projection differs')
        view = current_view
        size = current_view.get('display_image_size')
        if not isinstance(size, list) or len(size) != 2 or any(not _count(x) or x <= 0 for x in size) or size[0]*size[1] > 8388608:
            raise ValueError('Invalid or over-budget fixed projection size')
        if ('synthetic_ground_truth' in metric and
                (evidence['data_kind'] != 'synthetic' or not evidence.get('synthetic_ground_truth') or
                 metric['synthetic_ground_truth'].get('source') != evidence['synthetic_ground_truth'])):
            raise ValueError('Synthetic truth cannot be attached to real or unbound data')
        status = variant.get('optimization', {}).get('status')
        same = variant.get('same_as')
        if index == 0:
            if (status != 'baseline' or same is not None or variant.get('geometry_reused') is not False or
                    Path(variant.get('actual_dataset', '')).resolve() != Path(evidence['dataset']).resolve() or
                    variant.get('actual_pose_source') != evidence['pose_source'] or variant.get('masks_applied') != 0):
                raise ValueError('A is not original poses plus original keyframe depth')
        else:
            if status not in ('adopted', 'not_adopted'):
                raise ValueError('Missing actual optimization decision')
            if status == 'not_adopted':
                if (same != previous['name'] or variant.get('geometry_reused') is not True or
                        any(variant.get(k) != previous.get(k) for k in ('actual_dataset', 'actual_pose_source',
                            'mesh_sha256', 'initial_cloud_sha256', 'masks_applied'))):
                    raise ValueError('Not-adopted optimization must use the unchanged previous input and mesh')
            elif same is not None or variant.get('geometry_reused') is not False:
                raise ValueError('Adopted optimization incorrectly claims unchanged geometry reuse')
            if index == 1 and status == 'adopted' and (variant.get('actual_pose_source') != 'offline_robust_icp_research_grade' or variant.get('masks_applied') != 0):
                raise ValueError('Adopted refinement pose/mask source mismatch')
            if index == 2 and (variant.get('actual_pose_source') != previous.get('actual_pose_source') or
                               (status == 'adopted' and (not _count(variant.get('masks_applied')) or variant['masks_applied'] < 1))):
                raise ValueError('Adopted temporal filter must preserve pose source and apply an actual mask')
            if variant.get('initial_cloud_sha256') != variants[0].get('initial_cloud_sha256'):
                raise ValueError('Initial raw cloud differs between variants')
        signatures.append(_geometry_input_signature(variant, evidence, output, hashes, budget))
        if index == 2 and any(a['T_W0_Ct'] != b['T_W0_Ct'] for a, b in zip(signatures[1], signatures[2])):
            raise ValueError('Temporal filtering changed the selected poses')
        previous = variant
    return dict(key=key, path=path, report=report, evidence=evidence, hashes=hashes, geometry_inputs=signatures)


def _rms(rows, field='rmse_m', count='sample_count'):
    if not rows or any(not _count(row.get(count)) or
                       (row[count] and (not _finite(row.get(field)) or row[field] < 0)) for row in rows):
        return None
    total = sum(row[count] for row in rows)
    return math.sqrt(sum(row[count] * row[field] ** 2 for row in rows if row[count]) / total) if total else None


def _sum_counts(rows, key):
    return sum(row[key] for row in rows) if rows and all(_count(row.get(key)) for row in rows) else None


def _variant_summary(variant):
    metric = variant.get('metrics', {})
    geometry = metric.get('internal_geometric_residual', {}).get('frames', [])
    depth = metric.get('depth_consistency', {}).get('frames', [])
    pixels = metric.get('valid_pixels', {}).get('frames', [])
    valid_ratios = [row.get('actual_valid_pixel_ratio') for row in pixels]
    return dict(name=variant['name'], actual_dataset=variant.get('actual_dataset'),
        actual_pose_source=variant.get('actual_pose_source'), optimization=copy.deepcopy(variant.get('optimization')),
        same_as=variant.get('same_as'), geometry_reused=variant.get('geometry_reused'),
        mesh_sha256=variant.get('mesh_sha256'), initial_cloud_sha256=variant.get('initial_cloud_sha256'),
        counts=copy.deepcopy(variant.get('reconstruction_metrics', {})), resource_usage=copy.deepcopy(variant.get('resource_usage', {})),
        internal_geometry=dict(pooled_rmse_m=_rms(geometry), sample_count=_sum_counts(geometry, 'sample_count'),
                               per_frame=copy.deepcopy(geometry), definition='Pooled RMS from sample counts; no pooled percentile is inferred'),
        depth_consistency=dict(on_hits_rmse_m=_rms([r.get('error_on_hits', {}) for r in depth]),
            missing_as_zero_rmse_m=_rms([r.get('error_missing_as_zero', {}) for r in depth]),
            sampled_pixels=_sum_counts(depth, 'sampled_pixels'), missing_hits=_sum_counts(depth, 'missing_hit_pixels'),
            per_frame=copy.deepcopy(depth), definition='Raw depth vs first mesh hit; missing hits remain in the separate zero-depth residual'),
        valid_pixels=dict(raw_valid_pixels=_sum_counts(pixels, 'raw_valid_pixels'), actual_valid_pixels=_sum_counts(pixels, 'actual_valid_pixels'),
            masked_pixels=_sum_counts(pixels, 'masked_pixels'),
            mean_actual_valid_ratio=sum(valid_ratios)/len(valid_ratios) if valid_ratios and all(_finite(v) for v in valid_ratios) else None,
            per_frame=copy.deepcopy(pixels)),
        fixed_projection=copy.deepcopy(metric.get('fixed_projection', {})), topology=copy.deepcopy(metric.get('topology', {})),
        axis_aligned_extent_m=copy.deepcopy(metric.get('axis_aligned_extent_m')), mesh_surface_area_m2=metric.get('mesh_surface_area_m2'),
        synthetic_ground_truth=copy.deepcopy(metric.get('synthetic_ground_truth')),
        noise=dict(status='not_evaluated', reason='No predefined stationary planar ROI or independent reference'),
        ghosting=dict(status='not_evaluated', reason='No labelled double-surface reference'),
        roi_hole_filling=dict(status='not_evaluated', reason='No predetermined ROI with annotated new-view depth support'),
        physical_accuracy=dict(status='not_evaluated', accepted=None, reason='No independent physical reference measurement'))


def _resource_summary(report):
    stages = report.get('stages', [])
    def maximum(key):
        values = [s.get(key) for s in stages]
        return max(values) if values and all(_finite(v) and v >= 0 for v in values) else None
    return dict(comparison_elapsed_s=report.get('elapsed_s'), peak_job_committed_mb=maximum('peak_job_committed_mb'),
        peak_rss_sampled_mb=maximum('peak_rss_sampled_mb'), stages=copy.deepcopy(stages),
        definition='Original comparison elapsed; maximum of sequential stage peaks, never sum or RSS-to-commit substitution')


def _extent_delta(current, baseline):
    if (not isinstance(current, list) or not isinstance(baseline, list) or len(current) != 3 or len(baseline) != 3 or
            not all(_finite(v) and v >= 0 for v in current + baseline)):
        return dict(absolute_mm=None, relative_percent=None)
    return dict(absolute_mm=[(a-b)*1000 for a, b in zip(current, baseline)],
                relative_percent=[100*(a-b)/b if b else None for a, b in zip(current, baseline)])


def _validate_same_run(first, current):
    if any(first['evidence'].get(k) != current['evidence'].get(k) for k in IDENTITY_KEYS):
        raise ValueError('Parameter cases differ in source SHA, implementation SHA, W0 or accepted frames')
    ignored = {'voxel_m', 'sdf_trunc_m'}
    if ({k:v for k,v in first['report']['parameters'].items() if k not in ignored} !=
            {k:v for k,v in current['report']['parameters'].items() if k not in ignored} or
            first['report'].get('budgets') != current['report'].get('budgets')):
        raise ValueError('Parameter cases differ in non-TSDF parameters or task budgets')
    if first['report']['state'] == current['report']['state'] == 'complete':
        if first['geometry_inputs'] != current['geometry_inputs']:
            raise ValueError('Parameter cases have different actual poses, source depth or temporal masks')
        for a, b in zip(first['report']['variants'], current['report']['variants']):
            if (a['metrics']['fixed_view'] != b['metrics']['fixed_view'] or
                    a['initial_cloud_sha256'] != b['initial_cloud_sha256'] or
                    a['optimization']['status'] != b['optimization']['status'] or
                    a['actual_pose_source'] != b['actual_pose_source']):
                raise ValueError('Parameter cases differ in fixed projection or initial raw cloud')


def _safe_output(output, records):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Parameter study output must be a new directory')
    for record in records:
        for root in (Path(record['evidence']['dataset']), Path(record['evidence']['raw_source']), record['path'].parent):
            if output == root or output.is_relative_to(root):
                raise ValueError('Parameter study cannot write inside an existing source or comparison')
    return output


def summarize_existing(report_paths, output, *, allow_synthetic=False, cancelled=None,
                       cancel_file=None, timeout_s=300., progress=None):
    """Audit 1..8 immutable reports and write a separate, bilingual parameter study.

    A complete 5 mm / 20 mm baseline is mandatory. Failed/cancelled cases with
    valid source evidence are retained, without treating partial models as results.
    """
    paths = list(report_paths)
    if not 1 <= len(paths) <= 8:
        raise ValueError('A parameter study requires 1 to 8 reports')
    if Path(output).exists():
        raise FileExistsError('Parameter study output must be a new directory')
    budget = AuditBudget(timeout_s, cancelled=cancelled, cancel_file=cancel_file)
    records = [_load_case(path, budget, allow_synthetic=allow_synthetic) for path in paths]
    by_key = {record['key']: record for record in records}
    if len(by_key) != len(records):
        raise ValueError('Duplicate voxel / truncation candidate')
    baseline = by_key.get(parameter_key(*BASELINE))
    if baseline is None or baseline['report']['state'] != 'complete':
        raise ValueError('A complete 5 mm / 20 mm baseline is required')
    records = [baseline] + [r for r in records if r is not baseline]
    output = _safe_output(output, records)
    hashes, latest = {}, {}
    for record in records:
        budget.check()
        _validate_same_run(baseline, record)
        _merge_hashes(hashes, record['hashes'])
        # Old reports may predate a legitimate newer business result. Snapshot
        # present indices now; do not require the old value or restore it.
        for name in _hash_map(record['report'].get('latest_before', {}), missing_allowed=True):
            path = Path(name)
            latest[name] = budget.digest(path) if path.is_file() else None
    _verify_hashes(hashes, budget)
    budget.check()
    output.mkdir(parents=True, exist_ok=False)
    report = dict(schema=SCHEMA, job_id=uuid.uuid4().hex, state='running', mode='verified_existing_reports',
        created_local=time.strftime('%Y-%m-%d %H:%M:%S %z'), source_dataset=baseline['evidence']['dataset'],
        data_kind=baseline['evidence']['data_kind'], session_id=baseline['evidence']['session_id'],
        reference_frame_id=baseline['evidence']['reference_frame_id'], reference_provenance=baseline['evidence']['reference_provenance'],
        source_frame_count=baseline['evidence']['source_frame_count'], accepted_frame_ids=baseline['evidence']['accepted_frame_ids'],
        rejected_tracking=baseline['evidence'].get('rejected_tracking', []), coordinate_frame='world_rdf', unit='metre',
        transform_direction='T_W0_Ct', baseline=baseline['key'], default_changed=False, physical_accuracy_verified=False,
        budgets=dict(max_cases=8, timeout_s=timeout_s, json_file_limit_bytes=budget.json_limit,
                     cumulative_json_read_limit_bytes=budget.json_total_limit,
                     total_hash_read_limit_bytes=budget.hash_limit, reconstruction_workers=0),
        fixed_view=baseline['report']['variants'][0]['metrics']['fixed_view'],
        source_sha256=baseline['evidence']['source_sha256'], implementation_sha256=baseline['evidence']['implementation_sha256'],
        summarizer_sha256={str(path): budget.digest(path) for path in (Path(__file__).resolve(), ROOT/'scene_data_kind.py')}, cases=[], completed_cases=0,
        latest_before=latest, latest_unchanged=None, artifacts_unchanged=None,
        recommendation=dict(status='default_unchanged', voxel_m=.005, sdf_trunc_m=.02,
                            reason='Smaller voxels and lower internal residuals do not prove physical accuracy'),
        limitations=['Same source, frame IDs, W0 and gates; only voxel/truncation parameters differ.',
                     'Reports are reused after SHA verification; recorded elapsed is not this audit execution time.',
                     'Geometry residual, depth residual and valid pixels are separate internal measurements.',
                     'Bounds and mesh area are not independent object dimensions; boundary edges are not physical holes.',
                     'Real noise, ghosting, ROI benefit and physical accuracy remain unverified.'])
    def publish():
        report['audit_elapsed_s'] = time.monotonic()-budget.started
        _write(output / 'parameter_study_report.json', report)
        if progress:
            progress(copy.deepcopy(report))
    publish()
    try:
        base_variants = {v['name']: v for v in baseline['report']['variants']}
        for record in records:
            budget.check()
            original = record['report']
            row = dict(key=record['key'], parameters=original['parameters'], state=original['state'],
                       comparison_report=str(record['path']), report_sha256=record['hashes'][str(record['path'])],
                       execution='reused_verified_report', resource_usage=_resource_summary(original),
                       error=original.get('error'), variants=[])
            report['cases'].append(row)
            if original['state'] == 'complete':
                assets = output / 'assets' / record['key']
                assets.mkdir(parents=True, exist_ok=False)
                for variant in original['variants']:
                    budget.check()
                    summary = _variant_summary(variant)
                    signature = record['geometry_inputs'][list(VARIANTS).index(variant['name'])]
                    summary['geometry_input_sha256'] = hashlib.sha256(json.dumps(signature, sort_keys=True, allow_nan=False).encode()).hexdigest()
                    summary['extent_change_vs_5mm'] = _extent_delta(summary['axis_aligned_extent_m'],
                        base_variants[variant['name']]['metrics'].get('axis_aligned_extent_m'))
                    summary['images'] = {}
                    for kind in ('surface', 'reference_rgb', 'initial_raw'):
                        source = Path(variant['reconstruction']) / f'comparison_{kind}.png'
                        if source.stat().st_size > 16 * 1024**2:
                            raise ValueError('Comparison image exceeds 16 MiB report budget')
                        image_bytes = source.read_bytes()
                        if (not image_bytes.startswith(b'\x89PNG\r\n\x1a\n') or len(image_bytes) < 33 or
                                struct.unpack('>II', image_bytes[16:24]) != tuple(report['fixed_view']['display_image_size'])):
                            raise ValueError('Invalid recorded comparison PNG')
                        digest = hashlib.sha256(image_bytes).hexdigest()
                        _merge_hashes(hashes, {str(source): digest})
                        destination = assets / f'{variant["name"]}_{kind}.png'
                        with destination.open('xb') as stream:
                            stream.write(image_bytes)
                        summary['images'][kind] = destination.relative_to(output).as_posix()
                    row['variants'].append(summary)
                report['completed_cases'] += 1
            publish()
        _verify_hashes(hashes, budget)
        _verify_hashes(latest, budget)
        report.update(artifacts_unchanged=True, latest_unchanged=True,
                      state='complete' if all(r['state']=='complete' for r in report['cases']) else 'incomplete',
                      audited_file_count=len(hashes), audit_hash_bytes=budget.hashed_bytes)
        _write(output / 'audited_files.json', hashes)
        _write_html(output, report)
    except StudyStopped as error:
        report.update(state=error.state, error=str(error))
    except KeyboardInterrupt:
        report.update(state='cancelled', error='Parameter study interrupted; no business result published')
    except Exception as error:
        report.update(state='failed', error=str(error))
    finally:
        publish()
    return report


def run_study(dataset, output, *, candidates=DEFAULT_CANDIDATES, reuse_paths=(), allow_synthetic=False,
              stage_timeout_s=300., memory_mb=3072., render_width=640, max_views=20, total_timeout_s=1800.,
              cancelled=None, cancel_file=None, progress=None):
    """Run missing candidates serially with run_comparison, then audit their reports.

    Existing reports are not rerun. New outputs and the final summary live in this
    new task directory. The original 5 mm / 20 mm default remains untouched.
    """
    from local_comparison import run_comparison
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    candidates = tuple(candidates)
    keys = [parameter_key(*pair) for pair in candidates]
    if not 1 <= len(keys) <= 8 or len(set(keys)) != len(keys) or BASELINE not in candidates:
        raise ValueError('Unique candidate list including the 5 mm baseline is required; max 8')
    if (not _finite(stage_timeout_s) or not 1 <= stage_timeout_s <= 3600 or
            not _finite(memory_mb) or not 128 <= memory_mb <= 16384 or
            type(render_width) is not int or not 128 <= render_width <= 1280 or
            type(max_views) is not int or not 3 <= max_views <= 20):
        raise ValueError('Invalid stage time, memory, render or view budget')
    budget = AuditBudget(total_timeout_s, cancelled=cancelled, cancel_file=cancel_file)
    manifest = budget.read(dataset / 'manifest.json')
    raw = Path(manifest.get('source_dataset', dataset)).resolve()
    if output.exists():
        raise FileExistsError('Parameter study output must be a new directory')
    if output.is_relative_to(dataset) or output.is_relative_to(raw):
        raise ValueError('Parameter study output cannot be inside its source')
    reused = [_load_case(path, budget, allow_synthetic=allow_synthetic) for path in reuse_paths]
    reuse = {r['key']: r for r in reused}
    if len(reuse) != len(reused) or any(k not in keys for k in reuse):
        raise ValueError('Duplicate or unrequested reuse candidate')
    for record in reused:
        if Path(record['evidence']['dataset']).resolve() != dataset:
            raise ValueError('Reuse candidate belongs to another dataset')
        requested = dict(max_views=max_views, stage_timeout_s=stage_timeout_s, memory_mb=memory_mb, render_width=render_width)
        if any(record['report']['budgets'].get(k) != v for k,v in requested.items()):
            raise ValueError('Reuse task budgets differ from the requested study')
        _validate_same_run(reused[0], record)
        _verify_hashes(record['hashes'], budget)
    _safe_output(output, reused)
    output.mkdir(parents=True, exist_ok=False)
    job = dict(schema=SCHEMA, job_id=uuid.uuid4().hex, state='running', source_dataset=str(dataset),
               active_case=None, cases=[], worker_count=1, default_changed=False, physical_accuracy_verified=False)
    def publish():
        job['elapsed_s'] = time.monotonic()-budget.started
        _write(output / 'parameter_study_job.json', job)
        if progress:
            progress(copy.deepcopy(job))
    def stopped():
        try:
            budget.check()
            return False
        except StudyStopped:
            return True
    publish()
    try:
        reports, first = [], None
        for pair, key in zip(candidates, keys):
            budget.check()
            job['active_case'] = key
            publish()
            if key in reuse:
                record = reuse[key]
                execution = 'reused_verified_report'
            else:
                destination = output / 'runs' / key
                def update(current):
                    job['active_stage'] = copy.deepcopy(current['stages'][-1]) if current.get('stages') else None
                    publish()
                run_comparison(dataset, destination, voxel_m=pair[0], sdf_trunc_m=pair[1],
                    max_views=max_views, stage_timeout_s=stage_timeout_s, memory_mb=memory_mb,
                    render_width=render_width, allow_synthetic=allow_synthetic, cancelled=stopped, progress=update)
                budget.check()
                record = _load_case(destination, budget, allow_synthetic=allow_synthetic)
                execution = 'executed_now'
            if first is not None:
                _validate_same_run(first, record)
            else:
                first = record
            reports.append(record['path'])
            job['cases'].append(dict(key=key, state=record['report']['state'], report=str(record['path']), execution=execution))
            publish()
        budget.check()
        summary = summarize_existing(reports, output / 'summary', allow_synthetic=allow_synthetic,
            cancelled=cancelled, cancel_file=cancel_file, timeout_s=max(1., total_timeout_s-(time.monotonic()-budget.started)))
        budget.check()
        job.update(state=summary['state'], summary=str(output / 'summary/parameter_study_report.json'), active_case=None)
    except StudyStopped as error:
        job.update(state=error.state, error=str(error))
    except KeyboardInterrupt:
        job.update(state='cancelled', error='Parameter study interrupted; no business result published')
    except Exception as error:
        job.update(state='failed', error=str(error))
    finally:
        publish()
    return job


def _dual(zh, en):
    return f'<span data-zh="{html.escape(zh, quote=True)}" data-en="{html.escape(en, quote=True)}">{html.escape(zh)}</span>'


def _shown(value, scale=1, suffix=''):
    return f'{value*scale:.3f}{suffix}' if _finite(value) else _dual('未记录', 'Not recorded')


def _recorded(value):
    return _dual('未记录', 'Not recorded') if value is None else html.escape(str(value))


def _status(value):
    names = {'complete': ('已完成', 'Complete'), 'incomplete': ('部分未完成', 'Incomplete'),
             'failed': ('失败', 'Failed'), 'cancelled': ('已取消', 'Cancelled'),
             'real_femto': ('已有 Femto 实际数据回放', 'Recorded Femto data replay'),
             'synthetic': ('合成数据验证', 'Synthetic data validation'),
             'online_rgbd_research_grade': ('在线 RGB-D 研发位姿', 'Online RGB-D research poses'),
             'offline_robust_icp_research_grade': ('离线鲁棒 ICP 研发位姿', 'Offline robust ICP research poses'),
             'synthetic_ground_truth': ('合成真值位姿', 'Synthetic ground-truth poses')}
    return _dual(*names[value]) if value in names else _recorded(value)


def _write_html(output, report):
    cards = []
    width, height = report['fixed_view']['display_image_size']
    for case in report['cases']:
        versions = []
        for variant in case['variants']:
            count, geo, depth = variant['counts'], variant['internal_geometry'], variant['depth_consistency']
            details = html.escape(json.dumps(variant, ensure_ascii=False, indent=2))
            status = variant['optimization']['status']
            label = {'baseline': ('原始基线', 'Original baseline'), 'adopted': ('已采用', 'Adopted'),
                     'not_adopted': ('未采用', 'Not adopted')}.get(status, ('未记录', 'Not recorded'))
            same = (f'<p>{_dual("与上一版本相同，实际输入与几何未变", "Same input and geometry as previous version")}</p>'
                    if variant['same_as'] else '')
            delta = variant['extent_change_vs_5mm']['absolute_mm']
            delta_display = ', '.join(f'{x:.3f}' for x in delta) if delta is not None else _recorded(None)
            rows = [('实际位姿来源', 'Actual pose source', _status(variant['actual_pose_source'])),
                    ('点 / 面', 'Points / triangles', f'{_recorded(count.get("fused_point_count"))} / {_recorded(count.get("mesh_triangles"))}'),
                    ('内部几何 RMS', 'Internal geometric RMS', _shown(geo['pooled_rmse_m'], 1000, ' mm')),
                    ('深度 RMS（仅命中）', 'Depth RMS (hits only)', _shown(depth['on_hits_rmse_m'], 1000, ' mm')),
                    ('深度 RMS（缺失按零）', 'Depth RMS (missing as zero)', _shown(depth['missing_as_zero_rmse_m'], 1000, ' mm')),
                    ('无命中 / 采样', 'Missing hits / samples', f'{_recorded(depth["missing_hits"])} / {_recorded(depth["sampled_pixels"])}'),
                    ('平均有效像素', 'Mean valid pixels', _shown(variant['valid_pixels']['mean_actual_valid_ratio'], 100, '%')),
                    ('边界边（不是孔洞数）', 'Boundary edges (not hole count)', _shown(variant['topology'].get('boundary_edges'))),
                    ('W0 包围盒差值 mm', 'W0 extent delta mm', delta_display)]
            table = ''.join(f'<tr><th>{_dual(zh, en)}</th><td>{value}</td></tr>' for zh, en, value in rows)
            images = ' '.join(f'data-{key}="{html.escape(value, quote=True)}"' for key, value in variant['images'].items())
            versions.append(f'<section data-version="{variant["name"]}"><p>{_dual(*label)}</p>{same}'
                f'<div class="viewport" style="aspect-ratio:{width}/{height}"><img {images} src="{variant["images"]["surface"]}" width="{width}" height="{height}" alt="Fixed W0 projection"></div>'
                f'<table>{table}</table><details><summary>{_dual("逐帧指标与未采用原因", "Per-frame metrics and optimization decisions")}</summary><pre>{details}</pre></details></section>')
        resource = case['resource_usage']
        voxel, trunc = case['parameters']['voxel_m']*1000, case['parameters']['sdf_trunc_m']*1000
        title = _dual(f'体素 {voxel:g} mm / 截断 {trunc:g} mm', f'Voxel {voxel:g} mm / truncation {trunc:g} mm')
        cards.append(f'<article><h2>{title}</h2><p>{_status(case["state"])}</p>'
            f'<p>{_dual("原对照总耗时", "Original comparison total time")}: {_shown(resource["comparison_elapsed_s"],1," s")}<br>'
            f'{_dual("进程组峰值提交内存", "Peak job committed memory")}: {_shown(resource["peak_job_committed_mb"],1," MiB")}</p>'
            f'<a href="{html.escape(Path(case["comparison_report"]).with_suffix(".html").as_uri(),quote=True)}">{_dual("详细 A/B/C 报告", "Detailed A/B/C report")}</a>'
            f'<p>{html.escape(case.get("error") or "")}</p>{"".join(versions)}</article>')
    document = f'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Femto TSDF parameter study</title><style>
body{{font:16px system-ui,sans-serif;background:#111820;color:#e7eef5;margin:20px}}a{{color:#80d8f1}}select,input{{font:inherit;margin:6px;padding:5px}}.notice{{padding:14px;background:#333020;border:1px solid #988651}}.controls{{position:sticky;top:0;background:#18222d;padding:8px;z-index:2}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(440px,100%),1fr));gap:16px}}article{{padding:14px;background:#1b2632;min-width:0}}h2{{font-size:18px;overflow-wrap:anywhere}}table{{width:100%;border-collapse:collapse}}td,th{{padding:5px;text-align:left;border-bottom:1px solid #435264}}.viewport{{overflow:hidden;background:#181d24}}.viewport img{{width:100%;height:100%;object-fit:contain;transform:scale(var(--zoom,1));transform-origin:center}}pre{{white-space:pre-wrap;word-break:break-word;font-size:12px}}[hidden]{{display:none!important}}
</style><h1>{_dual('TSDF 参数对照', 'TSDF parameter study')}</h1>
<p class="notice">{_dual('默认仍为 5 mm / 20 mm。较小体素不代表更高实物精度；几何、深度残差及包围盒变化均为内部指标。噪声、重影、ROI 补洞和实物精度未验证。', 'Defaults remain 5 mm / 20 mm. Smaller voxels do not imply better physical accuracy. Geometry/depth residuals and extents are internal metrics. Noise, ghosting, ROI filling and physical accuracy are unverified.')}</p>
<p>{_dual('当前来源', 'Data source')}: {_status(report['data_kind'])} · {html.escape(report['session_id'])}<br>{_dual('源帧 / 采用视角', 'Source frames / accepted views')}: {report['source_frame_count']} / {len(report['accepted_frame_ids'])}</p>
<div class="controls"><label>{_dual('语言', 'Language')}<select id="lang"><option value="zh">中文</option><option value="en">English</option></select></label>
<label>{_dual('所有参数统一版本', 'Version for all parameters')}<select id="version">{''.join(f'<option>{name}</option>' for name in VARIANTS)}</select></label>
<label>{_dual('显示层', 'Display layer')}<select id="layer"><option value="surface" data-zh="几何" data-en="Geometry">几何</option><option value="reference_rgb" data-zh="初始RGB" data-en="Initial RGB">初始RGB</option><option value="initial_raw" data-zh="初始原始点云" data-en="Initial raw cloud">初始原始点云</option></select></label>
<label>{_dual('统一缩放', 'Shared zoom')}<input id="zoom" type="range" min="1" max="3" step=".1" value="1"></label></div>
<p>{_dual('固定初始相机 W0，所有参数使用相同投影、尺度和照明；只操作显示副本。原任务耗时与本次只读核验耗时分别记录。', 'Fixed initial camera W0, equal projection, scale and lighting for every parameter. Display copies only. Original processing and this read-only audit have separate elapsed times.')}</p>
<div class="cards">{''.join(cards)}</div><p><a href="parameter_study_report.json">JSON</a> · <a href="audited_files.json">SHA-256</a></p>
<script>
function update(){{let v=document.getElementById('version').value,l=document.getElementById('layer').value;document.querySelectorAll('[data-version]').forEach(e=>e.hidden=e.dataset.version!==v);document.querySelectorAll('.viewport img').forEach(e=>e.src=e.getAttribute('data-'+l));}}
document.getElementById('version').onchange=update;document.getElementById('layer').onchange=update;document.getElementById('zoom').oninput=e=>document.documentElement.style.setProperty('--zoom',e.target.value);
document.getElementById('lang').onchange=e=>{{document.documentElement.lang=e.target.value;document.querySelectorAll('[data-zh][data-en]').forEach(n=>n.textContent=n.getAttribute('data-'+e.target.value));}};update();
</script></html>'''
    (output / 'parameter_study_report.html').write_text(document, encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    summarize = sub.add_parser('summarize', help='Verify existing reports; no reconstruction')
    summarize.add_argument('--report', action='append', required=True, type=Path)
    summarize.add_argument('--output', required=True, type=Path)
    summarize.add_argument('--allow-synthetic', action='store_true')
    summarize.add_argument('--timeout-s', type=float, default=300.)
    summarize.add_argument('--cancel-file', type=Path)
    run = sub.add_parser('run', help='Run the five defined parameter candidates serially')
    run.add_argument('--dataset', required=True, type=Path)
    run.add_argument('--output', required=True, type=Path)
    run.add_argument('--reuse', action='append', default=[], type=Path)
    run.add_argument('--allow-synthetic', action='store_true')
    run.add_argument('--stage-timeout-s', type=float, default=300.)
    run.add_argument('--total-timeout-s', type=float, default=1800.)
    run.add_argument('--memory-mb', type=float, default=3072.)
    run.add_argument('--cancel-file', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == 'summarize':
            result = summarize_existing(args.report, args.output, allow_synthetic=args.allow_synthetic,
                timeout_s=args.timeout_s, cancel_file=args.cancel_file)
        else:
            result = run_study(args.dataset, args.output, reuse_paths=args.reuse, allow_synthetic=args.allow_synthetic,
                stage_timeout_s=args.stage_timeout_s, total_timeout_s=args.total_timeout_s,
                memory_mb=args.memory_mb, cancel_file=args.cancel_file)
        print(json.dumps(dict(state=result['state'], output=str(args.output), error=result.get('error')), ensure_ascii=False))
        return 0 if result['state'] == 'complete' else 2
    except (ValueError, FileNotFoundError, FileExistsError, StudyStopped) as error:
        print(json.dumps(dict(state=getattr(error, 'state', 'failed'), error=str(error)), ensure_ascii=False))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
