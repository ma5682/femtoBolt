"""Read-only evidence audit and manual distance experiments, never clinical certification."""
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import uuid

import numpy as np


def fingerprint(path):
    path = Path(path)
    if not path.is_file():
        return dict(path=str(path), status='missing')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return dict(path=str(path.resolve()), status='present', sha256=digest.hexdigest())


def layer_contract(store):
    """A role declaration; support metadata is evidence, not a physical accuracy claim."""
    return dict(schema='femto-data-layers-v1',
        coordinate_frame='world_rdf', unit='metre', axes='+X right, +Y down, +Z forward',
        origin_reference=store.origin_reference, geometry_sha256=store.digest,
        layers=dict(raw='immutable_RGB_Depth_calibration_pose',
                    geometry='derived_source_mesh_not_independent_ground_truth',
                    appearance='GLB_texture_and_display_normals_not_measurement_input',
                    planning='derived_surface_samples_not_robot_commands'),
        source_mesh=store.source, local_context=store.local_context,
        interpolation_policy='no_display_hole_fill_or_texture_used_as_measurement',
        physical_accuracy_verified=False, execution_ready=False)


def build_audit(model):
    root = model.reconstruction
    contract = layer_contract(model.store)
    evidence = {}
    diagnostics = {}
    warnings = []
    for name in ('local_geometry.json', 'reconstruction_report.json', 'multiview_texture_report.json'):
        path = root / name
        evidence[name] = fingerprint(path)
        if path.is_file():
            try:
                diagnostics[name] = json.loads(path.read_text(encoding='utf-8-sig'))
            except (ValueError, OSError):
                warnings.append('Unreadable report: ' + name)
    local = diagnostics.get('local_geometry.json', {})
    if not isinstance(local, dict):
        local = {}
    support_declared = (local.get('measured_only') is True
                        and local.get('inferred_repairs_integrated') is False
                        and isinstance(local.get('support'), dict)
                        and local['support'].get('rule') == 'all_vertices_and_centroid_supported_in_one_raw_view')
    if not support_declared:
        warnings.append('Raw-depth support policy missing or incomplete; do not infer measured support from appearance.')
    for key in ('tracking_report', 'temporal_support_report'):
        if local.get(key):
            path = Path(local[key])
            if not path.is_absolute():
                path = root / path
            evidence[key] = fingerprint(path)
            if path.is_file():
                try:
                    diagnostics[key] = json.loads(path.read_text(encoding='utf-8-sig'))
                except (ValueError, OSError):
                    warnings.append('Unreadable report: ' + key)
    source = model.store.origin_reference.get('source_dataset')
    calibration_match = None
    if source:
        dataset = Path(source)
        for name in ('manifest.json', 'frames.jsonl'):
            evidence[name] = fingerprint(dataset / name)
        try:
            rows = [json.loads(line) for line in (dataset/'frames.jsonl').read_text(encoding='utf-8-sig').splitlines() if line.strip()]
            for row in rows:
                for field in ('rgb', 'depth', 'pose'):
                    asset = (dataset/row[field]).resolve()
                    # Derived datasets intentionally reference immutable original
                    # RGB/Depth via ../../ paths. Audit reads, never writes them.
                    key = f"frame/{row['frame']}/{field}"
                    if field == 'pose' or row['frame'] == model.view_reference.get('reference_frame_id'):
                        evidence[key] = fingerprint(asset)
                        expected = model.view_reference.get(field + '_sha256')
                        if field != 'pose' and expected and evidence[key].get('sha256') != expected:
                            warnings.append('Reference image hash mismatch: ' + field)
                    else:
                        evidence[key] = dict(path=str(asset), status='present' if asset.is_file() else 'missing',
                                             content_hash_checked=False)
        except (OSError, ValueError, KeyError, TypeError):
            warnings.append('Frame evidence inventory incomplete.')
        calibration = (local.get('reference_provenance') or {}).get('calibration', {}).get('files', {})
        comparisons = []
        for name, expected in calibration.items():
            path = (dataset / 'calibration' / name).resolve()
            if path.parent != (dataset / 'calibration').resolve():
                warnings.append('Invalid calibration evidence path')
                comparisons.append(False)
                continue
            current = fingerprint(path)
            current['matches_recorded_hash'] = current.get('sha256') == expected
            comparisons.append(current['matches_recorded_hash'])
            evidence['calibration/' + name] = current
        if comparisons:
            calibration_match = all(comparisons)
        if calibration_match is not True:
            warnings.append('Calibration fingerprint unverified or changed.')
    evidence['measurement_mesh'] = fingerprint(model.store.source)
    if any(v['status'] == 'missing' for v in evidence.values()):
        warnings.append('Some referenced evidence files are missing.')
    projection = model.projection
    tracking = diagnostics.get('tracking_report', {})
    tracking_summary = {key: tracking.get(key) for key in (
        'state', 'frame_count', 'accepted_frames', 'valid_pose_count', 'rejected_frame_count',
        'rejection_counts', 'tracking_lost_events', 'termination', 'latency_p95_ms',
        'local_workspace_radius_m', 'local_workspace_rotation_deg')} if isinstance(tracking, dict) else {}
    return dict(schema='femto-measurement-audit-v1', created_at=datetime.now(timezone.utc).isoformat(),
        contract=contract, view_reference=model.view_reference, evidence=evidence, diagnostics=diagnostics,
        tracking_summary=tracking_summary,
        raw_support_policy_declared=support_declared, calibration_hashes_match=calibration_match,
        support_pixels=dict(total=int(projection.valid.size),
            reference_depth_consistent=int(np.count_nonzero(projection.valid & ~projection.new_support)),
            reference_depth_missing_mesh_supported=int(np.count_nonzero(projection.new_support)),
            unavailable=int(np.count_nonzero(~projection.valid))),
        warnings=warnings, physical_accuracy_verified=False,
        limitations=['Internal residuals are not physical positioning errors.',
            'Reference-depth-missing pixels are not proof of independent multi-view accuracy.',
            'Audit hashes metadata and mesh; it does not revalidate every raw frame or every triangle.',
            'Distance tests cannot certify absolute XYZ, surface normals, contact safety or clinical targeting.'])


def distance_record(model, points, *, test_id, reference_mm, reference_uncertainty_mm,
                    tolerance_mm, instrument, conditions):
    for name, value in (('reference_mm', reference_mm), ('reference_uncertainty_mm', reference_uncertainty_mm),
                        ('tolerance_mm', tolerance_mm)):
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (name != 'reference_uncertainty_mm' and value == 0):
            raise ValueError('Invalid ' + name)
    for value in (test_id, instrument, conditions):
        if not isinstance(value, str) or not value.strip():
            raise ValueError('Test ID, instrument and conditions are required')
    if len(points) != 2:
        raise ValueError('Select exactly two surface points')
    points = [model.store.normalized_point(p) for p in points]
    for point in points:
        if point.get('view_reference') != model.view_reference:
            raise ValueError('Point reference image mismatch')
    xyz = np.asarray([p['point_world_m'] for p in points], dtype=float)
    if xyz.shape != (2, 3) or not np.isfinite(xyz).all():
        raise ValueError('Invalid surface points')
    if np.array_equal(xyz[0], xyz[1]):
        raise ValueError('Select two distinct surface points')
    measured = float(np.linalg.norm(xyz[1] - xyz[0]) * 1000)
    error = measured - reference_mm
    bound = abs(error) + reference_uncertainty_mm
    # This interval classification is a user-reported ruler uncertainty check, not certification.
    outcome = ('within_entered_limit' if bound <= tolerance_mm else
               'outside_entered_limit' if abs(error) - reference_uncertainty_mm > tolerance_mm else
               'indeterminate_reference_uncertainty')
    return dict(schema='femto-manual-distance-test-v1', record_id=uuid.uuid4().hex,
        created_at=datetime.now(timezone.utc).isoformat(), test_id=test_id.strip(),
        reference_mm=reference_mm, reference_uncertainty_mm=reference_uncertainty_mm,
        tolerance_mm=tolerance_mm, instrument=instrument.strip(), conditions=conditions.strip(),
        points=points, measured_mm=measured, signed_error_mm=error, absolute_error_mm=abs(error),
        conservative_error_bound_mm=bound, outcome=outcome,
        contract=layer_contract(model.store), view_reference=model.view_reference,
        physical_accuracy_verified=False, scope='two_point_distance_only_not_absolute_XYZ_or_clinical_validation')


def summarize(records):
    """Never pool different reference targets, geometry versions or test conditions."""
    groups = {}
    for row in records:
        key = json.dumps([row['test_id'], row['reference_mm'], row['reference_uncertainty_mm'],
            row['tolerance_mm'], row['instrument'], row['conditions'], row['contract'], row['view_reference']],
            sort_keys=True, ensure_ascii=False)
        groups.setdefault(key, []).append(row)
    result = []
    for rows in groups.values():
        values = np.asarray([r['measured_mm'] for r in rows])
        errors = values - rows[0]['reference_mm']
        result.append(dict(test_id=rows[0]['test_id'], count=len(rows), record_ids=[r['record_id'] for r in rows],
            geometry_sha256=rows[0]['contract']['geometry_sha256'],
            signed_bias_mm=float(errors.mean()), rmse_mm=float(np.sqrt(np.mean(errors**2))),
            max_absolute_error_mm=float(np.max(np.abs(errors))), p95_absolute_error_mm=float(np.percentile(np.abs(errors), 95)),
            sample_stddev_mm=float(values.std(ddof=1)) if len(values) > 1 else None,
            repeatability_scope='same_geometry_manual_selection_only_not_recapture_repeatability',
            physical_accuracy_verified=False))
    return result


def save_record(model, row):
    folder = model.reconstruction / 'measurement_validation' / 'records'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (uuid.uuid4().hex + '.json')
    path.write_text(json.dumps(row, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    return path


def save_report(model):
    parent = model.reconstruction / 'measurement_validation'
    audit = build_audit(model)
    records, invalid = [], []
    for path in sorted((parent / 'records').glob('*.json')):
        try:
            row = json.loads(path.read_text(encoding='utf-8'))
            if row.get('contract') != layer_contract(model.store) or row.get('view_reference') != model.view_reference:
                invalid.append(dict(file=path.name, reason='model_or_reference_mismatch')); continue
            # Recompute every measurement; saved error/outcome fields are not trusted.
            verified = distance_record(model, row['points'], **{k: row[k] for k in
                ('test_id', 'reference_mm', 'reference_uncertainty_mm', 'tolerance_mm', 'instrument', 'conditions')})
            verified.update(record_id=row['record_id'], created_at=row['created_at'])
            records.append(verified)
        except (ValueError, KeyError, TypeError, OSError):
            invalid.append(dict(file=path.name, reason='invalid_record'))
    folder = parent / ('report_' + uuid.uuid4().hex)
    folder.mkdir(parents=True)
    audit.update(measurements=records, measurement_groups=summarize(records), excluded_records=invalid)
    (folder / 'report.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    keys = ['test_id','measured_mm','reference_mm','reference_uncertainty_mm','signed_error_mm',
            'absolute_error_mm','tolerance_mm','outcome','instrument','conditions','record_id']
    with (folder / 'measurements.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction='ignore'); writer.writeheader()
        for row in records:
            safe = {k: ("'" + v if isinstance(v, str) and v.startswith(('=', '+', '-', '@', '\t', '\r')) else v)
                    for k, v in row.items()}
            writer.writerow(safe)
    lines = ['# 定位数据质量与距离测试报告', '', '状态：实物绝对 XYZ 精度未验证；仅供研发。',
        f'测量记录：{len(records)}；排除记录：{len(invalid)}。',
        '统计仅适用于同模型、同参考长度、同测试条件；重复选点不等于重复采集。',
        '10–20 mm 未写入融合或定位放行门限。误差限由测试者逐次输入，仅用于该次距离实验。',
        '内部配准残差、深度一致性和覆盖比例都不是实际定位误差。', '',
        '## 数据分层', '原始 RGB/Depth/标定/位姿 → 派生几何 → RGB 展示 → 选区与路径。',
        'GLB 纹理、显示法线和路线图不参与测量；原始文件未改写。', '', '## 检查提示']
    lines.extend(audit['warnings'] or ['未发现本次证据检查告警；不代表临床验证通过。'])
    support=audit['support_pixels'];tracking=audit['tracking_summary']
    lines.extend(['', '## 几何与融合诊断',
        f"原始深度支持策略已声明：{audit['raw_support_policy_declared']}；标定指纹一致：{audit['calibration_hashes_match']}。",
        f"参考深度一致像素：{support['reference_depth_consistent']}；首帧缺深度但有网格支持：{support['reference_depth_missing_mesh_supported']}；无可用映射：{support['unavailable']}。",
        '像素统计不是曲面面积，也不证明几何精度。',
        f"跟踪状态：{tracking.get('state', '未记录')}；有效位姿帧：{tracking.get('valid_pose_count', '未记录')}；拒绝帧：{tracking.get('rejected_frame_count', '未记录')}。",
        '拒绝原因：'+json.dumps(tracking.get('rejection_counts'),ensure_ascii=False),
        '完整参数、支持筛选、纹理、跟踪与时间窗口报告快照见 report.json。'])
    for group in audit['measurement_groups']:
        lines.extend(['', f"测试 {group['test_id']}，n={group['count']}：",
            f"偏差 {group['signed_bias_mm']:.3f} mm；RMSE {group['rmse_mm']:.3f} mm；最大绝对误差 {group['max_absolute_error_mm']:.3f} mm。"])
    (folder / '报告.md').write_text('\n'.join(lines), encoding='utf-8')
    return folder
