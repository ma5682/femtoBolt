"""Read-only reference proof for completed sequential RGB-D reconstructions.

This adapter verifies the existing pose chain; it never estimates a pose,
rebases geometry, or invents local-capture metadata.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from femto.dataset.local_reference import calibration_identity, CALIBRATION_FILES
from femto.dataset.scene_dataset_validator import validate_pose

POSE_SOURCE = 'open3d_sequential_hybrid_rgbd_odometry'
PROVENANCE = 'sequential_rgbd_odometry_reference_pose'


def sequential_reference(source, *, check=lambda: None):
    """Return a verified reference, or None for a different source type.

Malformed evidence for this source type raises ValueError. Existing local
sources and standalone annotation origins keep their original behavior.
"""
    source = Path(source).resolve()
    root = source.parent
    if (root/'local_geometry.json').exists():
        return None
    evidence = {}

    def read_bytes(path):
        check()
        path = Path(path).resolve()
        value = path.read_bytes()
        evidence[str(path)] = hashlib.sha256(value).hexdigest()
        return value

    def read(path):
        value = json.loads(read_bytes(path).decode('utf-8-sig'))
        require(isinstance(value, dict), '来源记录格式无效')
        return value

    def require(condition, reason):
        if not condition:
            raise ValueError('通用场景参考来源无法核验：'+reason)

    def asset(base, name):
        path = (base/name).resolve()
        require(path.is_relative_to(base) and path.is_file(), '资产越界或缺失')
        return path

    def rows(path):
        data = [json.loads(line) for line in read_bytes(path).decode('utf-8-sig').splitlines() if line.strip()]
        require(0 < len(data) <= 10000 and all(isinstance(row, dict) for row in data), '帧列表无效')
        return data

    report = read(root/'reconstruction_report.json')
    dataset = Path(report['source_dataset']).resolve()
    manifest = read(dataset/'manifest.json')
    if manifest.get('pose', {}).get('source') != POSE_SOURCE:
        return None
    raw = Path(manifest['source_dataset']).resolve()
    original = read(raw/'manifest.json')
    summary = read(root.parent/'demo_summary.json')
    tracking_path = dataset/'pose_quality_report.json'
    tracking = read(tracking_path)
    validation = read(dataset/'quality/validation_report.json')
    require(dataset.is_relative_to(raw/'derived'), '派生数据集不属于声明的原始采集')
    require(manifest.get('schema') == original.get('schema') == 'femto-full-scene-rgbd-v1'
            and manifest.get('status') == 'derived_pose_estimated'
            and manifest.get('pose', {}).get('valid') is True
            and original.get('status') == 'captured_pose_pending'
            and original.get('parameters', {}).get('fixed_camera') is False,
            '采集类型或位姿状态不符')
    require(report.get('schema') == 'femto-scene-reconstruction-v1'
            and report.get('status') == 'complete_geometry_gaussian_init_untrained'
            and Path(report['outputs']['mesh']['ply']).resolve() == source,
            '模型与重建报告不符')
    require(summary.get('schema') == 'femto-complete-demo-v1'
            and summary.get('status') == 'pass_geometry_gaussian_init_untrained'
            and summary.get('data_kind') == 'real_femto'
            and summary.get('pose_stage') == POSE_SOURCE
            and summary.get('fixed_camera') is False
            and Path(summary['raw_dataset']).resolve() == raw
            and Path(summary['source_dataset']).resolve() == dataset
            and Path(summary['reconstruction']).resolve() == root,
            '结果摘要与采集来源不符')
    require(tracking.get('schema') == 'femto-rgbd-odometry-report-v1'
            and tracking.get('status') == 'estimated_research_grade'
            and Path(tracking['source_dataset']).resolve() == raw
            and Path(tracking['output_dataset']).resolve() == dataset,
            '位姿报告未通过或来源不符')
    require(validation.get('schema') == 'femto-scene-validation-v1'
            and Path(validation['dataset']).resolve() == dataset
            and validation.get('ready_for_multiview_reconstruction') is True
            and validation.get('errors') == [], '数据质量校验未通过')
    raw_rows = rows(raw/'frames.jsonl')
    selected_rows = rows(dataset/'frames.jsonl')
    count = len(selected_rows)
    require(count >= 3 and original.get('frame_count') == len(raw_rows)
            and tracking.get('source_frame_count') == len(raw_rows)
            and all(value == count for value in (manifest.get('frame_count'), report.get('frame_count'),
                summary.get('frame_count'), tracking.get('frame_count'),
                validation.get('metrics', {}).get('valid_pose_count'))), '帧数不一致')
    ids = [row.get('frame') for row in raw_rows]
    require(all(type(i) is int and i >= 0 for i in ids) and len(set(ids)) == len(ids), '原始帧编号无效')
    selection = tracking['keyframe_selection']
    tracked = selection['tracked_frame_ordinals']
    selected = selection['selected_frame_ordinals']
    for indices in (tracked, selected):
        require(isinstance(indices, list) and len(indices) >= 3
                and all(type(i) is int and 0 <= i < len(raw_rows) for i in indices)
                and indices == sorted(set(indices)), '跟踪或关键帧编号无效')
    require(selected == tracking['accepted_frame_ordinals'] and len(selected) == count
            and set(selected).issubset(tracked) and selected[0] == tracked[0], '参考帧或关键帧链不符')
    qualities = {item['frame_ordinal']: item for item in tracking['frame_quality']}
    require(all(qualities.get(i, {}).get('usable') is True for i in tracked), '帧质量不可用')
    parameters = tracking['parameters']
    pairs = tracking['pairs']
    require(len(pairs) == len(tracked)-1, '位姿链缺项')
    poses = {tracked[0]: np.eye(4)}
    for before, after, pair in zip(tracked, tracked[1:], pairs):
        check()
        matrix = np.asarray(pair['source_to_target'], dtype=float)
        require(pair['source_frame'] == ids[before] and pair['target_frame'] == ids[after]
                and validate_pose(matrix) is None
                and pair.get('cycle_consistency', {}).get('passed') is True,
                '位姿链断开或双向校验失败')
        require(all(np.isfinite(pair[key]) for key in ('fitness', 'reverse_fitness', 'inlier_rmse_m'))
                and pair['fitness'] >= parameters['min_fitness']
                and pair['reverse_fitness'] >= parameters['min_fitness']
                and 0 <= pair['inlier_rmse_m'] <= parameters['max_rmse_m'], '配准质量未通过')
        poses[after] = poses[before] @ np.linalg.inv(matrix)
    raw_calibration = calibration_identity(raw)
    require(calibration_identity(dataset) == raw_calibration, '标定与原始采集不一致')
    for base in (raw, dataset):
        for name in CALIBRATION_FILES:
            read_bytes(base/'calibration'/name)
    intrinsic = read(dataset/'calibration/color_intrinsic.json')
    require(all(np.isfinite(intrinsic[key]) and intrinsic[key] > 0 for key in ('width','height','fx','fy')),
            '标定参数无效')
    for ordinal, frame in zip(selected, selected_rows):
        original_frame = raw_rows[ordinal]
        # The producer copies all capture metadata; only asset paths and pose status change.
        fields = set(original_frame) - {'rgb', 'depth', 'pose', 'pose_valid'}
        require(all(frame.get(key) == original_frame[key] for key in fields)
                and frame.get('pose_valid') is True, '参考影像与原始帧错配')
        for key in ('rgb', 'depth'):
            original_path = asset(raw, original_frame[key])
            require((dataset/frame[key]).resolve() == original_path, '图像不属于声明的原始帧')
        pose_path = asset(dataset, frame['pose'])
        pose = read(pose_path)
        matrix = np.asarray(pose['T_WC'], dtype=float)
        require(pose.get('schema') == 'femto-scene-frame-pose-v1'
                and type(pose.get('frame')) is int and pose['frame'] == ids[ordinal]
                and pose.get('pose_valid') is True and pose.get('pose_source') == POSE_SOURCE
                and pose.get('coordinate_system') == '+X right, +Y down, +Z forward; metres'
                and Path(pose['source_dataset']).resolve() == raw
                and validate_pose(matrix) is None
                and np.allclose(matrix, poses[ordinal], rtol=0, atol=1e-8), '保存位姿与报告链不一致')
    first = selected_rows[0]
    pose_path = (dataset/first['pose']).resolve()
    require(np.allclose(poses[selected[0]], np.eye(4), rtol=0, atol=1e-9), '初始参考不是该位姿链的原点')
    for key in ('rgb', 'depth'):
        read_bytes((dataset/first[key]).resolve())
    # Stable source identifier, not a patient ID or a fabricated local capture session.
    identity = '\n'.join((evidence[str(raw/'manifest.json')], evidence[str(raw/'frames.jsonl')], raw_calibration['sha256']))
    session = 'rgbd-reference-'+hashlib.sha256(identity.encode('ascii')).hexdigest()
    return dict(kind='initial_color_camera', provenance=PROVENANCE, session_id=session,
                reference_frame_id=first['frame'], coordinate_frame='world_rdf', unit='metre',
                transform_direction='T_W0_Ct', source_dataset=str(dataset), reference_pose=str(pose_path),
                reference_pose_sha256=evidence[str(pose_path)], calibration_sha256=raw_calibration['sha256'],
                evidence_sha256=evidence,
                undistorted_intrinsic=report['parameters']['distortion_remap']['new_K'])


def verify_reference_unchanged(reference, *, check=lambda: None):
    """Reject source changes between proof construction and model installation."""
    for name, expected in reference['evidence_sha256'].items():
        check()
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError('读取期间通用场景参考来源已变化，请重新打开')
