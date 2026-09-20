"""Reconstruct an explicitly labelled, verified prefix of an interrupted W0 session.

This is not tracking recovery. It never resumes acquisition or writes a business
latest marker. A new result and all its derived evidence live under --output.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np

from femto.io_utils import load_json, load_jsonl, write_json_strict, write_jsonl
from femto.dataset.local_reference import calibration_identity, reference_provenance
from femto.tracking.moving_quality import POLICY, rotation_deg
from femto.dataset.scene_dataset_validator import validate_pose


def assess(source):
    source = Path(source).resolve()
    report = load_json(source/'derived/local_capture_report.json')
    records = load_jsonl(source/'derived/live_tracking.jsonl')
    raw = load_jsonl(source/'frames.jsonl')
    manifest = load_json(source/'manifest.json')
    reasons = []
    if report.get('state') != 'lost' or not report.get('termination'):
        reasons.append('not_a_recorded_tracking_interruption')
    if manifest.get('parameters', {}).get('fixed_camera') is not False:
        reasons.append('not_a_local_moving_capture')
    cutoff = (report.get('termination') or {}).get('last_valid_frame_id')
    accepted_ids = report.get('accepted_frames', [])
    if type(cutoff) is not int or not accepted_ids:
        reasons.append('no_verified_prefix')
    if len(records) != len(raw) or len({r['frame_id'] for r in records}) != len(records):
        raise ValueError('Tracking / raw frame index mismatch')
    indexed = {r['frame_id']: r for r in records}
    raw_indexed = {r['frame']: r for r in raw}
    if (not isinstance(accepted_ids, list) or any(type(i) is not int for i in accepted_ids)
            or len(set(accepted_ids)) != len(accepted_ids) or accepted_ids != sorted(accepted_ids)):
        raise ValueError('Invalid recorded accepted frame list')
    selected = []
    if type(cutoff) is int and indexed.get(cutoff, {}).get('state') != 'tracking':
        raise ValueError('Last trustworthy cutoff is not a trusted frame')
    calibration = calibration_identity(source)
    for index in accepted_ids:
        record, row = indexed.get(index), raw_indexed.get(index)
        if record is None or row is None or record.get('state') != 'tracking' or record.get('selected_for_fusion') is not True:
            raise ValueError('Prefix contains a frame without recorded admission')
        if type(cutoff) is not int or index > cutoff:
            raise ValueError('Accepted frame is after the last trustworthy time')
        if (record.get('session_id') != report.get('session_id') or record.get('reference_frame_id') != report.get('reference_frame_id')
                or record.get('unit') != 'metre' or validate_pose(record.get('T_WC'))):
            raise ValueError('Prefix frame belongs to a different or invalid W0')
        for raw_key, track_key in (('camera_timestamp_us', 'capture_timestamp_us'),
                                   ('color_timestamp_us', 'color_timestamp_us'), ('depth_timestamp_us', 'depth_timestamp_us')):
            if row.get(raw_key) != record.get(track_key):
                raise ValueError('Prefix RGB-D timestamp / pose mismatch')
        if abs(row['color_timestamp_us']-row['depth_timestamp_us']) > 5000:
            raise ValueError('Unsynchronized prefix frame')
        pose = np.asarray(record['T_WC'], float)
        radius, angle = report.get('local_workspace_radius_m'), report.get('local_workspace_rotation_deg')
        # Use the policy recorded by this acquisition, never a newer policy.
        if radius is not None and np.linalg.norm(pose[:3, 3]) > float(radius)+1e-12:
            raise ValueError('Prefix violates its recorded capture radius')
        if angle is not None and rotation_deg(pose) > float(angle)+1e-9:
            raise ValueError('Prefix violates its recorded total rotation limit')
        selected.append(record)
    if selected:
        proof = reference_provenance(report['session_id'], selected[0], calibration)
        if proof != report.get('reference_provenance') or proof != load_json(source/'derived/local_reference.json'):
            raise ValueError('Prefix initial origin provenance mismatch')
        first_valid = next((r for r in records if r.get('state') == 'tracking'), None)
        if not first_valid or selected[0]['frame_id'] != first_valid['frame_id']:
            raise ValueError('Prefix does not start at the first valid color camera')
    unique = []
    policy = report.get('selection_policy') or {}
    distinct_translation = policy.get('small_motion_translation_m', POLICY['keyframe_translation_m'])
    distinct_rotation = policy.get('small_motion_rotation_deg', POLICY['keyframe_rotation_deg'])
    for record in selected:
        pose = np.asarray(record['T_WC'], float)
        if all(np.linalg.norm((np.linalg.inv(p)@pose)[:3, 3]) >= distinct_translation
               or rotation_deg(np.linalg.inv(p)@pose) >= distinct_rotation for p in unique):
            unique.append(pose)
    if len(unique) < 3:
        reasons.append('need_three_distinct_verified_views')
    return dict(schema='femto-interrupted-prefix-v1', eligible=not reasons, reasons=reasons,
        source_dataset=str(source), source_session_id=report.get('session_id'),
        reference_frame_id=report.get('reference_frame_id'), last_trusted_frame_id=cutoff,
        accepted_frames=accepted_ids, distinct_views=len(unique),
        capture_scope='interrupted_prefix', capture_label='中断前数据建模 / Interrupted-prefix reconstruction',
        recorded_policy={k: report.get(k) for k in ('local_workspace_radius_m', 'local_workspace_rotation_deg', 'max_gap_s')},
        physical_accuracy_verified=False, tracking_resumed=False, business_latest_updated=False), selected, raw_indexed, manifest, report


def prepare(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents:
        raise ValueError('Prefix derivation must be outside the original session')
    decision, selected, raw, manifest, tracking = assess(source)
    output.mkdir(parents=True, exist_ok=False)
    write_json_strict(output/'interrupted_prefix_report.json', decision)
    if not decision['eligible']:
        return None, decision
    files = [source/'manifest.json', source/'frames.jsonl', source/'derived/live_tracking.jsonl',
             source/'derived/local_capture_report.json', source/'derived/local_reference.json',
             *sorted((source/'calibration').glob('*.json'))]
    files.extend(source/raw[r['frame_id']][k] for r in selected for k in ('rgb', 'depth', 'pose'))
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    dataset = output/'dataset'
    dataset.mkdir(); (dataset/'pose').mkdir()
    shutil.copytree(source/'calibration', dataset/'calibration')
    derived = []
    for ordinal, record in enumerate(selected):
        row = dict(raw[record['frame_id']])
        for key in ('rgb', 'depth'):
            row[key] = os.path.relpath(source/row[key], dataset).replace('\\', '/')
        row.update(pose=f'pose/{ordinal:06d}.json', pose_valid=True)
        write_json_strict(dataset/row['pose'], dict(schema='femto-scene-frame-pose-v1', frame=row['frame'],
            pose_valid=True, T_WC=record['T_WC'], pose_source='online_rgbd_research_grade',
            coordinate_system='+X right, +Y down, +Z forward; metres', session_id=tracking['session_id'],
            reference_frame_id=tracking['reference_frame_id'], capture_timestamp_us=row['camera_timestamp_us'],
            reference_provenance=tracking['reference_provenance']))
        derived.append(row)
    write_jsonl(dataset/'frames.jsonl', derived, compact=False)
    manifest = copy.deepcopy(manifest)
    manifest.update(status='derived_interrupted_prefix', source_dataset=str(source), frame_count=len(derived),
        pose=dict(valid=True, source='online_rgbd_research_grade'),
        local_capture=dict(session_id=tracking['session_id'], reference_frame_id=tracking['reference_frame_id'],
            reference_provenance=tracking['reference_provenance'], measured_only=True,
            tracking_report=str(source/'derived/local_capture_report.json'),
            **decision['recorded_policy'], capture_scope='interrupted_prefix', interrupted_prefix=decision))
    write_json_strict(dataset/'manifest.json', manifest)
    from femto.tracking.moving_quality import pose_diversity
    span, angle = pose_diversity([np.asarray(r['T_WC']) for r in selected])
    write_json_strict(dataset/'pose_quality_report.json', dict(status='estimated_research_grade_interrupted_prefix',
        source_dataset=str(source), source_frame_count=len(raw), frame_count=len(derived),
        accepted_frame_ordinals=decision['accepted_frames'], translation_span_m=span, rotation_span_deg=angle,
        capture_scope='interrupted_prefix', frame_quality=[dict(frame_ordinal=r['frame_id'], **r.get('quality', {})) for r in selected],
        rejected_frames=[], pairs=[], limitations=['Only the recorded admitted prefix in the original W0; no resumed tracking or accuracy certificate.']))
    decision['source_sha256'] = hashes
    decision['source_unchanged'] = all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in hashes.items())
    write_json_strict(output/'interrupted_prefix_report.json', decision)
    if not decision['source_unchanged']:
        raise ValueError('Source changed during interrupted prefix preparation')
    return dataset, decision


def reconstruct(source, output, *, cancelled=lambda: False, progress=None):
    output = Path(output).resolve()
    dataset, decision = prepare(source, output)
    if dataset is None:
        raise ValueError('; '.join(decision['reasons']))
    from femto.pipeline.local_offline import LocalOfflineJob
    job = LocalOfflineJob(dataset, output/'processing', output/'models', decision['source_session_id'], cancelled, progress)
    try:
        result, summary = job.run()
        if cancelled():
            raise RuntimeError('Interrupted-prefix reconstruction cancelled')
        receipt = job._owned_result(result)
        write_json_strict(result/'publication.json', dict(receipt, status='derived_complete',
                   capture_scope='interrupted_prefix', published_markers=[], business_latest_updated=False))
        decision.update(status='derived_complete', result=str(result), worker_resources=job.stats)
        write_json_strict(output/'interrupted_prefix_report.json', decision)
        return result, decision
    except Exception as error:
        decision.update(status='failed', error=str(error))
        write_json_strict(output/'interrupted_prefix_report.json', decision)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    value, report = prepare(args.source, args.output) if args.prepare_only else reconstruct(args.source, args.output)
    printed = dict(report, result=str(value) if value else None)
    printed.pop('source_sha256', None)
    print(json.dumps(printed, ensure_ascii=False, indent=2))
