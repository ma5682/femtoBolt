"""Isolated A/B/C local-geometry replay. Never publishes a business latest index.

The public command is a supervisor. Every expensive stage runs in one bounded
child process; neither this module nor its worker reads a camera. Source data,
accepted frame identities and W0 are checked before and after the run.
"""
from __future__ import annotations
from typing import Any, Callable


import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import uuid

from femto.io_utils import load_json_object, load_jsonl, write_json_strict


from femto.paths import REPO_ROOT as ROOT
SCHEMA = 'femto-local-comparison-v1'
VARIANTS = ('A_online_raw', 'B_refined_raw', 'C_refined_filtered')


def read_rows(path: Any) -> Any:
    rows = load_jsonl(path)
    if any(not isinstance(r, dict) for r in rows):
        raise ValueError('Expected JSON row objects')
    return rows


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def fingerprints(paths: Any) -> dict[Any, Any]:
    return {str(Path(p).resolve()): sha256(p) if Path(p).is_file() else None
            for p in sorted({Path(p).resolve() for p in paths}, key=str)}


def changed_files(snapshot: Any) -> list[Any]:
    return [p for p, digest in snapshot.items()
            if (sha256(p) if Path(p).is_file() else None) != digest]


def _frame_id(row, key='frame'):
    value = row.get(key)
    if type(value) is not int or value < 0:
        raise ValueError('Invalid frame identity')
    return value


def _index(rows, key):
    result = {}
    for row in rows:
        number = _frame_id(row, key)
        if number in result:
            raise ValueError('Duplicate frame identity')
        result[number] = row
    return result


def _same_source_image(dataset, frame, raw, raw_frame, kind):
    selected = (dataset / frame[kind]).resolve()
    original = (raw / raw_frame[kind]).resolve()
    if not original.is_relative_to(raw) or not original.is_file():
        raise ValueError('Raw RGB-D path is outside its source session or missing')
    if not selected.is_file() or sha256(selected) != sha256(original):
        raise ValueError('Selected RGB-D differs from the recorded source frame')
    return selected


def _validate_selected_frames(dataset, frames, raw, tracked, raw_index, expected_pose_source,
                              session, reference, angle_limit, radius, source_paths):
    import numpy as np
    from femto.dataset.scene_dataset_validator import validate_pose
    from femto.tracking.moving_quality import rotation_deg

    poses = []
    for frame in frames:
        number = frame['frame']
        if number not in tracked or number not in raw_index:
            raise ValueError('Selected source frame not found')
        record, raw_frame = tracked[number], raw_index[number]
        pose_path = (dataset / frame['pose']).resolve()
        pose = load_json_object(pose_path)
        matrix = np.asarray(pose.get('T_WC'), dtype=float)
        if (frame.get('pose_valid') is not True or pose.get('pose_valid') is not True
                or validate_pose(matrix) or pose.get('pose_source') != expected_pose_source
                or pose.get('frame') != number or pose.get('session_id') != session
                or pose.get('reference_frame_id') != reference or record.get('session_id') != session
                or record.get('reference_frame_id') != reference or record.get('state') != 'tracking'
                or record.get('selected_for_fusion') is not True or record.get('quality', {}).get('usable') is not True
                or validate_pose(record.get('T_WC'))
                or not np.allclose(matrix, record['T_WC'], rtol=0, atol=1e-9)):
            raise ValueError('Selected pose / trusted online record mismatch')
        age = record.get('latency_ms')
        if not isinstance(age, (int, float)) or not math.isfinite(age) or not 0 <= age <= 2000:
            raise ValueError('Selected online pose was stale or has no timing evidence')
        if angle_limit is not None and rotation_deg(matrix)>angle_limit+1e-9:
            raise ValueError('Pose exceeds recorded workspace rotation')
        if radius is not None and np.linalg.norm(matrix[:3, 3]) > radius:
            raise ValueError('Selected pose is outside the recorded workspace')
        for key, track_key in [('camera_timestamp_us', 'capture_timestamp_us'),
                               ('color_timestamp_us', 'color_timestamp_us'), ('depth_timestamp_us', 'depth_timestamp_us')]:
            value = frame.get(key)
            if type(value) is not int or value < 0 or value != raw_frame.get(key) or value != record.get(track_key):
                raise ValueError('RGB-D timestamp provenance mismatch')
        if (abs(frame['color_timestamp_us'] - frame['depth_timestamp_us']) > 5000
                or pose.get('capture_timestamp_us') != frame['camera_timestamp_us']):
            raise ValueError('RGB-D synchronization / pose timestamp mismatch')
        source_paths.add(pose_path)
        source_paths.update(_same_source_image(dataset, frame, raw, raw_frame, k) for k in ('rgb', 'depth'))
        poses.append(pose)
    return poses


def validate_comparison_input(dataset: str | Path, *, max_views: int=20, allow_synthetic: bool=False) -> dict[str, Any]:
    """Validate an original online posed dataset, without writing validation to it."""
    import numpy as np
    from femto.dataset.scene_data_kind import classify_dataset_data_kind
    from femto.dataset.scene_dataset_validator import validate_dataset, validate_pose
    from femto.dataset.local_reference import calibration_identity, validate_reference_provenance

    dataset = Path(dataset).resolve()
    manifest = load_json_object(dataset / 'manifest.json')
    local = manifest.get('local_capture', {})
    if (manifest.get('schema') != 'femto-full-scene-rgbd-v1'
            or not local.get('measured_only') or manifest.get('temporal_fusion')):
        raise ValueError('An original local observed-depth posed dataset is required')
    kind = classify_dataset_data_kind(dataset)
    synthetic = kind == 'synthetic'
    if kind == 'unknown' or (synthetic and not allow_synthetic):
        raise ValueError('Verified Femto lineage required; synthetic replay needs --allow-synthetic')
    expected_pose_source = 'synthetic_ground_truth' if synthetic else 'online_rgbd_research_grade'
    if (manifest.get('pose', {}).get('source') != expected_pose_source
            or manifest.get('refinement_source_dataset') or local.get('refinement_report')
            or local.get('temporal_support_report')):
        raise ValueError('A must start with original online poses and original keyframe depth')
    frames = read_rows(dataset / 'frames.jsonl')
    _index(frames, 'frame')
    if not 3 <= len(frames) <= max_views or max_views > 20 or max_views < 3:
        raise ValueError('Comparison requires 3 to max_views accepted views, at most 20; no truncation')
    if manifest.get('frame_count') != len(frames):
        raise ValueError('Manifest frame count mismatch')
    if any(f.get('local_temporal_support') or f.get('temporal_fusion') or f.get('reference_raw_depth')
           for f in frames):
        raise ValueError('A cannot contain a temporal mask or derived replacement depth')
    raw = Path(manifest['source_dataset']).resolve()
    tracking_path = Path(local['tracking_report']).resolve()
    if tracking_path != raw / 'derived' / 'local_capture_report.json':
        raise ValueError('Tracking report belongs to another source session')
    tracking = load_json_object(tracking_path)
    records = read_rows(raw / 'derived' / 'live_tracking.jsonl')
    raw_frames = read_rows(raw / 'frames.jsonl')
    tracked = _index(records, 'frame_id')
    raw_index = _index(raw_frames, 'frame')
    ids = [_frame_id(f) for f in frames]
    session, reference = local.get('session_id'), local.get('reference_frame_id')
    if (not isinstance(session, str) or not session or type(reference) is not int
            or ids[0] != reference or tracking.get('schema') != 'femto-local-tracking-v1'
            or tracking.get('session_id') != session or tracking.get('reference_frame_id') != reference
            or tracking.get('accepted_frames') != ids or tracking.get('state') != 'stopped'
            or tracking.get('tracking_lost_events', 0) or tracking.get('abort_reason')
            or tracking.get('error')):
        raise ValueError('Session, accepted frames or stopped tracking state mismatch')
    for key, expected in [('coordinate_frame', 'world_rdf'), ('unit', 'metre'), ('transform_direction', 'T_W0_Ct')]:
        if tracking.get(key) != expected:
            raise ValueError('Tracking coordinate system mismatch')
    if tracking.get('pose_source') != expected_pose_source:
        raise ValueError('Tracking pose source mismatch')
    if calibration_identity(dataset) != calibration_identity(raw):
        raise ValueError('Raw / posed calibration mismatch')
    radius = local.get('local_workspace_radius_m')
    angle_limit=local.get('local_workspace_rotation_deg')
    if angle_limit is not None and (not isinstance(angle_limit,(int,float)) or not math.isfinite(angle_limit) or not 0<angle_limit<=180):
        raise ValueError('Invalid recorded workspace rotation')
    if tracking.get('local_workspace_rotation_deg')!=angle_limit:
        raise ValueError('Workspace rotation policy mismatch')
    if radius is not None and (not isinstance(radius, (int, float)) or not math.isfinite(radius) or radius <= 0):
        raise ValueError('Invalid recorded workspace radius')
    if tracking.get('local_workspace_radius_m') != radius:
        raise ValueError('Workspace policy mismatch')
    source_paths = {dataset / 'manifest.json', dataset / 'frames.jsonl', dataset / 'pose_quality_report.json',
                    raw / 'manifest.json', raw / 'frames.jsonl', tracking_path, raw / 'derived' / 'live_tracking.jsonl'}
    # Hash every captured raw RGB-D/Pose/calibration file, not only the selected views.
    for folder in ('rgb', 'depth', 'pose', 'calibration'):
        source_paths.update(p for p in (raw / folder).rglob('*') if p.is_file())
    source_paths.update(p for p in (dataset / 'calibration').rglob('*') if p.is_file())
    source_paths.update((dataset / f[key]).resolve() for f in frames for key in ('rgb', 'depth', 'pose'))
    if manifest.get('comparison_ground_truth'):
        source_paths.add(Path(manifest['comparison_ground_truth']['mesh']).resolve())
    hashes = fingerprints(source_paths)
    if any(value is None for value in hashes.values()):
        raise ValueError('Required evidence file is missing')
    poses = _validate_selected_frames(dataset, frames, raw, tracked, raw_index,
        expected_pose_source, session, reference, angle_limit, radius, source_paths)
    proof = validate_reference_provenance(local.get('reference_provenance'), dataset, frames[0], poses[0])
    if tracking.get('reference_provenance') != proof or any(p.get('reference_provenance') != proof for p in poses):
        raise ValueError('Initial W0 reference provenance mismatch')
    params = manifest.get('parameters', {})
    limits = (float(params.get('depth_min_mm', 250)), float(params.get('depth_max_mm', 2210)))
    if limits != (250., 2210.):
        raise ValueError('Local comparison requires the existing 250 to 2210 mm range')
    validation, code = validate_dataset(argparse.Namespace(dataset=dataset, datasets_root=raw.parent,
        require_pose=True, max_sync_delta_ms=5., min_valid_depth_ratio=.01,
        min_depth_mm=limits[0], max_depth_mm=limits[1], no_write_report=True))
    if code or not validation.get('ready_for_multiview_reconstruction'):
        raise ValueError('Read-only dataset validation rejected: ' + str(validation.get('errors')))
    quality = load_json_object(dataset / 'pose_quality_report.json')
    if quality.get('accepted_frame_ordinals') != ids or quality.get('source_frame_count') != len(records):
        raise ValueError('Online quality report source frame mismatch')
    truth = manifest.get('comparison_ground_truth')
    if truth:
        truth_path = Path(truth['mesh']).resolve()
        if (not synthetic or truth.get('coordinate_frame') != 'world_rdf' or truth.get('unit') != 'metre'
                or truth.get('reference_provenance') != proof or sha256(truth_path) != truth.get('sha256')):
            raise ValueError('Synthetic ground-truth frame / source mismatch')
        source_paths.add(truth_path)
    if changed_files(hashes):
        raise ValueError('Source evidence changed during read-only validation')
    return dict(dataset=str(dataset), raw_source=str(raw), data_kind=kind, session_id=session,
        reference_frame_id=reference, reference_provenance=proof, accepted_frame_ids=ids,
        source_frame_count=len(raw_frames), processed_tracking_frame_count=len(records), accepted_frame_count=len(frames),
        rejected_tracking=quality.get('rejected_frames', []), workspace_radius_m=radius,
        workspace_rotation_deg=angle_limit,
        workspace_policy='recorded' if radius is not None else 'historical_policy_not_recorded',
        pose_source=expected_pose_source, source_sha256=hashes, validation=validation,
        synthetic_ground_truth=truth, physical_accuracy_verified=False,
        implementation_sha256=fingerprints(ROOT/name for name in (
            'femto/pipeline/local_comparison.py', 'femto/pipeline/local_comparison_viewer.py',
            'femto/pipeline/scene_reconstruction.py', 'femto/geometry/rgbd_frames.py',
            'femto/local/local_pose_refinement.py', 'femto/local/local_temporal_support.py',
            'femto/local/local_geometry.py', 'femto/dataset/local_reference.py',
            'femto/dataset/scene_dataset_validator.py', 'femto/annotation/fixed_rgb_geometry.py',
            'femto/tracking/local_stability.py', 'femto/io_utils.py')))


def validate_variant_input(baseline: Any, dataset: str | Path, *, allow_masks: bool=False) -> dict[str, Any]:
    """Forbid frame substitution, changed W0, or a mask reused on another RGB-D image."""
    import numpy as np
    from femto.dataset.local_reference import calibration_identity, validate_reference_provenance
    from femto.dataset.scene_dataset_validator import validate_pose
    from femto.io_utils import imread_unicode
    from femto.local.local_temporal_support import filtered_depth, stationary_window
    import cv2

    source, dataset = Path(baseline['dataset']), Path(dataset).resolve()
    manifest, original = load_json_object(dataset / 'manifest.json'), load_json_object(source / 'manifest.json')
    local = manifest.get('local_capture', {})
    if (local.get('session_id') != baseline['session_id']
            or local.get('reference_frame_id') != baseline['reference_frame_id']
            or local.get('reference_provenance') != baseline['reference_provenance']
            or local.get('local_workspace_radius_m') != baseline['workspace_radius_m']
            or local.get('local_workspace_rotation_deg') != baseline.get('workspace_rotation_deg')
            or not local.get('measured_only') or manifest.get('temporal_fusion')
            or calibration_identity(dataset) != calibration_identity(source)):
        raise ValueError('Variant source / session / calibration / W0 mismatch')
    frames, originals = read_rows(dataset / 'frames.jsonl'), read_rows(source / 'frames.jsonl')
    if [r.get('frame') for r in frames] != baseline['accepted_frame_ids']:
        raise ValueError('Variant changed accepted frames or their order')
    temporal = None
    tracking_records = None
    if any(f.get('local_temporal_support') for f in frames):
        if not allow_masks:
            raise ValueError('Unexpected temporal mask in raw-depth version')
        temporal = load_json_object(local['temporal_support_report'])
        if temporal.get('source_unchanged') is not True or changed_files(temporal.get('source_sha256', {})):
            raise ValueError('Temporal source evidence changed')
        if [r.get('frame') for r in temporal.get('frames', [])] != baseline['accepted_frame_ids']:
            raise ValueError('Temporal report frame mismatch')
        tracking_records = read_rows(Path(baseline['raw_source']) / 'derived/live_tracking.jsonl')
    rows = []
    for frame, reference_frame in zip(frames, originals):
        for key in ('rgb', 'depth'):
            if sha256(dataset / frame[key]) != sha256(source / reference_frame[key]):
                raise ValueError('Variant replaced a source RGB-D image')
        for key in ('camera_timestamp_us', 'color_timestamp_us', 'depth_timestamp_us'):
            if frame.get(key) != reference_frame.get(key):
                raise ValueError('Variant timestamp mismatch')
        pose = load_json_object(dataset / frame['pose'])
        if (validate_pose(pose.get('T_WC')) or pose.get('pose_valid') is not True
                or pose.get('frame') != frame['frame'] or pose.get('session_id') != baseline['session_id']
                or pose.get('reference_frame_id') != baseline['reference_frame_id']
                or pose.get('reference_provenance') != baseline['reference_provenance']):
            raise ValueError('Variant invalid pose / reference binding')
        radius = baseline['workspace_radius_m']
        from femto.tracking.moving_quality import rotation_deg
        angle_limit=baseline.get('workspace_rotation_deg')
        if angle_limit is not None and rotation_deg(np.asarray(pose['T_WC']))>angle_limit+1e-9:
            raise ValueError('Variant exceeds recorded workspace rotation')
        if radius is not None and np.linalg.norm(np.asarray(pose['T_WC'])[:3, 3]) > radius:
            raise ValueError('Variant pose outside original workspace')
        spec = frame.get('local_temporal_support')
        if spec:
            row = next(r for r in temporal['frames'] if r['frame'] == frame['frame'])
            if (row.get('applied') is not True or row.get('source_frames') != spec.get('source_frames')
                    or len(spec.get('source_frames', [])) != 5 or frame['frame'] not in spec['source_frames']):
                raise ValueError('Temporal mask frame provenance mismatch')
            expected_mask = Path(local['temporal_support_report']).resolve().parent / f"support_{frame['frame']:06}.png"
            if (dataset / spec['mask']).resolve() != expected_mask:
                raise ValueError('Temporal mask belongs to another source frame')
            window = stationary_window(tracking_records, frame['frame'])
            if (window is None or [r['frame_id'] for r in window[0]] != spec['source_frames']
                    or any(r.get('session_id') != baseline['session_id'] for r in window[0])):
                raise ValueError('Temporal mask stationary-window source mismatch')
            for key in ('rgb', 'depth'):
                path = str((dataset / frame[key]).resolve())
                if temporal['source_sha256'].get(path) != sha256(path):
                    raise ValueError('Temporal mask source image is unbound')
            depth = imread_unicode(dataset / frame['depth'], cv2.IMREAD_UNCHANGED)
            filtered_depth(dataset, frame, depth)  # hash, shape, dtype and binary-value gate
        rows.append(pose)
    validate_reference_provenance(baseline['reference_provenance'], dataset, frames[0], rows[0])
    return dict(actual_dataset=str(dataset), actual_pose_source=manifest.get('pose', {}).get('source'),
                frames=baseline['accepted_frame_ids'], masks_applied=sum(bool(f.get('local_temporal_support')) for f in frames))


def _replace_root(value, old, new):
    if isinstance(value, dict):
        return {k: _replace_root(v, old, new) for k, v in value.items()}
    if isinstance(value, list):
        return [_replace_root(v, old, new) for v in value]
    if isinstance(value, str):
        for source, target in ((str(old), str(new)), (old.as_posix(), new.as_posix())):
            if value == source or value.startswith(source + os.sep) or value.startswith(source + '/'):
                return target + value[len(source):]
    return value


def copy_unchanged_model(source: str | Path, output: str | Path) -> Any:
    """Copy this run's geometry byte-for-byte, and rebind report paths to its new version."""
    source, output = Path(source).resolve(), Path(output).resolve()
    shutil.copytree(source, output)
    for path in output.rglob('*.json'):
        value = load_json_object(path)
        write_json_strict(path, _replace_root(value, source, output))
    for path in source.rglob('*'):
        if path.is_file() and path.suffix.lower() in ('.ply', '.obj', '.glb', '.png', '.jpg'):
            if sha256(path) != sha256(output / path.relative_to(source)):
                raise ValueError('Unchanged model copy differs from previous version')
    return load_json_object(output / 'reconstruction_report.json')


def _worker(spec):
    stage, dataset, output = spec['stage'], Path(spec['dataset']), Path(spec['output'])
    if stage == 'validate':
        return validate_comparison_input(dataset, max_views=spec['max_views'], allow_synthetic=spec['allow_synthetic'])
    baseline = load_json_object(spec['baseline'])
    if changed_files(baseline['source_sha256']):
        raise ValueError('Original evidence changed before stage')
    if changed_files(baseline.get('implementation_sha256', {})):
        raise ValueError('Comparison implementation changed between stages; start a new independent run')
    if stage == 'refine':
        from femto.local.local_pose_refinement import refine
        result, report = refine(dataset, output)
        if (report.get('accepted') is True) == (Path(result).resolve() == dataset.resolve()):
            raise ValueError('Refinement adoption does not match the actual input')
        validate_variant_input(baseline, result)
        return dict(actual_dataset=str(result), accepted=report.get('accepted') is True, report=report,
                    report_path=str(output / 'refinement_report.json'))
    if stage == 'temporal':
        from femto.local.local_temporal_support import prepare
        result, report = prepare(dataset, output)
        if bool(report.get('applied_frames')) == (Path(result).resolve() == dataset.resolve()):
            raise ValueError('Temporal adoption does not match actual input')
        validate_variant_input(baseline, result, allow_masks=True)
        return dict(actual_dataset=str(result), applied_frames=report['applied_frames'], report=report,
                    report_path=str(output / 'temporal_support_report.json'))
    if stage == 'reconstruct':
        from femto.pipeline.scene_reconstruction import reconstruct_scene
        binding = validate_variant_input(baseline, dataset, allow_masks=spec['allow_masks'])
        if spec.get('same_reconstruction'):
            report = copy_unchanged_model(spec['same_reconstruction'], output)
        else:
            report = reconstruct_scene(dataset, output, image_scale=1., point_stride=1,
                voxel_m=spec['voxel_m'], sdf_trunc_m=spec['sdf_trunc_m'], depth_min_m=.25, depth_max_m=2.21)
        from femto.pipeline.local_comparison_viewer import evaluate_and_render
        metrics = evaluate_and_render(baseline, dataset, output, width=spec['render_width'])
        return dict(**binding, reconstruction=str(output), reconstruction_metrics=report['metrics'],
            metrics=metrics, mesh_sha256=sha256(output / 'scene_tsdf_mesh_m.ply'),
            initial_cloud_sha256=sha256(output / 'initial_observed_points_m.ply'))
    raise ValueError('Unknown comparison worker stage')


class StageStopped(RuntimeError):
    def __init__(self, status, reason, stats):
        super().__init__(reason)
        self.status, self.stats = status, stats


class _WindowsJob:
    """Kernel-enforced budget for the venv launcher AND its interpreter descendants."""
    def __init__(self, memory_mb):
        import ctypes
        from ctypes import wintypes
        self.ctypes, self.wintypes = ctypes, wintypes
        self.kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        class Basic(ctypes.Structure):
            _fields_ = [('PerProcessUserTimeLimit', ctypes.c_int64), ('PerJobUserTimeLimit', ctypes.c_int64),
                ('LimitFlags', wintypes.DWORD), ('MinimumWorkingSetSize', ctypes.c_size_t),
                ('MaximumWorkingSetSize', ctypes.c_size_t), ('ActiveProcessLimit', wintypes.DWORD),
                ('Affinity', ctypes.c_size_t), ('PriorityClass', wintypes.DWORD), ('SchedulingClass', wintypes.DWORD)]
        class IO(ctypes.Structure):
            _fields_ = [(key, ctypes.c_uint64) for key in ('ReadOperationCount', 'WriteOperationCount',
                'OtherOperationCount', 'ReadTransferCount', 'WriteTransferCount', 'OtherTransferCount')]
        class Extended(ctypes.Structure):
            _fields_ = [('BasicLimitInformation', Basic), ('IoInfo', IO),
                ('ProcessMemoryLimit', ctypes.c_size_t), ('JobMemoryLimit', ctypes.c_size_t),
                ('PeakProcessMemoryUsed', ctypes.c_size_t), ('PeakJobMemoryUsed', ctypes.c_size_t)]
        self.Extended = Extended
        self.kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self.kernel.CreateJobObjectW.restype = wintypes.HANDLE
        self.kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        self.kernel.SetInformationJobObject.restype = wintypes.BOOL
        self.kernel.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p,
                                                         wintypes.DWORD, ctypes.c_void_p]
        self.kernel.QueryInformationJobObject.restype = wintypes.BOOL
        self.kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self.kernel.AssignProcessToJobObject.restype = wintypes.BOOL
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel.CloseHandle.restype = wintypes.BOOL
        self.handle = self.kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = Extended()
        limits.BasicLimitInformation.LimitFlags = 0x2000 | 0x200  # KILL_ON_JOB_CLOSE | JOB_MEMORY
        limits.JobMemoryLimit = int(memory_mb * 1024**2)
        if not self.kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            self.close()
            raise ctypes.WinError(ctypes.get_last_error())

    def attach_and_resume(self, process: Any) -> None:
        ctypes, wintypes, kernel = self.ctypes, self.wintypes, self.kernel
        if not kernel.AssignProcessToJobObject(self.handle, wintypes.HANDLE(int(process._handle))):
            raise ctypes.WinError(ctypes.get_last_error())
        # Popen closes the initial thread handle. Enumerate its still-suspended
        # initial thread via documented Toolhelp APIs, then resume only our PID.
        class ThreadEntry(ctypes.Structure):
            _fields_ = [('dwSize', wintypes.DWORD), ('cntUsage', wintypes.DWORD),
                ('th32ThreadID', wintypes.DWORD), ('th32OwnerProcessID', wintypes.DWORD),
                ('tpBasePri', wintypes.LONG), ('tpDeltaPri', wintypes.LONG), ('dwFlags', wintypes.DWORD)]
        kernel.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
        kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
        kernel.Thread32First.argtypes = [wintypes.HANDLE, ctypes.POINTER(ThreadEntry)]
        kernel.Thread32First.restype = wintypes.BOOL
        kernel.Thread32Next.argtypes = [wintypes.HANDLE, ctypes.POINTER(ThreadEntry)]
        kernel.Thread32Next.restype = wintypes.BOOL
        kernel.OpenThread.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenThread.restype = wintypes.HANDLE
        kernel.ResumeThread.argtypes = [wintypes.HANDLE]
        kernel.ResumeThread.restype = wintypes.DWORD
        snapshot = kernel.CreateToolhelp32Snapshot(4, 0)  # TH32CS_SNAPTHREAD
        if snapshot == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            entry = ThreadEntry()
            entry.dwSize = ctypes.sizeof(entry)
            found = kernel.Thread32First(snapshot, ctypes.byref(entry))
            while found:
                if entry.th32OwnerProcessID == process.pid:
                    thread = kernel.OpenThread(2, False, entry.th32ThreadID)  # THREAD_SUSPEND_RESUME
                    if not thread:
                        raise ctypes.WinError(ctypes.get_last_error())
                    try:
                        if kernel.ResumeThread(thread) == 0xffffffff:
                            raise ctypes.WinError(ctypes.get_last_error())
                    finally:
                        kernel.CloseHandle(thread)
                    return
                found = kernel.Thread32Next(snapshot, ctypes.byref(entry))
            raise RuntimeError('No initial worker thread found; refusing unbudgeted execution')
        finally:
            kernel.CloseHandle(snapshot)

    def peak_memory(self) -> int:
        result = self.Extended()
        if not self.kernel.QueryInformationJobObject(self.handle, 9, self.ctypes.byref(result),
                                                     self.ctypes.sizeof(result), None):
            raise self.ctypes.WinError(self.ctypes.get_last_error())
        return int(result.PeakJobMemoryUsed)

    def close(self) -> None:
        if self.handle:
            self.kernel.CloseHandle(self.handle)  # Terminates the complete job, including launcher descendants.
            self.handle = None


def _rss_bytes(process):
    """Process working set on Windows / resident pages on Linux, without new dependencies."""
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD),
                ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]
        count = Counters()
        count.cb = ctypes.sizeof(count)
        function = ctypes.WinDLL('psapi').GetProcessMemoryInfo
        function.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        function.restype = wintypes.BOOL
        if not function(wintypes.HANDLE(int(process._handle)), ctypes.byref(count), count.cb):
            if process.poll() is not None:
                return 0
            raise RuntimeError('Cannot enforce child-process memory budget')
        return int(count.WorkingSetSize)
    stat = Path('/proc') / str(process.pid) / 'statm'
    try:
        return int(stat.read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')
    except FileNotFoundError:
        return 0


def supervise(command: Any, log_path: str | Path, *, timeout_s: float, memory_mb: float, cancelled: Callable[..., Any]=lambda: False, progress: Any=None) -> Any:
    """Run only this task's process group; Windows uses a hard job memory limit."""
    started, peak, last_update = time.monotonic(), 0, 0.
    log_path = Path(log_path)
    flags = (subprocess.CREATE_NO_WINDOW | 0x4) if os.name == 'nt' else 0  # CREATE_SUSPENDED
    job = None
    def measurements(elapsed):
        return dict(elapsed_s=elapsed,
            peak_job_committed_mb=peak / 1024**2 if job else None,
            peak_rss_sampled_mb=None if job else peak / 1024**2,
            memory_limit_mb=memory_mb,
            memory_measurement=('Windows kernel job peak committed memory, including venv launcher descendants; hard job memory limit'
                if job else 'Child RSS sampled every 100 ms; brief peaks may be missed'))
    with log_path.open('xb') as log:
        process = None
        try:
            job = _WindowsJob(memory_mb) if os.name == 'nt' else None
            process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, creationflags=flags, start_new_session=os.name != 'nt')
            if job:
                job.attach_and_resume(process)
            while process.poll() is None:
                elapsed = time.monotonic() - started
                peak = max(peak, job.peak_memory() if job else _rss_bytes(process))
                stats = measurements(elapsed)
                if cancelled():
                    raise StageStopped('cancelled', 'Cancellation requested; late results are not adopted', stats)
                if elapsed > timeout_s:
                    raise StageStopped('failed', 'Stage wall-time budget exceeded', stats)
                if peak > memory_mb * 1024**2:
                    raise StageStopped('failed', 'Stage sampled-memory budget exceeded', stats)
                if progress and elapsed - last_update >= 2:
                    progress(stats)
                    last_update = elapsed
                time.sleep(.1)
            if job:
                peak = max(peak, job.peak_memory())
            stats = dict(measurements(time.monotonic()-started), returncode=process.returncode)
            if cancelled():
                raise StageStopped('cancelled', 'Cancellation requested before adoption of completed result', stats)
            if stats['elapsed_s'] > timeout_s:
                raise StageStopped('failed', 'Stage wall-time budget exceeded before result adoption', stats)
            if process.returncode:
                raise StageStopped('failed', f'Worker exited with code {process.returncode}; see {log_path}', stats)
            return stats
        finally:
            if job:
                job.close()
            if process is not None and process.poll() is None:
                if os.name == 'nt':
                    process.terminate()
                else:
                    import signal
                    os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)


def _business_indices(raw):
    result = [ROOT / 'datasets/latest_scene.txt', ROOT / 'outputs/latest_complete_demo.txt']
    for directory in (ROOT / 'datasets', ROOT / 'outputs', raw / 'derived'):
        result.extend(directory.glob('latest*.txt'))
    return result


def run_comparison(dataset: str | Path, output: str | Path, *, voxel_m: float=.005, sdf_trunc_m: float=.02, max_views: int=20,
                   stage_timeout_s: float=300., memory_mb: float=3072., render_width: int=640,
                   allow_synthetic: bool=False, cancel_file: str | Path | None=None, progress: Any=None, latest_paths: Any=None, cancelled: Any=None) -> dict[str, Any]:
    """Synchronous supervisor API; a GUI must invoke it off its Tk thread or as a CLI."""
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Comparison output must be a new directory')
    manifest = load_json_object(dataset / 'manifest.json')
    raw = Path(manifest.get('source_dataset', dataset)).resolve()
    if output == dataset or output.is_relative_to(dataset) or output == raw or output.is_relative_to(raw):
        raise ValueError('Comparison output cannot be inside original or posed data')
    if (not all(math.isfinite(x) for x in (voxel_m, sdf_trunc_m, stage_timeout_s, memory_mb))
            or not .001 <= voxel_m <= .02 or not voxel_m < sdf_trunc_m <= .1
            or not 3 <= max_views <= 20 or not 1 <= stage_timeout_s <= 3600
            or not 128 <= memory_mb <= 16384 or not 128 <= render_width <= 1280):
        raise ValueError('Invalid finite parameter / task budget')
    output.mkdir(parents=True, exist_ok=False)
    latest = fingerprints(latest_paths if latest_paths is not None else _business_indices(raw))
    cancel_file = Path(cancel_file) if cancel_file else output / 'CANCEL'
    def is_cancelled():
        return cancel_file.exists() or bool(cancelled and cancelled())
    report_path = output / 'comparison_report.json'
    baseline_path = output / 'source_evidence.json'
    started = time.monotonic()
    report = dict(schema=SCHEMA, job_id=uuid.uuid4().hex, created_local=time.strftime('%Y-%m-%d %H:%M:%S %z'),
        state='running', source_dataset=str(dataset), output=str(output), session_id=None,
        coordinate_frame='world_rdf', unit='metre', transform_direction='T_W0_Ct',
        parameters=dict(voxel_m=voxel_m, sdf_trunc_m=sdf_trunc_m, image_scale=1., point_stride=1,
                        depth_min_m=.25, depth_max_m=2.21),
        budgets=dict(max_views=max_views, stage_timeout_s=stage_timeout_s, memory_mb=memory_mb,
                     worker_count=1, log_storage='streamed to disk', render_width=render_width),
        stages=[], variants=[], cancel_file=str(cancel_file.resolve()), physical_accuracy_verified=False,
        latest_before=latest, limitations=[
            'Same raw RGB-D and accepted frame identities; initial camera W0 remains fixed.',
            'Internal geometry and depth consistency are separate from independent physical accuracy.',
            'Binary five-frame rejection is not mean/median fusion; no inferred depth is integrated.',
            'Failed/cancelled stages retain their partial diagnostics and do not publish a latest result.',
            'Noise, ghosting and ROI hole filling require a predefined region and independent evidence; not inferred from point count.'])

    def publish():
        write_json_strict(report_path, report)
        if progress:
            progress(copy.deepcopy(report))

    def stage(name, operation, input_dataset, stage_output, **extra):
        if is_cancelled():
            raise StageStopped('cancelled', 'Cancelled before next stage', {})
        row = dict(name=name, operation=operation, state='running', input_dataset=str(input_dataset))
        report['stages'].append(row)
        spec = dict(stage=operation, dataset=str(input_dataset), output=str(stage_output),
            baseline=str(baseline_path), max_views=max_views, allow_synthetic=allow_synthetic,
            voxel_m=voxel_m, sdf_trunc_m=sdf_trunc_m, render_width=render_width, **extra)
        spec_path = output / (name + '_request.json')
        result_path = output / (name + '_result.json')
        write_json_strict(spec_path, spec)
        publish()
        def update(stats):
            row.update(stats)
            publish()
        try:
            stats = supervise([sys.executable, '-X', 'utf8', '-m', 'femto.pipeline.local_comparison',
                '--worker', str(spec_path), '--worker-result', str(result_path)], output / (name + '.log'),
                timeout_s=stage_timeout_s, memory_mb=memory_mb, cancelled=is_cancelled, progress=update)
            value = load_json_object(result_path)
            row.update(state='complete', **stats)
            publish()
            return value, stats
        except StageStopped as error:
            row.update(state=error.status, error=str(error), **error.stats)
            publish()
            raise
        except Exception as error:
            row.update(state='failed', error=str(error))
            publish()
            raise

    publish()
    try:
        baseline, _ = stage('00_validate', 'validate', dataset, output)
        write_json_strict(baseline_path, baseline)
        report.update(session_id=baseline['session_id'], reference_frame_id=baseline['reference_frame_id'],
            data_kind=baseline['data_kind'], source_frame_count=baseline['source_frame_count'],
            accepted_frame_ids=baseline['accepted_frame_ids'], source_evidence=str(baseline_path))
        selected = dataset
        for index, name in enumerate(VARIANTS):
            variant_root = output / name
            variant_root.mkdir(exist_ok=False)
            same_as, optimization = None, dict(status='baseline', reason='Original online poses and original keyframe depth')
            if index == 1:
                value, _ = stage('10_refinement', 'refine', selected, output / 'derived/refinement')
                selected = Path(value['actual_dataset'])
                adopted = value['accepted']
                optimization = dict(status='adopted' if adopted else 'not_adopted', details=value['report'], report=value['report_path'])
                same_as = None if adopted else VARIANTS[0]
            elif index == 2:
                value, _ = stage('20_temporal', 'temporal', selected, output / 'derived/temporal')
                selected = Path(value['actual_dataset'])
                adopted = value['applied_frames'] > 0
                optimization = dict(status='adopted' if adopted else 'not_adopted', details=value['report'], report=value['report_path'])
                same_as = None if adopted else VARIANTS[1]
            value, stats = stage(f'{index}1_reconstruct', 'reconstruct', selected, variant_root / 'reconstruction',
                allow_masks=index == 2, same_reconstruction=str(output / same_as / 'reconstruction') if same_as else None)
            variant = dict(name=name, state='complete', optimization=optimization, same_as=same_as,
                geometry_reused=bool(same_as), **value, resource_usage=stats,
                source_frame_count=baseline['source_frame_count'], accepted_frame_count=len(baseline['accepted_frame_ids']),
                physical_accuracy_verified=False)
            write_json_strict(variant_root / 'variant.json', variant)
            report['variants'].append(variant)
            publish()
        if changed_files(baseline['source_sha256']):
            raise ValueError('Original evidence changed during comparison')
        report['state'] = 'complete'
    except StageStopped as error:
        report.update(state=error.status, error=str(error))
    except KeyboardInterrupt:
        report.update(state='cancelled', error='Interrupted by user; no latest result published')
    except Exception as error:
        report.update(state='failed', error=str(error))
        (output / 'supervisor_error.log').write_text(traceback.format_exc(), encoding='utf-8')
    finally:
        report['elapsed_s'] = time.monotonic()-started
        report['latest_changed'] = changed_files(latest)
        report['latest_unchanged'] = not report['latest_changed']
        if baseline_path.is_file():
            report['source_changed'] = changed_files(load_json_object(baseline_path)['source_sha256'])
            report['source_unchanged'] = not report['source_changed']
        else:
            report['source_unchanged'] = None
        if report['state'] == 'complete' and (not report['source_unchanged'] or not report['latest_unchanged']):
            report.update(state='failed', error='Source or latest fingerprint changed; external changes are not restored')
        publish()
        from femto.pipeline.local_comparison_viewer import write_comparison_html
        write_comparison_html(output, report)
    return report


def main(argv: Any=None) -> Any:
    parser = argparse.ArgumentParser(description='Independent same-source local A/B/C replay; never writes a business latest index')
    parser.add_argument('dataset', nargs='?', type=Path)
    parser.add_argument('output', nargs='?', type=Path)
    parser.add_argument('--voxel-m', type=float, default=.005)
    parser.add_argument('--sdf-trunc-m', type=float, default=.02)
    parser.add_argument('--max-views', type=int, default=20)
    parser.add_argument('--stage-timeout-s', type=float, default=300.)
    parser.add_argument('--memory-mb', type=float, default=3072.)
    parser.add_argument('--render-width', type=int, default=640)
    parser.add_argument('--allow-synthetic', action='store_true')
    parser.add_argument('--cancel-file', type=Path)
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--worker-result', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        value = _worker(load_json_object(args.worker))
        write_json_strict(args.worker_result, value)
        return 0
    if args.dataset is None or args.output is None:
        parser.error('dataset and a new output directory are required')
    report = run_comparison(args.dataset, args.output, voxel_m=args.voxel_m, sdf_trunc_m=args.sdf_trunc_m,
        max_views=args.max_views, stage_timeout_s=args.stage_timeout_s, memory_mb=args.memory_mb,
        render_width=args.render_width, allow_synthetic=args.allow_synthetic, cancel_file=args.cancel_file)
    print(json.dumps(dict(state=report['state'], report=str(args.output / 'comparison_report.html'),
                         error=report.get('error'), physical_accuracy_verified=False), ensure_ascii=False))
    return 0 if report['state'] == 'complete' else 3 if report['state'] == 'cancelled' else 2


if __name__ == '__main__':
    raise SystemExit(main())
