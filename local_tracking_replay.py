"""Read-only recorded RGB-D replay. Timing is offline compute, never live FPS."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np

from local_tracking_diagnostics import distribution, summarize, image_diagnostics
from scene_live_tracker import LiveTracker, PairEstimator
from scene_reconstruction import _read_image


def read_json(path):
    return json.loads(Path(path).read_text('utf-8-sig'))


def rows(path):
    return [json.loads(s) for s in Path(path).read_text('utf-8-sig').splitlines() if s.strip()]


def frame_from_row(source, row):
    return SimpleNamespace(color_bgr=_read_image(source/row['rgb'], cv2.IMREAD_COLOR),
        depth_mm=_read_image(source/row['depth'], cv2.IMREAD_UNCHANGED),
        timestamp_us=row['camera_timestamp_us'], frame_index=row['camera_frame_index'],
        color_timestamp_us=row['color_timestamp_us'], depth_timestamp_us=row['depth_timestamp_us'],
        system_timestamp_us=row.get('system_timestamp_us', 0),
        color_system_timestamp_us=row.get('color_system_timestamp_us', 0),
        depth_system_timestamp_us=row.get('depth_system_timestamp_us', 0),
        frame_metadata=row.get('frame_metadata', {}), native_depth_mm=None)


def source_files(source):
    files = [source/'frames.jsonl', source/'manifest.json', source/'derived/live_tracking.jsonl',
             source/'derived/local_capture_report.json', *sorted((source/'calibration').glob('*.json'))]
    for row in rows(source/'frames.jsonl'):
        files.extend(source/row[key] for key in ('rgb', 'depth', 'pose'))
    return {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def load_baseline(directory):
    path = Path(directory)/'scene_live_tracker.py'
    spec = importlib.util.spec_from_file_location('recorded_baseline_tracker', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def replay(source, output, *, baseline=None, variants=('baseline', 'optimized')):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents:
        raise ValueError('Replay output must be outside original acquisition evidence')
    output.mkdir(parents=True, exist_ok=False)
    original = source_files(source)
    manifest = read_json(source/'manifest.json')
    log = rows(source/'derived/live_tracking.jsonl')
    raw = rows(source/'frames.jsonl')
    historical = read_json(source/'derived/local_capture_report.json')
    minimum = manifest['parameters']['depth_min_mm']
    maximum = manifest['parameters']['depth_max_mm']
    baseline_module = load_baseline(baseline) if baseline is not None else None
    reports = {};records_by_variant={}
    if len(raw)!=len(log):raise ValueError('Raw/tracking log lengths differ; no silent zip truncation')
    for variant in variants:
        if variant == 'baseline':
            if baseline_module is None:
                raise ValueError('Baseline source snapshot is required')
            estimator = baseline_module.PairEstimator(source, minimum, maximum)
            tracker_type = baseline_module.LiveTracker
        else:
            estimator = PairEstimator(source, minimum, maximum,
                reverse_seed='forward_inverse_experiment' if variant == 'reverse' else 'independent',
                background_mask=variant == 'background',feature_seed=variant in ('feature_seed','all'))
            tracker_type = LiveTracker
        logical = [0.]
        options={} if variant=='baseline' else dict(keyframe_anchor=variant in ('keyframe_anchor','all'),relocalization=variant in ('relocalize','all'))
        tracker = tracker_type(estimator, minimum, maximum, clock=lambda: logical[0],**options)
        from local_frame_selection import ViewSelector
        selector=ViewSelector()
        records, compute = [], []
        out_log = output/(variant+'.jsonl')
        with out_log.open('x', encoding='utf-8') as stream:
            for row, saved in zip(raw, log):
                if row['frame'] != saved['frame_id'] or row['camera_timestamp_us'] != saved['capture_timestamp_us']:
                    raise ValueError('Recorded RGB-D / tracking index mismatch')
                frame = frame_from_row(source, row)
                logical[0] = saved['received_monotonic']
                start = time.perf_counter()
                record = tracker.process(frame, row['frame'], logical[0])
                elapsed = (time.perf_counter()-start)*1000
                compute.append(elapsed)
                record['offline_compute_ms'] = elapsed
                record['source_selected_for_fusion'] = saved.get('selected_for_fusion', False)
                # A replay does not reconstruct unrecorded button presses.
                record['selected_for_fusion'] = False
                record['replay_keyframe_admitted']=False
                if variant!='baseline' and saved.get('selected_for_fusion') and record['state']=='tracking':
                    from local_capture import local_workspace
                    safe=(local_workspace(record['T_WC'])['within_limit'] and not record.get('motion_discontinuity')
                          and record.get('continuity',{}).get('current_frame_verified')
                          and record.get('step_m',0.)<=.003 and record.get('step_deg',0.)<=1.)
                    if safe:
                        view,selection=selector.evaluate(tracker.previous,record)
                        record['replay_admission']=selection
                        if selection['state']=='accepted':
                            selector.accept(view)
                            record['replay_keyframe_admitted']=tracker.add_keyframe(tracker.previous,record['T_WC'],record['frame_id'])
                if (row['frame']+1)%20==0:print(f'{source.name} {variant}: {row["frame"]+1}/{len(raw)}',flush=True)
                records.append(record)
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n')
        if getattr(tracker, 'relocalizing', False):
            tracker.mark_lost('replay_end_while_relocalizing')
        summary = summarize(records)
        summary.update(offline_compute_ms=distribution(compute),
                       offline_valid_pose_throughput=len([r for r in records if r['state']=='tracking'])/(sum(compute)/1000),
                       reference_frame_id=tracker.reference_id, lost=tracker.lost,
                       tracker_options=tracker.options() if hasattr(tracker,'options') else None,
                       relocalization_events=getattr(tracker,'relocalization_events',[]),
                       replay_keyframes=[r['frame_id'] for r in records if r.get('replay_keyframe_admitted')],
                       limitation='Recorded selected frames are opportunities only; variant pose/workspace/stillness/view gates rechecked. No new user requests inferred; lost recordings may end before any recoverable frames. Recorded receipt gaps retained; CPU age, input queue, disk IO, sensor and display are NOT replayed. No new keyframe acceptance or live FPS claim.')
        reports[variant] = summary;records_by_variant[variant]=records
        print(json.dumps(dict(source=source.name, variant=variant, valid=summary['valid_pose_frames'],
                              total=len(records), compute_p50_ms=summary['offline_compute_ms']['p50']), ensure_ascii=False), flush=True)
    comparisons={}
    reference='baseline' if 'baseline' in records_by_variant else 'optimized' if 'optimized' in records_by_variant else variants[0]
    base={r['frame_id']:r for r in records_by_variant[reference] if r['state']=='tracking'}
    from moving_quality import rotation_deg
    for variant,records in records_by_variant.items():
        common=[r for r in records if r['state']=='tracking' and r['frame_id'] in base]
        deltas=[np.linalg.inv(np.asarray(base[r['frame_id']]['T_WC']))@np.asarray(r['T_WC']) for r in common]
        comparisons[variant]=dict(reference=reference,common_frames=len(common),
            translation_mm=distribution([float(np.linalg.norm(t[:3,3])*1000) for t in deltas]),
            rotation_deg=distribution([rotation_deg(t) for t in deltas]),
            lost_baseline_valid_frames=[i for i in base if not any(r['frame_id']==i and r['state']=='tracking' for r in records)],
            interpretation='Pose disagreement, not absolute accuracy or a pass criterion')
    current = source_files(source)
    report = dict(schema='femto-local-tracking-replay-v1', source=str(source), source_sha256=original,
                  source_unchanged=current == original, recorded_policy={k: historical.get(k) for k in
                    ('local_workspace_radius_m', 'local_workspace_rotation_deg', 'max_gap_s')},
                  historical=summarize(log, historical.get('duration_s'), historical.get('dropped_input_frames')),
                  variants=reports, comparisons=comparisons,physical_accuracy_verified=False,
                  tracking_code_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ('scene_live_tracker.py','local_tracking_replay.py','local_frame_selection.py')})
    (output/'replay_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    if not report['source_unchanged']:
        raise ValueError('Raw source changed during replay')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sources', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--variants', nargs='+', choices=('baseline', 'optimized', 'reverse', 'background','feature_seed','keyframe_anchor','relocalize','all'), default=['baseline', 'optimized'])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for source in args.sources:
        results.append(replay(source, args.output/source.name, baseline=args.baseline, variants=args.variants))
    (args.output/'summary.json').write_text(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
