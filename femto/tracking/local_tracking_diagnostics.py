"""Evidence-based tracking diagnostics. Suspected scene causes are not truth labels."""
from __future__ import annotations
from typing import Any


from collections import Counter
import cv2
import numpy as np


def distribution(values: Any) -> dict[str, Any]:
    values = np.asarray([v for v in values if isinstance(v, (float, int)) and np.isfinite(v)], float)
    return dict(count=len(values), mean=float(values.mean()) if len(values) else None,
                p50=float(np.percentile(values, 50)) if len(values) else None,
                p95=float(np.percentile(values, 95)) if len(values) else None,
                maximum=float(values.max()) if len(values) else None)


def image_diagnostics(color: Any, depth: Any, minimum_mm: float, maximum_mm: float) -> dict[str, Any]:
    """Small image/coverage probes; no image mutation or semantic classification."""
    size = (min(480, color.shape[1]), max(1, round(color.shape[0]*min(1., 480/color.shape[1]))))
    gray = cv2.cvtColor(cv2.resize(color, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=160, qualityLevel=.02, minDistance=8)
    count = 0 if corners is None else len(corners)
    occupied = 0
    if count:
        xy = corners.reshape(-1, 2)
        cells = np.minimum((xy / [size[0], size[1]] * [4, 3]).astype(int), [3, 2])
        occupied = len(np.unique(cells, axis=0))
    # Sample only for these additional hints; the admission fraction continues
    # to be computed from the entire original depth frame by frame_quality.
    depth = depth[::4, ::4]
    finite = np.isfinite(depth)
    valid = finite & (depth > 0) & (depth >= minimum_mm) & (depth <= maximum_mm)
    return dict(texture_features=count, texture_grid_cells=occupied, texture_grid_total=12, depth_hint_stride=4,
                weak_texture_suspected=count < 30 or occupied < 4,
                invalid_depth_fraction=float(np.mean(~finite | (depth <= 0))),
                depth_below_range_fraction=float(np.mean(finite & (depth > 0) & (depth < minimum_mm))),
                depth_above_range_fraction=float(np.mean(finite & (depth > maximum_mm))),
                valid_depth_fraction=float(valid.mean()),
                valid_depth_p50_mm=float(np.median(depth[valid])) if valid.any() else None,
                interpretation='texture and depth evidence; blur, occlusion and object identity are not independently proven')


def classify(record: Any) -> dict[str, Any]:
    """Return observable failures separately from unproven physical causes."""
    reasons = []
    reason = record.get('rejection_reason')
    if reason:
        reasons.append(reason)
    quality = record.get('quality', {})
    image = record.get('image_diagnostics', {})
    if quality.get('valid_depth_fraction', 1.) < .20 and 'low_depth' not in reasons:
        reasons.append('low_depth')
    if quality.get('sharpness', 1e9) < 20:
        reasons.append('low_image_sharpness')
    if image.get('weak_texture_suspected'):
        reasons.append('weak_texture_suspected')
    metrics = record.get('pair_failure') or record.get('pair') or {}
    reasons.extend(metrics.get('rejected_checks', []))
    timing = record.get('timing_ms', {})
    measured = {k: float(v) for k, v in timing.items() if isinstance(v, (float, int)) and np.isfinite(v)}
    return dict(observed=list(dict.fromkeys(reasons)),
                dominant_measured_stage=max(measured, key=measured.get) if measured else None,
                dynamic_object='suspected' if (metrics.get('background') or {}).get('motion_outlier_count', 0) else 'not_established',
                occlusion='not_established', motion_blur='not_separable_from_low_texture_without_scene_evidence',
                synchronization_delta_us=(abs(record['color_timestamp_us']-record['depth_timestamp_us'])
                    if type(record.get('color_timestamp_us')) is int and type(record.get('depth_timestamp_us')) is int else None))


def summarize(records: Any, duration_s: float | None=None, dropped: Any=None) -> dict[str, Any]:
    valid = [r for r in records if r.get('state') == 'tracking']
    failures = [r for r in records if r.get('state') != 'tracking']
    first = failures[0] if failures else None
    stages = sorted({k for r in records for k in r.get('timing_ms', {})})
    return dict(schema='femto-tracking-diagnostics-v1', processed_frames=len(records),
                valid_pose_frames=len(valid), valid_pose_fps=len(valid)/duration_s if duration_s and duration_s > 0 else None,
                processed_fps=len(records)/duration_s if duration_s and duration_s > 0 else None,
                pose_latency_ms=distribution([r.get('latency_ms') for r in records]),
                durable_latency_ms=distribution([r.get('durable_latency_ms') for r in records]),
                timing_ms={k: distribution([r.get('timing_ms', {}).get(k) for r in records]) for k in stages},
                first_failure=dict(frame_id=first.get('frame_id'), **classify(first)) if first else None,
                maximum_consecutive_failures=max((r.get('consecutive_failures', 0) for r in records), default=0),
                rejection_counts=dict(Counter(r.get('rejection_reason', 'unrecorded') for r in failures)),
                selection_counts=dict(Counter(r.get('selection', {}).get('state', 'unrecorded') for r in records)),
                accepted_frames=[r['frame_id'] for r in records if r.get('selected_for_fusion')],
                dropped_input_frames=dropped, unknown_hardware_drops=True,
                relocalizing_frames=sum(r.get('state')=='relocalizing' for r in records),
                relocalized_frames=[r['frame_id'] for r in records if r.get('continuity',{}).get('relocalized')],
                anchor_disagreement_count=sum(r.get('rejection_reason')=='anchor_disagreement' for r in records),
                feature_seed_pair_count=sum(r.get('pair',{}).get('initialization')=='feature_pnp' and r['state']=='tracking' for r in records),
                limitations=['Host receipt to processing latency; not sensor-to-screen latency.',
                             'Queue replacements are not camera disconnects.',
                             'Consistency and repeatability only; no independent physical ground truth.'])


def quality_text(record: Any, language: str='zh') -> Any:
    quality = (record or {}).get('quality', {})
    depth, sharpness = quality.get('valid_depth_fraction'), quality.get('sharpness')
    if depth is None or sharpness is None:
        return ''
    zh = language == 'zh'
    state = ('深度不足' if zh else 'insufficient depth') if depth < .20 else (
            ('深度覆盖偏低' if zh else 'low depth coverage') if depth < .35 else ('深度可用' if zh else 'depth available'))
    texture = (record or {}).get('image_diagnostics', {})
    text = (f'有效深度 {depth:.1%} · 清晰度 {sharpness:.1f} · {state}' if zh else
            f'Valid depth {depth:.1%} · sharpness {sharpness:.1f} · {state}')
    if texture.get('weak_texture_suspected'):
        text += ' · 纹理支持偏少' if zh else ' · sparse texture support'
    return text


def current_feedback(record: Any, accepted: Any, now: float, active: bool=True) -> dict[str, Any]:
    """Current-frame status only; historical attempt errors never override recovery."""
    r=record or {};received=r.get('received_monotonic')
    age=max(0.,now-received) if isinstance(received,(int,float)) else None
    stale=bool(active and age is not None and age>2.)
    reason=('stale_frame' if stale else r.get('rejection_reason') if r.get('state')!='tracking' else
            None if r.get('selected_for_fusion') else r.get('selection',{}).get('reason'))
    return dict(schema='femto-current-capture-quality-v1',frame_id=r.get('frame_id'),
        tracking_state='stale' if stale else r.get('state','unknown'),accepted_frames=accepted,
        selected_for_fusion=bool(r.get('selected_for_fusion')) and not stale,
        current_reason=reason,valid_depth_fraction=r.get('quality',{}).get('valid_depth_fraction'),
        updated_at=r.get('status_updated_at'),frame_age_s=age,reference_frame_id=r.get('reference_frame_id'))


def feedback_text(record: Any,accepted: Any,now: float,active: bool=True,language: str='zh') -> Any:
    d=current_feedback(record,accepted,now,active);zh=language=='zh'
    stamp=d['updated_at'] or ('未记录' if zh else 'unrecorded')
    reason=d['current_reason'] or ('无' if zh else 'none')
    return (f"接纳 {accepted} 帧 · 当前接纳/拒绝原因：{reason}\n状态更新 {stamp}" if zh else
            f"Accepted {accepted} frames · current admission reason: {reason}\nUpdated {stamp}")
