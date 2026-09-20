"""Online candidate stable-frame status using the offline five-frame rules.

Pose-only checks over tracking records that the single capture worker has
already produced. No image or depth audit happens here; that full review runs
after the session ends (local_temporal_support). This module never reads the
camera SDK and never creates frames or windows that the records do not contain.
"""
from __future__ import annotations

import numpy as np

from moving_quality import rotation_deg
from scene_dataset_validator import validate_pose

WINDOW_FRAMES = 5
STILL_TRANSLATION_M = .001
STILL_ROTATION_DEG = .2
MAX_ADJACENT_GAP_US = 1_000_000
MAX_SPAN_US = 4_000_000
MAX_RGBD_SYNC_US = 5000
RULE = ('five consecutive trusted frames within 1 mm / 0.2 deg of the selected frame; '
        'adjacent capture gap <= 1 s; span <= 4 s; RGB-D sync <= 5 ms; same reference frame; pose-only')
CHECK = 'pose_only_candidate'


def frame_trusted(record):
    """A tracking record with a valid rigid pose, integer timestamp and synchronized RGB-D."""
    return bool(isinstance(record, dict) and record.get('state') == 'tracking'
                and not validate_pose(record.get('T_WC'))
                and type(record.get('capture_timestamp_us')) is int
                and type(record.get('color_timestamp_us')) is int
                and type(record.get('depth_timestamp_us')) is int
                and abs(record['color_timestamp_us']-record['depth_timestamp_us']) <= MAX_RGBD_SYNC_US)


def frame_still(record, reference):
    """Trusted and within the stillness tolerance relative to the reference record."""
    if not frame_trusted(record) or not frame_trusted(reference):
        return False
    if record.get('reference_frame_id') != reference.get('reference_frame_id'):
        return False
    relative = np.linalg.inv(np.asarray(reference['T_WC'], float)) @ np.asarray(record['T_WC'], float)
    return bool(np.linalg.norm(relative[:3, 3]) <= STILL_TRANSLATION_M
                and rotation_deg(relative) <= STILL_ROTATION_DEG)


def adjacent(earlier, later):
    a, b = earlier.get('capture_timestamp_us'), later.get('capture_timestamp_us')
    return type(a) is int and type(b) is int and a < b <= a+MAX_ADJACENT_GAP_US


def window_valid(group, reference):
    """The single definition shared by the offline filter and the online hint."""
    if len(group) != WINDOW_FRAMES or not all(frame_still(r, reference) for r in group):
        return False
    if not all(adjacent(a, b) for a, b in zip(group, group[1:])):
        return False
    return group[-1]['capture_timestamp_us']-group[0]['capture_timestamp_us'] <= MAX_SPAN_US


def find_window(records, frame_id):
    """Find an actual consecutive window containing the admitted anchor."""
    indices = [i for i, r in enumerate(records) if r.get('frame_id') == frame_id]
    if len(indices) != 1:
        return None
    i = indices[0]
    for start in sorted(range(max(0, i-4), min(i, len(records)-5)+1), key=lambda s: abs(s+2-i)):
        group = records[start:start+5]
        if window_valid(group, records[i]):
            return group, i-start
    return None


def _stop_reason(record, reference, previous, forward):
    if not isinstance(record, dict):
        return 'end_of_records'
    if record.get('state') != 'tracking':
        return 'frame_rejected'
    if not frame_trusted(record):
        return 'frame_untrusted'
    if record.get('reference_frame_id') != reference.get('reference_frame_id'):
        return 'reference_changed'
    a, b = (previous, record) if forward else (record, previous)
    if not adjacent(a, b):
        return 'time_gap'
    return 'moved'


def candidate_status(records, anchor_index):
    """Maximal stationary run containing the anchor, capped at the window size.

    ``candidate_frames`` counts actual records only. ``satisfied`` requires a
    full five-frame window that passes ``window_valid``; nothing is inferred.
    """
    anchor = records[anchor_index]
    status = dict(check=CHECK, rule=RULE, window_frames=WINDOW_FRAMES,
                  anchor_frame_id=anchor.get('frame_id'), candidate_frames=0, before=0, after=0,
                  satisfied=False, run_end_index=anchor_index, stopped_before='anchor_untrusted',
                  stopped_after='anchor_untrusted')
    if not frame_trusted(anchor):
        return status
    start = end = anchor_index
    reason_before = 'start_of_records'
    while start > 0 and anchor_index-start < WINDOW_FRAMES-1:
        candidate = records[start-1]
        if not frame_still(candidate, anchor) or not adjacent(candidate, records[start]):
            reason_before = _stop_reason(candidate, anchor, records[start], forward=False)
            break
        start -= 1
    else:
        reason_before = 'start_of_records' if start == 0 else 'window_full'
    reason_after = 'end_of_records'
    while end+1 < len(records) and end-start+1 < WINDOW_FRAMES:
        candidate = records[end+1]
        if not frame_still(candidate, anchor) or not adjacent(records[end], candidate):
            reason_after = _stop_reason(candidate, anchor, records[end], forward=True)
            break
        end += 1
    else:
        reason_after = 'end_of_records' if end+1 >= len(records) else 'window_full'
    group = records[start:end+1]
    relatives=[]
    anchor_pose=np.asarray(anchor['T_WC'],float)
    for index in range(max(0,anchor_index-4),min(len(records),anchor_index+5)):
        record=records[index]
        if frame_trusted(record) and record.get('reference_frame_id')==anchor.get('reference_frame_id'):
            relative=np.linalg.inv(anchor_pose)@np.asarray(record['T_WC'],float)
            relatives.append(dict(frame_id=record.get('frame_id'),offset=index-anchor_index,
                translation_mm=float(np.linalg.norm(relative[:3,3])*1000),rotation_deg=rotation_deg(relative)))
        else:
            relatives.append(dict(frame_id=record.get('frame_id'),offset=index-anchor_index,
                                  translation_mm=None,rotation_deg=None,state=record.get('state')))
    status.update(candidate_frames=len(group), before=anchor_index-start, after=end-anchor_index,
                  run_end_index=end, stopped_before=reason_before, stopped_after=reason_after,
                  satisfied=window_valid(group, anchor) if len(group) == WINDOW_FRAMES else False,
                  candidate_frame_ids=[r.get('frame_id') for r in group],nearby_relative_pose=relatives)
    return status


def all_stationary_windows(records):
    result=[]
    for start in range(0,max(0,len(records)-WINDOW_FRAMES+1)):
        group=records[start:start+WINDOW_FRAMES];reference=group[WINDOW_FRAMES//2]
        if window_valid(group,reference):
            result.append(dict(source_frame_ids=[r.get('frame_id') for r in group],
                reference_frame_id=reference.get('frame_id'),start_index=start,end_index=start+WINDOW_FRAMES-1))
    return result


def window_seconds_estimate(processed_fps):
    """Real-rate estimate of one five-frame window; informational only."""
    if not processed_fps or not np.isfinite(processed_fps) or processed_fps <= 0:
        return None
    return (WINDOW_FRAMES-1)/float(processed_fps)
