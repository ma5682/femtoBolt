"""Bounded capture feedback using the shared offline/online stillness rules."""
from __future__ import annotations
from typing import Any

from collections import deque
import math
import numpy as np
from femto.dataset.scene_dataset_validator import validate_pose
from femto.tracking.local_stability import (WINDOW_FRAMES, MAX_SPAN_US, MAX_ADJACENT_GAP_US, MAX_RGBD_SYNC_US,
                             frame_still, window_valid, find_window, adjacent)

TARGET = WINDOW_FRAMES
OBSERVATION_SECONDS = MAX_SPAN_US / 1e6


class CandidateStability:
    def __init__(self, session_id):
        self.session_id = session_id
        self.frames = deque(maxlen=TARGET)
        self.history = deque(maxlen=2*TARGET-1)
        self.state = 'idle'
        self.reason = None
        self.requested_at = None
        self.anchor = None
        self.accepted_frame_id = None
        self.reference_id = None
        self.deadline = None
        self.peak_count = 0
        self.last_episode = None
        self.deviation = None
        self.current_frame_id = None
        self.reason_frame_id = None
        self.adjacent_deltas = deque(maxlen=256)

    def reset(self, state: str='idle', reason: str | None=None, *, clear_history: bool=True) -> None:
        if self.anchor is not None:
            self.last_episode = dict(accepted_frame_id=self.accepted_frame_id, peak_count=self.peak_count,
                                     reason=reason, deviation=self.deviation)
        self.frames.clear()
        if clear_history:
            self.history.clear()
        self.state, self.reason = state, reason
        self.reason_frame_id = self.current_frame_id if reason else None
        self.requested_at = self.anchor = self.deadline = None
        self.accepted_frame_id = None

    def request(self, now: float) -> None:
        self.reset('waiting', clear_history=False)
        self.peak_count = 0
        self.deviation = None
        self.requested_at = now
        self.deadline = now + 3.0

    def expire(self, now: float) -> None:
        if self.deadline is not None and now > self.deadline:
            self.reset('timeout', 'request_timeout' if self.anchor is None else 'window_timeout')

    def _invalid_reason(self, record, now):
        if record.get('session_id') != self.session_id:
            return 'session_mismatch'
        if record.get('state') != 'tracking' or validate_pose(record.get('T_WC')):
            return record.get('rejection_reason') or 'untrusted_pose'
        if type(record.get('frame_id')) is not int or type(record.get('reference_frame_id')) is not int:
            return 'invalid_frame_id'
        received = record.get('received_monotonic')
        if not isinstance(received, (float, int)) or not math.isfinite(received) or not 0 <= now-received <= 2.0:
            return 'stale_frame'
        times = [record.get(k) for k in ('capture_timestamp_us', 'color_timestamp_us', 'depth_timestamp_us')]
        if any(type(t) is not int or t < 0 for t in times) or abs(times[1]-times[2]) > MAX_RGBD_SYNC_US:
            return 'rgb_depth_sync'
        if self.reference_id is not None and record.get('reference_frame_id') != self.reference_id:
            return 'reference_mismatch'
        return None

    @staticmethod
    def descriptor(record: Any) -> dict[str, Any]:
        return dict(frame_id=record['frame_id'], session_id=record['session_id'], state='tracking',
                    reference_frame_id=record['reference_frame_id'],
                    capture_timestamp_us=record['capture_timestamp_us'],
                    color_timestamp_us=record['color_timestamp_us'], depth_timestamp_us=record['depth_timestamp_us'],
                    received_monotonic=record['received_monotonic'],
                    T_WC=np.asarray(record['T_WC'], dtype=float).copy().tolist())

    def observe(self, record: Any, now: float, *, accepted: bool=False, paused: bool=False) -> Any:
        self.current_frame_id=record.get('frame_id') if isinstance(record,dict) else None
        if paused:
            self.reset('paused')
            return self.snapshot()
        if record.get('motion_discontinuity'):
            self.reset('reset','pose_correction_recheck')
            return self.snapshot()
        bad = self._invalid_reason(record, now)
        if bad:
            self.reset('lost' if record.get('state') == 'lost' else 'reset', bad)
            return self.snapshot()
        self.reference_id = record.get('reference_frame_id')
        self.expire(now)
        current = self.descriptor(record)
        if self.history and self.history[-1]['frame_id'] == current['frame_id']:
            return self.snapshot()
        if self.history and not adjacent(self.history[-1], current):
            self.reset('reset', 'timestamp_gap')
            self.history.append(current)
            return self.snapshot()
        if self.history:
            from femto.tracking.moving_quality import rotation_deg
            delta=np.linalg.inv(np.asarray(self.history[-1]['T_WC']))@np.asarray(current['T_WC'])
            self.adjacent_deltas.append((float(np.linalg.norm(delta[:3,3])*1000),float(rotation_deg(delta))))
        self.history.append(current)
        if accepted:
            if record.get('local_workspace', {}).get('within_limit') is False:
                self.reset('reset', 'outside_local_workspace')
                return self.snapshot()
            self.accepted_frame_id = record['frame_id']
            self.anchor = current
            self.deadline = record['received_monotonic'] + OBSERVATION_SECONDS
            self.state, self.reason = 'collecting', None
            self.peak_count = 0
            self.deviation = None
        elif self.state not in ('waiting', 'moving', 'collecting'):
            return self.snapshot()
        if self.anchor is None and self.requested_at is not None and record['received_monotonic'] < self.requested_at:
            return self.snapshot()
        anchor = self.anchor or current
        if self.anchor is not None:
            window = find_window(list(self.history), self.accepted_frame_id)
            if window is not None:
                self.frames.clear(); self.frames.extend(window[0])
                self.state, self.reason, self.deadline, self.peak_count = 'candidate_ready', None, None, TARGET
                return self.snapshot()
        elif record.get('selection', {}).get('state') == 'moving':
            self.frames.clear(); self.state = 'moving'
            return self.snapshot()
        else:
            self.state = 'waiting'
        candidates = list(self.history)
        if self.anchor is None and self.requested_at is not None:
            candidates = [r for r in candidates if r['received_monotonic'] >= self.requested_at]
        if not frame_still(current, anchor):
            from femto.tracking.moving_quality import rotation_deg
            delta = np.linalg.inv(np.asarray(anchor['T_WC'])) @ np.asarray(current['T_WC'])
            self.deviation = dict(translation_mm=float(np.linalg.norm(delta[:3, 3])*1000), rotation_deg=rotation_deg(delta))
            self.reset('reset', 'motion_after_capture')
            return self.snapshot()
        last_bad = max((i for i,r in enumerate(candidates) if not frame_still(r, anchor)), default=-1)
        candidates = candidates[last_bad+1:][-TARGET:]
        self.frames.clear()
        self.frames.extend(candidates)
        self.peak_count = max(self.peak_count, len(self.frames))
        if self.anchor is not None and not any(r['frame_id'] == self.accepted_frame_id for r in self.frames):
            self.reset('reset', 'accepted_frame_outside_window')
            return self.snapshot()
        if len(self.frames) == TARGET:
            if window_valid(list(self.frames), anchor):
                self.state = 'candidate_ready'
                self.deadline = None
            else:
                self.reset('reset', 'timestamp_gap')
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        count = len(self.frames)
        span = ((self.frames[-1]['capture_timestamp_us'] - self.frames[0]['capture_timestamp_us']) / 1e6
                if count > 1 else 0.0)
        jitter=(np.median(np.asarray(self.adjacent_deltas),axis=0).tolist()
                if self.adjacent_deltas else [None,None])
        return dict(state=self.state, reason=self.reason, count=count, target=TARGET,
                    reason_triggered_this_frame=bool(self.reason and self.reason_frame_id==self.current_frame_id),
                    reason_trigger_frame_id=self.reason_frame_id,current_frame_id=self.current_frame_id,
                    accepted_frame_id=self.accepted_frame_id, reference_frame_id=self.reference_id,
                    source_frame_ids=[r['frame_id'] for r in self.frames],
                    span_s=span, observed_pose_fps=(count-1)/span if span > 0 else None,
                    descriptor_limit=TARGET, check='pose_and_timestamps_only',
                    history_descriptor_limit=2*TARGET-1, peak_count=self.peak_count,
                    last_episode=self.last_episode, deviation=self.deviation,
                    adjacent_pose_delta_median_mm=jitter[0],adjacent_pose_delta_median_deg=jitter[1],
                    image_depth_audit='pending_offline', keyframes_added_by_monitor=0)


def format_stability(value: Any, language: str='zh') -> Any:
    zh = language == 'zh'
    state = value.get('state', 'idle')
    explanations = {
        'idle': ('请求补采后显示候选帧准备情况。', 'Request a view to see candidate preparation.'),
        'waiting': ('等待停稳单帧接纳，请保持不动。', 'Waiting to accept a still single view; keep still.'),
        'moving': ('正在移动；等待停稳后接纳。', 'Moving; waiting for the camera to settle.'),
        'collecting': ('单帧已接纳，请继续停稳以记录候选窗口。', 'Single view accepted; keep still for candidate observations.'),
        'candidate_ready': ('候选窗口已记录，图像/深度仍待结束审核。', 'Candidate window recorded; image/depth audit is still pending.'),
        'paused': ('补采暂停，候选计数清零；继续跟踪。', 'Keyframes paused; candidates cleared, tracking continues.'),
        'timeout': ('候选观察已超时；已接纳单帧仍保留。', 'Candidate observation timed out; any accepted single view is retained.'),
        'reset': ('候选计数已重置，不能视为稳定通过。', 'Candidate count reset; stability has not passed.'),
        'lost': ('失跟，候选无效；不得续接旧原点。', 'Tracking lost; candidates invalid, no continuation of the old origin.'),
        'cancelled': ('请求已取消，候选计数清零。', 'Request cancelled; candidates cleared.'),
        'ended': ('采集结束，完整审核在后台执行。', 'Capture ended; full audit runs in the background.'),
        'duplicate': ('视角重复，未增加关键帧。', 'Duplicate view; no keyframe added.'),
    }
    description = explanations.get(state, (state, state))[0 if zh else 1]
    heading = '候选稳定帧（仅位姿）' if zh else 'Candidate stable frames (Pose only)'
    text = f"{heading}: {value.get('count', 0)}/{value.get('target', TARGET)}"
    if value.get('accepted_frame_id') is not None:
        text += (' · 对应接纳帧 ' if zh else ' · accepted frame ') + str(value['accepted_frame_id'])
    fps = value.get('observed_pose_fps')
    if fps is not None:
        text += f" · {value['span_s']:.2f}s / {fps:.2f} Pose FPS"
    if value.get('reason'):
        reasons = {
            'motion_after_capture': ('接纳后位姿变化超过 1 mm / 0.2°', 'Pose changed over 1 mm / 0.2 degrees after capture'),
            'timestamp_gap': ('时间戳重复或间隔超过 1 秒', 'Duplicate timestamp or gap over 1 second'),
            'outside_local_workspace': ('锚帧超出本会话接纳范围', 'Anchor is outside this session capture envelope'),
            'stale_frame': ('位姿过期', 'Stale pose'),
            'rgb_depth_sync': ('RGB-D 不同步', 'Unsynchronized RGB-D'),
            'window_timeout': ('接纳后观察超过 4 秒', 'Over 4 seconds after acceptance'),
            'request_timeout': ('请求超过 3 秒', 'Request exceeded 3 seconds'),
            'untrusted_pose': ('位姿不可信', 'Untrusted pose'),
            'session_mismatch': ('会话不匹配', 'Session mismatch'),
            'reference_mismatch': ('参考帧不匹配', 'Reference mismatch'),
        }
        description += ' ' + reasons.get(value['reason'], (value['reason'], value['reason']))[0 if zh else 1]
        description += ('（本帧触发）' if value.get('reason_triggered_this_frame') else '（沿用上次结论）') if zh else (
            ' (triggered this frame)' if value.get('reason_triggered_this_frame') else ' (carried from prior frame)')
        if value.get('deviation'):
            deviation=value['deviation']
            description += (f" · 偏离 {deviation.get('translation_mm',0):.2f} mm / {deviation.get('rotation_deg',0):.3f}°"
                            if zh else f" · deviation {deviation.get('translation_mm',0):.2f} mm / {deviation.get('rotation_deg',0):.3f}°")
        jitter=value.get('adjacent_pose_delta_median_mm')
        jitter_deg=value.get('adjacent_pose_delta_median_deg')
        if jitter is not None and jitter_deg is not None:
            description += (f" · 会话相邻位姿变化中位 {jitter:.2f} mm / {jitter_deg:.3f}°"
                            if zh else f" · session median adjacent Pose delta {jitter:.2f} mm / {jitter_deg:.3f}°")
    episode = value.get('last_episode') or {}
    if value.get('count', 0) < value.get('peak_count', 0):
        description += (f" 本次最高 {value['peak_count']}/5。" if zh else f" Peak {value['peak_count']}/5.")
    deviation = value.get('deviation') or episode.get('deviation')
    if deviation:
        description += (f" 偏离 {deviation['translation_mm']:.2f} mm / {deviation['rotation_deg']:.2f}°。" if zh else
                        f" Deviation {deviation['translation_mm']:.2f} mm / {deviation['rotation_deg']:.2f} deg.")
    return text + '\n' + description + (' 不是降噪通过。' if zh else ' This is not a denoising approval.')
