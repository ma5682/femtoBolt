"""Conservative capture hints and derived moving keyframe selection."""
from __future__ import annotations
import cv2
import numpy as np

POLICY = dict(min_depth_fraction=0.20, min_sharpness=20.0,
              keyframe_translation_m=0.01, keyframe_rotation_deg=2.0,
              cycle_translation_m=0.01, cycle_rotation_deg=2.0,
              max_tracking_gap_frames=5)

def frame_quality(color, depth, minimum_mm, maximum_mm):
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY) if color.ndim == 3 else color
    ratio = min(1.0, 480 / gray.shape[1])
    gray = cv2.resize(gray, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    fraction = float(np.mean(np.isfinite(depth) & (depth > 0) & (depth >= minimum_mm) & (depth <= maximum_mm)))
    reason = 'low_depth' if fraction < POLICY['min_depth_fraction'] else ('blur' if sharpness < POLICY['min_sharpness'] else 'ok')
    return dict(sharpness=sharpness, valid_depth_fraction=fraction, state=reason, usable=reason == 'ok')

def capture_feedback(first, current, depth, minimum_mm, maximum_mm, *, scale=0.25):
    from scene_pose_estimator import assess_rgb_motion_pair
    motion = assess_rgb_motion_pair(first, current, scale=scale)
    quality = frame_quality(current, depth, minimum_mm, maximum_mm)
    state = quality['state']
    if state == 'ok':
        if not motion.get('reliable'):
            state = 'uncertain'
        elif motion.get('median_background_displacement_px', 0) > current.shape[1] * 0.15:
            state = 'large_motion'
        elif motion.get('appears_fixed'):
            state = 'small_motion'
    return dict(**motion, capture_quality=dict(**quality, guidance=state), hint_only=True)

def rotation_deg(transform):
    return float(np.degrees(np.arccos(np.clip((np.trace(transform[:3,:3])-1)/2, -1, 1))))

def cycle_consistency(forward, backward):
    cycle = backward @ forward
    translation = float(np.linalg.norm(cycle[:3,3]))
    rotation = rotation_deg(cycle)
    return dict(translation_m=translation, rotation_deg=rotation,
                passed=bool(np.isfinite(cycle).all() and translation <= POLICY['cycle_translation_m'] and rotation <= POLICY['cycle_rotation_deg']))

def select_keyframes(poses):
    if not poses:
        return []
    selected = [0]
    for index in range(1, len(poses)):
        relative = np.linalg.inv(poses[selected[-1]]) @ poses[index]
        if (np.linalg.norm(relative[:3,3]) >= POLICY['keyframe_translation_m'] or
                rotation_deg(relative) >= POLICY['keyframe_rotation_deg']):
            selected.append(index)
    return selected
