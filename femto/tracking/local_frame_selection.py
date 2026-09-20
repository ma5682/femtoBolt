"""View admission evidence in fixed W0, separate from acquisition/fusion pixels."""
from __future__ import annotations
from typing import Any

from dataclasses import dataclass
import cv2
import numpy as np
from femto.tracking.moving_quality import POLICY, rotation_deg
from femto.dataset.scene_dataset_validator import validate_pose


SELECTION_POLICY = dict(schema='femto-local-view-selection-v1',
    sample_budget=8192, minimum_samples=128, max_views=128,
    depth_consistency_m=.01, minimum_overlap=.60, maximum_front_conflict=.15,
    minimum_new_samples=64, minimum_new_fraction=.025,
    small_motion_translation_m=.002, small_motion_rotation_deg=.3,
    coarse_translation_m=POLICY['keyframe_translation_m'], coarse_rotation_deg=POLICY['keyframe_rotation_deg'])


@dataclass
class ViewEvidence:
    frame_id: int
    pose: np.ndarray
    k: np.ndarray
    depth: np.ndarray
    points: np.ndarray
    sampling_stride: int


def view_evidence(prepared: Any, record: dict[str, Any]) -> ViewEvidence:
    if validate_pose(record.get('T_WC')):
        raise ValueError('Invalid W0 view pose')
    depth = np.asarray(prepared.depth_mm)
    k = np.asarray(prepared.intrinsic.intrinsic_matrix)
    stride = max(1, int(np.ceil(np.sqrt(depth.size/SELECTION_POLICY['sample_budget']))))
    yy, xx = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
    z = depth[::stride, ::stride].astype(float)/1000
    valid = (z > 0) & np.isfinite(z)
    # Selection evidence avoids depth discontinuities. It never masks the raw
    # frames or the eventual reconstruction input, including thin structures.
    high = cv2.dilate(depth, np.ones((3, 3), np.uint8))[::stride, ::stride]/1000.
    low = cv2.erode(depth, np.ones((3, 3), np.uint8))[::stride, ::stride]/1000.
    valid &= (high-low <= .02)
    camera = np.c_[(xx[valid]-k[0, 2])*z[valid]/k[0, 0], (yy[valid]-k[1, 2])*z[valid]/k[1, 1], z[valid]]
    pose = np.asarray(record['T_WC'], float)
    world = camera @ pose[:3, :3].T+pose[:3, 3]
    return ViewEvidence(record['frame_id'], pose.copy(), k.copy(), depth.copy(), world, stride)


def assess_view(candidate: ViewEvidence, previous: list[ViewEvidence]) -> dict[str, Any]:
    policy = SELECTION_POLICY
    count = len(candidate.points)
    report = dict(policy=dict(policy), sample_count=count, sampling_stride=candidate.sampling_stride,
                  frame_id=candidate.frame_id, geometry_evaluated=True,
                  new_sampling_point_count=count if not previous else 0,
                  sampling_count_is_not_hole_filling_or_completeness=True)
    if not previous:
        return dict(report, state='accepted', reason='initial_reference', overlap_ratio=None)
    if count < policy['minimum_samples']:
        return dict(report, state='quality_insufficient', reason='insufficient_3d_samples')
    seen = np.zeros(count, bool)
    unknown_or_occluded = np.ones(count, bool)
    front_conflict = np.zeros(count, bool)
    differences = []
    overlap_per_view = []
    for view in previous:
        inverse = np.linalg.inv(view.pose)
        camera = candidate.points @ inverse[:3, :3].T+inverse[:3, 3]
        z = camera[:, 2]
        denominator = np.maximum(z, 1e-9)
        uv = np.c_[camera[:, 0]*view.k[0, 0]/denominator+view.k[0, 2],
                   camera[:, 1]*view.k[1, 1]/denominator+view.k[1, 2]]
        pixel = np.rint(uv).astype(int)
        inside = (z > 0) & (pixel[:, 0] >= 0) & (pixel[:, 0] < view.depth.shape[1]) & (pixel[:, 1] >= 0) & (pixel[:, 1] < view.depth.shape[0])
        observed = view.depth[np.clip(pixel[:, 1], 0, view.depth.shape[0]-1), np.clip(pixel[:, 0], 0, view.depth.shape[1]-1)]/1000.
        measured = inside & (observed > 0)
        consistent = measured & (np.abs(observed-z) <= policy['depth_consistency_m'])
        seen |= consistent
        front = measured & (z < observed-policy['depth_consistency_m'])
        front_conflict |= front
        # A newly appearing foreground can be a moving hand: do not count that
        # depth conflict as new surface evidence for the small-motion exception.
        unknown_or_occluded &= ~front & ~consistent
        overlap_per_view.append(float(consistent.mean()))
        delta = np.linalg.inv(view.pose) @ candidate.pose
        differences.append((float(np.linalg.norm(delta[:3, 3])), rotation_deg(delta)))
    overlap = float(seen.mean())
    conflicts = float((front_conflict & ~seen).mean())
    novel = int((unknown_or_occluded & ~seen).sum())
    coarse = all(t >= policy['coarse_translation_m'] or a >= policy['coarse_rotation_deg'] for t, a in differences)
    small = all(t >= policy['small_motion_translation_m'] or a >= policy['small_motion_rotation_deg'] for t, a in differences)
    report.update(overlap_ratio=overlap, per_view_overlap=overlap_per_view, front_conflict_fraction=conflicts,
                  new_sampling_point_count=novel, new_sampling_fraction=novel/count,
                  closest_translation_m=min(t for t, a in differences), closest_rotation_deg=min(a for t, a in differences),
                  comparison_views=len(previous), coarse_pose_novel=coarse, small_pose_novel=small)
    if overlap < policy['minimum_overlap']:
        return dict(report, state='quality_insufficient', reason='insufficient_3d_overlap')
    if conflicts > policy['maximum_front_conflict']:
        return dict(report, state='quality_insufficient', reason='foreground_depth_inconsistent')
    # Normal pose spacing remains admissible after the geometry/quality checks.
    # Revisited views are checked against every admitted view, not just the last.
    if coarse:
        return dict(report, state='accepted', reason='distinct_pose_with_3d_overlap')
    if small and novel >= policy['minimum_new_samples'] and novel/count >= policy['minimum_new_fraction']:
        return dict(report, state='accepted', reason='small_motion_with_new_observed_samples')
    return dict(report, state='duplicate', reason='redundant_view_or_insufficient_new_samples')


class ViewSelector:
    def __init__(self):
        self.views = []

    def evaluate(self, prepared: Any, record: dict[str, Any]) -> tuple[ViewEvidence | None, dict[str, Any]]:
        if len(self.views) >= SELECTION_POLICY['max_views']:
            return None, dict(state='quality_insufficient', reason='view_evidence_budget', geometry_evaluated=True)
        view = view_evidence(prepared, record)
        return view, assess_view(view, self.views)

    def accept(self, view: ViewEvidence | None) -> None:
        if view is not None:
            self.views.append(view)


LOCAL_RADIUS_M = .03  # Acquisition envelope, not measurement accuracy.

LOCAL_ROTATION_DEG = 10.0  # Total rotation relative to W0, not per Euler axis.

def local_workspace(transform: Any) -> dict[str, Any]:
    transform=np.asarray(transform,dtype=float)
    radius=float(np.linalg.norm(transform[:3,3]));angle=rotation_deg(transform)
    return dict(radius_m=radius,limit_m=LOCAL_RADIUS_M,rotation_deg=angle,
                rotation_limit_deg=LOCAL_ROTATION_DEG,
                within_limit=bool(np.isfinite([radius,angle]).all() and
                    radius<=LOCAL_RADIUS_M+1e-12 and angle<=LOCAL_ROTATION_DEG+1e-9))
