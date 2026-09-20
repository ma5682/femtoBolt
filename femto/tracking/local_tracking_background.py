"""Opt-in background-consensus experiment, never a semantic or motion truth mask.

Spatially distributed perimeter features propose motion; a symmetric projected
depth test masks disagreements only in tracking copies. The full original RGB-D
remains the acquisition evidence and reconstruction source. No relocalization.
"""
from __future__ import annotations
import cv2
import numpy as np
import open3d as o3d


def _depth_agreement(source, target, k, transform, tolerance=.02):
    yy, xx = np.indices(source.shape)
    z = source.astype(float)/1000
    p = np.stack(((xx-k[0, 2])*z/k[0, 0], (yy-k[1, 2])*z/k[1, 1], z), axis=-1)
    p = p @ transform[:3, :3].T+transform[:3, 3]
    positive = (source > 0) & (p[..., 2] > 0)
    denominator = np.maximum(p[..., 2], 1e-9)
    u = np.rint(p[..., 0]*k[0, 0]/denominator+k[0, 2]).astype(int)
    v = np.rint(p[..., 1]*k[1, 1]/denominator+k[1, 2]).astype(int)
    inside = positive & (u >= 0) & (u < target.shape[1]) & (v >= 0) & (v < target.shape[0])
    observed = target[np.clip(v, 0, target.shape[0]-1), np.clip(u, 0, target.shape[1]-1)]/1000.
    return inside & (observed > 0) & (np.abs(observed-p[..., 2]) <= tolerance)


def consensus(previous, current):
    a = cv2.cvtColor(previous.color_bgr, cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(current.color_bgr, cv2.COLOR_BGR2GRAY)
    height, width = a.shape
    yy, xx = np.indices(a.shape)
    perimeter = ((xx < width*.25) | (xx >= width*.75) | (yy < height*.25) | (yy >= height*.75))
    mask = (perimeter & (previous.depth_mm > 0)).astype(np.uint8)*255
    points = cv2.goodFeaturesToTrack(a, 400, .01, 7, mask=mask)
    report = dict(applied=False, method='perimeter_LK_PnP_and_symmetric_depth_reprojection',
                  feature_count=0 if points is None else len(points), reason='insufficient_features',
                  tracking_only=True, physical_background_verified=False, depth_tolerance_m=.02)
    if points is None or len(points) < 30:
        return None, None, report
    tracked, status, _ = cv2.calcOpticalFlowPyrLK(a, b, points, None, winSize=(21, 21), maxLevel=3)
    if tracked is None:
        return None, None, report
    back, status_back, _ = cv2.calcOpticalFlowPyrLK(b, a, tracked, None, winSize=(21, 21), maxLevel=3)
    if back is None:
        return None, None, report
    old, new = points.reshape(-1, 2), tracked.reshape(-1, 2)
    usable = status.ravel().astype(bool) & status_back.ravel().astype(bool)
    usable &= np.isfinite(new).all(axis=1) & (np.linalg.norm(back.reshape(-1, 2)-old, axis=1) <= 1.)
    usable &= (new[:, 0] >= 0) & (new[:, 0] < width) & (new[:, 1] >= 0) & (new[:, 1] < height)
    old, new = old[usable], new[usable]
    report['flow_consistent_features'] = len(old)
    if len(old) < 30:
        report['reason'] = 'insufficient_bidirectional_flow'
        return None, None, report
    k = np.asarray(previous.intrinsic.intrinsic_matrix)
    pix = np.rint(old).astype(int)
    z = previous.depth_mm[np.clip(pix[:, 1], 0, height-1), np.clip(pix[:, 0], 0, width-1)]/1000.
    xyz = np.c_[(old[:, 0]-k[0, 2])*z/k[0, 0], (old[:, 1]-k[1, 2])*z/k[1, 1], z]
    usable = z > 0
    xyz, new, old = xyz[usable], new[usable], old[usable]
    if len(xyz) < 30:
        report['reason'] = 'insufficient_feature_depth'
        return None, None, report
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(xyz, new, k, None, iterationsCount=100,
                                               reprojectionError=2., confidence=.999, flags=cv2.SOLVEPNP_EPNP)
    inliers = np.array([], int) if inliers is None else inliers.ravel()
    cells = np.minimum((old[inliers]/[width, height]*[4, 3]).astype(int), [3, 2])
    cell_count = len(np.unique(cells, axis=0))
    report.update(inlier_count=len(inliers), inlier_fraction=len(inliers)/len(xyz),
                  inlier_grid_cells=cell_count, motion_outlier_count=len(xyz)-len(inliers))
    if not ok or len(inliers) < 30 or len(inliers)/len(xyz) < .6 or cell_count < 6:
        report['reason'] = 'spatial_background_consensus_insufficient'
        return None, None, report
    transform = np.eye(4)
    transform[:3, :3] = cv2.Rodrigues(rvec)[0]
    transform[:3, 3] = tvec.ravel()
    from femto.dataset.scene_dataset_validator import validate_pose
    from femto.tracking.moving_quality import rotation_deg
    if validate_pose(transform) or np.linalg.norm(transform[:3, 3]) > .25 or rotation_deg(transform) > 25:
        report['reason'] = 'background_motion_out_of_bounds'
        return None, None, report
    old_keep = _depth_agreement(previous.depth_mm, current.depth_mm, k, transform)
    new_keep = _depth_agreement(current.depth_mm, previous.depth_mm, k, np.linalg.inv(transform))
    retained = [float(m.sum()/max(1, (d > 0).sum())) for m, d in
                ((old_keep, previous.depth_mm), (new_keep, current.depth_mm))]
    report.update(retained_fractions=retained, seed_T_current_previous=transform.tolist())
    if min(retained) < .55 or min(old_keep.sum(), new_keep.sum()) < 120:
        report['reason'] = 'insufficient_consistent_depth_after_mask'
        return None, None, report
    report.update(applied=True, reason=None)
    return old_keep, new_keep, report


def tracking_pair(previous, current, minimum, maximum):
    masks = consensus(previous, current)
    if not masks[2]['applied']:
        return previous, current, masks[2]
    from femto.tracking.scene_live_tracker import PreparedRGBD
    prepared = []
    for frame, keep in zip((previous, current), masks[:2]):
        depth = np.where(keep, frame.depth_mm, 0).astype(np.uint16)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(frame.color_bgr[..., ::-1])), o3d.geometry.Image(depth),
            depth_scale=1000., depth_trunc=maximum+.0005, convert_rgb_to_intensity=False)
        cloud = o3d.geometry.PointCloud.create_from_depth_image(
            rgbd.depth, frame.intrinsic, depth_scale=1., depth_trunc=maximum+.0005).voxel_down_sample(.02)
        prepared.append(PreparedRGBD(rgbd, frame.intrinsic, cloud, frame.color_bgr, depth))
    return *prepared, masks[2]
