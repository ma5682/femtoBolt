"""Incremental RGB-D pose estimates in the first valid color-camera RDF frame.

No SDK ownership, hardware motion control or IMU integration. Optional
keyframe anchoring and window-limited relocalization only ever re-solve
against keyframes accepted in the same session; the reference frame (W0)
is never reset and a seed transform never becomes a pose by itself.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from dataclasses import dataclass, field
import cv2
import numpy as np
import open3d as o3d

from femto.tracking.moving_quality import POLICY, frame_quality, cycle_consistency, rotation_deg
from femto.dataset.scene_dataset_validator import validate_pose
from femto.geometry.scene_reconstruction import _distortion_maps, _prepare_rgbd


FEATURE_SEED_POLICY = dict(
    detector='ORB', nfeatures=1000, ratio_test=.75, depth_neighborhood_span_mm=20,
    minimum_correspondences=12, minimum_inliers=12, minimum_inlier_fraction=.5,
    reprojection_error_px=2.5, ransac_iterations=500, rng_seed=20260918,
    max_translation_m=.25, max_rotation_deg=25., minimum_grid_cells=4,
    current_depth_tolerance_m=.02, minimum_current_depth_fraction=.5)
RELOCALIZATION_POLICY = dict(window_s=15., confirmations=2, candidate_keyframes=5,
                             agreement_translation_m=POLICY['cycle_translation_m'],
                             agreement_rotation_deg=POLICY['cycle_rotation_deg'])
ANCHOR_POLICY = dict(agreement_translation_m=POLICY['cycle_translation_m'],
                     agreement_rotation_deg=POLICY['cycle_rotation_deg'])


@dataclass
class PreparedRGBD:
    rgbd: object
    intrinsic: object
    cloud: object
    color_bgr: np.ndarray
    depth_mm: np.ndarray
    features: object = field(default=None, repr=False)

    def __iter__(self):
        return iter((self.rgbd, self.intrinsic, self.cloud))


def display_pose(transform):
    """R=Rz(roll) Ry(yaw) Rx(pitch); translation is camera origin in W0."""
    error = validate_pose(np.asarray(transform))
    if error:
        raise ValueError(error)
    r = np.asarray(transform)[:3, :3]
    yaw = np.arcsin(np.clip(-r[2, 0], -1, 1))
    singular = bool(abs(np.cos(yaw)) < 1e-6)
    angles = None if singular else np.degrees([
        np.arctan2(r[2, 1], r[2, 2]), yaw, np.arctan2(r[1, 0], r[0, 0])]).tolist()
    return dict(xyz_mm=(np.asarray(transform)[:3, 3] * 1000).tolist(),
                pitch_yaw_roll_deg=angles, angular_singularity=singular)


class PairEstimator:
    def __init__(self, dataset, minimum_mm, maximum_mm, *, reverse_seed='independent', background_mask=False,
                 feature_seed=False):
        if reverse_seed not in ('independent', 'forward_inverse_experiment'):
            raise ValueError('Unknown reverse initialization strategy')
        self.reverse_seed, self.background_mask = reverse_seed, background_mask
        self.feature_seed = bool(feature_seed)
        self.maps = _distortion_maps(dataset)
        self.minimum, self.maximum = minimum_mm / 1000, maximum_mm / 1000
        self.option = o3d.pipelines.odometry.OdometryOption()
        self.option.iteration_number_per_pyramid_level = o3d.utility.IntVector([30, 20, 10])
        self.option.depth_diff_max = .07
        self.option.depth_min, self.option.depth_max = self.minimum, self.maximum
        self.jacobian = o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()
        mx, my, k, width, height = self.maps
        self.size = (max(1, round(width*.25)), max(1, round(height*.25)))
        # Nearest-neighbour depth resizing and remapping commute when the maps
        # are sampled at exactly the original resize indices. RGB still uses
        # the original full remap + area resize; no change to RGB anti-aliasing.
        self.depth_maps = tuple(cv2.resize(m, self.size, interpolation=cv2.INTER_NEAREST) for m in (mx, my))
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            *self.size, k[0, 0]*.25, k[1, 1]*.25, k[0, 2]*.25, k[1, 2]*.25)

    def prepare(self, frame):
        mx, my, k, width, height = self.maps
        if frame.color_bgr.shape[:2] != (height, width) or frame.depth_mm.shape != (height, width):
            raise ValueError('Frame dimensions changed during local capture')
        depth_u16 = frame.depth_mm if frame.depth_mm.dtype == np.uint16 else np.rint(frame.depth_mm).astype(np.uint16)
        color = cv2.resize(cv2.remap(frame.color_bgr, mx, my, cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT), self.size, interpolation=cv2.INTER_AREA)
        depth = cv2.remap(depth_u16, *self.depth_maps, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        depth[(depth < round(self.minimum*1000)) | (depth > round(self.maximum*1000))] = 0
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(color[..., ::-1])), o3d.geometry.Image(depth),
            depth_scale=1000., depth_trunc=self.maximum+.0005, convert_rgb_to_intensity=False)
        # Registration evaluation uses XYZ only, so it needs no repeated color cloud.
        cloud = o3d.geometry.PointCloud.create_from_depth_image(
            rgbd.depth, self.intrinsic, depth_scale=1., depth_trunc=self.maximum+.0005).voxel_down_sample(.02)
        return PreparedRGBD(rgbd, self.intrinsic, cloud, color, depth)

    def features(self, prepared):
        """Lazily detect ORB features on the tracking-resolution image; cached on the frame."""
        if prepared.features is None:
            gray = cv2.cvtColor(prepared.color_bgr, cv2.COLOR_BGR2GRAY)
            orb = cv2.ORB_create(nfeatures=FEATURE_SEED_POLICY['nfeatures'])
            keypoints, descriptors = orb.detectAndCompute(gray, None)
            prepared.features = (keypoints or (), descriptors)
        return prepared.features

    def feature_seed_transform(self, previous, current):
        """Sparse ORB + raw depth + PnP proposal for T_current_previous, or None.

        The proposal is only ever used as an initial guess for the dense
        solver; it never becomes a pose and it reports its own evidence.
        """
        policy = FEATURE_SEED_POLICY
        report = dict(method='mutual_ORB_previous_depth_EPnP_RANSAC', matches=0, correspondences=0,
                      inliers=0, inlier_fraction=None, reprojection_median_px=None, reason=None)
        (kp_a, desc_a), (kp_b, desc_b) = self.features(previous), self.features(current)
        if desc_a is None or desc_b is None or len(kp_a) < policy['minimum_correspondences'] \
                or len(kp_b) < policy['minimum_correspondences']:
            report['reason'] = 'insufficient_features'
            return None, report
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        forward = [m for m in matcher.knnMatch(desc_a, desc_b, k=2) if len(m) == 2]
        reverse = [m for m in matcher.knnMatch(desc_b, desc_a, k=2) if len(m) == 2]
        reverse_good = {(m.trainIdx, m.queryIdx) for m, n in reverse if m.distance < policy['ratio_test']*n.distance}
        matches = [m for m, n in forward if m.distance < policy['ratio_test']*n.distance
                   and (m.queryIdx, m.trainIdx) in reverse_good]
        report['matches'] = len(matches)
        k = np.asarray(previous.intrinsic.intrinsic_matrix, float)
        depth = previous.depth_mm
        height, width = depth.shape
        span = policy['depth_neighborhood_span_mm']
        low, high = round(self.minimum*1000), round(self.maximum*1000)
        obj, uv, source_uv = [], [], []
        for m in matches:
            x, y = kp_a[m.queryIdx].pt
            ix, iy = int(round(x)), int(round(y))
            if not (1 <= ix < width-1 and 1 <= iy < height-1):
                continue
            neighborhood = depth[iy-1:iy+2, ix-1:ix+2]
            if neighborhood.min() < low or neighborhood.max() > high or np.ptp(neighborhood) > span:
                continue
            z = float(depth[iy, ix])/1000
            obj.append([(x-k[0, 2])*z/k[0, 0], (y-k[1, 2])*z/k[1, 1], z])
            uv.append(kp_b[m.trainIdx].pt); source_uv.append((x,y))
        report['correspondences'] = len(obj)
        if len(obj) < policy['minimum_correspondences']:
            report['reason'] = 'insufficient_depth_supported_correspondences'
            return None, report
        obj, uv = np.asarray(obj, float), np.asarray(uv, float)
        cv2.setRNGSeed(policy['rng_seed'])
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj, uv, k, None, iterationsCount=policy['ransac_iterations'],
            reprojectionError=policy['reprojection_error_px'], confidence=.999, flags=cv2.SOLVEPNP_EPNP)
        inliers = np.array([], int) if inliers is None else inliers.reshape(-1)
        report.update(inliers=int(len(inliers)), inlier_fraction=float(len(inliers)/len(obj)))
        if not ok or len(inliers) < policy['minimum_inliers'] or len(inliers)/len(obj) < policy['minimum_inlier_fraction']:
            report['reason'] = 'insufficient_pnp_inliers'
            return None, report
        rvec, tvec = cv2.solvePnPRefineLM(obj[inliers], uv[inliers], k, None, rvec, tvec)
        transform = np.eye(4)
        transform[:3, :3] = cv2.Rodrigues(rvec)[0]
        transform[:3, 3] = np.asarray(tvec, float).reshape(3)
        projected = cv2.projectPoints(obj[inliers], rvec, tvec, k, None)[0][:, 0]
        report['reprojection_median_px'] = float(np.median(np.linalg.norm(projected-uv[inliers], axis=1)))
        cells=[]
        for points in (np.asarray(source_uv)[inliers],uv[inliers]):
            grid=np.clip((points/[width,height]*[4,3]).astype(int),[0,0],[3,2])
            cells.append(len(np.unique(grid,axis=0)))
        moved=obj[inliers]@transform[:3,:3].T+transform[:3,3]
        pixels=np.rint(uv[inliers]).astype(int)
        inside=(pixels[:,0]>=0)&(pixels[:,0]<current.depth_mm.shape[1])&(pixels[:,1]>=0)&(pixels[:,1]<current.depth_mm.shape[0])
        measured=np.zeros(len(pixels))
        measured[inside]=current.depth_mm[pixels[inside,1],pixels[inside,0]]/1000.
        support=inside & (measured>=self.minimum)&(measured<=self.maximum)&(moved[:,2]>0)&(np.abs(measured-moved[:,2])<=policy['current_depth_tolerance_m'])
        report.update(inlier_grid_cells=cells,current_depth_support=int(support.sum()),current_depth_fraction=float(support.mean()))
        if (min(cells)<policy['minimum_grid_cells'] or support.sum()<policy['minimum_inliers']
                or support.mean()<policy['minimum_current_depth_fraction']
                or not np.isfinite(report['reprojection_median_px'])
                or report['reprojection_median_px']>policy['reprojection_error_px']):
            report['reason']='insufficient_distributed_depth_evidence'
            return None,report
        if (validate_pose(transform) or np.linalg.norm(transform[:3, 3]) > policy['max_translation_m']
                or rotation_deg(transform) > policy['max_rotation_deg']
                ):
            report['reason'] = 'seed_out_of_bounds'
            return None, report
        return transform, report

    def _estimate_once(self, previous, current, initial):
        a, intrinsic, cloud_a = previous
        b, _, cloud_b = current
        ok, forward, _ = o3d.pipelines.odometry.compute_rgbd_odometry(
            a, b, intrinsic, initial, self.jacobian, self.option)
        if not ok or validate_pose(forward):
            raise PairEstimationError('forward_pose')
        reverse_initial = np.linalg.inv(initial)
        seed_kind = 'independent'
        if (getattr(self, 'reverse_seed', 'independent') == 'forward_inverse_experiment'
                and np.linalg.norm(forward[:3, 3]) <= .25 and rotation_deg(forward) <= 25):
            reverse_initial = np.linalg.inv(forward)
            seed_kind = 'forward_inverse_experiment'
        ok, reverse, _ = o3d.pipelines.odometry.compute_rgbd_odometry(
            b, a, intrinsic, reverse_initial, self.jacobian, self.option)
        if not ok or validate_pose(reverse):
            raise PairEstimationError('reverse_pose')
        cycle = cycle_consistency(forward, reverse)
        ef = o3d.pipelines.registration.evaluate_registration(cloud_a, cloud_b, .06, forward)
        er = o3d.pipelines.registration.evaluate_registration(cloud_b, cloud_a, .06, reverse)
        # Keep failed metrics too: a generic pair_quality hid the reported fault.
        finite = lambda value: float(value) if np.isfinite(value) else None
        metrics = dict(fitness=finite(ef.fitness), rmse_m=finite(ef.inlier_rmse),
                       reverse_fitness=finite(er.fitness), reverse_rmse_m=finite(er.inlier_rmse),
                       cycle=cycle, step_m=float(np.linalg.norm(forward[:3, 3])),
                       step_deg=rotation_deg(forward),
                       reverse_step_m=float(np.linalg.norm(reverse[:3, 3])), reverse_step_deg=rotation_deg(reverse),
                       reverse_initialization=seed_kind)
        checks = []
        if not cycle['passed']:
            checks.append('cycle_consistency')
        if not np.isfinite([ef.fitness, er.fitness]).all() or min(ef.fitness, er.fitness) < .15:
            checks.append('fitness')
        if not np.isfinite([ef.inlier_rmse, er.inlier_rmse]).all() or max(ef.inlier_rmse, er.inlier_rmse) > .06:
            checks.append('rmse')
        if metrics['step_m'] > .25:
            checks.append('translation_step')
        if metrics['step_deg'] > 25:
            checks.append('rotation_step')
        if metrics['reverse_step_m'] > .25:
            checks.append('reverse_translation_step')
        if metrics['reverse_step_deg'] > 25:
            checks.append('reverse_rotation_step')
        metrics['rejected_checks'] = checks
        if checks:
            raise PairEstimationError('pair_quality', metrics)
        return forward, metrics

    def estimate(self, previous, current, *, initial_transform=None, initial_name='previous_valid_step',
                 allow_feature_seed=True):
        """Bounded retries from verified evidence only, within the live chain.

        Seeds are never pose results: identity, then the caller's verified seed
        (last step, chain prediction or a sparse proposal), then (only if both
        failed and enabled) a sparse ORB/PnP proposal. Every attempt solves
        both directions from RGB-D again and must pass the unchanged gates. No
        reference reset is performed here.
        """
        background = None
        if getattr(self, 'background_mask', False):
            from femto.tracking.local_tracking_background import tracking_pair
            previous, current, background = tracking_pair(previous, current, self.minimum, self.maximum)
            if not background['applied']:
                raise PairEstimationError('background_support_insufficient', dict(background=background))
        guesses = [('identity', np.eye(4))]
        if initial_transform is not None:
            seed = np.asarray(initial_transform, dtype=float)
            if (not validate_pose(seed) and np.linalg.norm(seed[:3, 3]) <= .25
                    and rotation_deg(seed) <= 25 and not np.allclose(seed, np.eye(4), rtol=0, atol=1e-9)):
                guesses.append((initial_name, seed.copy()))
        if initial_name=='feature_pnp' and len(guesses)>1:
            guesses.reverse()  # Relocalization starts with its independently proposed seed.
        attempts = []
        feature_report = None
        pending = list(guesses)
        while pending:
            initialization, initial = pending.pop(0)
            started = time.perf_counter()
            try:
                forward, metrics = self._estimate_once(previous, current, initial)
            except PairEstimationError as error:
                attempts.append(dict(initialization=initialization, passed=False, reason=str(error),
                                     metrics=error.metrics, processing_ms=(time.perf_counter()-started)*1000))
                if (not pending and feature_report is None and allow_feature_seed
                        and getattr(self, 'feature_seed', False)
                        and isinstance(previous, PreparedRGBD) and isinstance(current, PreparedRGBD)):
                    seed_started = time.perf_counter()
                    seed, feature_report = self.feature_seed_transform(previous, current)
                    feature_report['processing_ms'] = (time.perf_counter()-seed_started)*1000
                    # An identity feature proposal would only repeat the first
                    # dense attempt in ordinary frame-to-frame tracking.
                    if seed is not None and not np.allclose(seed, np.eye(4), rtol=0, atol=1e-9):
                        pending.append(('feature_pnp', seed))
                continue
            attempts.append(dict(initialization=initialization, passed=True,
                                 processing_ms=(time.perf_counter()-started)*1000))
            if background is not None:
                metrics['background'] = background
            if feature_report is not None:
                metrics['feature_seed'] = feature_report
            return forward, dict(metrics, initialization=initialization, attempts=attempts)
        last = attempts[-1]
        if background is not None:
            last['metrics']['background'] = background
        if feature_report is not None:
            last['metrics']['feature_seed'] = feature_report
        raise PairEstimationError(last['reason'], last['metrics'], attempts)


class PairEstimationError(ValueError):
    def __init__(self, reason, metrics=None, attempts=None):
        super().__init__(reason)
        self.metrics = metrics or {}
        self.attempts = attempts or []


MAX_KEYFRAMES = 128  # Matches local_frame_selection.SELECTION_POLICY['max_views'].


class LiveTracker:
    def __init__(self, estimator, minimum_mm, maximum_mm, *, max_gap_s=2.0, clock=time.monotonic,
                 keyframe_anchor=False, relocalization=False,
                 relocalization_window_s=RELOCALIZATION_POLICY['window_s'],
                 relocalization_confirmations=RELOCALIZATION_POLICY['confirmations']):
        self.estimator, self.minimum, self.maximum = estimator, minimum_mm, maximum_mm
        self.max_gap_s, self.clock = max_gap_s, clock
        self.pose = np.eye(4)
        self.previous = None
        self.reference_id = None
        self.last_valid_received = None
        self.last_timestamp = None
        self.last_valid_frame_id = None
        self.previous_step = None
        self.last_rejection = None
        self.loss_context = None
        self.failures = 0
        self.lost = False
        self.keyframe_anchor = bool(keyframe_anchor)
        self.relocalization = bool(relocalization)
        self.relocalization_window_s = float(relocalization_window_s)
        if not np.isfinite(self.relocalization_window_s) or not 0<self.relocalization_window_s<=15.:
            raise ValueError('Relocalization window must be in (0,15] seconds')
        if type(relocalization_confirmations) is not int or not 2<=relocalization_confirmations<=5:
            raise ValueError('Relocalization requires 2..5 confirmations')
        self.relocalization_confirmations = relocalization_confirmations
        self.last_relocalization_timestamp = None
        self.keyframes = []
        self.relocalizing_since = None
        self.relocalization_context = None
        self.relocalization_candidates = []
        self.relocalization_events = []
        self.lost_frames = 0

    @property
    def relocalizing(self):
        return self.relocalizing_since is not None and not self.lost

    def add_keyframe(self, prepared, pose, frame_id):
        """Remember an accepted view for anchoring/relocalization; never changes W0."""
        if (not (self.keyframe_anchor or self.relocalization) or self.lost or self.relocalizing
                or prepared is not self.previous or frame_id != self.last_valid_frame_id
                or any(k['frame_id']==frame_id for k in self.keyframes)
                or not isinstance(prepared, PreparedRGBD) or validate_pose(pose)
                or not np.allclose(np.asarray(pose),self.pose,rtol=0,atol=1e-12)):
            return False
        self.keyframes.append(dict(frame_id=frame_id, pose=np.asarray(pose, float).copy(), prepared=prepared))
        del self.keyframes[:-MAX_KEYFRAMES]
        return True

    def options(self):
        return dict(keyframe_anchor=self.keyframe_anchor, relocalization=self.relocalization,
                    relocalization_window_s=self.relocalization_window_s,
                    relocalization_confirmations=self.relocalization_confirmations,
                    feature_seed=bool(getattr(self.estimator, 'feature_seed', False)),
                    candidate_keyframes=RELOCALIZATION_POLICY['candidate_keyframes'],
                    agreement_translation_m=RELOCALIZATION_POLICY['agreement_translation_m'],
                    agreement_rotation_deg=RELOCALIZATION_POLICY['agreement_rotation_deg'],
                    keyframes_remembered=len(self.keyframes))

    def relocalization_remaining_s(self, now):
        if self.relocalizing_since is None or self.lost:
            return None
        return max(0., self.relocalization_window_s-(now-self.relocalizing_since))

    def _loss_context(self, reason, received=None, frame_id=None):
        clock_now = self.clock()
        now = clock_now if received is None else received
        return dict(reason=reason, frame_id=frame_id,
            last_valid_frame_id=self.last_valid_frame_id, reference_frame_id=self.reference_id,
            frame_age_s=max(0., clock_now-received) if received is not None else None,
            gap_since_last_valid_s=(max(0., now-self.last_valid_received)
                                   if self.last_valid_received is not None else None),
            consecutive_failures=self.failures, preceding_rejection=self.last_rejection,
            max_gap_s=self.max_gap_s, max_tracking_gap_frames=POLICY['max_tracking_gap_frames'])

    def mark_lost(self, reason, *, received=None, frame_id=None):
        """Latch the original loss cause, including watchdogs with no RGB-D row."""
        self.lost = True
        if self.loss_context is None:
            self.loss_context = self._loss_context(reason, received, frame_id)
            if self.relocalization_context is not None:
                self.loss_context.update(original_reason=self.relocalization_context['reason'],
                    original_frame_id=self.relocalization_context['frame_id'],
                    relocalization=dict(window_s=self.relocalization_window_s, frames_attempted=self.lost_frames,
                                        confirmations_reached=len(self.relocalization_candidates),
                                        required_confirmations=self.relocalization_confirmations))
                self.relocalization_events.append(dict(event='failed', reason=reason, frame_id=frame_id,
                    frames_attempted=self.lost_frames, original_reason=self.relocalization_context['reason']))
        return self.loss_context

    def watchdog(self, reason='tracking_gap'):
        now=self.clock();record=dict(T_WC=None,reference_frame_id=self.reference_id)
        if self.relocalizing:
            if self.relocalization_remaining_s(now)<=0:record['loss_context']=self.mark_lost('relocalization_timeout')
        elif not self.lost:self._enter_loss(reason,now,None,record)
        record.update(state='lost' if self.lost else 'relocalizing',rejection_reason='relocalization_timeout' if self.lost and self.relocalization_context else reason,
            continuity=dict(current_frame_verified=False,reference_reset=False,remaining_window_s=self.relocalization_remaining_s(now)))
        return self._finish_record(record,now)

    def _enter_loss(self, reason, received, frame_id, record):
        """Latch loss (default) or open the relocalization window if enabled and keyframes exist."""
        if self.relocalization and self.keyframes and self.relocalizing_since is None:
            self.relocalizing_since = self.clock()
            self.last_relocalization_timestamp = self.last_timestamp
            self.lost_frames = 0
            self.relocalization_candidates = []
            self.relocalization_context = self._loss_context(reason, received, frame_id)
            self.relocalization_events.append(dict(event='window_opened', reason=reason, frame_id=frame_id,
                last_valid_frame_id=self.last_valid_frame_id, window_s=self.relocalization_window_s,
                candidate_keyframes=[k['frame_id'] for k in self.keyframes[-RELOCALIZATION_POLICY['candidate_keyframes']:]]))
            record['loss_context'] = dict(self.relocalization_context, relocalization_window_opened=True)
            return
        record['loss_context'] = self.mark_lost(reason, received=received, frame_id=frame_id)

    def _anchor(self, current, candidate_frame):
        """Cross-check the frame-to-frame candidate against the latest accepted keyframe."""
        keyframe = self.keyframes[-1]
        anchor = dict(keyframe_id=keyframe['frame_id'], used=False, reason=None,
                      disagreement_mm=None, disagreement_deg=None)
        if keyframe['prepared'] is self.previous:
            anchor['reason'] = 'keyframe_is_previous_frame'
            return candidate_frame, anchor
        # Predicted T_current_keyframe from the chained candidate; a seed only.
        seed = np.linalg.inv(candidate_frame) @ keyframe['pose']
        try:
            forward, metrics = self.estimator.estimate(keyframe['prepared'], current, initial_transform=seed,
                                                       initial_name='chain_prediction', allow_feature_seed=False)
        except (PairEstimationError, ValueError, RuntimeError) as error:
            anchor['reason'] = str(error)
            anchor['pair_failure'] = getattr(error, 'metrics', None)
            return candidate_frame, anchor
        candidate_keyframe = keyframe['pose'] @ np.linalg.inv(forward)
        if validate_pose(candidate_keyframe):
            anchor['reason'] = 'invalid_composed_pose'
            return candidate_frame, anchor
        delta = np.linalg.inv(candidate_frame) @ candidate_keyframe
        translation, rotation = float(np.linalg.norm(delta[:3, 3])), rotation_deg(delta)
        anchor.update(disagreement_mm=translation*1000, disagreement_deg=rotation,
                      pair=dict(fitness=metrics.get('fitness'), rmse_m=metrics.get('rmse_m'),
                                initialization=metrics.get('initialization')))
        if (translation > ANCHOR_POLICY['agreement_translation_m']
                or rotation > ANCHOR_POLICY['agreement_rotation_deg']):
            anchor['reason'] = 'anchor_disagreement'
            raise AnchorDisagreement(anchor)
        anchor['used'] = True
        return candidate_keyframe, anchor

    def _relocalize_once(self, current):
        """One evidence-gated solve against the best-matching recent keyframe."""
        policy = RELOCALIZATION_POLICY
        reports, best, hypotheses = [], None, []
        def check_budget():
            if self.relocalization_remaining_s(self.clock())<=0:raise ValueError('relocalization_timeout')
        check_budget()
        for keyframe in reversed(self.keyframes[-policy['candidate_keyframes']:]):
            check_budget()
            seed, report = self.estimator.feature_seed_transform(keyframe['prepared'], current)
            if seed is not None:hypotheses.append(keyframe['pose']@np.linalg.inv(seed))
            reports.append(dict(keyframe_id=keyframe['frame_id'], **report))
            if seed is not None and (best is None or report['inliers'] > best[2]['inliers']):
                best = (keyframe, seed, report)
        if best is None:
            raise PairEstimationError('relocalization_no_feature_support', dict(keyframe_reports=reports))
        keyframe, seed, report = best
        if not self._candidates_consistent([dict(candidate=p) for p in hypotheses]):
            raise PairEstimationError('relocalization_ambiguous',dict(keyframe_reports=reports))
        check_budget()
        forward, metrics = self.estimator.estimate(keyframe['prepared'], current, initial_transform=seed,
                                                   initial_name='feature_pnp', allow_feature_seed=False)
        check_budget()
        candidate = keyframe['pose'] @ np.linalg.inv(forward)
        if not self._candidates_consistent([dict(candidate=keyframe['pose']@np.linalg.inv(seed)),dict(candidate=candidate)]):
            raise PairEstimationError('relocalization_seed_disagreement',dict(keyframe_reports=reports,pair=metrics))
        if validate_pose(candidate):
            raise ValueError('invalid_composed_pose')
        return dict(keyframe_id=keyframe['frame_id'], candidate=candidate, seed_report=report,
                    keyframe_reports=reports, pair=metrics)

    @staticmethod
    def _candidates_consistent(candidates):
        policy = RELOCALIZATION_POLICY
        for a, b in ((a,b) for i,a in enumerate(candidates) for b in candidates[i+1:]):
            delta = np.linalg.inv(a['candidate']) @ b['candidate']
            if (np.linalg.norm(delta[:3, 3]) > policy['agreement_translation_m']
                    or rotation_deg(delta) > policy['agreement_rotation_deg']):
                return False
        return True

    def _process_relocalizing(self, frame, frame_id, received, record, quality):
        self.lost_frames += 1
        context = self.relocalization_context
        reason = None
        if self.relocalization_remaining_s(self.clock())<=0:
            reason = 'relocalization_timeout'
            record['loss_context'] = self.mark_lost(reason, received=received, frame_id=frame_id)
        elif not 0<=self.clock()-received<=self.max_gap_s:
            reason = 'stale_frame'
        elif self.last_relocalization_timestamp is not None and frame.timestamp_us <= self.last_relocalization_timestamp:
            reason = 'nonmonotonic_timestamp'
        elif abs(frame.color_timestamp_us-frame.depth_timestamp_us) > 5000:
            reason = 'rgb_depth_sync'
        else:
            from femto.tracking.local_tracking_diagnostics import image_diagnostics
            record['image_diagnostics'] = image_diagnostics(frame.color_bgr, frame.depth_mm, self.minimum, self.maximum)
            if not quality['usable']:
                reason = quality['state']
        self.last_relocalization_timestamp=max(frame.timestamp_us,self.last_relocalization_timestamp or frame.timestamp_us)
        if self.relocalization_candidates and (received-self.relocalization_candidates[-1]['received']>self.max_gap_s
                or frame.timestamp_us<=self.relocalization_candidates[-1]['capture_timestamp_us']):self.relocalization_candidates=[]
        if reason is None:
            try:
                prepare_started = time.perf_counter()
                current = self.estimator.prepare(frame)
                record['prepare_ms'] = (time.perf_counter()-prepare_started)*1000
                attempt = self._relocalize_once(current)
                record['relocalization_attempt'] = dict(keyframe_id=attempt['keyframe_id'],
                    seed=attempt['seed_report'], keyframe_reports=attempt['keyframe_reports'],
                    pair=dict(fitness=attempt['pair'].get('fitness'), rmse_m=attempt['pair'].get('rmse_m'),
                              cycle=attempt['pair'].get('cycle'), initialization=attempt['pair'].get('initialization')))
                if self.relocalization_remaining_s(self.clock())<=0:raise ValueError('relocalization_timeout')
                if self.clock()-received > self.max_gap_s:
                    raise ValueError('processing_stale')
                self.relocalization_candidates.append(dict(frame_id=frame_id, received=received,capture_timestamp_us=frame.timestamp_us,candidate=attempt['candidate'],
                                                           keyframe_id=attempt['keyframe_id']))
                recent = self.relocalization_candidates[-self.relocalization_confirmations:]
                consistent = self._candidates_consistent(recent)
                if not consistent:
                    self.relocalization_candidates = [self.relocalization_candidates[-1]]
                    recent = self.relocalization_candidates
                if len(recent) >= self.relocalization_confirmations:
                    frames_lost = self.lost_frames
                    self.pose = attempt['candidate']
                    self.previous = current
                    self.previous_step = None
                    self.last_valid_received = received
                    self.last_timestamp = frame.timestamp_us
                    self.last_valid_frame_id = frame_id
                    self.failures = 0
                    self.last_rejection = None
                    self.relocalizing_since = None
                    self.relocalization_candidates = []
                    self.relocalization_context = None
                    record.update(state='tracking', T_WC=self.pose.tolist(),
                                  reference_frame_id=self.reference_id, **display_pose(self.pose))
                    record['motion_discontinuity']=True
                    record['continuity'] = dict(phase='relocalized', preceding_rejected_frames=frames_lost,
                        relocalized=True, relocalized_to_keyframe=attempt['keyframe_id'], frames_lost=frames_lost,
                        confirmations=len(recent), reference_reset=False, current_frame_verified=True,
                        original_loss_reason=context['reason'])
                    self.relocalization_events.append(dict(event='recovered', frame_id=frame_id,
                        keyframe_id=attempt['keyframe_id'], frames_lost=frames_lost,
                        confirmation_frame_ids=[c['frame_id'] for c in recent],
                        original_reason=context['reason'], original_frame_id=context['frame_id']))
                else:
                    reason = 'relocalization_pending'
            except (ValueError, RuntimeError) as exc:
                reason = str(exc)
                self.relocalization_candidates = []
                if isinstance(exc, PairEstimationError):
                    record.update(pair_failure=exc.metrics, pair_attempts=exc.attempts)
        if reason:
            if reason!='relocalization_pending':self.relocalization_candidates=[]
            if reason=='relocalization_timeout':record['loss_context']=self.mark_lost(reason,received=received,frame_id=frame_id)
            self.last_rejection = dict(frame_id=frame_id, reason=reason, pair_failure=record.get('pair_failure'))
            record.update(state='lost' if self.lost else 'relocalizing', rejection_reason=reason)
            record['continuity'] = dict(phase='lost' if self.lost else 'relocalizing',
                accepted_for_fusion=False, relocalized=False, reference_reset=False,
                current_frame_verified=False, max_gap_s=self.max_gap_s,
                remaining_window_s=None if self.lost else self.relocalization_remaining_s(self.clock()))
        record['relocalization'] = dict(window_s=self.relocalization_window_s, frames_attempted=self.lost_frames,
            confirmations=len(self.relocalization_candidates), required_confirmations=self.relocalization_confirmations,
            original_reason=context['reason'], original_frame_id=context['frame_id'],
            remaining_window_s=None if self.lost else self.relocalization_remaining_s(self.clock()))
        return record

    def process(self, frame, frame_id, received):
        record = dict(frame_id=frame_id, camera_frame_index=int(frame.frame_index),
            capture_timestamp_us=int(frame.timestamp_us), received_monotonic=received,
            color_timestamp_us=int(frame.color_timestamp_us), depth_timestamp_us=int(frame.depth_timestamp_us),
            reference_frame_id=self.reference_id, T_WC=None, state='uncertain',
            transform_direction='current_color_camera_to_initial_color_camera', unit='metre')
        # Record full-frame quality even when synchronization or stale-pose gates
        # reject before pose estimation. Never reuse the last good frame's value.
        quality_started = time.perf_counter()
        quality = frame_quality(frame.color_bgr, frame.depth_mm, self.minimum, self.maximum)
        record['quality'] = quality
        record['quality_processing_ms'] = (time.perf_counter()-quality_started)*1000
        if self.relocalizing:
            record = self._process_relocalizing(frame, frame_id, received, record, quality)
            return self._finish_record(record, received)
        reason = None
        loss_trigger = None
        if self.lost:
            reason = 'tracking_lost'
        elif self.clock() - received > self.max_gap_s:
            reason = loss_trigger = 'stale_frame'
        elif self.last_valid_received is not None and received-self.last_valid_received > self.max_gap_s:
            reason = loss_trigger = 'tracking_gap'
        elif self.last_timestamp is not None and frame.timestamp_us <= self.last_timestamp:
            reason = 'nonmonotonic_timestamp'
        elif abs(frame.color_timestamp_us-frame.depth_timestamp_us) > 5000:
            reason = 'rgb_depth_sync'
        else:
            from femto.tracking.local_tracking_diagnostics import image_diagnostics
            record['image_diagnostics'] = image_diagnostics(frame.color_bgr, frame.depth_mm, self.minimum, self.maximum)
            if not quality['usable']:
                reason = quality['state']
        if reason is None:
            try:
                prepare_started = time.perf_counter()
                current = self.estimator.prepare(frame)
                record['prepare_ms'] = (time.perf_counter()-prepare_started)*1000
                candidate = np.eye(4)
                if self.previous is not None:
                    forward, metrics = self.estimator.estimate(self.previous, current,
                                                               initial_transform=self.previous_step)
                    candidate = self.pose @ np.linalg.inv(forward)
                    record['pair'] = metrics
                    record['step_m'] = float(np.linalg.norm(forward[:3, 3]))
                    record['step_deg'] = rotation_deg(forward)
                    if self.keyframe_anchor and self.keyframes and not validate_pose(candidate):
                        try:
                            candidate, record['anchor'] = self._anchor(current, candidate)
                            adopted=np.linalg.inv(self.pose)@candidate
                            record['frame_pair_step_m']=record['step_m'];record['frame_pair_step_deg']=record['step_deg']
                            record['step_m']=float(np.linalg.norm(adopted[:3,3]));record['step_deg']=rotation_deg(adopted)
                            record['motion_discontinuity']=bool(record['anchor']['used'] and
                                (record['anchor']['disagreement_mm']>1. or record['anchor']['disagreement_deg']>.2))
                        except AnchorDisagreement as disagreement:
                            record['anchor'] = disagreement.anchor
                            raise ValueError('anchor_disagreement')
                if validate_pose(candidate):raise ValueError('invalid_composed_pose')
                if self.clock()-received > self.max_gap_s:
                    reason = loss_trigger = 'processing_stale'
                else:
                    previous_failures = self.failures
                    self.pose = candidate
                    if self.reference_id is None:
                        self.reference_id = frame_id
                    self.previous = current
                    if 'pair' in record:
                        self.previous_step = forward.copy()
                    self.last_valid_received = received
                    self.last_timestamp = frame.timestamp_us
                    self.last_valid_frame_id = frame_id
                    self.failures = 0
                    self.last_rejection = None
                    record.update(state='tracking', T_WC=self.pose.tolist(),
                                  reference_frame_id=self.reference_id, **display_pose(self.pose))
                    record['continuity'] = dict(phase='revalidated' if previous_failures else 'tracking',
                        preceding_rejected_frames=previous_failures, relocalized=False,
                        reference_reset=False, current_frame_verified=True)
            except (ValueError, RuntimeError) as exc:
                reason = str(exc)
                if isinstance(exc, PairEstimationError):
                    record.update(pair_failure=exc.metrics, pair_attempts=exc.attempts)
        if reason:
            self.failures += 1
            if self.reference_id is not None and self.failures >= POLICY['max_tracking_gap_frames']:
                loss_trigger = loss_trigger or reason
            if loss_trigger and not self.lost:
                self._enter_loss(loss_trigger, received, frame_id, record)
            elif self.lost and 'loss_context' not in record:
                record['loss_context'] = self.mark_lost(reason, received=received, frame_id=frame_id)
            self.last_rejection = dict(frame_id=frame_id, reason=reason,
                                       pair_failure=record.get('pair_failure'))
            if self.lost and record.get('pair_failure'):
                self.loss_context['current_pair_failure'] = record['pair_failure']
            relocalizing = self.relocalizing
            record.update(state='lost' if self.lost else 'relocalizing' if relocalizing else 'uncertain',
                          rejection_reason=reason)
            record['continuity'] = dict(phase='lost' if self.lost else 'relocalizing' if relocalizing else 'short_instability',
                accepted_for_fusion=False, relocalized=False, reference_reset=False,
                current_frame_verified=False, max_gap_s=self.max_gap_s,
                remaining_gap_s=max(0., self.max_gap_s-(self.clock()-self.last_valid_received))
                    if self.last_valid_received is not None else None)
            if relocalizing:
                record['continuity']['remaining_window_s'] = self.relocalization_remaining_s(self.clock())
        return self._finish_record(record, received)

    def _finish_record(self, record, received):
        record['consecutive_failures'] = self.failures
        record['last_valid_frame_id'] = self.last_valid_frame_id
        record['status_updated_at'] = datetime.now(timezone.utc).isoformat()
        record['status_updated_monotonic'] = self.clock()
        record['latency_ms'] = (self.clock()-received)*1000
        return record


class AnchorDisagreement(ValueError):
    def __init__(self, anchor):
        super().__init__('anchor_disagreement')
        self.anchor = anchor
