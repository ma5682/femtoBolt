"""Estimate research-grade sequential RGB-D camera poses without mutating raw capture."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any

import cv2
import numpy as np
import open3d as o3d

from femto.io_utils import imread_unicode as _read_image, load_json, load_jsonl
from femto.tracking.moving_quality import POLICY, frame_quality, cycle_consistency, select_keyframes
from femto.dataset.scene_dataset_validator import resolve_dataset, validate_pose
from femto.geometry.rgbd_frames import distortion_maps as _distortion_maps, prepare_rgbd as _prepare_rgbd
from femto.tracking.moving_quality import pose_diversity


from femto.paths import REPO_ROOT as ROOT
DEFAULT_DATASETS = ROOT / "datasets"
BACKGROUND_HOMOGRAPHY_INLIER_MIN = 0.85
FIXED_BACKGROUND_DISPLACEMENT_MAX_PX = 3.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate T_WC using sequential hybrid RGB-D odometry")
    parser.add_argument("dataset", nargs="?", type=Path)
    parser.add_argument("--datasets-root", type=Path, default=DEFAULT_DATASETS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--image-scale", type=float, default=0.25)
    parser.add_argument("--min-fitness", type=float, default=0.15)
    parser.add_argument("--max-rmse-m", type=float, default=0.06)
    parser.add_argument("--max-step-m", type=float, default=0.25)
    parser.add_argument("--max-step-deg", type=float, default=25.0)
    return parser.parse_args()


def _rotation_angle_deg(rotation: np.ndarray) -> float:
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _relative_path(target: Path, start: Path) -> str:
    return Path(os.path.relpath(target, start=start)).as_posix()


def assess_rgb_motion_pair(
    first: np.ndarray,
    last: np.ndarray,
    *,
    scale: float = 0.25,
) -> dict[str, Any]:
    """Assess one RGB pair for capture feedback and the dataset motion gate.

    This is deliberately a camera-motion *hint*, not a pose estimator.  Both the
    live UI and the persisted dataset audit call this function so their optical
    flow/homography semantics cannot drift apart.
    """
    if first.ndim == 3:
        first = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    if last.ndim == 3:
        last = cv2.cvtColor(last, cv2.COLOR_BGR2GRAY)
    first = cv2.resize(first, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    last = cv2.resize(last, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    features = cv2.goodFeaturesToTrack(
        first, maxCorners=1200, qualityLevel=0.01, minDistance=7, blockSize=7
    )
    if features is None or len(features) < 30:
        return {"reliable": False, "appears_fixed": None, "reason": "too few RGB features"}
    tracked, status, error = cv2.calcOpticalFlowPyrLK(
        first,
        last,
        features,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if tracked is None or status is None or error is None:
        return {"reliable": False, "appears_fixed": None, "reason": "optical flow failed"}
    valid = status.reshape(-1).astype(bool)
    source = features.reshape(-1, 2)[valid]
    target = tracked.reshape(-1, 2)[valid]
    errors = error.reshape(-1)[valid]
    if len(source) < 30:
        return {"reliable": False, "appears_fixed": None, "reason": "too few tracked RGB features"}
    threshold = float(np.quantile(errors, 0.80))
    keep = errors <= threshold
    source, target = source[keep], target[keep]
    if len(source) < 30:
        return {"reliable": False, "appears_fixed": None, "reason": "too few low-error RGB tracks"}
    _, inliers = cv2.findHomography(source, target, cv2.RANSAC, 2.0)
    if inliers is None:
        return {"reliable": False, "appears_fixed": None, "reason": "background homography failed"}
    inlier_mask = inliers.reshape(-1).astype(bool)
    if not np.any(inlier_mask):
        return {"reliable": False, "appears_fixed": None, "reason": "background homography has no inliers"}
    inlier_ratio = float(np.mean(inlier_mask))
    displacement = np.linalg.norm(target - source, axis=1) / scale
    median_displacement = float(np.median(displacement[inlier_mask]))
    metrics = {
        "tracked_features": int(len(source)),
        "homography_inlier_ratio": inlier_ratio,
        "median_background_displacement_px": median_displacement,
    }
    # A small RANSAC consensus can be produced by unrelated images, dynamic
    # foregrounds or lost overlap.  It is not positive evidence that the
    # camera moved.  Reuse the existing 0.85 consensus boundary as a
    # reliability requirement; inconclusive pairs remain auditable and the
    # downstream RGB-D odometry still decides whether usable poses exist.
    if inlier_ratio < BACKGROUND_HOMOGRAPHY_INLIER_MIN:
        return {
            **metrics,
            "reliable": False,
            "appears_fixed": None,
            "reason": "background homography inlier ratio is too low for a reliable motion hint",
        }
    return {
        **metrics,
        "reliable": True,
        "appears_fixed": bool(median_displacement < FIXED_BACKGROUND_DISPLACEMENT_MAX_PX),
    }


def _motion_sample_pairs(frame_count: int) -> list[tuple[int, int]]:
    """Return every adjacent pair plus the endpoint compatibility pair.

    Inspecting all adjacent saved frames prevents a brief excursion followed
    by a return to the same camera position from falling between sparse sample
    ordinals.  The endpoint pair is retained for compatibility and audit value.
    """
    if frame_count < 2:
        return []
    pairs = [(index, index + 1) for index in range(frame_count - 1)]
    endpoint_pair = (0, frame_count - 1)
    if endpoint_pair not in pairs:
        pairs.append(endpoint_pair)
    return pairs


def assess_background_motion(
    dataset: Path,
    frames: list[dict[str, Any]],
    *,
    scale: float = 0.25,
) -> dict[str, Any]:
    """Audit every adjacent RGB pair so a return cannot hide mid-capture motion."""
    pairs = _motion_sample_pairs(len(frames))
    if not pairs:
        return {
            "reliable": False,
            "appears_fixed": None,
            "reason": "fewer than two frames",
            "pair_details": [],
        }

    image_cache: dict[int, np.ndarray] = {}
    details: list[dict[str, Any]] = []
    for source_index, target_index in pairs:
        if source_index not in image_cache:
            image_cache[source_index] = _read_image(
                dataset / frames[source_index]["rgb"], cv2.IMREAD_GRAYSCALE
            )
        if target_index not in image_cache:
            image_cache[target_index] = _read_image(
                dataset / frames[target_index]["rgb"], cv2.IMREAD_GRAYSCALE
            )
        pair = assess_rgb_motion_pair(
            image_cache[source_index], image_cache[target_index], scale=scale
        )
        details.append(
            {
                "source_frame_ordinal": source_index,
                "target_frame_ordinal": target_index,
                "source_frame": frames[source_index].get("frame", source_index),
                "target_frame": frames[target_index].get("frame", target_index),
                # Keep every scheduled pair on one stable audit schema.  Failed
                # optical-flow/homography pairs cannot supply these metrics,
                # so JSON null is more truthful than omitting the fields or
                # manufacturing a numeric value.
                "tracked_features": None,
                "homography_inlier_ratio": None,
                "median_background_displacement_px": None,
                **pair,
            }
        )

    reliable = [item for item in details if item.get("reliable")]
    minimum_reliable_pairs = 1 if len(details) == 1 else 2
    if len(reliable) < minimum_reliable_pairs:
        return {
            "reliable": False,
            "appears_fixed": None,
            "reason": "too few reliable scheduled RGB pairs",
            "sampled_frame_ordinals": sorted({index for pair in pairs for index in pair}),
            "pair_count": len(details),
            "reliable_pair_count": len(reliable),
            "pair_details": details,
        }

    displacements = [float(item["median_background_displacement_px"]) for item in reliable]
    inlier_ratios = [float(item["homography_inlier_ratio"]) for item in reliable]
    fixed_pair_count = sum(bool(item["appears_fixed"]) for item in reliable)
    moving_pair_count = len(reliable) - fixed_pair_count
    all_pairs_reliable = len(reliable) == len(details)
    # A fixed-camera blocker needs positive evidence from every scheduled pair.
    # One reliable moving pair is sufficient to refute "fixed"; however, fixed
    # endpoint/adjacent pairs must not hide an unreliable blurred excursion.
    if moving_pair_count:
        assessment_reliable = True
        appears_fixed: bool | None = False
        reason = None
    elif all_pairs_reliable:
        assessment_reliable = True
        appears_fixed = True
        reason = None
    else:
        assessment_reliable = False
        appears_fixed = None
        reason = "not all scheduled RGB pairs were reliable; cannot declare camera fixed"
    report = {
        "reliable": assessment_reliable,
        "appears_fixed": appears_fixed,
        "sampled_frame_ordinals": sorted({index for pair in pairs for index in pair}),
        "pair_count": len(details),
        "reliable_pair_count": len(reliable),
        "unreliable_pair_count": len(details) - len(reliable),
        "all_pairs_reliable": all_pairs_reliable,
        "fixed_pair_count": fixed_pair_count,
        "fixed_pair_ratio": float(fixed_pair_count / len(reliable)),
        "tracked_features": int(np.median([item["tracked_features"] for item in reliable])),
        "homography_inlier_ratio": float(np.median(inlier_ratios)),
        # Kept for report/schema compatibility.  The maximum is also recorded
        # because it is the clearest evidence of an excursion in a loop path.
        "median_background_displacement_px": float(np.median(displacements)),
        "max_background_displacement_px": float(np.max(displacements)),
        "pair_details": details,
    }
    if reason is not None:
        report["reason"] = reason
    return report


def estimate_poses(
    dataset: Path,
    output: Path,
    *,
    image_scale: float = 0.25,
    min_fitness: float = 0.15,
    max_rmse_m: float = 0.06,
    max_step_m: float = 0.25,
    max_step_deg: float = 25.0,
) -> tuple[Path, dict[str, Any]]:
    """Preserve failure evidence as well as successful poses, without touching raw data."""
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists; refusing to overwrite: {output}")
    evidence: dict[str, Any] = {
        "schema": "femto-rgbd-odometry-report-v1",
        "created_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "source_dataset": str(dataset.resolve()),
        "output_dataset": str(output),
        "status": "BLOCKED",
        "pose_grade": "research_demo",
        "parameters": {"image_scale": image_scale, "min_fitness": min_fitness,
                       "max_rmse_m": max_rmse_m, "max_step_m": max_step_m,
                       "max_step_deg": max_step_deg},
        "limitations": ["Cumulative drift; no loop closure, relocalization or global pose graph optimization."],
    }
    try:
        return _estimate_poses(
            dataset, output, image_scale=image_scale, min_fitness=min_fitness,
            max_rmse_m=max_rmse_m, max_step_m=max_step_m, max_step_deg=max_step_deg,
            failure_evidence=evidence,
        )
    except Exception as error:
        evidence.update(error=f"{type(error).__name__}: {error}", pose_valid=False)
        evidence["accepted_candidate_count"] = len(evidence.get("accepted_frame_ordinals", []))
        evidence["valid_output_pose_count"] = 0
        evidence["rejected_frame_count"] = len(evidence.get("rejected_frames", []))
        output.mkdir(parents=True, exist_ok=True)
        (output / "pose_quality_report.json").write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        try:
            from femto.reports.scene_quality_report import write_report
            write_report(output)
        except Exception as report_error:
            evidence["quality_report_error"] = str(report_error)
            (output / "pose_quality_report.json").write_text(
                json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
        raise


def _estimate_poses(
    dataset: Path, output: Path, *, image_scale: float, min_fitness: float,
    max_rmse_m: float, max_step_m: float, max_step_deg: float,
    failure_evidence: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    dataset = dataset.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists; refusing to overwrite: {output}")
    if not 0.1 <= image_scale <= 0.75:
        raise ValueError("image_scale must be in [0.1, 0.75]")
    source_manifest = load_json(dataset / "manifest.json")
    frames = load_jsonl(dataset / "frames.jsonl")
    failure_evidence["source_frame_count"] = len(frames)
    if source_manifest.get("schema") != "femto-full-scene-rgbd-v1" or len(frames) < 2:
        raise ValueError("Pose estimation needs a femto-full-scene-rgbd-v1 dataset with at least two frames")
    motion = assess_background_motion(dataset, frames)
    failure_evidence["background_motion"] = motion
    appears_fixed = motion.get("appears_fixed")
    if appears_fixed is None and motion.get("reliable", False):
        # Backward-compatible path for callers/tests returning the v1 summary.
        appears_fixed = (
            motion["homography_inlier_ratio"] >= BACKGROUND_HOMOGRAPHY_INLIER_MIN
            and motion["median_background_displacement_px"]
            < FIXED_BACKGROUND_DISPLACEMENT_MAX_PX
        )
    if appears_fixed:
        raise ValueError(
            "Camera appears fixed: background displacement is only "
            f"{motion['median_background_displacement_px']:.2f} px with "
            f"{motion['homography_inlier_ratio'] * 100.0:.1f}% homography inliers. "
            "Choose 固定机位（单视角） in the operation page, or move the camera slowly "
            "with overlapping views for multiview reconstruction."
        )
    parameters = source_manifest.get("parameters", {})
    range_m = source_manifest.get("camera", {}).get("depth_range_m")
    depth_min_m = float(parameters.get("depth_min_mm", 1000.0 * range_m[0] if range_m else 500.0)) / 1000.0
    depth_max_m = float(parameters.get("depth_max_mm", 1000.0 * range_m[1] if range_m else 3860.0)) / 1000.0
    map_x, map_y, K, width, height = _distortion_maps(dataset)

    option = o3d.pipelines.odometry.OdometryOption()
    option.iteration_number_per_pyramid_level = o3d.utility.IntVector([30, 20, 10])
    option.depth_diff_max = 0.07
    option.depth_min = depth_min_m
    option.depth_max = depth_max_m
    jacobian = o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()

    poses = []
    accepted_frames = []
    accepted_ordinals = []
    frame_quality_metrics = []
    failure_evidence["frame_quality"] = frame_quality_metrics
    failure_evidence["moving_policy"] = dict(POLICY)
    pair_metrics: list[dict[str, Any]] = []
    rejected_frames: list[dict[str, Any]] = []
    failure_evidence.update(
        accepted_frame_ordinals=accepted_ordinals, pairs=pair_metrics, rejected_frames=rejected_frames
    )
    previous_rgbd: o3d.geometry.RGBDImage | None = None
    previous_cloud: o3d.geometry.PointCloud | None = None
    intrinsic: o3d.camera.PinholeCameraIntrinsic | None = None
    for ordinal, frame in enumerate(frames):
        color = _read_image(dataset / frame["rgb"], cv2.IMREAD_COLOR)
        depth = _read_image(dataset / frame["depth"], cv2.IMREAD_UNCHANGED)
        if color.shape[:2] != (height, width) or depth.shape != (height, width):
            raise ValueError(f"Frame {frame['frame']} dimensions do not match calibration")
        if accepted_ordinals and ordinal - accepted_ordinals[-1] > POLICY["max_tracking_gap_frames"]:
            raise ValueError("Tracking lost: more than 5 source frames since last valid pose; restart with overlapping views")
        quality = frame_quality(color, depth, depth_min_m * 1000, depth_max_m * 1000)
        frame_quality_metrics.append(dict(frame_ordinal=ordinal, **quality))
        if not quality["usable"]:
            rejected_frames.append(dict(frame_ordinal=ordinal, source_frame=int(frame["frame"]), reason=quality["state"]))
            continue
        current_rgbd, intrinsic, _, _ = _prepare_rgbd(
            color, depth, map_x, map_y, K, image_scale, depth_min_m, depth_max_m
        )
        current_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
            current_rgbd, intrinsic, project_valid_depth_only=True
        ).voxel_down_sample(0.02)
        if previous_rgbd is None:
            poses.append(np.eye(4, dtype=np.float64))
            accepted_frames.append(frame)
            accepted_ordinals.append(ordinal)
            previous_rgbd = current_rgbd
            previous_cloud = current_cloud
            continue
        assert previous_rgbd is not None and previous_cloud is not None and intrinsic is not None
        success, T_current_previous, information = o3d.pipelines.odometry.compute_rgbd_odometry(
            previous_rgbd,
            current_rgbd,
            intrinsic,
            np.eye(4, dtype=np.float64),
            jacobian,
            option,
        )
        transform_error = validate_pose(T_current_previous)
        source_ordinal = accepted_ordinals[-1]
        if not success or transform_error:
            rejected_frames.append(
                {
                    "frame_ordinal": ordinal,
                    "source_frame": int(frame["frame"]),
                    "last_accepted_frame_ordinal": source_ordinal,
                    "reason": transform_error or "insufficient overlap/photometric consistency",
                }
            )
            continue
        step_m = float(np.linalg.norm(T_current_previous[:3, 3]))
        step_deg = _rotation_angle_deg(T_current_previous[:3, :3])
        evaluation = o3d.pipelines.registration.evaluate_registration(
            previous_cloud, current_cloud, max_rmse_m, T_current_previous
        )
        fitness = float(evaluation.fitness)
        rmse_m = float(evaluation.inlier_rmse)
        rejection_reason = None
        if step_m > max_step_m or step_deg > max_step_deg:
            rejection_reason = f"pose jump {step_m:.3f} m, {step_deg:.2f} deg"
        elif fitness < min_fitness or not np.isfinite(rmse_m) or rmse_m > max_rmse_m:
            rejection_reason = f"quality fitness={fitness:.3f}, rmse={rmse_m:.4f} m"
        reverse_success, reverse_transform, _ = o3d.pipelines.odometry.compute_rgbd_odometry(
            current_rgbd, previous_rgbd, intrinsic, np.eye(4, dtype=np.float64), jacobian, option
        )
        reverse_error = validate_pose(reverse_transform)
        cycle = cycle_consistency(T_current_previous, reverse_transform) if reverse_success and not reverse_error else {"passed": False}
        reverse_evaluation = o3d.pipelines.registration.evaluate_registration(
            current_cloud, previous_cloud, max_rmse_m, reverse_transform
        ) if reverse_success and not reverse_error else None
        if not cycle["passed"]:
            rejection_reason = rejection_reason or "forward/backward pose inconsistency"
        elif (reverse_evaluation.fitness < min_fitness or not np.isfinite(reverse_evaluation.inlier_rmse)
              or reverse_evaluation.inlier_rmse > max_rmse_m):
            rejection_reason = rejection_reason or "insufficient reverse overlap/geometry quality"
        if rejection_reason is not None:
            rejected_frames.append(
                {
                    "frame_ordinal": ordinal,
                    "source_frame": int(frame["frame"]),
                    "last_accepted_frame_ordinal": source_ordinal,
                    "reason": rejection_reason,
                }
            )
            continue
        poses.append(poses[-1] @ np.linalg.inv(T_current_previous))
        accepted_frames.append(frame)
        accepted_ordinals.append(ordinal)
        pair_metrics.append(
            {
                "source_frame": int(frames[source_ordinal]["frame"]),
                "target_frame": int(frame["frame"]),
                "source_to_target": T_current_previous.tolist(),
                "translation_step_m": step_m,
                "rotation_step_deg": step_deg,
                "fitness": fitness,
                "inlier_rmse_m": rmse_m,
                "information_trace": float(np.trace(information)),
                "cycle_consistency": cycle,
                "reverse_fitness": float(reverse_evaluation.fitness),
            }
        )
        previous_rgbd = current_rgbd
        previous_cloud = current_cloud

    if accepted_ordinals and len(frames) - 1 - accepted_ordinals[-1] >= POLICY["max_tracking_gap_frames"]:
        raise ValueError("Tracking lost at end of capture; five consecutive frames rejected")
    tracked_ordinals = list(accepted_ordinals)
    selected = select_keyframes(poses)
    keyframe_selection = dict(tracked_frame_ordinals=tracked_ordinals,
        selected_frame_ordinals=[accepted_ordinals[i] for i in selected],
        redundant_frame_ordinals=[value for i, value in enumerate(accepted_ordinals) if i not in selected],
        policy=dict(POLICY), selection_basis="pose diversity after continuous tracking")
    failure_evidence["keyframe_selection"] = keyframe_selection
    accepted_frames = [accepted_frames[i] for i in selected]
    poses = [poses[i] for i in selected]
    accepted_ordinals = [accepted_ordinals[i] for i in selected]
    failure_evidence["accepted_frame_ordinals"] = accepted_ordinals
    if len(accepted_frames) < 3:
        raise ValueError(
            f"Only {len(accepted_frames)}/{len(frames)} frames passed RGB-D pose gates; "
            "move more slowly, keep about 70% overlap, and avoid moving people/objects"
        )
    translation_span_m, rotation_span_deg = pose_diversity(poses)
    failure_evidence.update(translation_span_m=translation_span_m, rotation_span_deg=rotation_span_deg)
    if translation_span_m < 0.02 and rotation_span_deg < 2.0:
        raise ValueError(
            "Estimated trajectory has no useful viewpoint diversity; move the camera slowly around the scene"
        )

    (output / "pose").mkdir(parents=True, exist_ok=False)
    shutil.copytree(dataset / "calibration", output / "calibration")
    derived_frames: list[dict[str, Any]] = []
    for ordinal, (frame, pose) in enumerate(zip(accepted_frames, poses)):
        pose_path = output / "pose" / f"{ordinal:06d}.json"
        pose_record = {
            "schema": "femto-scene-frame-pose-v1",
            "frame": int(frame["frame"]),
            "pose_valid": True,
            "T_WC": pose.tolist(),
            "pose_source": "open3d_sequential_hybrid_rgbd_odometry",
            "coordinate_system": "+X right, +Y down, +Z forward; metres",
            "source_dataset": str(dataset),
            "warning": "Research-grade sequential odometry; no loop closure or relocalization.",
        }
        pose_path.write_text(json.dumps(pose_record, ensure_ascii=False, indent=2), encoding="utf-8")
        derived = dict(frame)
        derived["rgb"] = _relative_path((dataset / frame["rgb"]).resolve(), output)
        derived["depth"] = _relative_path((dataset / frame["depth"]).resolve(), output)
        derived["pose"] = pose_path.relative_to(output).as_posix()
        derived["pose_valid"] = True
        derived_frames.append(derived)
    (output / "frames.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n" for item in derived_frames),
        encoding="utf-8",
    )
    derived_manifest = dict(source_manifest)
    derived_manifest.update(
        {
            "status": "derived_pose_estimated",
            "frame_count": len(accepted_frames),
            "source_dataset": str(dataset),
            "pose": {
                "valid": True,
                "source": "open3d_sequential_hybrid_rgbd_odometry",
                "warning": "No loop closure/relocalization; use quality report and geometric target before production.",
            },
            "files": {
                **dict(source_manifest.get("files", {})),
                "rgb": "references source dataset through frames.jsonl",
                "depth": "references source dataset through frames.jsonl",
                "pose": "pose/*.json",
            },
        }
    )
    (output / "manifest.json").write_text(
        json.dumps(derived_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = {
        "schema": "femto-rgbd-odometry-report-v1",
        "created_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "status": "estimated_research_grade",
        "source_dataset": str(dataset),
        "output_dataset": str(output),
        "source_frame_count": len(frames),
        "frame_count": len(accepted_frames),
        "accepted_frame_ordinals": accepted_ordinals,
        "rejected_frame_count": len(rejected_frames),
        "rejected_frames": rejected_frames,
        "background_motion": motion,
        "translation_span_m": translation_span_m,
        "rotation_span_deg": rotation_span_deg,
        "parameters": {
            "image_scale": image_scale,
            "min_fitness": min_fitness,
            "max_rmse_m": max_rmse_m,
            "max_step_m": max_step_m,
            "max_step_deg": max_step_deg,
            "depth_diff_max_m": option.depth_diff_max,
        },
        "pairs": pair_metrics,
        "frame_quality": frame_quality_metrics,
        "keyframe_selection": keyframe_selection,
        "moving_policy": dict(POLICY),
        "limitations": [
            "Sequential odometry accumulates drift.",
            "There is no loop closure or relocalization in this first demo.",
            "Robot T_BF + calibrated T_FC remains the preferred production pose source for eye-in-hand use.",
        ],
    }
    (output / "pose_quality_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    marker = dataset / "derived" / "latest_posed_dataset.txt"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(str(output), encoding="utf-8")
    return output, report


def main() -> int:
    args = parse_args()
    try:
        dataset = resolve_dataset(args.dataset, args.datasets_root)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output = (args.output or (dataset / "derived" / f"posed_rgbd_odometry_{stamp}")).resolve()
        output, report = estimate_poses(
            dataset,
            output,
            image_scale=args.image_scale,
            min_fitness=args.min_fitness,
            max_rmse_m=args.max_rmse_m,
            max_step_m=args.max_step_m,
            max_step_deg=args.max_step_deg,
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, RuntimeError) as error:
        print(json.dumps({"status": "blocked", "error": str(error)}, ensure_ascii=False, indent=2))
        return 2
    print(
        json.dumps(
            {
                "status": "estimated_research_grade",
                "output_dataset": str(output),
                "translation_span_m": report["translation_span_m"],
                "rotation_span_deg": report["rotation_span_deg"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
