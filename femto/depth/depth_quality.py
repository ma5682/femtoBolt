"""Auditable, conservative filling for small high-confidence RGB-D depth holes.

Raw capture files are never modified.  The caller supplies an explicit derived
directory for the original-depth copy, invalid mask, filled depth, confidence
map and JSON report.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from time import perf_counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class DepthQualityConfig:
    global_repair_limit: float = 0.03
    local_repair_limit: float = 0.10
    max_small_hole_area: int = 64
    max_local_hole_area: int = 24
    max_hole_width: int = 12
    max_hole_height: int = 12
    minimum_surrounding_valid_ratio: float = 0.60
    maximum_depth_std_mm: float = 30.0
    repaired_confidence: float = 0.30
    neighborhood_radius: int = 4
    maximum_depth_span_mm: float = 60.0
    mode: str = "quality"
    maximum_repair_error_mm: float = 50.0

    def __post_init__(self):
        if self.mode not in ("fast", "quality"):
            raise ValueError("Depth quality mode must be fast or quality")
        if not 0 < self.maximum_repair_error_mm <= 50:
            raise ValueError("Repair error gate must be positive and at most 50 mm")
        # Configuration may tighten the policy, never silently weaken it.
        if not 0 <= self.global_repair_limit <= 0.03 or not self.global_repair_limit <= self.local_repair_limit <= 0.10:
            raise ValueError("Repair rate limits must respect the 3%/10% gates")
        if not 0.60 <= self.minimum_surrounding_valid_ratio <= 1:
            raise ValueError("Original neighbor support must be at least 60%")
        if not 0 < self.maximum_depth_std_mm <= 30 or not 0 < self.maximum_depth_span_mm <= 60:
            raise ValueError("Depth continuity limits must be positive and conservative")
        if self.repaired_confidence != 0.30:
            raise ValueError("Downstream fusion requires repaired confidence 0.3")
        for name in ("max_small_hole_area", "max_local_hole_area", "max_hole_width", "max_hole_height", "neighborhood_radius"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_local_hole_area > self.max_small_hole_area:
            raise ValueError("Local repair must not allow larger holes than automatic repair")


def invalid_depth_mask(depth_mm: np.ndarray, min_range_mm: float, max_range_mm: float) -> np.ndarray:
    depth = np.asarray(depth_mm)
    if depth.ndim != 2:
        raise ValueError("Depth must be a 2-D image")
    if not (math.isfinite(min_range_mm) and math.isfinite(max_range_mm) and 0 <= min_range_mm < max_range_mm):
        raise ValueError("Invalid active depth range")
    return (~np.isfinite(depth)) | (depth == 0) | (depth < min_range_mm) | (depth > max_range_mm)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    return float(values[np.searchsorted(np.cumsum(weights), weights.sum() * 0.5)])


def _edge_aware_value(
    y: int, x: int, depth: np.ndarray, valid: np.ndarray, rgb: np.ndarray | None, radius: int
) -> float | None:
    y0, y1 = max(0, y-radius), min(depth.shape[0], y+radius+1)
    x0, x1 = max(0, x-radius), min(depth.shape[1], x+radius+1)
    local_valid = valid[y0:y1, x0:x1]
    yy, xx = np.nonzero(local_valid)
    if len(yy) < 5:
        return None
    yy, xx = yy+y0, xx+x0
    values = depth[yy, xx].astype(np.float64)
    spatial2 = (yy-y)**2 + (xx-x)**2
    weights = np.exp(-spatial2 / max(2.0*radius*radius, 1.0))
    if rgb is not None:
        difference = rgb[yy, xx].astype(np.float64) - rgb[y, x].astype(np.float64)
        weights *= np.exp(-np.sum(difference*difference, axis=1)/(2.0*35.0**2))
    if float(weights.sum()) <= 1e-9:
        return None
    return _weighted_median(values, weights)


def _local_median_value(y: int, x: int, depth: np.ndarray, valid: np.ndarray, radius: int) -> float | None:
    y0, y1 = max(0, y-radius), min(depth.shape[0], y+radius+1)
    x0, x1 = max(0, x-radius), min(depth.shape[1], x+radius+1)
    values = depth[y0:y1, x0:x1][valid[y0:y1, x0:x1]]
    return float(np.median(values)) if len(values) >= 3 else None


def _ransac_plane(boundary_yx: np.ndarray, boundary_depth: np.ndarray, targets_yx: np.ndarray,
                  threshold_mm: float) -> np.ndarray | None:
    """Deterministic small-sample RANSAC in pixel/depth space."""
    if len(boundary_yx) < 6:
        return None
    design = np.column_stack((boundary_yx[:, 1], boundary_yx[:, 0], np.ones(len(boundary_yx))))
    rng = np.random.default_rng(20260909 + len(boundary_yx))
    best: np.ndarray | None = None
    best_count = 0
    inlier_limit = max(3.0, threshold_mm / 3.0)
    for choice in rng.choice(len(boundary_yx), size=(min(32, max(8, len(boundary_yx))), 3), replace=True):
        if len(set(choice.tolist())) < 3:
            continue
        try:
            plane = np.linalg.solve(design[choice], boundary_depth[choice])
        except np.linalg.LinAlgError:
            continue
        residual = np.abs(design @ plane - boundary_depth)
        inliers = residual <= inlier_limit
        count = int(inliers.sum())
        if count > best_count:
            best, best_count = inliers, count
    if best is None or best_count < max(6, math.ceil(len(boundary_yx)*0.8)):
        return None
    plane, *_ = np.linalg.lstsq(design[best], boundary_depth[best], rcond=None)
    residual_std = float(np.std(design[best] @ plane - boundary_depth[best]))
    if residual_std >= threshold_mm / 2.0:
        return None
    target_design = np.column_stack((targets_yx[:, 1], targets_yx[:, 0], np.ones(len(targets_yx))))
    return target_design @ plane


def _check_and_report(depth_input,filled,invalid,confidence,labels,component_details,
                      min_range_mm,max_range_mm,config,started,total,valid,invalid_pixels,
                      invalid_rate,policy,count,ransac_checked,repaired_pixels,rejected_large,
                      rejected_reflection,rejected_low_support,rejected_global,method_pixels):
    from femto.depth.depth_error_check import depth_error_check
    error_mask, error_report = depth_error_check(depth_input, filled, min_range_mm, max_range_mm,
                                                max_error_mm=config.maximum_repair_error_mm)
    if not error_report['original_integrity_passed']:
        raise RuntimeError("Depth filling changed original valid observations")
    # All-or-none rollback also applies to the new error gate.
    rejected_error_regions = 0
    for detail in component_details:
        if detail['decision'] != 'repaired':
            continue
        x, y, w, h = (detail[k] for k in ('x','y','width','height'))
        region = labels[y:y+h,x:x+w] == detail['label']
        if np.any(error_mask[y:y+h,x:x+w] & region):
            filled[y:y+h,x:x+w][region] = 0
            confidence[y:y+h,x:x+w][region] = 0
            repaired_pixels -= detail['repaired_pixels']
            for method, amount in detail['methods'].items():
                method_pixels[method] -= amount
            detail.update(decision='rejected_depth_error', repaired_pixels=0,
                          classification='C_depth_error_over_limit_or_unsupported')
            rejected_error_regions += 1
    error_report['rejected_regions'] = rejected_error_regions
    _, final_check = depth_error_check(depth_input, filled, min_range_mm, max_range_mm,
                                      max_error_mm=config.maximum_repair_error_mm)
    error_report['after_rejection'] = final_check
    final_invalid = invalid_depth_mask(filled, min_range_mm, max_range_mm)
    report: dict[str, Any] = {
        "schema": "femto-depth-quality-v1",
        "total_pixels": total,
        "valid_depth_pixels": int(valid.sum()),
        "invalid_depth_pixels": invalid_pixels,
        "original_invalid_rate": invalid_rate,
        "invalid_rate": invalid_rate,
        "neighbor_policy": f"Every repaired pixel requires original 8-neighbor support >= {config.minimum_surrounding_valid_ratio}; std < {config.maximum_depth_std_mm} mm; boundary span <= {config.maximum_depth_span_mm} mm; all-or-none per component",
        "policy": policy,
        "processing_mode": config.mode,
        "component_analysis_scope": "isolated_single_pixel_candidates_only" if config.mode == "fast" else "all_invalid_components",
        "unanalysed_invalid_pixels": invalid_pixels-(count-1) if config.mode == "fast" else 0,
        "ransac_checked_regions": ransac_checked,
        "depth_error_check": error_report,
        "rejected_depth_error_regions": rejected_error_regions,
        "repaired_pixels": repaired_pixels,
        "repair_rate": repaired_pixels/max(invalid_pixels, 1),
        "repair_rate_denominator": "original invalid_depth_pixels (rates are fractions, not percentages)",
        "active_range_mm": [float(min_range_mm), float(max_range_mm)],
        "remaining_invalid_pixels": int(final_invalid.sum()),
        "filled_invalid_rate": float(final_invalid.mean()),
        "rejected_large_holes": rejected_large,
        "rejected_reflection_regions": rejected_reflection,
        "rejected_low_support_regions": rejected_low_support,
        "rejected_global_policy_regions": rejected_global,
        "material_identification": "not performed; depth discontinuity is a risk indicator, not a material diagnosis",
        "component_count": count-1,
        "method_pixels": method_pixels,
        "thresholds": asdict(config),
        "confidence": {"original_depth": 1.0, "repaired_depth": config.repaired_confidence, "unknown": 0.0},
        "geometry_policy": "Original observations remain authoritative; only A-class holes are estimated.",
        "components": component_details,
        "processing_ms": (perf_counter()-started)*1000.0,
    }
    return filled, invalid, confidence, report


def assess_and_fill_depth(
    depth_mm: np.ndarray,
    min_range_mm: float,
    max_range_mm: float,
    *,
    rgb_bgr: np.ndarray | None = None,
    config: DepthQualityConfig = DepthQualityConfig(),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Return filled uint16 depth, invalid mask, float confidence and report."""
    started = perf_counter()
    depth_input = np.asarray(depth_mm)
    if depth_input.dtype != np.uint16:
        raise ValueError("Depth evidence must be uint16 millimetres")
    if depth_input.size == 0:
        raise ValueError("Depth evidence must be non-empty")
    if rgb_bgr is not None and np.asarray(rgb_bgr).shape != (*depth_input.shape, 3):
        raise ValueError("Aligned RGB and depth dimensions differ")
    invalid = invalid_depth_mask(depth_input, min_range_mm, max_range_mm)
    valid = ~invalid
    total = int(depth_input.size)
    invalid_pixels = int(invalid.sum())
    invalid_rate = invalid_pixels / max(total, 1)
    if invalid_rate <= config.global_repair_limit:
        policy, area_limit = "automatic_small_holes", config.max_small_hole_area
    elif invalid_rate <= config.local_repair_limit:
        policy, area_limit = "local_high_confidence_only", config.max_local_hole_area
    else:
        policy, area_limit = "report_only", 0

    filled = np.where(valid, depth_input, 0).astype(np.uint16)
    confidence = np.where(valid, 1.0, 0.0).astype(np.float32)
    component_details: list[dict[str, Any]] = []
    repaired_pixels = rejected_large = rejected_reflection = rejected_low_support = 0
    rejected_global = 0
    method_pixels = {"edge_aware": 0, "local_interpolation": 0, "ransac_plane": 0}
    if config.mode == "fast":
        # An isolated hole has eight ORIGINAL valid neighbors: no labeling or
        # traversal of large missing regions in this deliberately narrower mode.
        neighbors = cv2.filter2D(valid.astype(np.uint8), cv2.CV_16U,
                                np.ones((3,3),np.uint8), borderType=cv2.BORDER_CONSTANT)
        fy, fx = np.nonzero(invalid & (neighbors == 8))
        labels = np.zeros(invalid.shape, np.int32)
        labels[fy,fx] = np.arange(1,len(fy)+1)
        stats = np.zeros((len(fy)+1,5),np.int32)
        stats[1:] = np.column_stack((fx,fy,np.ones((len(fy),3),np.int32)))
        count = len(fy)+1
    else:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(invalid.astype(np.uint8), connectivity=8)
    ransac_checked = 0
    kernel = np.ones((3, 3), np.uint8)
    for label in range(1, count):
        x, y, width, height, area = [int(v) for v in stats[label]]
        # Only inspect the bounding box and its one-pixel original boundary.
        # Full-frame dilation per speckle is quadratic in real noisy captures.
        y0, y1 = max(0, y-1), min(depth_input.shape[0], y+height+1)
        x0, x1 = max(0, x-1), min(depth_input.shape[1], x+width+1)
        component = labels[y0:y1, x0:x1] == label
        local_valid = valid[y0:y1, x0:x1]
        local_depth = depth_input[y0:y1, x0:x1]
        ring = cv2.dilate(component.astype(np.uint8), kernel, iterations=1).astype(bool) & ~component
        local_footprint = cv2.dilate(component.astype(np.uint8), kernel, iterations=1).astype(bool)
        surrounding_total = int(ring.sum())
        boundary_valid = ring & local_valid
        surrounding_ratio = float(boundary_valid.sum()/surrounding_total) if surrounding_total else 0.0
        local_support_ratio = float((local_footprint & local_valid).sum()/local_footprint.sum()) if local_footprint.any() else 0.0
        boundary_depth = local_depth[boundary_valid].astype(np.float64)
        depth_std = float(np.std(boundary_depth)) if len(boundary_depth) else None
        depth_span = float(np.ptp(boundary_depth)) if len(boundary_depth) else None
        touches_border = x == 0 or y == 0 or x+width == depth_input.shape[1] or y+height == depth_input.shape[0]
        detail: dict[str, Any] = {
            "label": label, "x": x, "y": y, "area": area, "width": width, "height": height,
            "surrounding_valid_depth_ratio": surrounding_ratio,
            "local_8_neighborhood_valid_ratio": local_support_ratio,
            "depth_std_mm": depth_std, "depth_span_mm": depth_span, "touches_image_border": touches_border,
            "classification": None, "decision": None, "repaired_pixels": 0,
        }
        if touches_border or area > (area_limit or config.max_small_hole_area) or width > config.max_hole_width or height > config.max_hole_height:
            detail.update(classification="B_large_or_unbounded", decision="rejected")
            rejected_large += 1
        elif depth_std is None or depth_std >= config.maximum_depth_std_mm or depth_span > config.maximum_depth_span_mm:
            detail.update(classification="C_depth_discontinuity_reflection_or_edge", decision="rejected")
            rejected_reflection += 1
        elif local_support_ratio < config.minimum_surrounding_valid_ratio:
            detail.update(classification="C_low_valid_support", decision="rejected")
            rejected_low_support += 1
        elif policy == "report_only":
            detail.update(classification="unverified_small_hole", decision="rejected_global_invalid_rate")
            rejected_global += 1
        else:
            targets = np.column_stack(np.nonzero(component)) + [y0, x0]
            values = np.full(len(targets), np.nan, dtype=np.float64)
            methods = np.full(len(targets), "", dtype=object)
            eligible = np.zeros(len(targets), dtype=bool)
            for index, (py, px) in enumerate(targets):
                neighbors = depth_input[py-1:py+2, px-1:px+2]
                neighbor_valid = valid[py-1:py+2, px-1:px+2]
                observed = neighbors[neighbor_valid]
                if observed.size / 8.0 < config.minimum_surrounding_valid_ratio:
                    continue
                if float(np.std(observed.astype(np.float64))) >= config.maximum_depth_std_mm:
                    continue
                eligible[index] = True
                # Estimate only from the same original neighbors that passed
                # the gate. Never propagate repaired depth or cross a wider edge.
                value = _edge_aware_value(int(py), int(px), depth_input, valid, rgb_bgr, 1)
                if value is not None:
                    values[index], methods[index] = value, "edge_aware"
                else:
                    value = _local_median_value(int(py), int(px), depth_input, valid, 1)
                    if value is not None:
                        values[index], methods[index] = value, "local_interpolation"
            # Ineligible pixels must not be filled by a fallback estimator.
            missing = eligible & ~np.isfinite(values)
            if config.mode == "quality" and eligible.all():
                boundary_yx = np.column_stack(np.nonzero(boundary_valid)) + [y0, x0]
                ransac_checked += 1
                plane_values = _ransac_plane(boundary_yx, boundary_depth,
                                             targets, config.maximum_depth_std_mm)
                if plane_values is not None:
                    # Keep edge-aware estimates first; a fitted local plane is
                    # only a consistency cross-check or a gated fallback.
                    inconsistent = np.isfinite(values) & (np.abs(values-plane_values) > config.maximum_repair_error_mm)
                    values[missing] = plane_values[missing]
                    methods[missing] = "ransac_plane"
                    values[inconsistent] = np.nan
            accepted = np.isfinite(values) & (values >= min_range_mm) & (values <= max_range_mm)
            if accepted.all():
                rounded = np.rint(values).astype(np.uint16)
                filled[targets[:, 0], targets[:, 1]] = rounded
                confidence[targets[:, 0], targets[:, 1]] = config.repaired_confidence
                repaired_pixels += area
                for method in method_pixels:
                    method_pixels[method] += int(np.sum(methods == method))
                detail.update(classification="A_small_continuous_hole", decision="repaired",
                              repaired_pixels=area, methods={m:int(np.sum(methods == m)) for m in method_pixels})
            else:
                detail.update(classification="C_insufficient_trustworthy_estimate", decision="rejected")
                rejected_low_support += 1
        component_details.append(detail)

    return _check_and_report(depth_input,filled,invalid,confidence,labels,component_details,
        min_range_mm,max_range_mm,config,started,total,valid,invalid_pixels,invalid_rate,
        policy,count,ransac_checked,repaired_pixels,rejected_large,rejected_reflection,
        rejected_low_support,rejected_global,method_pixels)


def save_depth_quality(
    output: Path,
    original_depth_mm: np.ndarray,
    invalid_mask: np.ndarray,
    filled_depth_mm: np.ndarray,
    confidence: np.ndarray,
    report: dict[str, Any],
) -> dict[str, str]:
    """Write only derived products; uint16 originals and filled values remain millimetres."""
    from femto.depth.depth_error_check import depth_error_check
    _, check = depth_error_check(original_depth_mm, filled_depth_mm, *report['active_range_mm'])
    if not check['passed']:
        raise ValueError("Refusing to save depth that failed geometric integrity")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    paths = {
        "original_depth": output / "original_depth_mm.png",
        "invalid_mask": output / "invalid_mask.png",
        "filled_depth": output / "filled_depth_mm.png",
        "confidence": output / "confidence.npy",
        "confidence_preview": output / "confidence.png",
        "quality_report": output / "depth_quality.json",
        "quality_report_alias": output / "quality_report.json",
        "depth_error_check": output / "depth_error_check.json",
    }
    Image.fromarray(np.asarray(original_depth_mm, dtype=np.uint16)).save(paths["original_depth"])
    Image.fromarray(np.asarray(invalid_mask, dtype=np.uint8)*255).save(paths["invalid_mask"])
    Image.fromarray(np.asarray(filled_depth_mm, dtype=np.uint16)).save(paths["filled_depth"])
    np.save(paths["confidence"], np.asarray(confidence, dtype=np.float32), allow_pickle=False)
    Image.fromarray(np.rint(np.clip(confidence, 0, 1)*255).astype(np.uint8)).save(paths["confidence_preview"])
    enriched = {**report, "outputs": {name: path.name for name, path in paths.items()}}
    paths["quality_report"].write_text(json.dumps(enriched, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["quality_report_alias"].write_text(json.dumps(enriched, ensure_ascii=False, indent=2), encoding="utf-8")
    paths['depth_error_check'].write_text(json.dumps(report['depth_error_check'],ensure_ascii=False,indent=2),encoding='utf-8')
    return {name: str(path.resolve()) for name, path in paths.items()}


def process_depth_quality(depth_mm: np.ndarray, min_range_mm: float, max_range_mm: float, output: Path,
                          *, rgb_bgr: np.ndarray | None = None,
                          config: DepthQualityConfig = DepthQualityConfig(),
                          keep_artifacts: bool = True):
    filled, mask, confidence, report = assess_and_fill_depth(
        depth_mm, min_range_mm, max_range_mm, rgb_bgr=rgb_bgr, config=config
    )
    if keep_artifacts:
        paths = save_depth_quality(output, depth_mm, mask, filled, confidence, report)
    else:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        quality = output / "depth_quality.json"
        quality.write_text(json.dumps({**report, "outputs": {}, "statistics_only": True},
                                      ensure_ascii=False, indent=2), encoding="utf-8")
        paths = {"quality_report": str(quality.resolve())}
    return filled, confidence, report, paths


def confidence_acceptance_mask(confidence: np.ndarray, frame_index: int = 0) -> np.ndarray:
    """Deterministically retain 30% of repaired samples for Open3D TSDF.

    Open3D's legacy scalable TSDF API has no fractional per-pixel weight. This
    spatially distributed acceptance realizes the confidence ratio while every
    original sample remains admitted.
    """
    confidence = np.asarray(confidence, dtype=np.float32)
    original = confidence >= 0.999
    repaired = (confidence > 0) & ~original
    yy, xx = np.indices(confidence.shape, dtype=np.uint64)
    bucket = (xx*73856093 + yy*19349663 + np.uint64(frame_index)*83492791) % 1000
    threshold = np.rint(np.clip(confidence, 0, 1)*1000).astype(np.uint64)
    return original | (repaired & (bucket < threshold))
