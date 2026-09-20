"""Fixed-grid temporal estimator. No poses, registration or synthetic observations."""
from dataclasses import dataclass, asdict
import math
from time import perf_counter
import warnings

import numpy as np


@dataclass(frozen=True)
class TemporalConfig:
    minimum_support_ratio: float = .60
    minimum_inlier_ratio: float = .80
    inlier_limit_mm: float = 30.0
    maximum_mad_mm: float = 15.0
    chunk_rows: int = 48

    def __post_init__(self):
        if not .6 <= self.minimum_support_ratio <= 1 or not .8 <= self.minimum_inlier_ratio <= 1:
            raise ValueError('Temporal support/inlier gates may only be tightened')
        if not 0 < self.inlier_limit_mm <= 50 or not 0 < self.maximum_mad_mm <= 15:
            raise ValueError('Invalid temporal continuity limits')
        if not isinstance(self.chunk_rows,int) or self.chunk_rows < 1:
            raise ValueError('chunk_rows must be positive')


def confidence_weighted_median(values, weights):
    """Axis-0 weighted median selecting an actual observation; ties choose lower."""
    values,weights = np.asarray(values),np.asarray(weights)
    if values.shape != weights.shape or values.ndim < 1 or values.shape[0] == 0:
        raise ValueError('Matching non-empty sample/weight arrays required')
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError('Weights must be finite and nonnegative')
    if not np.isfinite(values[weights>0]).all():
        raise ValueError('Weighted samples must be finite')
    order = np.argsort(np.where(weights>0,values,np.inf),axis=0,kind='stable')
    ordered_values = np.take_along_axis(values,order,axis=0)
    ordered_weights = np.take_along_axis(weights,order,axis=0)
    total = ordered_weights.sum(axis=0)
    indices = np.argmax(np.cumsum(ordered_weights,axis=0)>=total[None]/2,axis=0)
    selected = np.take_along_axis(ordered_values,indices[None],axis=0)[0]
    return np.where(total>0,selected,0)


def fuse_temporal_depth(depths, minimum_mm, maximum_mm, *, motion_mask=None, config=TemporalConfig()):
    started = perf_counter()
    if len(depths) not in (5,10,15):
        raise ValueError('Temporal mode requires 5, 10 or 15 distinct frames')
    shape = depths[0].shape
    if len(shape)!=2 or depths[0].size==0 or not all(a.shape==shape and a.dtype==np.uint16 for a in depths):
        raise ValueError('All frames must use the same uint16 millimetre pixel grid')
    if not 0 <= minimum_mm < maximum_mm <= 65535:
        raise ValueError('Invalid depth range')
    if motion_mask is None:
        motion_mask = np.zeros(shape,bool)
    if motion_mask.shape != shape:
        raise ValueError('Motion mask grid mismatch')
    n = len(depths); required = max(3,math.ceil(n*config.minimum_support_ratio))
    fused = np.zeros(shape,np.uint16); confidence = np.zeros(shape,np.float32)
    support_map = np.zeros(shape,np.uint8); inlier_map = np.zeros(shape,np.uint8)
    variance_map = np.zeros(shape,np.float32); mad_map = np.zeros(shape,np.float32)
    invalid_counts = np.zeros(n,np.int64)
    removed = removed_pixels = unsupported = unstable = 0
    for y in range(0,shape[0],config.chunk_rows):
        sl = slice(y,min(y+config.chunk_rows,shape[0]))
        values = np.stack([a[sl] for a in depths]).astype(np.float32)
        valid = (values>0)&(values>=minimum_mm)&(values<=maximum_mm)
        invalid_counts += (~valid).sum(axis=(1,2))
        count = valid.sum(axis=0)
        median = confidence_weighted_median(values,valid.astype(np.float32))
        deviations = np.abs(values-median)
        mad = confidence_weighted_median(deviations,valid.astype(np.float32))
        inliers = valid & (deviations<=config.inlier_limit_mm)
        inlier_count = inliers.sum(axis=0)
        purity = inlier_count/np.maximum(count,1)
        # All real samples start with confidence 1. Missing/repaired samples
        # have weight 0 in this measured-only temporal branch.
        result = confidence_weighted_median(values,inliers.astype(np.float32))
        mean = np.where(valid,values,0).sum(axis=0)/np.maximum(count,1)
        variance = np.where(valid,(values-mean)**2,0).sum(axis=0)/np.maximum(count,1)
        enough = inlier_count>=required
        consistent = (purity>=config.minimum_inlier_ratio)&(mad<=config.maximum_mad_mm)
        accepted = enough & consistent & ~motion_mask[sl]
        weights = (inlier_count/n)*purity*np.exp(-mad/config.maximum_mad_mm)
        fused[sl] = np.where(accepted,result,0).astype(np.uint16)
        confidence[sl] = np.where(accepted,weights,0).astype(np.float32)
        support_map[sl]=count; inlier_map[sl]=inlier_count
        variance_map[sl]=variance; mad_map[sl]=mad
        removed += int((valid & ~inliers).sum())
        removed_pixels += int(np.any(valid & ~inliers,axis=0).sum())
        unsupported += int((~enough).sum())
        unstable += int((enough & ~consistent).sum())
    total = fused.size
    valid_fused = fused>0
    report = dict(schema='femto-temporal-fusion-v1',frame_count=n,
                  original_invalid_rate=float(invalid_counts[0]/total),
                  original_invalid_rate_definition='first selected raw frame; rates are fractions [0,1]',
                  mean_input_invalid_rate=float(invalid_counts.mean()/total),
                  per_frame_invalid_rate=(invalid_counts/total).tolist(),
                  after_temporal_invalid_rate=float((~valid_fused).mean()),
                  stable_pixel_rate=float((confidence>=.8).mean()),
                  stable_pixel_definition='confidence >=0.8 / all pixels',
                  removed_outlier_observations=removed,removed_outlier_pixels=removed_pixels,
                  removed_outlier_pixels_definition='unique pixel locations with at least one rejected observation',
                  rejected_insufficient_support_pixels=unsupported,rejected_unstable_pixels=unstable,
                  rejected_motion_pixels=int(motion_mask.sum()),
                  point_count=int(valid_fused.sum()),reference_point_count=int(total-invalid_counts[0]),
                  mean_final_confidence=float(confidence[valid_fused].mean()) if valid_fused.any() else None,
                  minimum_real_support_frames=required,parameters=asdict(config),
                  confidence_formula='real_inlier_count/N * inlier_count/real_valid_count * exp(-MAD/15mm)',
                  evidence_policy='Only original valid depth; no per-frame spatial repairs used as temporal evidence',
                  fusion_ms=(perf_counter()-started)*1000)
    maps = dict(real_support=support_map,inlier_support=inlier_map,variance_mm2=variance_map,mad_mm=mad_map)
    return fused,confidence,maps,report
