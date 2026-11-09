"""Choose the local reconstruction branch from measured ROI contributions.

This pass only maps/filter depths; it constructs no cloud, mesh or TSDF. Its
depth arrays are the exact inputs reused by reconstruction and support checks.
The dataset and fixed reference are never rewritten when a view contributes
no surface to this selection.
"""
import cv2
import numpy as np

from femto.geometry.rgbd_frames import prepare_depth
from femto.io_utils import imread_unicode
from femto.local.local_temporal_support import filtered_depth
from femto.operation_diagnostics import diagnosed


@diagnosed('selection.prepare_contributing_frames')
def prepare_local_depths(dataset, frames, poses, maps, minimum, maximum, region_filter):
    mx, my, k, width, height = maps
    prepared = {}
    for frame, pose in zip(frames, poses):
        raw = imread_unicode(dataset / frame['depth'], cv2.IMREAD_UNCHANGED)
        if raw.shape != (height, width) or raw.dtype != np.uint16:
            raise ValueError(f"Frame {frame['frame']} depth does not match calibrated uint16 millimetres")
        measured = prepare_depth(filtered_depth(dataset, frame, raw), mx, my, minimum, maximum)
        local = region_filter.filter(measured, pose, k, frame['frame'])
        region_filter.metrics[-1].update(
            raw_nonzero_depth_pixels=int(np.count_nonzero(raw)),
            raw_in_range_pixels=int(np.count_nonzero((raw >= minimum*1000) & (raw <= maximum*1000))),
            surface_input_pixels=int(np.count_nonzero(local)),
            tsdf_integration_passes=0, surface_method=None)
        if np.any(local):
            prepared[frame['frame']] = local
    return prepared
