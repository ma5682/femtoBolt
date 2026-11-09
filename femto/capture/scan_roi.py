"""A clinician's image-space scan request, bound to the first captured RGB-D frame.

The rectangle guides acquisition.  It is not a treatment selection or a
robot-space target.  Raw RGB-D is always recorded at its full resolution.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


SCHEMA = 'femto-scan-roi-v1'
MIN_SIDE_PIXELS = 16
MIN_INITIAL_VALID_FRACTION = .5  # Engineering admission gate, not clinical coverage.


def pixel_bounds(rect, image_size):
    if (not isinstance(rect, (list, tuple)) or len(rect) != 4
            or not isinstance(image_size, (list, tuple)) or len(image_size) != 2):
        raise ValueError('扫描范围坐标或图像尺寸无效')
    width, height = image_size
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        raise ValueError('扫描范围图像尺寸无效')
    try:
        left, top, right, bottom = (float(value) for value in rect)
    except (TypeError, ValueError) as error:
        raise ValueError('扫描范围坐标无效') from error
    if (not np.isfinite([left, top, right, bottom]).all()
            or not (0 <= left < right <= 1 and 0 <= top < bottom <= 1)):
        raise ValueError('扫描范围须位于当前画面内')
    x0, y0 = int(np.floor(left * width)), int(np.floor(top * height))
    x1, y1 = int(np.ceil(right * width)), int(np.ceil(bottom * height))
    x1, y1 = min(width, x1), min(height, y1)
    if x1-x0 < MIN_SIDE_PIXELS or y1-y0 < MIN_SIDE_PIXELS:
        raise ValueError('扫描范围太小，请在画面中重新框选')
    return x0, y0, x1, y1


def depth_check(depth_mm, rect, minimum_mm, maximum_mm):
    depth = np.asarray(depth_mm)
    if depth.ndim != 2:
        raise ValueError('缺少与 RGB 对齐的深度画面')
    x0, y0, x1, y1 = pixel_bounds(rect, [depth.shape[1], depth.shape[0]])
    crop = depth[y0:y1, x0:x1]
    valid = np.isfinite(crop) & (crop >= minimum_mm) & (crop <= maximum_mm)
    count = int(valid.sum())
    return dict(valid_pixels=count, total_pixels=int(crop.size),
                valid_fraction=count / crop.size)


def prepare(rect, frame, minimum_mm, maximum_mm):
    color = np.asarray(frame.color_bgr)
    depth = np.asarray(frame.depth_mm)
    if color.ndim != 3 or color.shape[:2] != depth.shape or color.shape[2] != 3:
        raise ValueError('当前 RGB 与深度画面未对齐')
    image_size = [int(depth.shape[1]), int(depth.shape[0])]
    pixel_bounds(rect, image_size)
    check = depth_check(depth, rect, minimum_mm, maximum_mm)
    if check['valid_fraction'] < MIN_INITIAL_VALID_FRACTION:
        raise ValueError('所选范围内有效深度不足一半，请调整取景、距离或扫描范围')
    return dict(schema=SCHEMA, rect_norm=[float(value) for value in rect],
                image_size=image_size, depth_check=check,
                source_frame_index=int(frame.frame_index),
                source_timestamp_us=int(frame.timestamp_us),
                reference_frame_id=None, session_id=None, status='pending_reference',
                meaning='scan_guidance_not_treatment_selection',
                raw_rgbd_cropped=False)


def read(dataset):
    path = Path(dataset) / 'derived' / 'scan_roi.json'
    if not path.is_file():
        return None
    if path.stat().st_size > 16 * 1024:
        raise ValueError('扫描范围记录超出读取预算')
    value = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(value, dict) or value.get('schema') != SCHEMA:
        raise ValueError('扫描范围记录格式无效')
    pixel_bounds(value.get('rect_norm'), value.get('image_size'))
    if value.get('raw_rgbd_cropped') is not False:
        raise ValueError('扫描范围的原始数据语义无效')
    return value


def assess_model(model, roi, cancel=None, *, preview=False, evidence=False):
    """Check requested pixels against the verified mesh/reference-depth map."""
    reference = model.view_reference
    if (roi.get('status') != 'bound'
            or roi.get('reference_frame_id') != reference.get('reference_frame_id')
            or roi.get('session_id') != reference.get('session_id')
            or roi.get('image_size') != reference.get('image_size')):
        raise ValueError('扫描范围与当前模型首帧不一致，不能核验覆盖')
    from femto.geometry.rgbd_frames import distortion_maps
    from femto.annotation.reference_quality import assess_selection

    width, height = roi['image_size']
    mask_raw = np.zeros((height, width), np.uint8)
    x0, y0, x1, y1 = pixel_bounds(roi['rect_norm'], roi['image_size'])
    mask_raw[y0:y1, x0:x1] = 1
    dataset = Path(model.store.origin_reference['source_dataset'])
    map_x, map_y, _, map_width, map_height = distortion_maps(dataset)
    if (map_width, map_height) != (width, height):
        raise ValueError('扫描范围与首帧标定尺寸不一致')
    mask = cv2.remap(mask_raw, map_x, map_y, cv2.INTER_NEAREST,
                     borderMode=cv2.BORDER_CONSTANT)
    # Preserve original physician intent even where undistortion sampling
    # cannot represent a raw target pixel. Such pixels cannot disappear from
    # the completion conclusion merely because the mapped mask is smaller.
    used=np.flatnonzero(mask)
    raw_x=np.rint(map_x.ravel()[used]).astype(np.int64)
    raw_y=np.rint(map_y.ravel()[used]).astype(np.int64)
    inside=(raw_x>=x0)&(raw_x<x1)&(raw_y>=y0)&(raw_y<y1)
    represented=np.unique((raw_y[inside]-y0)*(x1-x0)+(raw_x[inside]-x0))
    raw_count=(x1-x0)*(y1-y0)
    raw_mapping=dict(raw_requested_pixels=raw_count,represented_raw_pixels=int(len(represented)),
                     unmapped_raw_target_pixels=raw_count-int(len(represented)),
                     raw_pixel_bounds=[x0,y0,x1,y1],
                     meaning='raw_target_sampling_accounting_not_surface_coverage')
    from femto.geometry.local_region import model_selection
    selected = model_selection(model)
    if selected is not None:
        mask = selected
        raw_mapping = None  # Intent was painted directly in undistorted pixels.
    result = assess_selection(model, mask, np.zeros_like(mask), cancel)
    summary = dict(status=result['status'], requested_pixels=result['selected'],
                supported_pixels=result['supported'], pending_pixels=result['pending'],
                supported_pixel_fraction=result['ratio'], message=result['message'],
                raw_target_mapping=raw_mapping,
                target_scope='painted_region' if selected is not None else 'scan_guidance_rectangle',
                meaning='reference_image_pixel_support_not_surface_area_or_treatment_coverage')
    if not preview and not evidence:
        return summary
    image = np.asarray(model.rgb).copy()
    rgba = result.get('rgba')
    if rgba is not None:
        alpha = rgba[:, :, 3:4].astype(np.float32) / 255
        image = np.rint(image * (1-alpha) + rgba[:, :, :3] * alpha).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (38, 214, 162), 3)
    if max(width, height) > 1600:
        scale = 1600 / max(width, height)
        image = cv2.resize(image, (round(width*scale), round(height*scale)),
                           interpolation=cv2.INTER_AREA)
    if evidence:
        return summary, image, result.get('classes')
    return summary, image
