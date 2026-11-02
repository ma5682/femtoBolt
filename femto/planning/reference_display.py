"""Verified full-resolution RGB for display only; never a planning input."""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
from PIL import Image

ASSET = 'display_rgb.png'


def install(model, rgb: np.ndarray) -> None:
    from femto.planning.planning_revision import array_record
    d = model.view_reference.get('derivation') or {}
    stride = d.get('stride')
    offset = d.get('source_pixel_offset_xy')
    if (d.get('schema') != 'reduced-planning-reference-v1'
            or type(stride) is not int or stride not in (4, 8, 16)
            or offset != [stride//2, stride//2]
            or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3
            or d.get('original_image_size') != [rgb.shape[1], rgb.shape[0]]
            or array_record(rgb) != d.get('original_rgb')
            or not np.array_equal(rgb[offset[1]::stride, offset[0]::stride], model.rgb)):
        raise ValueError('高清底图与原始参考及粗参考像素不一致')
    value = np.array(rgb, copy=True);value.setflags(write=False)
    model.display_rgb = value
    # QImage bounds use pixel edges, whereas the reference mapping uses centers.
    model.display_rect = ((stride/2-offset[0]-.5)/stride,
                          (stride/2-offset[1]-.5)/stride,
                          rgb.shape[1]/stride, rgb.shape[0]/stride)


def save(model, root: Path, check=lambda: None) -> None:
    rgb = getattr(model, 'display_rgb', None)
    if rgb is None:return
    check();install(model, rgb)
    Image.fromarray(rgb).save(Path(root)/ASSET, compress_level=1)
    check()


def load(model, root: Path, check=lambda: None) -> None:
    from femto.annotation.fixed_rgb_geometry import MAX_IMAGE_PIXELS
    path = Path(root)/ASSET
    if not path.is_file():return
    check()
    with Image.open(path) as im:
        if im.width*im.height > MAX_IMAGE_PIXELS or im.mode != 'RGB':
            raise ValueError('高清底图格式或尺寸无效')
        rgb = np.asarray(im).copy()
    install(model, rgb);check()


def recover(model, check=lambda: None) -> None:
    """Old local archives may recover the original photo, only with exact hashes.

    Missing original sources leave the original coarse display usable. A declared
    bundled display asset is validated by load(), never silently ignored.
    """
    if getattr(model, 'display_rgb', None) is not None:return
    d = model.view_reference.get('derivation') or {}
    if d.get('schema') != 'reduced-planning-reference-v1':return
    origin = model.store.origin_reference
    check()
    try:
        import cv2
        from femto.geometry.rgbd_frames import distortion_maps
        from femto.io_utils import imread_unicode
        dataset = Path(origin['source_dataset'])
        rows = [json.loads(line) for line in (dataset/'frames.jsonl').read_text(encoding='utf-8-sig').splitlines() if line.strip()]
        frame = next(r for r in rows if r['frame'] == origin['reference_frame_id'])
        mx, my, k, w, h = distortion_maps(dataset)
        if d['original_image_size'] != [w, h]:return
        bgr = imread_unicode(dataset/frame['rgb'], cv2.IMREAD_COLOR)
        if bgr is None or bgr.shape != (h, w, 3):return
        rgb = cv2.remap(bgr, mx, my, cv2.INTER_LINEAR)[:, :, ::-1].copy()
        install(model, rgb)
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        pass
    check()
