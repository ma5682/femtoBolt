"""Audited K6 XYZRGB plus one confidence per point, including PLY persistence."""
from pathlib import Path

import numpy as np

from depth_error_check import depth_error_check
from depth_quality import invalid_depth_mask


def build_confident_depth_cloud(original, filled, confidence, rgb_bgr, intrinsics, distortion,
                                min_range_mm, max_range_mm, *, temporal_layer=None):
    from scene_segmented_capture import depth_cloud
    if temporal_layer is not None:
        temporal_layer.validate()
        if not np.array_equal(filled,temporal_layer.fused_depth_mm) or not np.array_equal(confidence,temporal_layer.fused_confidence):
            raise ValueError('Cloud input differs from temporal intermediate layer')
        return depth_cloud(filled,rgb_bgr,intrinsics,distortion,min_range_mm,max_range_mm,
                           confidence=confidence,temporal_confidence=True)
    _, check = depth_error_check(original,filled,min_range_mm,max_range_mm)
    if not check['passed']:
        raise ValueError('Depth error check failed; refusing Point Cloud generation')
    expected = np.where(~invalid_depth_mask(original,min_range_mm,max_range_mm),1.0,
                        np.where(~invalid_depth_mask(filled,min_range_mm,max_range_mm),.3,0)).astype(np.float32)
    if not np.array_equal(confidence,expected):
        raise ValueError('Confidence does not match original/repaired/unknown provenance')
    return depth_cloud(filled,rgb_bgr,intrinsics,distortion,min_range_mm,max_range_mm,confidence=confidence)


def write_confident_point_cloud(path: Path, cloud, confidence: np.ndarray, *, temporal_confidence: bool = False):
    """Binary PLY extra property survives export; XYZ remains float64 metres.

    Legacy Open3D can display this PLY but discards custom properties on read;
    fusion must use the original depth+confidence pair, not a plain PLY reload.
    """
    points, colors = np.asarray(cloud.points), np.asarray(cloud.colors)
    confidence = np.asarray(confidence,dtype=np.float32)
    if confidence.shape != (len(points),) or colors.shape != points.shape:
        raise ValueError('Every XYZRGB point must carry one confidence')
    allowed = (np.isfinite(confidence) & (confidence > 0) & (confidence <= 1)) if temporal_confidence else np.isin(confidence,np.array([.3,1],np.float32))
    if not np.isfinite(points).all() or not allowed.all():
        raise ValueError('Unknown/non-finite points cannot be exported')
    dtype = np.dtype([(k,'<f8') for k in ('x','y','z')]+[(k,'u1') for k in ('red','green','blue')]+[('confidence','<f4')])
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:
        provenance = 'temporal confidence [0,1]; repair provenance retained in source layer' if temporal_confidence else 'confidence original=1 repair=0.3'
        stream.write(('ply\nformat binary_little_endian 1.0\ncomment XYZ metres; '+provenance+'\n'
                      f'element vertex {len(points)}\nproperty double x\nproperty double y\nproperty double z\n'
                      'property uchar red\nproperty uchar green\nproperty uchar blue\nproperty float confidence\nend_header\n').encode('ascii'))
        for start in range(0,len(points),100000):
            end = min(start+100000,len(points))
            block = np.empty(end-start,dtype=dtype)
            for i,k in enumerate(('x','y','z')):
                block[k] = points[start:end,i]
            rgb = np.rint(np.clip(colors[start:end],0,1)*255).astype(np.uint8)
            for i,k in enumerate(('red','green','blue')):
                block[k] = rgb[:,i]
            block['confidence'] = confidence[start:end]
            stream.write(block.tobytes())
