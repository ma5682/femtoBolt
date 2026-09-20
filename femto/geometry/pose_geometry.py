"""Pure NumPy geometry for RGB-D human landmarks and surface points."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import numpy as np


LANDMARK_NAMES = (
    "nose",
    "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
)

# MediaPipe Pose topology. Connections are kept locally so geometry/export does
# not depend on a particular MediaPipe package layout.
POSE_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
)

TORSO_LANDMARKS = (11, 12, 23, 24)


@dataclass(frozen=True)
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Landmark2D:
    x: float
    y: float
    visibility: float = 1.0
    presence: float = 1.0


def robust_depth_mm(
    depth_mm: np.ndarray,
    u: int,
    v: int,
    *,
    radius: int = 5,
    min_depth_mm: float = 500.0,
    max_depth_mm: float = 3860.0,
    cluster_half_width_mm: float = 60.0,
    mask: np.ndarray | None = None,
) -> tuple[float | None, int]:
    """Estimate landmark depth from a small patch, preferring the foreground.

    Joint pixels often fall on a body/background boundary. The 15th percentile
    supplies a foreground seed, then a local depth cluster is median-reduced.
    This is less sensitive to a single near fly-point than taking the minimum.
    """
    if depth_mm.ndim != 2:
        raise ValueError("depth_mm must be a 2-D array")
    height, width = depth_mm.shape
    if not (0 <= u < width and 0 <= v < height):
        return None, 0
    x0, x1 = max(0, u - radius), min(width, u + radius + 1)
    y0, y1 = max(0, v - radius), min(height, v + radius + 1)
    patch = depth_mm[y0:y1, x0:x1].astype(np.float32, copy=False)
    values = patch.ravel()
    if mask is not None:
        if mask.shape != depth_mm.shape:
            raise ValueError("depth mask must share the depth pixel grid")
        values = values[mask[y0:y1, x0:x1].astype(bool).ravel()]
    values = values[
        np.isfinite(values)
        & (values >= min_depth_mm)
        & (values <= max_depth_mm)
    ]
    if values.size < 3:
        return None, int(values.size)

    center = float(depth_mm[v, u])
    if min_depth_mm <= center <= max_depth_mm:
        seed = center
    else:
        seed = float(np.percentile(values, 15.0))
    cluster = values[np.abs(values - seed) <= cluster_half_width_mm]
    if cluster.size < 3:
        cluster = values
    return float(np.median(cluster)), int(cluster.size)


def back_project_pixel(u: float, v: float, depth_mm: float, intr: Intrinsics) -> np.ndarray:
    """Back-project a color pixel to Orbbec camera coordinates in metres.

    Coordinate convention: +X right, +Y down, +Z forward.
    """
    z_m = float(depth_mm) / 1000.0
    return np.array(
        [
            (float(u) - intr.cx) * z_m / intr.fx,
            (float(v) - intr.cy) * z_m / intr.fy,
            z_m,
        ],
        dtype=np.float64,
    )


def landmarks_to_3d(
    landmarks: Sequence[Landmark2D],
    depth_mm: np.ndarray,
    intr: Intrinsics,
    *,
    visibility_threshold: float = 0.45,
    presence_threshold: float = 0.45,
    depth_radius: int = 5,
    min_depth_mm: float = 500.0,
    max_depth_mm: float = 3860.0,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fuse normalized 2-D landmarks with aligned depth.

    Returns ``points_m`` (N×3, NaN when invalid), integer ``pixels`` (N×2),
    and the number of depth samples used for each joint.
    """
    count = len(landmarks)
    points = np.full((count, 3), np.nan, dtype=np.float64)
    pixels = np.full((count, 2), -1, dtype=np.int32)
    samples = np.zeros(count, dtype=np.int32)
    height, width = depth_mm.shape
    for index, landmark in enumerate(landmarks):
        u = int(round(float(landmark.x) * (width - 1)))
        v = int(round(float(landmark.y) * (height - 1)))
        pixels[index] = (u, v)
        if (
            landmark.visibility < visibility_threshold
            or landmark.presence < presence_threshold
        ):
            continue
        value, sample_count = robust_depth_mm(
            depth_mm,
            u,
            v,
            radius=depth_radius,
            min_depth_mm=min_depth_mm,
            max_depth_mm=max_depth_mm,
            mask=mask,
        )
        samples[index] = sample_count
        if value is not None:
            points[index] = back_project_pixel(u, v, value, intr)
    return points, pixels, samples


def valid_point(point: np.ndarray) -> bool:
    return bool(np.isfinite(point).all())


def _average(points: np.ndarray, indexes: Iterable[int]) -> np.ndarray | None:
    selected = np.asarray([points[index] for index in indexes], dtype=np.float64)
    valid = np.isfinite(selected).all(axis=1)
    if not valid.any():
        return None
    return np.mean(selected[valid], axis=0)


def _between(a: np.ndarray | None, b: np.ndarray | None, t: float) -> np.ndarray | None:
    if a is None or b is None:
        return None
    return a * (1.0 - t) + b * t


def derive_patient_regions(points_m: np.ndarray) -> dict[str, dict]:
    """Create coarse anatomical reference points from the 33 pose joints.

    These are navigation labels for visual review. They are not treatment
    targets and must not be used as clinical boundaries without validation.
    """
    if points_m.shape != (33, 3):
        raise ValueError("MediaPipe patient region mapping requires 33 landmarks")
    shoulder_center = _average(points_m, (11, 12))
    hip_center = _average(points_m, (23, 24))
    head = _average(points_m, (0, 2, 5, 7, 8))
    values: dict[str, tuple[str, np.ndarray | None]] = {
        "head": ("头", head),
        "left_shoulder": ("左肩", points_m[11] if valid_point(points_m[11]) else None),
        "right_shoulder": ("右肩", points_m[12] if valid_point(points_m[12]) else None),
        "chest_center": ("胸", _between(shoulder_center, hip_center, 0.35)),
        "abdomen_center": ("腹", _between(shoulder_center, hip_center, 0.62)),
        "waist_center": ("腰", _between(shoulder_center, hip_center, 0.78)),
        "hip_center": ("髋", hip_center),
        "left_hand": ("左手", points_m[15] if valid_point(points_m[15]) else None),
        "right_hand": ("右手", points_m[16] if valid_point(points_m[16]) else None),
        "left_knee": ("左膝", points_m[25] if valid_point(points_m[25]) else None),
        "right_knee": ("右膝", points_m[26] if valid_point(points_m[26]) else None),
        "left_ankle": ("左踝", points_m[27] if valid_point(points_m[27]) else None),
        "right_ankle": ("右踝", points_m[28] if valid_point(points_m[28]) else None),
    }
    result: dict[str, dict] = {}
    for key, (label, point) in values.items():
        result[key] = {
            "label_zh": label,
            "xyz_m": None if point is None else [float(value) for value in point],
            "status": "observed_or_interpolated" if point is not None else "unavailable",
        }
    return result


def torso_motion_mm(current: np.ndarray, reference: np.ndarray) -> float | None:
    """Median displacement of visible shoulders/hips between two frames."""
    current_torso = current[list(TORSO_LANDMARKS)]
    reference_torso = reference[list(TORSO_LANDMARKS)]
    valid = np.isfinite(current_torso).all(axis=1) & np.isfinite(reference_torso).all(axis=1)
    if valid.sum() < 2:
        return None
    distances_m = np.linalg.norm(current_torso[valid] - reference_torso[valid], axis=1)
    return float(np.median(distances_m) * 1000.0)


def masked_depth_residual_mm(
    current: np.ndarray,
    reference: np.ndarray,
    current_mask: np.ndarray,
    reference_mask: np.ndarray,
    *,
    stride: int = 8,
) -> float | None:
    """Median absolute depth residual over the common patient mask."""
    if current.shape != reference.shape:
        return None
    valid = (
        current_mask.astype(bool)
        & reference_mask.astype(bool)
        & np.isfinite(current)
        & np.isfinite(reference)
        & (current > 0)
        & (reference > 0)
    )
    values = np.abs(current[::stride, ::stride] - reference[::stride, ::stride])
    sampled_valid = valid[::stride, ::stride]
    if sampled_valid.sum() < 100:
        return None
    return float(np.median(values[sampled_valid]))


def depth_to_points(
    depth_mm: np.ndarray,
    intr: Intrinsics,
    *,
    mask: np.ndarray | None = None,
    stride: int = 3,
    min_depth_mm: float = 500.0,
    max_depth_mm: float = 3860.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized aligned-depth back-projection.

    Returns points in metres and the corresponding pixel coordinates ``u,v``.
    """
    height, width = depth_mm.shape
    vv, uu = np.mgrid[0:height:stride, 0:width:stride]
    depth = depth_mm[::stride, ::stride]
    valid = (
        np.isfinite(depth)
        & (depth >= min_depth_mm)
        & (depth <= max_depth_mm)
    )
    if mask is not None:
        valid &= mask[::stride, ::stride].astype(bool)
    z = depth[valid].astype(np.float64) / 1000.0
    u = uu[valid].astype(np.float64)
    v = vv[valid].astype(np.float64)
    points = np.column_stack(
        ((u - intr.cx) * z / intr.fx, (v - intr.cy) * z / intr.fy, z)
    )
    pixels = np.column_stack((u.astype(np.int32), v.astype(np.int32)))
    return points, pixels


def serialise_joints(
    points_m: np.ndarray,
    pixels: np.ndarray | None = None,
    visibility: Sequence[float] | None = None,
    presence: Sequence[float] | None = None,
) -> list[dict]:
    records = []
    for index, name in enumerate(LANDMARK_NAMES):
        valid = valid_point(points_m[index])
        record = {
            "index": index,
            "name": name,
            "xyz_m": [float(value) for value in points_m[index]] if valid else None,
            "valid_depth": valid,
        }
        if pixels is not None:
            record["pixel_uv"] = [int(value) for value in pixels[index]]
        if visibility is not None:
            record["visibility"] = float(visibility[index])
        if presence is not None:
            record["presence"] = float(presence[index])
        records.append(record)
    return records
