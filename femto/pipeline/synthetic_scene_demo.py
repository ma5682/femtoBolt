"""Generate a deterministic RGB-D multi-view scene with ground-truth poses.

The synthetic dataset exists to test the complete reconstruction contract
without depending on camera motion, pose tracking, or a person being present.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np
import open3d as o3d

from femto.geometry.fusion import export_mesh_formats
from femto.io_utils import imwrite_unicode as _imwrite, write_json as _write_json, write_jsonl as _write_jsonl


from femto.paths import REPO_ROOT as ROOT
DEFAULT_DATASETS = ROOT / "datasets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a known-pose full-scene RGB-D test dataset")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_DATASETS)
    parser.add_argument("--frames", type=int, default=14)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=270)
    return parser.parse_args()


def _look_at_rdf(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return camera-to-world with OpenCV/Orbbec right-down-forward axes."""
    forward = target - eye
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    down /= np.linalg.norm(down)
    T_WC = np.eye(4, dtype=np.float64)
    T_WC[:3, :3] = np.column_stack((right, down, forward))
    T_WC[:3, 3] = eye
    return T_WC


def _ellipsoid(scale: tuple[float, float, float], center: tuple[float, float, float]) -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=24)
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices) * np.asarray(scale))
    mesh.translate(center)
    mesh.compute_vertex_normals()
    return mesh


def _build_scene() -> tuple[o3d.t.geometry.RaycastingScene, o3d.geometry.TriangleMesh, dict[int, np.ndarray]]:
    objects: list[tuple[o3d.geometry.TriangleMesh, np.ndarray]] = []
    floor = o3d.geometry.TriangleMesh.create_box(3.2, 0.05, 3.2).translate([-1.6, 0.72, 0.15])
    bed = o3d.geometry.TriangleMesh.create_box(1.65, 0.18, 1.85).translate([-0.825, 0.39, 0.75])
    pillow = _ellipsoid((0.34, 0.11, 0.22), (0.0, 0.25, 1.02))
    torso = _ellipsoid((0.34, 0.16, 0.46), (0.0, 0.15, 1.60))
    head = _ellipsoid((0.20, 0.16, 0.23), (0.0, 0.08, 1.10))
    left_arm = _ellipsoid((0.12, 0.11, 0.43), (-0.39, 0.20, 1.62))
    right_arm = _ellipsoid((0.12, 0.11, 0.43), (0.39, 0.20, 1.62))
    left_leg = _ellipsoid((0.15, 0.13, 0.50), (-0.17, 0.22, 2.28))
    right_leg = _ellipsoid((0.15, 0.13, 0.50), (0.17, 0.22, 2.28))
    equipment = o3d.geometry.TriangleMesh.create_box(0.33, 0.55, 0.42).translate([0.83, 0.12, 1.12])
    objects.extend(
        [
            (floor, np.array([0.34, 0.38, 0.42])),
            (bed, np.array([0.25, 0.48, 0.72])),
            (pillow, np.array([0.78, 0.80, 0.72])),
            (torso, np.array([0.12, 0.34, 0.58])),
            (head, np.array([0.72, 0.48, 0.36])),
            (left_arm, np.array([0.14, 0.36, 0.60])),
            (right_arm, np.array([0.14, 0.36, 0.60])),
            (left_leg, np.array([0.10, 0.28, 0.50])),
            (right_leg, np.array([0.10, 0.28, 0.50])),
            (equipment, np.array([0.78, 0.24, 0.10])),
        ]
    )
    ray_scene = o3d.t.geometry.RaycastingScene()
    combined = o3d.geometry.TriangleMesh()
    colors: dict[int, np.ndarray] = {}
    for mesh, color in objects:
        mesh.compute_vertex_normals()
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.tile(color, (len(mesh.vertices), 1)))
        geometry_id = ray_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        colors[int(geometry_id)] = color
        combined += mesh
    combined.compute_vertex_normals()
    return ray_scene, combined, colors


def _render(
    scene: o3d.t.geometry.RaycastingScene,
    colors: dict[int, np.ndarray],
    K: np.ndarray,
    T_WC: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    rays = scene.create_rays_pinhole(
        o3d.core.Tensor(K, dtype=o3d.core.Dtype.Float64),
        o3d.core.Tensor(np.linalg.inv(T_WC), dtype=o3d.core.Dtype.Float64),
        width,
        height,
    )
    answer = scene.cast_rays(rays)
    rays_np = rays.numpy().astype(np.float64)
    t_hit = answer["t_hit"].numpy().astype(np.float64)
    ids = answer["geometry_ids"].numpy().astype(np.uint32)
    normals = answer["primitive_normals"].numpy().astype(np.float64)
    valid = np.isfinite(t_hit)
    safe_t_hit = np.where(valid, t_hit, 0.0)
    hit_world = rays_np[..., :3] + rays_np[..., 3:] * safe_t_hit[..., None]
    hit_h = np.concatenate((hit_world, np.ones((*hit_world.shape[:2], 1))), axis=2)
    hit_camera = hit_h @ np.linalg.inv(T_WC).T
    depth_m = np.where(valid, hit_camera[..., 2], 0.0)
    valid &= (depth_m >= 0.5) & (depth_m <= 3.86)
    depth_u16 = np.where(valid, np.rint(depth_m * 1000.0), 0).astype(np.uint16)

    rgb = np.full((height, width, 3), 0.035, dtype=np.float64)
    light = np.array([-0.25, -0.65, -0.72], dtype=np.float64)
    light /= np.linalg.norm(light)
    shading = np.clip(0.60 + 0.40 * np.abs(np.sum(normals * light, axis=2)), 0.45, 1.0)
    texture = 0.86 + 0.14 * np.sin(18.0 * hit_world[..., 0] + 13.0 * hit_world[..., 2])
    for geometry_id, base in colors.items():
        mask = valid & (ids == geometry_id)
        rgb[mask] = np.clip(base * (shading[mask] * texture[mask])[:, None], 0.0, 1.0)
    rgb_u8 = np.rint(rgb * 255.0).astype(np.uint8)
    return np.ascontiguousarray(rgb_u8[:, :, ::-1]), depth_u16


def generate_dataset(output_root: Path, *, frames: int = 14, width: int = 480, height: int = 270) -> Path:
    if frames < 4 or width < 128 or height < 96:
        raise ValueError("Synthetic demo needs at least 4 frames and 128x96 pixels")
    output_root = output_root.resolve()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    dataset = output_root / f"demo_scene_{timestamp}"
    suffix = 1
    while dataset.exists():
        dataset = output_root / f"demo_scene_{timestamp}_{suffix:02d}"
        suffix += 1
    for child in ("rgb", "depth", "pose", "calibration", "ground_truth"):
        (dataset / child).mkdir(parents=True, exist_ok=False)

    fx = width * 0.86
    fy = width * 0.86
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    intrinsic = {"width": width, "height": height, "fx": fx, "fy": fy, "cx": cx, "cy": cy}
    zero_distortion = {
        "model": "OBCameraDistortionModel.BROWN_CONRADY_K6",
        "k1": 0.0, "k2": 0.0, "k3": 0.0, "k4": 0.0, "k5": 0.0, "k6": 0.0,
        "p1": 0.0, "p2": 0.0,
    }
    _write_json(dataset / "calibration" / "color_intrinsic.json", intrinsic)
    _write_json(dataset / "calibration" / "depth_intrinsic.json", intrinsic)
    _write_json(
        dataset / "calibration" / "depth_to_color.json",
        {"rotation": np.eye(3).tolist(), "translation_mm": [0.0, 0.0, 0.0]},
    )
    _write_json(dataset / "calibration" / "distortion.json", {"rgb": zero_distortion, "depth": zero_distortion})
    _write_json(
        dataset / "calibration" / "factory_calibration.json",
        {
            "source": "synthetic_ground_truth",
            "color_intrinsic": intrinsic,
            "depth_intrinsic": intrinsic,
            "rgb_distortion": zero_distortion,
            "depth_distortion": zero_distortion,
            "depth_to_color": {"rotation": np.eye(3).tolist(), "translation_mm": [0.0, 0.0, 0.0]},
        },
    )
    _write_json(dataset / "calibration" / "camera_metadata.json", {"device": "synthetic_test_camera"})

    scene, ground_truth, colors = _build_scene()
    export_mesh_formats(
        ground_truth,
        dataset / "ground_truth",
        stem="synthetic_scene_ground_truth_m",
        report_name="ground_truth_mesh_report.json",
        note="Analytic synthetic test geometry used only for software verification.",
    )
    target = np.array([0.0, 0.20, 1.72], dtype=np.float64)
    records: list[dict[str, Any]] = []
    preview_images: list[np.ndarray] = []
    for index in range(frames):
        phase = index / (frames - 1)
        angle = np.deg2rad(-23.0 + 46.0 * phase)
        eye = np.array([0.76 * np.sin(angle), -0.34 + 0.05 * np.cos(phase * np.pi), 0.18 + 0.14 * (1.0 - np.cos(angle))])
        T_WC = _look_at_rdf(eye, target)
        color_bgr, depth_u16 = _render(scene, colors, K, T_WC, width, height)
        stem = f"{index:06d}"
        rgb_relative = Path("rgb") / f"{stem}.jpg"
        depth_relative = Path("depth") / f"{stem}.png"
        pose_relative = Path("pose") / f"{stem}.json"
        _imwrite(dataset / rgb_relative, color_bgr, [cv2.IMWRITE_JPEG_QUALITY, 96])
        _imwrite(dataset / depth_relative, depth_u16)
        _write_json(
            dataset / pose_relative,
            {
                "schema": "femto-scene-frame-pose-v1",
                "frame": index,
                "pose_valid": True,
                "T_WC": T_WC.tolist(),
                "pose_source": "synthetic_ground_truth",
                "coordinate_system": "+X right, +Y down, +Z forward; metres",
            },
        )
        timestamp_us = 1_000_000 + index * 100_000
        records.append(
            {
                "frame": index,
                "camera_frame_index": index,
                "camera_timestamp_us": timestamp_us,
                "system_timestamp_us": timestamp_us,
                "color_timestamp_us": timestamp_us,
                "color_system_timestamp_us": timestamp_us,
                "depth_timestamp_us": timestamp_us,
                "depth_system_timestamp_us": timestamp_us,
                "rgb_depth_timestamp_delta_us": 0,
                "rgb": rgb_relative.as_posix(),
                "depth": depth_relative.as_posix(),
                "pose": pose_relative.as_posix(),
                "pose_valid": True,
            }
        )
        if index in {0, frames // 2, frames - 1}:
            preview_images.append(color_bgr)
    _write_jsonl(dataset / "frames.jsonl", records)
    manifest = {
        "schema": "femto-full-scene-rgbd-v1",
        "created_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "status": "synthetic_ground_truth_complete",
        "scene_scope": "full synthetic camera field; no semantic mask",
        "frame_count": frames,
        "camera": {"device": "synthetic_test_camera", "intrinsics": intrinsic},
        "pose": {"valid": True, "source": "synthetic_ground_truth"},
        "parameters": {"frames": frames, "width": width, "height": height},
        "files": {"rgb": "rgb/*.jpg", "depth": "depth/*.png", "pose": "pose/*.json", "frames": "frames.jsonl"},
        "purpose": "deterministic end-to-end software demo; not Femto sensor evidence",
    }
    _write_json(dataset / "manifest.json", manifest)
    if preview_images:
        _imwrite(dataset / "synthetic_preview.jpg", np.hstack(preview_images), [cv2.IMWRITE_JPEG_QUALITY, 94])
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "latest_demo_scene.txt").write_text(str(dataset), encoding="utf-8")
    return dataset


def main() -> int:
    args = parse_args()
    try:
        dataset = generate_dataset(args.output_root, frames=args.frames, width=args.width, height=args.height)
    except (OSError, ValueError, RuntimeError) as error:
        print(json.dumps({"status": "blocked", "error": str(error)}, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps({"status": "generated", "dataset": str(dataset)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
