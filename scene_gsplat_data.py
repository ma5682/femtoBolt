"""Femto/Nerfstudio dataset reader for the direct gsplat product trainer.

The converter stores OpenGL/RUB camera-to-world matrices because that is what
Nerfstudio consumes.  gsplat projects along camera +Z, so this reader converts
the matrices back to the original Orbbec/OpenCV RDF convention before forming
world-to-camera view matrices.  Source RGB-D files are never modified.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image
from scene_gsplat_split import validate_split, file_sha256


CV_CAMERA_TO_OPENGL_CAMERA = np.diag([1.0, -1.0, -1.0, 1.0])


@dataclass(frozen=True)
class GsplatFrame:
    index: int
    rgb_path: Path
    depth_path: Path
    K: np.ndarray
    viewmat_rdf: np.ndarray
    camera_to_world_rdf: np.ndarray
    width: int
    height: int
    source_frame: int | None
    confidence_path: Path | None = None
    original_depth_path: Path | None = None


@dataclass(frozen=True)
class GsplatDataset:
    root: Path
    transforms_path: Path
    seed_path: Path
    frames: tuple[GsplatFrame, ...]
    train_indices: tuple[int, ...]
    eval_indices: tuple[int, ...]
    depth_unit_m: float
    fingerprint: str
    metadata: dict[str, Any]
    test_indices: tuple[int, ...] = ()


def _resolve_child(root: Path, relative: str, label: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes the dataset directory: {relative}") from error
    if not candidate.is_file():
        raise ValueError(f"Missing {label}: {candidate}")
    return candidate


def _scaled_intrinsic(payload: dict[str, Any], downscale: int) -> tuple[np.ndarray, int, int]:
    if downscale < 1:
        raise ValueError("downscale must be at least 1")
    source_width = int(payload["w"])
    source_height = int(payload["h"])
    if source_width <= 0 or source_height <= 0:
        raise ValueError("Camera image dimensions must be positive")
    width = max(1, source_width // downscale)
    height = max(1, source_height // downscale)
    sx = width / source_width
    sy = height / source_height
    K = np.array(
        [
            [float(payload["fl_x"]) * sx, 0.0, float(payload["cx"]) * sx],
            [0.0, float(payload["fl_y"]) * sy, float(payload["cy"]) * sy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    if not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError("Camera intrinsics must be finite with positive focal lengths")
    return K, width, height


def opengl_c2w_to_rdf(c2w_opengl: np.ndarray) -> np.ndarray:
    matrix = np.asarray(c2w_opengl, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Every transform_matrix must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-5):
        raise ValueError("Invalid camera homogeneous transform")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-2) or not np.isclose(np.linalg.det(rotation), 1.0, atol=2e-2):
        raise ValueError("Camera pose rotation must be orthonormal with determinant +1")
    return matrix @ CV_CAMERA_TO_OPENGL_CAMERA


def _hash_file(digest: Any, path: Path) -> None:
    digest.update(path.name.encode("utf-8"))
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)


def _fingerprint_files(transforms_path: Path, seed_path: Path, frames: Iterable[GsplatFrame]) -> str:
    """Hash every RGB-D input byte so resume can never cross data revisions."""
    digest = hashlib.sha256()
    _hash_file(digest, transforms_path)
    _hash_file(digest, seed_path)
    for frame in frames:
        for path in (frame.rgb_path, frame.depth_path, frame.confidence_path, frame.original_depth_path):
            if path is None:
                continue
            _hash_file(digest, path)
    return digest.hexdigest()


def load_gsplat_dataset(
    root: Path,
    *,
    downscale: int = 1,
    eval_stride: int = 8,
    minimum_frames: int = 2,
) -> GsplatDataset:
    root = root.resolve()
    transforms_path = root / "transforms.json"
    if not transforms_path.is_file():
        raise ValueError(f"Missing transforms.json: {transforms_path}")
    payload = json.loads(transforms_path.read_text(encoding="utf-8-sig"))
    if str(payload.get("camera_model", "")).upper() != "OPENCV":
        raise ValueError("Direct trainer currently requires rectified OPENCV camera input")
    for coefficient in ("k1", "k2", "k3", "k4", "p1", "p2"):
        if abs(float(payload.get(coefficient, 0.0))) > 1e-8:
            raise ValueError("Input must be rectified before direct gsplat training")
    K, width, height = _scaled_intrinsic(payload, int(downscale))
    depth_unit_m = float(payload.get("depth_unit_scale_factor", 0.001))
    if not np.isclose(depth_unit_m, 0.001, rtol=0, atol=1e-12):
        raise ValueError("Femto Depth PNG must declare millimetres (depth_unit_scale_factor=0.001)")
    seed_name = str(payload.get("ply_file_path", "")).strip()
    if not seed_name:
        raise ValueError("transforms.json does not reference an RGB-D initialization PLY")
    seed_path = _resolve_child(root, seed_name, "RGB-D initialization PLY")
    raw_frames = payload.get("frames")
    if not isinstance(raw_frames, list) or len(raw_frames) < minimum_frames:
        raise ValueError(f"Direct gsplat training needs at least {minimum_frames} frames")

    frames: list[GsplatFrame] = []
    for index, item in enumerate(raw_frames):
        if not isinstance(item, dict):
            raise ValueError(f"Invalid frame record at index {index}")
        rgb = _resolve_child(root, str(item.get("file_path", "")), "RGB image")
        depth = _resolve_child(root, str(item.get("depth_file_path", "")), "Depth image")
        confidence_name = str(item.get("depth_confidence_file_path", "")).strip()
        original_depth_name = str(item.get("original_depth_file_path", "")).strip()
        confidence = _resolve_child(root, confidence_name, "depth confidence") if confidence_name else None
        original_depth = _resolve_child(root, original_depth_name, "original Depth image") if original_depth_name else None
        for image_path in (rgb, depth):
            with Image.open(image_path) as image:
                if image.size != (int(payload["w"]), int(payload["h"])):
                    raise ValueError(f"Image dimensions do not match calibration: {image_path}")
                if image_path == depth and image.mode not in ("I;16", "I;16L", "I;16B"):
                    raise ValueError(f"Depth PNG must be uint16 millimetres: {image_path}")
        c2w_rdf = opengl_c2w_to_rdf(np.asarray(item.get("transform_matrix"), dtype=np.float64))
        viewmat = np.linalg.inv(c2w_rdf)
        source_frame = item.get("source_frame")
        frames.append(
            GsplatFrame(
                index=index,
                rgb_path=rgb,
                depth_path=depth,
                K=K.copy(),
                viewmat_rdf=viewmat.astype(np.float32),
                camera_to_world_rdf=c2w_rdf.astype(np.float32),
                width=width,
                height=height,
                source_frame=int(source_frame) if source_frame is not None else None,
                confidence_path=confidence,
                original_depth_path=original_depth,
            )
        )

    if eval_stride < 2:
        raise ValueError("eval_stride must be at least 2")
    eval_indices = tuple(i for i in range(len(frames)) if (i + 1) % eval_stride == 0)
    if not eval_indices:
        eval_indices = (len(frames) - 1,)
    train_indices = tuple(i for i in range(len(frames)) if i not in set(eval_indices))
    test_indices = ()
    split = payload.get('dataset_split')
    if split is not None and len(frames) >= 3:
        validate_split(split, len(frames))
        if file_sha256(seed_path) != split.get('seed_sha256'):
            raise ValueError('Seed PLY hash does not match training-only initialization provenance')
        train_indices = tuple(split['train'])
        eval_indices = tuple(split['validation'])
        test_indices = tuple(split['test'])
    if not train_indices:
        raise ValueError("Evaluation split left no training frames")
    fingerprint = _fingerprint_files(transforms_path, seed_path, frames)
    return GsplatDataset(
        root=root,
        transforms_path=transforms_path,
        seed_path=seed_path,
        frames=tuple(frames),
        train_indices=train_indices,
        eval_indices=eval_indices,
        depth_unit_m=depth_unit_m,
        fingerprint=fingerprint,
        metadata=payload,
        test_indices=test_indices,
    )


def load_frame_images(frame: GsplatFrame, depth_unit_m: float, *, observed_only: bool = False) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(frame.rgb_path) as image:
        rgb_image = image.convert("RGB")
        if rgb_image.size != (frame.width, frame.height):
            rgb_image = rgb_image.resize((frame.width, frame.height), Image.Resampling.BILINEAR)
        rgb = np.asarray(rgb_image, dtype=np.float32) / 255.0
    selected_depth = frame.original_depth_path if observed_only and frame.original_depth_path else frame.depth_path
    with Image.open(selected_depth) as image:
        if image.size != (frame.width, frame.height):
            image = image.resize((frame.width, frame.height), Image.Resampling.NEAREST)
        depth_raw = np.asarray(image)
    if depth_raw.ndim != 2 or not np.issubdtype(depth_raw.dtype, np.integer):
        raise ValueError(f"Depth image must be a single-channel integer image: {frame.depth_path}")
    depth_m = depth_raw.astype(np.float32) * float(depth_unit_m)
    return np.ascontiguousarray(rgb), np.ascontiguousarray(depth_m)


def load_frame_confidence(frame: GsplatFrame) -> np.ndarray:
    """Load optional float confidence; legacy inputs derive 1/0 from Depth."""
    if frame.confidence_path is None:
        _, depth = load_frame_images(frame, 1.0)
        return np.ascontiguousarray((depth > 0).astype(np.float32))
    confidence = np.load(frame.confidence_path, allow_pickle=False)
    if confidence.ndim != 2 or confidence.shape != (frame.height if frame.height else confidence.shape[0], frame.width if frame.width else confidence.shape[1]):
        if confidence.ndim != 2:
            raise ValueError(f"Depth confidence must be 2-D: {frame.confidence_path}")
        image = Image.fromarray(confidence.astype(np.float32))
        confidence = np.asarray(image.resize((frame.width, frame.height), Image.Resampling.NEAREST), dtype=np.float32)
    confidence = np.asarray(confidence, dtype=np.float32)
    if not np.isfinite(confidence).all() or np.any(confidence < 0) or np.any(confidence > 1):
        raise ValueError(f"Depth confidence must be finite in [0,1]: {frame.confidence_path}")
    # Continuous temporal confidence is preserved; no discretization/promotion.
    return np.ascontiguousarray(confidence)


def read_ply_vertex_records(path: Path) -> np.ndarray:
    """Read the vertex element of an ASCII/binary little-endian PLY into a record array.

    Shared by the XYZRGB seed cloud and the trained Gaussian model, which use the
    same writer conventions but very different property sets.
    """
    with path.open("rb") as stream:
        header_lines: list[str] = []
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"PLY header is incomplete: {path}")
            decoded = line.decode("ascii", errors="strict").strip()
            header_lines.append(decoded)
            if decoded == "end_header":
                break
        if header_lines[0] != "ply":
            raise ValueError(f"Not a PLY file: {path}")
        format_line = next((line for line in header_lines if line.startswith("format ")), "")
        vertex_line = next((line for line in header_lines if line.startswith("element vertex ")), "")
        if not vertex_line:
            raise ValueError("PLY does not declare vertices")
        count = int(vertex_line.split()[-1])
        properties: list[tuple[str, str]] = []
        in_vertex = False
        for line in header_lines:
            if line.startswith("element "):
                in_vertex = line.startswith("element vertex ")
            elif in_vertex and line.startswith("property "):
                parts = line.split()
                if len(parts) != 3 or parts[1] == "list":
                    raise ValueError("List properties are not supported in this PLY reader")
                properties.append((parts[2], parts[1]))
        type_map = {
            "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
            "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
            "ushort": "<u2", "uint16": "<u2", "int": "<i4", "int32": "<i4",
        }
        try:
            dtype = np.dtype([(name, type_map[kind]) for name, kind in properties])
        except KeyError as error:
            raise ValueError(f"Unsupported PLY scalar type: {error.args[0]}") from error
        if format_line == "format binary_little_endian 1.0":
            records = np.fromfile(stream, dtype=dtype, count=count)
        elif format_line == "format ascii 1.0":
            values = np.loadtxt(stream, dtype=np.float64, max_rows=count, ndmin=2)
            if values.shape != (count, len(properties)):
                raise ValueError("ASCII PLY vertex row count does not match its header")
            records = np.empty(count, dtype=dtype)
            for column, (name, _) in enumerate(properties):
                records[name] = values[:, column]
        else:
            raise ValueError(f"Unsupported PLY format: {format_line}")
    if len(records) != count:
        raise ValueError("PLY vertex payload is truncated")
    return records


def read_xyzrgb_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the ASCII/binary little-endian XYZRGB seed PLY written by this project."""
    records = read_ply_vertex_records(path)
    names = set(records.dtype.names or ())
    if not {"x", "y", "z"}.issubset(names):
        raise ValueError("Seed PLY is missing XYZ properties")
    points = np.column_stack([records[name] for name in ("x", "y", "z")]).astype(np.float32)
    if {"red", "green", "blue"}.issubset(names):
        colors = np.column_stack([records[name] for name in ("red", "green", "blue")]).astype(np.float32)
        if colors.size and float(np.nanmax(colors)) > 1.0:
            colors /= 255.0
    else:
        colors = np.full_like(points, 0.65, dtype=np.float32)
    valid = np.isfinite(points).all(axis=1) & np.isfinite(colors).all(axis=1)
    points = points[valid]
    colors = np.clip(colors[valid], 0.0, 1.0)
    if not len(points):
        raise ValueError("Seed PLY contains no finite points")
    return np.ascontiguousarray(points), np.ascontiguousarray(colors)
