"""Interactive renderer for trained direct-gsplat models.

The Open3D desktop viewer (``scene_demo_viewer.py``) can only draw Gaussian
*centres* as unlit points, which is a diagnostic display, not Gaussian Splatting.
This viewer instead evaluates the trained model with the very same public
``gsplat.rendering.rasterization`` entry point that produced it: anisotropic
covariance, opacity and view-dependent spherical harmonics all participate, so
what appears on screen is a real 3DGS render of the trained parameters.

It therefore runs inside ``.venv-gsplat`` (PyTorch + CUDA + gsplat), not in the
capture ``.venv``.  ``gsplat_backend.open_viewer`` launches it with that runtime.
Only the Python standard library, NumPy, Pillow and gsplat are used; no new
third-party dependency and no browser/WebGL stack is introduced.

Coordinates: the trained Gaussians live in the source ``T_WC`` world frame in
metres, using the Orbbec/OpenCV RDF camera convention (+X right, +Y down,
+Z forward).  Orbit cameras are constructed directly in that frame and inverted
into gsplat ``viewmat`` world-to-camera matrices; nothing on disk is modified.
"""
from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

from scene_gsplat_export import read_trained_gaussian_ply
from ui_i18n import LANGUAGE_NAMES, tr


ROOT = Path(__file__).resolve().parent
DEFAULT_RENDER_WIDTH = 960
MIN_PITCH_DEG = -85.0
MAX_PITCH_DEG = 85.0
MIN_RENDER_WIDTH = 160
RENDER_MEMORY_PER_GAUSSIAN = 320
RENDER_MEMORY_PER_PIXEL = 64
DEFAULT_FRAME_TIMEOUT_S = 120.0


def _append_viewer_log(log_dir: Path, payload: dict[str, Any]) -> Path:
    """Persist a viewer diagnostic even when no parent process captures stdout."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / "viewer_stdout.log"
    with path.open("a", encoding="utf-8") as log:
        log.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return path


def _viewer_log_directory(target: Path) -> Path:
    """Resolve the stable log directory for a run directory or direct PLY."""
    target = Path(target).resolve()
    if target.is_file() or target.suffix.lower() == ".ply":
        return target.parent.parent if target.parent.name.lower() == "model" else target.parent
    return target


def plan_render_budget(
    gaussian_count: int,
    width: int,
    height: int,
    *,
    free_cuda_bytes: int | None = None,
    total_cuda_bytes: int | None = None,
    max_pixels: int = 1_500_000,
) -> dict[str, Any]:
    """Plan a safe render footprint without changing the model on disk.

    The estimate deliberately includes a generous per-Gaussian rasterization
    allowance.  It is a pure function so the policy can be regression-tested
    on machines without CUDA.  ``effective_gaussian_count`` describes a
    deterministic in-memory sampling fallback only; the source PLY remains
    untouched and ``model_gaussian_count`` is always reported separately.
    """
    requested_count = max(0, int(gaussian_count))
    requested_width = max(1, int(width))
    requested_height = max(1, int(height))
    if free_cuda_bytes is None:
        # Unknown memory is treated conservatively, but not as a reason to
        # reject a valid model before the renderer can probe the device.
        available = 2 * 1024**3
    else:
        available = max(64 * 1024**2, int(free_cuda_bytes))
    budget = max(32 * 1024**2, int(available * 0.50))

    effective_width = requested_width
    effective_height = requested_height
    reasons: list[str] = []
    # Large turntable frames multiply tile/binning work.  Scale resolution
    # before considering a temporary Gaussian sample.
    max_pixels = max(1, int(max_pixels))
    pixels = effective_width * effective_height
    if pixels > max_pixels:
        factor = math.sqrt(max_pixels / pixels)
        effective_width = max(MIN_RENDER_WIDTH, int(effective_width * factor))
        effective_height = max(1, int(effective_height * factor))
        reasons.append("pixel budget")

    pixel_bytes = effective_width * effective_height * RENDER_MEMORY_PER_PIXEL
    capacity = max(1, (budget - pixel_bytes) // RENDER_MEMORY_PER_GAUSSIAN)
    effective_count = min(requested_count, int(capacity))
    if requested_count and effective_count < min(requested_count, 1024):
        # Keep a minimally useful cloud even on a nearly exhausted device;
        # the render call will still surface an OOM rather than pretending it
        # is a full-resolution/full-model result.
        effective_count = min(requested_count, 1024)
    if effective_count < requested_count:
        reasons.append("GPU memory budget")
    if effective_width != requested_width or effective_height != requested_height:
        state = "reduced_resolution" if effective_count == requested_count else "reduced_resolution_and_sampled"
    elif effective_count != requested_count:
        state = "sampled_gaussians"
    else:
        state = "full"
    estimated = effective_count * RENDER_MEMORY_PER_GAUSSIAN + effective_width * effective_height * RENDER_MEMORY_PER_PIXEL
    return {
        "state": state,
        "reason": "; ".join(reasons) if reasons else "within estimated GPU budget",
        "model_gaussian_count": requested_count,
        "effective_gaussian_count": int(effective_count),
        "requested_resolution": [requested_width, requested_height],
        "effective_resolution": [int(effective_width), int(effective_height)],
        "estimated_memory_bytes": int(estimated),
        "free_cuda_memory_bytes": int(free_cuda_bytes) if free_cuda_bytes is not None else None,
        "total_cuda_memory_bytes": int(total_cuda_bytes) if total_cuda_bytes is not None else None,
        "budget_bytes": int(budget),
        "temporary_sampling": bool(effective_count != requested_count),
    }


@dataclass
class CameraOptics:
    """Pinhole intrinsics carried over from the training dataset."""

    fl_x: float
    fl_y: float
    cx: float
    cy: float
    width: int
    height: int
    source: str

    def scaled(self, width: int, height: int) -> np.ndarray:
        sx = width / self.width
        sy = height / self.height
        return np.array(
            [[self.fl_x * sx, 0.0, self.cx * sx], [0.0, self.fl_y * sy, self.cy * sy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )


@dataclass
class OrbitCamera:
    """Orbit camera expressed directly in the RDF world frame of the model."""

    target: np.ndarray
    distance: float
    up_world: np.ndarray
    yaw_deg: float = 0.0
    pitch_deg: float = -15.0
    _basis: tuple[np.ndarray, np.ndarray] = field(init=False)

    def __post_init__(self) -> None:
        up = np.asarray(self.up_world, dtype=np.float64)
        norm = float(np.linalg.norm(up))
        self.up_world = up / norm if norm > 1e-9 else np.array([0.0, -1.0, 0.0])
        seed = np.array([1.0, 0.0, 0.0]) if abs(self.up_world[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
        first = np.cross(self.up_world, seed)
        first /= max(float(np.linalg.norm(first)), 1e-9)
        self._basis = (first, np.cross(self.up_world, first))

    def orbit(self, delta_yaw_deg: float, delta_pitch_deg: float) -> None:
        self.yaw_deg = (self.yaw_deg + delta_yaw_deg) % 360.0
        self.pitch_deg = float(np.clip(self.pitch_deg + delta_pitch_deg, MIN_PITCH_DEG, MAX_PITCH_DEG))

    def zoom(self, factor: float) -> None:
        self.distance = float(np.clip(self.distance * factor, 0.02, 1_000.0))

    def pan(self, right_m: float, down_m: float) -> None:
        c2w = self.camera_to_world_rdf()
        self.target = self.target + c2w[:3, 0] * right_m + c2w[:3, 1] * down_m

    def eye(self) -> np.ndarray:
        first, second = self._basis
        yaw = math.radians(self.yaw_deg)
        pitch = math.radians(self.pitch_deg)
        direction = (
            math.cos(pitch) * (math.cos(yaw) * first + math.sin(yaw) * second)
            + math.sin(pitch) * self.up_world
        )
        return self.target + self.distance * direction

    def camera_to_world_rdf(self) -> np.ndarray:
        """Build ``T_WC`` with RDF camera axes: +X right, +Y down, +Z forward."""
        eye = self.eye()
        forward = self.target - eye
        norm = float(np.linalg.norm(forward))
        forward = forward / norm if norm > 1e-9 else np.array([0.0, 0.0, 1.0])
        down_hint = -self.up_world
        right = np.cross(down_hint, forward)
        if float(np.linalg.norm(right)) < 1e-6:  # looking straight along the up axis
            right = np.cross(self._basis[0], forward)
        right /= max(float(np.linalg.norm(right)), 1e-9)
        down = np.cross(forward, right)
        c2w = np.eye(4)
        c2w[:3, 0] = right
        c2w[:3, 1] = down
        c2w[:3, 2] = forward
        c2w[:3, 3] = eye
        return c2w

    def viewmat(self) -> np.ndarray:
        return np.linalg.inv(self.camera_to_world_rdf())


class TrainedGaussianRenderer:
    """Thin wrapper that keeps the trained parameters resident on the GPU.

    The complete PLY is retained in CPU memory as the authoritative source.
    A deterministic in-memory sample may be selected when the device budget is
    too small; no Gaussian is removed or rewritten on disk.
    """

    def __init__(
        self,
        model_ply: Path,
        *,
        device_name: str | None = None,
        initial_width: int = DEFAULT_RENDER_WIDTH,
        initial_height: int | None = None,
        max_render_pixels: int = 1_500_000,
    ) -> None:
        import torch

        self.max_render_pixels = max_render_pixels

        arrays = read_trained_gaussian_ply(model_ply)
        coefficients = 1 + arrays["sh_rest"].shape[1]
        degree = int(round(math.sqrt(coefficients))) - 1
        if (degree + 1) ** 2 != coefficients:
            raise ValueError(f"Spherical-harmonic coefficient count {coefficients} is not a full SH band set")
        self.sh_degree = degree
        self.model_gaussian_count = len(arrays["means"])
        self.gaussian_count = self.model_gaussian_count  # backwards-compatible alias
        if device_name is None:
            if not torch.cuda.is_available():
                raise RuntimeError("Direct Gaussian rasterization requires the CUDA runtime")
            device_name = "cuda:0"
        self.device = torch.device(device_name)
        self.model_ply = Path(model_ply)
        self._torch = torch
        self._arrays = arrays
        self._active_indices = np.arange(self.model_gaussian_count, dtype=np.int64)
        self._forced_width: int | None = None
        self._forced_count: int | None = None
        self._last_requested_resolution = (max(1, int(initial_width)), max(1, int(initial_height or round(initial_width * 9 / 16))))
        free_bytes: int | None = None
        total_bytes: int | None = None
        if self.device.type == "cuda":
            try:
                free_bytes, total_bytes = (int(value) for value in torch.cuda.mem_get_info(self.device))
            except (RuntimeError, AttributeError, TypeError):
                pass
        initial_budget = plan_render_budget(
            self.model_gaussian_count,
            self._last_requested_resolution[0],
            self._last_requested_resolution[1],
            max_pixels=self.max_render_pixels,
            free_cuda_bytes=free_bytes,
            total_cuda_bytes=total_bytes,
        )
        self._budget = initial_budget
        self._activate_count(int(initial_budget["effective_gaussian_count"]))

    @property
    def budget(self) -> dict[str, Any]:
        return dict(self._budget)

    @property
    def rendered_gaussian_count(self) -> int:
        return int(len(self._active_indices))

    def _sample_indices(self, count: int) -> np.ndarray:
        count = max(1, min(int(count), self.model_gaussian_count))
        if count == self.model_gaussian_count:
            return np.arange(self.model_gaussian_count, dtype=np.int64)
        # Linspace sampling is deterministic and preserves the whole spatial
        # extent better than taking a contiguous prefix.
        return np.unique(np.rint(np.linspace(0, self.model_gaussian_count - 1, count)).astype(np.int64))

    def _activate_count(self, count: int) -> None:
        torch = self._torch
        self._active_indices = self._sample_indices(count)
        index = self._active_indices
        arrays = self._arrays
        self.means = torch.as_tensor(arrays["means"][index], device=self.device)
        self.quats = torch.as_tensor(arrays["quaternions_wxyz"][index], device=self.device)
        # Stored parameters are the raw trainer parameterization; apply the same activations.
        self.scales = torch.exp(torch.as_tensor(arrays["log_scales"][index], device=self.device))
        self.opacities = torch.sigmoid(torch.as_tensor(arrays["opacity_logits"][index], device=self.device))
        self.colors = torch.cat(
            (
                torch.as_tensor(arrays["sh0"][index], device=self.device),
                torch.as_tensor(arrays["sh_rest"][index], device=self.device),
            ),
            dim=1,
        )

    def _free_memory(self) -> tuple[int | None, int | None]:
        if self.device.type != "cuda":
            return None, None
        try:
            free, total = self._torch.cuda.mem_get_info(self.device)
            return int(free), int(total)
        except (RuntimeError, AttributeError, TypeError):
            return None, None

    def _replan(self, width: int, height: int, *, reason: str | None = None) -> None:
        free, total = self._free_memory()
        budget = plan_render_budget(
            self.model_gaussian_count,
            width,
            height,
            max_pixels=self.max_render_pixels,
            free_cuda_bytes=free,
            total_cuda_bytes=total,
        )
        if self._forced_width is not None:
            budget["effective_resolution"][0] = min(int(budget["effective_resolution"][0]), self._forced_width)
            ratio = budget["effective_resolution"][0] / max(int(budget["requested_resolution"][0]), 1)
            budget["effective_resolution"][1] = max(1, int(round(int(budget["requested_resolution"][1]) * ratio)))
        if self._forced_count is not None:
            budget["effective_gaussian_count"] = min(int(budget["effective_gaussian_count"]), self._forced_count)
        effective_count = int(budget["effective_gaussian_count"])
        effective_width, effective_height = budget["effective_resolution"]
        if effective_count != self.model_gaussian_count and effective_width != width:
            budget["state"] = "reduced_resolution_and_sampled"
        elif effective_count != self.model_gaussian_count:
            budget["state"] = "sampled_gaussians"
        elif effective_width != width:
            budget["state"] = "reduced_resolution"
        else:
            budget["state"] = "full"
        budget["temporary_sampling"] = bool(effective_count != self.model_gaussian_count)
        budget["estimated_memory_bytes"] = int(
            effective_count * RENDER_MEMORY_PER_GAUSSIAN
            + int(effective_width) * int(effective_height) * RENDER_MEMORY_PER_PIXEL
        )
        if reason:
            budget["reason"] = reason
        self._budget = budget
        if len(self._active_indices) != effective_count:
            # Release old tensors before allocating the replacement.
            self._active_indices = np.empty(0, dtype=np.int64)
            for name in ("means", "quats", "scales", "opacities", "colors"):
                if hasattr(self, name):
                    delattr(self, name)
            if self.device.type == "cuda":
                try:
                    self._torch.cuda.empty_cache()
                except RuntimeError:
                    pass
            self._activate_count(effective_count)

    def effective_resolution(self, width: int, height: int) -> tuple[int, int]:
        self._last_requested_resolution = (max(1, int(width)), max(1, int(height)))
        self._replan(width, height)
        effective = self._budget["effective_resolution"]
        return int(effective[0]), int(effective[1])

    def degrade_for_timeout(self, width: int, height: int) -> bool:
        current_width = int(self._budget.get("effective_resolution", [width, height])[0])
        if current_width > MIN_RENDER_WIDTH:
            self._forced_width = max(MIN_RENDER_WIDTH, int(current_width * 0.70))
            self._replan(width, height, reason="frame timeout; reduced render resolution")
            return True
        elif self.rendered_gaussian_count > 1024:
            self._forced_count = max(1024, self.rendered_gaussian_count // 2)
            self._replan(width, height, reason="frame timeout; temporary Gaussian sample")
            return True
        return False

    def degrade_for_oom(self, width: int, height: int) -> bool:
        current_width = int(self._budget.get("effective_resolution", [width, height])[0])
        if current_width > MIN_RENDER_WIDTH:
            self._forced_width = max(MIN_RENDER_WIDTH, int(current_width * 0.60))
            self._replan(width, height, reason="CUDA out of memory; reduced render resolution")
            return True
        if self.rendered_gaussian_count > 1024:
            self._forced_count = max(1024, self.rendered_gaussian_count // 2)
            self._replan(width, height, reason="CUDA out of memory; temporary Gaussian sample")
            return True
        return False

    def bounds(self) -> tuple[np.ndarray, float]:
        """Robust centre and 95th-percentile radius of the trained Gaussian centres."""
        # Use the complete CPU-resident model for framing even when rendering
        # is temporarily sampling a subset for GPU safety.
        points = np.asarray(self._arrays["means"], dtype=np.float32)
        centre = np.median(points, axis=0).astype(np.float64)
        radius = float(np.percentile(np.linalg.norm(points - centre, axis=1), 95.0))
        return centre, radius

    def render(self, viewmat: np.ndarray, intrinsic: np.ndarray, width: int, height: int) -> np.ndarray:
        from gsplat.rendering import rasterization

        requested_width, requested_height = max(1, int(width)), max(1, int(height))
        self.effective_resolution(requested_width, requested_height)
        while True:
            effective_width, effective_height = self._budget["effective_resolution"]
            if effective_width != requested_width or effective_height != requested_height:
                sx = effective_width / requested_width
                sy = effective_height / requested_height
                effective_intrinsic = np.asarray(intrinsic, dtype=np.float32).copy()
                effective_intrinsic[0, :] *= sx
                effective_intrinsic[1, :] *= sy
            else:
                effective_intrinsic = intrinsic
            try:
                with self._torch.no_grad():
                    renders, _, _ = rasterization(
                        means=self.means,
                        quats=self.quats,
                        scales=self.scales,
                        opacities=self.opacities,
                        colors=self.colors,
                        viewmats=self._torch.as_tensor(viewmat, dtype=self._torch.float32, device=self.device).unsqueeze(0),
                        Ks=self._torch.as_tensor(effective_intrinsic, dtype=self._torch.float32, device=self.device).unsqueeze(0),
                        width=int(effective_width),
                        height=int(effective_height),
                        sh_degree=self.sh_degree,
                        render_mode="RGB",
                        rasterize_mode="antialiased",
                        packed=True,
                    )
                    image = renders[0, ..., :3].clamp(0.0, 1.0)
                return (image.detach().cpu().numpy() * 255.0).round().astype(np.uint8)
            except RuntimeError as error:
                message = str(error).lower()
                is_oom = isinstance(error, getattr(self._torch.cuda, "OutOfMemoryError", ())) or "out of memory" in message
                if not is_oom:
                    raise
                try:
                    self._torch.cuda.empty_cache()
                except (RuntimeError, AttributeError):
                    pass
                if not self.degrade_for_oom(requested_width, requested_height):
                    raise


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _report_data_kind(report: dict[str, Any], manifest: dict[str, Any]) -> str:
    explicit = (report.get("input_provenance") or {}).get("data_kind")
    if explicit:
        return str(explicit)
    dataset_root = report.get("dataset_root") or (manifest.get("training_input") or {}).get("dataset_root")
    if dataset_root:
        try:
            from scene_gsplat_acceptance import input_provenance

            return str(input_provenance(Path(dataset_root))["data_kind"])
        except (OSError, ValueError, KeyError, TypeError):
            pass
    return "unknown"


def resolve_model(target: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Accept a direct_gsplat_* result directory or a trained PLY path."""
    target = Path(target).resolve()
    direct_ply = target.suffix.lower() == ".ply"
    if direct_ply:
        output_dir = target.parent.parent
    else:
        output_dir = target
    manifest_path = output_dir / "model" / "model_manifest.json"
    report_path = output_dir / "training_report.json"
    manifest = _load_json(manifest_path) if manifest_path.is_file() else {}
    report = _load_json(report_path) if report_path.is_file() else {}
    if direct_ply:
        model_ply = target
    else:
        stored = report.get("model_ply") or (manifest.get("outputs", {}).get("ply") or {}).get("path")
        model_ply = Path(stored) if stored else output_dir / "model" / "trained_gaussians.ply"
    if not model_ply.is_file():
        raise ValueError(f"Trained Gaussian PLY not found: {model_ply}")
    if "untrained" in model_ply.name.lower() or manifest.get("status") == "initialization_only_untrained":
        raise ValueError("Gaussian initialization is diagnostic data, not a trained model")
    return model_ply, manifest, report


def training_optics(manifest: dict[str, Any]) -> tuple[CameraOptics | None, np.ndarray | None]:
    """Reuse the training intrinsics and first camera so the first view matches capture."""
    transforms_value = (manifest.get("training_input") or {}).get("transforms")
    if not transforms_value:
        return None, None
    transforms_path = Path(transforms_value)
    if not transforms_path.is_file():
        return None, None
    try:
        payload = _load_json(transforms_path)
    except (OSError, json.JSONDecodeError):
        return None, None
    try:
        optics = CameraOptics(
            fl_x=float(payload["fl_x"]), fl_y=float(payload["fl_y"]),
            cx=float(payload["cx"]), cy=float(payload["cy"]),
            width=int(payload["w"]), height=int(payload["h"]),
            source=str(transforms_path),
        )
    except (KeyError, TypeError, ValueError):
        return None, None
    frames = payload.get("frames") or []
    first_c2w_rdf: np.ndarray | None = None
    if frames:
        try:
            from scene_gsplat_data import opengl_c2w_to_rdf

            first_c2w_rdf = opengl_c2w_to_rdf(np.asarray(frames[0]["transform_matrix"], dtype=np.float64))
        except (KeyError, TypeError, ValueError):
            first_c2w_rdf = None
    return optics, first_c2w_rdf


def initial_camera(renderer: TrainedGaussianRenderer, first_c2w_rdf: np.ndarray | None) -> OrbitCamera:
    centre, radius = renderer.bounds()
    distance = max(radius * 2.2, 0.35)
    if first_c2w_rdf is not None:
        eye = first_c2w_rdf[:3, 3]
        up_world = -first_c2w_rdf[:3, 1]  # camera +Y is down, so world "up" is its negation
        offset = centre - eye
        norm = float(np.linalg.norm(offset))
        if norm > 1e-6:
            distance = float(np.clip(norm, 0.15, 50.0))
    else:
        up_world = np.array([0.0, -1.0, 0.0])
    camera = OrbitCamera(target=np.asarray(centre, dtype=np.float64), distance=distance, up_world=up_world)
    if first_c2w_rdf is not None:
        first, second = camera._basis
        direction = first_c2w_rdf[:3, 3] - camera.target
        norm = float(np.linalg.norm(direction))
        if norm > 1e-6:
            direction = direction / norm
            camera.pitch_deg = float(
                np.clip(math.degrees(math.asin(float(np.dot(direction, camera.up_world)))), MIN_PITCH_DEG, MAX_PITCH_DEG)
            )
            camera.yaw_deg = math.degrees(math.atan2(float(np.dot(direction, second)), float(np.dot(direction, first)))) % 360.0
    return camera


def default_optics(renderer: TrainedGaussianRenderer, width: int, height: int) -> CameraOptics:
    focal = 0.5 * height / math.tan(math.radians(55.0) / 2.0)
    return CameraOptics(
        fl_x=focal, fl_y=focal, cx=width / 2.0, cy=height / 2.0,
        width=width, height=height, source="synthesized 55 degree vertical field of view",
    )


def render_orbit(
    target: Path,
    output: Path,
    *,
    frames: int = 36,
    width: int = DEFAULT_RENDER_WIDTH,
    frame_timeout_s: float = DEFAULT_FRAME_TIMEOUT_S,
    data_kind: str | None = None,
) -> dict[str, Any]:
    """Headless turntable render; used when no desktop session is available."""
    from PIL import Image

    model_ply, manifest, report = resolve_model(target)
    optics, first_c2w = training_optics(manifest)
    requested_width = max(1, int(width))
    requested_height = max(1, int(round(requested_width * (optics.height / optics.width)))) if optics else int(requested_width * 9 / 16)
    renderer = TrainedGaussianRenderer(model_ply, initial_width=requested_width, initial_height=requested_height)
    camera = initial_camera(renderer, first_c2w)
    optics = optics or default_optics(renderer, requested_width, requested_height)
    intrinsic = optics.scaled(requested_width, requested_height)
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    written: list[str] = []
    frame_times_ms: list[float] = []
    frame_status: list[dict[str, Any]] = []
    timeout_s = max(0.0, float(frame_timeout_s))
    try:
        renderer._torch.cuda.reset_peak_memory_stats(renderer.device)
    except (RuntimeError, AttributeError):
        pass
    for index in range(int(frames)):
        camera.yaw_deg = (360.0 * index / max(int(frames), 1)) % 360.0
        frame_started = time.time()
        image = renderer.render(camera.viewmat(), intrinsic, requested_width, requested_height)
        elapsed_frame = time.time() - frame_started
        frame_times_ms.append(elapsed_frame * 1000.0)
        timed_out = bool(timeout_s and elapsed_frame > timeout_s)
        # Snapshot the settings that actually produced this frame.  Timeout
        # degradation is a policy for the *next* frame; recording the mutated
        # budget here would mix the current image resolution with next-frame
        # Gaussian/resolution values and make the scale report unauditable.
        frame_budget = renderer.budget
        frame_gaussian_count = renderer.rendered_gaussian_count
        frame_status.append(
            {
                "index": index,
                "elapsed_ms": elapsed_frame * 1000.0,
                "timed_out": timed_out,
                "effective_resolution": list(image.shape[1::-1]),
                "rendered_gaussian_count": frame_gaussian_count,
                "degradation_state": frame_budget["state"],
            }
        )
        if timed_out:
            renderer.degrade_for_timeout(requested_width, requested_height)
        path = output / f"orbit_{index:03d}.png"
        Image.fromarray(image).save(path)
        written.append(path.name)
    peak_allocated: int | None = None
    peak_reserved: int | None = None
    try:
        peak_allocated = int(renderer._torch.cuda.max_memory_allocated(renderer.device))
        peak_reserved = int(renderer._torch.cuda.max_memory_reserved(renderer.device))
    except (RuntimeError, AttributeError):
        pass
    budget = renderer.budget
    report_payload = {
        "schema": "femto-direct-gsplat-orbit-render-v1",
        "created_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "renderer": "gsplat.rendering.rasterization (real 3D Gaussian Splatting)",
        "model_ply": str(model_ply),
        "model_sha256": hashlib.sha256(model_ply.read_bytes()).hexdigest(),
        "data_kind": data_kind or _report_data_kind(report, manifest),
        "gaussian_count": renderer.model_gaussian_count,
        "model_gaussian_count": renderer.model_gaussian_count,
        "rendered_gaussian_count": renderer.rendered_gaussian_count,
        "sh_degree": renderer.sh_degree,
        "frames": len(written),
        "resolution": list(budget["effective_resolution"]),
        "requested_resolution": [requested_width, requested_height],
        "effective_resolution": list(budget["effective_resolution"]),
        "frame_times_ms": frame_times_ms,
        "frame_status": frame_status,
        "frame_timeout_s": timeout_s,
        "peak_cuda_memory_bytes": peak_allocated,
        # ``peak_cuda_memory_mb`` is retained for report compatibility even
        # though its historical divisor is binary.  The explicit MiB key
        # removes the unit ambiguity for new consumers.
        "peak_cuda_memory_mb": (peak_allocated / (1024.0 * 1024.0)) if peak_allocated is not None else None,
        "peak_cuda_memory_mib": (peak_allocated / (1024.0 * 1024.0)) if peak_allocated is not None else None,
        "peak_cuda_reserved_bytes": peak_reserved,
        "degradation": budget,
        "intrinsics_source": optics.source,
        "world_frame": "source T_WC world; metres; Orbbec/OpenCV RDF camera axes",
        "training_accepted": report.get("accepted"),
        "elapsed_s": time.time() - started,
        "files": written,
    }
    (output / "orbit_render.json").write_text(
        json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report_payload


class GaussianViewerWindow:
    """Minimal Tkinter shell around the gsplat rasterizer."""

    def __init__(
        self,
        target: Path,
        language: str = "zh",
        *,
        frame_timeout_s: float = DEFAULT_FRAME_TIMEOUT_S,
    ) -> None:
        import tkinter as tk
        from PIL import Image, ImageTk

        self._tk = tk
        self._image_module = Image
        self._imagetk = ImageTk
        self.language = language if language in LANGUAGE_NAMES else "zh"
        self.model_ply, self.manifest, self.report = resolve_model(target)
        self.output_dir = _viewer_log_directory(Path(target))
        self.frame_timeout_s = max(0.0, float(frame_timeout_s))
        self.optics, first_c2w = training_optics(self.manifest)
        initial_height = int(round(DEFAULT_RENDER_WIDTH * (self.optics.height / self.optics.width))) if self.optics else int(DEFAULT_RENDER_WIDTH * 9 / 16)
        self.renderer = TrainedGaussianRenderer(
            self.model_ply, initial_width=DEFAULT_RENDER_WIDTH, initial_height=initial_height
        )
        self.camera = initial_camera(self.renderer, first_c2w)
        self._sensor_viewmat = np.linalg.inv(first_c2w) if first_c2w is not None else None
        self.render_width = DEFAULT_RENDER_WIDTH
        self.aspect = (self.optics.height / self.optics.width) if self.optics else 9 / 16
        self._pending = False
        self._drag: tuple[int, int] | None = None
        self._photo: Any = None
        self._last_render_s = 0.0

        self.root = tk.Tk()
        self.root.title(self._t("gsview.window_title", name=self.model_ply.parent.parent.name))
        self.root.configure(bg="#0b1118")
        self.root.geometry("1180x820")
        self.root.minsize(720, 520)
        self._build_ui()
        self.root.after(60, self._request_render)

    def _t(self, key: str, **values: Any) -> str:
        return tr(self.language, key, **values)

    def _build_ui(self) -> None:
        tk = self._tk
        header = tk.Frame(self.root, bg="#0b1118")
        header.pack(fill=tk.X, padx=14, pady=(12, 6))
        self.heading_label = tk.Label(
            header, text=self._t("gsview.heading"), bg="#0b1118", fg="#e8f0f8",
            font=("Microsoft YaHei UI", 15, "bold"),
        )
        self.heading_label.pack(side=tk.LEFT)
        self.language_var = tk.StringVar(value=LANGUAGE_NAMES[self.language])
        selector = tk.OptionMenu(header, self.language_var, *LANGUAGE_NAMES.values(), command=self._change_language)
        selector.configure(bg="#1a2633", fg="#dbe7f2", highlightthickness=0, relief=tk.FLAT, activebackground="#26364a")
        selector.pack(side=tk.RIGHT)

        self.backend_label = tk.Label(
            self.root, text=self._backend_text(), bg="#0b1118", fg="#7fd7a4",
            font=("Microsoft YaHei UI", 9), justify=tk.LEFT, anchor="w",
        )
        self.backend_label.pack(fill=tk.X, padx=14)

        self.canvas = tk.Label(self.root, bg="#05080c", text=self._t("gsview.rendering"), fg="#9cb1c7")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=14, pady=10)
        self.canvas.bind("<ButtonPress-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._orbit_drag)
        self.canvas.bind("<ButtonRelease-1>", self._release)
        self.canvas.bind("<ButtonPress-3>", self._press)
        self.canvas.bind("<B3-Motion>", self._pan_drag)
        self.canvas.bind("<ButtonRelease-3>", self._release)
        self.canvas.bind("<MouseWheel>", self._wheel)
        self.canvas.bind("<Configure>", lambda _event: self._request_render())

        controls = tk.Frame(self.root, bg="#111a24", padx=12, pady=8)
        controls.pack(fill=tk.X, padx=14, pady=(0, 12))
        self.reset_button = tk.Button(
            controls, text=self._t("gsview.reset_view"), command=self._reset_view,
            bg="#6d48a8", activebackground="#8058c1", fg="white", relief=tk.FLAT, bd=0,
            font=("Microsoft YaHei UI", 10, "bold"), cursor="hand2",
        )
        self.reset_button.pack(side=tk.LEFT)
        self.quality_label = tk.Label(
            controls, text=self._t("gsview.render_width"), bg="#111a24", fg="#dbe7f2",
            font=("Microsoft YaHei UI", 9),
        )
        self.quality_label.pack(side=tk.LEFT, padx=(14, 6))
        self.width_scale = tk.Scale(
            controls, from_=320, to=1600, resolution=80, orient=tk.HORIZONTAL, showvalue=True,
            bg="#111a24", fg="#dbe7f2", troughcolor="#1a2633", highlightthickness=0,
            command=self._set_render_width, length=220,
        )
        self.width_scale.set(self.render_width)
        self.width_scale.pack(side=tk.LEFT)
        self.status_var = tk.StringVar(value="")
        tk.Label(
            controls, textvariable=self.status_var, bg="#111a24", fg="#a3b6c8",
            font=("Microsoft YaHei UI", 9), anchor="e", justify=tk.RIGHT,
        ).pack(side=tk.RIGHT, fill=tk.X, expand=True)
        self.hint_label = tk.Label(
            self.root, text=self._t("gsview.controls_hint"), bg="#0b1118", fg="#7d8fa3",
            font=("Microsoft YaHei UI", 9), anchor="w", justify=tk.LEFT,
        )
        self.hint_label.pack(fill=tk.X, padx=14, pady=(0, 10))

    def _backend_text(self) -> str:
        evaluation = (self.report.get("evaluation") or {}).get("aggregate") or {}
        psnr = evaluation.get("psnr")
        kind = _report_data_kind(self.report, self.manifest)
        budget = self.renderer.budget
        return self._t(f"gsview.data.{kind}") + " · " + self._t(
            "gsview.backend",
            count=self.renderer.model_gaussian_count,
            degree=self.renderer.sh_degree,
            accepted=self._t("gsview.accepted" if self.report.get("accepted") else "gsview.not_accepted"),
            psnr=f"{psnr:.2f}" if isinstance(psnr, (int, float)) else "-",
        ) + " · " + self._t(
            "gsview.degradation",
            state=self._t(f"gsview.degradation.{budget['state']}"),
            rendered=self.renderer.rendered_gaussian_count,
        )

    def _change_language(self, display: str) -> None:
        for code, name in LANGUAGE_NAMES.items():
            if name == display:
                self.language = code
                break
        self.root.title(self._t("gsview.window_title", name=self.model_ply.parent.parent.name))
        self.heading_label.configure(text=self._t("gsview.heading"))
        self.backend_label.configure(text=self._backend_text())
        self.reset_button.configure(text=self._t("gsview.reset_view"))
        self.quality_label.configure(text=self._t("gsview.render_width"))
        self.hint_label.configure(text=self._t("gsview.controls_hint"))
        self._update_status()

    def _set_render_width(self, value: str) -> None:
        self.render_width = int(float(value))
        self._request_render()

    def _reset_view(self) -> None:
        _, first_c2w = training_optics(self.manifest)
        self.camera = initial_camera(self.renderer, first_c2w)
        self._sensor_viewmat = np.linalg.inv(first_c2w) if first_c2w is not None else None
        self._request_render()

    def _press(self, event: Any) -> None:
        self._drag = (event.x, event.y)

    def _release(self, _event: Any) -> None:
        self._drag = None

    def _orbit_drag(self, event: Any) -> None:
        self._sensor_viewmat = None
        if self._drag is None:
            self._drag = (event.x, event.y)
            return
        dx, dy = event.x - self._drag[0], event.y - self._drag[1]
        self._drag = (event.x, event.y)
        self.camera.orbit(-dx * 0.35, dy * 0.35)
        self._request_render()

    def _pan_drag(self, event: Any) -> None:
        self._sensor_viewmat = None
        if self._drag is None:
            self._drag = (event.x, event.y)
            return
        dx, dy = event.x - self._drag[0], event.y - self._drag[1]
        self._drag = (event.x, event.y)
        metres_per_pixel = self.camera.distance / max(self.render_width, 1) * 1.4
        self.camera.pan(-dx * metres_per_pixel, -dy * metres_per_pixel)
        self._request_render()

    def _wheel(self, event: Any) -> None:
        self._sensor_viewmat = None
        self.camera.zoom(0.9 if event.delta > 0 else 1.0 / 0.9)
        self._request_render()

    def _request_render(self) -> None:
        if self._pending:
            return
        self._pending = True
        self.root.after_idle(self._render_now)

    def _render_now(self) -> None:
        self._pending = False
        width = int(self.render_width)
        height = max(1, int(round(width * self.aspect)))
        optics = self.optics or default_optics(self.renderer, width, height)
        started = time.time()
        try:
            viewmat = self._sensor_viewmat if self._sensor_viewmat is not None else self.camera.viewmat()
            image = self.renderer.render(viewmat, optics.scaled(width, height), width, height)
        except Exception as error:  # keep the window usable and name the failure
            self.status_var.set(f"{type(error).__name__}: {error}")
            try:
                _append_viewer_log(
                    self.output_dir,
                    {
                        "status": "blocked",
                        "phase": "interactive_render",
                        "error": f"{type(error).__name__}: {error}",
                    },
                )
            except OSError:
                pass
            return
        self._last_render_s = time.time() - started
        rerender_degraded = False
        if self.frame_timeout_s and self._last_render_s > self.frame_timeout_s:
            rerender_degraded = self.renderer.degrade_for_timeout(width, height)
            try:
                _append_viewer_log(
                    self.output_dir,
                    {
                        "status": "degraded",
                        "phase": "interactive_render",
                        "reason": "frame timeout",
                        "elapsed_ms": self._last_render_s * 1000.0,
                        "frame_timeout_s": self.frame_timeout_s,
                        "degradation": self.renderer.budget,
                    },
                )
            except OSError:
                pass
        available_width = max(self.canvas.winfo_width(), 200)
        available_height = max(self.canvas.winfo_height(), 150)
        picture = self._image_module.fromarray(image)
        scale = min(available_width / width, available_height / height)
        if scale > 0 and abs(scale - 1.0) > 0.02:
            picture = picture.resize(
                (max(1, int(width * scale)), max(1, int(height * scale))),
                self._image_module.Resampling.BILINEAR,
            )
        self._photo = self._imagetk.PhotoImage(picture)
        self.canvas.configure(image=self._photo, text="")
        self._update_status()
        if rerender_degraded:
            self._request_render()

    def _update_status(self) -> None:
        budget = self.renderer.budget
        self.status_var.set(
            self._t(
                "gsview.status",
                width=int(self.render_width),
                ms=self._last_render_s * 1000.0,
                distance=self.camera.distance,
                state=self._t(f"gsview.degradation.{budget['state']}"),
            )
        )

    def run(self) -> None:
        self.root.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a trained direct-gsplat model with gsplat")
    parser.add_argument("model", type=Path, help="direct_gsplat_* result directory or trained_gaussians.ply")
    parser.add_argument("--lang", choices=tuple(LANGUAGE_NAMES), default="zh")
    parser.add_argument("--orbit", type=int, help="Headless mode: render this many turntable frames")
    parser.add_argument("--orbit-output", type=Path, help="Directory for headless orbit frames")
    parser.add_argument("--width", type=int, default=DEFAULT_RENDER_WIDTH)
    parser.add_argument("--frame-timeout-s", type=float, default=DEFAULT_FRAME_TIMEOUT_S)
    parser.add_argument("--data-kind", choices=("synthetic", "real_femto", "unknown"), default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.orbit:
            output = args.orbit_output or (Path(args.model).resolve() / "orbit_render")
            payload = render_orbit(
                args.model,
                output,
                frames=args.orbit,
                width=args.width,
                frame_timeout_s=args.frame_timeout_s,
                data_kind=args.data_kind,
            )
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0
        GaussianViewerWindow(
            args.model,
            language=args.lang,
            frame_timeout_s=args.frame_timeout_s,
        ).run()
    except Exception as error:
        try:
            log_dir = _viewer_log_directory(args.model)
            log_dir.mkdir(parents=True, exist_ok=True)
            _append_viewer_log(
                log_dir,
                {"status": "blocked", "phase": "startup", "error": f"{type(error).__name__}: {error}"},
            )
        except OSError:
            pass
        print(
            json.dumps(
                {"status": "blocked", "error": f"{type(error).__name__}: {error}"},
                ensure_ascii=False, indent=2,
            )
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
