"""Commercial-safe Femto RGB-D trainer using the Apache-2.0 gsplat API.

This is an independently written product adapter.  It consumes the project's
rectified full-scene RGB-D dataset, optimizes Gaussian parameters with a real
metric depth loss, delegates split/duplicate/prune decisions to gsplat's public
DefaultStrategy, supports resumable checkpoints, evaluates held-out views, and
exports a trained Gaussian PLY plus optional official Niantic SPZ.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import inspect
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np
from femto.gsplat.scene_gsplat_guard import ResourceLimitError, bounded_strategy_class, guard_cuda, cuda_memory_snapshot
from PIL import Image

from femto.gsplat.scene_gsplat_data import (
    GsplatDataset,
    GsplatFrame,
    load_frame_confidence,
    load_frame_images,
    load_gsplat_dataset,
    read_xyzrgb_ply,
)
from femto.gsplat.scene_gsplat_export import export_official_spz, write_model_lineage, write_trained_gaussian_ply
from femto.gsplat.scene_gsplat_metrics import (
    LPIPS_BACKBONE_FILE,
    LpipsMetric,
    depth_metrics_numpy,
    install_lpips_backbone,
    lpips_backbone_status,
    metric_depth_loss_torch,
    psnr_numpy,
    ssim_torch,
)


from femto.paths import REPO_ROOT as ROOT
OUTPUTS_ROOT = ROOT / "outputs"
RUNTIME_ROOT = ROOT / ".venv-gsplat"
ASCII_RUNTIME_ROOT = Path(r"C:\FemtoGsplatRuntime") if os.name == "nt" else RUNTIME_ROOT
LATEST_MARKER = OUTPUTS_ROOT / "latest_direct_gsplat.txt"
SH_C0 = 0.28209479177387814


@dataclass(frozen=True)
class TrainerConfig:
    profile: str = "low_vram"
    iterations: int = 6_000
    downscale: int = 4
    max_init_points: int = 25_000
    max_gaussians: int = 400_000
    max_cuda_reserved_mib: int = 5120
    min_cuda_free_mib: int = 1536
    step_pause_ms: int = 10
    sh_degree: int = 3
    eval_stride: int = 8
    seed: int = 20260903
    rgb_l1_weight: float = 0.80
    rgb_ssim_weight: float = 0.20
    depth_weight: float = 0.10
    depth_beta_m: float = 0.02
    scale_regularization_weight: float = 0.002
    opacity_regularization_weight: float = 0.0005
    packed: bool = True
    antialiased: bool = True
    means_lr_final_ratio: float = 0.01
    publish_latest: bool = True
    refine_start: int = 500
    refine_stop: int = 3_000
    refine_every: int = 100
    reset_every: int = 3_000
    prune_opacity: float = 0.005
    grow_grad2d: float = 0.0002
    grow_scale3d: float = 0.01
    prune_scale3d: float = 0.10
    checkpoint_every: int = 250
    progress_every: int = 25
    acceptance_psnr: float = 20.0
    acceptance_ssim: float = 0.65
    acceptance_lpips: float = 0.40
    acceptance_depth_rmse_m: float = 0.12
    acceptance_depth_coverage: float = 0.90
    export_spz: bool = True
    spz_source_coordinate: str = "RDF"


PROFILES: dict[str, TrainerConfig] = {
    "low_vram": TrainerConfig(),
    "balanced": TrainerConfig(
        profile="balanced", iterations=30_000, downscale=1, max_init_points=250_000,
        max_gaussians=800_000, refine_stop=20_000, checkpoint_every=500,
    ),
    "quality": TrainerConfig(
        profile="quality", iterations=45_000, downscale=1, max_init_points=500_000,
        max_gaussians=1_000_000, refine_stop=30_000, checkpoint_every=500,
    ),
}


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def runtime_python() -> Path:
    """Interpreter of the product 3DGS runtime, preferring the ASCII junction.

    CUDA JIT compilation fails under this repository's non-ASCII path, so
    ``setup_direct_gsplat_environment.cmd`` creates ``C:\\FemtoGsplatRuntime`` as
    a directory junction to ``.venv-gsplat``; every launcher goes through it.
    """
    relative = "Scripts/python.exe" if os.name == "nt" else "bin/python"
    ascii_executable = ASCII_RUNTIME_ROOT / relative
    return ascii_executable if ascii_executable.is_file() else RUNTIME_ROOT / relative


def runtime_status() -> dict[str, Any]:
    executable = runtime_python()
    result: dict[str, Any] = {
        "ready": False,
        "runtime_root": str(RUNTIME_ROOT),
        "python": str(executable) if executable.is_file() else None,
        "backend": "direct nerfstudio-project/gsplat public API",
        "backend_license": "Apache-2.0",
        "graphdeco_dependency": False,
        "nerfstudio_dependency": False,
        "pymeshlab_dependency": False,
        "lpips_backbone": lpips_backbone_status(),
    }
    result["lpips_ready"] = bool(result["lpips_backbone"]["ready"])
    if not executable.is_file():
        result["error"] = "Run setup_direct_gsplat_environment.cmd"
        return result
    command = [
        str(executable), "-c",
        "import json,torch,gsplat; print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'cuda_ready':torch.cuda.is_available(),'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'gsplat':gsplat.__version__}))",
    ]
    try:
        completed = subprocess.run(
            command, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", timeout=45,
        )
        if completed.returncode == 0:
            details = json.loads(completed.stdout.strip().splitlines()[-1])
            result.update(details)
            result["ready"] = bool(details.get("cuda_ready"))
        else:
            result["error"] = completed.stderr.strip() or completed.stdout.strip()
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError, IndexError) as error:
        result["error"] = str(error)
    return result


def _resolve_input_path(value: Path | None) -> Path:
    if value is not None:
        candidate = value.resolve()
        if (candidate / "transforms.json").is_file():
            return candidate
        summary = candidate / "demo_summary.json"
        if summary.is_file():
            payload = json.loads(summary.read_text(encoding="utf-8-sig"))
            return Path(payload.get("nerfstudio", candidate / "nerfstudio")).resolve()
        raise ValueError(f"Cannot find transforms.json under {candidate}")
    marker = OUTPUTS_ROOT / "latest_complete_demo.txt"
    if not marker.is_file():
        raise ValueError("No completed reconstruction is available")
    return _resolve_input_path(Path(marker.read_text(encoding="utf-8-sig").strip()).parent)


def assess_training_input(input_path: Path | None = None) -> dict[str, Any]:
    from femto.gsplat.scene_gsplat_acceptance import input_provenance

    root: Path | None = None
    try:
        root = _resolve_input_path(input_path)
        dataset = load_gsplat_dataset(root, downscale=1, minimum_frames=12)
        seed_points, _ = read_xyzrgb_ply(dataset.seed_path)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        resolved = root or input_path
        return {"ready": False, "input": str(resolved) if resolved else None, "reasons": [str(error)],
                "provenance": input_provenance(root) if root else {}}
    reasons: list[str] = []
    if not dataset.test_indices:
        reasons.append('Reconvert RGB-D input: strict split and training-only Seed PLY required')
    if len(dataset.frames) < 12:
        reasons.append("direct 3DGS needs at least 12 accepted moving-camera views")
    manifest_path = root / "conversion_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig")) if manifest_path.is_file() else {}
    # Use the actual matrices, not a potentially stale span copied into a manifest.
    from femto.gsplat.scene_gsplat_acceptance import input_provenance

    provenance = input_provenance(root)
    cameras = np.stack([frame.camera_to_world_rdf for frame in dataset.frames])
    centers = cameras[:, :3, 3]
    translation_span = float(np.linalg.norm(centers[:, None] - centers[None, :], axis=-1).max())
    relative_rotations = cameras[0, :3, :3].T @ cameras[:, :3, :3]
    rotation_span = float(np.max(np.degrees(np.arccos(np.clip(
        (np.trace(relative_rotations, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0
    )))))
    if provenance["fixed_camera"]:
        reasons.append("fixed-camera datasets cannot train multiview 3DGS")
    if provenance["geometry_and_gaussian_share_pose_dataset"] is False:
        reasons.append("geometry reconstruction and Gaussian input use different pose datasets")
    if translation_span < 0.05 and rotation_span < 5.0:
        reasons.append("camera viewpoint diversity is too small")
    return {
        "ready": not reasons,
        "input": str(root),
        "frame_count": len(dataset.frames),
        "train_frames": len(dataset.train_indices),
        "eval_frames": len(dataset.eval_indices),
        "translation_span_m": translation_span,
        "rotation_span_deg": rotation_span,
        "dataset_fingerprint_sha256": dataset.fingerprint,
        "seed_ply": str(dataset.seed_path),
        "seed_point_count": len(seed_points),
        "provenance": provenance,
        "reasons": reasons,
    }


def _scene_scale(points: np.ndarray) -> tuple[np.ndarray, float]:
    center = np.median(points, axis=0).astype(np.float32)
    radii = np.linalg.norm(points - center, axis=1)
    scale = max(float(np.percentile(radii, 90.0)), 0.1)
    return center, scale


def _seed_scale_m(dataset: GsplatDataset, points: np.ndarray) -> float:
    manifest_path = dataset.root / "conversion_manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
            voxel = float((manifest.get("initialization") or {}).get("voxel_m", 0.0))
            if 0.0005 <= voxel <= 0.10:
                return float(np.clip(voxel * 0.75, 0.002, 0.05))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    span = np.ptp(points, axis=0)
    volume = max(float(np.prod(np.maximum(span, 1e-3))), 1e-9)
    return float(np.clip((volume / max(len(points), 1)) ** (1.0 / 3.0) * 0.65, 0.002, 0.05))


def _initial_arrays(dataset: GsplatDataset, config: TrainerConfig) -> tuple[dict[str, np.ndarray], float]:
    points, colors = read_xyzrgb_ply(dataset.seed_path)
    if len(points) > config.max_init_points:
        rng = np.random.default_rng(config.seed)
        indices = np.sort(rng.choice(len(points), size=config.max_init_points, replace=False))
        points = points[indices]
        colors = colors[indices]
    _, scene_scale = _scene_scale(points)
    scale_m = _seed_scale_m(dataset, points)
    coefficient_count = (config.sh_degree + 1) ** 2
    arrays = {
        "means": points.astype(np.float32),
        "scales": np.full((len(points), 3), math.log(scale_m), dtype=np.float32),
        "quats": np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (len(points), 1)),
        "opacities": np.full((len(points),), math.log(0.1 / 0.9), dtype=np.float32),
        "sh0": ((colors[:, None, :] - 0.5) / SH_C0).astype(np.float32),
        "shN": np.zeros((len(points), coefficient_count - 1, 3), dtype=np.float32),
    }
    return arrays, scene_scale


def _make_parameters(arrays: dict[str, np.ndarray], device: Any) -> dict[str, Any]:
    import torch

    return {
        name: torch.nn.Parameter(torch.as_tensor(array, device=device, dtype=torch.float32))
        for name, array in arrays.items()
    }


def _make_optimizers(params: dict[str, Any], scene_scale: float) -> dict[str, Any]:
    import torch

    rates = {
        "means": 1.6e-4 * scene_scale,
        "scales": 5.0e-3,
        "quats": 1.0e-3,
        "opacities": 5.0e-2,
        "sh0": 2.5e-3,
        "shN": 1.25e-4,
    }
    return {
        name: torch.optim.Adam([{"params": [params[name]], "lr": rate, "name": name}], eps=1e-15)
        for name, rate in rates.items()
    }


def means_learning_rate(scene_scale: float, step: int, iterations: int, final_ratio: float) -> float:
    if iterations < 1 or not 0.0 < final_ratio <= 1.0:
        raise ValueError('Invalid means learning-rate schedule')
    return 1.6e-4 * scene_scale * final_ratio ** (min(max(step, 0), iterations) / iterations)


def _to_cpu(value: Any) -> Any:
    try:
        import torch

        if torch.is_tensor(value):
            return value.detach().cpu()
    except ImportError:
        pass
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return value


def _to_device(value: Any, device: Any) -> Any:
    import torch

    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    return value


def _atomic_checkpoint(
    path: Path,
    *,
    step: int,
    params: dict[str, Any],
    optimizers: dict[str, Any],
    strategy_state: dict[str, Any],
    config: TrainerConfig,
    dataset: GsplatDataset,
    scene_scale: float,
) -> None:
    import torch

    payload = {
        "schema": "femto-direct-gsplat-checkpoint-v1",
        "step": int(step),
        "config": asdict(config),
        "dataset_root": str(dataset.root),
        "dataset_fingerprint_sha256": dataset.fingerprint,
        "scene_scale": float(scene_scale),
        "params": {name: value.detach().cpu() for name, value in params.items()},
        "optimizers": {name: _to_cpu(optimizer.state_dict()) for name, optimizer in optimizers.items()},
        "strategy_state": _to_cpu(strategy_state),
        "random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.random.get_rng_state(),
        "torch_cuda_random_state": torch.cuda.get_rng_state_all(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as checkpoint_file:
        torch.save(payload, checkpoint_file)
        checkpoint_file.flush()
        os.fsync(checkpoint_file.fileno())
    temporary.replace(path)


def _load_checkpoint(path: Path, dataset: GsplatDataset, device: Any) -> dict[str, Any]:
    import torch

    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("schema") != "femto-direct-gsplat-checkpoint-v1":
        raise ValueError("Unsupported checkpoint schema")
    if payload.get("dataset_fingerprint_sha256") != dataset.fingerprint:
        raise ValueError("Checkpoint dataset fingerprint does not match current RGB-D evidence")
    return payload


def _create_strategy(config: TrainerConfig) -> Any:
    from gsplat.strategy import DefaultStrategy

    strategy = bounded_strategy_class(DefaultStrategy)(
        prune_opa=config.prune_opacity,
        grow_grad2d=config.grow_grad2d,
        grow_scale3d=config.grow_scale3d,
        prune_scale3d=config.prune_scale3d,
        refine_start_iter=config.refine_start,
        refine_stop_iter=min(config.refine_stop, config.iterations),
        refine_every=config.refine_every,
        reset_every=config.reset_every,
        absgrad=True,
        verbose=True,
    )
    strategy.max_gaussians = config.max_gaussians
    strategy.guard_config = config
    return strategy


def _rasterize(params: dict[str, Any], frame: GsplatFrame, device: Any, config: TrainerConfig) -> tuple[Any, Any, dict[str, Any]]:
    import torch
    from gsplat.rendering import rasterization

    guard_cuda(device, config)
    if len(params['means']) > config.max_gaussians:
        raise ResourceLimitError('Gaussian count exceeds configured cap')
    colors = torch.cat((params["sh0"], params["shN"]), dim=1)
    viewmat = torch.as_tensor(frame.viewmat_rdf, dtype=torch.float32, device=device).unsqueeze(0)
    intrinsic = torch.as_tensor(frame.K, dtype=torch.float32, device=device).unsqueeze(0)
    renders, alphas, info = rasterization(
        means=params["means"],
        quats=params["quats"],
        scales=torch.exp(params["scales"]),
        opacities=torch.sigmoid(params["opacities"]),
        colors=colors,
        viewmats=viewmat,
        Ks=intrinsic,
        width=frame.width,
        height=frame.height,
        packed=config.packed,
        absgrad=True,
        sh_degree=config.sh_degree,
        render_mode="RGB+ED",
        rasterize_mode="antialiased" if config.antialiased else "classic",
    )
    return renders[0], alphas[0], info


def _frame_tensors(frame: GsplatFrame, dataset: GsplatDataset, device: Any) -> tuple[Any, Any, Any]:
    import torch

    rgb, depth = load_frame_images(frame, dataset.depth_unit_m)
    confidence = load_frame_confidence(frame)
    return (
        torch.as_tensor(rgb, dtype=torch.float32, device=device),
        torch.as_tensor(depth, dtype=torch.float32, device=device),
        torch.as_tensor(confidence, dtype=torch.float32, device=device),
    )


def _strategy_post(strategy: Any, params: dict[str, Any], optimizers: dict[str, Any], state: dict[str, Any], step: int, info: dict[str, Any], config: TrainerConfig) -> None:
    signature = inspect.signature(strategy.step_post_backward)
    kwargs: dict[str, Any] = {}
    if "packed" in signature.parameters:
        kwargs["packed"] = config.packed
    strategy.step_post_backward(params, optimizers, state, step, info, **kwargs)


def _train_step(
    *,
    params: dict[str, Any],
    optimizers: dict[str, Any],
    strategy: Any,
    strategy_state: dict[str, Any],
    dataset: GsplatDataset,
    frame_index: int,
    step: int,
    device: Any,
    config: TrainerConfig,
) -> dict[str, float]:
    import torch

    frame = dataset.frames[frame_index]
    target_rgb, target_depth, target_confidence = _frame_tensors(frame, dataset, device)
    render, alpha, info = _rasterize(params, frame, device, config)
    prediction_rgb = render[..., :3].clamp(0.0, 1.0)
    prediction_depth = render[..., 3]
    l1 = torch.mean(torch.abs(prediction_rgb - target_rgb))
    ssim_value = ssim_torch(prediction_rgb, target_rgb)
    depth_loss, depth_valid = metric_depth_loss_torch(
        prediction_depth, target_depth, config.depth_beta_m, target_confidence
    )
    scales = torch.exp(params["scales"])
    scale_regularizer = torch.relu(scales.max(dim=1).values - 0.05).mean()
    opacity_regularizer = torch.sigmoid(params["opacities"]).mean()
    total = (
        config.rgb_l1_weight * l1
        + config.rgb_ssim_weight * (1.0 - ssim_value)
        + config.depth_weight * depth_loss
        + config.scale_regularization_weight * scale_regularizer
        + config.opacity_regularization_weight * opacity_regularizer
    )
    strategy.step_pre_backward(params, optimizers, strategy_state, step, info)
    total.backward()
    for optimizer in optimizers.values():
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    _strategy_post(strategy, params, optimizers, strategy_state, step, info, config)
    return {
        "loss": float(total.detach().cpu()),
        "rgb_l1": float(l1.detach().cpu()),
        "ssim": float(ssim_value.detach().cpu()),
        "depth_loss_m": float(depth_loss.detach().cpu()),
        "depth_valid_pixels": int(depth_valid.sum().detach().cpu()),
        "depth_effective_weight": float(target_confidence[depth_valid].sum().detach().cpu()),
        "mean_alpha": float(alpha.mean().detach().cpu()),
        "gaussians": len(params["means"]),
    }


def _save_rgb(path: Path, image: np.ndarray) -> None:
    pixels = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(path)


def _save_depth(path: Path, depth_m: np.ndarray) -> None:
    depth_mm = np.where(np.isfinite(depth_m) & (depth_m > 0.0), np.rint(depth_m * 1000.0), 0.0)
    depth_mm = np.clip(depth_mm, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(depth_mm).save(path)


def evaluate_model(
    params: dict[str, Any],
    dataset: GsplatDataset,
    device: Any,
    config: TrainerConfig,
    output_dir: Path,
    *,
    enable_lpips: bool = True,
    partition: str = 'validation',
) -> dict[str, Any]:
    import torch

    lpips_metric = None
    lpips_error = None
    lpips_backbone: dict[str, Any] | None = None
    if enable_lpips:
        try:
            lpips_metric = LpipsMetric(device)
            lpips_backbone = lpips_metric.backbone
        except Exception as error:  # installation/cache errors must be visible in the report
            if isinstance(error, torch.cuda.OutOfMemoryError):
                raise
            lpips_error = f"{type(error).__name__}: {error}"
    else:
        lpips_error = "LPIPS skipped by --skip-lpips (smoke test only)"
    frame_reports: list[dict[str, Any]] = []
    if partition not in ('validation', 'test'):
        raise ValueError('Unknown evaluation partition')
    selected_indices = dataset.test_indices if partition == 'test' else dataset.eval_indices
    if not selected_indices:
        raise ValueError('Requested evaluation partition is empty')
    for frame_index in selected_indices:
        frame = dataset.frames[frame_index]
        target_rgb_np, target_depth_np = load_frame_images(
            frame, dataset.depth_unit_m, observed_only=True
        )
        target_rgb = torch.as_tensor(target_rgb_np, dtype=torch.float32, device=device)
        with torch.no_grad():
            render, _, _ = _rasterize(params, frame, device, config)
            pred_rgb = render[..., :3].clamp(0.0, 1.0)
            pred_depth = render[..., 3]
            ssim_value = float(ssim_torch(pred_rgb, target_rgb).detach().cpu())
            lpips_value = lpips_metric(pred_rgb, target_rgb) if lpips_metric is not None else None
        pred_rgb_np = pred_rgb.detach().cpu().numpy()
        pred_depth_np = pred_depth.detach().cpu().numpy()
        _save_rgb(output_dir / "renders" / f"frame_{frame.index:06d}_rgb.png", pred_rgb_np)
        _save_depth(output_dir / "renders" / f"frame_{frame.index:06d}_depth_mm.png", pred_depth_np)
        frame_report = {
            "frame_index": frame.index,
            "source_frame": frame.source_frame,
            "psnr": psnr_numpy(pred_rgb_np, target_rgb_np),
            "ssim": ssim_value,
            "lpips": lpips_value,
            "depth": depth_metrics_numpy(pred_depth_np, target_depth_np),
        }
        frame_reports.append(frame_report)

    def mean_value(path: tuple[str, ...]) -> float | None:
        values: list[float] = []
        for item in frame_reports:
            value: Any = item
            for key in path:
                value = value.get(key) if isinstance(value, dict) else None
            if value is not None and np.isfinite(value):
                values.append(float(value))
        return float(np.mean(values)) if values else None

    aggregate = {
        "psnr": mean_value(("psnr",)),
        "ssim": mean_value(("ssim",)),
        "lpips": mean_value(("lpips",)),
        "depth_rmse_m": mean_value(("depth", "rmse_m")),
        "depth_mae_m": mean_value(("depth", "mae_m")),
        "depth_abs_rel": mean_value(("depth", "abs_rel")),
        "depth_coverage_ratio": mean_value(("depth", "coverage_ratio")),
    }
    checks = {
        "psnr": aggregate["psnr"] is not None and aggregate["psnr"] >= config.acceptance_psnr,
        "ssim": aggregate["ssim"] is not None and aggregate["ssim"] >= config.acceptance_ssim,
        "lpips": aggregate["lpips"] is not None and aggregate["lpips"] <= config.acceptance_lpips,
        "depth_rmse": aggregate["depth_rmse_m"] is not None and aggregate["depth_rmse_m"] <= config.acceptance_depth_rmse_m,
        "depth_coverage": aggregate["depth_coverage_ratio"] is not None and aggregate["depth_coverage_ratio"] >= config.acceptance_depth_coverage,
    }
    report = {
        "schema": "femto-direct-gsplat-evaluation-v1",
        "held_out_frames": len(frame_reports),
        "partition": partition,
        "dataset_split": dataset.metadata.get('dataset_split'),
        "dataset_fingerprint_sha256": dataset.fingerprint,
        "evaluation_downscale": config.downscale,
        "aggregate": aggregate,
        "thresholds": {
            "psnr_min": config.acceptance_psnr,
            "ssim_min": config.acceptance_ssim,
            "lpips_max": config.acceptance_lpips,
            "depth_rmse_m_max": config.acceptance_depth_rmse_m,
            "depth_coverage_min": config.acceptance_depth_coverage,
        },
        "checks": checks,
        "accepted": all(checks.values()),
        "depth_definition": (
            "all valid target pixels; unrendered holes scored as 0 m misses; "
            "coverage_ratio = rendered valid pixels / valid pixels"
        ),
        "lpips_error": lpips_error,
        "lpips_backbone": lpips_backbone,
        "frames": frame_reports,
    }
    _json_dump(output_dir / "evaluation.json", report)
    return report


def _export_parameters(params: dict[str, Any], output_dir: Path) -> tuple[Path, dict[str, Any]]:
    arrays = {name: value.detach().cpu().numpy() for name, value in params.items()}
    path = output_dir / "model" / "trained_gaussians.ply"
    report = write_trained_gaussian_ply(
        path,
        means=arrays["means"],
        sh0=arrays["sh0"],
        sh_rest=arrays["shN"],
        opacity_logits=arrays["opacities"],
        log_scales=arrays["scales"],
        quaternions_wxyz=arrays["quats"],
    )
    return path, report


def _prepare_training(input_dir: Path, output_dir: Path, config: TrainerConfig,
                      resume: Path | None, enable_lpips: bool):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the direct gsplat environment")
    assessment = assess_training_input(input_dir)
    if not assessment["ready"]:
        raise ValueError("Direct gsplat input blocked: " + "; ".join(assessment["reasons"]))
    if enable_lpips:
        backbone = lpips_backbone_status()
        if not backbone["ready"]:
            raise FileNotFoundError(backbone["error"])
    if min(config.iterations, config.max_init_points, config.checkpoint_every, config.progress_every, config.refine_every) < 1:
        raise ValueError("iteration, point, checkpoint, progress and refinement counts must be positive")
    if config.refine_start < 0 or config.refine_stop < config.refine_start:
        raise ValueError("refinement interval is invalid")
    if (config.max_gaussians < config.max_init_points or config.max_cuda_reserved_mib <= 0
            or config.min_cuda_free_mib <= 0 or config.step_pause_ms < 0 or config.grow_grad2d <= 0):
        raise ValueError('Invalid Gaussian/memory limits or pause')
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    device = torch.device("cuda:0")
    # PyTorch allocator ceiling is an additional backstop, not a driver/OS guard.
    total = torch.cuda.get_device_properties(device).total_memory
    fraction = min(config.max_cuda_reserved_mib*2**20/total, 1-config.min_cuda_free_mib*2**20/total)
    if fraction <= 0:
        raise ResourceLimitError('GPU is too small for configured memory reserve')
    torch.cuda.set_per_process_memory_fraction(fraction, device)
    guard_cuda(device, config)
    dataset = load_gsplat_dataset(
        input_dir, downscale=config.downscale, eval_stride=config.eval_stride, minimum_frames=12
    )
    if not dataset.test_indices:
        raise ValueError('Reconvert RGB-D input: training-only seed and strict train/validation/test split required')
    means_learning_rate(1.0, 0, config.iterations, config.means_lr_final_ratio)
    if not np.isfinite(config.depth_weight) or config.depth_weight < 0:
        raise ValueError('Depth weight must be finite and nonnegative')
    output_dir.mkdir(parents=True, exist_ok=True)
    if resume is None and (output_dir / "training_config.json").exists():
        raise FileExistsError("Existing training output requires explicit --resume")
    return assessment, device, dataset, output_dir / "training_metrics.jsonl", _create_strategy(config)


def _train_impl(
    input_dir: Path,
    output_dir: Path,
    config: TrainerConfig,
    *,
    resume: Path | None = None,
    enable_lpips: bool = True,
) -> tuple[Path, dict[str, Any]]:
    import torch

    assessment, device, dataset, log_path, strategy = _prepare_training(
        input_dir, output_dir, config, resume, enable_lpips
    )
    start_step = 0
    if resume is not None:
        checkpoint = _load_checkpoint(resume.resolve(), dataset, "cpu")
        if len(checkpoint['params']['means']) > config.max_gaussians:
            raise ResourceLimitError('Checkpoint exceeds Gaussian cap; preserved unchanged. Use a fresh conservative run.')
        saved_config = checkpoint.get("config") or {}
        for name in ("downscale", "sh_degree", "packed"):
            if saved_config.get(name) != getattr(config, name):
                raise ValueError(f"Resume requires the original {name} setting")
        for name in ('iterations', 'means_lr_final_ratio', 'depth_weight', 'grow_grad2d', 'antialiased'):
            if saved_config.get(name) != getattr(config, name):
                raise ValueError(f'Resume requires the original {name}; start a new experiment')
        params = {name: torch.nn.Parameter(value.to(device)) for name, value in checkpoint["params"].items()}
        scene_scale = float(checkpoint["scene_scale"])
        optimizers = _make_optimizers(params, scene_scale)
        for name, state in checkpoint["optimizers"].items():
            optimizers[name].load_state_dict(state)
        strategy_state = _to_device(checkpoint["strategy_state"], device)
        start_step = int(checkpoint["step"])
        random.setstate(checkpoint["random_state"])
        np.random.set_state(checkpoint["numpy_random_state"])
        torch.random.set_rng_state(checkpoint["torch_random_state"].cpu())
        torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["torch_cuda_random_state"]])
        del checkpoint
    else:
        arrays, scene_scale = _initial_arrays(dataset, config)
        params = _make_parameters(arrays, device)
        optimizers = _make_optimizers(params, scene_scale)
        strategy_state = strategy.initialize_state(scene_scale=scene_scale)
    strategy.check_sanity(params, optimizers)
    _json_dump(output_dir / "training_config.json", asdict(config))
    initial_gaussians = len(params["means"])
    torch.cuda.reset_peak_memory_stats(device)
    final_checkpoint = output_dir / "checkpoints" / "latest.pt"
    started = time.time()
    with log_path.open("a", encoding="utf-8") as log:
        for step in range(start_step, config.iterations):
            guard_cuda(device, config)
            learning_rate = means_learning_rate(scene_scale, step, config.iterations, config.means_lr_final_ratio)
            for group in optimizers['means'].param_groups:
                group['lr'] = learning_rate
            frame_index = dataset.train_indices[step % len(dataset.train_indices)]
            metrics = _train_step(
                params=params, optimizers=optimizers, strategy=strategy, strategy_state=strategy_state,
                dataset=dataset, frame_index=frame_index, step=step, device=device, config=config,
            )
            memory = guard_cuda(device, config)
            metrics.update(memory)
            metrics['means_lr'] = learning_rate
            metrics['max_gaussians'] = config.max_gaussians
            if step == start_step or (step + 1) % config.progress_every == 0 or step + 1 == config.iterations:
                record = {"step": step + 1, "frame_index": frame_index, "elapsed_s": time.time() - started, **metrics}
                log.write(json.dumps(record, ensure_ascii=False) + "\n")
                log.flush()
                os.fsync(log.fileno())
                print(json.dumps(record, ensure_ascii=False), flush=True)
            if (step + 1) % config.checkpoint_every == 0 or step + 1 == config.iterations:
                _atomic_checkpoint(
                    final_checkpoint, step=step + 1, params=params, optimizers=optimizers,
                    strategy_state=strategy_state, config=config, dataset=dataset, scene_scale=scene_scale,
                )
            if config.step_pause_ms:
                time.sleep(config.step_pause_ms / 1000.)

    evaluation = evaluate_model(params, dataset, device, config, output_dir, enable_lpips=enable_lpips)
    ply_path, ply_report = _export_parameters(params, output_dir)
    spz_report: dict[str, Any] | None = None
    spz_error = None
    if config.export_spz:
        try:
            spz_report = export_official_spz(
                ply_path, output_dir / "model" / "trained_gaussians.spz",
                source_coordinate=config.spz_source_coordinate,
            )
        except Exception as error:
            spz_error = f"{type(error).__name__}: {error}"
            spz_report = {"status": "not_exported", "error": spz_error}
    lineage = write_model_lineage(
        output_dir / "model" / "model_manifest.json",
        dataset_root=dataset.root, dataset_fingerprint=dataset.fingerprint,
        transforms_path=dataset.transforms_path, checkpoint_path=final_checkpoint,
        ply_report=ply_report, spz_report=spz_report, training_config=asdict(config), metrics=evaluation,
    )
    report = {
        "schema": "femto-direct-gsplat-training-v1",
        "status": "trained",
        "data_kind": assessment["provenance"].get("data_kind", "unknown"),
        "accepted": evaluation["accepted"],
        "iterations": config.iterations,
        "resumed_from_step": start_step,
        "elapsed_s": time.time() - started,
        "initial_gaussians": int(initial_gaussians),
        "final_gaussians": len(params["means"]),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "dataset_root": str(dataset.root),
        "dataset_fingerprint_sha256": dataset.fingerprint,
        "train_frames": len(dataset.train_indices),
        "eval_frames": len(dataset.eval_indices),
        "checkpoint": str(final_checkpoint.resolve()),
        "model_ply": str(ply_path.resolve()),
        "model_spz": spz_report.get("path") if spz_report else None,
        "spz_error": spz_error,
        "evaluation": evaluation,
        "lineage": lineage,
        "input_provenance": assessment["provenance"],
        "runtime": {
            "gpu": torch.cuda.get_device_name(device),
            "cuda": torch.version.cuda,
            "torch": torch.__version__,
            "gsplat": __import__("gsplat").__version__,
        },
        "training_command": [sys.executable, *sys.argv],
    }
    evaluation["input_provenance"] = assessment["provenance"]
    evaluation["pose_note"] = "Research/demo sequential RGB-D odometry has drift and no loop closure, relocalization or global pose optimization."
    _json_dump(output_dir / "evaluation.json", evaluation)
    lineage["training_input"]["provenance"] = assessment["provenance"]
    _json_dump(output_dir / "model" / "model_manifest.json", lineage)
    _json_dump(output_dir / "training_report.json", report)
    from femto.gsplat.scene_gsplat_acceptance import build_acceptance_report

    _json_dump(output_dir / "acceptance_report.json", build_acceptance_report(output_dir))
    if config.publish_latest:
        LATEST_MARKER.write_text(str(output_dir.resolve()), encoding="utf-8")
    return ply_path, report


def train(input_dir: Path, output_dir: Path, config: TrainerConfig, *, resume: Path | None = None,
          enable_lpips: bool = True) -> tuple[Path, dict[str, Any]]:
    """Never publish a successful model or overwrite a checkpoint on resource stop."""
    import torch
    try:
        return _train_impl(input_dir, output_dir, config, resume=resume, enable_lpips=enable_lpips)
    except (ResourceLimitError, torch.cuda.OutOfMemoryError) as error:
        snapshot = {}
        try:
            snapshot = cuda_memory_snapshot(torch.device('cuda:0'))
        except Exception:
            pass
        stamp = time.strftime('%Y%m%d_%H%M%S') + '_' + str(time.time_ns())
        checkpoint = output_dir / 'checkpoints/latest.pt'
        report = dict(status='stopped_resource_limit',error=str(error),config=asdict(config),memory=snapshot,
                      checkpoint=str(checkpoint) if checkpoint.is_file() else None,
                      note='Last completed periodic checkpoint preserved; failed step not checkpointed; not an accepted model.')
        _json_dump(output_dir / ('resource_stop_' + stamp + '.json'), report)
        raise ResourceLimitError('Training stopped by resource protection: '+str(error)) from error


def train_latest(
    iterations: int | None = None,
    profile: str = "low_vram",
    *,
    input_path: Path | None = None,
    resume: Path | None = None,
    enable_lpips: bool = True,
) -> tuple[Path, dict[str, Any]]:
    if profile not in PROFILES:
        raise ValueError(f"Unknown direct gsplat profile: {profile}")
    config = PROFILES[profile]
    if iterations is not None:
        config = replace(config, iterations=int(iterations), refine_stop=min(config.refine_stop, int(iterations)))
    input_dir = _resolve_input_path(input_path)
    assessment = assess_training_input(input_dir)
    if not assessment["ready"]:
        raise ValueError("Direct gsplat input blocked: " + "; ".join(assessment["reasons"]))
    if resume is not None:
        output_dir = resume.resolve().parent.parent
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = OUTPUTS_ROOT / f"direct_gsplat_{stamp}"
        suffix = 1
        while output_dir.exists():
            output_dir = OUTPUTS_ROOT / f"direct_gsplat_{stamp}_{suffix:02d}"
            suffix += 1
    return train(input_dir, output_dir, config, resume=resume, enable_lpips=enable_lpips)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direct Apache-2.0 gsplat trainer for Femto RGB-D scenes")
    parser.add_argument("action", choices=("status", "train", "install-lpips"), nargs="?", default="status")
    parser.add_argument("--input", type=Path, help="Nerfstudio-format directory or complete demo root")
    parser.add_argument(
        "--output", type=Path,
        help="Explicit result directory; a caller that already owns the run layout passes it here",
    )
    parser.add_argument(
        "--source", type=Path, default=Path.home() / "Downloads" / LPIPS_BACKBONE_FILE,
        help="install-lpips: manually downloaded official AlexNet checkpoint to verify and install",
    )
    parser.add_argument("--profile", choices=tuple(PROFILES), default="low_vram")
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--downscale", type=int)
    parser.add_argument("--max-init-points", type=int)
    parser.add_argument("--max-gaussians", type=int)
    parser.add_argument("--max-cuda-reserved-mib", type=int)
    parser.add_argument("--min-cuda-free-mib", type=int)
    parser.add_argument("--checkpoint-every", type=int)
    parser.add_argument("--refine-start", type=int)
    parser.add_argument("--refine-stop", type=int)
    parser.add_argument("--refine-every", type=int)
    parser.add_argument("--progress-every", type=int)
    parser.add_argument("--grow-grad2d", type=float)
    parser.add_argument("--prune-opacity", type=float)
    parser.add_argument('--depth-weight', type=float)
    parser.add_argument('--means-lr-final-ratio', type=float)
    parser.add_argument('--antialiased', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--no-publish-latest', action='store_true')
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--no-spz", action="store_true")
    parser.add_argument("--skip-lpips", action="store_true", help="Smoke-test only; acceptance will fail LPIPS")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.action == "status":
            payload = {"runtime": runtime_status(), "input": assess_training_input(args.input)}
        elif args.action == "install-lpips":
            payload = {"install": install_lpips_backbone(args.source), "lpips_backbone": lpips_backbone_status()}
        else:
            config = PROFILES[args.profile]
            overrides: dict[str, Any] = {}
            for argument, field in (
                (args.iterations, "iterations"), (args.downscale, "downscale"),
                (args.max_init_points, "max_init_points"), (args.checkpoint_every, "checkpoint_every"),
                (args.max_gaussians, "max_gaussians"), (args.max_cuda_reserved_mib, "max_cuda_reserved_mib"),
                (args.min_cuda_free_mib, "min_cuda_free_mib"),
                (args.refine_start, "refine_start"), (args.refine_stop, "refine_stop"),
                (args.refine_every, "refine_every"), (args.progress_every, "progress_every"),
                (args.grow_grad2d, "grow_grad2d"), (args.prune_opacity, "prune_opacity"),
                (args.depth_weight, 'depth_weight'), (args.means_lr_final_ratio, 'means_lr_final_ratio'),
            ):
                if argument is not None:
                    overrides[field] = int(argument) if field not in {"grow_grad2d", "prune_opacity", 'depth_weight', 'means_lr_final_ratio'} else float(argument)
            if args.antialiased is not None:
                overrides['antialiased'] = args.antialiased
            if args.no_publish_latest:
                overrides['publish_latest'] = False
            if args.no_spz:
                overrides["export_spz"] = False
            config = replace(config, **overrides)
            input_dir = _resolve_input_path(args.input)
            status = assess_training_input(input_dir)
            if not status["ready"]:
                raise ValueError("Direct gsplat input blocked: " + "; ".join(status["reasons"]))
            if args.resume:
                output_dir = args.resume.resolve().parent.parent
            elif args.output is not None:
                output_dir = args.output.resolve()
            else:
                stamp = time.strftime("%Y%m%d_%H%M%S")
                output_dir = OUTPUTS_ROOT / f"direct_gsplat_{stamp}"
            model, report = train(
                input_dir, output_dir, config, resume=args.resume,
                enable_lpips=not args.skip_lpips,
            )
            payload = {"status": "trained", "model": str(model), "report": report}
    except Exception as error:
        if os.environ.get("FEMTO_GSPLAT_DEBUG") == "1":
            import traceback

            traceback.print_exc()
        print(json.dumps({"status": "blocked", "error": f"{type(error).__name__}: {error}"}, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

