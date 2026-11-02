"""Differentiable image/depth losses and 3DGS acceptance metrics.

Depth acceptance definition: every target pixel with a valid Femto depth counts.
Pixels the model fails to render (expected depth 0, i.e. a hole) are scored as a
full miss against the target depth, not excluded, so RMSE/MAE/AbsRel cannot be
flattered by dropping black holes.  ``coverage_ratio`` reports the rendered share
separately so a low-coverage model is blocked even if its covered pixels look
accurate.
"""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np


# torchmetrics LPIPS ("alex") bundles the BSD-2 linear heads but pulls the AlexNet
# backbone through torchvision, which downloads this exact file into the torch hub
# cache unless it is already present.  The product runtime must not fetch weights
# implicitly, so the file is provisioned offline and pinned by full SHA-256.
LPIPS_BACKBONE_FILE = "alexnet-owt-7be5be79.pth"
LPIPS_BACKBONE_SHA256 = "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02"
LPIPS_BACKBONE_URL = "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth"
LPIPS_BACKBONE_SIZE_BYTES = 244_408_911


def torch_hub_checkpoint_dir() -> Path:
    """Mirror ``torch.hub.get_dir()/checkpoints`` without importing torch.

    Kept torch-free so status reporting and unit tests work in the capture ``.venv``.
    """
    torch_home = os.environ.get("TORCH_HOME")
    if torch_home:
        hub = Path(os.path.expanduser(torch_home)) / "hub"
    else:
        from femto.paths import WORKSPACE_CACHE_ROOT
        hub = WORKSPACE_CACHE_ROOT / "torch" / "hub"
    return hub / "checkpoints"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lpips_backbone_status(
    expected_sha256: str = LPIPS_BACKBONE_SHA256, *, verify_hash: bool = True,
    checkpoint_dir: Path | None = None,
) -> dict[str, Any]:
    path = (Path(checkpoint_dir) if checkpoint_dir is not None else torch_hub_checkpoint_dir()) / LPIPS_BACKBONE_FILE
    status: dict[str, Any] = {
        "file": LPIPS_BACKBONE_FILE,
        "path": str(path),
        "source_url": LPIPS_BACKBONE_URL,
        "expected_sha256": expected_sha256,
        "present": path.is_file(),
        "sha256": None,
        "sha256_verified": False,
        "ready": False,
        "error": None,
    }
    if not path.is_file():
        status["error"] = (
            f"LPIPS AlexNet backbone missing at {path}; download {LPIPS_BACKBONE_URL} "
            "and run install_lpips_weights.cmd"
        )
        return status
    if verify_hash:
        digest = sha256_file(path)
        status["sha256"] = digest
        status["sha256_verified"] = digest == expected_sha256
        status["ready"] = status["sha256_verified"]
        if not status["ready"]:
            status["error"] = f"cached {LPIPS_BACKBONE_FILE} SHA-256 {digest} differs from the official {expected_sha256}"
    else:
        status["ready"] = True
    return status


def install_lpips_backbone(source: Path, expected_sha256: str = LPIPS_BACKBONE_SHA256) -> dict[str, Any]:
    """Copy a manually downloaded AlexNet checkpoint into the torch hub cache.

    The source is hashed before and the copy after writing; a mismatch never lands
    under the official filename, so torchvision cannot silently load a tampered file.
    """
    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError(f"LPIPS backbone source not found: {source}")
    digest = sha256_file(source)
    if digest != expected_sha256:
        raise ValueError(
            f"{source.name} SHA-256 {digest} does not match the official {expected_sha256}; refusing to install"
        )
    target_dir = torch_hub_checkpoint_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / LPIPS_BACKBONE_FILE
    if target.is_file() and sha256_file(target) == expected_sha256:
        return {"status": "already_installed", "path": str(target), "sha256": digest, "source": str(source)}
    temporary = target.with_name(target.name + ".femto_tmp")
    shutil.copyfile(source, temporary)
    if sha256_file(temporary) != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"copy of {source.name} failed verification; nothing installed")
    os.replace(temporary, target)
    return {"status": "installed", "path": str(target), "sha256": digest, "source": str(source)}


def psnr_numpy(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape:
        raise ValueError("PSNR arrays must have the same shape")
    mse = float(np.mean(np.square(prediction - target)))
    return float("inf") if mse <= 1e-15 else float(-10.0 * math.log10(mse))


def _as_nchw(value: Any) -> Any:
    if value.ndim == 3:
        value = value.unsqueeze(0)
    if value.ndim != 4:
        raise ValueError("Image tensor must be HWC, NHWC or NCHW")
    if value.shape[-1] == 3:
        value = value.permute(0, 3, 1, 2)
    if value.shape[1] != 3:
        raise ValueError("Image tensor must have three colour channels")
    return value


def ssim_torch(prediction: Any, target: Any, window_size: int = 11) -> Any:
    """Channel-averaged Gaussian-window SSIM using only public PyTorch ops."""
    import torch
    import torch.nn.functional as functional

    x = _as_nchw(prediction)
    y = _as_nchw(target)
    if x.shape != y.shape:
        raise ValueError("SSIM tensors must have the same shape")
    size = min(int(window_size), int(x.shape[-2]), int(x.shape[-1]))
    if size < 1:
        raise ValueError("SSIM needs a non-empty image")
    if size % 2 == 0:
        size -= 1
    sigma = max(0.5, 1.5 * size / 11.0)
    coordinates = torch.arange(size, device=x.device, dtype=x.dtype) - (size - 1) / 2.0
    kernel_1d = torch.exp(-(coordinates**2) / (2.0 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel = (kernel_1d[:, None] * kernel_1d[None, :]).expand(3, 1, size, size)
    padding = size // 2
    mu_x = functional.conv2d(x, kernel, padding=padding, groups=3)
    mu_y = functional.conv2d(y, kernel, padding=padding, groups=3)
    mu_x2 = mu_x.square()
    mu_y2 = mu_y.square()
    mu_xy = mu_x * mu_y
    sigma_x2 = functional.conv2d(x * x, kernel, padding=padding, groups=3) - mu_x2
    sigma_y2 = functional.conv2d(y * y, kernel, padding=padding, groups=3) - mu_y2
    sigma_xy = functional.conv2d(x * y, kernel, padding=padding, groups=3) - mu_xy
    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    return (numerator / denominator.clamp_min(1e-12)).mean()


def metric_depth_loss_torch(
    prediction_m: Any,
    target_m: Any,
    beta_m: float = 0.02,
    confidence: Any | None = None,
) -> tuple[Any, Any]:
    """Confidence-weighted metric-depth Smooth-L1 loss.

    Original Femto observations normally carry weight 1.0, conservative repairs
    carry weight 0.3, and unknown pixels carry weight 0.0.  Legacy datasets that
    do not provide a confidence map retain the previous unweighted behaviour.
    """
    import torch
    import torch.nn.functional as functional

    prediction = prediction_m.squeeze(-1) if prediction_m.ndim == 3 else prediction_m
    target = target_m.squeeze(-1) if target_m.ndim == 3 else target_m
    valid = torch.isfinite(target) & torch.isfinite(prediction) & (target > 0.0)
    weights = None
    if confidence is not None:
        weights = confidence.squeeze(-1) if confidence.ndim == 3 else confidence
        if weights.shape != target.shape:
            raise ValueError("Depth confidence and target depth must have the same shape")
        if not bool(torch.isfinite(weights).all()) or bool((weights < 0.0).any()) or bool((weights > 1.0).any()):
            raise ValueError("Depth confidence must be finite in [0,1]")
        valid = valid & (weights > 0.0)
    if not bool(valid.any()):
        return prediction.sum() * 0.0, valid
    errors = functional.smooth_l1_loss(
        prediction[valid], target[valid], beta=float(beta_m), reduction="none"
    )
    if weights is None:
        return errors.mean(), valid
    valid_weights = weights[valid]
    return (errors * valid_weights).sum() / valid_weights.sum().clamp_min(1e-12), valid


def depth_metrics_numpy(prediction_m: np.ndarray, target_m: np.ndarray) -> dict[str, float | int | None]:
    """Metric depth errors over every valid target pixel; unrendered holes count as 0 m."""
    prediction = np.asarray(prediction_m, dtype=np.float64).squeeze()
    target = np.asarray(target_m, dtype=np.float64).squeeze()
    if prediction.shape != target.shape:
        raise ValueError("Depth metric arrays must have the same shape")
    valid = np.isfinite(target) & (target > 0.0)
    count = int(valid.sum())
    if not count:
        return {
            "valid_pixels": 0, "predicted_pixels": 0, "coverage_ratio": None,
            "rmse_m": None, "mae_m": None, "abs_rel": None,
        }
    predicted = np.isfinite(prediction) & (prediction > 0.0)
    prediction_safe = np.where(predicted, prediction, 0.0)
    error = prediction_safe[valid] - target[valid]
    return {
        "valid_pixels": count,
        "predicted_pixels": int((valid & predicted).sum()),
        "coverage_ratio": float((valid & predicted).sum() / count),
        "rmse_m": float(np.sqrt(np.mean(error * error))),
        "mae_m": float(np.mean(np.abs(error))),
        "abs_rel": float(np.mean(np.abs(error) / np.maximum(target[valid], 1e-6))),
    }


class LpipsMetric:
    """One reusable torchmetrics LPIPS network with explicit availability state."""

    def __init__(self, device: Any) -> None:
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        # Refuse to construct when the backbone is absent: torchvision would otherwise
        # start an unverified network download inside the product trainer.
        self.backbone = lpips_backbone_status()
        if not self.backbone["ready"]:
            raise FileNotFoundError(self.backbone["error"])
        import torch
        torch.hub.set_dir(str(torch_hub_checkpoint_dir().parent))
        self.metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)
        self.metric.eval()

    def __call__(self, prediction: Any, target: Any) -> float:
        import torch

        with torch.no_grad():
            x = _as_nchw(prediction).clamp(0.0, 1.0)
            y = _as_nchw(target).clamp(0.0, 1.0)
            return float(self.metric(x, y).detach().cpu())
