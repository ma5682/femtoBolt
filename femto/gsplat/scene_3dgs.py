"""Development 3DGS launcher with a clean boundary around Nerfstudio/gsplat CLIs.

The project owns the Femto dataset conversion and quality gates.  Training and
rasterization are delegated to a separate Nerfstudio/gsplat installation.  No
Graphdeco source code is copied or linked here.  The Nerfstudio environment is
not approved for redistribution until its transitive dependencies are audited;
the production target is a direct Apache-2.0 gsplat adapter.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any

from femto.io_utils import load_json, write_json


from femto.paths import REPO_ROOT as ROOT
OUTPUTS_ROOT = ROOT / "outputs"
RUNTIME_ROOT = ROOT / ".venv-3dgs"
LATEST_GEOMETRY_MARKER = OUTPUTS_ROOT / "latest_complete_demo.txt"
LATEST_3DGS_MARKER = OUTPUTS_ROOT / "latest_3dgs_config.txt"
TRAINING_PROFILES: dict[str, dict[str, Any]] = {
    "low_vram": {
        "iterations": 15_000,
        "description": "8 GB GPU baseline; antialiasing and scale regularization when supported",
    },
    "balanced": {
        "iterations": 30_000,
        "description": "longer optimization for normal production-quality tests",
    },
    "quality": {
        "iterations": 45_000,
        "description": "long quality run; monitor VRAM and training convergence",
    },
}
OPTION_FLAGS = {
    "antialiasing": ("--pipeline.model.rasterize-mode", "antialiased"),
    "scale_regularization": ("--pipeline.model.use-scale-regularization", "True"),
    "bilateral_grid": ("--pipeline.model.use-bilateral-grid", "True"),
}


def _runtime_command(name: str) -> Path | None:
    suffix = ".exe" if os.name == "nt" else ""
    local = RUNTIME_ROOT / ("Scripts" if os.name == "nt" else "bin") / f"{name}{suffix}"
    if local.is_file():
        return local
    found = shutil.which(name)
    return Path(found).resolve() if found else None


def setup_prerequisites() -> dict[str, Any]:
    python_310 = False
    py_launcher = shutil.which("py")
    if py_launcher:
        try:
            result = subprocess.run(
                [py_launcher, "-3.10", "-c", "import sys; print(sys.executable)"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=10,
            )
            python_310 = result.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            pass
    cuda_128 = shutil.which("nvcc")
    if not cuda_128 and os.name == "nt":
        candidate = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / (
            r"NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe"
        )
        cuda_128 = str(candidate) if candidate.is_file() else None
    compiler = shutil.which("cl")
    vswhere = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / (
        r"Microsoft Visual Studio\Installer\vswhere.exe"
    )
    if not compiler and vswhere.is_file():
        try:
            result = subprocess.run(
                [
                    str(vswhere), "-latest", "-products", "*", "-requires",
                    "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=10,
            )
            compiler = result.stdout.strip() or None
        except (OSError, subprocess.TimeoutExpired):
            pass
    blockers = []
    if not python_310:
        blockers.append("Python 3.10 x64")
    if not cuda_128:
        blockers.append("CUDA Toolkit 12.8 (nvcc)")
    if not compiler:
        blockers.append("Visual Studio 2022 C++ Build Tools")
    return {
        "ready_for_install": not blockers,
        "python_310": python_310,
        "nvcc": str(cuda_128) if cuda_128 else None,
        "cpp_build_tools": str(compiler) if compiler else None,
        "blockers": blockers,
    }


def latest_demo_root() -> Path:
    if not LATEST_GEOMETRY_MARKER.is_file():
        raise ValueError("No completed reconstruction is available")
    reconstruction = Path(LATEST_GEOMETRY_MARKER.read_text(encoding="utf-8-sig").strip()).resolve()
    root = reconstruction.parent
    if not (root / "demo_summary.json").is_file():
        raise ValueError(f"Missing demo_summary.json beside {reconstruction}")
    return root


def assess_training_input(demo_root: Path | None = None) -> dict[str, Any]:
    root = (demo_root or latest_demo_root()).resolve()
    summary = load_json(root / "demo_summary.json")
    conversion = dict(summary.get("nerfstudio_conversion") or {})
    input_dir = Path(summary.get("nerfstudio", root / "nerfstudio")).resolve()
    conversion_path = input_dir / "conversion_manifest.json"
    if conversion_path.is_file():
        conversion = load_json(conversion_path)
    frame_count = int(conversion.get("frame_count", summary.get("frame_count", 0)))
    translation_span_m = float(conversion.get("translation_span_m", 0.0))
    rotation_span_deg = float(conversion.get("rotation_span_deg", 0.0))
    fixed_camera = bool(summary.get("fixed_camera", frame_count <= 1))
    reasons: list[str] = []
    if not (input_dir / "transforms.json").is_file():
        reasons.append("missing transforms.json")
    transforms: dict[str, Any] = {}
    if (input_dir / "transforms.json").is_file():
        transforms = load_json(input_dir / "transforms.json")
    seed_name = str(transforms.get("ply_file_path", ""))
    seed_path = input_dir / seed_name if seed_name else None
    if seed_path is None or not seed_path.is_file():
        reasons.append("missing RGB-D initialization PLY; rebuild the multiview result")
    if fixed_camera or frame_count < 12:
        reasons.append("3DGS needs at least 12 accepted moving-camera views")
    if translation_span_m < 0.05 and rotation_span_deg < 5.0:
        reasons.append("camera viewpoint diversity is too small")
    return {
        "ready": not reasons,
        "demo_root": str(root),
        "input": str(input_dir),
        "frame_count": frame_count,
        "translation_span_m": translation_span_m,
        "rotation_span_deg": rotation_span_deg,
        "fixed_camera": fixed_camera,
        "rgbd_seed_ply": str(seed_path) if seed_path else None,
        "rgbd_seed_points": int((conversion.get("initialization") or {}).get("point_count", 0)),
        "photometric_unstable": bool((conversion.get("photometric") or {}).get("unstable", False)),
        "quality_report": conversion.get("quality_report"),
        "reasons": reasons,
    }


def runtime_status() -> dict[str, Any]:
    train = _runtime_command("ns-train")
    viewer = _runtime_command("ns-viewer")
    return {
        "ready": train is not None and viewer is not None,
        "runtime_root": str(RUNTIME_ROOT),
        "ns_train": str(train) if train else None,
        "ns_viewer": str(viewer) if viewer else None,
        "backend": "Development trainer: Nerfstudio Splatfacto + gsplat",
        "production_backend_target": "direct Apache-2.0 gsplat adapter",
        "distribution_status": (
            "do not redistribute the Nerfstudio environment in a closed-source product "
            "until transitive licenses, including PyMeshLab GPL-3.0, are reviewed"
        ),
        "forbidden_backend": "Graphdeco reference implementation is not a product dependency",
        "setup_prerequisites": setup_prerequisites(),
    }


def _training_help(ns_train: str) -> str:
    try:
        completed = subprocess.run(
            [ns_train, "splatfacto", "--help"],
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=45,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return completed.stdout or ""


def build_training_command(
    ns_train: str,
    data: dict[str, Any],
    output: Path,
    *,
    profile: str,
    iterations: int,
    help_text: str,
) -> tuple[list[str], dict[str, Any]]:
    """Build a version-compatible Splatfacto command from flags advertised by its CLI."""
    if profile not in TRAINING_PROFILES:
        raise ValueError(f"Unknown 3DGS training profile: {profile}")
    command = [
        ns_train,
        "splatfacto",
        "--data",
        str(data["input"]),
        "--output-dir",
        str(output),
        "--max-num-iterations",
        str(int(iterations)),
        "--viewer.quit-on-train-completion",
        "True",
    ]
    requested = ["antialiasing", "scale_regularization"]
    if data.get("photometric_unstable"):
        requested.append("bilateral_grid")
    enabled: list[str] = []
    skipped: dict[str, str] = {}
    for feature in requested:
        flag, value = OPTION_FLAGS[feature]
        if flag in help_text:
            command.extend((flag, value))
            enabled.append(feature)
        else:
            skipped[feature] = "not advertised by the installed ns-train splatfacto --help"
    return command, {
        "requested": requested,
        "enabled": enabled,
        "skipped": skipped,
        "compatibility_source": "installed ns-train splatfacto --help",
    }


def train_latest(
    iterations: int | None = None,
    profile: str = "low_vram",
) -> tuple[Path, dict[str, Any]]:
    if profile not in TRAINING_PROFILES:
        raise ValueError(f"Unknown 3DGS training profile: {profile}")
    selected_iterations = int(iterations or TRAINING_PROFILES[profile]["iterations"])
    if selected_iterations <= 0:
        raise ValueError("3DGS iteration count must be positive")
    data = assess_training_input()
    runtime = runtime_status()
    if not data["ready"]:
        raise ValueError("3DGS input blocked: " + "; ".join(data["reasons"]))
    if not runtime["ready"]:
        raise RuntimeError(
            "The legacy Nerfstudio runtime is unavailable and its setup launcher has been retired. "
            "Use the maintained direct gsplat workflow documented in doc/使用说明/启动入口说明.md."
        )
    stamp = time.strftime("%Y%m%d_%H%M%S")
    output = OUTPUTS_ROOT / f"trained_3dgs_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    help_text = _training_help(str(runtime["ns_train"]))
    command, features = build_training_command(
        str(runtime["ns_train"]),
        data,
        output,
        profile=profile,
        iterations=selected_iterations,
        help_text=help_text,
    )
    log_path = output / "training.log"
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        completed = subprocess.run(command, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)
    configs = sorted(output.rglob("config.yml"), key=lambda item: item.stat().st_mtime, reverse=True)
    status = "trained" if completed.returncode == 0 and configs else "failed"
    report = {
        "schema": "femto-trained-3dgs-v2",
        "created_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "status": status,
        "backend": runtime["backend"],
        "source": data,
        "profile": profile,
        "profile_description": TRAINING_PROFILES[profile]["description"],
        "iterations": selected_iterations,
        "command": command,
        "exit_code": completed.returncode,
        "log": str(log_path),
        "config": str(configs[0]) if configs else None,
        "features": features,
        "rasterization": (
            "antialiased" if "antialiasing" in features["enabled"] else "installed default"
        ),
    }
    write_json(output / "training_report.json", report)
    if status != "trained":
        raise RuntimeError(f"3DGS training failed; inspect {log_path}")
    config = configs[0]
    LATEST_3DGS_MARKER.write_text(str(config), encoding="utf-8")
    return config, report


def latest_trained_config() -> Path:
    if not LATEST_3DGS_MARKER.is_file():
        raise ValueError("No trained 3DGS model is available")
    config = Path(LATEST_3DGS_MARKER.read_text(encoding="utf-8-sig").strip()).resolve()
    if not config.is_file():
        raise ValueError(f"Trained 3DGS config no longer exists: {config}")
    return config


def open_viewer(config: Path | None = None) -> subprocess.Popen[Any]:
    viewer = _runtime_command("ns-viewer")
    if viewer is None:
        raise RuntimeError("ns-viewer is unavailable; the separate 3DGS runtime is not installed")
    target = (config or latest_trained_config()).resolve()
    return subprocess.Popen([str(viewer), "--load-config", str(target)], cwd=str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/view commercial-friendly Femto 3DGS")
    parser.add_argument("action", choices=("status", "train", "view"), default="status", nargs="?")
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--profile", choices=tuple(TRAINING_PROFILES), default="low_vram")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.action == "status":
            payload = {"runtime": runtime_status(), "input": assess_training_input()}
        elif args.action == "train":
            config, report = train_latest(args.iterations, args.profile)
            payload = {"status": "trained", "config": str(config), "report": report}
        else:
            process = open_viewer()
            payload = {"status": "viewer_started", "pid": process.pid}
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "blocked", "error": str(error)}, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
