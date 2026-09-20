"""Product 3DGS backend controller for the Femto operator page.

The operator page runs in the capture ``.venv`` (Python 3.12, no PyTorch/CUDA),
while the product trainer ``scene_gsplat_trainer.py`` must execute inside
``.venv-gsplat`` (Python 3.10, PyTorch 2.7.1+cu128, gsplat 1.5.3).  This module
owns that process boundary and nothing else:

* resolve a training input from an existing complete-demo reconstruction;
* launch the trainer with the same environment ``start_direct_gsplat_train.cmd``
  prepares, streaming its progress records back to the caller;
* locate the trained model and open the direct-gsplat viewer.

Training, densification, evaluation, acceptance and export logic stay in
``scene_gsplat_trainer`` / ``scene_gsplat_export``; none of it is duplicated here
or in the UI.  ``scene_3dgs`` remains the internal Nerfstudio/Splatfacto
comparison backend, reachable through its own CLI and ``start_3dgs_*.cmd``
entries; it is no longer the default backend of the operator page.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Any, Callable

from io_utils import load_json as _load_json
from scene_gsplat_trainer import (
    ASCII_RUNTIME_ROOT,
    LATEST_MARKER,
    OUTPUTS_ROOT,
    PROFILES,
    assess_training_input as assess_direct_input,
    runtime_python,
    runtime_status,
)


__all__ = [
    "BACKEND_NAME",
    "PROFILES",
    "PROFILE_KEYS",
    "TrainedModel",
    "assess_training_input",
    "latest_trained_model",
    "open_viewer",
    "resolve_training_input",
    "runtime_environment",
    "runtime_status",
    "status",
    "train",
]


ROOT = Path(__file__).resolve().parent
PROFILE_KEYS: tuple[str, ...] = tuple(PROFILES)
BACKEND_NAME = "direct_gsplat"
BACKEND_DESCRIPTION = "direct nerfstudio-project/gsplat public API (Apache-2.0)"
INTERNAL_COMPARISON_BACKEND = "scene_3dgs.py (Nerfstudio Splatfacto; internal only, not redistributable)"
# CUDA JIT and the MSVC toolchain both fail under the repository's non-ASCII path,
# so the runtime is reached through the ASCII directory junction created by
# setup_direct_gsplat_environment.cmd.  Mirrors start_direct_gsplat_train.cmd.
CUDA_TOOLKIT_VERSION = "v12.8"
TORCH_CUDA_ARCH_LIST = "12.0"
_MSVC_ENVIRONMENT_KEYS = (
    "PATH", "INCLUDE", "LIB", "LIBPATH", "VCINSTALLDIR", "VCTOOLSINSTALLDIR",
    "VSINSTALLDIR", "WINDOWSSDKDIR", "WINDOWSSDKVERSION", "UCRTVERSION",
)


@dataclass(frozen=True)
class TrainedModel:
    output_dir: Path
    model_ply: Path
    model_spz: Path | None
    report: dict[str, Any]
    manifest: dict[str, Any]

    @property
    def accepted(self) -> bool:
        return bool(self.report.get("accepted", False))


def _cuda_home() -> Path:
    return (
        Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
        / "NVIDIA GPU Computing Toolkit"
        / "CUDA"
        / CUDA_TOOLKIT_VERSION
    )


def _visual_studio_environment() -> dict[str, str]:
    """Capture the MSVC variables gsplat's CUDA JIT needs when its extension cache is cold.

    Best effort by design: a warm extension cache trains fine without a compiler,
    so a missing Visual Studio installation must not block an otherwise ready
    runtime.  The developer command file is invoked through a generated batch file
    in an ASCII temporary directory: passing ``cmd.exe /c`` a quoted command string
    goes through ``subprocess.list2cmdline``, whose backslash-escaped quotes
    ``cmd.exe`` does not understand, and this repository's own path is non-ASCII.
    """
    if os.name != "nt":
        return {}
    vswhere = (
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
        / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    )
    if not vswhere.is_file():
        return {}
    try:
        located = subprocess.run(
            [
                str(vswhere), "-latest", "-products", "*", "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property", "installationPath",
            ],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8", errors="replace", timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}
    install = located.stdout.strip().splitlines()
    if not install:
        return {}
    developer_command = Path(install[0]) / "Common7" / "Tools" / "VsDevCmd.bat"
    if not developer_command.is_file():
        return {}
    workspace = Path(tempfile.mkdtemp(prefix="femto_vsenv_"))
    script = workspace / "capture_env.bat"
    script.write_text(
        f'@echo off\r\ncall "{developer_command}" -arch=x64 >nul\r\nset\r\n', encoding="ascii"
    )
    try:
        captured = subprocess.run(
            ["cmd.exe", "/c", str(script)], cwd=str(workspace),
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding="mbcs", errors="replace", timeout=180,
        )
    except (OSError, subprocess.TimeoutExpired, LookupError):
        return {}
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
    if captured.returncode != 0:
        return {}
    environment: dict[str, str] = {}
    for line in captured.stdout.splitlines():
        name, separator, value = line.partition("=")
        if separator and name.upper() in _MSVC_ENVIRONMENT_KEYS:
            environment[name] = value
    return environment


def runtime_environment() -> dict[str, str]:
    """Build the trainer subprocess environment (mirrors start_direct_gsplat_train.cmd)."""
    environment = dict(os.environ)
    environment.update(_visual_studio_environment())
    cuda_home = _cuda_home()
    prefix = [str(ASCII_RUNTIME_ROOT / "Scripts"), str(cuda_home / "bin")]
    environment["CUDA_HOME"] = str(cuda_home)
    environment["PATH"] = os.pathsep.join(prefix + [environment.get("PATH", "")])
    environment["TORCH_CUDA_ARCH_LIST"] = TORCH_CUDA_ARCH_LIST
    environment["VSLANG"] = "1033"
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    return environment


def resolve_training_input(demo_root: Path | None) -> Path:
    """Map an operator-page reconstruction result to its Nerfstudio training input."""
    if demo_root is None:
        marker = OUTPUTS_ROOT / "latest_complete_demo.txt"
        if not marker.is_file():
            raise ValueError("No completed reconstruction is available")
        demo_root = Path(marker.read_text(encoding="utf-8-sig").strip()).resolve().parent
    candidate = Path(demo_root).resolve()
    if (candidate / "transforms.json").is_file():
        return candidate
    summary = candidate / "demo_summary.json"
    if summary.is_file():
        return Path(_load_json(summary).get("nerfstudio", candidate / "nerfstudio")).resolve()
    nested = candidate / "nerfstudio"
    if (nested / "transforms.json").is_file():
        return nested
    raise ValueError(f"Cannot find a Nerfstudio training input under {candidate}")


def prepare_training_input(demo_root: Path | None) -> Path:
    input_dir = resolve_training_input(demo_root)
    if _load_json(input_dir / 'transforms.json').get('dataset_split') is not None:
        return input_dir
    manifest = _load_json(input_dir / 'conversion_manifest.json')
    source = Path(manifest['source_dataset'])
    if not source.is_absolute():
        source = input_dir / source
    from scene_to_nerfstudio import convert
    output = input_dir.parent / ('nerfstudio_strict_' + str(time.time_ns()))
    initialization = manifest.get('initialization') or {}
    convert(source.resolve(), output,
            seed_voxel_m=float(initialization.get('voxel_m', 0.01)),
            max_seed_points=int(initialization.get('max_points', 1000000)))
    return output


def assess_training_input(demo_root: Path | None = None, *, require_real: bool = False) -> dict[str, Any]:
    """Direct-gsplat input gate for the operator page.

    Delegates the actual gate to ``scene_gsplat_trainer`` so the page and the CLI
    can never drift apart on the accepted-view, viewpoint-span and seed-PLY rules.
    """
    try:
        input_dir = resolve_training_input(demo_root)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return {"ready": False, "backend": BACKEND_NAME, "input": None, "reasons": [str(error)]}
    assessment = assess_direct_input(input_dir)
    if require_real and assessment.get("provenance", {}).get("data_kind") != "real_femto":
        assessment["ready"] = False
        assessment["reasons"].append("Real Femto input required; synthetic/unknown inputs are for explicit regression tests only")
    assessment["backend"] = BACKEND_NAME
    quality = input_dir / "quality" / "3dgs_input_report.json"
    if quality.is_file():
        try:
            assessment["input_quality"] = _load_json(quality)
        except (OSError, json.JSONDecodeError):
            pass
    return assessment


def _new_output_dir() -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUTS_ROOT / f"direct_gsplat_{stamp}"
    suffix = 1
    while output_dir.exists():
        output_dir = OUTPUTS_ROOT / f"direct_gsplat_{stamp}_{suffix:02d}"
        suffix += 1
    return output_dir


def build_training_command(
    executable: Path,
    input_dir: Path,
    output_dir: Path,
    *,
    profile: str,
    iterations: int | None = None,
    downscale: int | None = None,
    max_init_points: int | None = None,
    max_gaussians: int | None = None,
    max_cuda_reserved_mib: int | None = None,
    min_cuda_free_mib: int | None = None,
    export_spz: bool = True,
    enable_lpips: bool = True,
) -> list[str]:
    if profile not in PROFILES:
        raise ValueError(f"Unknown direct gsplat profile: {profile}")
    command = [
        str(executable), str(ROOT / "scene_gsplat_trainer.py"), "train",
        "--profile", profile,
        "--input", str(input_dir),
        "--output", str(output_dir),
    ]
    for value, flag in ((iterations, "--iterations"), (downscale, "--downscale"), (max_init_points, "--max-init-points"),
                        (max_gaussians, "--max-gaussians"), (max_cuda_reserved_mib, "--max-cuda-reserved-mib"),
                        (min_cuda_free_mib, "--min-cuda-free-mib")):
        if value is not None:
            command.extend((flag, str(int(value))))
    if not export_spz:
        command.append("--no-spz")
    if not enable_lpips:
        command.append("--skip-lpips")
    return command


def train(
    demo_root: Path | None = None,
    profile: str = "low_vram",
    *,
    output_dir: Path | None = None,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    require_real: bool = False,
    **overrides: Any,
) -> tuple[Path, dict[str, Any]]:
    """Run the product trainer in ``.venv-gsplat`` and return its result directory."""
    input_dir = prepare_training_input(demo_root)
    assessment = assess_training_input(input_dir, require_real=require_real)
    if not assessment["ready"]:
        raise ValueError("Direct gsplat input blocked: " + "; ".join(assessment["reasons"]))
    executable = runtime_python()
    if not executable.is_file():
        raise RuntimeError(
            "The direct gsplat runtime is not installed. Run setup_direct_gsplat_environment.cmd."
        )
    output_dir = (output_dir or _new_output_dir()).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    command = build_training_command(executable, input_dir, output_dir, profile=profile, **overrides)
    log_path = output_dir / "trainer_stdout.log"
    tail: list[str] = []
    with subprocess.Popen(
        command, cwd=str(ROOT), env=runtime_environment(),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    ) as process, log_path.open("w", encoding="utf-8") as log:
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            stripped = line.strip()
            if not stripped:
                continue
            tail.append(stripped)
            del tail[:-40]
            if on_progress is not None and stripped.startswith("{") and stripped.endswith("}"):
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                if "step" in record:
                    on_progress(record)
        return_code = process.wait()
    report_path = output_dir / "training_report.json"
    if return_code != 0 or not report_path.is_file():
        raise RuntimeError(
            f"Direct gsplat training failed (exit {return_code}); see {log_path}\n" + "\n".join(tail[-8:])
        )
    return output_dir, _load_json(report_path)


def latest_trained_model(output_dir: Path | None = None) -> TrainedModel:
    """Load the most recent trained direct-gsplat model, or the one at ``output_dir``."""
    if output_dir is None:
        if not LATEST_MARKER.is_file():
            raise ValueError("No trained direct gsplat model is available")
        output_dir = Path(LATEST_MARKER.read_text(encoding="utf-8-sig").strip())
    output_dir = Path(output_dir).resolve()
    report_path = output_dir / "training_report.json"
    if not report_path.is_file():
        raise ValueError(f"Trained direct gsplat result is missing its report: {report_path}")
    report = _load_json(report_path)
    model_ply = Path(report.get("model_ply") or (output_dir / "model" / "trained_gaussians.ply"))
    if not model_ply.is_file():
        raise ValueError(f"Trained Gaussian PLY no longer exists: {model_ply}")
    spz_value = report.get("model_spz")
    spz = Path(spz_value) if spz_value and Path(spz_value).is_file() else None
    manifest_path = output_dir / "model" / "model_manifest.json"
    manifest = _load_json(manifest_path) if manifest_path.is_file() else {}
    return TrainedModel(
        output_dir=output_dir, model_ply=model_ply, model_spz=spz, report=report, manifest=manifest
    )


def open_viewer(output_dir: Path | None = None, *, language: str = "zh") -> subprocess.Popen[Any]:
    """Open the trained model in the direct-gsplat renderer inside ``.venv-gsplat``.

    The Open3D desktop viewer can only draw Gaussian *positions* as points, so the
    trained model is rendered by ``scene_gsplat_viewer.py`` through the same gsplat
    rasterizer that produced it.
    """
    model = latest_trained_model(output_dir)
    executable = runtime_python()
    if not executable.is_file():
        raise RuntimeError(
            "The direct gsplat runtime is not installed; the trained PLY exists but cannot be rendered."
        )
    command = [
        str(executable), str(ROOT / "scene_gsplat_viewer.py"),
        str(model.output_dir), "--lang", language,
    ]
    log_path = model.output_dir / "viewer_stdout.log"
    with log_path.open("a", encoding="utf-8") as log:
        return subprocess.Popen(
            command, cwd=str(ROOT), env=runtime_environment(), stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )


def status(demo_root: Path | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "backend": BACKEND_NAME,
        "backend_description": BACKEND_DESCRIPTION,
        "internal_comparison_backend": INTERNAL_COMPARISON_BACKEND,
        "profiles": list(PROFILE_KEYS),
        "runtime": runtime_status(),
        "input": assess_training_input(demo_root),
    }
    try:
        model = latest_trained_model()
        from scene_gsplat_acceptance import input_provenance

        provenance = model.report.get("input_provenance") or input_provenance(Path(model.report.get("dataset_root") or model.output_dir))
        payload["trained_model"] = {
            "output_dir": str(model.output_dir),
            "model_ply": str(model.model_ply),
            "model_spz": str(model.model_spz) if model.model_spz else None,
            "accepted": model.accepted,
            "final_gaussians": model.report.get("final_gaussians"),
            "data_kind": provenance["data_kind"],
        }
    except (OSError, ValueError, json.JSONDecodeError) as error:
        payload["trained_model"] = {"available": False, "reason": str(error)}
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Operator-page controller for the direct gsplat backend")
    parser.add_argument("action", choices=("status", "train", "view"), nargs="?", default="status")
    parser.add_argument("--input", type=Path, help="Complete demo root or Nerfstudio directory")
    parser.add_argument("--profile", choices=PROFILE_KEYS, default="low_vram")
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--downscale", type=int, help="Override training image downscale")
    parser.add_argument("--max-init-points", type=int, help="Override initial Gaussian point cap")
    parser.add_argument("--no-spz", action="store_true", help="Keep the PLY only and record SPZ as not requested")
    parser.add_argument("--skip-lpips", action="store_true", help="Smoke test only; acceptance cannot pass")
    parser.add_argument("--require-real", action="store_true", help="Block synthetic or unknown input lineage")
    parser.add_argument("--lang", choices=("zh", "en"), default="zh")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.action == "status":
            payload: dict[str, Any] = status(args.input)
        elif args.action == "train":
            output_dir, report = train(
                args.input,
                args.profile,
                iterations=args.iterations,
                downscale=args.downscale,
                max_init_points=args.max_init_points,
                export_spz=not args.no_spz,
                enable_lpips=not args.skip_lpips,
                require_real=args.require_real,
            )
            payload = {
                "status": "trained",
                "output": str(output_dir),
                "accepted": report.get("accepted"),
                "evaluation": (report.get("evaluation") or {}).get("aggregate"),
            }
        else:
            process = open_viewer(args.input, language=args.lang)
            payload = {"status": "viewer_started", "pid": process.pid}
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "blocked", "error": str(error)}, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
