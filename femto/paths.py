"""Repository-relative locations shared by every module (stdlib only).

Modules used to compute ``ROOT = Path(__file__).resolve().parent`` while they
lived flat in the repository root.  Inside the package tree that expression no
longer points at the repository, so every data-tree location is defined here.
Only the marker/evidence trees are listed; ``outputs/`` and ``datasets/`` stay
outside version control.
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

REPO_ROOT = (Path(sys.executable).resolve().parent if getattr(sys, 'frozen', False)
             else Path(__file__).resolve().parents[1])
DATASETS_ROOT = REPO_ROOT / "datasets"
OUTPUTS_ROOT = REPO_ROOT / "outputs"
PHOTOS_ROOT = REPO_ROOT / "photos"
MODELS_ROOT = REPO_ROOT / "models"
LOG_ROOT = REPO_ROOT / "Log"
COMPLIANCE_ROOT = REPO_ROOT / "compliance"
GSPLAT_RUNTIME_ROOT = REPO_ROOT / ".venv-gsplat"
SPLATFACTO_RUNTIME_ROOT = REPO_ROOT / ".venv-3dgs"


def execution_database(outputs: Path | None = None) -> Path:
    return Path(outputs if outputs is not None else OUTPUTS_ROOT) / 'history_metadata' / 'execution.sqlite3'


def execution_resource_root(outputs: Path | None = None) -> Path:
    return Path(outputs if outputs is not None else OUTPUTS_ROOT) / 'simulation_resources'


def hardware_configuration_root(outputs: Path | None = None) -> Path:
    return Path(outputs if outputs is not None else OUTPUTS_ROOT) / 'hardware_configuration'


def installation_calibration_root(outputs: Path | None = None) -> Path:
    return Path(outputs if outputs is not None else OUTPUTS_ROOT) / 'installation_calibrations'


def child_path(root: Path, name: str) -> Path:
    """Join ``name`` under ``root`` and refuse anything that escapes ``root``."""
    path = (Path(root) / name).resolve()
    path.relative_to(Path(root).resolve())
    return path


def workspace_id(root: Path) -> str:
    """Stable identity for a checkout, independent of its active Git branch."""
    canonical = os.path.normcase(str(Path(root).resolve()))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


WORKSPACE_ID = workspace_id(REPO_ROOT)
WORKSPACE_CACHE_ROOT = (
    Path(os.environ.get("LOCALAPPDATA", str(Path.home() / ".cache")))
    / "FemtoWorkspaces" / WORKSPACE_ID
)
ASCII_GSPLAT_RUNTIME_ROOT = (
    Path("C:/FemtoRuntimes") / WORKSPACE_ID / "gsplat"
    if os.name == "nt" else GSPLAT_RUNTIME_ROOT
)


def workspace_runtime_environment() -> dict[str, str]:
    """Writable ML caches belong to this checkout, never another branch."""
    return {
        "TORCH_HOME": str(WORKSPACE_CACHE_ROOT / "torch"),
        "TORCH_EXTENSIONS_DIR": str(WORKSPACE_CACHE_ROOT / "torch_extensions"),
        "HF_HOME": str(WORKSPACE_CACHE_ROOT / "huggingface"),
        "XDG_CACHE_HOME": str(WORKSPACE_CACHE_ROOT / "cache"),
    }


def workspace_subprocess_environment() -> dict[str, str]:
    """Shared environment for product launch and status probes."""
    environment = dict(os.environ)
    for name in ('PYTHONHOME', 'PYTHONPATH'):
        environment.pop(name, None)
    environment.update(workspace_runtime_environment())
    environment.update(PYTHONNOUSERSITE='1', PYTHONUTF8='1',
                       PYTHONIOENCODING='utf-8', VIRTUAL_ENV=str(ASCII_GSPLAT_RUNTIME_ROOT))
    return environment
