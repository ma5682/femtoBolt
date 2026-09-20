"""Repository-relative locations shared by every module (stdlib only).

Modules used to compute ``ROOT = Path(__file__).resolve().parent`` while they
lived flat in the repository root.  Inside the package tree that expression no
longer points at the repository, so every data-tree location is defined here.
Only the marker/evidence trees are listed; ``outputs/`` and ``datasets/`` stay
outside version control.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS_ROOT = REPO_ROOT / "datasets"
OUTPUTS_ROOT = REPO_ROOT / "outputs"
PHOTOS_ROOT = REPO_ROOT / "photos"
MODELS_ROOT = REPO_ROOT / "models"
LOG_ROOT = REPO_ROOT / "Log"
COMPLIANCE_ROOT = REPO_ROOT / "compliance"
GSPLAT_RUNTIME_ROOT = REPO_ROOT / ".venv-gsplat"
SPLATFACTO_RUNTIME_ROOT = REPO_ROOT / ".venv-3dgs"


def child_path(root: Path, name: str) -> Path:
    """Join ``name`` under ``root`` and refuse anything that escapes ``root``."""
    path = (Path(root) / name).resolve()
    path.relative_to(Path(root).resolve())
    return path
