"""Trained Gaussian PLY and optional official Niantic SPZ export.

The PLY layout follows the widely supported scalar 3DGS property convention.
SPZ conversion is delegated to Niantic's MIT-licensed ``spz`` package; this
project does not invent a binary SPZ writer or silently rename a PLY file.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any

import numpy as np

from femto.gsplat.scene_gsplat_data import read_ply_vertex_records
from femto.io_utils import write_json


SPZ_SOURCE_TAG = "v3.0.0"
SPZ_SOURCE_COMMIT = "5bf2945de1a003cee07133b1e495fe9c6ffdc7e7"

# Keep DLL directory handles alive for the lifetime of the process.  Python's
# Windows extension loader intentionally does not search arbitrary sibling
# package directories; the official SPZ wheel may depend on zlib.dll.
_SPZ_DLL_HANDLES: list[Any] = []


def _import_spz_binding() -> Any:
    """Import Niantic's binding with a deterministic Windows DLL search path."""
    if os.name == "nt":
        spec = importlib.util.find_spec("spz")
        package_dir = Path(spec.submodule_search_locations[0]) if spec and spec.submodule_search_locations else None
        candidates: list[Path] = []
        if package_dir is not None:
            candidates.extend((package_dir, package_dir / "bin", package_dir.parent / "bin"))
        # torchvision is the known provider of the matching zlib.dll in the
        # local CUDA runtime.  It is only a fallback; setup_spz_environment.cmd
        # also copies the DLL beside the SPZ extension for offline use.
        if package_dir is not None:
            candidates.append(package_dir.parent / "torchvision")
        seen: set[str] = set()
        for directory in candidates:
            key = str(directory.resolve()).lower()
            if key in seen or not directory.is_dir():
                continue
            seen.add(key)
            try:
                _SPZ_DLL_HANDLES.append(os.add_dll_directory(str(directory)))
            except (AttributeError, OSError):
                # Older Python or a directory without loadable DLLs: import
                # can still succeed through the normal module-directory path.
                pass
    try:
        return importlib.import_module("spz")
    except ImportError as error:
        raise RuntimeError(
            "Niantic SPZ Python bindings are not usable; check spz extension DLLs "
            "(zlib.dll) and run setup_spz_environment.cmd"
        ) from error


def _stage_ascii_paths(ply_path: Path, spz_path: Path) -> tuple[tempfile.TemporaryDirectory[str], Path, Path]:
    """Stage native-library IO through an ASCII temp path on Windows.

    The official C++ PLY reader predates Python's Unicode path handling and
    returns an empty cloud for Chinese paths.  Staging is lossless and leaves
    the source PLY untouched; the final SPZ is copied back to the requested
    derived/output path by Python.
    """
    temp_dir = tempfile.TemporaryDirectory(prefix="femto_spz_")
    root = Path(temp_dir.name)
    if not str(root).isascii():
        temp_dir.cleanup()
        from femto.paths import ASCII_GSPLAT_RUNTIME_ROOT
        fallback = ASCII_GSPLAT_RUNTIME_ROOT.parent / "spz-tmp"
        fallback.mkdir(parents=True, exist_ok=True)
        temp_dir = tempfile.TemporaryDirectory(prefix="femto_spz_", dir=str(fallback))
        root = Path(temp_dir.name)
    staged_ply = root / "input.ply"
    staged_spz = root / "output.spz"
    shutil.copyfile(ply_path, staged_ply)
    return temp_dir, staged_ply, staged_spz


def write_trained_gaussian_ply(
    path: Path,
    *,
    means: np.ndarray,
    sh0: np.ndarray,
    sh_rest: np.ndarray,
    opacity_logits: np.ndarray,
    log_scales: np.ndarray,
    quaternions_wxyz: np.ndarray,
    comment: str = "Femto direct gsplat trained model",
) -> dict[str, Any]:
    means = np.asarray(means, dtype=np.float32)
    sh0 = np.asarray(sh0, dtype=np.float32)
    sh_rest = np.asarray(sh_rest, dtype=np.float32)
    opacity_logits = np.asarray(opacity_logits, dtype=np.float32).reshape(-1)
    log_scales = np.asarray(log_scales, dtype=np.float32)
    quaternions = np.asarray(quaternions_wxyz, dtype=np.float32)
    count = len(means)
    if means.shape != (count, 3) or log_scales.shape != (count, 3) or quaternions.shape != (count, 4):
        raise ValueError("means/scales/quaternions have inconsistent shapes")
    if sh0.shape == (count, 3):
        sh0 = sh0[:, None, :]
    if sh0.shape != (count, 1, 3):
        raise ValueError("sh0 must have shape (N, 1, 3)")
    if sh_rest.ndim != 3 or sh_rest.shape[0] != count or sh_rest.shape[2] != 3:
        raise ValueError("sh_rest must have shape (N, K, 3)")
    if opacity_logits.shape != (count,):
        raise ValueError("opacity_logits must have shape (N,)")
    for array in (means, sh0, sh_rest, opacity_logits, log_scales, quaternions):
        if not np.isfinite(array).all():
            raise ValueError("Cannot export non-finite Gaussian parameters")

    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    quaternions = quaternions / np.maximum(norms, 1e-12)
    rest_flat = np.transpose(sh_rest, (0, 2, 1)).reshape(count, -1)
    property_names = [
        "x", "y", "z", "nx", "ny", "nz",
        "f_dc_0", "f_dc_1", "f_dc_2",
        *[f"f_rest_{index}" for index in range(rest_flat.shape[1])],
        "opacity", "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3",
    ]
    dtype = np.dtype([(name, "<f4") for name in property_names])
    records = np.zeros(count, dtype=dtype)
    for axis, name in enumerate(("x", "y", "z")):
        records[name] = means[:, axis]
    for channel, name in enumerate(("f_dc_0", "f_dc_1", "f_dc_2")):
        records[name] = sh0[:, 0, channel]
    for index in range(rest_flat.shape[1]):
        records[f"f_rest_{index}"] = rest_flat[:, index]
    records["opacity"] = opacity_logits
    for axis in range(3):
        records[f"scale_{axis}"] = log_scales[:, axis]
    for axis in range(4):
        records[f"rot_{axis}"] = quaternions[:, axis]

    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"comment {comment}",
        f"element vertex {count}",
        *[f"property float {name}" for name in property_names],
        "end_header",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        output.write("\n".join(header).encode("ascii", errors="replace"))
        records.tofile(output)
    return {
        "path": str(path.resolve()),
        "gaussian_count": count,
        "sh_rest_coefficients_per_channel": int(sh_rest.shape[1]),
        "property_count": len(property_names),
        "format": "binary_little_endian PLY",
    }


def read_trained_gaussian_ply(path: Path) -> dict[str, np.ndarray]:
    """Read back a PLY written by :func:`write_trained_gaussian_ply`.

    Returns the raw stored parameterization — SH coefficients, opacity *logits*,
    *log* scales and a scalar-first quaternion — so a renderer applies exactly the
    same activations the trainer used.  ``sh_rest`` is restored to ``(N, K, 3)``
    from the writer's channel-major flattening.
    """
    records = read_ply_vertex_records(Path(path))
    names = list(records.dtype.names or ())
    required = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity"]
    required += [f"scale_{axis}" for axis in range(3)] + [f"rot_{axis}" for axis in range(4)]
    missing = [name for name in required if name not in names]
    if missing:
        raise ValueError(f"{path} is not a trained 3DGS PLY; missing: {', '.join(missing)}")
    count = len(records)
    rest_names = sorted(
        (name for name in names if name.startswith("f_rest_")),
        key=lambda name: int(name.split("_")[-1]),
    )
    if len(rest_names) % 3:
        raise ValueError("f_rest property count must be divisible by the three colour channels")
    coefficients = len(rest_names) // 3
    rest_flat = (
        np.column_stack([records[name] for name in rest_names]).astype(np.float32)
        if rest_names
        else np.zeros((count, 0), dtype=np.float32)
    )
    sh_rest = rest_flat.reshape(count, 3, coefficients).transpose(0, 2, 1)
    model = {
        "means": np.column_stack([records[name] for name in ("x", "y", "z")]).astype(np.float32),
        "sh0": np.column_stack(
            [records[name] for name in ("f_dc_0", "f_dc_1", "f_dc_2")]
        ).astype(np.float32)[:, None, :],
        "sh_rest": np.ascontiguousarray(sh_rest, dtype=np.float32),
        "opacity_logits": np.asarray(records["opacity"], dtype=np.float32),
        "log_scales": np.column_stack(
            [records[f"scale_{axis}"] for axis in range(3)]
        ).astype(np.float32),
        "quaternions_wxyz": np.column_stack(
            [records[f"rot_{axis}"] for axis in range(4)]
        ).astype(np.float32),
    }
    for name, array in model.items():
        if not np.isfinite(array).all():
            raise ValueError(f"Trained Gaussian PLY contains non-finite {name}")
    return model


def export_official_spz(
    ply_path: Path,
    spz_path: Path,
    *,
    source_coordinate: str = "RDF",
) -> dict[str, Any]:
    """Convert a trained PLY with Niantic's official Python bindings.

    Femto capture worlds anchored to the first RDF camera use ``RDF`` here.
    The official loader converts into RUB before packing, and the packer is
    explicitly told that its input cloud is RUB.  Unknown world conventions
    are rejected instead of applying an unrecorded axis guess.
    """
    coordinate = source_coordinate.upper()
    if coordinate != "RDF":
        raise ValueError(
            "The official PLY loader defines its input as RDF. Convert a non-RDF world into a "
            "documented RDF export frame before requesting SPZ."
        )
    ply_path = Path(ply_path).resolve()
    spz_path = Path(spz_path).resolve()
    if not ply_path.is_file():
        raise FileNotFoundError(f"Trained Gaussian PLY not found: {ply_path}")
    spz = _import_spz_binding()
    required = ("UnpackOptions", "PackOptions", "CoordinateSystem", "load_splat_from_ply", "save_spz")
    missing = [name for name in required if not hasattr(spz, name)]
    if missing:
        raise RuntimeError("Installed spz package lacks the official conversion API: " + ", ".join(missing))
    ply_count = int(len(read_ply_vertex_records(ply_path)))
    unpack = spz.UnpackOptions()
    unpack.to_coord = spz.CoordinateSystem.RUB
    pack = spz.PackOptions()
    pack.from_coord = spz.CoordinateSystem.RUB
    spz_path.parent.mkdir(parents=True, exist_ok=True)
    # The C++ binding accepts only ASCII filenames on Windows.  Always stage
    # through a short temporary path so a Chinese repository path cannot turn
    # into an apparently successful zero-point export.
    temp_dir, staged_ply, staged_spz = _stage_ascii_paths(ply_path, spz_path)
    try:
        cloud = spz.load_splat_from_ply(str(staged_ply), unpack)
        loaded_count = int(getattr(cloud, "num_points", 0))
        if loaded_count != ply_count:
            raise RuntimeError(
                f"Official SPZ PLY reader loaded {loaded_count} Gaussian(s), expected {ply_count}"
            )
        saved = spz.save_spz(cloud, pack, str(staged_spz))
        if saved is False or not staged_spz.is_file() or staged_spz.stat().st_size == 0:
            raise RuntimeError("Official SPZ converter returned without creating a valid file")
        roundtrip = spz.load_spz(str(staged_spz), unpack)
        roundtrip_count = int(getattr(roundtrip, "num_points", 0))
        if roundtrip_count != ply_count:
            raise RuntimeError(
                f"SPZ round-trip contains {roundtrip_count} Gaussian(s), expected {ply_count}"
            )
        shutil.copyfile(staged_spz, spz_path)
    finally:
        temp_dir.cleanup()
    try:
        binding_version = importlib.metadata.version("spz")
    except importlib.metadata.PackageNotFoundError:
        binding_version = None
    return {
        "status": "exported",
        "path": str(spz_path.resolve()),
        "bytes": spz_path.stat().st_size,
        "backend": "Niantic spz Python bindings",
        "binding_distribution": "spz",
        "binding_version": binding_version,
        "source_tag": SPZ_SOURCE_TAG,
        "source_commit": SPZ_SOURCE_COMMIT,
        "source_coordinate": coordinate,
        "packed_coordinate": "RUB",
        "axis_conversion": f"{coordinate}->RUB performed by official spz loader",
        "ply_gaussian_count": ply_count,
        "spz_gaussian_count": roundtrip_count,
        "roundtrip_count_matches": True,
        "roundtrip_io": "official binding load_spz through ASCII staging path (Windows Unicode-path workaround)",
        "sha256": hashlib.sha256(spz_path.read_bytes()).hexdigest(),
    }


def export_run_spz(run_directory: Path, *, source_coordinate: str = "RDF") -> dict[str, Any]:
    """Export an existing direct-gsplat run and update its lineage reports.

    This is intentionally a post-training operation: the original PLY and
    training metrics remain authoritative even if the optional SPZ conversion
    fails.  The function is used by the reproducible ``scene_gsplat_export.py``
    command and is safe for runs whose output directory contains non-ASCII
    characters.
    """
    run_directory = Path(run_directory).resolve()
    model_dir = run_directory / "model"
    ply_path = model_dir / "trained_gaussians.ply"
    if not ply_path.is_file():
        raise FileNotFoundError(f"Trained Gaussian PLY not found: {ply_path}")
    spz_path = model_dir / "trained_gaussians.spz"
    spz_report = export_official_spz(ply_path, spz_path, source_coordinate=source_coordinate)

    report_path = run_directory / "training_report.json"
    manifest_path = model_dir / "model_manifest.json"
    report = json.loads(report_path.read_text(encoding="utf-8-sig")) if report_path.is_file() else {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig")) if manifest_path.is_file() else {}
    report["model_spz"] = str(spz_path)
    report["spz_error"] = None
    report["spz_conversion"] = spz_report
    lineage = report.get("lineage")
    if isinstance(lineage, dict):
        outputs = lineage.setdefault("outputs", {})
        outputs["spz"] = spz_report
        lineage["model_spz"] = str(spz_path)
        lineage["spz_error"] = None
    manifest.setdefault("outputs", {})["spz"] = spz_report
    # Keep the detailed nested output while also mirroring the training-report
    # compatibility fields expected by downstream release tooling.
    manifest["model_spz"] = str(spz_path)
    manifest["spz_error"] = None
    write_json(report_path, report)
    write_json(manifest_path, manifest)
    return {
        "status": "exported",
        "run_directory": str(run_directory),
        "ply": str(ply_path),
        "spz": spz_report,
        "training_report": str(report_path),
        "model_manifest": str(manifest_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a direct-gsplat trained PLY with Niantic SPZ")
    parser.add_argument("run", type=Path, help="direct_gsplat_* run directory or trained_gaussians.ply")
    parser.add_argument("--source-coordinate", default="RDF", choices=("RDF", "rdf"))
    args = parser.parse_args()
    target = Path(args.run)
    run = target.parent.parent if target.is_file() else target
    try:
        result = export_run_spz(run, source_coordinate=args.source_coordinate)
    except Exception as error:
        payload = {"status": "blocked", "error": f"{type(error).__name__}: {error}", "run_directory": str(run.resolve())}
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def write_model_lineage(
    path: Path,
    *,
    dataset_root: Path,
    dataset_fingerprint: str,
    transforms_path: Path,
    checkpoint_path: Path,
    ply_report: dict[str, Any],
    spz_report: dict[str, Any] | None,
    training_config: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    manifest = {
        "schema": "femto-direct-gsplat-model-v1",
        "created_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "status": "trained",
        "backend": "direct nerfstudio-project/gsplat public API",
        "backend_license": "Apache-2.0",
        "graphdeco_dependency": False,
        "training_input": {
            "dataset_root": str(dataset_root.resolve()),
            "transforms": str(transforms_path.resolve()),
            "fingerprint_sha256": dataset_fingerprint,
            "semantic_mask_applied": False,
            "rgb": "rectified full-scene RGB",
            "depth": "aligned uint16 depth converted to metres and used in the loss",
        },
        "coordinates": {
            "stored_transforms_camera": "OpenGL/RUB camera-to-world",
            "training_camera": "Orbbec/OpenCV RDF; +Z forward",
            "training_world": "source T_WC world coordinates; metres",
            "camera_axis_conversion": "T_WC_RDF = T_WC_OpenGL @ diag(1,-1,-1,1)",
        },
        "checkpoint": str(checkpoint_path.resolve()),
        "training_config": training_config,
        "evaluation": metrics,
        "model_spz": spz_report.get("path") if spz_report else None,
        "spz_error": spz_report.get("error") if spz_report else None,
        "outputs": {"ply": ply_report, "spz": spz_report},
    }
    write_json(path, manifest)
    return manifest


if __name__ == "__main__":
    raise SystemExit(main())
