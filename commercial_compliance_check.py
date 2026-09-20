"""Mechanical release checks for the commercial clean-room codebase.

This script catches packaging mistakes. It does not replace legal review.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any
import zipfile

from scene_gsplat_metrics import lpips_backbone_status


ROOT = Path(__file__).resolve().parent
# Virtual environments and data trees are not product source. `.venv-gsplat` must be
# excluded explicitly: gsplat ships an optional Graphdeco compatibility wrapper whose
# import would otherwise be reported as a product-tree violation.
EXCLUDED_SCAN_DIRS = frozenset(
    {".venv", ".venv-3dgs", ".venv-gsplat", "outputs", "datasets", "references", "__pycache__"}
)
EXPECTED_PACKAGES = {
    "pyorbbecsdk2": "2.1.2",
    "mediapipe": "1.0.1",
    "numpy": "2.2.6",
    "open3d": "0.19.0",
    "opencv-contrib-python": "5.0.0.93",
    "pillow": "12.3.0",
    "trimesh": "5.1.0",
}
REQUIRED_LICENSES = (
    "orbbec/OrbbecSDK_v2_LICENSE.txt",
    "orbbec/OrbbecSDK_v2_EULA.txt",
    "pyorbbecsdk2/LICENSE",
    "mediapipe/LICENSE",
    "mediapipe/Model_Card_BlazePose_GHUM_3D.pdf",
    "open3d/LICENSE",
    "trimesh/LICENSE.md",
    "numpy/LICENSE.txt",
    "opencv-contrib-python/LICENSE.txt",
    "opencv-contrib-python/LICENSE-3RD-PARTY.txt",
    "pillow/LICENSE",
    "gsplat/LICENSE",
    "pymeshlab/LICENSE",
    "torch/LICENSE",
    "torch/NOTICE",
    "torchvision/LICENSE",
    "torchmetrics/LICENSE",
    "lightning-utilities/LICENSE",
    "imageio/LICENSE",
    "tqdm/LICENCE",
    "lpips/LICENSE",
    "spz/LICENSE",
    "spz/ADOBE_NOTICE.txt",
    "zlib/LICENSE.txt",
    "zstd/LICENSE",
    "zstd/LEGACY_BSD2_NOTICES.txt",
    "divsufsort/LICENSE",
    "nanobind/LICENSE",
    "robin-map/LICENSE",
)
EXPECTED_MODEL_SHA256 = "5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1"
BLOCKED_IMPORT_PREFIXES = (
    "diff_gaussian_rasterization",
    "simple_knn",
    "pymeshlab",
)
# Direct product 3DGS runtime (.venv-gsplat). Versions mirror requirements-gsplat-lock.txt
# plus the separately installed official cu128 wheels.
GSPLAT_RUNTIME_SITE_PACKAGES = ROOT / ".venv-gsplat" / "Lib" / "site-packages"
EXPECTED_GSPLAT_PACKAGES = {
    "gsplat": "1.5.3",
    "torch": "2.7.1+cu128",
    "torchvision": "0.22.1+cu128",
    "torchmetrics": "1.9.0",
    "lightning-utilities": "0.15.3",
    "imageio": "2.37.0",
    "tqdm": "4.67.1",
    "numpy": "2.2.6",
    "pillow": "11.3.0",
    # Niantic official source tag v3.0.0 builds a distribution whose CMake
    # project/version is 1.1.0; lock the actual installed distribution value.
    "spz": "1.1.0",
}
# Build inputs that are compiled into the SPZ extension and therefore do not
# appear as installed runtime distributions.  The nanobind wheel contains the
# pinned robin-map headers referenced by the nanobind v3.0.1 submodule.
SPZ_STATIC_COMPONENTS = {
    "spz-source": {
        "version": "3.0.0",
        "tag": "v3.0.0",
        "commit": "5bf2945de1a003cee07133b1e495fe9c6ffdc7e7",
        "license": "MIT",
        "source_archive": "C:/FemtoSpzDeps/spz-v3.0.0.zip",
        "source_archive_sha256": "7f0f08ab144ecc940475b4edbfbb967d504aa2c78a1fbed17fd857bef8d95665",
    },
    "zlib": {
        "version": "1.3.1",
        "license": "zlib License",
        "source_archive": "C:/FemtoSpzDeps/zlib-1.3.1.tar.gz",
        "source_archive_sha256": "9a93b2b7dfdac77ceba5a558a580e74667dd6fede4585b91eefb60f03b72df23",
        "runtime_artifact": "C:/FemtoGsplatRuntime/Lib/site-packages/spz/zlib.dll",
        "runtime_artifact_sha256": "2f08f389724c8a05c672eeb3dec1ac1ff7f589d133d4011d807e489bee522baa",
        "linkage": "dynamic runtime DLL built from the pinned source archive",
    },
    "zstd": {
        "version": "1.5.6",
        "tag": "v1.5.6",
        "license": "BSD-3-Clause",
        "source_archive": "C:/FemtoSpzDeps/zstd-1.5.6.tar.gz",
        "source_archive_sha256": "8c29e06cf42aacc1eafc4077ae2ec6c6fcb96a626157e0593d5e82a34fd403c1",
        "linkage": "static library compiled into the SPZ extension",
    },
    "divsufsort-lite": {
        "version": "bundled snapshot in zstd 1.5.6",
        "copyright": "Copyright (c) 2003-2008 Yuta Mori",
        "license": "MIT",
        "source_archive": "C:/FemtoSpzDeps/zstd-1.5.6.tar.gz",
        "covered_by_component": "zstd",
        "linkage": "compiled as part of the pinned zstd static-library build",
    },
    "zstd-legacy-code": {
        "version": "bundled in zstd 1.5.6",
        "copyright": "Copyright (c) 2013-2016 Yann Collet",
        "license": "BSD-2-Clause",
        "source_archive": "C:/FemtoSpzDeps/zstd-1.5.6.tar.gz",
        "covered_by_component": "zstd",
        "linkage": "legacy decoder sources are compiled into the pinned zstd static-library build",
    },
    "nanobind": {
        "version": "3.0.1",
        "tag": "v3.0.1",
        "commit": "db4827f06f6f1680e5d4004c95fc8d69299dba8b",
        "license": "BSD-3-Clause",
        "build_artifact": "C:/FemtoSpzDeps/nanobind-3.0.1-py3-none-any.whl",
        "build_artifact_sha256": "4b49491fe8bf483a5e0342125b8fe56237e490ba642c7e296170b8cd256a7764",
    },
    "robin-map": {
        "version": "1.4.0",
        "tag": "v1.4.0",
        "commit": "4ec1bf19c6a96125ea22062f38c2cf5b958e448e",
        "license": "MIT",
        "delivery": "headers vendored in the pinned nanobind 3.0.1 wheel",
    },
}
SPZ_BUILD_CONSTRAINT = ROOT / "spz-build-constraints.txt"
EXPECTED_SPZ_EXTENSION_SHA256 = "0b9dfdfbc8d53f882d861b12e7ad014efefd8d932ae62a6c3eab8cad41e7a59e"
# Distributions that must never be installed in the product training runtime.
BLOCKED_RUNTIME_DISTRIBUTIONS = ("diff-gaussian-rasterization", "simple-knn", "pymeshlab", "nerfstudio")


def _normalize_distribution_name(name: str) -> str:
    return name.lower().replace("_", "-")


def gsplat_runtime_versions() -> dict[str, str] | None:
    if not GSPLAT_RUNTIME_SITE_PACKAGES.is_dir():
        return None
    versions: dict[str, str] = {}
    for distribution in importlib.metadata.distributions(path=[str(GSPLAT_RUNTIME_SITE_PACKAGES)]):
        name = distribution.metadata["Name"]
        if name:
            versions[_normalize_distribution_name(name)] = distribution.version
    return versions


def issue(target: list[dict[str, Any]], code: str, message: str, **context: Any) -> None:
    target.append({"code": code, "message": message, **context})


def python_files() -> list[Path]:
    return [
        path
        for path in ROOT.rglob("*.py")
        if not any(part in EXCLUDED_SCAN_DIRS for part in path.relative_to(ROOT).parts)
    ]


def imported_modules(path: Path) -> list[str]:
    # Match Python's own source decoder, including BOM and encoding cookies.
    tree = ast.parse(path.read_bytes(), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    return modules


def main() -> int:
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    package_versions: dict[str, str | None] = {}

    for package, expected in EXPECTED_PACKAGES.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        package_versions[package] = actual
        if actual != expected:
            issue(
                errors,
                "package_version_mismatch",
                f"{package} must be {expected}, found {actual}",
                package=package,
                expected=expected,
                actual=actual,
            )

    license_hashes: dict[str, str] = {}
    for relative in REQUIRED_LICENSES:
        path = ROOT / "licenses" / relative
        if not path.is_file():
            issue(errors, "license_missing", f"Missing licenses/{relative}", file=relative)
            continue
        license_hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()

    model = ROOT / "models" / "pose_landmarker_full.task"
    if not model.is_file():
        issue(errors, "model_missing", "MediaPipe Pose model is missing")
        model_sha256 = None
    else:
        model_sha256 = hashlib.sha256(model.read_bytes()).hexdigest()
        if model_sha256 != EXPECTED_MODEL_SHA256:
            issue(
                errors,
                "model_hash_mismatch",
                "MediaPipe Pose model hash differs from the audited asset",
                expected=EXPECTED_MODEL_SHA256,
                actual=model_sha256,
            )

    blocked_checkout = ROOT / "references" / "graphdeco-gaussian-splatting"
    if blocked_checkout.exists():
        issue(errors, "restricted_source_present", "Graphdeco source checkout must not enter the product tree")

    runtime_versions = gsplat_runtime_versions()
    if runtime_versions is None:
        issue(warnings, "gsplat_runtime_missing", "Direct gsplat runtime (.venv-gsplat) is not installed on this machine")
    else:
        for package, expected in EXPECTED_GSPLAT_PACKAGES.items():
            actual = runtime_versions.get(package)
            if actual != expected:
                issue(
                    errors,
                    "gsplat_runtime_version_mismatch",
                    f".venv-gsplat {package} must be {expected}, found {actual}",
                    package=package,
                    expected=expected,
                    actual=actual,
                )
        for blocked in BLOCKED_RUNTIME_DISTRIBUTIONS:
            if blocked in runtime_versions:
                issue(
                    errors,
                    "restricted_runtime_package",
                    f"Restricted distribution installed in the product 3DGS runtime: {blocked} {runtime_versions[blocked]}",
                    package=blocked,
                )

    spz_static_components = json.loads(json.dumps(SPZ_STATIC_COMPONENTS))
    for component_name in ("spz-source", "zlib", "zstd"):
        component = spz_static_components[component_name]
        source_archive = Path(component["source_archive"])
        if source_archive.is_file():
            actual_sha256 = hashlib.sha256(source_archive.read_bytes()).hexdigest()
            component["actual_source_archive_sha256"] = actual_sha256
            component["source_archive_sha256_verified"] = actual_sha256 == component["source_archive_sha256"]
            if not component["source_archive_sha256_verified"]:
                issue(
                    errors,
                    "spz_static_component_hash_mismatch",
                    f"The cached {component_name} archive differs from the audited SPZ build input",
                    path=str(source_archive),
                    expected=component["source_archive_sha256"],
                    actual=actual_sha256,
                )
        else:
            component["actual_source_archive_sha256"] = None
            component["source_archive_sha256_verified"] = False
            issue(
                warnings,
                "spz_static_component_cache_missing",
                f"The audited {component_name} source archive is absent; installed SPZ can run, but an offline rebuild is not reproducible",
                path=str(source_archive),
            )

    zlib_component = spz_static_components["zlib"]
    zlib_runtime = Path(zlib_component["runtime_artifact"])
    if zlib_runtime.is_file():
        actual_zlib_sha256 = hashlib.sha256(zlib_runtime.read_bytes()).hexdigest()
        zlib_component["actual_runtime_artifact_sha256"] = actual_zlib_sha256
        zlib_component["runtime_artifact_sha256_verified"] = (
            actual_zlib_sha256 == zlib_component["runtime_artifact_sha256"]
        )
        if not zlib_component["runtime_artifact_sha256_verified"]:
            issue(
                errors,
                "spz_runtime_component_hash_mismatch",
                "The SPZ runtime zlib DLL differs from the audited zlib 1.3.1 build output",
                path=str(zlib_runtime),
                expected=zlib_component["runtime_artifact_sha256"],
                actual=actual_zlib_sha256,
            )
    else:
        zlib_component["actual_runtime_artifact_sha256"] = None
        zlib_component["runtime_artifact_sha256_verified"] = False
        issue(errors, "spz_runtime_component_missing", "The pinned SPZ runtime zlib DLL is missing", path=str(zlib_runtime))

    spz_static_components["divsufsort-lite"]["covered_source_archive_sha256_verified"] = (
        spz_static_components["zstd"]["source_archive_sha256_verified"]
    )
    spz_static_components["zstd-legacy-code"]["covered_source_archive_sha256_verified"] = (
        spz_static_components["zstd"]["source_archive_sha256_verified"]
    )
    expected_constraint = "nanobind==3.0.1"
    if not SPZ_BUILD_CONSTRAINT.is_file():
        issue(errors, "spz_build_constraint_missing", "spz-build-constraints.txt is missing")
        constraint_entries: list[str] = []
    else:
        constraint_entries = [
            line.strip()
            for line in SPZ_BUILD_CONSTRAINT.read_text(encoding="utf-8-sig").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if constraint_entries != [expected_constraint]:
            issue(
                errors,
                "spz_build_constraint_mismatch",
                f"SPZ build constraint must contain only {expected_constraint}",
                actual=constraint_entries,
            )
    spz_static_components["nanobind"]["constraint_verified"] = constraint_entries == [expected_constraint]
    nanobind_artifact = Path(SPZ_STATIC_COMPONENTS["nanobind"]["build_artifact"])
    if nanobind_artifact.is_file():
        nanobind_artifact_sha256 = hashlib.sha256(nanobind_artifact.read_bytes()).hexdigest()
        spz_static_components["nanobind"]["actual_build_artifact_sha256"] = nanobind_artifact_sha256
        spz_static_components["nanobind"]["build_artifact_sha256_verified"] = (
            nanobind_artifact_sha256 == SPZ_STATIC_COMPONENTS["nanobind"]["build_artifact_sha256"]
        )
        if not spz_static_components["nanobind"]["build_artifact_sha256_verified"]:
            issue(
                errors,
                "spz_static_component_hash_mismatch",
                "The cached nanobind wheel differs from the audited SPZ build input",
                path=str(nanobind_artifact),
                expected=SPZ_STATIC_COMPONENTS["nanobind"]["build_artifact_sha256"],
                actual=nanobind_artifact_sha256,
            )
        required_robin_headers = {
            "nanobind/ext/robin_map/include/tsl/robin_growth_policy.h",
            "nanobind/ext/robin_map/include/tsl/robin_hash.h",
            "nanobind/ext/robin_map/include/tsl/robin_map.h",
        }
        try:
            with zipfile.ZipFile(nanobind_artifact) as wheel:
                missing_robin_headers = sorted(required_robin_headers.difference(wheel.namelist()))
        except (OSError, zipfile.BadZipFile) as error:
            missing_robin_headers = sorted(required_robin_headers)
            issue(
                errors,
                "spz_static_component_wheel_invalid",
                f"Cannot inspect the audited nanobind wheel: {error}",
                path=str(nanobind_artifact),
            )
        spz_static_components["robin-map"]["headers_verified_in_nanobind_wheel"] = not missing_robin_headers
        if missing_robin_headers:
            issue(
                errors,
                "spz_static_component_missing",
                "The audited nanobind wheel does not contain the expected robin-map headers",
                missing=missing_robin_headers,
            )
    else:
        spz_static_components["nanobind"]["actual_build_artifact_sha256"] = None
        spz_static_components["nanobind"]["build_artifact_sha256_verified"] = False
        spz_static_components["robin-map"]["headers_verified_in_nanobind_wheel"] = False
        issue(
            warnings,
            "spz_static_component_cache_missing",
            "The audited nanobind wheel cache is absent; installed SPZ can run, but an offline rebuild is not reproducible",
            path=str(nanobind_artifact),
        )

    spz_extension = GSPLAT_RUNTIME_SITE_PACKAGES / "spz" / "spz.cp310-win_amd64.pyd"
    spz_extension_sha256 = hashlib.sha256(spz_extension.read_bytes()).hexdigest() if spz_extension.is_file() else None
    spz_extension_sha256_verified = spz_extension_sha256 == EXPECTED_SPZ_EXTENSION_SHA256
    if runtime_versions is not None:
        if not spz_extension.is_file():
            issue(errors, "spz_extension_missing", "The pinned SPZ Python extension is missing", path=str(spz_extension))
        elif not spz_extension_sha256_verified:
            issue(
                errors,
                "spz_extension_hash_mismatch",
                "The installed SPZ Python extension differs from the audited build artifact",
                path=str(spz_extension),
                expected=EXPECTED_SPZ_EXTENSION_SHA256,
                actual=spz_extension_sha256,
            )

    lpips_backbone = lpips_backbone_status()
    if not lpips_backbone["present"]:
        issue(warnings, "lpips_backbone_missing", "LPIPS AlexNet backbone is not provisioned; run install_lpips_weights.cmd")
    elif not lpips_backbone["ready"]:
        issue(errors, "lpips_backbone_hash_mismatch", lpips_backbone["error"], path=lpips_backbone["path"])

    for path in python_files():
        try:
            modules = imported_modules(path)
        except (SyntaxError, UnicodeDecodeError) as error:
            issue(errors, "source_parse_failed", str(error), file=str(path.relative_to(ROOT)))
            continue
        for module in modules:
            if module.startswith(BLOCKED_IMPORT_PREFIXES):
                issue(
                    errors,
                    "restricted_import",
                    f"Restricted product dependency import: {module}",
                    file=str(path.relative_to(ROOT)),
                )

    manual_release_blockers = [
        "Register the actual copyright owner and contributor assignments.",
        "Include license bundle and notices in the real installer/customer delivery.",
        "Flow down the Orbbec Extension EULA and satisfy its LGPL third-party obligations.",
        "Do not redistribute the development Nerfstudio environment; it pulls GPL-3.0 PyMeshLab.",
        "Complete privacy impact assessment and separate consent for human RGB/body/health data.",
        "Complete medical-device classification before making diagnostic/treatment/control claims.",
        "Run final legal, patent, trademark, security and SBOM review on the exact release artifact.",
    ]
    for blocker in manual_release_blockers:
        issue(warnings, "manual_release_gate", blocker)

    report = {
        "schema": "commercial-clean-room-check-v1",
        "development_baseline_passed": not errors,
        "release_ready": False,
        "package_versions": package_versions,
        "gsplat_runtime_versions": (
            {name: runtime_versions.get(name) for name in EXPECTED_GSPLAT_PACKAGES} if runtime_versions else None
        ),
        "spz_static_components": spz_static_components,
        "spz_extension": {
            "file": str(spz_extension),
            "present": spz_extension.is_file(),
            "sha256": spz_extension_sha256,
            "expected_sha256": EXPECTED_SPZ_EXTENSION_SHA256,
            "sha256_verified": spz_extension_sha256_verified,
        },
        "lpips_backbone": {
            "file": lpips_backbone["file"],
            "present": lpips_backbone["present"],
            "sha256_verified": lpips_backbone["sha256_verified"],
        },
        "model_sha256": model_sha256,
        "license_sha256": license_hashes,
        "errors": errors,
        "warnings": warnings,
    }
    report_dir = ROOT / "compliance"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "current_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report_path"] = str(report_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 2 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
