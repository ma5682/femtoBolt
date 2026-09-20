"""Stage B of the package migration: make every cross-package import point downward.

Run from the repository root:  python tools/_restructure_layers.py . [--dry]

Moves misplaced pure functions to the layer they belong to, extracts the
capture-session code out of the Tk application, relocates the orchestration
modules that lived in low-level packages, and rewrites every import / mock
target accordingly.  Symbol moves preserve source text exactly.
"""
from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
DRY = "--dry" in sys.argv
PENDING: dict[Path, str] = {}
MOVED_FILES: dict[Path, Path] = {}  # dry run only: destination -> original location


def log(*args: object) -> None:
    print(*args)


def real_path(path: Path) -> Path:
    """Where the current text of ``path`` lives during a dry run (files are not moved)."""
    return MOVED_FILES.get(path, path)


def read(path: Path) -> str:
    if path in PENDING:
        return PENDING[path]
    source = real_path(path) if DRY else path
    return source.read_bytes().decode("utf-8-sig")


def exists(path: Path) -> bool:
    return path in PENDING or (real_path(path) if DRY else path).exists()


def write(path: Path, text: str) -> None:
    if DRY:
        PENDING[path] = text
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))


def eol_of(text: str) -> str:
    return "\r\n" if text.count("\r\n") > text.count("\n") - text.count("\r\n") else "\n"


def git_mv(src: Path, dst: Path) -> None:
    log(f"  git mv {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")
    if DRY:
        if src in PENDING:
            PENDING[dst] = PENDING.pop(src)
        else:
            MOVED_FILES[dst] = src
        return
    subprocess.run(["git", "mv", str(src.relative_to(ROOT)), str(dst.relative_to(ROOT))], cwd=ROOT, check=True)


def all_python_files() -> list[Path]:
    files = list((ROOT / "femto").rglob("*.py")) + list((ROOT / "tests").glob("*.py")) + list((ROOT / "tools").glob("*.py"))
    files.append(ROOT / "commercial_compliance_check.py")
    known = set(files) | set(PENDING) | set(MOVED_FILES)
    originals = set(MOVED_FILES.values())
    return sorted(p for p in known if p.suffix == ".py" and not p.name.startswith("_") and p not in originals)


# ---------------------------------------------------------------------------
# symbol moves
# ---------------------------------------------------------------------------

def _line_offsets(text: str) -> list[int]:
    offsets = [0]
    for line in text.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


def extract_definitions(text: str, names: list[str]) -> tuple[str, str]:
    """Return (text without the named top-level definitions, the moved text)."""
    tree = ast.parse(text)
    offsets = _line_offsets(text)
    lines = text.splitlines(keepends=True)
    spans: list[tuple[int, int]] = []
    found: set[str] = set()
    for node in tree.body:
        target_names: list[str] = []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            target_names = [node.name]
        elif isinstance(node, ast.Assign):
            target_names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target_names = [node.target.id]
        if not any(n in names for n in target_names):
            continue
        found.update(n for n in target_names if n in names)
        start_line = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
        while start_line > 1 and lines[start_line - 2].lstrip().startswith("#"):
            start_line -= 1
        spans.append((start_line, node.end_lineno))
    missing = set(names) - found
    assert not missing, f"definitions not found: {sorted(missing)}"
    moved_parts = []
    keep = text
    for start_line, end_line in sorted(spans, reverse=True):
        start = offsets[start_line - 1]
        end = offsets[end_line]
        chunk = keep[start:end]
        tail = end
        while tail < len(keep) and keep[tail] in "\r\n":
            tail += 1
        moved_parts.insert(0, chunk.rstrip("\r\n"))
        keep = keep[:start] + keep[tail:]
    eol = eol_of(text)
    moved = (eol * 2).join(part.replace("\r\n", "\n").replace("\n", eol) for part in moved_parts) + eol
    keep = re.sub(r"(\r?\n){4,}", eol * 3, keep)
    return keep, moved


def ensure_import(text: str, line: str) -> str:
    if re.search(r"^" + re.escape(line) + r"[ \t]*\r?$", text, re.M):
        return text
    eol = eol_of(text)
    lines = text.splitlines(keepends=True)
    tree = ast.parse(text)
    last_import_end = 0
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            last_import_end = node.end_lineno
        elif last_import_end:
            break
        elif isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant):
            last_import_end = node.end_lineno  # module docstring
    lines.insert(last_import_end, line + eol)
    return "".join(lines)


def move_symbols(src: Path, dst: Path, names: list[str], *, header: str | None = None,
                 dst_imports: tuple[str, ...] = (), substitutions: tuple[tuple[str, str], ...] = (),
                 src_import: str | None = None) -> None:
    log(f"  move {names}: {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")
    text = read(src)
    remaining, moved = extract_definitions(text, names)
    for old, new in substitutions:
        assert old in moved, f"{src.relative_to(ROOT)}: substitution anchor missing: {old!r}"
        moved = moved.replace(old, new)
    if exists(dst):
        target = read(dst)
        eol = eol_of(target)
        for line in dst_imports:
            target = ensure_import(target, line)
        target = target.rstrip("\r\n") + eol * 3 + moved.replace("\r\n", "\n").replace("\n", eol)
    else:
        assert header is not None, f"{dst} does not exist and no header given"
        target = header.rstrip("\n") + "\n\n\n" + moved.replace("\r\n", "\n")
    write(dst, target)
    if src_import:
        remaining = ensure_import(remaining, src_import)
    write(src, remaining)


# ---------------------------------------------------------------------------
# import rewriting
# ---------------------------------------------------------------------------

def _abs(offsets: list[int], lineno: int, col: int, text_lines: list[str]) -> int:
    prefix = text_lines[lineno - 1].encode("utf-8")[:col].decode("utf-8", errors="replace")
    return offsets[lineno - 1] + len(prefix)


def split_symbol_imports(text: str, module: str, moved: dict[str, tuple[str, str]]) -> str:
    """Rewrite ``from module import a, b`` when some names moved to other modules.

    ``moved`` maps old imported name -> (new module, new name).  The local alias
    is preserved: ``from old import _x`` becomes ``from new import x as _x``.
    """
    tree = ast.parse(text)
    text_lines = text.splitlines(keepends=True)
    offsets = _line_offsets(text)
    edits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.module != module or node.level:
            continue
        if not any(alias.name in moved for alias in node.names):
            continue
        indent = text_lines[node.lineno - 1][: node.col_offset]
        keep = [a for a in node.names if a.name not in moved]
        groups: dict[str, list[str]] = {}
        for alias in node.names:
            if alias.name in moved:
                new_module, new_name = moved[alias.name]
                local = alias.asname or alias.name
                groups.setdefault(new_module, []).append(new_name if new_name == local else f"{new_name} as {local}")
        statements = []
        if keep:
            statements.append(f"from {module} import " + ", ".join(a.name + (f" as {a.asname}" if a.asname else "") for a in keep))
        for new_module, items in groups.items():
            statements.append(f"from {new_module} import " + ", ".join(items))
        eol = eol_of(text)
        replacement = (eol + indent).join(statements)
        start = _abs(offsets, node.lineno, node.col_offset, text_lines)
        end = _abs(offsets, node.end_lineno, node.end_col_offset, text_lines)
        edits.append((start, end, replacement))
    for start, end, replacement in sorted(edits, reverse=True):
        text = text[:start] + replacement + text[end:]
    return text


def apply_symbol_split(module: str, moved: dict[str, tuple[str, str]]) -> None:
    for path in all_python_files():
        text = read(path)
        updated = split_symbol_imports(text, module, moved)
        if updated != text:
            write(path, updated)


def rename_module_everywhere(old: str, new: str) -> None:
    pattern = re.compile(r"(?<![\w.])" + re.escape(old) + r"(?![\w])")
    for path in all_python_files():
        text = read(path)
        updated = pattern.sub(new, text)
        if updated != text:
            write(path, updated)
    log(f"  renamed module references {old} -> {new}")


def replace_in(path: Path, old: str, new: str, count: int = 1) -> None:
    text = read(path)
    pattern = re.escape(old).replace("\\\n", "\r?\n")
    matches = list(re.finditer(pattern, text))
    assert len(matches) == count, f"{path.relative_to(ROOT)}: expected {count} of {old[:70]!r}, found {len(matches)}"
    eol = eol_of(text)
    for match in reversed(matches):
        text = text[: match.start()] + new.replace("\n", eol) + text[match.end():]
    write(path, text)


# ---------------------------------------------------------------------------
# the plan
# ---------------------------------------------------------------------------

def F(rel: str) -> Path:
    return ROOT / rel


def stage_symbols() -> None:
    # A. K6 back-projection is a geometry primitive
    move_symbols(F("femto/capture/scene_segmented_capture.py"), F("femto/geometry/depth_cloud.py"), ["depth_cloud"],
                 header='"""Canonical K6 (BROWN_CONRADY_K6) back-projection of an aligned depth image to a metric point cloud."""\n'
                        'from __future__ import annotations\n\nfrom typing import Any\n\nimport cv2\nimport numpy as np\nimport open3d as o3d\n',
                 src_import="from femto.geometry.depth_cloud import depth_cloud")
    apply_symbol_split("femto.capture.scene_segmented_capture", {"depth_cloud": ("femto.geometry.depth_cloud", "depth_cloud")})

    # B. pose span is tracking policy math
    move_symbols(F("femto/pipeline/scene_to_nerfstudio.py"), F("femto/tracking/moving_quality.py"), ["pose_diversity"],
                 src_import="from femto.tracking.moving_quality import pose_diversity")
    apply_symbol_split("femto.pipeline.scene_to_nerfstudio", {"pose_diversity": ("femto.tracking.moving_quality", "pose_diversity")})

    # C. temporal confidence reading belongs with the depth layer
    move_symbols(F("femto/temporal/temporal_keyframe.py"), F("femto/depth/temporal_depth_layer.py"),
                 ["load_temporal_confidence", "prepared_depth"],
                 dst_imports=("from femto.depth.depth_quality import process_depth_quality", "from femto.paths import child_path"),
                 substitutions=(("_child(Path(dataset),", "child_path(Path(dataset),"),),
                 src_import="from femto.depth.temporal_depth_layer import load_temporal_confidence, prepared_depth")
    apply_symbol_split("femto.temporal.temporal_keyframe", {
        "load_temporal_confidence": ("femto.depth.temporal_depth_layer", "load_temporal_confidence"),
        "prepared_depth": ("femto.depth.temporal_depth_layer", "prepared_depth")})

    # E. local acquisition envelope is view-selection policy
    move_symbols(F("femto/capture/local_capture.py"), F("femto/tracking/local_frame_selection.py"),
                 ["LOCAL_RADIUS_M", "LOCAL_ROTATION_DEG", "local_workspace"],
                 src_import="from femto.tracking.local_frame_selection import LOCAL_RADIUS_M, LOCAL_ROTATION_DEG, local_workspace")
    apply_symbol_split("femto.capture.local_capture", {"local_workspace": ("femto.tracking.local_frame_selection", "local_workspace")})

    # F. calibration files are dataset schema
    move_symbols(F("femto/capture/scene_capture.py"), F("femto/dataset/calibration_files.py"), ["write_calibration"],
                 header='"""Write the calibration JSON set that every dataset carries (see local_reference.CALIBRATION_FILES)."""\n'
                        'from __future__ import annotations\n\nfrom pathlib import Path\nfrom typing import Any\n\n'
                        'from femto.io_utils import write_json as json_dump\n',
                 substitutions=(("camera: OrbbecAlignedCamera", "camera: Any"),),
                 src_import="from femto.dataset.calibration_files import write_calibration")
    apply_symbol_split("femto.capture.scene_capture", {"write_calibration": ("femto.dataset.calibration_files", "write_calibration")})

    # G. capture-session lifecycle out of the Tk application
    move_symbols(F("femto/app/femto_control_app.py"), F("femto/capture/scan_session.py"),
                 ["MOTION_CHECK_INTERVAL_S", "MOTION_CHECK_SCALE", "MOTION_FINAL_WAIT_S", "ALIGNED_FRAME_STARTUP_TIMEOUT_S",
                  "unique_timestamp_dir", "copy_frame", "save_rgbd_photo", "ActiveScan", "update_capture_motion_state",
                  "consume_capture_motion_future", "harvest_capture_motion", "wait_for_aligned_frame", "save_scan_frame",
                  "finalize_scan"],
                 header='"""Capture-session lifecycle shared by the Tk application and the local capture worker.\n\n'
                        'Frame and photo writers, the ``ActiveScan`` record and the sticky camera-motion\n'
                        'hint state live here so that ``femto.capture`` never imports the application.\n"""\n'
                        'from __future__ import annotations\n\n'
                        'from concurrent.futures import Future, TimeoutError as FutureTimeoutError\n'
                        'from dataclasses import dataclass, field\nimport importlib.metadata\nfrom pathlib import Path\n'
                        'import time\nfrom typing import Any, Callable\n\nimport cv2\nimport numpy as np\n\n'
                        'from femto.dataset.calibration_files import write_calibration\n'
                        'from femto.geometry.depth_cloud import depth_cloud\n'
                        'from femto.geometry.fusion import write_point_cloud_safe\n'
                        'from femto.hardware.orbbec_rgbd import AlignedFrame, OrbbecAlignedCamera\n'
                        'from femto.io_utils import append_jsonl, imwrite_unicode, write_json as json_dump\n'
                        'from femto.capture.scene_capture import depth_preview\n'
                        'from femto.paths import PHOTOS_ROOT\n',
                 src_import="from femto.capture.scan_session import (ActiveScan, ALIGNED_FRAME_STARTUP_TIMEOUT_S, MOTION_CHECK_INTERVAL_S, "
                            "MOTION_CHECK_SCALE, MOTION_FINAL_WAIT_S, consume_capture_motion_future, copy_frame, finalize_scan, "
                            "harvest_capture_motion, save_rgbd_photo, save_scan_frame, unique_timestamp_dir, "
                            "update_capture_motion_state, wait_for_aligned_frame)")
    for name in ("ActiveScan", "copy_frame"):
        replace_in(F("femto/capture/local_capture.py"), f"from femto.app.femto_control_app import {name}",
                   f"from femto.capture.scan_session import {name}")
    replace_in(F("femto/capture/local_capture.py"), "from femto.app.femto_control_app import save_scan_frame, finalize_scan",
               "from femto.capture.scan_session import save_scan_frame, finalize_scan")
    for test, old, new in (
        ("tests/test_local_resilience.py", "mock.patch('femto.app.femto_control_app.save_scan_frame'", "mock.patch('femto.capture.scan_session.save_scan_frame'"),
        ("tests/test_local_tracking_recovery.py", "mock.patch('femto.app.femto_control_app.save_scan_frame'", "mock.patch('femto.capture.scan_session.save_scan_frame'"),
        ("tests/test_local_offline.py", "mock.patch('femto.app.femto_control_app.finalize_scan')", "mock.patch('femto.capture.scan_session.finalize_scan')"),
    ):
        replace_in(F(test), old, new, read(F(test)).count(old))

    # H. exclusion polygon validation is part of the annotation contract
    move_symbols(F("femto/planning/region_exclusions.py"), F("femto/annotation/scene_annotations.py"), ["validate_exclusions"],
                 dst_imports=("import cv2",), src_import="from femto.annotation.scene_annotations import validate_exclusions")
    replace_in(F("femto/annotation/scene_annotations.py"), "        from femto.planning.region_exclusions import validate_exclusions\n", "")

    # I. single-frame source resolution is dataset schema
    move_symbols(F("femto/gsplat/scene_single_gs.py"), F("femto/dataset/single_frame_source.py"), ["resolve_source"],
                 header='"""Resolve a fixed-keyframe model or RGB-D photo into its single source frame (schema checks only)."""\n'
                        'from __future__ import annotations\n\nimport hashlib\nimport json\nfrom pathlib import Path\n\n'
                        'from femto.io_utils import load_json as read_json\n',
                 src_import="from femto.dataset.single_frame_source import resolve_source")
    apply_symbol_split("femto.gsplat.scene_single_gs", {"resolve_source": ("femto.dataset.single_frame_source", "resolve_source")})
    replace_in(F("tests/test_reconstruction_sources.py"), "@patch('femto.gsplat.scene_single_gs.resolve_source')",
               "@patch('femto.dataset.single_frame_source.resolve_source')", 2)

    # J. RGB-D frame primitives out of the TSDF orchestrator (which then moves to pipeline)
    move_symbols(F("femto/geometry/scene_reconstruction.py"), F("femto/geometry/rgbd_frames.py"),
                 ["_distortion_maps", "_prepare_rgbd", "_load_poses"],
                 header='"""Calibrated RGB-D frame primitives: K6 undistortion maps, Open3D RGBD preparation, pose loading."""\n'
                        'from __future__ import annotations\n\nfrom pathlib import Path\nfrom typing import Any\n\n'
                        'import cv2\nimport numpy as np\nimport open3d as o3d\n\n'
                        'from femto.dataset.scene_dataset_validator import validate_pose\nfrom femto.io_utils import load_json\n',
                 substitutions=(("def _distortion_maps(", "def distortion_maps("), ("def _prepare_rgbd(", "def prepare_rgbd("),
                                ("def _load_poses(", "def load_poses(")),
                 src_import="from femto.geometry.rgbd_frames import distortion_maps as _distortion_maps, load_poses as _load_poses, prepare_rgbd as _prepare_rgbd")
    apply_symbol_split("femto.geometry.scene_reconstruction", {
        "_distortion_maps": ("femto.geometry.rgbd_frames", "distortion_maps"),
        "_prepare_rgbd": ("femto.geometry.rgbd_frames", "prepare_rgbd"),
        "_load_poses": ("femto.geometry.rgbd_frames", "load_poses")})
    replace_in(F("tests/test_capture_diagnostics.py"), "mock.patch('femto.geometry.scene_reconstruction._distortion_maps'",
               "mock.patch('femto.geometry.rgbd_frames.distortion_maps'")


MODULE_MOVES = [
    ("femto/geometry/scene_reconstruction.py", "femto/pipeline/scene_reconstruction.py"),
    ("femto/geometry/local_geometry.py", "femto/local/local_geometry.py"),
    ("femto/local/local_comparison.py", "femto/pipeline/local_comparison.py"),
    ("femto/local/local_comparison_viewer.py", "femto/pipeline/local_comparison_viewer.py"),
    ("femto/local/local_parameter_study.py", "femto/pipeline/local_parameter_study.py"),
    ("femto/pipeline/local_processing.py", "femto/local/local_processing.py"),
    ("femto/annotation/local_history.py", "femto/pipeline/local_history.py"),
]

FINGERPRINT_OLD = ("            'femto/local/local_comparison.py', 'femto/local/local_comparison_viewer.py',\n"
                   "            'femto/geometry/scene_reconstruction.py', 'femto/local/local_pose_refinement.py',\n"
                   "            'femto/local/local_temporal_support.py', 'femto/geometry/local_geometry.py',\n"
                   "            'femto/dataset/local_reference.py', 'femto/dataset/scene_dataset_validator.py',\n"
                   "            'femto/annotation/fixed_rgb_geometry.py', 'femto/tracking/local_stability.py',\n"
                   "            'femto/io_utils.py')))")
FINGERPRINT_NEW = ("            'femto/pipeline/local_comparison.py', 'femto/pipeline/local_comparison_viewer.py',\n"
                   "            'femto/pipeline/scene_reconstruction.py', 'femto/geometry/rgbd_frames.py',\n"
                   "            'femto/local/local_pose_refinement.py', 'femto/local/local_temporal_support.py',\n"
                   "            'femto/local/local_geometry.py', 'femto/dataset/local_reference.py',\n"
                   "            'femto/dataset/scene_dataset_validator.py', 'femto/annotation/fixed_rgb_geometry.py',\n"
                   "            'femto/tracking/local_stability.py', 'femto/io_utils.py')))")
STUDY_OLD = ("    'femto/local/local_comparison.py', 'femto/local/local_comparison_viewer.py',\n"
             "    'femto/geometry/scene_reconstruction.py', 'femto/local/local_pose_refinement.py',\n"
             "    'femto/local/local_temporal_support.py', 'femto/geometry/local_geometry.py',\n"
             "    'femto/dataset/local_reference.py', 'femto/dataset/scene_dataset_validator.py',\n"
             "    'femto/annotation/fixed_rgb_geometry.py', 'femto/tracking/local_stability.py',\n"
             "    'femto/io_utils.py',\n)")
STUDY_NEW = ("    'femto/pipeline/local_comparison.py', 'femto/pipeline/local_comparison_viewer.py',\n"
             "    'femto/pipeline/scene_reconstruction.py', 'femto/geometry/rgbd_frames.py',\n"
             "    'femto/local/local_pose_refinement.py', 'femto/local/local_temporal_support.py',\n"
             "    'femto/local/local_geometry.py', 'femto/dataset/local_reference.py',\n"
             "    'femto/dataset/scene_dataset_validator.py', 'femto/annotation/fixed_rgb_geometry.py',\n"
             "    'femto/tracking/local_stability.py', 'femto/io_utils.py',\n)")


def stage_modules() -> None:
    for src, dst in MODULE_MOVES:
        git_mv(F(src), F(dst))
    for src, dst in MODULE_MOVES:
        rename_module_everywhere(src[:-3].replace("/", "."), dst[:-3].replace("/", "."))
    replace_in(F("femto/pipeline/local_comparison.py"), FINGERPRINT_OLD, FINGERPRINT_NEW)
    replace_in(F("femto/pipeline/local_parameter_study.py"), STUDY_OLD, STUDY_NEW)


def main() -> None:
    log("== symbols")
    stage_symbols()
    log("== modules")
    stage_modules()
    log("done" + (" (dry run)" if DRY else ""))


if __name__ == "__main__":
    main()
