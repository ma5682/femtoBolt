"""One-shot migration of the flat repository into the ``femto`` package tree.

Run from the repository root:  python tools/_migrate_packages.py . [--dry]

Stage A only: moves files with ``git mv``, rewrites every ``import`` /
``from ... import`` and ``mock.patch("module.attr")`` target, replaces the
per-module ``ROOT = Path(__file__)...`` with ``femto.paths``, converts every
script-path subprocess launch and ``.cmd`` entry point to ``python -m``.
Function-level cycle breaking is a separate stage.
"""
from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
DRY = "--dry" in sys.argv

PLAN: dict[str, list[str]] = {
    "femto": ["io_utils", "ui_i18n"],
    "femto.depth": ["depth_quality", "depth_error_check", "temporal_depth_fusion", "temporal_depth_layer"],
    "femto.dataset": ["scene_dataset_validator", "scene_data_kind", "local_reference", "scene_gsplat_split"],
    "femto.geometry": ["pose_geometry", "fusion", "single_frame_mesh", "single_frame_quality", "multiview_texture",
                       "scene_gaussian", "depth_confidence_cloud", "local_geometry", "scene_reconstruction"],
    "femto.hardware": ["orbbec_rgbd"],
    "femto.reports": ["scene_quality_report", "scene_scale_validation", "research_validation"],
    "femto.tracking": ["moving_quality", "scene_pose_estimator", "scene_live_tracker", "local_tracking_background",
                       "local_tracking_diagnostics", "local_tracking_replay", "local_frame_selection", "local_stability",
                       "local_stability_feedback"],
    "femto.temporal": ["scene_temporal_fusion", "temporal_keyframe", "scene_static_fusion", "scene_keyframe"],
    "femto.local": ["local_pose_refinement", "local_temporal_support", "local_comparison", "local_comparison_viewer",
                    "local_parameter_study", "local_geometry_diagnostics"],
    "femto.capture": ["scene_capture", "local_raw_writer", "local_live_preview", "local_capture", "scene_segmented_capture",
                      "top_scan"],
    "femto.pipeline": ["synthetic_scene_demo", "scene_to_nerfstudio", "complete_demo", "local_processing", "local_offline",
                       "local_interrupted", "reconstruction_sources", "depth_quality_replay"],
    "femto.gsplat": ["scene_gsplat_data", "scene_gsplat_guard", "scene_gsplat_metrics", "scene_gsplat_export",
                     "scene_gsplat_trainer", "scene_gsplat_viewer", "scene_gsplat_acceptance", "gsplat_backend",
                     "scene_single_gs", "scene_single_gs_viewer", "scene_3dgs"],
    "femto.annotation": ["scene_annotations", "scene_annotation_ui", "scene_surface_picker", "fixed_rgb_geometry",
                         "measurement_audit", "region_texture", "square_brush", "local_history"],
    "femto.planning": ["region_coverage", "region_exclusions", "region_path", "region_path_glb", "region_path_guard",
                       "region_project", "region_library", "metric_region_path", "serpentine_path", "route_motion",
                       "route_optimization", "route_quality", "path_presentation", "planning_drafts", "planning_io",
                       "planning_state", "planning_workspace", "coverage_inspection", "coverage_inspector",
                       "surface_spacing", "surface_spacing_report", "fixed_rgb_viewer", "export_history"],
    "femto.viewer": ["scene_demo_viewer", "live_point_cloud", "view_latest", "segmented_cloud_viewer", "depth_debug_viewer",
                     "export_preview"],
    "femto.app": ["femto_control_app", "app_local_panel", "app_local_history", "app_local_comparison", "app_temporal",
                  "app_temporal_panel"],
    "tools": ["download_model", "download_verified_file", "patch_gsplat_windows"],
}
PACKAGE_DOCS = {
    "femto": "Femto Bolt RGB-D capture, reconstruction, measurement and planning product code.",
    "femto.depth": "Per-frame depth quality, hole filling, confidence and temporal depth fusion primitives.",
    "femto.dataset": "Raw dataset schema, validation, provenance classification and split rules.",
    "femto.geometry": "Back-projection, TSDF/mesh fusion, texture and single-frame geometry.",
    "femto.hardware": "Orbbec Femto Bolt SDK adapter (the only pyorbbecsdk wrapper).",
    "femto.reports": "Research-grade quality, scale and measurement reports over results.",
    "femto.tracking": "RGB-D odometry, live tracking, motion checks and frame selection policies.",
    "femto.temporal": "Fixed-camera temporal bursts and keyframe derivation.",
    "femto.local": "Offline refinement stages for local supplementary capture sessions.",
    "femto.capture": "Dataset writers and capture workers on top of the hardware adapter.",
    "femto.pipeline": "End-to-end orchestration: validate, pose, reconstruct, convert, publish.",
    "femto.gsplat": "Product 3DGS training/export/viewing through the Apache-2.0 gsplat public API.",
    "femto.annotation": "Annotation data model, fixed-RGB geometry and measurement audit.",
    "femto.planning": "Region planning workbench: regions, exclusions, routes, coverage, projects.",
    "femto.viewer": "Open3D / Tk viewers launched as separate processes.",
    "femto.app": "Tk control panel and its panels.",
}
MAP = {m: pkg for pkg, ms in PLAN.items() for m in ms if pkg.startswith("femto")}
TOOLS = PLAN["tools"]
KEYS = sorted(MAP, key=len, reverse=True)
ALT = "|".join(re.escape(k) for k in KEYS)

log = print


def run(*cmd: str) -> None:
    log("  $", " ".join(cmd))
    if not DRY:
        subprocess.run(cmd, cwd=ROOT, check=True)


def source_path(path: Path) -> Path:
    """In a dry run nothing has moved yet: fall back to the flat location."""
    if DRY and not path.exists() and (path.is_relative_to(ROOT / "femto") or path.is_relative_to(ROOT / "tools")):
        return ROOT / path.name
    return path


PENDING: dict[Path, str] = {}  # dry-run only: text as it would be after earlier stages


def read(path: Path) -> str:
    if path in PENDING:
        return PENDING[path]
    path = source_path(path)
    return path.read_bytes().decode("utf-8-sig") if path.suffix == ".py" else path.read_text(encoding="utf-8")


def write(path: Path, text: str, original: str) -> None:
    if text == original:
        return
    log(f"  ~ {path.relative_to(ROOT)}")
    if DRY:
        PENDING[path] = text
        return
    # keep a BOM-free UTF-8 file; line endings are normalised by .gitattributes
    path.write_bytes(text.encode("utf-8"))


# ---------------------------------------------------------------------------
# 1. package directories, git mv, femto/paths.py
# ---------------------------------------------------------------------------

def stage_move() -> None:
    for pkg, doc in PACKAGE_DOCS.items():
        directory = ROOT / pkg.replace(".", "/")
        init = directory / "__init__.py"
        log(f"  + {init.relative_to(ROOT)}")
        if not DRY:
            directory.mkdir(parents=True, exist_ok=True)
            if not init.exists():
                init.write_text(f'"""{doc}"""\n', encoding="utf-8")
    for module, pkg in MAP.items():
        src = ROOT / f"{module}.py"
        dst = ROOT / pkg.replace(".", "/") / f"{module}.py"
        if src.exists():
            run("git", "mv", str(src.relative_to(ROOT)), str(dst.relative_to(ROOT)))
    for module in TOOLS:
        src = ROOT / f"{module}.py"
        if src.exists():
            run("git", "mv", f"{module}.py", f"tools/{module}.py")
    paths_py = ROOT / "femto" / "paths.py"
    content = '''"""Repository-relative locations shared by every module (stdlib only).

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
'''
    log(f"  + {paths_py.relative_to(ROOT)}")
    if not DRY:
        paths_py.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# 2. import rewriting (AST driven, formatting preserving)
# ---------------------------------------------------------------------------

def _abs_index(lines: list[str], lineno: int, col: int) -> int:
    before = sum(len(line) for line in lines[: lineno - 1])
    prefix = lines[lineno - 1].encode("utf-8")[:col].decode("utf-8", errors="replace")
    return before + len(prefix)


def rewrite_imports(text: str) -> str:
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    edits: list[tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level or not node.module:
                continue
            base = node.module.split(".")[0]
            if base not in MAP:
                continue
            new = f"{MAP[base]}.{node.module}"
            start = _abs_index(lines, node.lineno, node.col_offset)
            end = _abs_index(lines, node.end_lineno, node.end_col_offset)
            segment = text[start:end]
            replaced, count = re.subn(r"^(from\s+)" + re.escape(node.module) + r"(\s+import\b)", r"\g<1>" + new + r"\2", segment)
            assert count == 1, segment
            edits.append((start, end, replaced))
        elif isinstance(node, ast.Import):
            if not any(alias.name.split(".")[0] in MAP for alias in node.names):
                continue
            parts = []
            for alias in node.names:
                base = alias.name.split(".")[0]
                if base in MAP:
                    parts.append(f"import {MAP[base]}.{alias.name} as {alias.asname or alias.name}")
                else:
                    parts.append(f"import {alias.name}" + (f" as {alias.asname}" if alias.asname else ""))
            start = _abs_index(lines, node.lineno, node.col_offset)
            end = _abs_index(lines, node.end_lineno, node.end_col_offset)
            edits.append((start, end, "; ".join(parts)))
    for start, end, replacement in sorted(edits, key=lambda e: e[0], reverse=True):
        text = text[:start] + replacement + text[end:]
    return text


PATCH_TARGET = re.compile(r"(patch(?:\.object)?\(\s*['\"])(" + ALT + r")(\.)")


def rewrite_patch_targets(text: str) -> str:
    return PATCH_TARGET.sub(lambda m: m.group(1) + MAP[m.group(2)] + "." + m.group(2) + m.group(3), text)


ROOT_LINE = re.compile(r"^ROOT\s*=\s*Path\(__file__\)\.resolve\(\)\.parent[ \t]*(?=\r?$)", re.M)


def rewrite_root(text: str) -> str:
    return ROOT_LINE.sub("from femto.paths import REPO_ROOT as ROOT", text)


def python_targets() -> list[Path]:
    targets = list((ROOT / "femto").rglob("*.py")) + list((ROOT / "tests").glob("*.py")) + list((ROOT / "tools").glob("*.py"))
    targets.append(ROOT / "commercial_compliance_check.py")
    if DRY:  # validate the rewrite against the not-yet-moved flat modules too
        targets.extend(ROOT / f"{module}.py" for module in MAP if (ROOT / f"{module}.py").exists())
    return [p for p in targets if p.name != "_migrate_packages.py"]


def stage_rewrite() -> None:
    for path in python_targets():
        original = read(path)
        text = rewrite_imports(original)
        text = rewrite_patch_targets(text)
        if path.is_relative_to(ROOT / "femto") or (DRY and path.parent == ROOT and path.stem in MAP):
            text = rewrite_root(text)
        if DRY and path.parent == ROOT and path.stem in MAP:
            # register under the future package location so stage 3 sees stage 2's result
            PENDING[ROOT / MAP[path.stem].replace(".", "/") / path.name] = text
            continue
        write(path, text, original)


# ---------------------------------------------------------------------------
# 3. targeted edits: subprocess launches, data roots, tests, tools
# ---------------------------------------------------------------------------

def dominant_eol(text: str) -> str:
    return "\r\n" if text.count("\r\n") > text.count("\n") - text.count("\r\n") else "\n"


def flex_replace(text: str, old: str, new: str, count: int, label: str) -> str:
    """Replace ``old`` with ``new`` while tolerating CRLF / LF / mixed line endings."""
    pattern = re.escape(old).replace("\\\n", "\r?\n").replace("\\\r\\\n", "\r?\n")
    matches = list(re.finditer(pattern, text))
    assert len(matches) == count, f"{label}: expected {count} occurrence(s) of {old[:60]!r}, found {len(matches)}"
    for match in reversed(matches):
        eol = "\r\n" if "\r\n" in match.group(0) else dominant_eol(text)
        text = text[: match.start()] + new.replace("\n", eol) + text[match.end():]
    return text


def ensure_import(text: str, line: str) -> str:
    if line in text:
        return text
    eol = dominant_eol(text)
    lines = text.splitlines(keepends=True)
    insert_at = None
    for index, current in enumerate(lines):
        stripped = current.strip()
        if stripped.startswith("from __future__"):
            insert_at = index + 1
            continue
        if stripped.startswith(("import ", "from ")) and insert_at is None:
            insert_at = index
            break
        if stripped.startswith(("import ", "from ")):
            break
    if insert_at is None:
        insert_at = 0
    lines.insert(insert_at, line + eol)
    return "".join(lines)


EDITS: list[tuple[str, list[tuple[str, str, int]], list[str]]] = [
    ("femto/app/app_local_history.py", [
        ("script='fixed_rgb_viewer.py' if key=='fixed' else 'scene_demo_viewer.py'",
         "module='femto.planning.fixed_rgb_viewer' if key=='fixed' else 'femto.viewer.scene_demo_viewer'", 1),
        ("subprocess.Popen([sys.executable,str(Path(__file__).with_name(script)),str(Path(row['result'])/'reconstruction'),'--lang',self.language],cwd=Path(__file__).parent)",
         "subprocess.Popen([sys.executable,'-m',module,str(Path(row['result'])/'reconstruction'),'--lang',self.language],cwd=REPO_ROOT)", 1),
    ], ["from femto.paths import REPO_ROOT"]),
    ("femto/app/app_local_panel.py", [
        ("subprocess.Popen([sys.executable,str(Path(__file__).parent/'fixed_rgb_viewer.py'),str(self.result/'reconstruction'),'--lang',self.app.language],cwd=Path(__file__).parent)",
         "subprocess.Popen([sys.executable,'-m','femto.planning.fixed_rgb_viewer',str(self.result/'reconstruction'),'--lang',self.app.language],cwd=REPO_ROOT)", 1),
    ], ["from femto.paths import REPO_ROOT"]),
    ("femto/app/app_temporal_panel.py", [
        ("subprocess.Popen([sys.executable,str(Path(__file__).with_name('segmented_cloud_viewer.py')),str(output/'fused_pointcloud.ply')],\n",
         "subprocess.Popen([sys.executable,'-m','femto.viewer.segmented_cloud_viewer',str(output/'fused_pointcloud.ply')],cwd=REPO_ROOT,\n", 1),
    ], ["from femto.paths import REPO_ROOT"]),
    ("femto/planning/export_history.py", [
        ("subprocess.Popen([sys.executable,str(Path(__file__).with_name('export_preview.py')),str(row['asset'])])",
         "subprocess.Popen([sys.executable,'-m','femto.viewer.export_preview',str(row['asset'])],cwd=REPO_ROOT)", 1),
        ("subprocess.Popen([sys.executable,str(Path(__file__).with_name('export_preview.py')),str(asset)])",
         "subprocess.Popen([sys.executable,'-m','femto.viewer.export_preview',str(asset)],cwd=REPO_ROOT)", 1),
    ], ["from femto.paths import REPO_ROOT"]),
    ("femto/planning/fixed_rgb_viewer.py", [
        ("subprocess.Popen([sys.executable,str(Path(__file__).with_name('scene_demo_viewer.py')),str(self.result),'--lang',self.language])",
         "subprocess.Popen([sys.executable,'-m','femto.viewer.scene_demo_viewer',str(self.result),'--lang',self.language],cwd=REPO_ROOT)", 1),
    ], ["from femto.paths import REPO_ROOT"]),
    ("femto/app/femto_control_app.py", [
        ('from femto.paths import REPO_ROOT as ROOT\nDATASETS_ROOT = ROOT / "datasets"\nPHOTOS_ROOT = ROOT / "photos"\nOUTPUTS_ROOT = ROOT / "outputs"\n',
         "from femto.paths import DATASETS_ROOT, LOG_ROOT, OUTPUTS_ROOT, PHOTOS_ROOT, REPO_ROOT as ROOT\n", 1),
        ('                str(ROOT / "scene_demo_viewer.py"),\n', '                "-m",\n                "femto.viewer.scene_demo_viewer",\n', 1),
        ('"entry": "femto_control_app.py",', '"entry": "femto.app.femto_control_app",', 1),
        ('log_dir = ROOT / "Log"', "log_dir = LOG_ROOT", 1),
    ], []),
    ("femto/gsplat/gsplat_backend.py", [
        ('str(executable), str(ROOT / "scene_gsplat_trainer.py"), "train",', 'str(executable), "-m", "femto.gsplat.scene_gsplat_trainer", "train",', 1),
        ('str(executable), str(ROOT / "scene_gsplat_viewer.py"),', 'str(executable), "-m", "femto.gsplat.scene_gsplat_viewer",', 1),
    ], []),
    ("femto/local/local_comparison.py", [
        ("stats = supervise([sys.executable, '-X', 'utf8', str(Path(__file__).resolve()),",
         "stats = supervise([sys.executable, '-X', 'utf8', '-m', 'femto.local.local_comparison',", 1),
        ("            'local_comparison.py', 'local_comparison_viewer.py', 'scene_reconstruction.py',\n"
         "            'local_pose_refinement.py', 'local_temporal_support.py', 'local_geometry.py',\n"
         "            'local_reference.py', 'scene_dataset_validator.py', 'fixed_rgb_geometry.py', 'local_stability.py',\n"
         "            'io_utils.py')))",
         "            'femto/local/local_comparison.py', 'femto/local/local_comparison_viewer.py',\n"
         "            'femto/geometry/scene_reconstruction.py', 'femto/local/local_pose_refinement.py',\n"
         "            'femto/local/local_temporal_support.py', 'femto/geometry/local_geometry.py',\n"
         "            'femto/dataset/local_reference.py', 'femto/dataset/scene_dataset_validator.py',\n"
         "            'femto/annotation/fixed_rgb_geometry.py', 'femto/tracking/local_stability.py',\n"
         "            'femto/io_utils.py')))", 1),
    ], []),
    ("femto/pipeline/local_offline.py", [
        ("self.stats = supervise([sys.executable, '-X', 'utf8', str(Path(__file__).resolve()),",
         "self.stats = supervise([sys.executable, '-X', 'utf8', '-m', 'femto.pipeline.local_offline',", 1),
    ], []),
    ("femto/gsplat/scene_single_gs.py", [
        ("command = [str(runtime_python()), str(Path(__file__)), 'train', str(output), '--iterations', str(iterations)]",
         "command = [str(runtime_python()), '-m', 'femto.gsplat.scene_single_gs', 'train', str(output), '--iterations', str(iterations)]", 1),
    ], []),
    ("femto/gsplat/scene_single_gs_viewer.py", [
        ("return subprocess.Popen([str(runtime_python()),str(Path(__file__)),str(output),'--lang',language],\n            cwd=Path(__file__).parent,",
         "return subprocess.Popen([str(runtime_python()),'-m','femto.gsplat.scene_single_gs_viewer',str(output),'--lang',language],\n            cwd=REPO_ROOT,", 1),
    ], ["from femto.paths import REPO_ROOT"]),
    ("tools/scene_gsplat_experiments.py", [
        ("command = [str(runtime_python()), str(REPO_ROOT / 'scene_gsplat_trainer.py'),",
         "command = [str(runtime_python()), '-m', 'femto.gsplat.scene_gsplat_trainer',", 1),
        ("result = subprocess.run(command, env=runtime_environment(), stdout=log,",
         "result = subprocess.run(command, cwd=REPO_ROOT, env=runtime_environment(), stdout=log,", 1),
    ], []),
    ("femto/tracking/local_tracking_replay.py", [
        ("tracking_code_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ('scene_live_tracker.py','local_tracking_replay.py','local_frame_selection.py')})",
         "tracking_code_sha256={p:hashlib.sha256((REPO_ROOT/p).read_bytes()).hexdigest() for p in ('femto/tracking/scene_live_tracker.py','femto/tracking/local_tracking_replay.py','femto/tracking/local_frame_selection.py')})", 1),
        ("def load_baseline(directory):\n    path = Path(directory)/'scene_live_tracker.py'\n",
         "# Baseline snapshots recorded before the package migration import their siblings by\n"
         "# bare name; alias those names to the packaged modules before executing the snapshot.\n"
         "LEGACY_FLAT_MODULES = {\n"
         "    'io_utils': 'femto.io_utils',\n"
         "    'moving_quality': 'femto.tracking.moving_quality',\n"
         "    'scene_dataset_validator': 'femto.dataset.scene_dataset_validator',\n"
         "    'scene_reconstruction': 'femto.geometry.scene_reconstruction',\n"
         "    'local_tracking_background': 'femto.tracking.local_tracking_background',\n"
         "    'local_tracking_diagnostics': 'femto.tracking.local_tracking_diagnostics',\n"
         "}\n\n\n"
         "def load_baseline(directory):\n"
         "    import importlib\n"
         "    for legacy, packaged in LEGACY_FLAT_MODULES.items():\n"
         "        sys.modules.setdefault(legacy, importlib.import_module(packaged))\n"
         "    path = Path(directory)/'scene_live_tracker.py'\n", 1),
    ], ["from femto.paths import REPO_ROOT"]),
    ("femto/local/local_parameter_study.py", [
        ("    'local_comparison.py', 'local_comparison_viewer.py', 'scene_reconstruction.py',\n"
         "    'local_pose_refinement.py', 'local_temporal_support.py', 'local_geometry.py',\n"
         "    'local_reference.py', 'scene_dataset_validator.py', 'fixed_rgb_geometry.py', 'local_stability.py',\n"
         "    'io_utils.py',\n)",
         "    'femto/local/local_comparison.py', 'femto/local/local_comparison_viewer.py',\n"
         "    'femto/geometry/scene_reconstruction.py', 'femto/local/local_pose_refinement.py',\n"
         "    'femto/local/local_temporal_support.py', 'femto/geometry/local_geometry.py',\n"
         "    'femto/dataset/local_reference.py', 'femto/dataset/scene_dataset_validator.py',\n"
         "    'femto/annotation/fixed_rgb_geometry.py', 'femto/tracking/local_stability.py',\n"
         "    'femto/io_utils.py',\n)", 1),
        ("ROOT/'scene_data_kind.py'", "ROOT/'femto/dataset/scene_data_kind.py'", 1),
    ], []),
    ("femto/viewer/export_preview.py", [
        ("folder=Path(__file__).parent/'Log';folder.mkdir(exist_ok=True)", "folder=LOG_ROOT;folder.mkdir(parents=True,exist_ok=True)", 1),
    ], ["from femto.paths import LOG_ROOT"]),
    ("femto/viewer/live_point_cloud.py", [
        ('log = Path(__file__).resolve().parent / "Log" / "live_point_cloud.log"', 'log = LOG_ROOT / "live_point_cloud.log"', 1),
    ], ["from femto.paths import LOG_ROOT"]),
    ("femto/planning/planning_drafts.py", [
        ("ROOT=Path(__file__).parent/'outputs'/'planning_drafts'", "ROOT=OUTPUTS_ROOT/'planning_drafts'", 1),
    ], ["from femto.paths import OUTPUTS_ROOT"]),
    ("femto/planning/region_project.py", [
        ("Path(__file__).parent/'outputs'/'region_workspaces'", "OUTPUTS_ROOT/'region_workspaces'", 2),
    ], ["from femto.paths import OUTPUTS_ROOT"]),
    ("femto/local/local_geometry_diagnostics.py", [
        ("Path(__file__).parent/'outputs'/'local_geometry_diagnostics'", "OUTPUTS_ROOT/'local_geometry_diagnostics'", 1),
    ], ["from femto.paths import OUTPUTS_ROOT"]),
    ("femto/temporal/scene_temporal_fusion.py", [
        ("capture_temporal_frames(Path(__file__).resolve().parent/'datasets',count)", "capture_temporal_frames(DATASETS_ROOT,count)", 1),
    ], ["from femto.paths import DATASETS_ROOT"]),
    ("tools/download_model.py", [
        ("ROOT = Path(__file__).resolve().parent\n", "ROOT = Path(__file__).resolve().parents[1]\n", 1),
    ], []),
    ("tests/test_planning_recovery.py", [
        ("from planning_drafts import Journal", "from femto.planning.planning_drafts import Journal", 1),
    ], []),
    ("tests/test_local_history.py", [
        ("self.assertTrue(args[1].endswith('fixed_rgb_viewer.py'))", "self.assertEqual(args[1:3],['-m','femto.planning.fixed_rgb_viewer'])", 1),
        ("self.assertEqual(Path(args[2]),self.result/'reconstruction')", "self.assertEqual(Path(args[3]),self.result/'reconstruction')", 1),
    ], []),
    ("tests/test_temporal_gui.py", [
        ("self.assertIn('segmented_cloud_viewer.py',launch.call_args.args[0][1])",
         "self.assertEqual(launch.call_args.args[0][1:3],['-m','femto.viewer.segmented_cloud_viewer'])", 1),
    ], []),
    ("tests/test_gsplat_backend.py", [
        ('self.assertEqual(command[2], "train")', 'self.assertEqual(command[1:4], ["-m", "femto.gsplat.scene_gsplat_trainer", "train"])', 1),
        ('        self.assertTrue(command[1].endswith("scene_gsplat_trainer.py"))\n', "", 1),
    ], []),
    ("tests/test_export_history.py", [
        ("with patch('femto.viewer.export_preview.__file__',str(Path(folder)/'export_preview.py')):",
         "with patch('femto.viewer.export_preview.LOG_ROOT',Path(folder)/'Log'):", 1),
    ], []),
    ("清理历史数据.ps1", [
        ("(Join-Path $projectRoot 'fixed_rgb_viewer.py')", "(Join-Path $projectRoot 'commercial_compliance_check.py')", 1),
    ], []),
]


def stage_edits() -> None:
    for relative, replacements, imports in EDITS:
        path = ROOT / relative
        original = read(path)
        text = original
        for old, new, count in replacements:
            text = flex_replace(text, old, new, count, relative)
        for line in imports:
            text = ensure_import(text, line)
        write(path, text, original)
    # test_reconstruction_sources: viewer argv shifts by one because of "-m <module>"
    path = ROOT / "tests/test_reconstruction_sources.py"
    original = read(path)
    text = original.replace("viewer.call_args.args[0][2]", "viewer.call_args.args[0][3]")
    write(path, text, original)


# ---------------------------------------------------------------------------
# 4. .cmd entry points
# ---------------------------------------------------------------------------

CMD_EDITS: list[tuple[str, bytes, bytes]] = [
    ("convert_latest_scene_to_nerfstudio.cmd", b'"%~dp0scene_to_nerfstudio.py"', b"-m femto.pipeline.scene_to_nerfstudio"),
    ("install_lpips_weights.cmd", b"scene_gsplat_trainer.py install-lpips", b"-m femto.gsplat.scene_gsplat_trainer install-lpips"),
    ("setup_direct_gsplat_environment.cmd", b'"C:\\FemtoGsplatRuntime\\Scripts\\python.exe" patch_gsplat_windows.py', b'"C:\\FemtoGsplatRuntime\\Scripts\\python.exe" tools\\patch_gsplat_windows.py'),
    ("setup_environment.cmd", b'".venv\\Scripts\\python.exe" download_model.py', b'".venv\\Scripts\\python.exe" tools\\download_model.py'),
    ("start_3dgs_status.cmd", b"scene_3dgs.py status", b"-m femto.gsplat.scene_3dgs status"),
    ("start_3dgs_viewer.cmd", b"scene_3dgs.py view", b"-m femto.gsplat.scene_3dgs view"),
    ("start_complete_demo.cmd", b"complete_demo.py --mode synthetic", b"-m femto.pipeline.complete_demo --mode synthetic"),
    ("start_complete_demo_viewer.cmd", b'".venv\\Scripts\\python.exe" scene_demo_viewer.py', b'".venv\\Scripts\\python.exe" -m femto.viewer.scene_demo_viewer'),
    ("start_depth_debug.cmd", b"depth_debug_viewer.py %*", b"-m femto.viewer.depth_debug_viewer %*"),
    ("start_direct_gsplat_status.cmd", b"scene_gsplat_trainer.py status", b"-m femto.gsplat.scene_gsplat_trainer status"),
    ("start_direct_gsplat_train.cmd", b"scene_gsplat_trainer.py train", b"-m femto.gsplat.scene_gsplat_trainer train"),
    ("start_direct_gsplat_viewer.cmd", b"scene_gsplat_viewer.py %*", b"-m femto.gsplat.scene_gsplat_viewer %*"),
    ("start_femto_app.cmd", b"femto_control_app.py", b"-m femto.app.femto_control_app"),
    ("start_fixed_rgb_viewer.cmd", b"-X utf8 fixed_rgb_viewer.py", b"-X utf8 -m femto.planning.fixed_rgb_viewer"),
    ("start_latest_viewer.cmd", b'-u "view_latest.py"', b"-u -m femto.viewer.view_latest"),
    ("start_real_scene_demo.cmd", b'".venv\\Scripts\\python.exe" scene_capture.py', b'".venv\\Scripts\\python.exe" -m femto.capture.scene_capture'),
    ("start_real_scene_demo.cmd", b'".venv\\Scripts\\python.exe" complete_demo.py', b'".venv\\Scripts\\python.exe" -m femto.pipeline.complete_demo'),
    ("start_region_planner.cmd", b'"%~dp0region_project.py"', b"-m femto.planning.region_project"),
    ("start_scene_capture.cmd", b'-u "scene_capture.py"', b"-u -m femto.capture.scene_capture"),
    ("start_temporal_fusion.cmd", b"scene_temporal_fusion.py --capture", b"-m femto.temporal.scene_temporal_fusion --capture"),
    ("start_top_scan.cmd", b'-u "top_scan.py"', b"-u -m femto.capture.top_scan"),
    ("validate_latest_scene.cmd", b'"%~dp0scene_dataset_validator.py"', b"-m femto.dataset.scene_dataset_validator"),
    ("validate_scene_scale.cmd", b"scene_scale_validation.py %*", b"-m femto.reports.scene_scale_validation %*"),
    ("生成实拍验收报告.cmd", b'"%~dp0scene_quality_report.py"', b"-m femto.reports.scene_quality_report"),
    ("单帧3DGS建模.cmd", b"from scene_single_gs import launch,ROOT; from scene_single_gs_viewer import open_viewer",
     b"from femto.gsplat.scene_single_gs import launch,ROOT; from femto.gsplat.scene_single_gs_viewer import open_viewer"),
    ("查看单帧3DGS.cmd", b"from scene_single_gs import MARKER; from scene_single_gs_viewer import open_viewer",
     b"from femto.gsplat.scene_single_gs import MARKER; from femto.gsplat.scene_single_gs_viewer import open_viewer"),
]


def stage_cmd() -> None:
    for name, old, new in CMD_EDITS:
        path = ROOT / name
        data = path.read_bytes()
        assert data.count(old) == 1, f"{name}: {old!r} occurs {data.count(old)} times"
        log(f"  ~ {name}")
        if not DRY:
            path.write_bytes(data.replace(old, new))


def main() -> None:
    log("== stage 1: move")
    stage_move()
    log("== stage 2: rewrite imports / patch targets / ROOT")
    stage_rewrite()
    log("== stage 3: targeted edits")
    stage_edits()
    log("== stage 4: .cmd entry points")
    stage_cmd()
    log("done" + (" (dry run)" if DRY else ""))


if __name__ == "__main__":
    main()
