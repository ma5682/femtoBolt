"""Atomic JSON/CSV export and reproducible reload for SurfaceTrajectory."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
import csv
import hashlib
import json
import os
import uuid

import trimesh

from femto.planning.planning_io import publish, staging_directory
from femto.planning.surface_trajectory import Gap, ScanSegment, SurfaceTrajectory, canonical_json_bytes, validate_trajectory


FILES_SCHEMA = "femto-surface-trajectory-files-v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: list[list[Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerows(rows)


def _write_core(folder: Path, trajectory: SurfaceTrajectory, check: Callable[[], None]) -> None:
    payload = trajectory.to_dict()
    (folder / "trajectory.json").write_bytes(canonical_json_bytes(payload))
    blocks = trajectory.blocks
    point_rows = [["route_index", "segment_index", "row_index", "point_index",
                   "x_m", "y_m", "z_m", "triangle_index", "geometry_support"]
                  + (["block_index"] if blocks else [])]
    gap_rows = [["gap_index", "reason", "inferred_cause", "row_index_from", "row_index_to",
                 "previous_route_index", "next_route_index", "prev_x_m", "prev_y_m", "prev_z_m",
                 "next_x_m", "next_y_m", "next_z_m", "chord_length_m",
                 "scan_dir_x", "scan_dir_y", "scan_dir_z"]
                + (["block_from", "block_to", "row_turn_failure_reason"] if blocks else [])]
    previous_block = None
    for item in trajectory.ordered_items:
        check()
        if isinstance(item, ScanSegment):
            for point_index, point in enumerate(item.points):
                point_rows.append([point["route_index"], item.segment_index, item.row_index, point_index,
                                   *point["xyz_m"], point["triangle_index"], point["geometry_support"]]
                                  + ([item.block_index] if blocks else []))
            previous_block = item.block_index
        else:
            gap_rows.append([item.gap_index, item.reason, item.detail["inferred_cause"],
                             item.row_index_from, item.row_index_to,
                             item.previous_point["route_index"], item.next_point["route_index"],
                             *item.previous_point["xyz_m"], *item.next_point["xyz_m"],
                             item.chord_length_m, *item.scan_direction_w0])
            if blocks:
                gap_rows[-1].extend([previous_block, _next_block(trajectory, item.gap_index),
                                     item.detail.get("row_turn_failure_reason", "")])
    _write_csv(folder / "trajectory.csv", point_rows)
    _write_csv(folder / "gaps.csv", gap_rows)
    if blocks:
        # One row per mesh block in visit order; the same authority as trajectory.json.
        block_rows = [["block_index", "component_id", "status", "reason", "face_count", "face_index_min",
                       "face_index_max", "face_indices_sha256", "area_m2", "segment_index_first",
                       "segment_count", "gap_count", "point_count", "row_index_first", "row_count",
                       "scan_length_m", "connector_length_m", "connected_row_turns"]]
        for block in blocks["items"]:
            block_rows.append([block.get(key) for key in block_rows[0]])
        _write_csv(folder / "blocks.csv", block_rows)


def _next_block(trajectory: SurfaceTrajectory, gap_index: int) -> Any:
    segments = [item for item in trajectory.ordered_items if isinstance(item, ScanSegment)]
    return segments[gap_index + 1].block_index if gap_index + 1 < len(segments) else None


def write_files_manifest(folder: str | Path, trajectory: SurfaceTrajectory) -> Path:
    folder = Path(folder)
    files = {path.name: _sha256(path) for path in sorted(folder.iterdir(), key=lambda value: value.name)
             if path.is_file() and path.name != "trajectory_files.json"}
    payload = {"schema": FILES_SCHEMA, "result_sha256": trajectory.to_dict()["result_sha256"],
               "files": files}
    path = folder / "trajectory_files.json"
    path.write_bytes(canonical_json_bytes(payload))
    return path


def export_trajectory(trajectory: SurfaceTrajectory, parent: str | Path, *,
                      check: Callable[[], None] = lambda: None,
                      commit: Callable[[Callable[[], Any]], Any] | None = None,
                      augment: Callable[[Path, SurfaceTrajectory, Callable[[], None]], None] | None = None,
                      include_visuals: bool = True) -> Path:
    """Publish a complete result directory; cancellation leaves no partial folder."""
    if include_visuals:
        from femto.planning.trajectory_diagram import prepare_export_trajectory, write_visual_outputs
        trajectory = prepare_export_trajectory(trajectory)
        if augment is None:
            augment = write_visual_outputs
    validate_trajectory(trajectory)
    parent = Path(parent).resolve()
    parent.mkdir(parents=True, exist_ok=True)
    final = parent / ("trajectory_" + uuid.uuid4().hex[:12])
    with staging_directory(parent, ".trajectory_staging_") as staging:
        check()
        _write_core(staging, trajectory, check)
        if augment is not None:
            augment(staging, trajectory, check)
        check()
        write_files_manifest(staging, trajectory)
        check()

        def finish() -> Path:
            if final.exists():
                raise FileExistsError(final)
            os.replace(staging, final)
            return final

        return publish(finish, check, commit)


def export_blocked(parent: str | Path, source: str | Path, error: Exception,
                   *, details: Any = None) -> Path:
    """Publish one explicit validation refusal, not a partial trajectory."""
    parent = Path(parent).resolve()
    parent.mkdir(parents=True, exist_ok=True)
    final = parent / ("trajectory_" + uuid.uuid4().hex[:12])
    with staging_directory(parent, ".trajectory_blocked_staging_") as staging:
        payload = {"schema": "femto-surface-trajectory-blocked-v1", "source": str(Path(source).resolve()),
                   "reason": type(error).__name__, "message": str(error), "details": details,
                   "execution_ready": False, "physical_accuracy_verified": False}
        (staging / "trajectory_blocked.json").write_bytes(canonical_json_bytes(payload))
        os.replace(staging, final)
    return final


def _verify_files(folder: Path, result_sha256: str) -> None:
    path = folder / "trajectory_files.json"
    if not path.is_file():
        raise ValueError("Trajectory file integrity manifest is missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != FILES_SCHEMA or payload.get("result_sha256") != result_sha256:
        raise ValueError("Trajectory file integrity manifest mismatch")
    files = payload.get("files")
    if not isinstance(files, dict) or "trajectory.json" not in files:
        raise ValueError("Trajectory file integrity list is invalid")
    for name, expected in files.items():
        target = (folder / name).resolve()
        if not target.is_relative_to(folder.resolve()) or not target.is_file() or _sha256(target) != expected:
            raise ValueError("Trajectory output file hash mismatch: " + str(name))


def load_trajectory(path: str | Path, *, check: Callable[[], None] = lambda: None,
                    reproduce: bool = True) -> SurfaceTrajectory:
    path = Path(path).resolve()
    if path.is_dir():
        path = path / "trajectory.json"
    if not path.is_file() or path.stat().st_size > 256 * 1024 * 1024:
        raise ValueError("Trajectory JSON is missing or exceeds 256 MiB")
    payload = json.loads(path.read_text(encoding="utf-8"))
    trajectory = SurfaceTrajectory.from_dict(payload)
    _verify_files(path.parent, payload["result_sha256"])
    source = trajectory.source
    manifest_path = Path(source.get("region_manifest", "")).resolve()
    if not manifest_path.is_file() or _sha256(manifest_path) != source.get("region_manifest_sha256"):
        raise ValueError("Trajectory region manifest source mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    mesh_path = (manifest_path.parent / manifest["files"]["mesh_file"]).resolve()
    if _sha256(mesh_path) != source.get("mesh_file_sha256"):
        raise ValueError("Trajectory region mesh source mismatch")
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    validate_trajectory(trajectory, mesh=mesh, expected_result_sha256=payload["result_sha256"])
    if reproduce and trajectory.planner.get("kind") == "mesh_slice":
        check()
        from femto.planning.mesh_slice_planner import plan_mesh_slice
        regenerated = plan_mesh_slice(manifest_path, check=check, **trajectory.planner["parameters"])
        current = regenerated.planner.get("algorithm_version")
        if trajectory.planner.get("algorithm_version") != current:
            raise ValueError("Trajectory was generated by an older algorithm version "
                             f"({trajectory.planner.get('algorithm_version')} vs {current}); regenerate it")
        saved = trajectory.to_dict()
        fresh = regenerated.to_dict()
        for optional in ("route_diagram_status",):
            saved.pop(optional, None)
            fresh.pop(optional, None)
        if saved != fresh:
            raise ValueError("Trajectory does not reproduce from its bound mesh and parameters")
    return trajectory
