"""Small, dependency-free provenance and lifecycle contracts.

The capture, reconstruction, and planning layers write different assets, but
they must be able to state which revision they belong to.  This module keeps
that contract independent of Qt, Open3D, and the planning implementation.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from femto.io_utils import sha256_file


LINEAGE_SCHEMA = "femto-data-lineage-v1"
CAPTURE_STATES = frozenset({
    "disconnected", "connecting", "streaming", "stalled", "stopping",
    "saved", "failed",
})
CAPTURE_TRANSITIONS = {
    "disconnected": frozenset({"connecting", "failed"}),
    "connecting": frozenset({"streaming", "stalled", "failed", "disconnected"}),
    "streaming": frozenset({"stalled", "stopping", "failed"}),
    "stalled": frozenset({"streaming", "stopping", "failed", "disconnected"}),
    "stopping": frozenset({"saved", "failed"}),
    "saved": frozenset(),
    "failed": frozenset({"disconnected", "connecting"}),
}


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def value_hash(value: Any) -> str:
    return sha256(canonical_json_bytes(value)).hexdigest()


def file_records(root: str | Path, relative_paths: Iterable[str | Path],
                 check: Callable[[], None] | None = None) -> dict[str, str]:
    """Return deterministic file hashes and reject paths outside *root*."""
    root = Path(root).resolve()
    result: dict[str, str] = {}
    for relative in sorted((Path(value) for value in relative_paths), key=lambda value: value.as_posix()):
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Missing provenance asset: {relative}")
        if check is not None:
            check()
        key = path.relative_to(root).as_posix()
        result[key] = sha256_file(path, check)
    return result


def transition_capture_state(current: str, target: str) -> str:
    if current not in CAPTURE_STATES:
        raise ValueError(f"Unknown capture state: {current}")
    if target not in CAPTURE_STATES:
        raise ValueError(f"Unknown capture state: {target}")
    if target != current and target not in CAPTURE_TRANSITIONS[current]:
        raise ValueError(f"Invalid capture state transition: {current} -> {target}")
    return target


def make_lineage(*, session_id: str, revision_id: str,
                 source_manifest: str | None,
                 algorithm_version: str,
                 coordinate_frame: str,
                 unit: str,
                 source_files: Mapping[str, str] | None = None,
                 calibration_hash: str | None = None,
                 pose_hash: str | None = None,
                 geometry_hash: str | None = None,
                 region_hash: str | None = None,
                 planning_hash: str | None = None) -> dict[str, Any]:
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("session_id is required")
    if not isinstance(revision_id, str) or not revision_id:
        raise ValueError("revision_id is required")
    if not isinstance(algorithm_version, str) or not algorithm_version:
        raise ValueError("algorithm_version is required")
    if not isinstance(coordinate_frame, str) or not isinstance(unit, str):
        raise ValueError("coordinate_frame and unit are required")
    payload = {
        "schema": LINEAGE_SCHEMA,
        "session_id": session_id,
        "revision_id": revision_id,
        "source_manifest": source_manifest,
        "calibration_hash": calibration_hash,
        "pose_hash": pose_hash,
        "geometry_hash": geometry_hash,
        "region_hash": region_hash,
        "planning_hash": planning_hash,
        "algorithm_version": algorithm_version,
        "coordinate_frame": coordinate_frame,
        "unit": unit,
        "source_files": dict(source_files or {}),
    }
    payload["content_hash"] = value_hash(payload)
    return payload


def validate_lineage(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or value.get("schema") != LINEAGE_SCHEMA:
        raise ValueError("Invalid data lineage schema")
    data = dict(value)
    expected = data.pop("content_hash", None)
    if not isinstance(expected, str) or expected != value_hash(data):
        raise ValueError("Data lineage content hash mismatch")
    for key in ("session_id", "revision_id", "algorithm_version", "coordinate_frame", "unit"):
        if not isinstance(data.get(key), str) or not data[key]:
            raise ValueError(f"Data lineage field is missing: {key}")
    if not isinstance(data.get("source_files"), dict):
        raise ValueError("Data lineage source_files must be an object")
    data["content_hash"] = expected
    return data


__all__ = [
    "CAPTURE_STATES", "CAPTURE_TRANSITIONS", "LINEAGE_SCHEMA",
    "canonical_json_bytes", "file_records", "make_lineage",
    "transition_capture_state", "validate_lineage", "value_hash",
]
