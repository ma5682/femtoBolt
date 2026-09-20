"""Conservative dataset-kind classification shared by reporting entry points."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


FEMTO_BOLT_DEVICE = "Orbbec Femto Bolt"
SYNTHETIC_TEST_DEVICE = "synthetic_test_camera"


def classify_manifest_data_kind(
    manifest: Mapping[str, Any],
    *,
    dataset: Path | None = None,
) -> str:
    """Return ``synthetic``, verified ``real_femto``, or ``unknown``.

    A non-empty camera device is not evidence that the capture came from the
    supported Femto hardware.  ``real_femto`` therefore requires the exact
    canonical device value written by the Orbbec capture adapter.  Synthetic
    markers take precedence so test data cannot be promoted by a stray camera
    label.
    """
    camera = manifest.get("camera") or {}
    pose = manifest.get("pose") or {}
    status = str(manifest.get("status", "")).lower()
    device = camera.get("device")
    device_marker = str(device).lower()
    pose_source = str(pose.get("source", "")).lower()
    if (
        device_marker == SYNTHETIC_TEST_DEVICE
        or "synthetic" in device_marker
        or "synthetic" in status
        or "synthetic" in pose_source
        or (dataset is not None and dataset.name.lower().startswith("demo_scene_"))
    ):
        return "synthetic"
    if device == FEMTO_BOLT_DEVICE:
        return "real_femto"
    return "unknown"


def load_dataset_manifest_chain(dataset: Path | None) -> tuple[list[tuple[Path, dict[str, Any]]], bool]:
    """Load the current-to-raw manifest chain and report whether it is complete.

    A missing manifest, malformed JSON, or cycle makes the chain incomplete.
    Callers may still recognize explicit synthetic markers in the readable
    portion, but an incomplete chain can never establish ``real_femto``.
    """
    if dataset is None:
        return [], False
    current = Path(dataset).resolve()
    chain: list[tuple[Path, dict[str, Any]]] = []
    visited: set[str] = set()
    while True:
        marker = str(current).casefold()
        if marker in visited:
            return chain, False
        visited.add(marker)
        try:
            value = json.loads((current / "manifest.json").read_text(encoding="utf-8-sig"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return chain, False
        if not isinstance(value, Mapping):
            return chain, False
        manifest = dict(value)
        chain.append((current, manifest))
        source = manifest.get("source_dataset")
        if not source:
            return chain, True
        current = Path(str(source)).resolve()


def classify_dataset_data_kind(dataset: Path | None) -> str:
    """Classify a dataset from its full lineage, trusting only the raw end."""
    chain, complete = load_dataset_manifest_chain(dataset)
    if any(classify_manifest_data_kind(manifest, dataset=path) == "synthetic" for path, manifest in chain):
        return "synthetic"
    if not complete or not chain:
        return "unknown"
    raw_path, raw_manifest = chain[-1]
    return classify_manifest_data_kind(raw_manifest, dataset=raw_path)
