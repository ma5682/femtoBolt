"""Stage C, step 1: derived work-surface *candidates* from bound raw depth (diagnostic only).

Selected pixels that have reference-frame depth but no measured mesh
(`selection_support` class 2) keep their coverage requirement. This module binds the raw
depth / calibration / pose evidence of the region export, builds candidate triangles from the
reference frame's *measured* depth with the existing pixel-adjacency rule (no interpolation,
no smoothing, no hole filling), keeps only triangles whose complete projection stays inside
one selection component, and saves the candidate surface with per-pixel and per-view
diagnostics next to the export.

Nothing here is admitted into measurement or planning: the region export, the source mesh,
the selection support record and every saved trajectory stay untouched, the planner keeps
`derived_surface.status=not_generated`, and acceptance thresholds (hole size, density,
cross-view support, fit error) remain user decisions. Residuals against the reference depth
are not independent accuracy evidence because the candidates come from that same depth.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import cv2
import numpy as np
import open3d as o3d
import trimesh

from femto.annotation.selection_intent import (
    SUPPORT_CLASSES, component_labels, intent_mask, selection_contours, selection_layers, support_classes,
    validate_selection_intent, validate_selection_support,
)
from femto.dataset.local_reference import calibration_identity
from femto.geometry.rgbd_frames import distortion_maps, load_poses
from femto.geometry.single_frame_mesh import build_pixel_mesh
from femto.io_utils import imread_unicode, load_json
from femto.planning.mesh_slice_planner import Unplannable, _load_input
from femto.planning.planning_io import publish, staging_directory
from femto.planning.selection_blocks import INTENT_MISSING_MESSAGE
from femto.planning.surface_row_turns import segment_pixels
from femto.planning.surface_trajectory import canonical_json_bytes


SCHEMA = "femto-derived-surface-candidates-v1"
FILES_SCHEMA = "femto-derived-surface-candidates-files-v1"
METHOD = "reference_frame_measured_depth_pixel_adjacency_candidates"
STATUS = "candidate_not_admitted"
ADMISSION = "not_allowed_pending_user_confirmed_thresholds"
DIAGNOSTIC_TOLERANCE_M = 0.01  # existing 10 mm diagnostic scale; not an acceptance gate
FOLDER_PREFIX = "derived_surface_candidates_"
JSON_NAME = "derived_surface_candidates.json"
PLY_NAME = "candidate_surface_m.ply"
FACES_CSV = "candidate_faces.csv"
PIXELS_CSV = "candidate_pixels.csv"
DIAGRAM_NAME = "derived_surface_diagram.png"
FILES_NAME = "derived_surface_files.json"
CANDIDATE_CLASS = 2
CONFLICT_CLASS = 5
MEASURED_CLASS = 1
FACE_REJECTIONS = ("outside_image_or_unselected", "spans_two_selection_components", "touches_depth_conflict_pixel",
                   "outside_reference_depth_range", "no_missing_mesh_vertex", "crosses_unselected_pixels")
LIMITATIONS = [
    "candidates come from the reference frame's measured depth only; no pixel without depth is interpolated",
    "candidate points are the reference depth itself, so agreement with that depth is not accuracy evidence",
    "other-view residuals are diagnostics at the existing 10 mm scale; no acceptance gate is defined",
    "no depth-step or normal gate against the surrounding measured mesh is applied (user decision pending)",
    "seam vertices lie on pixels whose mesh/depth agree within 10 mm; no stitching to the measured mesh",
    "candidate faces are not admitted into selection_support, measurement or trajectory planning",
    "depth_conflict pixels are never attempted; they need recapture or a user decision",
]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(dataset: Path) -> list[dict[str, Any]]:
    text = (dataset / "frames.jsonl").read_text(encoding="utf-8-sig")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _dataset_depth_range_m(dataset: Path) -> list[float]:
    """Depth range the reconstruction and the fixed RGB page used for this dataset (same lookup as
    `complete_demo._depth_range_mm`, without its defaults: a dataset without a range is refused)."""
    manifest = load_json(dataset / "manifest.json")
    parameters = manifest.get("parameters") or {}
    range_m = (manifest.get("camera") or {}).get("depth_range_m")
    minimum_mm = parameters.get("depth_min_mm", 1000.0 * range_m[0] if range_m else None)
    maximum_mm = parameters.get("depth_max_mm", 1000.0 * range_m[1] if range_m else None)
    if minimum_mm is None or maximum_mm is None:
        raise ValueError("Bound dataset manifest records no depth range")
    return [float(minimum_mm) / 1000.0, float(maximum_mm) / 1000.0]


def evidence_record(binding: dict[str, Any]) -> dict[str, Any]:
    """The `evidence` block of the record; `load_candidates` recomputes it from disk and requires equality."""
    reference = binding["frames"][binding["reference_index"]]
    return dict(dataset=binding["dataset"], dataset_manifest_sha256=binding["dataset_manifest_sha256"],
                frames_jsonl_sha256=binding["frames_jsonl_sha256"], calibration_sha256=binding["calibration"]["sha256"],
                reference_frame_id=reference["frame"], frames=binding["frames"],
                depth_range_m=binding["depth_range_m"], transform_direction="T_W0_Ct",
                kind="raw_per_view_depth_calibration_pose_not_display_cloud_or_texture")


def bind_evidence(manifest: dict[str, Any], *, check: Callable[[], None] = lambda: None) -> dict[str, Any]:
    """Resolve and verify raw depth, calibration and poses bound by a region export.

    Requires the export's selection intent + support; the reference pose must be the W0
    identity, every hash in the export must match the bytes on disk, and the support record's
    depth range must be the range the bound dataset was reconstructed with."""
    region = manifest["region"]
    view = region["view_reference"]
    origin = manifest["origin_reference"]
    if not isinstance(region.get("selection_intent"), dict) or not isinstance(region.get("selection_support"), dict):
        raise Unplannable("selection_intent_missing", INTENT_MISSING_MESSAGE)
    intent = validate_selection_intent(region["selection_intent"], view["image_size"])
    dataset = Path(origin["source_dataset"]).resolve()
    if not dataset.is_dir():
        raise ValueError("Bound posed dataset is missing: " + str(dataset))
    support = validate_selection_support(region["selection_support"], intent_mask(intent),
                                         expected_depth_range_m=_dataset_depth_range_m(dataset))
    depth_range_m = [float(v) for v in support["depth_range_m"]]
    rows = _rows(dataset)
    reference_id = origin["reference_frame_id"]
    matches = [index for index, row in enumerate(rows) if row.get("frame") == reference_id]
    if len(matches) != 1:
        raise ValueError("Reference frame is not unique in the bound dataset")
    reference_index = matches[0]
    calibration = calibration_identity(dataset)
    if calibration["sha256"] != origin.get("calibration_sha256"):
        raise ValueError("Calibration identity does not match the region export")
    pose_path = Path(origin["reference_pose"]).resolve()
    if pose_path != (dataset / rows[reference_index]["pose"]).resolve():
        raise ValueError("Reference pose path does not match the bound dataset")
    if _sha(pose_path) != origin.get("reference_pose_sha256"):
        raise ValueError("Reference pose SHA-256 mismatch")
    check()
    poses = load_poses(dataset, rows)
    if not np.allclose(poses[reference_index], np.eye(4), rtol=0, atol=1e-9):
        raise ValueError("Reference pose is not the W0 identity")
    frames = []
    for index, row in enumerate(rows):
        check()
        record = dict(frame=int(row["frame"]), rgb=str((dataset / row["rgb"]).resolve()),
                      depth=str((dataset / row["depth"]).resolve()), pose=str((dataset / row["pose"]).resolve()))
        record.update(rgb_sha256=_sha(Path(record["rgb"])), depth_sha256=_sha(Path(record["depth"])),
                      pose_sha256=_sha(Path(record["pose"])), T_W0_Ct=poses[index].tolist(),
                      role="reference" if index == reference_index else "other_view")
        frames.append(record)
    reference = frames[reference_index]
    if reference["rgb_sha256"] != view.get("rgb_sha256") or reference["depth_sha256"] != view.get("depth_sha256"):
        raise ValueError("Reference RGB / depth SHA-256 mismatch with the region export")
    map_x, map_y, k, width, height = distortion_maps(dataset)
    if [width, height] != list(view["image_size"]) or not np.allclose(k, view["undistorted_intrinsic"], rtol=0, atol=1e-6):
        raise ValueError("Undistorted reference intrinsic / image size mismatch")
    return dict(dataset=str(dataset), frames=frames, reference_index=reference_index, poses=poses,
                calibration=calibration, map_x=map_x, map_y=map_y, k=k, width=width, height=height,
                dataset_manifest_sha256=_sha(dataset / "manifest.json"), frames_jsonl_sha256=_sha(dataset / "frames.jsonl"),
                intrinsic=SimpleNamespace(**load_json(dataset / "calibration" / "color_intrinsic.json")),
                distortion=load_json(dataset / "calibration" / "distortion.json")["rgb"],
                intent=intent, support=support, depth_range_m=depth_range_m)


def _pixels(uv: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    pix = np.floor(uv + .5)
    inside = np.isfinite(uv).all(axis=1) & (pix[:, 0] >= 0) & (pix[:, 0] < width) & (pix[:, 1] >= 0) & (pix[:, 1] < height)
    safe = np.where(inside[:, None], pix, 0).astype(np.int64)
    return safe, inside


def _triangle_inside_component(uv: np.ndarray, allowed: np.ndarray) -> bool:
    """Every pixel touched by the projected triangle (edges and interior) belongs to `allowed`."""
    height, width = allowed.shape
    for a, b in ((0, 1), (1, 2), (2, 0)):
        for u, v in segment_pixels(uv[a], uv[b]):
            if not (0 <= u < width and 0 <= v < height and allowed[v, u]):
                return False
    lo = np.floor(uv.min(axis=0) + .5).astype(int)
    hi = np.floor(uv.max(axis=0) + .5).astype(int)
    for v in range(lo[1], hi[1] + 1):
        for u in range(lo[0], hi[0] + 1):
            p = np.array([u, v], float)
            d = uv[1:] - uv[0]
            det = d[0, 0] * d[1, 1] - d[0, 1] * d[1, 0]
            if abs(det) < 1e-12:
                continue
            s = ((p[0] - uv[0, 0]) * d[1, 1] - (p[1] - uv[0, 1]) * d[1, 0]) / det
            t = (-(p[0] - uv[0, 0]) * d[0, 1] + (p[1] - uv[0, 1]) * d[0, 0]) / det
            if s >= 0 and t >= 0 and s + t <= 1 and not (0 <= u < width and 0 <= v < height and allowed[v, u]):
                return False
    return True


def _select_faces(vertices: np.ndarray, faces: np.ndarray, uv: np.ndarray, labels: np.ndarray, classes: np.ndarray,
                  depth_range: list[float], check: Callable[[], None]) -> tuple[np.ndarray, dict[str, int]]:
    height, width = labels.shape
    pix, inside = _pixels(uv, width, height)
    owner = np.where(inside, labels[pix[:, 1], pix[:, 0]], 0)
    cls = np.where(inside, classes[pix[:, 1], pix[:, 0]], 0)
    counts = {name: 0 for name in FACE_REJECTIONS}
    face_owner = owner[faces]
    face_cls = cls[faces]
    z = vertices[:, 2][faces]
    keep = np.ones(len(faces), bool)
    stage = (face_owner > 0).all(axis=1)
    counts["outside_image_or_unselected"] = int((~stage).sum()); keep &= stage
    stage = (face_owner == face_owner[:, :1]).all(axis=1)
    counts["spans_two_selection_components"] = int((keep & ~stage).sum()); keep &= stage
    stage = ~(face_cls == CONFLICT_CLASS).any(axis=1)
    counts["touches_depth_conflict_pixel"] = int((keep & ~stage).sum()); keep &= stage
    stage = ((z >= depth_range[0]) & (z <= depth_range[1])).all(axis=1)
    counts["outside_reference_depth_range"] = int((keep & ~stage).sum()); keep &= stage
    stage = (face_cls == CANDIDATE_CLASS).any(axis=1)
    counts["no_missing_mesh_vertex"] = int((keep & ~stage).sum()); keep &= stage
    retained = []
    for face_index in np.flatnonzero(keep):
        if len(retained) % 2048 == 0:
            check()
        component = int(face_owner[face_index, 0])
        if _triangle_inside_component(uv[faces[face_index]], labels == component):
            retained.append(int(face_index))
        else:
            counts["crosses_unselected_pixels"] += 1
    return np.asarray(retained, np.int64), counts


def _percentiles(values: np.ndarray) -> list[float] | None:
    values = np.asarray(values, float)
    return None if not values.size else [float(v) for v in np.percentile(values, [50, 95, 100])]


def _coverage(covered_image: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """A candidate pixel is covered iff its own measured depth point is a vertex of a retained face.

    Exact by construction (candidates are pixel points); pixel-centre rays would be numerically
    unreliable exactly at vertices and would add no information."""
    return covered_image[ys, xs]


def _cross_view(binding: dict[str, Any], xyz: np.ndarray) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    k, width, height = binding["k"], binding["width"], binding["height"]
    low, high = binding["depth_range_m"]
    observed_count = np.zeros(len(xyz), np.int64)
    within_count = np.zeros(len(xyz), np.int64)
    views = []
    for index, frame in enumerate(binding["frames"]):
        if index == binding["reference_index"]:
            continue
        depth = imread_unicode(Path(frame["depth"]), cv2.IMREAD_UNCHANGED)
        if depth.shape != (height, width):
            raise ValueError("Other-view depth dimensions mismatch")
        aligned = cv2.remap(depth, binding["map_x"], binding["map_y"], cv2.INTER_NEAREST).astype(float) / 1000
        inverse = np.linalg.inv(binding["poses"][index])
        camera = xyz @ inverse[:3, :3].T + inverse[:3, 3]
        z = camera[:, 2]
        ok = z > 0
        safe_z = np.where(ok, z, 1.0)
        uv = np.c_[camera[:, 0] / safe_z * k[0, 0] + k[0, 2], camera[:, 1] / safe_z * k[1, 1] + k[1, 2]]
        pix, inside = _pixels(uv, width, height)
        ok &= inside
        observed = np.where(ok, aligned[pix[:, 1], pix[:, 0]], np.nan)
        ok &= (observed >= low) & (observed <= high)
        error_mm = np.where(ok, np.abs(observed - z) * 1000, np.nan)
        within = ok & (error_mm <= DIAGNOSTIC_TOLERANCE_M * 1000)
        observed_count += ok
        within_count += within
        views.append(dict(frame=frame["frame"], depth_sha256=frame["depth_sha256"], observed_pixels=int(ok.sum()),
                          within_diagnostic_tolerance=int(within.sum()), over_diagnostic_tolerance=int((ok & ~within).sum()),
                          residual_percentiles_mm=_percentiles(error_mm[ok])))
    return views, observed_count, within_count


def _seam_distances(candidate: trimesh.Trimesh | None, region_mesh: trimesh.Trimesh, seam: np.ndarray) -> list[float] | None:
    if candidate is None or not seam.any():
        return None
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(np.asarray(region_mesh.vertices, np.float32)),
                        o3d.core.Tensor(np.asarray(region_mesh.faces, np.uint32)))
    points = o3d.core.Tensor(np.asarray(candidate.vertices, np.float32)[seam])
    closest = scene.compute_closest_points(points)["points"].numpy()
    return _percentiles(np.linalg.norm(closest - np.asarray(candidate.vertices)[seam], axis=1) * 1000)


def build_candidates(source: str | Path, *, check: Callable[[], None] = lambda: None) -> dict[str, Any]:
    """Generate and diagnose candidate faces for one region export; returns a result dict.

    The dict carries `candidate_mesh` (trimesh or None), `pixel_table`, `face_table` and
    `diagram` arrays for export; `payload` is the JSON record."""
    manifest_path, manifest, region_mesh = _load_input(source)
    binding = bind_evidence(manifest, check=check)
    intent, support = binding["intent"], binding["support"]
    labels = component_labels(intent)
    classes = support_classes(support)
    height, width, k = binding["height"], binding["width"], binding["k"]
    reference = binding["frames"][binding["reference_index"]]
    raw_depth = imread_unicode(Path(reference["depth"]), cv2.IMREAD_UNCHANGED)
    bgr = imread_unicode(Path(reference["rgb"]), cv2.IMREAD_COLOR)
    if raw_depth.shape != (height, width) or bgr.shape != (height, width, 3):
        raise ValueError("Reference RGB / depth dimensions mismatch")
    check()
    # The reference camera *is* W0 (identity pose verified above): camera XYZ == W0 XYZ.
    pixel_mesh, _, mesher = build_pixel_mesh(raw_depth, bgr, binding["intrinsic"], binding["distortion"], np.eye(4))
    vertices = np.asarray(pixel_mesh.vertices, np.float64)
    faces = np.asarray(pixel_mesh.triangles, np.int64)
    check()
    z = vertices[:, 2]
    uv = np.c_[vertices[:, 0] / z * k[0, 0] + k[0, 2], vertices[:, 1] / z * k[1, 1] + k[1, 2]]
    retained, rejections = _select_faces(vertices, faces, uv, labels, classes, binding["depth_range_m"], check)
    candidate = None
    face_components = np.zeros(0, np.int64)
    seam = np.zeros(0, bool)
    covered_image = np.zeros((height, width), bool)
    if len(retained):
        used, inverse = np.unique(faces[retained], return_inverse=True)
        candidate = trimesh.Trimesh(vertices=vertices[used], faces=inverse.reshape(-1, 3), process=False)
        pix, _ = _pixels(uv[used], width, height)
        face_components = labels[pix[:, 1], pix[:, 0]][np.asarray(candidate.faces)[:, 0]].astype(np.int64) - 1
        seam = classes[pix[:, 1], pix[:, 0]] == MEASURED_CLASS
        covered_image[pix[:, 1], pix[:, 0]] = True
    check()
    aligned = cv2.remap(raw_depth, binding["map_x"], binding["map_y"], cv2.INTER_NEAREST).astype(float) / 1000
    ys, xs = np.nonzero(classes == CANDIDATE_CLASS)
    covered = _coverage(covered_image, ys, xs)
    rays = np.column_stack(((xs - k[0, 2]) / k[0, 0], (ys - k[1, 2]) / k[1, 1], np.ones(len(xs))))
    xyz = rays * aligned[ys, xs][:, None]
    views, observed_count, within_count = _cross_view(binding, xyz)
    check()
    pixel_components = labels[ys, xs].astype(np.int64) - 1
    per_component = []
    for component in intent["components"]:
        cid = component["component_id"]
        mine = pixel_components == cid
        per_component.append(dict(
            component_id=cid, selected_pixels=component["pixel_count"], candidate_pixels=int(mine.sum()),
            covered_pixels=int((mine & covered).sum()), uncovered_pixels=int((mine & ~covered).sum()),
            depth_conflict_pixels=int(((classes == CONFLICT_CLASS) & (labels == cid + 1)).sum()),
            candidate_faces=int((face_components == cid).sum()),
            note="component_id equals blocks[].component_id of mesh-slice-serpentine-v4; pixel counts are not areas"))
    other_views = max(len(binding["frames"]) - 1, 0)
    payload = dict(
        schema=SCHEMA, status=STATUS, admission_to_planning=ADMISSION, method=METHOD,
        execution_ready=False, physical_accuracy_verified=False,
        coordinate_frame="world_rdf", unit="metre", origin_reference=manifest["origin_reference"],
        source=dict(region_export_dir=str(manifest_path.parent), region_manifest=str(manifest_path),
                    region_manifest_sha256=_sha(manifest_path), mesh_file=manifest["files"]["mesh_file"],
                    mesh_file_sha256=manifest["file_sha256"][manifest["files"]["mesh_file"]],
                    geometry_sha256=manifest["geometry_sha256"], selection_intent_sha256=intent["selected_mask_sha256"],
                    selection_support_sha256=support["class_sha256"], region_id=manifest["region"].get("annotation_id")),
        evidence=evidence_record(binding),
        mesher=dict(mesher, note="existing single-frame pixel-adjacency rule on the reference raw depth; unchanged thresholds"),
        selection=dict(component_count=len(intent["components"]), hole_count=len(intent["explicit_unselected_holes"]),
                       support_counts=support["counts"], candidate_pixel_count=int(len(xs)),
                       depth_conflict_pixel_count=int((classes == CONFLICT_CLASS).sum()),
                       strict_rule="every pixel touched by a candidate triangle (edges and interior) lies in one selection component"),
        faces=dict(retained=int(len(retained)), rejected=rejections, seam_vertex_count=int(seam.sum()),
                   seam_distance_to_region_mesh_percentiles_mm=_seam_distances(candidate, region_mesh, seam),
                   seam_rule="vertices on measured_mesh pixels (mesh/depth agree within 10 mm); not stitched"),
        coverage=dict(rule="candidate pixel covered iff its measured depth point is a vertex of a retained candidate face",
                      covered_pixels=int(covered.sum()), uncovered_pixels=int((~covered).sum()),
                      covered_ratio=(float(covered.mean()) if len(xs) else None),
                      meaning="pixel bookkeeping of the same-input candidate surface, not independent accuracy",
                      per_component=per_component),
        cross_view=dict(diagnostic_tolerance_m=DIAGNOSTIC_TOLERANCE_M, gate="none", other_view_count=other_views, views=views,
                        pixels_observed_by_all_other_views=int((observed_count == other_views).sum()) if other_views else 0,
                        pixels_within_tolerance_in_all_observing_views=int(((observed_count > 0) & (within_count == observed_count)).sum()),
                        pixels_over_tolerance_in_any_view=int((within_count < observed_count).sum()),
                        pixels_unobserved_by_other_views=int((observed_count == 0).sum())),
        completeness=dict(all_candidate_pixels_covered=bool(len(xs)) and bool(covered.all()),
                          depth_conflict_pixels_remaining=int((classes == CONFLICT_CLASS).sum()),
                          statement="not complete while any candidate pixel is uncovered or any depth_conflict pixel remains"),
        limitations=LIMITATIONS)
    pixel_table = np.c_[xs, ys, pixel_components, covered.astype(int), observed_count, within_count]
    face_table = None if candidate is None else np.c_[np.arange(len(candidate.faces)), face_components]
    diagram = _diagram(bgr, binding, manifest["region"], ys, xs, covered, classes)
    return dict(payload=payload, candidate_mesh=candidate, pixel_table=pixel_table, face_table=face_table, diagram=diagram)


def _diagram(bgr: np.ndarray, binding: dict[str, Any], region: dict[str, Any], ys: np.ndarray, xs: np.ndarray,
             covered: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Undistorted reference RGB: green = candidate covered, orange = uncovered, red = depth conflict; cyan outlines."""
    rgb = cv2.remap(bgr, binding["map_x"], binding["map_y"], cv2.INTER_LINEAR)[:, :, ::-1].astype(np.float32)
    rgb[ys[covered], xs[covered]] = rgb[ys[covered], xs[covered]] * .35 + np.array([40, 220, 90], np.float32) * .65
    rgb[ys[~covered], xs[~covered]] = rgb[ys[~covered], xs[~covered]] * .35 + np.array([245, 150, 40], np.float32) * .65
    conflict = classes == CONFLICT_CLASS
    rgb[conflict] = rgb[conflict] * .35 + np.array([230, 40, 60], np.float32) * .65
    image = np.clip(rgb, 0, 255).astype(np.uint8)
    layers = selection_layers(region)
    if layers is not None and tuple(layers["mask"].shape) == image.shape[:2]:
        for _, polyline in selection_contours(layers)["components"]:
            cv2.polylines(image, [polyline.astype(np.int32).reshape(-1, 1, 2)], True, (56, 230, 216), 1)
    return image


def _cell(value: Any) -> Any:
    value = float(value)
    if not np.isfinite(value):
        return ""
    return int(value) if value.is_integer() else f"{value:.6f}"


def _write_csv(path: Path, header: list[str], rows: Any) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow([_cell(v) for v in row])


def export_candidates(result: dict[str, Any], parent: str | Path | None = None, *,
                      check: Callable[[], None] = lambda: None) -> Path:
    """Write the diagnostic folder atomically next to the region export (or under `parent`)."""
    payload = dict(result["payload"])
    parent = Path(parent).resolve() if parent else Path(payload["source"]["region_export_dir"]).resolve()
    parent.mkdir(parents=True, exist_ok=True)
    final = parent / (FOLDER_PREFIX + uuid.uuid4().hex[:12])
    with staging_directory(parent, ".derived_surface_staging_") as staging:
        check()
        candidate = result["candidate_mesh"]
        files = dict(json=JSON_NAME, pixels_csv=PIXELS_CSV, diagram=DIAGRAM_NAME)
        if candidate is not None:
            (staging / PLY_NAME).write_bytes(candidate.export(file_type="ply"))
            _write_csv(staging / FACES_CSV, ["face_index", "component_id"], result["face_table"])
            files.update(ply=PLY_NAME, faces_csv=FACES_CSV)
        _write_csv(staging / PIXELS_CSV, ["u", "v", "component_id", "covered",
                                          "other_views_observed", "other_views_within_10mm"], result["pixel_table"])
        cv2.imencode(".png", np.ascontiguousarray(result["diagram"][:, :, ::-1]))[1].tofile(str(staging / DIAGRAM_NAME))
        payload["files"] = files
        payload["candidate_face_count"] = 0 if candidate is None else int(len(candidate.faces))
        (staging / JSON_NAME).write_bytes(canonical_json_bytes(payload))
        check()
        hashes = {p.name: _sha(p) for p in sorted(staging.iterdir()) if p.is_file()}
        (staging / FILES_NAME).write_bytes(canonical_json_bytes(dict(schema=FILES_SCHEMA, files=hashes)))

        def finish() -> Path:
            if final.exists():
                raise FileExistsError(final)
            os.replace(staging, final)
            return final

        return publish(finish, check)


def export_blocked(parent: str | Path, source: str | Path, error: Exception) -> Path:
    parent = Path(parent).resolve()
    parent.mkdir(parents=True, exist_ok=True)
    final = parent / (FOLDER_PREFIX + uuid.uuid4().hex[:12])
    with staging_directory(parent, ".derived_surface_blocked_staging_") as staging:
        payload = dict(schema="femto-derived-surface-candidates-blocked-v1", source=str(Path(source).resolve()),
                       reason=getattr(error, "reason", type(error).__name__), message=str(error),
                       execution_ready=False, physical_accuracy_verified=False)
        (staging / "derived_surface_blocked.json").write_bytes(canonical_json_bytes(payload))
        os.replace(staging, final)
    return final


def load_candidates(folder: str | Path) -> dict[str, Any]:
    """Re-read a diagnostic folder.

    Re-verified: file hashes, the record's status / admission / safety flags, the region export
    and its selection intent + support, the bound raw evidence (every view's RGB / depth / pose
    bytes, `frames.jsonl`, dataset manifest, calibration identity, reference W0 identity pose,
    depth range) exactly as recorded, and every candidate face's whole-pixel containment."""
    folder = Path(folder).resolve()
    files = json.loads((folder / FILES_NAME).read_text(encoding="utf-8"))
    if files.get("schema") != FILES_SCHEMA or not isinstance(files.get("files"), dict):
        raise ValueError("Derived surface file manifest is invalid")
    for name, expected in files["files"].items():
        target = (folder / name).resolve()
        if not target.is_relative_to(folder) or not target.is_file() or _sha(target) != expected:
            raise ValueError("Derived surface file hash mismatch: " + name)
    payload = json.loads((folder / JSON_NAME).read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("status") != STATUS or payload.get("admission_to_planning") != ADMISSION:
        raise ValueError("Derived surface record is not a non-admitted candidate diagnostic")
    if payload.get("execution_ready") is not False or payload.get("physical_accuracy_verified") is not False:
        raise ValueError("Derived surface safety flags are invalid")
    manifest_path = Path(payload["source"]["region_manifest"])
    if not manifest_path.is_file() or _sha(manifest_path) != payload["source"]["region_manifest_sha256"]:
        raise ValueError("Region export changed since the candidates were generated")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    region = manifest["region"]
    binding = bind_evidence(manifest)
    intent, support = binding["intent"], binding["support"]
    if (intent["selected_mask_sha256"] != payload["source"]["selection_intent_sha256"]
            or support["class_sha256"] != payload["source"]["selection_support_sha256"]):
        raise ValueError("Selection intent / support changed since the candidates were generated")
    if json.loads(canonical_json_bytes(evidence_record(binding))) != payload.get("evidence"):
        raise ValueError("Bound raw evidence (RGB / depth / pose / calibration / frames) changed since the candidates were generated")
    mesh = None
    if "ply" in payload["files"]:
        loaded = trimesh.load(folder / payload["files"]["ply"], force="mesh", process=False)
        if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) != payload["candidate_face_count"]:
            raise ValueError("Candidate surface PLY does not match its record")
        labels = component_labels(intent)
        k = np.asarray(region["view_reference"]["undistorted_intrinsic"], float)
        vertices = np.asarray(loaded.vertices, float)
        z = vertices[:, 2]
        if np.any(z <= 0):
            raise ValueError("Candidate surface has vertices behind the reference camera")
        uv = np.c_[vertices[:, 0] / z * k[0, 0] + k[0, 2], vertices[:, 1] / z * k[1, 1] + k[1, 2]]
        pix, inside = _pixels(uv, labels.shape[1], labels.shape[0])
        if not inside.all():
            raise ValueError("Candidate surface leaves the reference image")
        owners = labels[pix[:, 1], pix[:, 0]]
        for face in np.asarray(loaded.faces):
            component = int(owners[face[0]])
            if component <= 0 or not _triangle_inside_component(uv[face], labels == component):
                raise ValueError("Candidate surface face leaves its selection component")
        mesh = loaded
    return dict(payload=payload, candidate_mesh=mesh, folder=folder)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage C diagnostic: derived work-surface candidates (not admitted)")
    parser.add_argument("source", help="Region export directory or region_N.json")
    parser.add_argument("--output", help="Parent folder for the diagnostic (default: the region export directory)")
    args = parser.parse_args(argv)
    source = Path(args.source).resolve()
    output = Path(args.output).resolve() if args.output else (source if source.is_dir() else source.parent)
    try:
        folder = export_candidates(build_candidates(args.source), output)
    except (OSError, ValueError, TypeError, KeyError, IndexError) as error:
        print(export_blocked(output, args.source, error))
        return 2
    print(folder)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
