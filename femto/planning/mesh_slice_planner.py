"""Deterministic parallel-plane surface trajectories on exported region meshes."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable
from types import SimpleNamespace
import argparse
import hashlib
import json
import math

import numpy as np
import trimesh

from femto.planning.surface_trajectory import (
    BLOCK_ALGORITHM, BLOCKS_SCHEMA, Gap, POINT_BUDGET, ScanSegment, SurfaceTrajectory,
    face_components, face_list_sha256, validate_trajectory,
)


ALGORITHM_VERSION = "mesh-slice-serpentine-v1"
DEFAULTS = {
    "row_spacing_mm": 5.0, "sample_step_mm": 1.0,
    "row_direction": "auto", "start_corner": "top_left",
    "min_segment_mm": 2.0, "spacing_tolerance": 0.5,
    "inplane_ambiguity_ratio_max": 0.8, "planarity_ratio_max": 0.5,
    "normal_view_angle_max_deg": 60.0, "adaptive_rows": False,
}
FROZEN = {
    "sample_step_mm": 1.0, "min_segment_mm": 2.0,
    "inplane_ambiguity_ratio_max": 0.8, "planarity_ratio_max": 0.5,
    "normal_view_angle_max_deg": 60.0, "adaptive_rows": False,
}
GAP_DETAIL_METHOD = "sweep_plane_outer_loop_containment"


class Unplannable(ValueError):
    """A surface (whole mesh or one block) that the frozen rules cannot plan."""

    def __init__(self, reason: str, message: str, details: Any = None):
        super().__init__(message)
        self.reason = reason
        self.details = details


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_path(source: str | Path) -> Path:
    path = Path(source).resolve()
    if path.is_file():
        return path
    if not path.is_dir():
        raise ValueError("Region export path does not exist")
    candidates = sorted(path.glob("region_*.json"))
    if len(candidates) != 1:
        raise ValueError("Export directory must contain exactly one region_N.json; pass the manifest explicitly")
    return candidates[0]


def _load_input(source: str | Path) -> tuple[Path, dict[str, Any], trimesh.Trimesh]:
    manifest_path = _manifest_path(source)
    if manifest_path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("Region manifest exceeds 64 MiB")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if not isinstance(manifest, dict) or manifest.get("schema") != "femto-region-export-v1":
        raise ValueError("Expected a femto-region-export-v1 manifest")
    if manifest.get("coordinate_frame") != "world_rdf" or manifest.get("unit") != "metre":
        raise ValueError("Region export must use W0 world_rdf metres")
    origin = manifest.get("origin_reference")
    if not isinstance(origin, dict) or origin.get("kind") != "initial_color_camera":
        raise ValueError("Initial color-camera origin cannot be verified")
    region = manifest.get("region")
    if not isinstance(region, dict) or not isinstance(region.get("view_reference"), dict):
        raise ValueError("Region/view reference is missing")
    view = region["view_reference"]
    intrinsic = np.asarray(view.get("undistorted_intrinsic"), dtype=np.float64)
    size = np.asarray(view.get("image_size"))
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all() or size.shape != (2,) or np.any(size <= 0):
        raise ValueError("Valid undistorted reference intrinsic and image size are required")
    files = manifest.get("files")
    hashes = manifest.get("file_sha256")
    if not isinstance(files, dict) or not isinstance(hashes, dict) or not isinstance(files.get("mesh_file"), str):
        raise ValueError("Region mesh binding is missing")
    mesh_path = (manifest_path.parent / files["mesh_file"]).resolve()
    if not mesh_path.is_relative_to(manifest_path.parent.resolve()) or not mesh_path.is_file():
        raise ValueError("Bound region mesh is missing")
    expected_hash = hashes.get(files["mesh_file"])
    if not isinstance(expected_hash, str) or _sha256(mesh_path) != expected_hash:
        raise ValueError("Region mesh SHA-256 mismatch")
    loaded = trimesh.load(mesh_path, force="mesh", process=False)
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError("Region PLY is not a triangle mesh")
    mesh = loaded
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if not len(vertices) or not len(faces) or not np.isfinite(vertices).all() or faces.shape[1:] != (3,):
        raise ValueError("Region mesh is empty or invalid")
    indices = region.get("triangle_indices")
    if not isinstance(indices, list) or len(faces) != len(indices):
        raise ValueError("Region PLY triangle count does not match the manifest")
    edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    if np.any(counts > 2):
        raise ValueError("Region mesh has non-manifold edges")
    return manifest_path, manifest, mesh


def _parameters(values: dict[str, Any]) -> dict[str, Any]:
    params = dict(DEFAULTS)
    params.update({key: values[key] for key in DEFAULTS if key in values and values[key] is not None})
    for key, expected in FROZEN.items():
        if params[key] != expected:
            raise ValueError(f"{key} is frozen at {expected} in v1")
    for key, low, high in (("row_spacing_mm", 0.5, 100.0),
                           ("sample_step_mm", 0.1, 50.0),
                           ("min_segment_mm", 0.0, 100.0),
                           ("spacing_tolerance", 0.0, 2.0),
                           ("inplane_ambiguity_ratio_max", 0.5, 1.0),
                           ("planarity_ratio_max", 0.1, 1.0),
                           ("normal_view_angle_max_deg", 10.0, 89.0)):
        value = params[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"Invalid {key}")
        params[key] = float(value)
    if params["start_corner"] not in ("top_left", "top_right", "bottom_left", "bottom_right"):
        raise ValueError("Invalid start_corner")
    direction = params["row_direction"]
    if not (direction in ("auto", "w0_x", "w0_y") if isinstance(direction, str) else type(direction) in (int, float)):
        try:
            params["row_direction"] = float(direction)
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid row_direction") from exc
    return params


def _unit_projection(axis: np.ndarray, normal: np.ndarray, label: str) -> np.ndarray:
    projected = axis - normal * float(np.dot(axis, normal))
    length = float(np.linalg.norm(projected))
    if length < math.sin(math.radians(10)):
        raise Unplannable("axis_perpendicular", f"{label} is almost perpendicular to the surface; choose another axis")
    return projected / length


def _pca_frame(mesh: trimesh.Trimesh, params: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    areas = np.linalg.norm(cross, axis=1) / 2
    if not np.isfinite(areas).all() or float(areas.sum()) <= 0:
        raise Unplannable("no_positive_area", "Region mesh has no positive-area triangles")
    centers = triangles.mean(axis=1)
    mu = np.average(centers, axis=0, weights=areas)
    delta = centers - mu
    covariance = np.einsum("i,ij,ik->jk", areas, delta, delta) / float(areas.sum())
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    if eigenvalues[1] <= 1e-20 or eigenvalues[0] <= 1e-20:
        raise Unplannable("insufficient_extent", "Region mesh has insufficient two-dimensional extent")
    view_axis = np.array([0.0, 0.0, -1.0])
    n_pca = eigenvectors[:, 2]
    normal_angle = math.degrees(math.acos(float(np.clip(abs(np.dot(n_pca, view_axis)), 0, 1))))
    planarity = float(eigenvalues[2] / eigenvalues[1])
    inplane = float(eigenvalues[1] / eigenvalues[0])
    if planarity <= params["planarity_ratio_max"] and normal_angle <= params["normal_view_angle_max_deg"]:
        n = n_pca
        normal_source = "pca"
    else:
        n = view_axis.copy()
        normal_source = "view_axis_fallback"
    if float(np.dot(n, view_axis)) < 0:
        n = -n
    direction = params["row_direction"]
    if direction == "auto":
        candidate = eigenvectors[:, 0] - n * float(np.dot(eigenvectors[:, 0], n))
        if inplane <= params["inplane_ambiguity_ratio_max"] and np.linalg.norm(candidate) > 1e-12:
            e1 = candidate / np.linalg.norm(candidate)
            direction_source = "pca"
        else:
            e1 = _unit_projection(np.array([1.0, 0.0, 0.0]), n, "W0 X")
            direction_source = "w0_x_fallback"
    elif direction in ("w0_x", "w0_y"):
        axis = np.array([1.0, 0.0, 0.0]) if direction == "w0_x" else np.array([0.0, 1.0, 0.0])
        e1 = _unit_projection(axis, n, "W0 X" if direction == "w0_x" else "W0 Y")
        direction_source = direction
    else:
        base = _unit_projection(np.array([1.0, 0.0, 0.0]), n, "W0 X")
        angle = math.radians(float(direction))
        e1 = base * math.cos(angle) + np.cross(n, base) * math.sin(angle)
        e1 /= np.linalg.norm(e1)
        direction_source = "w0_x_angle"
    for axis in np.eye(3):
        dot = float(np.dot(e1, axis))
        if abs(dot) >= 1e-6:
            if dot < 0:
                e1 = -e1
            break
    e2 = np.cross(n, e1)
    e2 /= np.linalg.norm(e2)
    frame = {
        "origin": mu.tolist(), "e1": e1.tolist(), "e2": e2.tolist(), "n": n.tolist(),
        "eigenvalues_m2": [float(v) for v in eigenvalues],
        "ratios": {"inplane": inplane, "planarity": planarity,
                   "normal_view_angle_deg": normal_angle},
        "direction_source": direction_source, "normal_source": normal_source,
        "thresholds": {key: params[key] for key in
                       ("inplane_ambiguity_ratio_max", "planarity_ratio_max", "normal_view_angle_max_deg")},
    }
    return mu, e1, e2, n, frame


_face_components = face_components


def _quantized(point: np.ndarray) -> tuple[int, int, int]:
    return tuple(int(v) for v in np.rint(np.asarray(point) / 1e-7))


def _chains(lines: np.ndarray, face_ids: np.ndarray) -> tuple[list[tuple[np.ndarray, list[int]]], bool]:
    grouped: dict[tuple[Any, Any], tuple[int, np.ndarray, np.ndarray]] = {}
    for line, face in sorted(zip(lines, face_ids), key=lambda item: (int(item[1]), _quantized(item[0][0]), _quantized(item[0][1]))):
        left, right = _quantized(line[0]), _quantized(line[1])
        if left == right:
            continue
        key = tuple(sorted((left, right)))
        candidate = (int(face), np.asarray(line[0], float), np.asarray(line[1], float))
        if key not in grouped or candidate[0] < grouped[key][0]:
            grouped[key] = candidate
    coordinates: dict[Any, np.ndarray] = {}
    edges: list[tuple[Any, Any, int]] = []
    for key in sorted(grouped):
        face, a, b = grouped[key]
        qa, qb = _quantized(a), _quantized(b)
        for q, point in ((qa, a), (qb, b)):
            if q not in coordinates or tuple(point) < tuple(coordinates[q]):
                coordinates[q] = point
        edges.append((qa, qb, face))
    adjacency: dict[Any, list[int]] = defaultdict(list)
    for index, (a, b, _) in enumerate(edges):
        adjacency[a].append(index)
        adjacency[b].append(index)
    for node in adjacency:
        adjacency[node].sort(key=lambda index: (edges[index][2], edges[index][1] if edges[index][0] == node else edges[index][0]))
    branched = any(len(value) > 2 for value in adjacency.values())
    unused = set(range(len(edges)))
    result: list[tuple[np.ndarray, list[int]]] = []

    def walk(start: Any, first: int) -> None:
        nodes = [start]
        used_faces: list[int] = []
        current, edge_index = start, first
        while edge_index in unused:
            unused.remove(edge_index)
            a, b, face = edges[edge_index]
            current = b if a == current else a
            nodes.append(current)
            used_faces.append(face)
            if current == start or len(adjacency[current]) != 2:
                break
            candidates = [candidate for candidate in adjacency[current] if candidate in unused]
            if not candidates:
                break
            edge_index = candidates[0]
        if used_faces:
            result.append((np.asarray([coordinates[node] for node in nodes]), used_faces))

    for node in sorted(adjacency):
        if len(adjacency[node]) != 2:
            for edge_index in adjacency[node]:
                if edge_index in unused:
                    walk(node, edge_index)
    while unused:
        edge_index = min(unused, key=lambda index: (min(edges[index][:2]), edges[index][2], max(edges[index][:2])))
        a, b, face = edges[edge_index]
        start = min(a, b)
        candidates = [candidate for candidate in adjacency[start] if candidate in unused]
        first = min(candidates, key=lambda index: (edges[index][2], edges[index][1] if edges[index][0] == start else edges[index][0]))
        walk(start, first)
    return result, branched


def _resample(points: np.ndarray, edge_faces: list[int], step_m: float) -> list[tuple[np.ndarray, int]]:
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = lengths > 1e-12
    if not np.any(keep):
        return []
    clean = [points[0]]
    faces: list[int] = []
    for index, valid in enumerate(keep):
        if valid:
            clean.append(points[index + 1])
            faces.append(int(edge_faces[index]))
    points = np.asarray(clean)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(lengths)]
    total = float(cumulative[-1])
    distances = np.r_[np.arange(0.0, total, step_m), total]
    distances = np.unique(np.minimum(distances, total))
    samples = []
    for distance in distances:
        edge = min(len(lengths) - 1, max(0, int(np.searchsorted(cumulative, distance, side="right") - 1)))
        fraction = (distance - cumulative[edge]) / max(lengths[edge], 1e-15)
        samples.append((points[edge] * (1 - fraction) + points[edge + 1] * fraction, faces[edge]))
    return samples


def _project(point: np.ndarray, intrinsic: np.ndarray, image_size: np.ndarray) -> list[float] | None:
    if point[2] <= 0:
        return None
    u = intrinsic[0, 0] * point[0] / point[2] + intrinsic[0, 2]
    v = intrinsic[1, 1] * point[1] / point[2] + intrinsic[1, 2]
    if not np.isfinite([u, v]).all() or not (0 <= u < image_size[0] and 0 <= v < image_size[1]):
        return None
    return [float(u), float(v)]


def _pixel_extent(points: np.ndarray, intrinsic: np.ndarray,
                  image_size: np.ndarray) -> tuple[list[float] | None, list[float] | None]:
    """Display-only projected bounds/centre of the points that fall inside the reference image."""
    z = points[:, 2]
    ok = z > 0
    safe = np.where(ok, z, 1.0)
    u = intrinsic[0, 0] * points[:, 0] / safe + intrinsic[0, 2]
    v = intrinsic[1, 1] * points[:, 1] / safe + intrinsic[1, 2]
    ok &= np.isfinite(u) & np.isfinite(v) & (u >= 0) & (u < image_size[0]) & (v >= 0) & (v < image_size[1])
    if not ok.any():
        return None, None
    return ([float(u[ok].min()), float(v[ok].min()), float(u[ok].max()), float(v[ok].max())],
            [float(u[ok].mean()), float(v[ok].mean())])


def _split_supported(samples: list[tuple[np.ndarray, int]], intrinsic: np.ndarray,
                     image_size: np.ndarray, min_length_m: float) -> tuple[list[list[dict[str, Any]]], int]:
    runs: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    dropped = 0
    for point, face in samples:
        pixel = _project(point, intrinsic, image_size)
        if pixel is None:
            if current:
                runs.append(current)
            current = []
            dropped += 1
        else:
            current.append({"xyz_m": [float(v) for v in point], "triangle_index": int(face),
                            "geometry_support": "measured_mesh", "pixel": pixel})
    if current:
        runs.append(current)
    retained = []
    for run in runs:
        length = float(np.linalg.norm(np.diff(np.asarray([p["xyz_m"] for p in run]), axis=0), axis=1).sum()) if len(run) > 1 else 0.0
        if len(run) >= 2 and length + 1e-12 >= min_length_m:
            retained.append(run)
        else:
            dropped += 1
    return retained, dropped


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    if len(polygon) < 3:
        return False
    x, y = point
    inside = False
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        if (a[1] > y) != (b[1] > y):
            inside ^= x < (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]) + a[0]
    return bool(inside)


def _outer_loop(vertices: np.ndarray, faces: np.ndarray, origin: np.ndarray,
                e1: np.ndarray, e2: np.ndarray) -> np.ndarray | None:
    """Largest projected boundary loop of one connected face set, or None when undetermined."""
    counts: Counter[tuple[int, int]] = Counter()
    for face in faces:
        counts.update(tuple(sorted((int(a), int(b)))) for a, b in
                      ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])))
    boundary = [edge for edge, count in counts.items() if count == 1]
    adjacency: dict[int, list[int]] = defaultdict(list)
    for a, b in boundary:
        adjacency[a].append(b)
        adjacency[b].append(a)
    if not boundary or any(len(values) != 2 for values in adjacency.values()):
        return None
    unused = {tuple(sorted(edge)) for edge in boundary}
    loops = []
    while unused:
        first = min(unused)
        start, current, previous = first[0], first[0], None
        loop = [start]
        while True:
            choices = sorted(node for node in adjacency[current]
                             if tuple(sorted((current, node))) in unused)
            if not choices:
                break
            nxt = choices[0]
            unused.remove(tuple(sorted((current, nxt))))
            previous, current = current, nxt
            loop.append(current)
            if current == start:
                break
        if len(loop) >= 4 and loop[-1] == start:
            xyz = vertices[np.asarray(loop[:-1])]
            uv = np.c_[(xyz - origin) @ e1, (xyz - origin) @ e2]
            loops.append(uv)
    if not loops:
        return None

    def area(poly: np.ndarray) -> float:
        return abs(float(np.sum(poly[:, 0] * np.roll(poly[:, 1], -1)
                                - poly[:, 1] * np.roll(poly[:, 0], -1)) / 2))
    return max(loops, key=lambda poly: (area(poly), tuple(poly.ravel())))


def _outer_loops(vertices: np.ndarray, faces: np.ndarray, components: np.ndarray,
                 origin: np.ndarray, e1: np.ndarray, e2: np.ndarray) -> dict[int, np.ndarray | None]:
    return {component: _outer_loop(vertices, faces[np.flatnonzero(components == component)], origin, e1, e2)
            for component in sorted(set(int(v) for v in components))}


def _forbidden_polygons(region: dict[str, Any]) -> list[np.ndarray]:
    values = []
    if region.get("forbidden_screen_polygon"):
        values.append(region["forbidden_screen_polygon"])
    values.extend(region.get("exclusion_polygons", []))
    return [np.asarray(value, dtype=np.float64) for value in values if len(value) >= 3]


def _gap_reason(left: dict[str, Any], right: dict[str, Any], region: dict[str, Any],
                intrinsic: np.ndarray, image_size: np.ndarray, step_m: float,
                outer: dict[int, np.ndarray | None], origin: np.ndarray,
                e1: np.ndarray, e2: np.ndarray) -> tuple[str, str]:
    if left["row_index"] != right["row_index"]:
        return "row_turn", "not_applicable"
    if left.get("unsupported") or right.get("unsupported"):
        return "unsupported_geometry", "not_applicable"
    a = np.asarray(left["points"][-1]["xyz_m"])
    b = np.asarray(right["points"][0]["xyz_m"])
    count = max(2, int(math.ceil(float(np.linalg.norm(b - a)) / step_m)) + 1)
    pixels = [_project(a * (1 - value) + b * value, intrinsic, image_size)
              for value in np.linspace(0, 1, count)]
    polygons = _forbidden_polygons(region)
    if any(pixel is not None and any(_point_in_polygon(np.asarray(pixel), polygon) for polygon in polygons)
           for pixel in pixels):
        return "forbidden_region", "not_applicable"
    if any(pixel is None for pixel in pixels):
        return "unsupported_geometry", "not_applicable"
    if left["component"] != right["component"]:
        return "disconnected_surface", "not_applicable"
    polygon = outer.get(left["component"])
    if polygon is None:
        return "surface_gap", "undetermined"
    midpoint = (a + b) / 2 - origin
    uv = np.array([np.dot(midpoint, e1), np.dot(midpoint, e2)])
    return "surface_gap", "interior_hole_likely" if _point_in_polygon(uv, polygon) else "boundary_concavity_likely"


def _direction(points: list[dict[str, Any]]) -> list[float]:
    delta = np.asarray(points[-1]["xyz_m"]) - np.asarray(points[-2]["xyz_m"])
    length = float(np.linalg.norm(delta))
    return (delta / max(length, 1e-15)).tolist()


def _gap_detail(inferred: str) -> dict[str, Any]:
    return {"inferred_cause": inferred, "method": GAP_DETAIL_METHOD,
            "confidence": "heuristic", "decision_use": "none"}


def _slice_rows(sliced: trimesh.Trimesh, face_lookup: np.ndarray | None, components: np.ndarray,
                origin: np.ndarray, e2: np.ndarray, positions: np.ndarray, params: dict[str, Any],
                intrinsic: np.ndarray, image_size: np.ndarray, check: Callable[[], None],
                budget: dict[str, int], block_index: int | None = None
                ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Slice one surface at every row position; the shared budget counts every retained sample."""
    spacing_m = params["row_spacing_mm"] / 1000
    step_m = params["sample_step_mm"] / 1000
    records_by_slice: list[dict[str, Any]] = []
    dropped_fragments: list[dict[str, Any]] = []

    def dropped(slice_index: int, length: Any, reason: str) -> dict[str, Any]:
        record = {"slice_index": slice_index, "length_m": length, "reason": reason}
        if block_index is not None:
            record["block_index"] = block_index
        return record

    for slice_index, position in enumerate(positions):
        check()
        lines, face_ids = trimesh.intersections.mesh_plane(
            sliced, plane_normal=e2, plane_origin=origin + e2 * position, return_faces=True)
        face_ids = np.asarray(face_ids)
        if face_lookup is not None and len(face_ids):
            face_ids = face_lookup[face_ids.astype(np.int64)]
        chains, branched = _chains(np.asarray(lines), face_ids)
        candidates = []
        row_dropped = 0
        for chain_index, (polyline, edge_faces) in enumerate(chains):
            length = float(np.linalg.norm(np.diff(polyline, axis=0), axis=1).sum())
            if length + 1e-12 < params["min_segment_mm"] / 1000:
                row_dropped += 1
                dropped_fragments.append(dropped(slice_index, length, "below_min_segment"))
                continue
            samples = _resample(polyline, edge_faces, step_m)
            runs, unsupported = _split_supported(samples, intrinsic, image_size,
                                                  params["min_segment_mm"] / 1000)
            row_dropped += unsupported
            dropped_fragments.extend(dropped(slice_index, None, "outside_reference_image_or_short_supported_run")
                                     for _ in range(unsupported))
            if branched:
                dropped_fragments.append(dropped(slice_index, None, "branched_slice_split"))
            for run in runs:
                budget["points"] += len(run)
                if budget["points"] > POINT_BUDGET:
                    raise ValueError(f"Trajectory exceeds {POINT_BUDGET} samples; increase row spacing or shrink the region")
                candidates.append({"points": run, "component": int(components[run[0]["triangle_index"]]),
                                   "unsupported": bool(unsupported or branched),
                                   "mean_u": float(np.mean([point["pixel"][0] for point in run])),
                                   "mean_v": float(np.mean([point["pixel"][1] for point in run])),
                                   "level_m": float((slice_index + 0.5) * spacing_m)})
        if candidates:
            records_by_slice.append({"slice_index": slice_index, "position_m": float(position),
                                     "mean_v": float(np.mean([item["mean_v"] for item in candidates])),
                                     "candidates": candidates, "dropped": row_dropped})
    return records_by_slice, dropped_fragments


def _order_rows(records_by_slice: list[dict[str, Any]], params: dict[str, Any],
                row_offset: int = 0) -> list[dict[str, Any]]:
    """Serpentine row order from the preferred corner; parity restarts at row_offset."""
    reverse_rows = "bottom" in params["start_corner"]
    records_by_slice.sort(key=lambda row: (row["mean_v"], row["position_m"]), reverse=reverse_rows)
    ordered_records = []
    for local_row, row in enumerate(records_by_slice):
        row_index = row_offset + local_row
        left_to_right = ("right" not in params["start_corner"]) ^ bool(local_row % 2)
        candidates = sorted(row["candidates"], key=lambda item: (item["mean_u"], item["component"]),
                            reverse=not left_to_right)
        for candidate in candidates:
            points = candidate["points"]
            endpoint_order = points[0]["pixel"][0] <= points[-1]["pixel"][0]
            if endpoint_order != left_to_right:
                points = list(reversed(points))
            candidate.update(points=points, row_index=row_index,
                             slice_index=row["slice_index"], position_m=row["position_m"],
                             row_dropped=row["dropped"])
            ordered_records.append(candidate)
    return ordered_records


def _image_axis_signs(origin, e1, e2, intrinsic, image_size):
    """Sign of image u along e1 and image v along e2 at the block origin; +1 when unprojectable.

    The reference image only decides which end of each sweep axis the start corner
    names; it never orders rows or segments."""
    signs = []
    base = _project(origin, intrinsic, image_size)
    for axis, index in ((e1, 0), (e2, 1)):
        moved = _project(origin + axis * 1e-3, intrinsic, image_size)
        delta = None if base is None or moved is None else moved[index] - base[index]
        signs.append(-1.0 if delta is not None and delta < 0 else 1.0)
    return signs[0], signs[1]


def _order_rows_by_frame(records_by_slice, params, row_offset, origin, e1, e2, intrinsic, image_size):
    """Block-mode order: rows by slice position along e2, segments by e1, serpentine parity.

    Ties on position fall back to slice_index then the smallest triangle index; image
    coordinates are never a sort key."""
    u_sign, v_sign = _image_axis_signs(origin, e1, e2, intrinsic, image_size)
    start_high_e2 = ("top" in params["start_corner"]) == (v_sign < 0)
    records_by_slice.sort(key=lambda row: (row["position_m"], row["slice_index"]), reverse=start_high_e2)
    ordered_records = []
    for local_row, row in enumerate(records_by_slice):
        row_index = row_offset + local_row
        left_to_right = ("right" not in params["start_corner"]) ^ bool(local_row % 2)
        increasing_e1 = left_to_right == (u_sign > 0)
        for candidate in row["candidates"]:
            xyz = np.asarray([point["xyz_m"] for point in candidate["points"]])
            candidate["mean_s"] = float(((xyz - origin) @ e1).mean())
            candidate["min_face"] = min(point["triangle_index"] for point in candidate["points"])
        candidates = sorted(row["candidates"], key=lambda item: (item["mean_s"], item["min_face"]),
                            reverse=not increasing_e1)
        for candidate in candidates:
            points = candidate["points"]
            first = float((np.asarray(points[0]["xyz_m"]) - origin) @ e1)
            last = float((np.asarray(points[-1]["xyz_m"]) - origin) @ e1)
            if (first <= last) != increasing_e1:
                points = list(reversed(points))
            candidate.update(points=points, row_index=row_index,
                             slice_index=row["slice_index"], position_m=row["position_m"],
                             row_dropped=row["dropped"])
            ordered_records.append(candidate)
    return ordered_records


def _build_items(ordered_records: list[dict[str, Any]], region: dict[str, Any],
                 intrinsic: np.ndarray, image_size: np.ndarray, step_m: float,
                 outer: dict[int, np.ndarray | None], origin: np.ndarray,
                 e1: np.ndarray, e2: np.ndarray) -> tuple[list[ScanSegment], list[Gap]]:
    """Locally numbered scan segments and the gap between each adjacent pair."""
    route_index = 0
    segments = []
    for segment_index, record in enumerate(ordered_records):
        points = []
        for point in record["points"]:
            points.append(dict(point, row_index=record["row_index"], route_index=route_index))
            route_index += 1
        segments.append(ScanSegment(segment_index, record["row_index"], points))
    gaps = []
    for index in range(len(segments) - 1):
        left, right = ordered_records[index], ordered_records[index + 1]
        reason, inferred = _gap_reason(left, right, region, intrinsic, image_size, step_m,
                                       outer, origin, e1, e2)
        before, after = segments[index].points[-1], segments[index + 1].points[0]
        gaps.append(Gap(index, reason, _gap_detail(inferred),
                        {"route_index": before["route_index"], "xyz_m": before["xyz_m"]},
                        {"route_index": after["route_index"], "xyz_m": after["xyz_m"]},
                        _direction(segments[index].points), segments[index].row_index,
                        segments[index + 1].row_index,
                        float(np.linalg.norm(np.asarray(after["xyz_m"]) - before["xyz_m"]))))
    return segments, gaps


def _segment_lengths(segments: list[ScanSegment]) -> list[float]:
    return [float(np.linalg.norm(np.diff(np.asarray([point["xyz_m"] for point in segment.points]), axis=0), axis=1).sum())
            for segment in segments]


def _row_report(segments: list[ScanSegment], lengths: list[float], gaps: list[Gap],
                block_index: int | None = None) -> list[dict[str, Any]]:
    report = []
    for row_index in sorted(set(segment.row_index for segment in segments)):
        selected = [segment for segment in segments if segment.row_index == row_index]
        z = np.asarray([point["xyz_m"][2] for segment in selected for point in segment.points])
        row = {"row_index": row_index, "segment_count": len(selected),
               "point_count": int(sum(len(segment.points) for segment in selected)),
               "scan_length_m": float(sum(lengths[segment.segment_index] for segment in selected)),
               "depth_min_m": float(z.min()), "depth_max_m": float(z.max()),
               "depth_span_m": float(z.max() - z.min()),
               "gap_count": sum(gap.row_index_from == row_index for gap in gaps)}
        if block_index is not None:
            row["block_index"] = block_index
        report.append(row)
    return report


def _level_segments(ordered_records: list[dict[str, Any]], segments: list[ScanSegment]) -> list[list[dict[str, Any]]]:
    return [[dict(point, surface_level_m=record["level_m"]) for point in segment.points]
            for record, segment in zip(ordered_records, segments)]


def _source_record(manifest_path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    from femto.planning.planning_state import region_signature
    region = manifest["region"]
    return {
        "region_export_dir": str(manifest_path.parent), "region_manifest": str(manifest_path),
        "region_manifest_sha256": _sha256(manifest_path),
        "mesh_file": manifest["files"]["mesh_file"],
        "mesh_file_sha256": manifest["file_sha256"][manifest["files"]["mesh_file"]],
        "geometry_sha256": manifest["geometry_sha256"],
        "region_id": region.get("annotation_id"), "region_signature": region_signature(region),
    }


def _depth_statistics(points: np.ndarray) -> dict[str, float]:
    return {"depth_min_m": float(points.min()), "depth_max_m": float(points.max()),
            "depth_span_m": float(points.max() - points.min())}


def _plan_single_surface(manifest_path: Path, manifest: dict[str, Any], mesh: trimesh.Trimesh,
                         params: dict[str, Any], check: Callable[[], None]) -> tuple[SurfaceTrajectory, list[dict[str, Any]]]:
    """v1: one sweep frame for the whole mesh, rows ordered globally across every component."""
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    region = manifest["region"]
    intrinsic = np.asarray(region["view_reference"]["undistorted_intrinsic"], dtype=np.float64)
    image_size = np.asarray(region["view_reference"]["image_size"], dtype=np.int64)
    origin, e1, e2, normal, sweep_frame = _pca_frame(mesh, params)
    components = face_components(faces)
    outer = _outer_loops(vertices, faces, components, origin, e1, e2)
    coordinates = (vertices - origin) @ e2
    low, high = float(coordinates.min()), float(coordinates.max())
    spacing_m = params["row_spacing_mm"] / 1000
    step_m = params["sample_step_mm"] / 1000
    positions = np.arange(low + spacing_m / 2, high + 1e-12, spacing_m)
    if not len(positions):
        raise ValueError("Region is narrower than one row spacing")
    records_by_slice, dropped_fragments = _slice_rows(mesh, None, components, origin, e2, positions, params,
                                                      intrinsic, image_size, check, {"points": 0})
    if not records_by_slice:
        raise ValueError("No supported scan segment was produced")
    ordered_records = _order_rows(records_by_slice, params)
    segments, gaps = _build_items(ordered_records, region, intrinsic, image_size, step_m, outer, origin, e1, e2)
    items: list[ScanSegment | Gap] = []
    for index, segment in enumerate(segments):
        items.append(segment)
        if index < len(gaps):
            items.append(gaps[index])
    gap_counts = Counter(gap.reason for gap in gaps)
    lengths = _segment_lengths(segments)
    all_z = np.asarray([point["xyz_m"][2] for segment in segments for point in segment.points])
    row_report = _row_report(segments, lengths, gaps)
    from femto.planning.surface_spacing_report import spacing_report as measure_spacing
    check()
    spacing_mesh = SimpleNamespace(vertices=vertices, triangles=faces)
    spacing = measure_spacing(spacing_mesh, {"triangle_indices": list(range(len(faces)))},
                              {"segments": _level_segments(ordered_records, segments)}, params["row_spacing_mm"],
                              params["row_spacing_mm"] * params["spacing_tolerance"], check)
    spacing["spacing_semantics"] = "parallel_plane_spacing_not_uniform_surface_spacing"
    spacing["warning_threshold_mm"] = params["row_spacing_mm"] * (1 + params["spacing_tolerance"])
    planner = {"kind": "mesh_slice", "algorithm_version": ALGORITHM_VERSION,
               "parameters": params, "sweep_frame": sweep_frame}
    route_index = sum(len(segment.points) for segment in segments)
    trajectory = SurfaceTrajectory(
        manifest["origin_reference"], planner, _source_record(manifest_path, manifest), items,
        {"segment_count": len(segments), "gap_count": len(segments) - 1,
         "gap_counts": dict(sorted(gap_counts.items())), "point_count": route_index,
         "scan_length_m": float(sum(lengths)), "row_count": len(row_report),
         **_depth_statistics(all_z)},
        {"counts": dict(sorted(gap_counts.items())), "dropped_fragments": dropped_fragments,
         "dropped_fragment_count": len(dropped_fragments)},
        spacing, row_report, manifest.get("data_layers"))
    validate_trajectory(trajectory, mesh=mesh)
    return trajectory, ordered_records


def _eligible_turns(ordered_records: list[dict[str, Any]]) -> set[int]:
    return {index for index, (left, right) in enumerate(zip(ordered_records, ordered_records[1:]))
            if left["component"] == right["component"] and
            abs(left["slice_index"] - right["slice_index"]) == 1 and
            left["row_index"] != right["row_index"]}


def _block_descriptors(vertices: np.ndarray, faces: np.ndarray, components: np.ndarray,
                       intrinsic: np.ndarray, image_size: np.ndarray, corner: str) -> list[dict[str, Any]]:
    """Blocks are mesh connectivity components; visit order is reading order from the start corner."""
    descriptors = []
    for component in sorted(set(int(v) for v in components)):
        ids = np.flatnonzero(components == component)
        triangles = vertices[faces[ids]]
        area = float(np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                             triangles[:, 2] - triangles[:, 0]), axis=1).sum() / 2)
        bounds, center = _pixel_extent(vertices[np.unique(faces[ids])], intrinsic, image_size)
        if bounds is None:
            key = (1, 0.0, 0.0, component)
        else:
            key = (0, bounds[1] if "top" in corner else -bounds[3],
                   bounds[0] if "left" in corner else -bounds[2], component)
        descriptors.append({"component_id": component, "face_ids": ids, "area_m2": area,
                            "pixel_bounds": bounds, "center_pixel": center, "key": key})
    descriptors.sort(key=lambda item: item["key"])
    return descriptors


def _plan_block(vertices: np.ndarray, faces: np.ndarray, descriptor: dict[str, Any], components: np.ndarray,
                params: dict[str, Any], region: dict[str, Any], intrinsic: np.ndarray, image_size: np.ndarray,
                check: Callable[[], None], budget: dict[str, int], row_offset: int,
                block_index: int) -> dict[str, Any]:
    """Plan one block alone: its own sweep frame, rows, gaps and boundary row turns."""
    from femto.planning.surface_row_turns import BoundaryTurns, connect_segments
    ids = descriptor["face_ids"]
    component = descriptor["component_id"]
    spacing_m = params["row_spacing_mm"] / 1000
    step_m = params["sample_step_mm"] / 1000
    block_mesh = trimesh.Trimesh(vertices=vertices, faces=faces[ids], process=False)
    origin, e1, e2, _normal, frame = _pca_frame(block_mesh, params)
    outer = {component: _outer_loop(vertices, faces[ids], origin, e1, e2)}
    coordinates = (vertices[np.unique(faces[ids])] - origin) @ e2
    positions = np.arange(float(coordinates.min()) + spacing_m / 2, float(coordinates.max()) + 1e-12, spacing_m)
    if not len(positions):
        raise Unplannable("narrower_than_row_spacing", "Block is narrower than one row spacing")
    records_by_slice, dropped_fragments = _slice_rows(block_mesh, ids, components, origin, e2, positions, params,
                                                      intrinsic, image_size, check, budget, block_index)
    if not records_by_slice:
        raise Unplannable("no_supported_segment", "No supported scan segment was produced",
                          details={"dropped_fragments": dropped_fragments})
    ordered_records = _order_rows_by_frame(records_by_slice, params, row_offset, origin, e1, e2,
                                           intrinsic, image_size)
    segments, gaps = _build_items(ordered_records, region, intrinsic, image_size, step_m, outer, origin, e1, e2)
    # The boundary graph is built from this block's faces only, indexed in the whole PLY.
    # Same router and same frozen snap tolerance as v2; nothing is relaxed per block.
    router = BoundaryTurns(SimpleNamespace(vertices=vertices, faces=faces), region, frame, check, face_ids=ids)
    eligible = _eligible_turns(ordered_records)
    chains, kept, connected, connector_length, added = connect_segments(
        segments, gaps, eligible, router, step_m, check, e2=e2, used=budget["points"])
    budget["points"] += added
    # Diagnostic only: why a row turn stayed a gap. Never changes any decision.
    diagnosed = []
    failure_counts = Counter()
    for gap in kept:
        if gap.reason != "row_turn":
            diagnosed.append(gap)
            continue
        check()
        if gap.gap_index not in eligible:
            why = "non_adjacent_slice"
        else:
            why = router.diagnose(segments[gap.gap_index].points[-1], segments[gap.gap_index + 1].points[0], step_m, e2)
        failure_counts[why] += 1
        diagnosed.append(replace(gap, detail=dict(gap.detail, row_turn_failure_reason=why)))
    return {"frame": frame, "ordered_records": ordered_records, "segments": segments, "chains": chains,
            "kept": diagnosed, "connected": connected, "connector_length_m": connector_length,
            "row_turn_failure_counts": dict(sorted(failure_counts.items())),
            "dropped_fragments": dropped_fragments, "lengths": _segment_lengths(segments)}


def _merge_spacing(reports: list[dict[str, Any]], params: dict[str, Any], budget: dict[str, int],
                   scan_rows: list[int]) -> dict[str, Any]:
    from femto.planning.surface_spacing_report import spacing_distribution
    target = params["row_spacing_mm"]
    rows = [row for report in reports for row in report["measurements"]]
    counts = {status: sum(report["counts"][status] for report in reports)
              for status in ("within_bounds", "outside_bounds", "uncertain_bounds", "unmeasurable")}
    values = [row["surface_graph_distance_mm"] for row in rows if row["surface_graph_distance_mm"] is not None]
    outlier_threshold = 2 * target
    effective = [value for value in values if value <= outlier_threshold]
    template = reports[0]
    return dict(schema=template["schema"], target_spacing_mm=target, tolerance_mm=template["tolerance_mm"],
                raw_spacing_statistics=spacing_distribution(values, target),
                effective_spacing_statistics=spacing_distribution(effective, target),
                spacing_outlier_threshold_mm=outlier_threshold,
                spacing_outlier_count=len(values) - len(effective),
                spacing_outlier_ratio=(len(values) - len(effective)) / len(values) if values else None,
                spacing_outlier_semantics="graph_upper_bounds_above_2x_nominal_spacing_usually_cross_holes_or_missing_surface_not_row_pitch",
                method=template["method"], direction=template["direction"], sampling=template["sampling"],
                sample_count=len(rows),
                status="within_sampled_bounds" if rows and counts["within_bounds"] == len(rows) else "not_fully_verified",
                counts=counts, distribution=spacing_distribution(values, target), measurements=rows,
                budget=dict(template["budget"], expansions=budget["expansions"], distance_pairs=budget["distance_pairs"]),
                limitations=list(template["limitations"]) + ["每块单独测量，只与同一块内的前一扫描层比较；块之间不测量间距。"],
                spacing_semantics="parallel_plane_spacing_not_uniform_surface_spacing",
                warning_threshold_mm=target * (1 + params["spacing_tolerance"]),
                scan_segment_rows=scan_rows,
                blocks=[dict(block_index=report["block_index"], sample_count=report["sample_count"],
                             counts=report["counts"], status=report["status"], distribution=report["distribution"])
                        for report in reports])


def _plan_blocks(manifest_path: Path, manifest: dict[str, Any], mesh: trimesh.Trimesh,
                 params: dict[str, Any], check: Callable[[], None]) -> SurfaceTrajectory:
    """v3: every connectivity component is planned alone and visited one after another."""
    from femto.planning.surface_row_turns import chain_items
    from femto.planning.surface_spacing_report import spacing_report as measure_spacing
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    region = manifest["region"]
    intrinsic = np.asarray(region["view_reference"]["undistorted_intrinsic"], dtype=np.float64)
    image_size = np.asarray(region["view_reference"]["image_size"], dtype=np.int64)
    components = face_components(faces)
    descriptors = _block_descriptors(vertices, faces, components, intrinsic, image_size, params["start_corner"])
    budget = {"points": 0}
    spacing_budget = {"expansions": 0, "distance_pairs": 0}
    offsets = {"row": 0, "route": 0, "segment": 0, "gap": 0, "pre_segment": 0}
    items: list[ScanSegment | Gap] = []
    blocks: list[dict[str, Any]] = []
    unplannable: list[dict[str, Any]] = []
    dropped_fragments: list[dict[str, Any]] = []
    row_report: list[dict[str, Any]] = []
    spacing_reports: list[dict[str, Any]] = []
    scan_rows: list[int] = []
    final_gaps: list[Gap] = []
    spacing_mesh = SimpleNamespace(vertices=vertices, triangles=faces)
    for block_index, descriptor in enumerate(descriptors):
        check()
        ids = descriptor["face_ids"]
        record = {"block_index": block_index, "component_id": descriptor["component_id"],
                  "face_count": int(len(ids)), "face_index_min": int(ids.min()), "face_index_max": int(ids.max()),
                  "face_indices_sha256": face_list_sha256(ids), "area_m2": descriptor["area_m2"],
                  "pixel_bounds": descriptor["pixel_bounds"], "center_pixel": descriptor["center_pixel"]}
        try:
            planned = _plan_block(vertices, faces, descriptor, components, params, region, intrinsic,
                                  image_size, check, budget, offsets["row"], block_index)
        except Unplannable as error:
            record.update(status="unplannable", reason=error.reason, message=str(error))
            if error.details:
                dropped_fragments.extend(error.details.get("dropped_fragments", []))
                record["dropped_fragment_count"] = len(error.details.get("dropped_fragments", []))
            blocks.append(record)
            unplannable.append({key: record[key] for key in
                                ("block_index", "component_id", "reason", "message", "face_count", "area_m2")})
            continue
        inter_gap = None
        if items:
            inter_gap = offsets["gap"]
            offsets["gap"] += 1
        block_items, route_end = chain_items(planned["chains"], planned["kept"], route_start=offsets["route"],
                                             segment_start=offsets["segment"], gap_start=offsets["gap"],
                                             block_index=block_index)
        segments_final = [item for item in block_items if isinstance(item, ScanSegment)]
        kept = [item for item in block_items if isinstance(item, Gap)]
        if inter_gap is not None:
            previous = items[-1]
            assert isinstance(previous, ScanSegment)
            first = segments_final[0]
            before, after = previous.points[-1], first.points[0]
            gap = Gap(inter_gap, "disconnected_surface", _gap_detail("not_applicable"),
                      {"route_index": before["route_index"], "xyz_m": before["xyz_m"]},
                      {"route_index": after["route_index"], "xyz_m": after["xyz_m"]},
                      _direction(previous.points), previous.row_index, first.row_index,
                      float(np.linalg.norm(np.asarray(after["xyz_m"]) - before["xyz_m"])))
            items.append(gap)
            final_gaps.append(gap)
        items.extend(block_items)
        final_gaps.extend(kept)
        lengths = planned["lengths"]
        scan_length = float(sum(lengths))
        depth = np.asarray([point["xyz_m"][2] for segment in segments_final for point in segment.points])
        row_report.extend(_row_report(planned["segments"], lengths, [], block_index))
        check()
        report = measure_spacing(spacing_mesh, {"triangle_indices": [int(v) for v in ids]},
                                 {"segments": _level_segments(planned["ordered_records"], planned["segments"])},
                                 params["row_spacing_mm"], params["row_spacing_mm"] * params["spacing_tolerance"],
                                 check, budget=spacing_budget)
        report["measurements"] = [dict(row, chain_index=row["chain_index"] + offsets["pre_segment"], block_index=block_index)
                                  for row in report["measurements"]]
        report["block_index"] = block_index
        spacing_reports.append(report)
        scan_rows.extend(segment.row_index for segment in planned["segments"])
        dropped_fragments.extend(planned["dropped_fragments"])
        row_count = len({segment.row_index for segment in planned["segments"]})
        record.update(status="planned", sweep_frame=planned["frame"],
                      segment_index_first=segments_final[0].segment_index, segment_count=len(segments_final),
                      gap_count=len(kept), gap_counts=dict(sorted(Counter(g.reason for g in kept).items())),
                      point_count=route_end - offsets["route"], row_index_first=offsets["row"], row_count=row_count,
                      scan_length_m=scan_length, connector_length_m=planned["connector_length_m"],
                      connected_row_turns=planned["connected"], row_turn_failure_counts=planned["row_turn_failure_counts"],
                      dropped_fragment_count=len(planned["dropped_fragments"]),
                      **_depth_statistics(depth))
        blocks.append(record)
        offsets.update(row=offsets["row"] + row_count, route=route_end,
                       segment=segments_final[-1].segment_index + 1, gap=offsets["gap"] + len(kept),
                       pre_segment=offsets["pre_segment"] + len(planned["segments"]))
    planned_blocks = [block for block in blocks if block["status"] == "planned"]
    if not planned_blocks:
        raise Unplannable("all_blocks_unplannable",
                          "No block can be planned: " + "; ".join(
                              f"block {b['block_index'] + 1} ({b['face_count']} faces): {b['reason']}" for b in blocks),
                          details={"blocks": blocks})
    for row in row_report:
        row["gap_count"] = sum(gap.row_index_from == row["row_index"] for gap in final_gaps)
    segments_all = [item for item in items if isinstance(item, ScanSegment)]
    gap_counts = Counter(gap.reason for gap in final_gaps)
    all_z = np.asarray([point["xyz_m"][2] for segment in segments_all for point in segment.points])
    scan_length = float(sum(block["scan_length_m"] for block in planned_blocks))
    connector_length = float(sum(block["connector_length_m"] for block in planned_blocks))
    try:
        display_frame = dict(_pca_frame(mesh, dict(params, row_direction="auto"))[4],
                             role="display_only_whole_mesh_frame")
    except Unplannable:
        display_frame = dict(planned_blocks[0]["sweep_frame"], role="display_only_first_planned_block_frame")
    planner = {"kind": "mesh_slice", "algorithm_version": BLOCK_ALGORITHM,
               "parameters": dict(params, connect_rows=True, block_planning=True), "sweep_frame": display_frame}
    statistics = {
        "segment_count": len(segments_all), "gap_count": len(final_gaps),
        "gap_counts": dict(sorted(gap_counts.items())), "point_count": offsets["route"],
        "scan_length_m": scan_length, "connector_length_m": connector_length,
        "connected_row_turns": int(sum(block["connected_row_turns"] for block in planned_blocks)),
        "total_length_m": scan_length + connector_length, "row_count": len(row_report),
        **_depth_statistics(all_z),
        "block_count": len(blocks), "plannable_block_count": len(planned_blocks),
        "unplannable_block_count": len(blocks) - len(planned_blocks),
        "inter_block_gap_count": int(gap_counts.get("disconnected_surface", 0)),
        "coverage_status": "complete" if len(planned_blocks) == len(blocks) else "partial",
        "unplannable_face_count": int(sum(block["face_count"] for block in unplannable)),
        "unplannable_area_m2": float(sum(block["area_m2"] for block in unplannable)),
    }
    block_record = {
        "schema": BLOCKS_SCHEMA,
        "block_definition": "faces sharing an edge in the bound region PLY; no vertex welding, no distance merging",
        "order_rule": {"visit_order": "reference_image_reading_order_from_start_corner",
                       "start_corner": params["start_corner"],
                       "keys": "top: v_min ascending / bottom: v_max descending, then left: u_min ascending / right: u_max descending",
                       "unprojectable_blocks": "after all projectable blocks",
                       "tie_break": "component_id ascending (component ids follow the smallest face index)"},
        "items": blocks,
    }
    trajectory = SurfaceTrajectory(
        manifest["origin_reference"], planner, _source_record(manifest_path, manifest), items, statistics,
        {"counts": dict(sorted(gap_counts.items())), "dropped_fragments": dropped_fragments,
         "dropped_fragment_count": len(dropped_fragments), "unplannable_blocks": unplannable,
         "row_turn_failure_counts": dict(sorted(Counter(
             gap.detail["row_turn_failure_reason"] for gap in final_gaps
             if gap.reason == "row_turn" and "row_turn_failure_reason" in gap.detail).items()))},
        _merge_spacing(spacing_reports, params, spacing_budget, scan_rows), row_report,
        manifest.get("data_layers"), None, block_record)
    validate_trajectory(trajectory, mesh=mesh)
    return trajectory


def plan_mesh_slice(export_dir_or_manifest: str | Path, *, row_spacing_mm: float = 5.0,
                    sample_step_mm: float = 1.0, row_direction: Any = "auto",
                    start_corner: str = "top_left", min_segment_mm: float = 2.0,
                    spacing_tolerance: float = 0.5,
                    inplane_ambiguity_ratio_max: float = 0.8,
                    planarity_ratio_max: float = 0.5,
                    normal_view_angle_max_deg: float = 60.0,
                    adaptive_rows: bool = False,
                    connect_rows: bool = False,
                    block_planning: bool = False,
                    check: Callable[[], None] = lambda: None) -> SurfaceTrajectory:
    """Generate only measured-mesh scan samples; no gap interpolation or motion."""
    params = _parameters(locals())
    if type(connect_rows) is not bool:
        raise ValueError('connect_rows must be boolean')
    if type(block_planning) is not bool:
        raise ValueError('block_planning must be boolean')
    if block_planning and not connect_rows:
        raise ValueError('block_planning requires connect_rows=True')
    check()
    manifest_path, manifest, mesh = _load_input(export_dir_or_manifest)
    if block_planning:
        return _plan_blocks(manifest_path, manifest, mesh, params, check)
    trajectory, ordered_records = _plan_single_surface(manifest_path, manifest, mesh, params, check)
    if connect_rows:
        from femto.planning.surface_row_turns import connect_row_turns
        return connect_row_turns(trajectory, mesh, manifest["region"], _eligible_turns(ordered_records), check)
    return trajectory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a W0 measured-mesh surface trajectory")
    parser.add_argument("source", help="Region export directory or region_N.json")
    parser.add_argument("--output")
    parser.add_argument("--row-mm", type=float, default=5.0)
    parser.add_argument("--step-mm", type=float, default=1.0)
    parser.add_argument("--direction", default="auto")
    parser.add_argument("--start-corner", default="top_left",
                        choices=("top_left", "top_right", "bottom_left", "bottom_right"))
    args = parser.parse_args(argv)
    direction: Any = args.direction
    if direction not in ("auto", "w0_x", "w0_y"):
        direction = float(direction)
    from femto.planning.trajectory_export import export_blocked, export_trajectory
    output = Path(args.output).resolve() if args.output else (Path(args.source).resolve()
             if Path(args.source).is_dir() else Path(args.source).resolve().parent)
    try:
        trajectory = plan_mesh_slice(args.source, row_spacing_mm=args.row_mm,
                                     sample_step_mm=args.step_mm, row_direction=direction,
                                     start_corner=args.start_corner, connect_rows=True,
                                     block_planning=True)
        folder = export_trajectory(trajectory, output)
    except (OSError, ValueError, TypeError, KeyError, IndexError) as error:
        folder = export_blocked(output, args.source, error, details=getattr(error, "details", None))
        print(folder)
        return 2
    print(folder)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
