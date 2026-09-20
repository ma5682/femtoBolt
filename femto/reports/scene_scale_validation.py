"""Real-world scale and coordinate validation for reconstructed Femto geometry.

This module answers one narrow question: *does a distance measured on the
reconstructed geometry match a physically measured distance in the room?*  It
therefore reads the measurable geometry layer — the fused point cloud or the
TSDF mesh — and never the Gaussian model.  Trained Gaussians are a rendering
representation; their centres are optimized for image likelihood, not for metric
surface position, so using them as engineering geometry would silently overstate
what the system can measure.

Every measurement records its geometry source, both world coordinates, the
coordinate frame, the unit and the dataset lineage, per the project rule that no
bare XYZ may exist without a frame.

Scope: this validates metric scale and coordinate consistency of the
reconstruction pipeline only.  It is not robot positioning accuracy, not a
clinical targeting accuracy, and it does not authorize any treatment claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import open3d as o3d

from femto.geometry.fusion import read_geometry_safe
from femto.io_utils import load_json, write_json
from femto.dataset.scene_data_kind import classify_dataset_data_kind


from femto.paths import REPO_ROOT as ROOT
OUTPUTS_ROOT = ROOT / "outputs"
SPEC_SCHEMA = "femto-scale-validation-spec-v1"
REPORT_SCHEMA = "femto-scale-validation-v1"
COORDINATE_FRAME = "world (source T_WC world frame; Orbbec/OpenCV RDF: +X right, +Y down, +Z forward)"
UNIT = "metre"
TRANSFORM_DIRECTION = "p_W = T_WC * p_C; points below are already expressed in W"
SCOPE_DISCLAIMER = (
    "research geometry scale validation only. Validates metric scale and coordinate consistency. "
    "Not a robot positioning accuracy, not a clinical targeting accuracy, and not a treatment claim."
)
GEOMETRY_SOURCES = {
    "fused_point_cloud": "scene_fused_points_m.ply",
    "tsdf_mesh": "scene_tsdf_mesh_m.ply",
}
DEFAULT_SNAP_RADIUS_MM = 15.0


def resolve_reconstruction(value: Path | None) -> Path:
    """Accept a reconstruction directory, a complete-demo root, or fall back to the latest."""
    if value is None:
        marker = OUTPUTS_ROOT / "latest_complete_demo.txt"
        if not marker.is_file():
            raise ValueError("No completed reconstruction is available")
        value = Path(marker.read_text(encoding="utf-8-sig").strip())
    candidate = Path(value).resolve()
    if (candidate / "reconstruction_report.json").is_file():
        return candidate
    nested = candidate / "reconstruction"
    if (nested / "reconstruction_report.json").is_file():
        return nested
    raise ValueError(f"Not a reconstruction directory: {candidate}")


def load_measurement_geometry(reconstruction: Path, source: str) -> tuple[np.ndarray, Path]:
    """Return world-frame vertices in metres from the measurable geometry layer."""
    if source not in GEOMETRY_SOURCES:
        raise ValueError(f"Unknown geometry source: {source}; expected one of {sorted(GEOMETRY_SOURCES)}")
    path = reconstruction / GEOMETRY_SOURCES[source]
    if source == "tsdf_mesh":
        report = load_json(reconstruction / "reconstruction_report.json")
        mesh_path = report.get("outputs", {}).get("mesh", {}).get("ply")
        if mesh_path:
            path = Path(mesh_path)
    if not path.is_file():
        raise ValueError(f"Missing {source} geometry: {path}")
    geometry = read_geometry_safe(path, mesh=source == "tsdf_mesh")
    points = np.asarray(geometry.vertices if source == "tsdf_mesh" else geometry.points, dtype=np.float64)
    if not len(points) or not np.isfinite(points).all():
        raise ValueError(f"{path} contains no geometry")
    return points, path


class _NearestSurface:
    """Snap requested world points onto measured geometry, never onto empty space."""

    def __init__(self, points: np.ndarray) -> None:
        self.points = points
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        self.index = o3d.geometry.KDTreeFlann(cloud)

    def snap(self, requested_m: np.ndarray, radius_mm: float) -> dict[str, Any]:
        if requested_m.shape != (3,) or not np.isfinite(requested_m).all():
            raise ValueError("World landmark must contain three finite metre coordinates")
        if not np.isfinite(radius_mm) or radius_mm <= 0:
            raise ValueError("snap_radius_mm must be positive and finite")
        radius_m = float(radius_mm) / 1000.0
        count, indices, _ = self.index.search_radius_vector_3d(requested_m, radius_m)
        if count < 1:
            nearest_count, nearest_index, nearest_sq = self.index.search_knn_vector_3d(requested_m, 1)
            nearest_mm = float(np.sqrt(nearest_sq[0]) * 1000.0) if nearest_count else None
            raise ValueError(
                f"No reconstructed surface within {radius_mm:.1f} mm of {requested_m.tolist()} "
                f"(nearest measured point is {nearest_mm:.1f} mm away)"
                if nearest_mm is not None
                else f"No reconstructed surface within {radius_mm:.1f} mm of {requested_m.tolist()}"
            )
        neighbours = self.points[np.asarray(indices, dtype=np.int64)]
        # Averaging a neighbourhood moves edge landmarks inward or into empty space.
        # Use one measured vertex; never synthesize a measurement location.
        snapped = neighbours[np.argmin(np.linalg.norm(neighbours - requested_m, axis=1))]
        centered=neighbours-neighbours.mean(axis=0)
        if len(neighbours)>=3:
            _,vectors=np.linalg.eigh(centered.T@centered/max(1,len(neighbours)-1))
            normal=vectors[:,0];normal_spread=float(np.ptp(centered@normal)*1000)
        else:normal_spread=0.0
        return {
            "requested_world_m": [float(value) for value in requested_m],
            "snapped_world_m": [float(value) for value in snapped],
            "snap_distance_mm": float(np.linalg.norm(snapped - requested_m) * 1000.0),
            "snap_radius_mm": float(radius_mm),
            "support_points": int(count),
            "neighbourhood_std_mm": float(np.linalg.norm(neighbours.std(axis=0)) * 1000.0),
            "normal_spread_mm": normal_spread,
            "normal_spread_warning": bool(normal_spread>5.0),
            "coordinate_frame": COORDINATE_FRAME,
            "unit": UNIT,
        }


def _lineage(reconstruction: Path) -> dict[str, Any]:
    report = load_json(reconstruction / "reconstruction_report.json")
    lineage: dict[str, Any] = {
        "reconstruction": str(reconstruction),
        "source_dataset": report.get("source_dataset"),
        "frame_count": report.get("frame_count"),
        "reconstruction_parameters": report.get("parameters"),
        "fused_point_count": (report.get("metrics") or {}).get("fused_point_count"),
        "mesh_triangles": (report.get("metrics") or {}).get("mesh_triangles"),
    }
    summary_path = reconstruction.parent / "demo_summary.json"
    if summary_path.is_file():
        summary = load_json(summary_path)
        source = summary.get("source_dataset") or report.get("source_dataset")
        source_path = Path(source) if source else None
        source_kind = classify_dataset_data_kind(source_path)
        if summary.get("mode") == "synthetic" or summary.get("data_kind") == "synthetic":
            data_kind = "synthetic"
        else:
            # A summary may preserve a prior claim, but only the source camera
            # manifest can establish that the data came from a Femto Bolt.
            data_kind = source_kind
        lineage["data_kind"] = data_kind
        lineage["pose_stage"] = summary.get("pose_stage")
        lineage["fixed_camera"] = summary.get("fixed_camera")
        lineage["raw_dataset"] = summary.get("raw_dataset")
    return lineage


def validate_scale(
    reconstruction: Path,
    measurements: list[dict[str, Any]],
    *,
    geometry_source: str = "auto",
) -> dict[str, Any]:
    """Measure each requested point pair on the reconstruction and score it."""
    if not measurements:
        raise ValueError("At least one measurement is required")
    lineage = _lineage(reconstruction)
    if geometry_source=='auto':
        report=load_json(reconstruction/'reconstruction_report.json')
        geometry_source='tsdf_mesh' if report.get('local_capture',{}).get('measured_only') else 'fused_point_cloud'
    points, geometry_path = load_measurement_geometry(reconstruction, geometry_source)
    data_kind = lineage.get("data_kind", "unknown")
    geometry_sha256 = hashlib.sha256(geometry_path.read_bytes()).hexdigest()
    surface = _NearestSurface(points)
    results: list[dict[str, Any]] = []
    for item in measurements:
        ground_truth_mm = float(item["ground_truth_distance_mm"])
        if not np.isfinite(ground_truth_mm) or ground_truth_mm <= 0.0:
            raise ValueError("ground_truth_distance_mm must be a positive, finite millimetre value")
        radius_mm = float(item.get("snap_radius_mm", DEFAULT_SNAP_RADIUS_MM))
        point_a = surface.snap(np.asarray(item["point_a_world_m"], dtype=np.float64), radius_mm)
        point_b = surface.snap(np.asarray(item["point_b_world_m"], dtype=np.float64), radius_mm)
        measured_mm = float(
            np.linalg.norm(
                np.asarray(point_b["snapped_world_m"]) - np.asarray(point_a["snapped_world_m"])
            )
            * 1000.0
        )
        signed_error_mm = measured_mm - ground_truth_mm
        absolute_error_mm = abs(signed_error_mm)
        results.append(
            {
                "label": str(item.get("label", f"measurement_{len(results) + 1}")),
                "description": item.get("description"),
                "ground_truth_distance_mm": ground_truth_mm,
                "ground_truth_source": item.get("ground_truth_source", "manually measured physical object"),
                "measured_distance_mm": measured_mm,
                "absolute_error_mm": absolute_error_mm,
                "signed_error_mm": signed_error_mm,
                "relative_error_percent": absolute_error_mm / ground_truth_mm * 100.0,
                "geometry_source": geometry_source,
                "geometry_file": str(geometry_path),
                "coordinate_frame": COORDINATE_FRAME,
                "unit": UNIT,
                "transform_direction": TRANSFORM_DIRECTION,
                "point_a": point_a,
                "point_b": point_b,
                "point_a_world_m": point_a["snapped_world_m"],
                "point_b_world_m": point_b["snapped_world_m"],
                "source_geometry_path": str(geometry_path.resolve()),
                "source_geometry_sha256": geometry_sha256,
                "dataset_path": lineage["source_dataset"],
                "measurement_method": "Euclidean distance between nearest measured world vertices; no neighbourhood averaging",
            }
        )
    absolute_errors = [abs(item["absolute_error_mm"]) for item in results]
    relative_errors = [abs(item["relative_error_percent"]) for item in results]
    return {
        "schema": REPORT_SCHEMA,
        "data_kind": data_kind,
        "created_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "geometry_source": geometry_source,
        "geometry_file": str(geometry_path),
        "geometry_note": (
            ("Local measured-only workflow defaults to the supported TSDF mesh surface; fused point cloud is a "
             "multi-view observation union and remains available explicitly for diagnostics. " if geometry_source=='tsdf_mesh' else "")+
            "The trained Gaussian model is a rendering representation and is deliberately not used as measurement geometry."
        ),
        "coordinate_frame": COORDINATE_FRAME,
        "unit": UNIT,
        "transform_direction": TRANSFORM_DIRECTION,
        "lineage": lineage,
        "source_geometry_sha256": geometry_sha256,
        "status": "measured_requires_review",
        "accepted": None,
        "acceptance_note": "No engineering tolerance specified; not an automatic scale PASS.",
        "measurements": results,
        "summary": {
            "measurement_count": len(results),
            "max_absolute_error_mm": float(np.max(absolute_errors)),
            "mean_absolute_error_mm": float(np.mean(absolute_errors)),
            "max_relative_error_percent": float(np.max(relative_errors)),
            "mean_relative_error_percent": float(np.mean(relative_errors)),
        },
        "scope_disclaimer": SCOPE_DISCLAIMER,
    }


def load_spec(path: Path) -> tuple[list[dict[str, Any]], str]:
    payload = load_json(path)
    if payload.get("schema") != SPEC_SCHEMA:
        raise ValueError(f"Measurement spec must declare schema {SPEC_SCHEMA}")
    measurements = payload.get("measurements")
    if not isinstance(measurements, list) or not measurements:
        raise ValueError("Measurement spec contains no measurements")
    return measurements, str(payload.get("geometry_source", "auto"))


def pick_points(reconstruction: Path, geometry_source: str) -> list[list[float]]:
    """Open a picking window so an operator can designate landmarks on real geometry.

    Uses the Open3D editing visualizer: shift + left click marks a point,
    closing the window returns the picks.  Kept separate from :func:`validate_scale`
    so the scoring path stays headless and reproducible from a spec file.
    """
    points, _ = load_measurement_geometry(reconstruction, geometry_source)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.paint_uniform_color([0.7, 0.75, 0.8])
    visualizer = o3d.visualization.VisualizerWithEditing()
    visualizer.create_window(window_name="Shift+Click landmarks, then close")
    visualizer.add_geometry(cloud)
    visualizer.run()
    visualizer.destroy_window()
    return [points[index].tolist() for index in visualizer.get_picked_points()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate real-world scale on reconstructed Femto geometry")
    parser.add_argument("--reconstruction", type=Path, help="Reconstruction directory or complete-demo root")
    parser.add_argument("--spec", type=Path, help=f"Measurement spec JSON ({SPEC_SCHEMA})")
    parser.add_argument(
        "--geometry-source", choices=('auto',*GEOMETRY_SOURCES), default="auto"
    )
    parser.add_argument("--output", type=Path, help="Report path; defaults into the reconstruction directory")
    parser.add_argument("--pick", action="store_true", help="Interactively pick landmarks and write a spec")
    parser.add_argument("--ground-truth-mm", type=float, help="--pick: physically measured distance")
    parser.add_argument("--label", default="known_object", help="--pick: measurement label")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        reconstruction = resolve_reconstruction(args.reconstruction)
        if args.pick:
            if args.ground_truth_mm is None:
                raise ValueError("--pick requires --ground-truth-mm")
            picked = pick_points(reconstruction, args.geometry_source)
            if len(picked) != 2:
                raise ValueError(f"Pick exactly two landmarks; got {len(picked)}")
            spec = {
                "schema": SPEC_SCHEMA,
                "geometry_source": args.geometry_source,
                "measurements": [
                    {
                        "label": args.label,
                        "ground_truth_distance_mm": float(args.ground_truth_mm),
                        "point_a_world_m": picked[0],
                        "point_b_world_m": picked[1],
                    }
                ],
            }
            spec_path = args.spec or (reconstruction / "scale_validation_spec.json")
            write_json(spec_path, spec)
            measurements, geometry_source = spec["measurements"], args.geometry_source
        else:
            if args.spec is None:
                raise ValueError("Provide --spec with measured landmarks, or use --pick")
            measurements, geometry_source = load_spec(args.spec)
        report = validate_scale(reconstruction, measurements, geometry_source=geometry_source)
        output = args.output or (reconstruction / "scale_validation.json")
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, report)
        report["report_path"] = str(output)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "blocked", "error": str(error)}, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
