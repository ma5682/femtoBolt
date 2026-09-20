from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import open3d as o3d

import femto.reports.scene_scale_validation as scale


def _build_reconstruction(root: Path, points: np.ndarray) -> Path:
    """Write the minimum reconstruction layout the validator consumes."""
    reconstruction = root / "reconstruction"
    reconstruction.mkdir(parents=True)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(str(reconstruction / "scene_fused_points_m.ply"), cloud)
    (reconstruction / "reconstruction_report.json").write_text(
        json.dumps(
            {
                "schema": "femto-scene-reconstruction-v1",
                "source_dataset": str(root / "dataset"),
                "frame_count": 14,
                "parameters": {"voxel_m": 0.01},
                "metrics": {"fused_point_count": int(len(points)), "mesh_triangles": 100},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (root / "demo_summary.json").write_text(
        json.dumps({"pose_stage": "open3d_sequential_hybrid_rgbd_odometry", "fixed_camera": False}),
        encoding="utf-8",
    )
    return reconstruction


def _plate(centre: np.ndarray, half: float = 0.01, step: float = 0.0025) -> np.ndarray:
    grid = np.arange(-half, half + 1e-9, step)
    x, y = np.meshgrid(grid, grid)
    patch = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    return patch + centre


class ScaleValidationTests(unittest.TestCase):
    def test_short_measurement_reports_positive_absolute_error(self) -> None:
        points = np.array([[0., 0., 1.], [.297, 0., 1.]])
        with TemporaryDirectory() as temporary:
            reconstruction = _build_reconstruction(Path(temporary), points)
            report = scale.validate_scale(reconstruction, [{"ground_truth_distance_mm": 300.,
                "point_a_world_m": points[0].tolist(), "point_b_world_m": points[1].tolist()}])
        measurement = report["measurements"][0]
        self.assertAlmostEqual(measurement["absolute_error_mm"], 3.)
        self.assertAlmostEqual(measurement["signed_error_mm"], -3.)
        self.assertAlmostEqual(measurement["relative_error_percent"], 1.)
        self.assertEqual(measurement["point_b_world_m"], points[1].tolist())
        self.assertIsNone(report["accepted"])

    def test_snap_keeps_measured_edge_not_neighbourhood_average(self) -> None:
        surface = scale._NearestSurface(np.array([[0., 0., 1.], [.005, 0., 1.]]))
        self.assertEqual(surface.snap(np.array([0., 0., 1.]), 15.)["snapped_world_m"], [0., 0., 1.])
        for point, radius in (([0, 1], 15), ([0, 0, float('nan')], 15), ([0, 0, 1], -1)):
            with self.assertRaises(ValueError):
                surface.snap(np.array(point), radius)

    def test_exact_geometry_reports_zero_error_with_full_provenance(self) -> None:
        # Two landmark patches exactly 300 mm apart along world +X.
        points = np.vstack((_plate(np.array([0.0, 0.0, 1.0])), _plate(np.array([0.3, 0.0, 1.0]))))
        with TemporaryDirectory() as temporary:
            reconstruction = _build_reconstruction(Path(temporary), points)
            report = scale.validate_scale(
                reconstruction,
                [
                    {
                        "label": "calibration_box_long_edge",
                        "ground_truth_distance_mm": 300.0,
                        "point_a_world_m": [0.0, 0.0, 1.0],
                        "point_b_world_m": [0.3, 0.0, 1.0],
                    }
                ],
            )
        measurement = report["measurements"][0]
        self.assertAlmostEqual(measurement["measured_distance_mm"], 300.0, places=6)
        self.assertAlmostEqual(measurement["absolute_error_mm"], 0.0, places=6)
        self.assertAlmostEqual(measurement["relative_error_percent"], 0.0, places=6)
        self.assertEqual(measurement["geometry_source"], "fused_point_cloud")
        self.assertIn("RDF", measurement["coordinate_frame"])
        self.assertEqual(measurement["unit"], "metre")
        self.assertEqual(len(measurement["point_a"]["snapped_world_m"]), 3)
        self.assertEqual(len(measurement["point_b"]["snapped_world_m"]), 3)
        self.assertEqual(report["lineage"]["frame_count"], 14)
        self.assertEqual(report["lineage"]["pose_stage"], "open3d_sequential_hybrid_rgbd_odometry")
        self.assertIn("not a clinical", report["scope_disclaimer"].lower())

    def test_synthetic_lineage_is_explicit_in_scale_report(self) -> None:
        points = np.array([[0.0, 0.0, 1.0], [0.3, 0.0, 1.0]])
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            reconstruction = _build_reconstruction(root, points)
            (root / "demo_summary.json").write_text(
                json.dumps({"mode": "synthetic", "data_kind": "synthetic"}),
                encoding="utf-8",
            )
            report = scale.validate_scale(
                reconstruction,
                [{
                    "ground_truth_distance_mm": 300.0,
                    "point_a_world_m": points[0].tolist(),
                    "point_b_world_m": points[1].tolist(),
                }],
            )
        self.assertEqual(report["data_kind"], "synthetic")
        self.assertEqual(report["lineage"]["data_kind"], "synthetic")

    def test_scaled_geometry_surfaces_a_proportional_error(self) -> None:
        # A geometry that is 1% too long must be reported as +1%, not absorbed by snapping.
        points = np.vstack((_plate(np.array([0.0, 0.0, 1.0])), _plate(np.array([0.303, 0.0, 1.0]))))
        with TemporaryDirectory() as temporary:
            reconstruction = _build_reconstruction(Path(temporary), points)
            report = scale.validate_scale(
                reconstruction,
                [
                    {
                        "label": "one_percent_long",
                        "ground_truth_distance_mm": 300.0,
                        "point_a_world_m": [0.0, 0.0, 1.0],
                        "point_b_world_m": [0.303, 0.0, 1.0],
                    }
                ],
            )
        measurement = report["measurements"][0]
        self.assertAlmostEqual(measurement["measured_distance_mm"], 303.0, places=5)
        self.assertAlmostEqual(measurement["absolute_error_mm"], 3.0, places=5)
        self.assertAlmostEqual(measurement["relative_error_percent"], 1.0, places=5)
        self.assertAlmostEqual(report["summary"]["max_absolute_error_mm"], 3.0, places=5)

    def test_landmarks_without_measured_surface_are_blocked(self) -> None:
        points = _plate(np.array([0.0, 0.0, 1.0]))
        with TemporaryDirectory() as temporary:
            reconstruction = _build_reconstruction(Path(temporary), points)
            with self.assertRaises(ValueError) as raised:
                scale.validate_scale(
                    reconstruction,
                    [
                        {
                            "label": "empty_space",
                            "ground_truth_distance_mm": 300.0,
                            "point_a_world_m": [0.0, 0.0, 1.0],
                            "point_b_world_m": [5.0, 5.0, 5.0],
                        }
                    ],
                )
            self.assertIn("No reconstructed surface", str(raised.exception))

    def test_gaussian_model_is_not_an_accepted_measurement_source(self) -> None:
        self.assertEqual(set(scale.GEOMETRY_SOURCES), {"fused_point_cloud", "tsdf_mesh"})
        with TemporaryDirectory() as temporary:
            reconstruction = _build_reconstruction(Path(temporary), _plate(np.array([0.0, 0.0, 1.0])))
            with self.assertRaises(ValueError):
                scale.load_measurement_geometry(reconstruction, "gaussian")

    def test_spec_schema_is_enforced(self) -> None:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "spec.json"
            path.write_text(json.dumps({"measurements": []}), encoding="utf-8")
            with self.assertRaises(ValueError):
                scale.load_spec(path)
            path.write_text(
                json.dumps(
                    {
                        "schema": scale.SPEC_SCHEMA,
                        "geometry_source": "tsdf_mesh",
                        "measurements": [
                            {
                                "label": "edge",
                                "ground_truth_distance_mm": 100.0,
                                "point_a_world_m": [0, 0, 1],
                                "point_b_world_m": [0.1, 0, 1],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            measurements, source = scale.load_spec(path)
            self.assertEqual(source, "tsdf_mesh")
            self.assertEqual(len(measurements), 1)

    def test_local_auto_measurement_uses_tsdf_mesh_and_reports_normal_spread(self) -> None:
        points=np.array([[0.,0.,1.],[.3,0.,1.]])
        with TemporaryDirectory() as temporary:
            root=Path(temporary);reconstruction=_build_reconstruction(root,points)
            mesh=o3d.geometry.TriangleMesh()
            mesh.vertices=o3d.utility.Vector3dVector([[0,0,1],[.3,0,1],[0,.01,1],[.3,.01,1]])
            mesh.triangles=o3d.utility.Vector3iVector([[0,1,2],[1,3,2]])
            o3d.io.write_triangle_mesh(str(reconstruction/'scene_tsdf_mesh_m.ply'),mesh)
            report_path=reconstruction/'reconstruction_report.json'
            report=json.loads(report_path.read_text())
            report['local_capture']={'measured_only':True}
            report_path.write_text(json.dumps(report))
            result=scale.validate_scale(reconstruction,[{'ground_truth_distance_mm':300.,
                'point_a_world_m':[0,0,1],'point_b_world_m':[.3,0,1]}])
            self.assertEqual(result['geometry_source'],'tsdf_mesh')
            self.assertEqual(result['measurements'][0]['geometry_source'],'tsdf_mesh')
            self.assertIn('normal_spread_mm',result['measurements'][0]['point_a'])


if __name__ == "__main__":
    unittest.main()
