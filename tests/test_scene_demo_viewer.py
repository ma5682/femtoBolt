"""Unit tests for the viewer's scene-object management and camera helpers.

These cover the geometry-level logic only, so they run without a GPU window.
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import open3d as o3d

import femto.viewer.scene_demo_viewer as viewer


IDENTITY_POSE = np.eye(4).tolist()
INTRINSIC = {"width": 640, "height": 480, "fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0}


class FakeScene:
    """Minimal stand-in for rendering.Open3DScene used by SceneObjectManager."""

    def __init__(self) -> None:
        self.geometries: dict[str, object] = {}
        self.materials: dict[str, object] = {}
        self.shown: dict[str, bool] = {}

    def add_geometry(self, name, geometry, material):
        self.geometries[name] = geometry
        self.materials[name] = material
        self.shown[name] = True

    def has_geometry(self, name):
        return name in self.geometries

    def remove_geometry(self, name):
        self.geometries.pop(name, None)
        self.materials.pop(name, None)
        self.shown.pop(name, None)

    def show_geometry(self, name, show):
        self.shown[name] = bool(show)

    def modify_geometry_material(self, name, material):
        self.materials[name] = material


def _cloud(points) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    return cloud


def _material():
    return viewer.rendering.MaterialRecord()


def _write_dataset(root: Path, poses) -> Path:
    (root / "pose").mkdir(parents=True, exist_ok=True)
    (root / "calibration").mkdir(parents=True, exist_ok=True)
    (root / "calibration" / "color_intrinsic.json").write_text(
        json.dumps(INTRINSIC), encoding="utf-8"
    )
    lines = []
    for index, pose in enumerate(poses):
        relative = f"pose/{index:06d}.json"
        (root / relative).write_text(
            json.dumps({"frame": index, "pose_valid": True, "T_WC": pose}), encoding="utf-8"
        )
        lines.append(json.dumps({"frame": index, "pose": relative}))
    (root / "frames.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


class CameraGeometryTests(unittest.TestCase):
    def test_single_view_has_no_camera_path(self) -> None:
        report = {"frames": [{"camera_center_m": [0.0, 0.0, 0.0]}]}
        self.assertIsNone(viewer.camera_path_from_report(report))

    def test_multi_view_path_connects_every_centre(self) -> None:
        report = {"frames": [{"camera_center_m": [float(i), 0.0, 0.0]} for i in range(4)]}
        path = viewer.camera_path_from_report(report)
        self.assertIsNotNone(path)
        self.assertEqual(len(path.points), 4)
        self.assertEqual(len(path.lines), 3)
        self.assertEqual(len(path.colors), 3)

    def test_frustum_apex_is_the_camera_centre(self) -> None:
        pose = np.eye(4)
        pose[:3, 3] = [0.2, -0.1, 1.5]
        frustums = viewer.camera_frustums([pose], INTRINSIC, 0.3)
        self.assertIsNotNone(frustums)
        points = np.asarray(frustums.points)
        self.assertEqual(points.shape, (5, 3))
        self.assertEqual(len(frustums.lines), 8)
        np.testing.assert_allclose(points[0], pose[:3, 3])
        # Corner rays must sit exactly at the requested display depth.
        np.testing.assert_allclose(points[1:, 2] - pose[2, 3], np.full(4, 0.3))

    def test_frustums_are_indexed_per_pose(self) -> None:
        poses = [np.eye(4) for _ in range(3)]
        frustums = viewer.camera_frustums(poses, INTRINSIC, 0.25)
        self.assertEqual(len(frustums.points), 15)
        self.assertEqual(len(frustums.lines), 24)
        self.assertEqual(int(np.asarray(frustums.lines).max()), 14)

    def test_capture_poses_load_from_the_source_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = _write_dataset(Path(directory) / "scene", [IDENTITY_POSE, IDENTITY_POSE])
            poses, intrinsic = viewer.load_capture_poses({"source_dataset": str(dataset)})
        self.assertEqual(len(poses), 2)
        self.assertEqual(intrinsic["fx"], 500.0)

    def test_missing_dataset_reports_no_poses_instead_of_inventing_them(self) -> None:
        poses, intrinsic = viewer.load_capture_poses({"source_dataset": "E:/does/not/exist"})
        self.assertEqual(poses, [])
        self.assertIsNone(intrinsic)
        poses, intrinsic = viewer.load_capture_poses({})
        self.assertEqual(poses, [])
        self.assertIsNone(intrinsic)

    def test_invalid_pose_blocks_the_whole_frustum_set(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = _write_dataset(Path(directory) / "scene", [IDENTITY_POSE, IDENTITY_POSE])
            (dataset / "pose" / "000001.json").write_text(
                json.dumps({"frame": 1, "pose_valid": False, "T_WC": IDENTITY_POSE}),
                encoding="utf-8",
            )
            poses, intrinsic = viewer.load_capture_poses({"source_dataset": str(dataset)})
        self.assertEqual(poses, [])
        self.assertIsNone(intrinsic)

    def test_non_rigid_pose_is_rejected_for_frustums(self) -> None:
        invalid = np.eye(4)
        invalid[0, 0] = 2.0
        with tempfile.TemporaryDirectory() as directory:
            dataset = _write_dataset(Path(directory) / "scene", [invalid.tolist()])
            poses, intrinsic = viewer.load_capture_poses({"source_dataset": str(dataset)})
        self.assertEqual(poses, [])
        self.assertIsNone(intrinsic)


class HelperGeometryTests(unittest.TestCase):
    def test_grid_step_keeps_the_division_count_bounded(self) -> None:
        for span in (0.2, 1.0, 3.0, 12.0, 40.0):
            step = viewer.grid_step_m(span)
            self.assertIn(step, viewer.GRID_STEP_CANDIDATES_M)
            self.assertLessEqual(span / step, viewer.GRID_MAX_DIVISIONS)

    def test_grid_sits_on_the_lowest_measured_surface(self) -> None:
        bounds = o3d.geometry.AxisAlignedBoundingBox(
            np.array([-1.0, -0.5, 1.0]), np.array([1.0, 0.8, 3.0])
        )
        grid = viewer.reference_grid(bounds)
        points = np.asarray(grid.points)
        # +Y is down on disk, so the floor side is the maximum Y bound.
        np.testing.assert_allclose(points[:, 1], np.full(len(points), 0.8))
        self.assertGreater(len(grid.lines), 0)

    def test_degenerate_bounds_are_inflated_for_near_far_setup(self) -> None:
        point = np.array([0.0, 0.0, 1.0])
        bounds = o3d.geometry.AxisAlignedBoundingBox(point, point)
        inflated = viewer.inflated_bounds(bounds, 0.05)
        np.testing.assert_allclose(np.asarray(inflated.get_extent()), np.full(3, 0.05))


class SceneObjectManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scene = FakeScene()
        self.manager = viewer.SceneObjectManager(self.scene)

    def test_objects_use_unique_stable_geometry_names(self) -> None:
        self.manager.add_geometry("mesh", _cloud([[0, 0, 0]]), _material(), visible=True)
        self.manager.add_geometry("cloud", _cloud([[1, 1, 1]]), _material())
        self.assertEqual(set(self.scene.geometries), {"scene_mesh", "scene_pointcloud"})
        self.assertTrue(self.scene.shown["scene_mesh"])
        self.assertFalse(self.scene.shown["scene_pointcloud"])
        self.assertEqual(len(set(viewer.GEOMETRY_NAMES.values())), len(viewer.GEOMETRY_NAMES))

    def test_mesh_and_point_cloud_can_be_visible_at_the_same_time(self) -> None:
        self.manager.add_geometry("mesh", _cloud([[0, 0, 0]]), _material(), visible=True)
        self.manager.add_geometry("cloud", _cloud([[1, 1, 1]]), _material())
        self.manager.set_geometry_visible("cloud", True)
        self.assertTrue(self.manager.is_visible("mesh"))
        self.assertTrue(self.manager.is_visible("cloud"))
        self.assertTrue(all(self.scene.shown.values()))

    def test_hidden_objects_do_not_take_part_in_fit_to_view(self) -> None:
        self.manager.add_geometry("mesh", _cloud([[0, 0, 0], [1, 1, 1]]), _material(), visible=True)
        self.manager.add_geometry("cloud", _cloud([[9, 9, 9]]), _material())
        bounds = self.manager.visible_bounds()
        np.testing.assert_allclose(np.asarray(bounds.get_max_bound()), [1, 1, 1])
        self.manager.set_geometry_visible("cloud", True)
        bounds = self.manager.visible_bounds()
        np.testing.assert_allclose(np.asarray(bounds.get_max_bound()), [9, 9, 9])

    def test_helper_objects_are_excluded_from_fit_to_view(self) -> None:
        self.manager.add_geometry("mesh", _cloud([[0, 0, 0], [1, 1, 1]]), _material(), visible=True)
        self.manager.add_geometry(
            "grid", _cloud([[50, 50, 50]]), _material(), visible=True, fittable=False
        )
        bounds = self.manager.visible_bounds()
        np.testing.assert_allclose(np.asarray(bounds.get_max_bound()), [1, 1, 1])
        self.assertIn("scene_grid", self.scene.geometries)

    def test_replacing_geometry_keeps_name_and_visibility(self) -> None:
        self.manager.add_geometry("cloud", _cloud([[0, 0, 0]]), _material(), visible=True)
        replacement = _cloud([[2, 2, 2]])
        self.manager.replace_geometry("cloud", replacement)
        self.assertIs(self.scene.geometries["scene_pointcloud"], replacement)
        self.assertTrue(self.scene.shown["scene_pointcloud"])
        self.assertIs(self.manager.geometry_of("cloud"), replacement)

    def test_material_update_keeps_the_same_geometry_object(self) -> None:
        geometry = _cloud([[0, 0, 0]])
        self.manager.add_geometry("cloud", geometry, _material(), visible=True)
        material = _material()
        self.manager.update_material("cloud", material)
        self.assertIs(self.scene.geometries["scene_pointcloud"], geometry)
        self.assertIs(self.scene.materials["scene_pointcloud"], material)

    def test_removing_an_object_clears_the_render_scene(self) -> None:
        self.manager.add_geometry("gaussian", _cloud([[0, 0, 0]]), _material())
        self.manager.remove_geometry("gaussian")
        self.assertFalse(self.manager.has("gaussian"))
        self.assertNotIn("gaussian_points", self.scene.geometries)

    def test_empty_scene_has_no_bounds(self) -> None:
        self.assertIsNone(self.manager.visible_bounds())
        self.assertIsNone(self.manager.fittable_bounds())


class ViewDirectionTests(unittest.TestCase):
    def test_every_view_button_has_a_direction_and_a_valid_up_vector(self) -> None:
        self.assertEqual(set(viewer.VIEW_KEYS), set(viewer.VIEW_DIRECTIONS))
        for key, (direction, up) in viewer.VIEW_DIRECTIONS.items():
            forward = -np.asarray(direction, dtype=np.float64)
            self.assertAlmostEqual(float(np.linalg.norm(forward)), 1.0, msg=key)
            self.assertAlmostEqual(float(np.dot(forward, np.asarray(up, dtype=np.float64))), 0.0, msg=key)


if __name__ == "__main__":
    unittest.main()
