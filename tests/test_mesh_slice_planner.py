import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import trimesh

from femto.planning.mesh_slice_planner import plan_mesh_slice
from femto.planning.surface_trajectory import Gap, ScanSegment, canonical_json_bytes, to_player_plan
from femto.planning.trajectory_export import export_trajectory, load_trajectory


def grid_mesh(width=0.06, height=0.03, cell=0.003, *, hole=None, tilt_deg=0.0):
    xs = np.linspace(-width / 2, width / 2, int(round(width / cell)) + 1)
    ys = np.linspace(-height / 2, height / 2, int(round(height / cell)) + 1)
    vertices = np.asarray([[x, y, 0.5] for y in ys for x in xs], dtype=float)
    columns = len(xs)
    faces = []
    for y in range(len(ys) - 1):
        for x in range(len(xs) - 1):
            cx, cy = (xs[x] + xs[x + 1]) / 2, (ys[y] + ys[y + 1]) / 2
            if hole is not None and hole[0] <= cx <= hole[1] and hole[2] <= cy <= hole[3]:
                continue
            a = y * columns + x
            faces.extend([[a, a + 1, a + columns + 1], [a, a + columns + 1, a + columns]])
    if tilt_deg:
        angle = np.radians(tilt_deg)
        center = np.array([0.0, 0.0, 0.5])
        rotation = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)],
                             [0, np.sin(angle), np.cos(angle)]])
        vertices = (vertices - center) @ rotation.T + center
    return trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces), process=False)


def two_patches():
    left = grid_mesh(0.025, 0.025, 0.0025)
    right = left.copy()
    left.apply_translation([-0.022, 0, 0])
    right.apply_translation([0.022, 0, 0])
    return trimesh.util.concatenate([left, right])


INTRINSIC = [[200.0, 0.0, 100.0], [0.0, 200.0, 100.0], [0.0, 0.0, 1.0]]
IMAGE_SIZE = [200, 200]


def project_uv(points):
    """Reference-image pixel coordinates of W0 points for the synthetic 200×200 / f=200 camera."""
    points = np.asarray(points, dtype=float)
    return np.c_[200.0 * points[:, 0] / points[:, 2] + 100.0, 200.0 * points[:, 1] / points[:, 2] + 100.0]


def pixel_mask(*boxes):
    """Selection mask from (u0, v0, u1, v1) inclusive pixel boxes."""
    mask = np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0]), bool)
    for u0, v0, u1, v1 in boxes:
        mask[v0:v1 + 1, u0:u1 + 1] = True
    return mask


def selected_faces(mesh, mask, depth=None):
    """Whole triangles whose visible centroid pixel lies in `mask`, via the real FixedProjection gate."""
    projection = fixture_projection(mesh, depth)
    return projection.region([], mask=mask), projection


def fixture_projection(mesh, depth=None):
    import open3d as o3d
    from femto.annotation.fixed_rgb_geometry import FixedProjection
    legacy = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.asarray(mesh.vertices, float)),
                                       o3d.utility.Vector3iVector(np.asarray(mesh.faces, np.int32)))
    depth = np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0]), np.uint16) if depth is None else np.asarray(depth, np.uint16)
    return FixedProjection(legacy, np.asarray(INTRINSIC), depth)


def make_export(root, mesh=None, *, forbidden=None, selection=None, depth=None):
    """Synthetic femto-region-export-v1 manifest.

    Without `selection` the whole mesh is the region (legacy polygon rule, no intent record).
    With `selection` = dict(mask, forbidden_mask=None, method='work_only', trim_to_support=False)
    the region PLY holds only the faces the physician mask selects (same rule as the product
    page), and the manifest carries `selection_intent` + `selection_support` exactly like a real
    v4-capable export. `trim_to_support=True` keeps only mask pixels the mesh supports, i.e. a
    physician painting exactly the measured surface (no geometry-unconfirmed pixels)."""
    root = Path(root)
    mesh = mesh or grid_mesh()
    region_extra = {}
    if selection is not None:
        from femto.annotation.selection_intent import SELECTION_RULE, build_selection_intent, build_selection_support
        mask = np.asarray(selection["mask"], bool)
        projection = fixture_projection(mesh, depth)
        if selection.get("trim_to_support"):
            mask = mask & projection.valid
        ids = projection.region([], mask=mask)
        if not ids:
            raise ValueError("selection mask selects no face of the fixture mesh")
        subset = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces)[ids], process=False)
        subset.remove_unreferenced_vertices()
        region_extra = {"selection_intent": build_selection_intent(mask, forbidden_mask=selection.get("forbidden_mask"),
                                                                   method=selection.get("method", "work_only")),
                        "selection_support": build_selection_support(mask, projection),
                        "selection_rule": SELECTION_RULE}
        mesh = subset
        triangle_indices = list(ids)
    else:
        triangle_indices = list(range(len(mesh.faces)))
    mesh_path = root / "region_1_mesh_m.ply"
    mesh_path.write_bytes(mesh.export(file_type="ply"))
    region = {
        "annotation_id": "region-test", "triangle_indices": triangle_indices,
        "screen_polygon": [[0, 0], [199, 0], [199, 199], [0, 199]],
        "work_screen_polygon": [[0, 0], [199, 0], [199, 199], [0, 199]],
        "forbidden_screen_polygon": forbidden or [], "exclusion_polygons": [],
        "selection_method": "work_minus_forbidden" if forbidden else "work_only",
        "view_reference": {"undistorted_intrinsic": INTRINSIC,
                           "image_size": IMAGE_SIZE, "rgb_sha256": "0" * 64,
                           "depth_sha256": "1" * 64, "reference_frame_id": 0},
        **region_extra,
    }
    manifest = {
        "schema": "femto-region-export-v1", "region": region,
        "source_mesh": str(root / "source.ply"), "geometry_sha256": "2" * 64,
        "coordinate_frame": "world_rdf", "unit": "metre",
        "origin_reference": {"kind": "initial_color_camera", "source_dataset": str(root / "missing_dataset"),
                             "reference_frame_id": 0},
        "data_layers": {"schema": "femto-data-layers-v1"},
        "files": {"mesh_file": mesh_path.name},
        "file_sha256": {mesh_path.name: hashlib.sha256(mesh_path.read_bytes()).hexdigest()},
    }
    path = root / "region_1.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    return path


class MeshSlicePlannerTests(unittest.TestCase):
    def test_tilted_plane_support_spacing_and_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_export(tmp, grid_mesh(0.06, 0.025, 0.0025, tilt_deg=30))
            trajectory = plan_mesh_slice(path)
            payload = trajectory.to_dict()
            self.assertEqual(payload["planner"]["sweep_frame"]["direction_source"], "pca")
            self.assertEqual(payload["ordered_items"][0]["kind"], "scan_segment")
            self.assertEqual(payload["ordered_items"][-1]["kind"], "scan_segment")
            self.assertTrue(all(item["kind"] == ("scan_segment" if index % 2 == 0 else "gap")
                                for index, item in enumerate(payload["ordered_items"])))
            points = [point for item in payload["ordered_items"] if item["kind"] == "scan_segment"
                      for point in item["points"]]
            self.assertEqual([point["route_index"] for point in points], list(range(len(points))))
            self.assertTrue(all(point["geometry_support"] == "measured_mesh" for point in points))
            self.assertFalse(payload["execution_ready"])
            self.assertFalse(payload["physical_accuracy_verified"])
            self.assertIsNone(payload["orientation_fields"])
            self.assertGreater(payload["statistics"]["depth_span_m"], 0)

    def test_square_uses_w0_x_fallback_and_user_y_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_export(tmp, grid_mesh(0.04, 0.04, 0.0025))
            automatic = plan_mesh_slice(path)
            self.assertEqual(automatic.planner["sweep_frame"]["direction_source"], "w0_x_fallback")
            manual = plan_mesh_slice(path, row_direction="w0_y")
            self.assertEqual(manual.planner["sweep_frame"]["direction_source"], "w0_y")
            np.testing.assert_allclose(manual.planner["sweep_frame"]["e1"], [0, 1, 0], atol=1e-8)

    def test_hole_is_surface_gap_without_points_inside(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_export(tmp, grid_mesh(0.06, 0.03, 0.002, hole=(-0.003, 0.003, -0.01, 0.01)))
            trajectory = plan_mesh_slice(path)
            gaps = [item for item in trajectory.ordered_items if isinstance(item, Gap)]
            surface = [gap for gap in gaps if gap.reason == "surface_gap"]
            self.assertTrue(surface)
            self.assertTrue(any(gap.detail["inferred_cause"] == "interior_hole_likely" for gap in surface))
            for item in trajectory.ordered_items:
                if isinstance(item, ScanSegment):
                    self.assertFalse(any(-0.002 < point["xyz_m"][0] < 0.002 and
                                         -0.009 < point["xyz_m"][1] < 0.009 for point in item.points))

    def test_forbidden_has_priority_over_surface_gap(self):
        with tempfile.TemporaryDirectory() as tmp:
            polygon = [[97, 90], [103, 90], [103, 110], [97, 110]]
            path = make_export(tmp, grid_mesh(0.06, 0.03, 0.002, hole=(-0.003, 0.003, -0.01, 0.01)), forbidden=polygon)
            trajectory = plan_mesh_slice(path)
            gaps = [item for item in trajectory.ordered_items if isinstance(item, Gap)]
            self.assertTrue(any(gap.reason == "forbidden_region" for gap in gaps))
            self.assertTrue(all(gap.detail["inferred_cause"] == "not_applicable"
                                for gap in gaps if gap.reason == "forbidden_region"))

    def test_disconnected_patches_never_bridge(self):
        with tempfile.TemporaryDirectory() as tmp:
            trajectory = plan_mesh_slice(make_export(tmp, two_patches()))
            gaps = [item for item in trajectory.ordered_items if isinstance(item, Gap)]
            self.assertTrue(any(gap.reason == "disconnected_surface" for gap in gaps))
            player = to_player_plan(trajectory)
            self.assertEqual(len(player["chain_breaks"]), len(player["segments"]) - 1)

    def test_deterministic_export_reload_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_export(tmp, grid_mesh(0.045, 0.02, 0.0025))
            first = plan_mesh_slice(path)
            second = plan_mesh_slice(path)
            self.assertEqual(canonical_json_bytes(first.to_dict()), canonical_json_bytes(second.to_dict()))
            output = export_trajectory(first, Path(tmp) / "results")
            loaded = load_trajectory(output).to_dict()
            expected = first.to_dict()
            loaded.pop("route_diagram_status", None)
            expected.pop("route_diagram_status", None)
            self.assertEqual(loaded, expected)
            trajectory_path = output / "trajectory.json"
            payload = json.loads(trajectory_path.read_text(encoding="utf-8"))
            payload["planner"]["parameters"]["planarity_ratio_max"] = 0.4
            trajectory_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_trajectory(output)

    def test_bad_mesh_hash_and_origin_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_export(tmp)
            manifest = json.loads(path.read_text())
            manifest["file_sha256"][manifest["files"]["mesh_file"]] = "bad"
            path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                plan_mesh_slice(path)
        with tempfile.TemporaryDirectory() as tmp:
            path = make_export(tmp)
            manifest = json.loads(path.read_text())
            manifest["origin_reference"]["kind"] = "model_coordinate_origin"
            path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                plan_mesh_slice(path)

    def test_cancelled_export_leaves_no_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            trajectory = plan_mesh_slice(make_export(tmp))
            parent = Path(tmp) / "results"
            calls = [0]
            def cancel():
                calls[0] += 1
                if calls[0] > 2:
                    raise RuntimeError("cancelled")
            with self.assertRaises(RuntimeError):
                export_trajectory(trajectory, parent, check=cancel)
            self.assertFalse(list(parent.glob("trajectory_*")))


if __name__ == "__main__":
    unittest.main()
