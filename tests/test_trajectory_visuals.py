import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import trimesh
from PIL import Image

from femto.planning.mesh_slice_planner import plan_mesh_slice
from femto.planning.trajectory_export import export_trajectory, load_trajectory
from test_mesh_slice_planner import grid_mesh, make_export


def attach_reference_rgb(manifest_path):
    manifest_path = Path(manifest_path)
    dataset = manifest_path.parent / "dataset"
    (dataset / "calibration").mkdir(parents=True)
    bgr = np.zeros((200, 200, 3), np.uint8)
    bgr[:, :, 1] = np.arange(200, dtype=np.uint8)[:, None]
    bgr[:, :, 2] = np.arange(200, dtype=np.uint8)[None, :]
    ok, encoded = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("PNG fixture encoding failed")
    rgb_path = dataset / "rgb.png"
    rgb_path.write_bytes(encoded.tobytes())
    (dataset / "frames.jsonl").write_text(json.dumps({"frame": 0, "rgb": "rgb.png"}) + "\n", encoding="utf-8")
    (dataset / "calibration" / "color_intrinsic.json").write_text(json.dumps(
        {"width": 200, "height": 200, "fx": 200.0, "fy": 200.0, "cx": 100.0, "cy": 100.0}), encoding="utf-8")
    (dataset / "calibration" / "distortion.json").write_text(json.dumps({"rgb": {
        "model": "BROWN_CONRADY_K6", "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0,
        "k3": 0.0, "k4": 0.0, "k5": 0.0, "k6": 0.0}}), encoding="utf-8")
    manifest = json.loads(manifest_path.read_text())
    manifest["origin_reference"]["source_dataset"] = str(dataset)
    manifest["region"]["view_reference"]["rgb_sha256"] = hashlib.sha256(rgb_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


class TrajectoryVisualTests(unittest.TestCase):
    def test_visual_bundle_with_verified_rgb(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = make_export(tmp, grid_mesh(0.04, 0.02, 0.0025))
            attach_reference_rgb(manifest)
            folder = export_trajectory(plan_mesh_slice(manifest), Path(tmp) / "results")
            expected = {"trajectory.json", "trajectory.csv", "gaps.csv", "trajectory_files.json",
                        "region_with_trajectory.glb", "route_diagram.png", "depth_diagram.png",
                        "trajectory_report.txt"}
            self.assertEqual({path.name for path in folder.iterdir()}, expected)
            loaded = load_trajectory(folder)
            self.assertEqual(loaded.route_diagram_status["status"], "available")
            with Image.open(folder / "route_diagram.png") as image:
                self.assertGreater(image.width, 100)
            with Image.open(folder / "depth_diagram.png") as image:
                self.assertEqual(image.size, (1200, 870))
            scene = trimesh.load(folder / "region_with_trajectory.glb", force="scene", process=False)
            names = set(scene.geometry)
            self.assertIn("selected_region", names)
            self.assertIn("surface_trajectory", names)
            report = (folder / "trajectory_report.txt").read_text(encoding="utf-8-sig")
            self.assertIn("execution_ready=false", report)
            files = json.loads((folder / "trajectory_files.json").read_text(encoding="utf-8"))
            self.assertNotIn("trajectory_files.json", files["files"])
            self.assertEqual(set(files["files"]), expected - {"trajectory_files.json"})

    def test_missing_rgb_is_nonfatal_and_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = make_export(tmp, grid_mesh(0.04, 0.02, 0.0025))
            folder = export_trajectory(plan_mesh_slice(manifest), Path(tmp) / "results")
            self.assertFalse((folder / "route_diagram.png").exists())
            payload = json.loads((folder / "trajectory.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["route_diagram_status"]["status"], "unavailable")
            self.assertTrue((folder / "depth_diagram.png").is_file())
            self.assertTrue((folder / "region_with_trajectory.glb").is_file())


if __name__ == "__main__":
    unittest.main()
