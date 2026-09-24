"""Treatment-head parameters: row spacing from width/overlap, recorded-only fields, coverage estimate."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests.test_mesh_slice_planner import make_export, two_patches
from tests.test_selection_blocks import LEFT_PATCH, RIGHT_PATCH, history_row
from femto.planning.planning_io import TaskControl
from femto.planning.treatment_head import FILE_NAME, coverage_estimate, head_parameters, scan_samples
from femto.planning.treatment_head_dialog import collect


def export(tmp):
    reconstruction = Path(tmp) / "reconstruction"
    export_dir = reconstruction / "annotations" / "20260924_000000_deadbeef"
    export_dir.mkdir(parents=True)
    source = make_export(export_dir, two_patches(), selection=dict(mask=LEFT_PATCH | RIGHT_PATCH, trim_to_support=True))
    return reconstruction, export_dir, source


class TreatmentHeadParameterTests(unittest.TestCase):
    def test_row_spacing_rules_and_ranges(self):
        record = head_parameters("from_effective_width", effective_treatment_width_mm=5, overlap_percent=20)
        self.assertEqual(record["row_spacing_mm"], 4.0)
        self.assertTrue(record["overlap_used_for_row_spacing"])
        self.assertEqual(record["values_status"], "placeholder_not_device_parameters")
        self.assertEqual(record["deferred"]["usage"], "recorded_only_not_used_by_planner")
        self.assertIn("not_region_inset", record["deferred"]["safety_margin_semantics"])
        self.assertFalse(record["execution_ready"])
        self.assertFalse(record["physical_accuracy_verified"])
        direct = head_parameters("direct_row_spacing", row_spacing_mm=6, overlap_percent=50)
        self.assertEqual(direct["row_spacing_mm"], 6.0)
        self.assertFalse(direct["overlap_used_for_row_spacing"])
        for bad in (dict(overlap_percent=95), dict(overlap_percent=100), dict(overlap_percent=-1),
                    dict(effective_treatment_width_mm=True), dict(effective_treatment_width_mm=float("nan")),
                    dict(effective_treatment_width_mm=0.5, overlap_percent=90),  # 0.05 mm: outside, never clamped
                    dict(treatment_depth_mm=-1), dict(head_diameter_mm=0), dict(safety_margin_mm=101)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                head_parameters("from_effective_width", **bad)
        for bad in (dict(), dict(row_spacing_mm=0.4), dict(row_spacing_mm=101)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                head_parameters("direct_row_spacing", **bad)
        with self.assertRaises(ValueError):
            head_parameters("auto")

    def test_dialog_parsing(self):
        texts = dict(row_spacing_mm="5", effective_treatment_width_mm="10", overlap_percent="20",
                     treatment_depth_mm="20", head_diameter_mm="40", safety_margin_mm="5")
        self.assertEqual(collect("from_effective_width", texts)["row_spacing_mm"], 8.0)
        self.assertEqual(collect("direct_row_spacing", texts)["row_spacing_mm"], 5.0)
        with self.assertRaisesRegex(ValueError, "有效治疗宽度"):
            collect("from_effective_width", dict(texts, effective_treatment_width_mm="abc"))
        # In auto mode the (disabled) direct entry is ignored even when invalid.
        self.assertEqual(collect("from_effective_width", dict(texts, row_spacing_mm="x"))["row_spacing_mm"], 8.0)


class TreatmentHeadTrajectoryTests(unittest.TestCase):
    def test_generate_records_parameters_reloads_and_rejects_tampering(self):
        from femto.planning.trajectory_export import load_trajectory
        from femto.planning.trajectory_history import generate, prepare_report
        with tempfile.TemporaryDirectory() as tmp:
            _, export_dir, source = export(tmp)
            row = history_row(export_dir, source)
            head = head_parameters("from_effective_width", effective_treatment_width_mm=5, overlap_percent=20)
            first = generate(row, TaskControl(), head)
            folder = first["folder"]
            payload = json.loads((folder / "trajectory.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["planner"]["parameters"]["row_spacing_mm"], 4.0)
            files = json.loads((folder / "trajectory_files.json").read_text(encoding="utf-8"))["files"]
            self.assertIn(FILE_NAME, files)
            record = json.loads((folder / FILE_NAME).read_text(encoding="utf-8"))
            self.assertEqual(record["result_sha256"], payload["result_sha256"])
            coverage = record["coverage_estimate"]
            self.assertEqual(coverage["claim"], "geometric_estimate_on_measured_region_mesh_not_treatment_coverage")
            self.assertTrue(0 < coverage["covered_area_ratio"] <= 1)
            self.assertEqual(coverage["covered"]["face_count"] + coverage["uncovered_near_boundary_or_hole"]["face_count"]
                             + coverage["uncovered_interior"]["face_count"], coverage["region_face_count"])
            report = (folder / "trajectory_report.txt").read_text(encoding="utf-8-sig")
            self.assertIn("【治疗头参数】（占位值，非设备参数）", report)
            self.assertIn("非治疗覆盖", report)
            trajectory = load_trajectory(folder)  # full reproduce with the saved 4 mm spacing
            self.assertIn("【治疗头参数】", prepare_report(dict(kind="三维表面轨迹", source=folder / "trajectory.json"), TaskControl()))
            # Brute-force check of the coverage rule.
            import trimesh
            mesh = trimesh.load(export_dir / "region_1_mesh_m.ply", process=False)
            centroids = np.asarray(mesh.vertices)[np.asarray(mesh.faces)].mean(axis=1)
            samples = scan_samples(trajectory)
            nearest = np.sqrt(((centroids[:, None, :] - samples[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
            self.assertEqual(int((nearest <= 0.0025).sum()), coverage["covered"]["face_count"])
            wider = coverage_estimate(trajectory, 20.0, mesh)
            self.assertGreaterEqual(wider["covered"]["area_m2"], coverage["covered"]["area_m2"])
            # Tampered record (hash list updated too) must not reproduce.
            import hashlib
            original = (folder / FILE_NAME).read_bytes()
            for change in (lambda r: r["coverage_estimate"]["covered"].update(face_count=0),
                           lambda r: r.update(overlap_percent=10.0),
                           lambda r: r.update(execution_ready=True)):
                tampered = json.loads(original)
                change(tampered)
                data = json.dumps(tampered).encode("utf-8")
                (folder / FILE_NAME).write_bytes(data)
                manifest = json.loads((folder / "trajectory_files.json").read_text(encoding="utf-8"))
                manifest["files"][FILE_NAME] = hashlib.sha256(data).hexdigest()
                (folder / "trajectory_files.json").write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_trajectory(folder, reproduce=False)
            (folder / FILE_NAME).write_bytes(original)
            manifest["files"][FILE_NAME] = hashlib.sha256(original).hexdigest()
            (folder / "trajectory_files.json").write_text(json.dumps(manifest), encoding="utf-8")
            load_trajectory(folder, reproduce=False)
            # Recorded-only inputs are protected by the file hash list.
            edited = json.loads(original)
            edited["deferred"]["head_diameter_mm"] = 41.0
            (folder / FILE_NAME).write_bytes(json.dumps(edited).encode("utf-8"))
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_trajectory(folder, reproduce=False)
            (folder / FILE_NAME).write_bytes(original)
            # A later complete result replaces the old folder even though it holds treatment_head.json.
            second = generate(row, TaskControl(), head_parameters("direct_row_spacing", row_spacing_mm=5))
            self.assertEqual(second["cleanup"], "done")
            self.assertEqual(second["removed"], [str(folder)])
            self.assertEqual(json.loads((second["folder"] / "trajectory.json").read_text(encoding="utf-8"))
                             ["planner"]["parameters"]["row_spacing_mm"], 5.0)
            # Without a head record the legacy output is unchanged (no record file).
            third = generate(row, TaskControl())
            self.assertFalse((third["folder"] / FILE_NAME).exists())
            load_trajectory(third["folder"])

    def test_cli_head_options(self):
        from femto.planning.mesh_slice_planner import main
        with tempfile.TemporaryDirectory() as tmp:
            _, export_dir, source = export(tmp)
            self.assertEqual(main([str(source), "--head-width-mm", "5", "--overlap-percent", "20"]), 0)
            folders = sorted(export_dir.glob("trajectory_*"))
            record = json.loads((folders[-1] / FILE_NAME).read_text(encoding="utf-8"))
            self.assertEqual((record["mode"], record["row_spacing_mm"], record["values_status"]),
                             ("from_effective_width", 4.0, "user_entered_unverified"))
            for argv in (["--row-mm", "5", "--head-width-mm", "5"], ["--overlap-percent", "20"],
                         ["--head-width-mm", "0.5", "--overlap-percent", "90"]):
                with self.subTest(argv=argv), self.assertRaises(SystemExit):
                    main([str(source), *argv])


if __name__ == "__main__":
    unittest.main()
