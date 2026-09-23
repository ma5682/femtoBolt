"""Stage C step 1: derived work-surface candidates are generated, strictly checked and saved as a
diagnostic only. They never enter selection support, planning or trajectory replacement."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import trimesh

from tests.test_mesh_slice_planner import IMAGE_SIZE, INTRINSIC, grid_mesh, make_export, pixel_mask
from femto.dataset.local_reference import CALIBRATION_FILES, calibration_identity
from femto.planning.derived_surface_candidates import (
    ADMISSION, FILES_NAME, JSON_NAME, PLY_NAME, STATUS, build_candidates, export_candidates, load_candidates, main,
)
from femto.planning.mesh_slice_planner import Unplannable, plan_mesh_slice


FULL = pixel_mask((0, 0, 199, 199))
AROUND_HOLE = pixel_mask((74, 86, 126, 114))  # mesh footprint plus a margin of no-mesh pixels; away from the image border


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_png(path, image):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".png", image)[1].tofile(str(path))


def bound_export(tmp, mesh, depth, mask, *, other_depths=(), other_poses=()):
    """Region export + posed dataset whose calibration, reference RGB-D and poses are all bound and hashable.

    The synthetic camera has zero distortion, so the undistorted intrinsic equals INTRINSIC."""
    tmp = Path(tmp)
    dataset = tmp / "raw" / "derived" / "local_posed"
    calibration = dataset / "calibration"
    calibration.mkdir(parents=True)
    intrinsic = dict(width=IMAGE_SIZE[0], height=IMAGE_SIZE[1], fx=200.0, fy=200.0, cx=100.0, cy=100.0)
    zero = dict(model="OBCameraDistortionModel.BROWN_CONRADY_K6", k1=0.0, k2=0.0, k3=0.0, k4=0.0, k5=0.0, k6=0.0, p1=0.0, p2=0.0)
    contents = {"color_intrinsic.json": intrinsic, "depth_intrinsic.json": intrinsic, "distortion.json": dict(rgb=zero, depth=zero),
                "camera_metadata.json": dict(device="synthetic"), "depth_to_color.json": dict(T=np.eye(4).tolist()),
                "factory_calibration.json": dict(synthetic=True)}
    for name in CALIBRATION_FILES:
        (calibration / name).write_text(json.dumps(contents[name]), encoding="utf-8")
    depths = [np.asarray(depth, np.uint16)] + [np.asarray(d, np.uint16) for d in other_depths]
    poses = [np.eye(4)] + [np.asarray(p, float) for p in other_poses]
    rows = []
    for index, (d, pose) in enumerate(zip(depths, poses)):
        rgb = np.full((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), 90, np.uint8)
        write_png(tmp / "raw" / "rgb" / f"{index:06d}.png", rgb)
        write_png(tmp / "raw" / "depth" / f"{index:06d}.png", d)
        pose_path = dataset / "pose" / f"{index:06d}.json"
        pose_path.parent.mkdir(exist_ok=True)
        pose_path.write_text(json.dumps(dict(frame=index, pose_valid=True, T_WC=pose.tolist())), encoding="utf-8")
        rows.append(dict(frame=index, rgb=f"../../rgb/{index:06d}.png", depth=f"../../depth/{index:06d}.png",
                         pose=f"pose/{index:06d}.json", pose_valid=True))
    (dataset / "frames.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    # the range the dataset was reconstructed with; FixedProjection's fixture default is the same 0.25–2.21 m
    (dataset / "manifest.json").write_text(json.dumps(dict(schema="synthetic-posed-dataset", data_kind="synthetic",
                                                           camera=dict(depth_range_m=[0.25, 2.21]))), encoding="utf-8")
    export = tmp / "export"
    export.mkdir()
    manifest_path = make_export(export, mesh, selection=dict(mask=mask), depth=depths[0])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["origin_reference"].update(
        source_dataset=str(dataset), reference_frame_id=0, reference_pose=str(dataset / "pose" / "000000.json"),
        reference_pose_sha256=sha(dataset / "pose" / "000000.json"), calibration_sha256=calibration_identity(dataset)["sha256"])
    manifest["region"]["view_reference"].update(rgb_sha256=sha(tmp / "raw" / "rgb" / "000000.png"),
                                                depth_sha256=sha(tmp / "raw" / "depth" / "000000.png"))
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    return manifest_path


def hole_mesh():
    return grid_mesh(0.06, 0.03, 0.002, hole=(-0.006, 0.006, -0.008, 0.008))


def flat_depth(value=500):
    return np.full((IMAGE_SIZE[1], IMAGE_SIZE[0]), value, np.uint16)


def shifted_pose(dx=0.004, dy=-0.002):
    pose = np.eye(4)
    pose[:3, 3] = [dx, dy, 0.0]
    return pose


class DerivedSurfaceCandidateTests(unittest.TestCase):
    def test_hole_with_reference_depth_is_covered_by_strictly_contained_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = bound_export(tmp, hole_mesh(), flat_depth(), AROUND_HOLE, other_depths=[flat_depth()], other_poses=[shifted_pose()])
            support = json.loads(source.read_text(encoding="utf-8"))["region"]["selection_support"]
            candidates = support["counts"]["no_surface_reference_depth_present"]
            self.assertGreater(candidates, 0)
            result = build_candidates(source)
            payload = result["payload"]
            self.assertEqual(payload["status"], STATUS)
            self.assertEqual(payload["admission_to_planning"], ADMISSION)
            self.assertFalse(payload["execution_ready"])
            self.assertFalse(payload["physical_accuracy_verified"])
            self.assertEqual(payload["selection"]["candidate_pixel_count"], candidates)
            self.assertEqual(payload["coverage"]["uncovered_pixels"], 0)
            self.assertTrue(payload["completeness"]["all_candidate_pixels_covered"])
            self.assertEqual(payload["completeness"]["depth_conflict_pixels_remaining"], 0)
            mesh = result["candidate_mesh"]
            self.assertGreater(len(mesh.faces), 0)
            np.testing.assert_allclose(np.asarray(mesh.vertices)[:, 2], 0.5, atol=1e-9)
            # every candidate face has at least one vertex on a missing-mesh pixel and lies in one component
            self.assertEqual(payload["faces"]["rejected"]["spans_two_selection_components"], 0)
            self.assertEqual(payload["faces"]["rejected"]["crosses_unselected_pixels"], 0)
            self.assertLessEqual(payload["faces"]["seam_distance_to_region_mesh_percentiles_mm"][2], 3.0)
            view = payload["cross_view"]["views"][0]
            self.assertEqual(view["observed_pixels"], candidates)
            self.assertEqual(view["within_diagnostic_tolerance"], candidates)
            self.assertEqual(payload["cross_view"]["gate"], "none")
            self.assertEqual(payload["evidence"]["frames"][1]["role"], "other_view")

    def test_explicit_unselected_strip_and_two_components_are_never_crossed(self):
        with tempfile.TemporaryDirectory() as tmp:
            mask = FULL.copy()
            mask[:, 100] = False  # one-pixel unselected strip through the hole: two components
            source = bound_export(tmp, hole_mesh(), flat_depth(), mask)
            intent = json.loads(source.read_text(encoding="utf-8"))["region"]["selection_intent"]
            self.assertEqual(len(intent["components"]), 2)
            result = build_candidates(source)
            payload = result["payload"]
            rejected = payload["faces"]["rejected"]
            self.assertGreater(rejected["outside_image_or_unselected"] + rejected["spans_two_selection_components"]
                               + rejected["crosses_unselected_pixels"], 0)
            mesh = result["candidate_mesh"]
            vertices = np.asarray(mesh.vertices)
            uv = np.c_[200 * vertices[:, 0] / vertices[:, 2] + 100, 200 * vertices[:, 1] / vertices[:, 2] + 100]
            pix = np.floor(uv + .5).astype(int)
            self.assertFalse(np.any(pix[:, 0] == 100), "no candidate vertex on the unselected strip")
            for face in np.asarray(mesh.faces):
                self.assertEqual(len(set(np.sign(pix[face, 0] - 100))), 1, "no face spans the strip")
            components = {row["component_id"] for row in payload["coverage"]["per_component"]}
            self.assertEqual(components, {0, 1})
            self.assertEqual(payload["coverage"]["uncovered_pixels"], 0)

    def test_depth_conflict_and_missing_depth_pixels_are_not_attempted(self):
        with tempfile.TemporaryDirectory() as tmp:
            depth = flat_depth()
            depth[92:95, 90:96] = 300  # rows 94.. lie on the measured mesh (0.5 m) but reference says 0.3 m: conflict
            depth[98:102, 98:102] = 0  # inside the hole with no reference depth at all
            source = bound_export(tmp, hole_mesh(), depth, AROUND_HOLE)
            support = json.loads(source.read_text(encoding="utf-8"))["region"]["selection_support"]
            self.assertGreater(support["counts"]["depth_conflict"], 0)
            self.assertGreater(support["counts"]["no_surface_no_reference_depth"], 0)
            result = build_candidates(source)
            payload = result["payload"]
            self.assertGreater(payload["faces"]["rejected"]["touches_depth_conflict_pixel"], 0)
            self.assertEqual(payload["completeness"]["depth_conflict_pixels_remaining"], support["counts"]["depth_conflict"])
            from femto.annotation.selection_intent import support_classes
            classes = support_classes(support)
            vertices = np.asarray(result["candidate_mesh"].vertices)
            uv = np.floor(np.c_[200 * vertices[:, 0] / vertices[:, 2] + 100, 200 * vertices[:, 1] / vertices[:, 2] + 100] + .5).astype(int)
            self.assertTrue(np.all(np.isin(classes[uv[:, 1], uv[:, 0]], [1, 2])), "vertices only on measured or candidate pixels")
            self.assertFalse(np.any((uv[:, 1] >= 98) & (uv[:, 1] < 102) & (uv[:, 0] >= 98) & (uv[:, 0] < 102)), "no vertex without depth")
            # a candidate ring at 0.3 m next to 0.5 m pixels fails the frozen continuity gate on that side: still no interpolation
            self.assertTrue(np.all(np.isin(np.round(vertices[:, 2], 6), [0.3, 0.5])))
            self.assertGreater(payload["coverage"]["covered_pixels"], 0)
            self.assertEqual(payload["selection"]["candidate_pixel_count"], support["counts"]["no_surface_reference_depth_present"])

    def test_export_reload_and_planner_stay_separated(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = bound_export(tmp, hole_mesh(), flat_depth(), FULL)
            before = source.read_bytes()
            folder = export_candidates(build_candidates(source))
            self.assertEqual(folder.parent, source.parent)
            self.assertTrue(folder.name.startswith("derived_surface_candidates_"))
            self.assertEqual(source.read_bytes(), before, "region export untouched")
            loaded = load_candidates(folder)
            self.assertEqual(loaded["payload"]["schema"], "femto-derived-surface-candidates-v1")
            self.assertIsNotNone(loaded["candidate_mesh"])
            # v4 planning of the same export still reports the derived surface as not generated
            result = plan_mesh_slice(source, connect_rows=True, block_planning=True, selection_blocks=True)
            self.assertEqual(result.gap_report["geometry_unconfirmed"]["derived_surface"]["status"], "not_generated")
            from femto.planning.surface_trajectory import ScanSegment
            self.assertTrue(all(p["geometry_support"] == "measured_mesh" for i in result.ordered_items
                                if isinstance(i, ScanSegment) for p in i.points))
            self.assertEqual(result.statistics["coverage_status"], "partial")
            # tampering: file hash, and a candidate face pushed off its selection component
            ply = folder / PLY_NAME
            mesh = trimesh.load(ply, force="mesh", process=False)
            vertices = np.asarray(mesh.vertices).copy()
            vertices[0] = [0.3, 0.3, 0.5]  # projects outside the 200x200 image
            trimesh.Trimesh(vertices=vertices, faces=mesh.faces, process=False).export(ply)
            with self.assertRaises(ValueError):
                load_candidates(folder)
            files = json.loads((folder / FILES_NAME).read_text(encoding="utf-8"))
            files["files"][PLY_NAME] = sha(ply)
            (folder / FILES_NAME).write_text(json.dumps(files), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "leaves"):
                load_candidates(folder)
            record = json.loads((folder / JSON_NAME).read_text(encoding="utf-8"))
            self.assertEqual(record["status"], STATUS)

    def test_reload_reverifies_raw_evidence_and_binding_checks_depth_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = bound_export(tmp, hole_mesh(), flat_depth(), AROUND_HOLE, other_depths=[flat_depth()], other_poses=[shifted_pose()])
            folder = export_candidates(build_candidates(source))
            evidence = load_candidates(folder)["payload"]["evidence"]
            raw = Path(tmp) / "raw"
            dataset = raw / "derived" / "local_posed"
            self.assertEqual(evidence["frames_jsonl_sha256"], sha(dataset / "frames.jsonl"))
            self.assertEqual(evidence["dataset_manifest_sha256"], sha(dataset / "manifest.json"))
            self.assertEqual(evidence["depth_range_m"], [0.25, 2.21])
            self.assertEqual([f["role"] for f in evidence["frames"]], ["reference", "other_view"])
            tampers = [  # (file, new content writer, expected refusal message)
                (raw / "depth" / "000001.png", lambda p: write_png(p, flat_depth(503)), "evidence"),
                (raw / "rgb" / "000001.png", lambda p: write_png(p, np.full((200, 200, 3), 91, np.uint8)), "evidence"),
                (dataset / "pose" / "000001.json", lambda p: p.write_text(json.dumps(dict(
                    frame=1, pose_valid=True, T_WC=shifted_pose(0.006, 0.0).tolist())), encoding="utf-8"), "evidence"),
                (dataset / "frames.jsonl", lambda p: p.write_text(p.read_text(encoding="utf-8") + "\n", encoding="utf-8"), "evidence"),
                (dataset / "pose" / "000000.json", lambda p: p.write_text(json.dumps(dict(
                    frame=0, pose_valid=True, T_WC=shifted_pose(0.001, 0.0).tolist())), encoding="utf-8"), "SHA-256"),
                (raw / "depth" / "000000.png", lambda p: write_png(p, flat_depth(501)), "SHA-256"),
                (dataset / "calibration" / "camera_metadata.json", lambda p: p.write_text("{}", encoding="utf-8"), "Calibration"),
                (dataset / "manifest.json", lambda p: p.write_text(json.dumps(dict(camera=dict(depth_range_m=[9.0, 10.0]))),
                                                                   encoding="utf-8"), "depth range"),
                (dataset / "manifest.json", lambda p: p.write_text(json.dumps(dict(camera=dict(depth_range_m=[0.25, 2.21]),
                                                                                   note="x")), encoding="utf-8"), "evidence"),
            ]
            for target, tamper, message in tampers:
                original = target.read_bytes()
                tamper(target)
                with self.assertRaisesRegex(ValueError, message, msg=str(target)):
                    load_candidates(folder)
                target.write_bytes(original)
            self.assertIsNotNone(load_candidates(folder)["candidate_mesh"], "restored evidence loads again")
            # the export's own depth range is not trusted either: it must equal the bound dataset's range
            manifest = json.loads(source.read_text(encoding="utf-8"))
            manifest["region"]["selection_support"]["depth_range_m"] = [9.0, 10.0]
            source.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "depth range"):
                build_candidates(source)
            with self.assertRaisesRegex(ValueError, "Region export changed"):
                load_candidates(folder)
            manifest["region"]["selection_support"]["depth_range_m"] = [0.25, 2.21]
            manifest["region"]["selection_support"]["derived_surface"]["status"] = "admitted_to_planning"
            source.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not_generated"):
                build_candidates(source)

    def test_binding_refusals_and_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = bound_export(tmp, hole_mesh(), flat_depth(), FULL)
            manifest = json.loads(source.read_text(encoding="utf-8"))
            # changed raw depth bytes
            depth_path = Path(tmp) / "raw" / "depth" / "000000.png"
            original = depth_path.read_bytes()
            write_png(depth_path, flat_depth(501))
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                build_candidates(source)
            depth_path.write_bytes(original)
            # changed calibration
            calibration = Path(tmp) / "raw" / "derived" / "local_posed" / "calibration" / "camera_metadata.json"
            saved = calibration.read_bytes()
            calibration.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Calibration"):
                build_candidates(source)
            calibration.write_bytes(saved)
            # missing selection intent (old export)
            legacy = Path(tmp) / "legacy"
            legacy.mkdir()
            stripped = dict(manifest, region={k: v for k, v in manifest["region"].items()
                                              if k not in ("selection_intent", "selection_support")})
            (legacy / "region_1.json").write_text(json.dumps(stripped), encoding="utf-8")
            (legacy / "region_1_mesh_m.ply").write_bytes((source.parent / "region_1_mesh_m.ply").read_bytes())
            with self.assertRaises(Unplannable):
                build_candidates(legacy / "region_1.json")
            out = Path(tmp) / "cli"
            self.assertEqual(main([str(legacy / "region_1.json"), "--output", str(out)]), 2)
            blocked = list(out.glob("derived_surface_candidates_*/derived_surface_blocked.json"))
            self.assertEqual(len(blocked), 1)
            self.assertEqual(json.loads(blocked[0].read_text(encoding="utf-8"))["reason"], "selection_intent_missing")
            self.assertEqual(main([str(source.parent), "--output", str(out)]), 0)
            self.assertEqual(len(list(out.glob("derived_surface_candidates_*/" + JSON_NAME))), 1)


if __name__ == "__main__":
    unittest.main()
