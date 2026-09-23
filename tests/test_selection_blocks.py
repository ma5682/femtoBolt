"""Acceptance tests for mesh-slice-serpentine-v4: work blocks are the physician's selection components.

Numbering follows the twelve acceptance items of the task. Item 4 (derived work surface from point
cloud support) is not implemented: its test pins the explicit "candidate reported, nothing generated"
behaviour until the user confirms a method and thresholds.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import trimesh

from tests.test_mesh_slice_planner import grid_mesh, make_export, pixel_mask, two_patches
from femto.annotation.selection_intent import build_selection_intent
from femto.planning.mesh_slice_planner import Unplannable, main, plan_mesh_slice
from femto.planning.route_motion import MotionPlayer
from femto.planning.surface_trajectory import (
    SELECTION_ALGORITHM, Gap, ScanSegment, SurfaceTrajectory, _point_on_triangle, face_components, to_player_plan,
    validate_trajectory,
)
from femto.planning.trajectory_export import current_algorithm_version, export_trajectory, load_trajectory


FULL = pixel_mask((0, 0, 199, 199))
LEFT_PATCH = pixel_mask((80, 80, 98, 120))
RIGHT_PATCH = pixel_mask((102, 80, 120, 120))


def plan(source, **kwargs):
    return plan_mesh_slice(source, connect_rows=True, block_planning=True, selection_blocks=True, **kwargs)


def sub(tmp, name):
    folder = Path(tmp) / name
    folder.mkdir()
    return folder


def segments(trajectory):
    return [item for item in trajectory.ordered_items if isinstance(item, ScanSegment)]


def gaps(trajectory):
    return [item for item in trajectory.ordered_items if isinstance(item, Gap)]


def region_mesh(source):
    return trimesh.load(Path(source).parent / "region_1_mesh_m.ply", process=False)


def history_row(tmp, manifest):
    data = json.loads(Path(manifest).read_text(encoding="utf-8"))
    annotations = Path(tmp) / "annotations.json"
    annotations.write_text(json.dumps({"regions": [dict(data["region"], region_manifest_file=Path(manifest).name)]}))
    return dict(kind="区域", source=annotations, region_index=0)


class SelectionBlockAcceptanceTests(unittest.TestCase):
    def test_01_one_selection_component_over_mesh_fragments_is_one_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = make_export(tmp, two_patches(), selection=dict(mask=FULL))
            result = plan(source)
            stats = result.statistics
            self.assertEqual(result.planner["algorithm_version"], SELECTION_ALGORITHM)
            self.assertEqual(stats["block_count"], 1)
            self.assertEqual(stats["mesh_fragment_count"], 2)
            block = result.blocks["items"][0]
            self.assertEqual(block["status"], "incomplete")
            self.assertEqual(block["continuity"], {"status": "broken", "chain_count": 2, "internal_gap_count": 1,
                                                   "link_failure_counts": {"disconnected_measured_surface": 1}})
            self.assertEqual([g.reason for g in gaps(result)], ["disconnected_surface"])
            self.assertEqual(gaps(result)[0].detail["link_failure_reason"], "disconnected_measured_surface")
            self.assertEqual(stats["continuity_status"], "partial")
            # The same mesh under the v3.1 rule is two mesh blocks: v4 does not take mesh topology as intent.
            legacy = plan_mesh_slice(source, connect_rows=True, block_planning=True)
            self.assertEqual(legacy.statistics["block_count"], 2)
            self.assertEqual(legacy.planner["algorithm_version"], "mesh-slice-serpentine-v3.1")

    def test_02_two_separate_selection_components_are_two_blocks_with_a_pause_between(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = make_export(tmp, two_patches(), selection=dict(mask=LEFT_PATCH | RIGHT_PATCH))
            result = plan(source)
            stats = result.statistics
            self.assertEqual((stats["block_count"], stats["planned_block_count"], stats["incomplete_block_count"]), (2, 2, 0))
            self.assertEqual(stats["gap_counts"], {"selection_break": 1})
            self.assertEqual(stats["continuity_status"], "complete")
            self.assertEqual([b["status"] for b in result.blocks["items"]], ["planned", "planned"])
            self.assertEqual([len(b["hole_ids"]) for b in json.loads(source.read_text())["region"]["selection_intent"]["components"]], [0, 0])
            player = to_player_plan(result)
            brk = player["chain_breaks"][0]
            self.assertTrue(brk["inter_block"])
            self.assertEqual(brk["reason"], "selection_break")
            self.assertIn("选择分离", brk["explanation"])
            self.assertFalse(brk["transition_planned"])
            motion = MotionPlayer(player)
            motion.play()
            motion.advance(1e9)
            self.assertTrue(motion.at_break and not motion.finished)
            self.assertTrue(motion.play())
            motion.advance(1e9)
            self.assertTrue(motion.finished)
            # Each block is exactly one chain owned by one selection component.
            components = face_components(region_mesh(source).faces)
            for block, segment in zip(result.blocks["items"], segments(result)):
                self.assertEqual(block["segment_count"], 1)
                self.assertEqual(len({int(components[p["triangle_index"]]) for p in segment.points}), 1)

    def test_03_explicit_unselected_hole_is_kept_and_routed_around(self):
        with tempfile.TemporaryDirectory() as tmp:
            hole = pixel_mask((98, 97, 102, 103))
            source = make_export(tmp, grid_mesh(0.06, 0.03, 0.002), selection=dict(mask=FULL & ~hole))
            intent = json.loads(source.read_text())["region"]["selection_intent"]
            self.assertEqual(len(intent["components"]), 1)
            self.assertEqual(len(intent["explicit_unselected_holes"]), 1)
            self.assertEqual(intent["explicit_unselected_holes"][0]["pixel_count"], int(hole.sum()))
            mesh = region_mesh(source)
            centroids = mesh.vertices[mesh.faces].mean(axis=1)
            self.assertFalse(np.any((np.abs(centroids[:, 0]) < 0.004) & (np.abs(centroids[:, 1]) < 0.0065)),
                             "faces whose centroid the physician left unselected are not in the region PLY")
            result = plan(source)
            stats = result.statistics
            self.assertEqual((stats["block_count"], stats["planned_block_count"], stats["gap_count"]), (1, 1, 0))
            self.assertGreater(stats["connected_detours"], 0)
            self.assertGreater(stats["connected_row_turns"], 0)
            block = result.blocks["items"][0]
            self.assertEqual(block["selection"]["hole_count"], 1)
            self.assertEqual(block["fragments"][0]["link_methods"].get("surface_edges", 0) + block["fragments"][0]["link_methods"].get("boundary_band", 0),
                             stats["connected_detours"] + stats["connected_row_turns"])
            for point in segments(result)[0].points:
                self.assertTrue(_point_on_triangle(point["xyz_m"], mesh.vertices, mesh.faces[point["triangle_index"]]))
                x, y = point["xyz_m"][:2]
                self.assertFalse(abs(x) < 0.0025 and abs(y) < 0.005, "no sample crosses the unselected hole interior")
            self.assertEqual(result.gap_report["gap_cause_counts"], {})
            self.assertGreaterEqual(stats["revisit_connector_sample_count"], 0)

    def test_04_missing_mesh_with_reference_depth_is_reported_not_used(self):
        """Phase C is not implemented: pixels with reference depth but no mesh are recorded as derived-surface
        candidates, no XYZ is generated for them and every trajectory point stays measured_mesh."""
        with tempfile.TemporaryDirectory() as tmp:
            depth = np.full((200, 200), 500, np.uint16)
            source = make_export(tmp, grid_mesh(0.06, 0.03, 0.002, hole=(-0.006, 0.006, -0.008, 0.008)),
                                 selection=dict(mask=FULL), depth=depth)
            support = json.loads(source.read_text())["region"]["selection_support"]
            self.assertGreater(support["counts"]["no_surface_reference_depth_present"], 0)
            self.assertEqual(support["derived_surface"]["status"], "not_generated")
            result = plan(source)
            report = result.gap_report["geometry_unconfirmed"]
            self.assertEqual(report["derived_surface"]["status"], "not_generated")
            self.assertEqual(report["derived_surface"]["reason"], "pending_user_confirmed_method_and_thresholds")
            self.assertGreater(report["by_reason"]["no_surface_reference_depth_present"], 0)
            self.assertTrue(all(p["geometry_support"] == "measured_mesh" for s in segments(result) for p in s.points))
            self.assertEqual(result.statistics["coverage_status"], "partial")
            self.assertEqual(result.statistics["planned_block_count"], 1, "the measured surface itself is still one chain")
            from femto.planning.trajectory_diagram import report_text
            self.assertIn("not_generated", report_text(result))

    def test_05_insufficient_or_ambiguous_depth_blocks_and_asks_for_recapture(self):
        with tempfile.TemporaryDirectory() as tmp:
            depth = np.full((200, 200), 500, np.uint16)
            depth[94:97, :] = 300  # mesh at 0.5 m, reference says 0.3 m: ambiguous, never selected
            source = make_export(tmp, grid_mesh(0.06, 0.03, 0.002), selection=dict(mask=FULL), depth=depth)
            result = plan(source)
            unconfirmed = result.gap_report["geometry_unconfirmed"]
            self.assertGreater(unconfirmed["by_reason"]["depth_conflict"], 0)
            self.assertEqual(result.statistics["coverage_status"], "partial")
            for segment in segments(result):
                for point in segment.points:
                    self.assertGreaterEqual(point["pixel"][1], 96.5, "no sample on ambiguous-depth rows")
            from femto.planning.trajectory_diagram import report_text
            text = report_text(result)
            self.assertIn("depth_conflict", text)
            # A component with no supported geometry at all is an unplannable block that asks for recapture.
            source2 = make_export(sub(tmp, "empty"), grid_mesh(0.02, 0.02, 0.002),
                                  selection=dict(mask=pixel_mask((90, 90, 110, 110), (150, 150, 190, 190))))
            result2 = plan(source2)
            statuses = [(b["status"], b.get("reason")) for b in result2.blocks["items"]]
            self.assertEqual(statuses, [("planned", None), ("unplannable", "no_supported_geometry")])
            self.assertIn("需要补采", result2.blocks["items"][1]["message"])
            self.assertEqual(result2.statistics["coverage_status"], "partial")
            self.assertEqual(result2.statistics["gap_count"], 0)

    def test_06_close_or_overlapping_but_separate_surfaces_are_never_joined(self):
        with tempfile.TemporaryDirectory() as tmp:
            tile = grid_mesh(0.02, 0.02, 0.002)
            close = trimesh.util.concatenate([tile.copy().apply_translation([-0.01005, 0, 0]),
                                              tile.copy().apply_translation([0.01005, 0, 0])])
            behind = trimesh.util.concatenate([tile, grid_mesh(0.03, 0.03, 0.003).apply_translation([0.004, 0.004, 0.08])])
            for name, mesh in (("close", close), ("behind", behind)):
                with self.subTest(mesh=name):
                    source = make_export(sub(tmp, name), mesh, selection=dict(mask=FULL))
                    result = plan(source)
                    self.assertEqual(result.statistics["block_count"], 1)
                    block = result.blocks["items"][0]
                    self.assertGreaterEqual(block["mesh_fragment_count"], 2)
                    self.assertEqual(block["status"], "incomplete")
                    self.assertTrue(all(g.reason == "disconnected_surface" for g in gaps(result)))
                    loaded = region_mesh(source)
                    components = face_components(loaded.faces)
                    for segment in segments(result):
                        owners = {int(components[p["triangle_index"]]) for p in segment.points}
                        self.assertEqual(len(owners), 1, "a chain never leaves its mesh fragment")
                        previous = None
                        for point in segment.points:
                            if point.get("role") == "connector" and previous is not None:
                                triangle = loaded.faces[point["triangle_index"]]
                                self.assertTrue(_point_on_triangle(previous, loaded.vertices, triangle))
                                self.assertTrue(_point_on_triangle(point["xyz_m"], loaded.vertices, triangle))
                            previous = point["xyz_m"]

    def test_07_planned_block_is_one_chain_and_an_injected_gap_fails_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = make_export(tmp, grid_mesh(0.06, 0.03, 0.002), selection=dict(mask=FULL))
            result = plan(source)
            block = result.blocks["items"][0]
            self.assertEqual((block["status"], block["segment_count"], block["gap_count"]), ("planned", 1, 0))
            self.assertEqual(result.statistics["gap_count"], 0)
            payload = result.to_dict()
            chain = payload["ordered_items"][0]
            cut = len(chain["points"]) // 2
            head = dict(chain, points=chain["points"][:cut])
            tail = dict(chain, segment_index=1, row_index=chain["points"][cut]["row_index"], points=chain["points"][cut:])
            gap = {"kind": "gap", "gap_index": 0, "reason": "surface_gap",
                   "detail": {"inferred_cause": "undetermined", "method": "sweep_plane_outer_loop_containment",
                              "confidence": "heuristic", "decision_use": "none", "link_failure_reason": "no_surface_path"},
                   "previous_point": {"route_index": chain["points"][cut - 1]["route_index"], "xyz_m": chain["points"][cut - 1]["xyz_m"]},
                   "next_point": {"route_index": chain["points"][cut]["route_index"], "xyz_m": chain["points"][cut]["xyz_m"]},
                   "scan_direction_w0": [1.0, 0.0, 0.0], "row_index_from": chain["points"][cut - 1]["row_index"],
                   "row_index_to": chain["points"][cut]["row_index"], "chord_length_m": 0.001}
            payload["ordered_items"] = [head, gap, tail]
            with self.assertRaises(ValueError):
                SurfaceTrajectory.from_dict(payload)
            # Even with consistent counts a "planned" block with an internal gap is refused.
            payload["statistics"].update(segment_count=2, gap_count=1, gap_counts={"surface_gap": 1})
            payload["blocks"]["items"][0].update(segment_count=2, gap_count=1, gap_counts={"surface_gap": 1})
            with self.assertRaises(ValueError):
                SurfaceTrajectory.from_dict(payload)

    def test_08_continuous_route_can_still_miss_selected_pixels_and_reports_the_gap(self):
        with tempfile.TemporaryDirectory() as tmp:
            speck = trimesh.Trimesh(vertices=[[0.05, 0.02, 0.5], [0.052, 0.02, 0.5], [0.05, 0.022, 0.5]],
                                    faces=[[0, 1, 2]], process=False)
            source = make_export(tmp, trimesh.util.concatenate([grid_mesh(0.06, 0.03, 0.002), speck]),
                                 selection=dict(mask=pixel_mask((60, 60, 140, 140))))
            result = plan(source)
            stats = result.statistics
            # Continuity: the measured main surface is one chain. Coverage: the selected speck fragment
            # (too small to hold a row) and the selected pixels without any mesh are reported as gaps.
            self.assertEqual(stats["continuity_status"], "complete")
            block = result.blocks["items"][0]
            self.assertEqual(block["status"], "planned")
            self.assertEqual((block["mesh_fragment_count"], block["planned_fragment_count"]), (2, 1))
            self.assertEqual(block["coverage"]["uncovered_fragment_count"], 1)
            self.assertEqual(block["coverage"]["uncovered_face_count"], 1)
            self.assertEqual(block["coverage"]["status"], "partial")
            self.assertGreater(block["selection"]["support"]["geometry_unconfirmed_pixels"], 0)
            self.assertEqual(stats["coverage_status"], "partial")
            self.assertEqual(stats["uncovered_fragment_count"], 1)
            self.assertGreater(stats["geometry_unconfirmed_pixel_count"], 0)
            self.assertEqual(stats["treatment_coverage_claim"], "none_without_effective_head_width")
            self.assertEqual([f["status"] for f in block["fragments"]], ["planned", "unplannable"])
            # Without the speck the surface is one chain, yet coverage is still partial: selected
            # pixels without geometry keep their requirement and are listed as a coverage gap.
            source2 = make_export(sub(tmp, "single"), grid_mesh(0.06, 0.03, 0.002),
                                  selection=dict(mask=pixel_mask((60, 60, 140, 140))))
            result2 = plan(source2)
            self.assertEqual(result2.statistics["continuity_status"], "complete")
            self.assertEqual(result2.statistics["coverage_status"], "partial")
            self.assertGreater(result2.gap_report["geometry_unconfirmed"]["pixel_count"], 0)
            from femto.planning.trajectory_diagram import report_text
            self.assertIn("几何待确认像素", report_text(result2))
            self.assertIn("覆盖缺口", report_text(result))

    def test_09_statistic_tamper_source_mismatch_and_forbidden_crossing_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            hole = pixel_mask((98, 97, 102, 103))
            source = make_export(tmp, grid_mesh(0.06, 0.03, 0.002), selection=dict(mask=FULL & ~hole))
            result = plan(source)
            base = result.to_dict()
            self.assertEqual(SurfaceTrajectory.from_dict(base).to_dict(), base)
            tampers = [
                lambda p: p["statistics"].__setitem__("connected_detours", p["statistics"]["connected_detours"] + 1),
                lambda p: p["statistics"].__setitem__("revisit_connector_sample_count", p["statistics"]["revisit_connector_sample_count"] + 1),
                lambda p: p["statistics"].__setitem__("connector_length_m", p["statistics"]["connector_length_m"] + 1e-6),
                lambda p: p["statistics"].__setitem__("coverage_status", "complete" if p["statistics"]["coverage_status"] != "complete" else "partial"),
                lambda p: p["statistics"].__setitem__("continuity_status", "partial"),
                lambda p: p["blocks"]["items"][0].__setitem__("status", "incomplete"),
                lambda p: p["blocks"]["items"][0]["continuity"].__setitem__("chain_count", 2),
                lambda p: p["blocks"]["items"][0]["coverage"].__setitem__("uncovered_face_count", 5),
                lambda p: p["blocks"]["items"][0]["revisit"].__setitem__("connector_sample_count", 999),
                lambda p: p["blocks"]["selection_binding"].__setitem__("intent_sha256", "0" * 64),
                lambda p: p["gap_report"].__setitem__("link_failure_counts", {"no_surface_path": 1}),
            ]
            for index, tamper in enumerate(tampers):
                payload = json.loads(json.dumps(base))
                tamper(payload)
                with self.assertRaises(ValueError, msg=f"tamper {index}"):
                    SurfaceTrajectory.from_dict(payload)
            folder = export_trajectory(result, Path(tmp) / "results", include_visuals=False)
            self.assertEqual(load_trajectory(folder).to_dict(), base)
            # Source mismatch: the export's selection intent changes after the trajectory was made.
            manifest = json.loads(source.read_text(encoding="utf-8"))
            manifest["region"]["selection_intent"] = build_selection_intent(FULL, method="work_only")
            backup = source.read_bytes()
            source.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_trajectory(folder)
            source.write_bytes(backup)
            self.assertEqual(load_trajectory(folder).to_dict(), base)
            # A connector edge that crosses an explicitly excluded region is refused when the region is known.
            mesh = region_mesh(source)
            region = json.loads(source.read_text(encoding="utf-8"))["region"]
            connector = next(p for p in segments(result)[0].points if p.get("role") == "connector")
            excluded = dict(region, exclusion_polygons=[[[connector["pixel"][0] - 0.5, connector["pixel"][1] - 0.5],
                                                         [connector["pixel"][0] + 0.5, connector["pixel"][1] - 0.5],
                                                         [connector["pixel"][0] + 0.5, connector["pixel"][1] + 0.5],
                                                         [connector["pixel"][0] - 0.5, connector["pixel"][1] + 0.5]]])
            with self.assertRaisesRegex(ValueError, "forbidden|excluded"):
                validate_trajectory(result, mesh=mesh, region=excluded)
            validate_trajectory(result, mesh=mesh, region=region)
            # A scan point moved onto a face of another selection component is refused.
            two = plan(make_export(sub(tmp, "two"), two_patches(), selection=dict(mask=LEFT_PATCH | RIGHT_PATCH)))
            payload = two.to_dict()
            first, second = payload["ordered_items"][0], payload["ordered_items"][2]
            first["points"][0]["triangle_index"] = second["points"][0]["triangle_index"]
            first["points"][0]["xyz_m"] = second["points"][0]["xyz_m"]
            two_region = json.loads((Path(tmp) / "two" / "region_1.json").read_text(encoding="utf-8"))["region"]
            with self.assertRaises(ValueError):
                validate_trajectory(SurfaceTrajectory.from_dict(dict(payload, result_sha256=None)) if False else
                                    SurfaceTrajectory(two.origin_reference, two.planner, two.source,
                                                      [ScanSegment.from_dict(i) if i["kind"] == "scan_segment" else Gap.from_dict(i)
                                                       for i in payload["ordered_items"]],
                                                      two.statistics, two.gap_report, two.spacing_report, two.row_report,
                                                      two.data_layers, None, two.blocks),
                                    mesh=region_mesh(Path(tmp) / "two" / "region_1.json"), region=two_region)

    def test_10_json_csv_glb_preview_and_playback_share_one_order(self):
        from tests.test_trajectory_visuals import attach_reference_rgb
        from femto.planning.path_presentation import render_diagram
        from femto.planning.trajectory_diagram import render_route_diagram, report_text
        with tempfile.TemporaryDirectory() as tmp:
            source = make_export(tmp, two_patches(), selection=dict(mask=LEFT_PATCH | RIGHT_PATCH))
            attach_reference_rgb(source)
            result = plan(source)
            folder = export_trajectory(result, Path(tmp) / "results")
            payload = json.loads((folder / "trajectory.json").read_text(encoding="utf-8"))
            self.assertTrue((folder / "route_diagram.png").is_file())
            self.assertIn("selection_semantics", to_player_plan(result))
            json_points = [(p["route_index"], s["segment_index"], s["block_index"], p.get("role", "scan"))
                           for s in payload["ordered_items"] if s["kind"] == "scan_segment" for p in s["points"]]
            header, *rows = (folder / "trajectory.csv").read_text(encoding="utf-8").splitlines()
            columns = header.split(",")
            self.assertEqual(columns[-3:], ["block_index", "role", "revisit"])
            csv_points = [(int(r[0]), int(r[1]), int(r[columns.index("block_index")]), r[columns.index("role")])
                          for r in (row.split(",") for row in rows)]
            self.assertEqual(csv_points, json_points)
            gap_header, *gap_rows = (folder / "gaps.csv").read_text(encoding="utf-8").splitlines()
            self.assertTrue(gap_header.endswith("row_turn_failure_reason,link_failure_reason"))
            self.assertEqual([r.split(",")[1] for r in gap_rows], [g["reason"] for g in payload["ordered_items"] if g["kind"] == "gap"])
            blocks_csv = (folder / "blocks.csv").read_text(encoding="utf-8").splitlines()
            self.assertIn("continuity_status", blocks_csv[0])
            self.assertEqual(len(blocks_csv), 1 + payload["statistics"]["block_count"])
            scene = trimesh.load(folder / "region_with_trajectory.glb", force="scene", process=False)
            names = sorted(name for name in scene.geometry if name.startswith("surface_trajectory_block_"))
            self.assertEqual(names, ["surface_trajectory_block_001", "surface_trajectory_block_002"])
            self.assertIn("gap_selection_break", scene.geometry)
            self.assertEqual([item["chain_indices"] for item in scene.metadata["blocks"]["items"]], [[0], [1]])
            player = to_player_plan(result)
            self.assertEqual([[p["route_index"] for p in chain] for chain in player["segments"]],
                             [[p["route_index"] for p in s["points"]] for s in payload["ordered_items"] if s["kind"] == "scan_segment"])
            self.assertEqual(player["blocks"]["chain_blocks"], [0, 1])
            region = json.loads(source.read_text(encoding="utf-8"))["region"]
            rgb = np.zeros((200, 200, 3), np.uint8)
            plain = render_diagram(rgb, np.zeros((200, 200), int), {"triangle_indices": [0]}, player)
            with_layers = render_diagram(rgb, np.zeros((200, 200), int), region, player)
            self.assertGreater(with_layers.size[1], plain.size[1] - 1)
            route = render_route_diagram(rgb, result, region)
            self.assertGreater(route.size[1], 200)
            text = report_text(result)
            self.assertIn("工作块规划（v4", text)
            self.assertIn("selection_break", text)
            report_file = (folder / "trajectory_report.txt").read_text(encoding="utf-8-sig")
            reloaded = load_trajectory(folder)
            self.assertEqual(report_file, report_text(reloaded))
            self.assertEqual(reloaded.to_dict()["ordered_items"], result.to_dict()["ordered_items"])
            self.assertEqual(reloaded.to_dict()["blocks"], result.to_dict()["blocks"])

    def test_11_old_versions_and_exports_without_intent_follow_the_explicit_policy(self):
        from femto.planning.trajectory_export import product_algorithm_notice
        from femto.planning.trajectory_history import generate, previous_trajectories
        from femto.planning.planning_io import TaskControl
        with tempfile.TemporaryDirectory() as tmp:
            legacy = make_export(tmp, two_patches())  # no selection intent: an old export
            with self.assertRaises(Unplannable) as caught:
                plan(legacy)
            self.assertEqual(caught.exception.reason, "selection_intent_missing")
            self.assertIn("重新选择并导出", str(caught.exception))
            # The old rule stays available and byte-stable; its file still loads and plays with a notice.
            old = plan_mesh_slice(legacy, connect_rows=True, block_planning=True)
            folder = export_trajectory(old, Path(tmp), include_visuals=False)
            self.assertEqual(load_trajectory(folder).to_dict(), old.to_dict())
            played = load_trajectory(folder, allow_older_versions=True)
            notice = product_algorithm_notice(played.planner["parameters"], played.planner["algorithm_version"])
            self.assertIn(SELECTION_ALGORITHM, notice)
            self.assertIn("重新选择并导出", notice)
            self.assertIsNone(product_algorithm_notice({"connect_rows": True, "block_planning": True, "selection_blocks": True}, SELECTION_ALGORITHM))
            self.assertEqual(current_algorithm_version({"connect_rows": True, "block_planning": True}), "mesh-slice-serpentine-v3.1")
            self.assertEqual(current_algorithm_version({"connect_rows": True, "block_planning": True, "selection_blocks": True}), SELECTION_ALGORITHM)
            # History regeneration on the old export refuses and keeps the old result untouched.
            row = history_row(tmp, legacy)
            before = (folder / "trajectory.json").read_bytes()
            with self.assertRaises(Unplannable):
                generate(row, TaskControl())
            self.assertEqual((folder / "trajectory.json").read_bytes(), before)
            self.assertEqual(previous_trajectories(legacy), [folder])
            # CLI: v4 by default writes an explicit blocked record; the legacy rule needs an explicit flag.
            code = main([str(legacy), "--output", str(Path(tmp) / "cli")])
            self.assertEqual(code, 2)
            blocked = json.loads(next((Path(tmp) / "cli").glob("trajectory_*/trajectory_blocked.json")).read_text(encoding="utf-8"))
            self.assertIn("选择意图", blocked["message"])
            self.assertEqual(main([str(legacy), "--output", str(Path(tmp) / "cli_v3"), "--mesh-blocks-v3"]), 0)
            # A v4 file whose intent record is removed from the export is refused on read-back.
            source = make_export(sub(tmp, "v4"), two_patches(), selection=dict(mask=LEFT_PATCH | RIGHT_PATCH))
            fresh = export_trajectory(plan(source), Path(tmp) / "v4", include_visuals=False)
            manifest = json.loads(source.read_text(encoding="utf-8"))
            manifest["region"].pop("selection_intent")
            manifest["region"].pop("selection_support")
            source.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_trajectory(fresh, allow_older_versions=True)
            # A v3.1 payload cannot masquerade as v4 and a v4 payload cannot drop its selection flag.
            payload = old.to_dict()
            payload["planner"]["algorithm_version"] = SELECTION_ALGORITHM
            with self.assertRaises(ValueError):
                SurfaceTrajectory.from_dict(payload)

    def test_12_cancel_failure_or_unfinished_blocks_keep_the_old_result(self):
        import shutil
        from femto.planning.trajectory_history import generate, history_rows, previous_trajectories
        from femto.planning.planning_io import TaskControl
        with tempfile.TemporaryDirectory() as tmp:
            # Real layout: <reconstruction>/annotations/<export>/{annotations.json, region_1.json, trajectory_*/}.
            reconstruction = Path(tmp) / "reconstruction"
            export_dir = reconstruction / "annotations" / "20260922_000000_deadbeef"
            export_dir.mkdir(parents=True)
            source = make_export(export_dir, two_patches(), selection=dict(mask=LEFT_PATCH | RIGHT_PATCH))
            row = history_row(export_dir, source)
            first = generate(row, TaskControl())
            self.assertEqual(first["cleanup"], "done")
            good = first["folder"]
            before = (good / "trajectory.json").read_bytes()
            # Cancelled regeneration: nothing removed.
            control = TaskControl()
            control.set()
            with self.assertRaises(RuntimeError):
                generate(row, control)
            self.assertEqual(previous_trajectories(source), [good])
            # Unfinished blocks: the export now carries one component spanning both patches, which
            # cannot become one chain. The partial result is published for preview; the old one stays
            # on disk and in the history next to it.
            manifest = json.loads(source.read_text(encoding="utf-8"))
            incomplete_source = make_export(sub(tmp, "incomplete"), two_patches(), selection=dict(mask=FULL))
            incomplete_manifest = json.loads(incomplete_source.read_text(encoding="utf-8"))
            manifest["region"].update(selection_intent=incomplete_manifest["region"]["selection_intent"],
                                      selection_support=incomplete_manifest["region"]["selection_support"],
                                      triangle_indices=incomplete_manifest["region"]["triangle_indices"])
            shutil.copy(Path(incomplete_source).parent / "region_1_mesh_m.ply", export_dir / "region_1_mesh_m.ply")
            manifest["file_sha256"] = incomplete_manifest["file_sha256"]
            source.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
            (export_dir / "annotations.json").write_text(json.dumps({"regions": [dict(manifest["region"], region_manifest_file=source.name)]}))
            result = generate(row, TaskControl())
            self.assertEqual(result["cleanup"], "skipped_incomplete_result")
            self.assertFalse(result["blocks"]["complete"])
            self.assertIn("旧轨迹已保留", result["blocks"]["text"])
            self.assertTrue(good.is_dir())
            self.assertTrue(result["folder"].is_dir())
            self.assertEqual(result["removed"], [])
            self.assertEqual(result["kept_previous"], [str(good)])
            self.assertEqual((good / "trajectory.json").read_bytes(), before)
            payload = json.loads((result["folder"] / "trajectory.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["statistics"]["incomplete_block_count"], 1)
            rows = history_rows(reconstruction)
            trajectories = [(r["folder"].name, r.get("result_complete")) for r in rows if r["kind"] == "三维表面轨迹"]
            self.assertEqual(sorted(trajectories), sorted([(good.name, True), (result["folder"].name, False)]))
            statuses = {r["folder"].name: r["status"] for r in rows if r["kind"] == "三维表面轨迹"}
            self.assertIn("未完成结果，旧轨迹保留", statuses[result["folder"].name])
            # Verification failure of a later regeneration: the old results are untouched.
            with mock.patch("femto.planning.trajectory_export.load_trajectory", side_effect=ValueError("bad output")):
                with self.assertRaises(ValueError):
                    generate(row, TaskControl())
            self.assertEqual((good / "trajectory.json").read_bytes(), before)
            self.assertTrue(result["folder"].is_dir())


if __name__ == "__main__":
    unittest.main()
