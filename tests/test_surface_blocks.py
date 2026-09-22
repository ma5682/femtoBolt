"""Block planning: each connected mesh surface is scanned completely before the next one."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import trimesh

from tests.test_mesh_slice_planner import grid_mesh, make_export, two_patches
from femto.planning.mesh_slice_planner import Unplannable, main, plan_mesh_slice
from femto.planning.route_motion import MotionPlayer
from femto.planning.surface_trajectory import (
    BLOCK_ALGORITHM, Gap, ScanSegment, SurfaceTrajectory, _point_on_triangle, block_view,
    face_components, to_player_plan,
)
from femto.planning.trajectory_export import export_trajectory, load_trajectory


def shifted(mesh, offset):
    copy = mesh.copy()
    copy.apply_translation(offset)
    return copy


def rotated(mesh, degrees, axis):
    copy = mesh.copy()
    center = np.array([0.0, 0.0, 0.5])
    copy.apply_translation(-center)
    copy.apply_transform(trimesh.transformations.rotation_matrix(np.radians(degrees), axis))
    copy.apply_translation(center)
    return copy


def staggered_blocks():
    """Four separate patches whose rows interleave in the reference image."""
    tile = grid_mesh(0.02, 0.014, 0.002)
    return trimesh.util.concatenate([
        shifted(tile, [-0.03, -0.012, 0]), shifted(tile, [0.0, -0.004, 0]),
        shifted(tile, [0.03, -0.012, 0]), shifted(tile, [-0.015, 0.012, 0])])


def plan(source, **kwargs):
    return plan_mesh_slice(source, connect_rows=True, block_planning=True, **kwargs)


def chain_blocks(trajectory):
    return [item.block_index for item in trajectory.ordered_items if isinstance(item, ScanSegment)]


def gaps(trajectory):
    return [item for item in trajectory.ordered_items if isinstance(item, Gap)]


class BlockOrderTests(unittest.TestCase):
    def test_staggered_blocks_finish_one_block_before_the_next(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = make_export(tmp, staggered_blocks())
            unblocked = plan_mesh_slice(source, connect_rows=True)
            mesh = trimesh.load(Path(tmp) / "region_1_mesh_m.ply", process=False)
            components = face_components(mesh.faces)
            old_sequence = [int(components[item.points[0]["triangle_index"]])
                            for item in unblocked.ordered_items if isinstance(item, ScanSegment)]
            self.assertGreater(len([1 for a, b in zip(old_sequence, old_sequence[1:]) if a != b]), 3,
                               "the global row order switches between surfaces, which block mode must remove")
            result = plan(source)
            sequence = chain_blocks(result)
            self.assertEqual(sequence, sorted(sequence))
            self.assertEqual(len(set(sequence)), 4)
            self.assertEqual(result.planner["algorithm_version"], BLOCK_ALGORITHM)
            self.assertEqual(result.statistics["coverage_status"], "complete")
            self.assertEqual(result.statistics["inter_block_gap_count"], 3)
            for gap in gaps(result):
                self.assertEqual(gap.reason, "disconnected_surface")
                self.assertEqual(gap.detail["decision_use"], "none")
            items = result.blocks["items"]
            centers = [item["center_pixel"] for item in items]
            self.assertEqual([round(c[1]) for c in centers], sorted(round(c[1]) for c in centers))
            for item in items:
                faces = np.flatnonzero(components == item["component_id"])
                self.assertEqual(item["face_count"], len(faces))
                for segment in result.ordered_items:
                    if isinstance(segment, ScanSegment) and segment.block_index == item["block_index"]:
                        self.assertTrue(all(components[p["triangle_index"]] == item["component_id"] for p in segment.points))
            points = [p for s in result.ordered_items if isinstance(s, ScanSegment) for p in s.points]
            self.assertEqual([p["route_index"] for p in points], list(range(len(points))))
            self.assertEqual(sum(b["point_count"] for b in items), len(points))
            self.assertEqual(sum(b["segment_count"] for b in items), result.statistics["segment_count"])
            self.assertAlmostEqual(sum(b["scan_length_m"] for b in items), result.statistics["scan_length_m"])
            self.assertAlmostEqual(sum(b["connector_length_m"] for b in items), result.statistics["connector_length_m"])
            self.assertEqual(sum(b["connected_row_turns"] for b in items), result.statistics["connected_row_turns"])

    def test_start_corner_changes_block_visit_order_deterministically(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = make_export(tmp, staggered_blocks())
            top = plan(source, start_corner="top_left")
            bottom = plan(source, start_corner="bottom_right")
            top_ids = [b["component_id"] for b in top.blocks["items"]]
            bottom_ids = [b["component_id"] for b in bottom.blocks["items"]]
            self.assertEqual(sorted(top_ids), sorted(bottom_ids))
            self.assertNotEqual(top_ids, bottom_ids)
            self.assertEqual(top_ids[0], bottom_ids[-1])
            for corner in ("top_left", "top_right", "bottom_left", "bottom_right"):
                first = plan(source, start_corner=corner)
                second = plan(source, start_corner=corner)
                self.assertEqual(first.to_dict(), second.to_dict())
                self.assertEqual(first.blocks["order_rule"]["start_corner"], corner)

    def test_blocks_with_different_orientations_get_their_own_sweep_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            tilted_x = rotated(grid_mesh(0.03, 0.016, 0.002), 30, [1, 0, 0])
            tilted_y = rotated(grid_mesh(0.03, 0.016, 0.002), 30, [0, 1, 0])
            mesh = trimesh.util.concatenate([shifted(tilted_x, [-0.022, 0, 0]), shifted(tilted_y, [0.022, 0, 0])])
            source = make_export(tmp, mesh)
            result = plan(source)
            frames = [b["sweep_frame"] for b in result.blocks["items"]]
            self.assertEqual(len(frames), 2)
            self.assertGreater(np.degrees(np.arccos(abs(np.dot(frames[0]["n"], frames[1]["n"])))), 20)
            self.assertTrue(all(f["direction_source"] == "pca" for f in frames))
            manual = plan(source, row_direction="w0_y")
            for block in manual.blocks["items"]:
                e1 = np.asarray(block["sweep_frame"]["e1"])
                n = np.asarray(block["sweep_frame"]["n"])
                expected = np.array([0, 1, 0.0]) - n * np.dot([0, 1, 0.0], n)
                np.testing.assert_allclose(e1, expected / np.linalg.norm(expected), atol=1e-9)
                self.assertEqual(block["sweep_frame"]["direction_source"], "w0_y")
            self.assertEqual(result.planner["sweep_frame"]["role"], "display_only_whole_mesh_frame")
            reloaded_mesh = trimesh.load(Path(tmp) / "region_1_mesh_m.ply", process=False)
            for segment in result.ordered_items:
                if isinstance(segment, ScanSegment):
                    for point in segment.points:
                        self.assertTrue(_point_on_triangle(point["xyz_m"], reloaded_mesh.vertices,
                                                           reloaded_mesh.faces[point["triangle_index"]]))


class BlockGapTests(unittest.TestCase):
    def test_hole_and_forbidden_gaps_survive_inside_a_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Block 1: hole fully covered by the forbidden polygon (u 97..103 ↔ x ±7.5 mm).
            # Block 2: an uncovered hole far below, so its gaps stay surface_gap.
            covered = grid_mesh(0.06, 0.03, 0.002, hole=(-0.003, 0.003, -0.01, 0.01))
            open_hole = shifted(grid_mesh(0.03, 0.02, 0.002, hole=(-0.003, 0.003, -0.005, 0.005)), [0, 0.06, 0])
            polygon = [[97, 90], [103, 90], [103, 110], [97, 110]]
            source = make_export(tmp, trimesh.util.concatenate([covered, open_hole]), forbidden=polygon)
            result = plan(source)
            reasons = {gap.reason for gap in gaps(result)}
            self.assertEqual(reasons, {"surface_gap", "forbidden_region", "disconnected_surface"})
            self.assertEqual(result.statistics["inter_block_gap_count"], 1)
            blocks = chain_blocks(result)
            for gap in gaps(result):
                crossing = blocks[gap.gap_index] != blocks[gap.gap_index + 1]
                self.assertEqual(crossing, gap.reason == "disconnected_surface")
                if gap.reason == "forbidden_region":
                    self.assertEqual(blocks[gap.gap_index], 0)
                    self.assertEqual(gap.detail["inferred_cause"], "not_applicable")
                if gap.reason == "surface_gap":
                    self.assertEqual(blocks[gap.gap_index], 1)
            for segment in result.ordered_items:
                if isinstance(segment, ScanSegment):
                    for point in segment.points:
                        self.assertFalse(-0.002 < point["xyz_m"][0] < 0.002 and -0.009 < point["xyz_m"][1] < 0.009)
                        self.assertFalse(-0.002 < point["xyz_m"][0] < 0.002 and 0.056 < point["xyz_m"][1] < 0.064)

    def test_near_or_overlapping_but_separate_surfaces_are_never_joined(self):
        with tempfile.TemporaryDirectory() as tmp:
            tile = grid_mesh(0.02, 0.02, 0.002)
            close = trimesh.util.concatenate([shifted(tile, [-0.01005, 0, 0]), shifted(tile, [0.01005, 0, 0])])
            behind = trimesh.util.concatenate([tile, shifted(grid_mesh(0.03, 0.03, 0.003), [0.004, 0.004, 0.08])])
            for mesh in (close, behind):
                with self.subTest(mesh=len(mesh.faces)):
                    source = make_export(tmp, mesh)
                    result = plan(source)
                    self.assertEqual(result.statistics["block_count"], 2)
                    self.assertEqual(result.statistics["inter_block_gap_count"], 1)
                    self.assertEqual(chain_blocks(result), sorted(chain_blocks(result)))
                    mesh_loaded = trimesh.load(Path(tmp) / "region_1_mesh_m.ply", process=False)
                    components = face_components(mesh_loaded.faces)
                    for segment in result.ordered_items:
                        if isinstance(segment, ScanSegment):
                            owners = {int(components[p["triangle_index"]]) for p in segment.points}
                            self.assertEqual(len(owners), 1)
                    player = to_player_plan(result)
                    inter = [b for b in player["chain_breaks"] if b.get("inter_block")]
                    self.assertEqual(len(inter), 1)
                    self.assertFalse(inter[0]["transition_planned"])

    def test_row_turns_inside_blocks_keep_face_support_and_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            tilted = shifted(grid_mesh(0.03, 0.02, 0.0025, tilt_deg=25), [-0.02, 0, 0])
            flat = shifted(grid_mesh(0.03, 0.02, 0.0025), [0.02, 0, 0])
            source = make_export(tmp, trimesh.util.concatenate([tilted, flat]))
            result = plan(source)
            mesh = trimesh.load(Path(tmp) / "region_1_mesh_m.ply", process=False)
            connectors = 0
            for segment in result.ordered_items:
                if not isinstance(segment, ScanSegment):
                    continue
                for a, b in zip(segment.points, segment.points[1:]):
                    self.assertLessEqual(np.linalg.norm(np.array(a["xyz_m"]) - b["xyz_m"]), 0.001000001)
                    if b.get("role") == "connector":
                        connectors += 1
                        triangle = mesh.faces[b["triangle_index"]]
                        self.assertTrue(_point_on_triangle(a["xyz_m"], mesh.vertices, triangle))
                        self.assertTrue(_point_on_triangle(b["xyz_m"], mesh.vertices, triangle))
            self.assertGreater(connectors, 0)
            self.assertEqual(result.statistics["gap_counts"]["disconnected_surface"], 1)
            # Each block must reproduce exactly what the v2 planner yields for that surface alone:
            # same rows, same connected turns, same kept gaps, same XYZ. No block gets a relaxed rule.
            # Visit order is image reading order: the flat patch has the smaller v_min, so it is block 1.
            for block, part in zip(result.blocks["items"], (flat, tilted)):
                alone_dir = Path(tmp) / f"alone_{block['block_index']}"
                alone_dir.mkdir()
                alone = plan_mesh_slice(make_export(alone_dir, part), connect_rows=True)
                self.assertEqual(block["connected_row_turns"], alone.statistics["connected_row_turns"])
                self.assertEqual(block["row_count"], alone.statistics["row_count"])
                self.assertEqual(block["segment_count"], alone.statistics["segment_count"])
                self.assertEqual(block["gap_counts"], alone.statistics["gap_counts"])
                self.assertAlmostEqual(block["connector_length_m"], alone.statistics["connector_length_m"])
                own = [p["xyz_m"] for s in result.ordered_items if isinstance(s, ScanSegment)
                       and s.block_index == block["block_index"] for p in s.points]
                theirs = [p["xyz_m"] for s in alone.ordered_items if isinstance(s, ScanSegment) for p in s.points]
                np.testing.assert_allclose(own, theirs, atol=1e-9)


class BlockStatusTests(unittest.TestCase):
    def test_small_fragments_are_reported_not_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            fragment = trimesh.Trimesh(vertices=[[0.05, 0.02, 0.5], [0.052, 0.02, 0.5], [0.05, 0.022, 0.5]],
                                       faces=[[0, 1, 2]], process=False)
            narrow = grid_mesh(0.02, 0.002, 0.002)
            source = make_export(tmp, trimesh.util.concatenate([grid_mesh(0.04, 0.02, 0.002), fragment,
                                                                shifted(narrow, [0, -0.03, 0])]))
            result = plan(source)
            self.assertEqual(result.statistics["block_count"], 3)
            self.assertEqual(result.statistics["plannable_block_count"], 1)
            self.assertEqual(result.statistics["coverage_status"], "partial")
            statuses = {b["block_index"]: (b["status"], b.get("reason")) for b in result.blocks["items"]}
            self.assertEqual(sorted(v[0] for v in statuses.values()), ["planned", "unplannable", "unplannable"])
            self.assertEqual({v[1] for v in statuses.values() if v[0] == "unplannable"},
                             {"insufficient_extent", "narrower_than_row_spacing"})
            self.assertEqual(result.statistics["unplannable_face_count"], 1 + len(narrow.faces))
            self.assertEqual(len(result.gap_report["unplannable_blocks"]), 2)
            self.assertEqual(result.statistics["gap_count"], len(gaps(result)))
            self.assertEqual(result.statistics["inter_block_gap_count"], 0)
            view = block_view(result)
            self.assertEqual(view["summary"]["coverage_status"], "partial")
            self.assertEqual([len(item["chain_indices"]) > 0 for item in view["items"]],
                             [b["status"] == "planned" for b in result.blocks["items"]])
            from femto.planning.trajectory_diagram import report_text
            text = report_text(result)
            self.assertIn("部分覆盖", text)
            self.assertIn("insufficient_extent", text)

    def test_all_unplannable_is_refused_and_old_result_is_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            good = make_export(tmp, grid_mesh(0.03, 0.02, 0.002))
            old = export_trajectory(plan(good), Path(tmp), include_visuals=False)
            before = (old / "trajectory.json").read_bytes()
            fragments = trimesh.util.concatenate([
                trimesh.Trimesh(vertices=[[x, 0.0, 0.5], [x + 0.001, 0.0, 0.5], [x, 0.001, 0.5]],
                                faces=[[0, 1, 2]], process=False) for x in (0.0, 0.01, 0.02)])
            bad = make_export(Path(tmp) / "bad", fragments) if (Path(tmp) / "bad").mkdir() is None else None
            with self.assertRaises(Unplannable) as caught:
                plan(bad)
            self.assertEqual(caught.exception.reason, "all_blocks_unplannable")
            self.assertEqual(len(caught.exception.details["blocks"]), 3)
            from femto.planning.trajectory_history import generate, previous_trajectories
            from femto.planning.planning_io import TaskControl
            data = json.loads(good.read_text())
            annotations = Path(tmp) / "annotations.json"
            annotations.write_text(json.dumps({"regions": [dict(data["region"], region_manifest_file=good.name)]}))
            row = dict(kind="区域", source=annotations, region_index=0)
            with mock.patch("femto.planning.mesh_slice_planner.plan_mesh_slice",
                            side_effect=Unplannable("all_blocks_unplannable", "none")):
                with self.assertRaises(Unplannable):
                    generate(row, TaskControl())
            self.assertEqual((old / "trajectory.json").read_bytes(), before)
            self.assertEqual(previous_trajectories(good), [old])
            code = main([str(bad), "--output", str(Path(tmp) / "cli")])
            self.assertEqual(code, 2)
            blocked = json.loads(next((Path(tmp) / "cli").glob("trajectory_*/trajectory_blocked.json")).read_text(encoding="utf-8"))
            self.assertEqual(len(blocked["details"]["blocks"]), 3)
            self.assertFalse(blocked["execution_ready"])

    def test_generate_reports_block_summary_and_replaces_old(self):
        with tempfile.TemporaryDirectory() as tmp:
            from femto.planning.trajectory_history import generate, history_rows
            from femto.planning.planning_io import TaskControl
            manifest = make_export(tmp, two_patches())
            data = json.loads(manifest.read_text())
            annotations = Path(tmp) / "annotations.json"
            annotations.write_text(json.dumps({"regions": [dict(data["region"], region_manifest_file=manifest.name)]}))
            row = dict(kind="区域", source=annotations, region_index=0)
            result = generate(row, TaskControl())
            self.assertEqual(result["blocks"]["coverage_status"], "complete")
            self.assertIn("共 2 块", result["blocks"]["text"])
            self.assertTrue((result["folder"] / "blocks.csv").is_file())
            loaded = load_trajectory(result["folder"])
            self.assertEqual(loaded.statistics["block_count"], 2)


class BlockBudgetTests(unittest.TestCase):
    def test_total_point_budget_is_shared_across_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = make_export(tmp, two_patches())
            single = make_export(Path(tmp) / "one", grid_mesh(0.025, 0.025, 0.0025)) if (Path(tmp) / "one").mkdir() is None else None
            alone = plan(single).statistics["point_count"]
            with mock.patch("femto.planning.mesh_slice_planner.POINT_BUDGET", alone + 10), \
                 mock.patch("femto.planning.surface_row_turns.POINT_BUDGET", alone + 10):
                plan(single)
                with self.assertRaisesRegex(ValueError, "exceeds"):
                    plan(source)

    def test_cancel_propagates_and_leaves_no_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = make_export(tmp, two_patches())
            calls = [0]
            def cancel():
                calls[0] += 1
                if calls[0] > 40:
                    raise RuntimeError("cancelled")
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                plan(source, check=cancel)
            trajectory = plan(source)
            parent = Path(tmp) / "results"
            calls[0] = 0
            def cancel_export():
                calls[0] += 1
                if calls[0] > 2:
                    raise RuntimeError("cancelled")
            with self.assertRaises(RuntimeError):
                export_trajectory(trajectory, parent, check=cancel_export, include_visuals=False)
            self.assertFalse(list(parent.glob("trajectory_*")))


class BlockCompatibilityTests(unittest.TestCase):
    def test_export_reload_reproduce_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = make_export(tmp, two_patches())
            result = plan(source)
            folder = export_trajectory(result, Path(tmp) / "results", include_visuals=False)
            loaded = load_trajectory(folder)
            self.assertEqual(loaded.to_dict(), result.to_dict())
            self.assertEqual(plan(source).to_dict(), result.to_dict())
            csv_rows = (folder / "trajectory.csv").read_text(encoding="utf-8").splitlines()
            self.assertTrue(csv_rows[0].endswith(",block_index"))
            blocks_csv = (folder / "blocks.csv").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(blocks_csv), 1 + result.statistics["block_count"])
            gap_rows = (folder / "gaps.csv").read_text(encoding="utf-8").splitlines()
            self.assertTrue(gap_rows[0].endswith("block_from,block_to"))
            payload = json.loads((folder / "trajectory.json").read_text(encoding="utf-8"))
            payload["blocks"]["items"][0]["face_count"] += 1
            (folder / "trajectory.json").write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_trajectory(folder)
            payload = json.loads((folder / "trajectory.json").read_text(encoding="utf-8"))
            payload["blocks"]["items"][0]["face_count"] -= 1
            payload["blocks"]["items"][0]["face_indices_sha256"] = "0" * 64
            (folder / "trajectory.json").write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_trajectory(folder)

    def test_v1_and_v2_results_are_unchanged_and_still_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = make_export(tmp, two_patches())
            v1 = plan_mesh_slice(source)
            v2 = plan_mesh_slice(source, connect_rows=True)
            self.assertEqual(v1.planner["algorithm_version"], "mesh-slice-serpentine-v1")
            self.assertEqual(v2.planner["algorithm_version"], "mesh-slice-serpentine-v2")
            for trajectory in (v1, v2):
                payload = trajectory.to_dict()
                self.assertNotIn("blocks", payload)
                self.assertNotIn("block_planning", payload["planner"]["parameters"])
                self.assertTrue(all("block_index" not in item for item in payload["ordered_items"]))
                folder = export_trajectory(trajectory, Path(tmp) / "results", include_visuals=False)
                self.assertFalse((folder / "blocks.csv").exists())
                self.assertEqual(load_trajectory(folder).to_dict(), payload)
                header = (folder / "trajectory.csv").read_text(encoding="utf-8").splitlines()[0]
                self.assertEqual(header, "route_index,segment_index,row_index,point_index,x_m,y_m,z_m,triangle_index,geometry_support")
            v3 = plan(source)
            self.assertNotEqual(chain_blocks(v3), [])
            payload = v3.to_dict()
            payload.pop("blocks")
            with self.assertRaises(ValueError):
                SurfaceTrajectory.from_dict(payload)
            payload = v2.to_dict()
            payload["blocks"] = v3.to_dict()["blocks"]
            with self.assertRaises(ValueError):
                SurfaceTrajectory.from_dict(payload)


class BlockPlaybackTests(unittest.TestCase):
    def test_player_pauses_between_blocks_and_continue_enters_next_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = plan(make_export(tmp, staggered_blocks()))
            player_plan = to_player_plan(result)
            blocks = player_plan["blocks"]["chain_blocks"]
            player = MotionPlayer(player_plan)
            visited = []
            player.play()
            for _ in range(200):
                player.advance(1e9)
                if player.finished:
                    break
                self.assertTrue(player.at_break)
                brk = player_plan["chain_breaks"][player.chain]
                self.assertEqual(brk["inter_block"], blocks[player.chain] != blocks[player.chain + 1])
                if brk["inter_block"]:
                    visited.append((blocks[player.chain], blocks[player.chain + 1]))
                    self.assertIn("块间", brk["explanation"])
                self.assertTrue(player.play())
            self.assertTrue(player.finished)
            self.assertEqual(visited, [(0, 1), (1, 2), (2, 3)])

    def test_window_filters_blocks_for_display_only_and_reports_block_stops(self):
        import tkinter as tk
        from tkinter import ttk
        from types import SimpleNamespace
        from femto.planning.path_presentation import render_diagram, show_presentation
        with tempfile.TemporaryDirectory() as tmp:
            result = plan(make_export(tmp, two_patches()))
            player_plan = to_player_plan(result)
            before = json.dumps(player_plan, sort_keys=True)
            rgb = np.zeros((200, 200, 3), np.uint8)
            model = SimpleNamespace(rgb=rgb, projection=SimpleNamespace(ids=np.zeros((200, 200), int)))
            region = {"triangle_indices": [0]}
            everything = np.asarray(render_diagram(rgb, model.projection.ids, region, player_plan))
            only_second = np.asarray(render_diagram(rgb, model.projection.ids, region, player_plan,
                                                    chain_indices=player_plan["blocks"]["items"][1]["chain_indices"]))
            self.assertEqual(everything.shape, only_second.shape)
            yellow = lambda image: int(((image[:, :, 0] > 200) & (image[:, :, 1] > 180) & (image[:, :, 2] < 120)).sum())
            self.assertLess(yellow(only_second), yellow(everything))
            self.assertGreater(yellow(only_second), 0)
            root = tk.Tk()
            root.withdraw()
            try:
                show_presentation(root, model, region, player_plan)
                window = root._route_demo_window
                root.update()
                def children(widget):
                    for child in widget.winfo_children():
                        yield child
                        yield from children(child)
                combo = next(w for w in children(window) if isinstance(w, ttk.Combobox) and "全部块" in w.cget("values"))
                self.assertEqual(len(combo.cget("values")), 3)
                combo.set(combo.cget("values")[2])
                combo.event_generate("<<ComboboxSelected>>")
                root.update()
                self.assertEqual(window.motion_state["chain_index"], player_plan["blocks"]["items"][1]["chain_indices"][0])
                window.block_selection.set("全部块")
                window.select_block()
                root.update()
                window.motion_player.reset()
                window.motion_player.play()
                window.motion_player.advance(1e9)
                buttons = {w.cget("text"): w for w in children(window) if isinstance(w, ttk.Button)}
                buttons["暂停"].invoke()
                root.update()
                self.assertTrue(window.motion_player.at_break)
                state = next(w for w in children(window) if isinstance(w, ttk.Label) and "块间停止" in str(w.cget("text")))
                self.assertIn("不播放跨块运动", str(state.cget("text")))
                buttons["继续"].invoke()
                root.update()
                self.assertEqual(window.motion_player.chain, 1)
                buttons["暂停"].invoke()
                window.close_demo()
            finally:
                root.destroy()
            self.assertEqual(json.dumps(player_plan, sort_keys=True), before)


if __name__ == "__main__":
    unittest.main()
