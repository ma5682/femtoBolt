"""End-to-end checks for v4.1 without changing v4's scan or selection semantics."""
from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import trimesh

from tests.test_mesh_slice_planner import IMAGE_SIZE, INTRINSIC, grid_mesh, make_export, pixel_mask, two_patches
from femto.annotation.selection_intent import component_labels
from femto.planning.mesh_slice_planner import main, plan_mesh_slice
from femto.planning.planning_io import TaskControl
from femto.planning.surface_row_turns import edge_pixels_allowed
from femto.planning.surface_trajectory import (
    BIDIRECTIONAL_SELECTION_ALGORITHM, FACE_PORTAL_SELECTION_ALGORITHM, ORDERED_SELECTION_ALGORITHM,
    OPTIMIZED_SELECTION_ALGORITHM, SELECTION_ALGORITHM, Gap, ScanSegment,
    _length_split, _point_on_triangle, validate_trajectory,
)
from femto.planning.trajectory_export import export_trajectory, load_trajectory


FULL = pixel_mask((0, 0, 199, 199))


def plan(source, version):
    return plan_mesh_slice(source, connect_rows=True, block_planning=True, selection_blocks=True,
                           selection_algorithm=version)


def chains(trajectory):
    return [item for item in trajectory.ordered_items if isinstance(item, ScanSegment)]


def scan_signature(trajectory):
    # Both versions replace the first scan point after a successful link by the last connector
    # sample. Compare the retained scans exactly and compare connector endpoints separately.
    return [(chain.block_index, point["row_index"], point["xyz_m"], point["pixel"], point["triangle_index"])
            for chain in chains(trajectory) for point in chain.points if point.get("role") != "connector"]


def connectors(trajectory):
    result = []
    for chain in chains(trajectory):
        index = 1
        while index < len(chain.points):
            if chain.points[index].get("role") != "connector":
                index += 1
                continue
            start = index - 1
            while index < len(chain.points) and chain.points[index].get("role") == "connector":
                index += 1
            result.append((chain.block_index, chain.points[start:index]))
    return result


class OptimizedSelectionRouteTests(unittest.TestCase):
    def compare_versions(self, mesh, mask):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        source = make_export(directory.name, mesh, selection=dict(mask=mask))
        source_bytes = {path.name: path.read_bytes() for path in source.parent.iterdir() if path.is_file()}
        old, new = [plan(source, version) for version in (SELECTION_ALGORITHM, OPTIMIZED_SELECTION_ALGORITHM)]
        self.assertEqual(old.planner["algorithm_version"], SELECTION_ALGORITHM)
        self.assertEqual(new.planner["algorithm_version"], OPTIMIZED_SELECTION_ALGORITHM)
        self.assertEqual(scan_signature(old), scan_signature(new))
        self.assertEqual(old.planner["parameters"], new.planner["parameters"])
        self.assertEqual(old.planner["sweep_frame"], new.planner["sweep_frame"])
        self.assertEqual(old.source, new.source)
        self.assertEqual(old.gap_report, new.gap_report)
        self.assertEqual(old.spacing_report, new.spacing_report)
        for key in ("block_count", "planned_block_count", "incomplete_block_count", "unplannable_block_count",
                    "row_count", "segment_count", "gap_count", "gap_counts", "continuity_status", "coverage_status",
                    "mesh_fragment_count", "geometry_unconfirmed_pixel_count", "uncovered_fragment_count",
                    "connected_row_turns", "connected_detours", "link_failure_counts"):
            self.assertEqual(old.statistics[key], new.statistics[key], key)
        for before, after in zip(old.blocks["items"], new.blocks["items"]):
            for key in ("component_id", "status", "selection", "coverage", "continuity", "mesh_fragment_count",
                        "planned_fragment_count", "segment_count", "gap_count", "gap_counts", "link_failure_counts"):
                self.assertEqual(before[key], after[key], key)
        before_gaps = [item for item in old.ordered_items if isinstance(item, Gap)]
        after_gaps = [item for item in new.ordered_items if isinstance(item, Gap)]
        for before, after in zip(before_gaps, after_gaps):
            for key in ("reason", "detail", "row_index_from", "row_index_to", "chord_length_m", "scan_direction_w0"):
                self.assertEqual(getattr(before, key), getattr(after, key), key)
            self.assertEqual(before.previous_point["xyz_m"], after.previous_point["xyz_m"])
            self.assertEqual(before.next_point["xyz_m"], after.next_point["xyz_m"])
        old_links, new_links = connectors(old), connectors(new)
        self.assertEqual(len(old_links), len(new_links))
        for (old_block, before), (new_block, after) in zip(old_links, new_links):
            self.assertEqual(old_block, new_block)
            np.testing.assert_array_equal(before[0]["xyz_m"], after[0]["xyz_m"])
            np.testing.assert_allclose(before[-1]["xyz_m"], after[-1]["xyz_m"], rtol=0, atol=1e-15)
            self.assertLessEqual(_length_split(after)[1], _length_split(before)[1] + 1e-12)
        region = json.loads(source.read_text(encoding="utf-8"))["region"]
        loaded_mesh = trimesh.load(source.parent / "region_1_mesh_m.ply", process=False)
        labels = component_labels(region["selection_intent"])
        for trajectory in (old, new):
            validate_trajectory(trajectory, mesh=loaded_mesh, region=region)
            for chain in chains(trajectory):
                selected = labels == trajectory.blocks["items"][chain.block_index]["component_id"] + 1
                for previous, point in zip(chain.points, chain.points[1:]):
                    self.assertTrue(edge_pixels_allowed(previous["xyz_m"], point["xyz_m"],
                                                        np.asarray(INTRINSIC), np.asarray(IMAGE_SIZE), selected))
                    if point.get("role") == "connector":
                        face = loaded_mesh.faces[point["triangle_index"]]
                        self.assertTrue(_point_on_triangle(previous["xyz_m"], loaded_mesh.vertices, face))
                        self.assertTrue(_point_on_triangle(point["xyz_m"], loaded_mesh.vertices, face))
            payload = trajectory.to_dict()
            self.assertFalse(payload["execution_ready"])
            self.assertFalse(payload["physical_accuracy_verified"])
            self.assertIsNone(payload["orientation_fields"])
        self.assertEqual(source_bytes, {path.name: path.read_bytes() for path in source.parent.iterdir() if path.is_file()})
        return source, old, new

    def test_unselected_hole_keeps_scan_and_shortens_supported_detour(self):
        _, old, new = self.compare_versions(grid_mesh(.06, .03, .002), FULL & ~pixel_mask((98, 97, 102, 103)))
        self.assertEqual(new.statistics["gap_counts"], {})
        self.assertGreater(new.statistics["connected_detours"], 0)
        self.assertLess(new.statistics["connector_length_m"], old.statistics["connector_length_m"] - 1e-5)

    def test_tilted_surface_shortens_links_without_flattening_xyz(self):
        _, old, new = self.compare_versions(grid_mesh(.06, .03, .002, tilt_deg=25),
                                             FULL & ~pixel_mask((98, 98, 102, 102)))
        self.assertGreater(new.statistics["depth_span_m"], .001)
        self.assertLess(new.statistics["connector_length_m"], old.statistics["connector_length_m"] - 1e-5)

    def test_missing_mesh_fragments_stay_one_incomplete_selection_block(self):
        _, _, new = self.compare_versions(two_patches(), FULL)
        self.assertEqual(new.statistics["block_count"], 1)
        self.assertEqual(new.statistics["incomplete_block_count"], 1)
        self.assertEqual(new.statistics["gap_counts"], {"disconnected_surface": 1})
        self.assertGreater(new.statistics["geometry_unconfirmed_pixel_count"], 0)

    def test_multiple_selection_blocks_stay_separate_with_a_pause(self):
        _, _, new = self.compare_versions(two_patches(), pixel_mask((80, 80, 98, 120), (102, 80, 120, 120)))
        self.assertEqual(new.statistics["block_count"], 2)
        self.assertEqual(new.statistics["planned_block_count"], 2)
        self.assertEqual(new.statistics["gap_counts"], {"selection_break": 1})

    def test_export_formats_and_reproduction_use_each_saved_version(self):
        source, old, new = self.compare_versions(grid_mesh(.06, .03, .002), FULL & ~pixel_mask((98, 97, 102, 103)))
        self.assertEqual(set(old.to_dict()), set(new.to_dict()))
        self.assertEqual(set(old.planner), set(new.planner))
        self.assertNotIn("selection_algorithm", new.planner["parameters"])
        folders = []
        for label, trajectory in (("old", old), ("new", new)):
            folder = export_trajectory(trajectory, source.parent / label, include_visuals=False)
            folders.append(folder)
            before = {path.name: path.read_bytes() for path in folder.iterdir() if path.is_file()}
            with mock.patch("femto.planning.mesh_slice_planner.plan_mesh_slice", wraps=plan_mesh_slice) as replan:
                loaded = load_trajectory(folder)
            self.assertEqual(loaded.to_dict(), trajectory.to_dict())
            self.assertEqual(replan.call_count, 1)
            self.assertEqual(replan.call_args.kwargs["selection_algorithm"], trajectory.planner["algorithm_version"])
            self.assertEqual(before, {path.name: path.read_bytes() for path in folder.iterdir() if path.is_file()})
        self.assertEqual({path.name for path in folders[0].iterdir()}, {path.name for path in folders[1].iterdir()})
        for name in ("trajectory.csv", "gaps.csv", "blocks.csv"):
            self.assertEqual((folders[0] / name).read_text().splitlines()[0],
                             (folders[1] / name).read_text().splitlines()[0])

    def test_old_playback_is_validated_but_never_replanned_or_upgraded(self):
        from femto.planning.trajectory_history import prepare_playback
        with tempfile.TemporaryDirectory() as tmp:
            source = make_export(tmp, selection=dict(mask=FULL))
            old = plan(source, SELECTION_ALGORITHM)
            folder = export_trajectory(old, Path(tmp) / "saved", include_visuals=False)
            region = json.loads(source.read_text())["region"]
            model = SimpleNamespace(store=SimpleNamespace(digest=old.source["geometry_sha256"],
                                                          origin_reference=old.origin_reference),
                                    view_reference=region["view_reference"])
            with mock.patch("femto.annotation.fixed_rgb_geometry.FixedRGBModel", return_value=model), \
                    mock.patch("femto.planning.mesh_slice_planner.plan_mesh_slice", side_effect=AssertionError("replanned")):
                returned, _, playback = prepare_playback({"source": folder / "trajectory.json"}, Path(tmp), TaskControl())
            self.assertIs(returned, model)
            self.assertIn(SELECTION_ALGORITHM, playback["algorithm_notice"])
            self.assertIn("不会自动升级", playback["algorithm_notice"])
            self.assertEqual(json.loads((folder / "trajectory.json").read_text())["planner"]["algorithm_version"],
                             SELECTION_ALGORITHM)

    def test_cli_default_and_explicit_legacy_flags_dispatch_without_extra_parameters(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = make_export(tmp, selection=dict(mask=FULL))
            for flag, version in (([], ORDERED_SELECTION_ALGORITHM),
                                  (["--selection-v4-3"], BIDIRECTIONAL_SELECTION_ALGORITHM),
                                  (["--selection-v4-2"], FACE_PORTAL_SELECTION_ALGORITHM),
                                  (["--selection-v4-1"], OPTIMIZED_SELECTION_ALGORITHM),
                                  (["--selection-v4"], SELECTION_ALGORITHM),
                                  (["--mesh-blocks-v3"], "mesh-slice-serpentine-v3.1")):
                with self.subTest(flag=flag), redirect_stdout(io.StringIO()), \
                        mock.patch("femto.planning.trajectory_export.export_trajectory", return_value=Path(tmp)) as export:
                    self.assertEqual(main([str(source), "--output", tmp, *flag]), 0)
                trajectory = export.call_args.args[0]
                self.assertEqual(trajectory.planner["algorithm_version"], version)
                self.assertNotIn("selection_algorithm", trajectory.planner["parameters"])

    def test_api_keeps_v4_default_and_rejects_an_unsupported_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = make_export(tmp, selection=dict(mask=FULL))
            frozen = plan_mesh_slice(source, connect_rows=True, block_planning=True, selection_blocks=True)
            self.assertEqual(frozen.planner["algorithm_version"], SELECTION_ALGORITHM)
            with self.assertRaisesRegex(ValueError, "Unsupported selection_algorithm"):
                plan(source, "mesh-slice-serpentine-v999")
            with self.assertRaisesRegex(ValueError, "requires selection_blocks"):
                plan_mesh_slice(source, connect_rows=True, block_planning=True,
                                selection_algorithm=OPTIMIZED_SELECTION_ALGORITHM)


if __name__ == "__main__":
    unittest.main()
