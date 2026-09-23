"""v4.4 traversal changes do not change coverage samples or physical row reports."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import trimesh

from tests.test_mesh_slice_planner import grid_mesh, make_export, pixel_mask, two_patches
from femto.planning.mesh_slice_planner import _prepare_block, plan_mesh_slice
from femto.planning.surface_trajectory import (
    BIDIRECTIONAL_SELECTION_ALGORITHM, ORDERED_SELECTION_ALGORITHM, Gap, ScanSegment,
    validate_trajectory,
)
from femto.planning.trajectory_export import export_trajectory, load_trajectory


FULL = pixel_mask((0, 0, 199, 199))


def plan(source, version=ORDERED_SELECTION_ALGORITHM, **kwargs):
    return plan_mesh_slice(source, connect_rows=True, block_planning=True,
                           selection_blocks=True, selection_algorithm=version, **kwargs)


def points(result):
    return [p for item in result.ordered_items if isinstance(item, ScanSegment) for p in item.points]


def measured_spacing(report):
    result = deepcopy(report)
    # Only work accounting changes: v4.4 stops once all queried vertices settle.
    result['budget'].pop('expansions')
    return result


class OrderedSelectionRouteTests(unittest.TestCase):
    def test_rectangular_single_strip_preserves_existing_snake_for_every_corner(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = make_export(tmp, selection=dict(mask=FULL))
            for corner in ('top_left', 'top_right', 'bottom_left', 'bottom_right'):
                with self.subTest(corner=corner):
                    old = plan(source, BIDIRECTIONAL_SELECTION_ALGORITHM, start_corner=corner)
                    new = plan(source, start_corner=corner)
                    self.assertEqual(new.ordered_items, old.ordered_items)
                    self.assertEqual(new.row_report, old.row_report)
                    self.assertEqual(measured_spacing(new.spacing_report), measured_spacing(old.spacing_report))
                    self.assertLessEqual(new.spacing_report['budget']['expansions'], old.spacing_report['budget']['expansions'])

    def test_hole_reduces_repeated_detours_without_changing_scans_or_spacing(self):
        with tempfile.TemporaryDirectory() as tmp:
            mesh = grid_mesh(.10, .07, .002, hole=(-.016, .016, -.022, .022))
            source = make_export(tmp, mesh, selection=dict(mask=FULL))
            old = plan(source, BIDIRECTIONAL_SELECTION_ALGORITHM)
            new = plan(source)
            self.assertLess(new.statistics['connector_length_m'], old.statistics['connector_length_m'])
            self.assertLess(new.statistics['point_count'], old.statistics['point_count'])
            self.assertEqual(new.statistics['gap_count'], old.statistics['gap_count'])
            self.assertEqual(measured_spacing(new.spacing_report), measured_spacing(old.spacing_report))
            for key in ('scan_length_m', 'row_count', 'block_count', 'mesh_fragment_count',
                        'geometry_unconfirmed_pixel_count', 'coverage_status', 'uncovered_fragment_count'):
                self.assertEqual(new.statistics[key], old.statistics[key])
            self.assertEqual(points(new)[0], points(old)[0])
            rows = [p['row_index'] for p in points(new)]
            self.assertTrue(any(b < a for a, b in zip(rows, rows[1:])))

    def test_all_original_samples_and_row_reports_survive_reordering(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = make_export(tmp, grid_mesh(.08, .05, .002, hole=(-.012, .012, -.012, .012)),
                                 selection=dict(mask=FULL))
            original = _prepare_block
            snapshots = []

            def capture(*args, **kwargs):
                prepared = original(*args, **kwargs)
                snapshots.append((prepared, deepcopy(prepared['ordered_records'])))
                return prepared

            with mock.patch('femto.planning.mesh_slice_planner._prepare_block', capture):
                result = plan(source)
            observed = Counter((p['row_index'], tuple(p['xyz_m'])) for p in points(result))
            for prepared, snapshot in snapshots:
                self.assertEqual(prepared['ordered_records'], snapshot)
                expected = Counter((s.row_index, tuple(p['xyz_m']))
                                   for s in prepared['segments'] for p in s.points)
                self.assertFalse(expected - observed)
            self.assertEqual([p['route_index'] for p in points(result)], list(range(len(points(result)))))

    def test_unselected_pixels_and_forbidden_polygon_still_constrain_every_edge(self):
        with tempfile.TemporaryDirectory() as tmp:
            mask = FULL & ~pixel_mask((98, 96, 102, 104))
            source = make_export(tmp, grid_mesh(.08, .05, .002), selection=dict(mask=mask),
                                 forbidden=[[98, 96], [102, 96], [102, 104], [98, 104]])
            before = {p.name: p.read_bytes() for p in Path(tmp).iterdir() if p.is_file()}
            result = plan(source)
            manifest = json.loads(source.read_text(encoding='utf-8'))
            mesh = trimesh.load(source.parent / manifest['files']['mesh_file'], process=False)
            validate_trajectory(result, mesh=mesh, region=manifest['region'])
            self.assertEqual(before, {p.name: p.read_bytes() for p in Path(tmp).iterdir() if p.is_file()})
            self.assertFalse(result.to_dict()['execution_ready'])
            self.assertFalse(result.to_dict()['physical_accuracy_verified'])
            self.assertIsNone(result.to_dict()['orientation_fields'])

    def test_mesh_fragments_do_not_become_connected_or_new_work_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = make_export(tmp, two_patches(), selection=dict(mask=FULL))
            result = plan(source)
            self.assertEqual(result.statistics['block_count'], 1)
            self.assertEqual(result.statistics['mesh_fragment_count'], 2)
            self.assertEqual(result.blocks['items'][0]['status'], 'incomplete')
            gaps = [g for g in result.ordered_items if isinstance(g, Gap)]
            self.assertEqual([g.reason for g in gaps], ['disconnected_surface'])
            self.assertEqual(gaps[0].detail['link_failure_reason'], 'disconnected_measured_surface')

    def test_work_blocks_remain_separate_with_actual_endpoint_row_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            mask = pixel_mask((80, 90, 97, 110), (102, 90, 119, 110))
            source = make_export(tmp, two_patches(), selection=dict(mask=mask))
            result = plan(source)
            self.assertEqual(result.statistics['block_count'], 2)
            items = result.ordered_items
            for index, gap in enumerate(items):
                if isinstance(gap, Gap):
                    self.assertEqual(gap.reason, 'selection_break')
                    self.assertEqual(gap.row_index_from, items[index - 1].points[-1]['row_index'])
                    self.assertEqual(gap.row_index_to, items[index + 1].points[0]['row_index'])

    def test_old_version_never_calls_reordering_and_both_saved_versions_reproduce(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = make_export(tmp, grid_mesh(.06, .04, .002, hole=(-.008, .008, -.008, .008)),
                                 selection=dict(mask=FULL))
            with mock.patch('femto.planning.surface_scan_order.order_scan_records',
                            side_effect=AssertionError('old version reordered')):
                old = plan(source, BIDIRECTIONAL_SELECTION_ALGORITHM)
            new = plan(source)
            for result in (old, new):
                folder = export_trajectory(result, Path(tmp) / result.planner['algorithm_version'], include_visuals=False)
                before = {p.name: p.read_bytes() for p in folder.iterdir()}
                self.assertEqual(load_trajectory(folder).to_dict(), result.to_dict())
                self.assertEqual(before, {p.name: p.read_bytes() for p in folder.iterdir()})


if __name__ == '__main__':
    unittest.main()
