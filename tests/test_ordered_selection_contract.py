"""v4.4 retains row identities while allowing non-monotonic visit order."""
from __future__ import annotations

from copy import deepcopy
import csv
from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from femto.planning.surface_trajectory import (
    BIDIRECTIONAL_SELECTION_ALGORITHM, ORDERED_SELECTION_ALGORITHM, ScanSegment,
    to_player_plan, validate_trajectory,
)
from femto.planning.surface_visit_validation import player_spacing_report, validate_visit_rows
from femto.planning.trajectory_export import _write_core
from tests.test_face_portal_selection_routes import coarse_selected_export, plan
from tests.test_mesh_slice_planner import make_export, two_patches


def scan(row, face, x=0.):
    return dict(row_index=row, triangle_index=face, xyz_m=[x, 0., 1.], role='scan')


def fragment_fixture():
    blocks = [dict(block_index=0, status='incomplete', row_index_first=0, row_count=4, fragments=[
        dict(status='planned', mesh_component_id=0, row_index_first=0, row_count=2,
             face_index_min=0, face_index_max=0),
        dict(status='planned', mesh_component_id=1, row_index_first=2, row_count=2,
             face_index_min=1, face_index_max=1)])]
    segments = [ScanSegment(0, 1, [scan(1, 0), scan(0, 0, .001)], 0),
                ScanSegment(1, 3, [scan(3, 1, .1), scan(2, 1, .101)], 0)]
    gaps = [SimpleNamespace(row_index_from=0, row_index_to=3)]
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    return blocks, segments, gaps, faces


class OrderedSelectionContractTests(unittest.TestCase):
    def test_descending_visit_rows_are_valid_within_their_true_fragment(self):
        blocks, segments, gaps, faces = fragment_fixture()
        validate_visit_rows(blocks, segments, gaps, faces=faces)
        validate_visit_rows(blocks, segments, gaps)

    def test_negative_noninteger_and_out_of_block_rows_are_rejected(self):
        for row in (-1, 4, True, 1.0):
            with self.subTest(row=row):
                blocks, segments, gaps, faces = fragment_fixture()
                segments[0].points[-1]['row_index'] = row
                with self.assertRaisesRegex(ValueError, 'outside its work block'):
                    validate_visit_rows(blocks, segments, gaps, faces=faces)

    def test_row_owned_by_another_measured_fragment_is_rejected(self):
        blocks, segments, gaps, faces = fragment_fixture()
        segments[0].points[-1]['row_index'] = 2
        with self.assertRaisesRegex(ValueError, 'outside its measured mesh fragment'):
            validate_visit_rows(blocks, segments, gaps, faces=faces)
        with self.assertRaisesRegex(ValueError, 'no compatible mesh fragment'):
            validate_visit_rows(blocks, segments, gaps)

    def test_public_validation_rejects_tampered_row_in_bound_mesh_fragment(self):
        with tempfile.TemporaryDirectory() as tmp:
            mesh = two_patches()
            source = make_export(tmp, mesh, selection=dict(mask=np.ones((200, 200), dtype=bool)))
            trajectory = plan(source, BIDIRECTIONAL_SELECTION_ALGORITHM)
            trajectory.planner['algorithm_version'] = ORDERED_SELECTION_ALGORITHM
            for index in range(1, len(trajectory.ordered_items), 2):
                before = trajectory.ordered_items[index - 1].points[-1]
                after = trajectory.ordered_items[index + 1].points[0]
                trajectory.ordered_items[index] = replace(trajectory.ordered_items[index],
                    row_index_from=before['row_index'], row_index_to=after['row_index'])
            validate_trajectory(trajectory, mesh=mesh)
            other_fragment = trajectory.blocks['items'][0]['fragments'][1]['row_index_first']
            trajectory.ordered_items[0].points[-1]['row_index'] = other_fragment
            with self.assertRaisesRegex(ValueError, 'outside its measured mesh fragment'):
                validate_trajectory(trajectory, mesh=mesh)

    def test_overlapping_or_missing_fragment_row_ranges_are_rejected(self):
        for field, value in (('row_index_first', 1), ('row_count', 0), ('mesh_component_id', 0)):
            with self.subTest(field=field):
                blocks, segments, gaps, faces = fragment_fixture()
                blocks[0]['fragments'][1][field] = value
                with self.assertRaisesRegex(ValueError, 'fragment row'):
                    validate_visit_rows(blocks, segments, gaps, faces=faces)

    def test_chain_cannot_join_two_mesh_fragments_even_when_each_row_is_in_range(self):
        blocks, segments, gaps, faces = fragment_fixture()
        points = segments[0].points + segments[1].points
        with self.assertRaisesRegex(ValueError, 'cannot bridge disconnected mesh fragments'):
            validate_visit_rows(blocks, [ScanSegment(0, 1, points, 0)], [], faces=faces)

    def test_chain_header_and_gap_rows_refer_to_actual_endpoints_not_minimum_or_header(self):
        blocks, segments, gaps, faces = fragment_fixture()
        bad_header = [replace(segments[0], row_index=0), segments[1]]
        with self.assertRaisesRegex(ValueError, 'header must identify its first point'):
            validate_visit_rows(blocks, bad_header, gaps, faces=faces)
        gaps[0].row_index_from = segments[0].row_index
        with self.assertRaisesRegex(ValueError, 'actual endpoint rows'):
            validate_visit_rows(blocks, segments, gaps, faces=faces)

    def test_old_version_keeps_its_original_row_lower_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = plan(coarse_selected_export(tmp), BIDIRECTIONAL_SELECTION_ALGORITHM)
            first = old.ordered_items[0]
            self.assertGreater(old.statistics['row_count'], 1)
            changed = deepcopy(old)
            points = changed.ordered_items[0].points
            points[0]['row_index'] = 1
            changed.ordered_items[0] = replace(changed.ordered_items[0], row_index=1)
            with self.assertRaisesRegex(ValueError, 'row binding'):
                validate_trajectory(changed)
            # The v4.4 row-identity validator accepts non-monotonic identities;
            # this isolated check does not change or weaken any older contract.
            blocks = old.blocks['items']
            validate_visit_rows(blocks, changed.ordered_items[::2], changed.ordered_items[1::2])
            self.assertEqual(first.row_index, 0)

    def test_only_new_version_csv_writes_each_points_original_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = plan(coarse_selected_export(tmp), BIDIRECTIONAL_SELECTION_ALGORITHM)
            new = deepcopy(old)
            new.planner['algorithm_version'] = ORDERED_SELECTION_ALGORITHM
            rows = {}
            for name, trajectory in [('old', old), ('new', new)]:
                folder = Path(tmp) / name
                folder.mkdir()
                _write_core(folder, trajectory, lambda: None)
                with (folder / 'trajectory.csv').open(newline='', encoding='utf-8') as stream:
                    rows[name] = list(csv.DictReader(stream))
            self.assertEqual(list(rows['old'][0]), list(rows['new'][0]))
            expected = [p['row_index'] for item in new.ordered_items if isinstance(item, ScanSegment) for p in item.points]
            self.assertEqual([int(row['row_index']) for row in rows['new']], expected)
            self.assertGreater(len(set(expected)), 1)
            self.assertEqual({int(row['row_index']) for row in rows['old']}, {old.ordered_items[0].row_index})

    def test_exact_spacing_lookup_survives_reordering_and_reversal_without_mutation(self):
        report = dict(scan_segment_rows=[4, 5], measurements=[
            dict(chain_index=1, sample_index=9, xyz_m=[.2, 0., 1.], level_m=.025)])
        original = deepcopy(report)
        points = [[scan(5, 0, .2), scan(5, 0, .1), scan(4, 0, .3)]]
        adapted = player_spacing_report(report, points)
        row = adapted['measurements'][0]
        self.assertEqual(row['player_location'], dict(status='unique', chain_index=0, sample_index=0))
        self.assertEqual(row['chain_index'], 1)
        self.assertEqual(row['sample_index'], 9)
        self.assertEqual(report, original)

    def test_spacing_lookup_never_uses_nearest_point_or_another_row(self):
        report = dict(scan_segment_rows=[5], measurements=[dict(chain_index=0, sample_index=0, xyz_m=[.2, 0., 1.])])
        for points in ([[scan(5, 0, .2 + 1e-15)]], [[scan(4, 0, .2)]],
                       [[scan(5, 0, .2), scan(5, 0, .2)]]):
            with self.subTest(points=points):
                location = player_spacing_report(report, points)['measurements'][0]['player_location']
                self.assertIn(location['status'], ('unavailable', 'ambiguous'))
                self.assertNotIn('chain_index', location)
                self.assertIn('未使用最近邻', location['explanation'])

    def test_spacing_lookup_prefers_original_scan_evidence_over_connector_duplicate(self):
        report = dict(scan_segment_rows=[5], measurements=[dict(chain_index=0, sample_index=0, xyz_m=[.2, 0., 1.])])
        connector = dict(scan(5, 0, .2), role='connector')
        points = [[connector, scan(5, 0, .2)]]
        location = player_spacing_report(report, points)['measurements'][0]['player_location']
        self.assertEqual(location, dict(status='unique', chain_index=0, sample_index=1))

    def test_player_adapter_is_versioned_and_does_not_change_saved_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = plan(coarse_selected_export(tmp), BIDIRECTIONAL_SELECTION_ALGORITHM)
            old_player = to_player_plan(old)
            self.assertIs(old_player['spacing_report'], old.spacing_report)
            new = deepcopy(old)
            new.planner['algorithm_version'] = ORDERED_SELECTION_ALGORITHM
            before = deepcopy(new.spacing_report)
            new_player = to_player_plan(new)
            self.assertIsNot(new_player['spacing_report'], new.spacing_report)
            self.assertEqual(new.spacing_report, before)
            self.assertTrue(all('player_location' in row for row in new_player['spacing_report']['measurements']))


if __name__ == '__main__':
    unittest.main()
