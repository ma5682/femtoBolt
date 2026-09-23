"""Scan strip ordering changes visits, never samples, row identities or geometry."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import unittest

import numpy as np

from femto.planning.surface_scan_order import order_scan_records


FRAME = {'origin': [0., 0., 0.], 'e1': [1., 0., 0.]}


def record(tag, row, low, high, *, reverse=False, y=None):
    y = row if y is None else y
    points = [dict(xyz_m=[float(x), float(y), 1.], pixel=[float(x + 100), float(y + 100)],
                   triangle_index=row * 100 + index, row_index=row, role='scan')
              for index, x in enumerate((low, (low + high) / 2, high))]
    if reverse:
        points.reverse()
    return dict(tag=tag, points=points, row_index=row, slice_index=row,
                level_m=float(row), component=0, metadata={'evidence': [tag]})


def signatures(records):
    return Counter((item['tag'], item['row_index'], item['slice_index'],
                    tuple(point['xyz_m']), tuple(point['pixel']), point['triangle_index'], point['row_index'])
                   for item in records for point in item['points'])


def chord_length(records):
    return sum(float(np.linalg.norm(np.asarray(right['points'][0]['xyz_m']) - left['points'][-1]['xyz_m']))
               for left, right in zip(records, records[1:]))


class SurfaceScanOrderTests(unittest.TestCase):
    def ordered(self, records, frame=FRAME):
        before = deepcopy(records)
        result = order_scan_records(records, frame, lambda: None)
        self.assertEqual(records, before)
        self.assertEqual(Counter(item['tag'] for item in result), Counter(item['tag'] for item in records))
        self.assertEqual(signatures(result), signatures(records))
        if records:
            self.assertEqual(result[0], records[0])
        return result

    def test_plain_rectangle_keeps_original_serpentine_exactly(self):
        records = [record(str(row), row, 0, 10, reverse=bool(row % 2)) for row in range(5)]
        self.assertEqual(self.ordered(records), records)

    def test_first_scan_direction_and_bottom_up_start_are_preserved(self):
        records = [record(str(row), row, 0, 10, reverse=not bool(index % 2))
                   for index, row in enumerate((4, 3, 2, 1, 0))]
        self.assertEqual(self.ordered(records), records)

    def test_single_row_multiple_segments_are_each_visited_once(self):
        records = [record('a', 0, 0, 1), record('b', 0, 3, 4), record('c', 0, 6, 7)]
        self.assertEqual(self.ordered(records), records)

    def test_hole_sides_are_swept_as_strips_instead_of_crossing_every_row(self):
        records = []
        for row in range(4):
            halves = [('L', 0, 2), ('R', 8, 10)]
            if row % 2:
                halves.reverse()
            records.extend(record(f'{tag}{row}', row, low, high, reverse=bool(row % 2))
                           for tag, low, high in halves)
        result = self.ordered(records)
        self.assertEqual([item['tag'] for item in result], ['L0', 'L1', 'L2', 'L3', 'R3', 'R2', 'R1', 'R0'])
        self.assertLess(chord_length(result), chord_length(records))
        # This is a cheap visit-cost comparison, not an assertion that chords
        # across the hole are valid motion; downstream routing still decides.

    def test_split_and_merge_terminate_strips_without_losing_branches(self):
        records = [record('top', 0, 0, 10), record('R1', 1, 6, 10, reverse=True),
                   record('L1', 1, 0, 4, reverse=True), record('L2', 2, 0, 4),
                   record('R2', 2, 6, 10), record('bottom', 3, 0, 10, reverse=True)]
        result = self.ordered(records)
        self.assertEqual([item['tag'] for item in result], ['top', 'R1', 'R2', 'bottom', 'L2', 'L1'])

    def test_nonadjacent_physical_slices_do_not_create_a_strip(self):
        records = [record('a', 0, 0, 1), record('near', 0, 2, 3), record('far', 2, 0, 1, y=100)]
        self.assertEqual([item['tag'] for item in self.ordered(records)], ['a', 'near', 'far'])

    def test_touching_intervals_do_not_establish_positive_overlap(self):
        records = [record('a', 0, 0, 1), record('near', 0, 1.1, 1.2), record('touch', 1, 1, 2, y=100)]
        self.assertEqual([item['tag'] for item in self.ordered(records)], ['a', 'near', 'touch'])

    def test_order_uses_w0_geometry_not_reference_image_coordinates(self):
        records = [record('a', 0, 0, 2), record('far', 0, 8, 10), record('b', 1, 0, 2, reverse=True)]
        expected = self.ordered(records)
        changed = deepcopy(records)
        for index, item in enumerate(changed):
            item.update(mean_u=-1000 * index, mean_v=1000 * index)
            for point in item['points']:
                point['pixel'] = [-point['pixel'][0], 9000 - point['pixel'][1]]
        actual = self.ordered(changed)
        self.assertEqual([item['tag'] for item in actual], [item['tag'] for item in expected])
        self.assertEqual([[point['xyz_m'] for point in item['points']] for item in actual],
                         [[point['xyz_m'] for point in item['points']] for item in expected])

    def test_symmetric_distance_ties_use_stable_original_indices(self):
        records = [record('start', 0, -1, 0), record('first', 2, 1, 2, y=1), record('second', 4, 1, 2, y=-1)]
        expected = self.ordered(records)
        self.assertEqual([item['tag'] for item in expected], ['start', 'first', 'second'])
        for _ in range(5):
            self.assertEqual(self.ordered(records), expected)

    def test_every_returned_record_point_and_nested_value_is_independent(self):
        records = [record('a', 0, 0, 2), record('b', 1, 0, 2, reverse=True)]
        result = self.ordered(records)
        for original, copied in zip(records, result):
            self.assertIsNot(original, copied)
            for left, right in zip(original['points'], copied['points']):
                self.assertIsNot(left, right)
                self.assertIsNot(left['xyz_m'], right['xyz_m'])
                self.assertIsNot(left['pixel'], right['pixel'])
        before = deepcopy(records)
        result[0]['points'][0]['xyz_m'][0] = 999
        result[0]['points'][0]['pixel'][0] = 999
        result[0]['metadata']['evidence'].append('changed')
        self.assertEqual(records, before)

    def test_noncanonical_interior_start_is_still_first_and_no_scan_is_lost(self):
        records = [record('middle', 1, 0, 2), record('top', 0, 0, 2), record('bottom', 2, 0, 2)]
        self.assertEqual(self.ordered(records)[0], records[0])

    def test_cancellation_interrupts_ordering_without_mutating_input(self):
        records = [record(str(row), row, 0, 2) for row in range(20)]
        before = deepcopy(records)
        calls = 0
        def cancelled():
            nonlocal calls
            calls += 1
            if calls == 10:
                raise RuntimeError('cancel requested')
        with self.assertRaisesRegex(RuntimeError, 'cancel requested'):
            order_scan_records(records, FRAME, cancelled)
        self.assertEqual(records, before)

    def test_empty_input_is_supported(self):
        self.assertEqual(self.ordered([]), [])


if __name__ == '__main__':
    unittest.main()
