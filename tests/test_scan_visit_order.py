import itertools
import unittest
from copy import deepcopy

import numpy as np

from femto.planning.surface_trajectory import ScanSegment
from femto.planning.scan_visit_order import propose, oriented, endpoint_cost, _directions, optimize


def segment(index, a, b, block=0):
    return ScanSegment(index, index, [dict(xyz_m=list(a), role='scan', row_index=index),
                                     dict(xyz_m=list(b), role='scan', row_index=index)], block)


class ScanVisitTests(unittest.TestCase):
    def test_direction_program_matches_exhaustive_and_keeps_start(self):
        endpoints = np.random.default_rng(8).normal(size=(7, 2, 3))
        visits = _directions(endpoints, list(range(7)), lambda: None)
        exhaustive = min(endpoint_cost(endpoints, list(enumerate((False,)+choices)))
                         for choices in itertools.product((False, True), repeat=6))
        self.assertAlmostEqual(endpoint_cost(endpoints, visits), exhaustive)
        self.assertEqual(visits[0], (0, False))

    def test_visits_improve_reordered_rows_preserve_blocks_and_source(self):
        scans = [segment(i, (0, y, 0), (1, y, 0)) for i, y in enumerate((0, 4, 1, 3, 2))]
        scans += [segment(5, (0, 0, 1), (1, 0, 1), 1)]
        original = deepcopy(scans)
        visits = propose(scans)
        self.assertEqual(sorted(i for i, _ in visits), list(range(6)))
        self.assertEqual(visits[0], (0, False))
        self.assertEqual(visits[-1], (5, False))
        self.assertEqual(scans, original)
        endpoints = np.asarray([[s.points[0]['xyz_m'], s.points[-1]['xyz_m']] for s in scans])
        self.assertLess(endpoint_cost(endpoints, visits[:-1]), endpoint_cost(endpoints, [(i, False) for i in range(5)]))
        self.assertEqual(propose(scans), visits)

    def test_reverse_preserves_edge_roles_and_is_independent(self):
        scan = segment(0, (0, 0, 0), (1, 0, 0))
        scan.points.extend([dict(xyz_m=[2, 0, 0], role='connector', row_index=1),
                            dict(xyz_m=[3, 0, 0], role='scan', row_index=1)])
        scan.points[1]['role'] = 'connector'
        reversed_scan = oriented(scan, True)
        self.assertEqual([p['role'] for p in reversed_scan.points], ['scan', 'scan', 'connector', 'connector'])
        restored = oriented(reversed_scan, True)
        self.assertEqual(restored.points, scan.points)
        reversed_scan.points[0]['xyz_m'][0] = 100
        self.assertEqual(scan.points[-1]['xyz_m'][0], 3)

    def test_cancel_propagates_and_empty(self):
        self.assertEqual(propose([]), [])
        def cancel():
            raise RuntimeError('cancel')
        with self.assertRaisesRegex(RuntimeError, 'cancel'):
            propose([segment(0, (0, 0, 0), (1, 0, 0))], cancel)

    def test_shorter_optimistic_proposal_is_rejected_if_surface_links_fail(self):
        scans = [segment(i, (0, y, 0), (1, y, 0)) for i, y in enumerate((0, 4, 1, 3, 2))]
        class Graph:
            def route(self, left, right, *, left_reverse, right_reverse):
                if right != left+1 or left_reverse or right_reverse:
                    return None
                return [dict(scans[right].points[0], role='connector')]
        chains, links, added, decision = optimize(Graph(), scans, lambda: None, 100)
        self.assertFalse(decision['adopted'])
        self.assertEqual(decision['visits'], [[i, False] for i in range(5)])
        self.assertEqual(len(chains), 1)
        self.assertEqual(added, 4)
        self.assertEqual(links, decision['baseline_connections'])
