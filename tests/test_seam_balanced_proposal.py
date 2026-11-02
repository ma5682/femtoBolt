"""Tests for the isolated seam-then-balanced ordering experiment."""
from copy import deepcopy
import unittest
from unittest.mock import patch

from test_verified_visit_improvement import Graph, make_segment
from femto.planning import seam_balanced_proposal as combined
from femto.planning.scan_visit_order import connect


class SeamBalancedTests(unittest.TestCase):
    def setUp(self):
        self.segments = [make_segment(i, float(i)) for i in range(5)]
        self.graph = Graph(self.segments)
        self.visits = [[i, False] for i in range(5)]
        self.links = connect(self.graph, self.segments, self.visits, lambda: None, 1000)

    def test_both_decisions_validate_and_inputs_are_unchanged(self):
        before = deepcopy((self.segments, self.visits, self.links))
        _, final, _, record = combined.propose(self.graph, self.segments, self.visits, self.links,
                                               {}, .01, lambda: None, 1000)
        visits, intermediate = combined.validate_record(record, self.segments, self.visits,
                                                         self.links, final, .01, lambda: None)
        self.assertEqual(len(visits), len(self.visits))
        self.assertTrue(intermediate)
        self.assertEqual((self.segments, self.visits, self.links), before)
        bad = deepcopy(record)
        bad['balanced']['baseline']['detours'] += 1
        with self.assertRaises(ValueError):
            combined.validate_record(bad, self.segments, self.visits, self.links, final, .01, lambda: None)

    def test_each_budget_exhaustion_retains_verified_input(self):
        error = ValueError(combined.GEOMETRY_BUDGET)
        with patch.object(combined.seam_visit_proposal, 'propose', side_effect=error), \
             patch.object(combined.balanced_visit_improvement, 'propose', side_effect=error):
            _, final, _, record = combined.propose(self.graph, self.segments, self.visits, self.links,
                                                   {}, .01, lambda: None, 1000)
        self.assertEqual(final, self.links)
        self.assertEqual(record['seam']['reason'], 'geometry_budget')
        self.assertEqual(record['balanced']['skipped'], 'geometry_budget')
        combined.validate_record(record, self.segments, self.visits, self.links, final, .01, lambda: None)

    def test_second_budget_exhaustion_preserves_seam_result(self):
        with patch.object(combined.balanced_visit_improvement, 'propose',
                          side_effect=ValueError(combined.GEOMETRY_BUDGET)):
            _, final, _, record = combined.propose(self.graph, self.segments, self.visits, self.links,
                                                   {}, .01, lambda: None, 1000)
        self.assertEqual(final, record['middle_connections'])
        combined.validate_record(record, self.segments, self.visits, self.links, final, .01, lambda: None)

    def test_unrelated_exception_and_cancellation_are_propagated(self):
        with patch.object(combined.seam_visit_proposal, 'propose', side_effect=ValueError('other')):
            with self.assertRaisesRegex(ValueError, 'other'):
                combined.propose(self.graph, self.segments, self.visits, self.links, {}, .01, lambda: None, 1000)
        def cancel(): raise RuntimeError('cancelled')
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):
            combined.propose(self.graph, self.segments, self.visits, self.links, {}, .01, cancel, 1000)


if __name__ == '__main__':
    unittest.main()
