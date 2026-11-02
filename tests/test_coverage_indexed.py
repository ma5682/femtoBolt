"""Exact indexed connectivity keeps proof rules and legacy replay separate."""
from fractions import Fraction as Q
import math
import random
import unittest
from unittest.mock import patch

import numpy as np

from femto.planning import coverage_certificate as coverage, footprint_model, coverage_partition
from femto.planning.coverage_indexed import ExactIntervalIndex, POLICY
from femto.planning.planning_revision import digest
from test_coverage_certificate import triangle_domain, seed, reseal


class ExactIntervalIndexTests(unittest.TestCase):
    def test_rational_overlaps_match_linear_scan_in_insertion_order(self):
        rng = random.Random(9401)
        index, rows = ExactIntervalIndex(), []
        for number in range(1000):
            low = Q(rng.randrange(1000), 1000)
            high = low + Q(rng.randrange(1, 100), 1000)
            expected = [n for a, b, n in rows if min(high, b) > max(low, a)]
            self.assertEqual(index.overlaps(low, high, lambda:None), expected)
            index.add(low, high, number, lambda:None)
            rows.append((low, high, number))

    def test_touching_and_positive_rational_gaps_do_not_connect(self):
        index = ExactIntervalIndex()
        index.add(Q(0), Q(1,2), 0, lambda:None)
        self.assertEqual(index.overlaps(Q(1,2), Q(1), lambda:None), [])
        self.assertEqual(index.overlaps(Q(1,2)+Q(1,2**400), Q(1), lambda:None), [])
        self.assertEqual(index.overlaps(Q(1,2)-Q(1,2**400), Q(1), lambda:None), [0])

    def test_dense_monotone_input_stays_balanced_and_prunes_disjoint_intervals(self):
        index = ExactIntervalIndex()
        for number in range(4096):
            index.add(Q(number), Q(number+1), number, lambda:None)
        work = []
        self.assertEqual(index.overlaps(Q(4000), Q(4001), lambda:work.append(1)), [4000])
        self.assertLess(len(work), 64)
        self.assertLessEqual(index.root.height, 2*math.ceil(math.log2(index.count+1)))
        self.assertEqual(index.count, 4096)


class IndexedCoverageTests(unittest.TestCase):
    def evaluate(self, algorithm, **kwargs):
        return coverage.evaluate(triangle_domain(), [seed()], footprint_model.nominal(125),
            algorithm=algorithm, **kwargs)

    def test_complete_result_preserves_legacy_confirmation_and_independent_replay(self):
        legacy = self.evaluate(coverage.CURRENT_ALGORITHM)
        current = self.evaluate(coverage.INDEXED_ALGORITHM)
        self.assertEqual(current.metadata['inputs']['search_policy'], POLICY)
        self.assertEqual(coverage_partition.compare(legacy, current, triangle_domain().areas)['lost_weight_units'], 0)
        for key in current.arrays:np.testing.assert_array_equal(current.arrays[key], legacy.arrays[key])
        coverage.validate(current, triangle_domain(), [seed()], footprint_model.nominal(125))
        coverage.validate(legacy, triangle_domain(), [seed()], footprint_model.nominal(125))

    def test_partial_frontier_and_adjacency_stop_preserve_full_partition_and_replay(self):
        for budget in (coverage.CoverageBudget(max_adjacencies=0), coverage.CoverageBudget(max_proofs=1)):
            result = self.evaluate(coverage.INDEXED_ALGORITHM, budget=budget)
            coverage_partition.intervals(result, 1)
            coverage.validate(result, triangle_domain(), [seed()], footprint_model.nominal(125))
        complete = self.evaluate(coverage.INDEXED_ALGORITHM)
        result = self.evaluate(coverage.INDEXED_ALGORITHM,
            _frontier=dict(stop_reason='time_budget', steps=complete.metadata['metrics']['steps']//2))
        coverage.validate(result, triangle_domain(), [seed()], footprint_model.nominal(125))

    def test_integer_index_matches_fraction_comparison_at_fixed_frontiers(self):
        complete = self.evaluate(coverage.INDEXED_ALGORITHM)
        for frontier in (None, dict(stop_reason='time_budget', steps=17),
                         dict(stop_reason='time_budget', steps=complete.metadata['metrics']['steps']//2)):
            with patch('femto.planning.coverage_indexed._less',
                       lambda a,b: Q(*a) < Q(*b)):
                reference = self.evaluate(coverage.INDEXED_ALGORITHM, _frontier=frontier)
            current = self.evaluate(coverage.INDEXED_ALGORITHM, _frontier=frontier)
            self.assertEqual(current.metadata, reference.metadata)
            for key in current.arrays:
                np.testing.assert_array_equal(current.arrays[key], reference.arrays[key])

    def test_cancellation_and_policy_relabelling_do_not_install_a_certificate(self):
        def cancel():raise RuntimeError('cancelled')
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):
            self.evaluate(coverage.INDEXED_ALGORITHM, check=cancel)
        result = self.evaluate(coverage.INDEXED_ALGORITHM)
        metadata = dict(result.metadata)
        metadata['inputs'] = dict(metadata['inputs'], search_policy=dict(POLICY, adjacency_budget='unmetered'))
        metadata['coverage_computation_sha256'] = digest(metadata['inputs'])
        forged = reseal(result, metadata=metadata)
        with self.assertRaises(ValueError):
            coverage.validate(forged, triangle_domain(), [seed()], footprint_model.nominal(125))


if __name__ == '__main__':unittest.main()
