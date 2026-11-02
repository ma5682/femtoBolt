"""Equivalent current P2/P3 preparation never carries a proof between visits."""
from unittest.mock import patch
import unittest

import numpy as np

from femto.geometry import exact_surface as exact
from femto.planning import coverage_certificate as coverage, coverage_partition, footprint_model
from femto.planning.coverage_prepared import PreparedResidualEngine
from femto.planning.coverage_residual import ResidualEngine
from femto.planning.coverage_recalculation import RecalculationEngine
from femto.resource_policy import MemorySnapshot, GiB
from test_coverage_certificate import triangle_domain, seed


class PreparedCoverageTests(unittest.TestCase):
    def setUp(self):
        memory = patch('femto.resource_policy.memory_snapshot', return_value=MemorySnapshot(16*GiB, 8*GiB, 'simulated'))
        memory.start(); self.addCleanup(memory.stop)
        self.domain = triangle_domain()
        self.records = [seed()]
        self.model = footprint_model.nominal(125.)
        self.budget = coverage.CoverageBudget(max_depth=3)
        self.guide = coverage.evaluate(self.domain, self.records, self.model,
            budget=self.budget, algorithm=coverage.CURRENT_ALGORITHM)

    def evaluate(self, algorithm, frontier=None):
        records = self.records if algorithm == coverage.RECALCULATION_ALGORITHM else self.records + [seed((.125, 0, 0))]
        return coverage.evaluate(self.domain, records, self.model, budget=self.budget,
            algorithm=algorithm, guide=self.guide, _frontier=frontier)

    def test_both_current_algorithms_match_reference_arrays_and_work_counts(self):
        for algorithm in (coverage.RECALCULATION_ALGORITHM, coverage.INTERLEAVED_ALGORITHM):
            with self.subTest(algorithm=algorithm):
                with patch.object(PreparedResidualEngine, '_visit', ResidualEngine._visit):
                    expected = self.evaluate(algorithm)
                actual = self.evaluate(algorithm)
                self.assertEqual(actual.metadata, expected.metadata)
                for key in actual.arrays:np.testing.assert_array_equal(actual.arrays[key], expected.arrays[key])
                self.assertEqual(coverage_partition.compare(self.guide, actual, self.domain.areas)['lost_weight_units'], 0)

    def test_partial_work_frontiers_preserve_reference_state_and_replay(self):
        for algorithm in (coverage.RECALCULATION_ALGORITHM, coverage.INTERLEAVED_ALGORITHM):
            complete = self.evaluate(algorithm)
            for stop in (17, complete.metadata['metrics']['steps'] // 2, complete.metadata['metrics']['steps'] - 1):
                frontier = dict(stop_reason='time_budget', steps=stop)
                with self.subTest(algorithm=algorithm, stop=stop):
                    with patch.object(PreparedResidualEngine, '_visit', ResidualEngine._visit):
                        expected = self.evaluate(algorithm, frontier)
                    actual = self.evaluate(algorithm, frontier)
                    self.assertEqual(actual.metadata, expected.metadata)
                    records = self.records if algorithm == coverage.RECALCULATION_ALGORITHM else self.records + [seed((.125, 0, 0))]
                    coverage.validate(actual, self.domain, records, self.model,
                        **({'guide': self.guide} if algorithm == coverage.INTERLEAVED_ALGORITHM else {}))

    def test_candidate_preparation_tracks_current_seed_subset_and_is_immutable(self):
        records = self.records + [seed((.125, 0, 0))]
        engine = PreparedResidualEngine(self.domain, records, self.model, self.budget, lambda: None, None, guide=self.guide)
        engine.seed()
        all_ids, all_points = engine._candidate_inputs(0, None)
        limited = {0}
        ids, points = engine._candidate_inputs(0, sum(1 << i for i in limited))
        np.testing.assert_array_equal(ids, [0])
        limited.add(1)
        next_ids, _ = engine._candidate_inputs(0, sum(1 << i for i in limited))
        np.testing.assert_array_equal(next_ids, all_ids)
        np.testing.assert_array_equal(ids, [0])
        for array in (all_ids, all_points, ids, points, engine._radii):
            self.assertFalse(array.flags.writeable)
        self.assertGreater(engine._candidate_inputs.cache_info().maxsize, 0)
        self.assertLessEqual(engine._candidate_inputs.cache_info().maxsize, 4096)

    def test_new_instance_does_not_reuse_old_nominal_radius(self):
        engines = [RecalculationEngine(self.domain, self.records, model, self.budget, lambda: None, None, guide=self.guide)
            for model in (self.model, footprint_model.nominal(100.))]
        for engine in engines:engine.seed()
        self.assertNotEqual(engines[0]._radii[0], engines[1]._radii[0])
        self.assertIsNot(engines[0]._candidate_inputs(0, None)[1], engines[1]._candidate_inputs(0, None)[1])

    def test_dense_face_does_not_grow_the_prepared_array_cache(self):
        records = [seed() for _ in range(160)]
        engine = PreparedResidualEngine(self.domain, records, self.model, self.budget, lambda: None, None, guide=self.guide)
        engine.seed(); self.assertTrue(engine._prepare()); engine._pixel_bounds()
        engine._visit(0)
        self.assertEqual(engine.meter.counts['spatial_candidates'], 160)
        self.assertEqual(engine._candidate_inputs.cache_info().currsize, 0)

    def test_restriction_key_size_depends_on_local_candidates(self):
        engine = PreparedResidualEngine(self.domain, self.records, self.model, self.budget, lambda: None, None, guide=self.guide)
        engine.seed(); self.assertTrue(engine._prepare()); engine._pixel_bounds()
        cache = engine._candidate_inputs
        with patch.object(engine, '_candidate_inputs', wraps=cache) as prepared:
            engine.limited_seeds = set(range(20000)); engine._visit(0)
            large_key = prepared.call_args.args[1]
            engine.limited_seeds = {0}; engine._visit(0)
            small_key = prepared.call_args.args[1]
        self.assertEqual(large_key, small_key)
        self.assertIs(type(large_key), int)
        self.assertLessEqual(large_key.bit_length(), 128)
        self.assertEqual(cache.cache_info().currsize, 1)

    def test_interrupted_new_link_retains_the_current_reproved_witness(self):
        engine = RecalculationEngine(self.domain, self.records, self.model, self.budget, lambda: None, None, guide=self.guide)
        engine.seed(); self.assertTrue(engine._prepare()); engine._pixel_bounds(); self.assertTrue(engine._reprove())
        number = int(engine.old_confirmed[0]); witness = engine.nodes[number].witness
        with patch.object(engine, '_link', side_effect=coverage._BudgetStop), self.assertRaises(coverage._BudgetStop):
            engine._visit(number)
        arrays, remap = engine.leaf_arrays()
        self.assertEqual(arrays['status'][remap[number]], 1)
        self.assertEqual(arrays['witness'][remap[number]], witness)

    def test_shared_coefficients_have_no_geometry_and_keep_exact_dyadic_values(self):
        rng = np.random.default_rng(51010)
        for depth in range(9):
            for code in rng.integers(0, 4**depth, 12):
                code = int(code)
                actual = exact._dyadic_coefficients(depth, code)
                expected = np.asarray(exact.descend(exact.IDENTITY, depth, code), float)
                np.testing.assert_array_equal(actual, expected)
                self.assertFalse(actual.flags.writeable)
        self.assertEqual(exact._dyadic_coefficients.cache_info().maxsize, 4096)


if __name__ == '__main__':unittest.main()
