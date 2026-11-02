"""Simulated resource conditions verify bounds and exact proof equivalence."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from femto import resource_policy as resources
from femto.io_utils import sha256_file
from femto.planning import coverage_certificate as coverage, footprint_model
from femto.planning.coverage_prepared import PreparedResidualEngine
from femto.planning.coverage_residual import ResidualEngine
from test_coverage_certificate import triangle_domain, seed


def simulated(total, available):
    return resources.MemorySnapshot(total*resources.GiB, available*resources.GiB, 'simulated-only')


class ResourcePolicyTests(unittest.TestCase):
    def test_available_memory_not_installed_total_controls_cache(self):
        for total in (16, 32, 64):
            with self.subTest(total=total):
                low = resources.preparation_resources(100000, memory=simulated(total, 1))
                self.assertEqual(low.candidate_cache_entries, 0)
                self.assertLessEqual(low.hash_buffer_bytes, 65536)
                self.assertGreaterEqual(low.reserve_bytes, 2*resources.GiB)
        high = resources.preparation_resources(100000, memory=simulated(32, 24))
        self.assertGreater(high.candidate_cache_entries, 4096)
        self.assertLessEqual(high.candidate_cache_entries*8192, high.cache_budget_bytes)
        self.assertLessEqual(high.cache_budget_bytes, 128*resources.MiB)

    def test_task_size_and_unknown_memory_have_bounded_fallbacks(self):
        small = resources.preparation_resources(1, file_bytes=15, memory=simulated(64, 48))
        self.assertLessEqual(small.candidate_cache_entries, 64)
        self.assertEqual(small.hash_buffer_bytes, 15)
        for memory in (resources.MemorySnapshot(None, None, 'unavailable'), simulated(16, 17), simulated(0, 0)):
            unknown = resources.preparation_resources(100000, memory=memory)
            self.assertEqual(unknown.candidate_cache_entries, 4096)
            self.assertLessEqual(unknown.hash_buffer_bytes, resources.MiB)
        with self.assertRaises(ValueError):resources.preparation_resources(-1)
        with self.assertRaises(ValueError):resources.preparation_resources(0, file_bytes=-1)

    def test_live_observation_is_explicit_and_sane_or_unknown(self):
        observed = resources.memory_snapshot()
        self.assertNotEqual(observed.source, 'simulated-only')
        if observed.source != 'unavailable':
            self.assertGreater(observed.total_bytes, 0)
            self.assertGreaterEqual(observed.available_bytes, 0)
            self.assertLessEqual(observed.available_bytes, observed.total_bytes)

    def test_low_and_high_memory_hash_every_byte_and_keep_cancellation(self):
        class Cancelled(RuntimeError):pass
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'中文.bin'
            data = bytes(range(256))*16385
            path.write_bytes(data)
            for memory in (simulated(16, 1), simulated(32, 24)):
                with patch.object(resources, 'memory_snapshot', return_value=memory):
                    self.assertEqual(sha256_file(path), hashlib.sha256(data).hexdigest())
                    calls = 0
                    def check():
                        nonlocal calls
                        calls += 1
                        if calls == 3:raise Cancelled('requested')
                    with self.assertRaises(Cancelled):sha256_file(path, check)
                    self.assertEqual(path.read_bytes(), data)

    def test_cache_disabled_small_large_caps_preserve_full_and_partial_proofs(self):
        domain, model = triangle_domain(), footprint_model.nominal(125.)
        budget = coverage.CoverageBudget(max_depth=3)
        guide = coverage.evaluate(domain, [seed()], model, budget=budget, algorithm=coverage.CURRENT_ALGORITHM)
        for algorithm in (coverage.RECALCULATION_ALGORITHM, coverage.INTERLEAVED_ALGORITHM):
            records = [seed()] if algorithm == coverage.RECALCULATION_ALGORITHM else [seed(), seed((.125, 0, 0))]
            for frontier in (None, dict(stop_reason='time_budget', steps=17)):
                with patch.object(PreparedResidualEngine, '_visit', ResidualEngine._visit):
                    expected = coverage.evaluate(domain, records, model, budget=budget, algorithm=algorithm,
                        guide=guide, _frontier=frontier)
                for cap in (0, 64, 16384):
                    policy = resources.preparation_resources(100000, memory=simulated(32, 24))
                    from dataclasses import replace
                    with patch('femto.planning.coverage_prepared.preparation_resources', return_value=replace(policy, candidate_cache_entries=cap)):
                        actual = coverage.evaluate(domain, records, model, budget=budget, algorithm=algorithm,
                            guide=guide, _frontier=frontier)
                    self.assertEqual(actual.metadata, expected.metadata)
                    for key in actual.arrays:
                        np.testing.assert_array_equal(actual.arrays[key], expected.arrays[key])


if __name__ == '__main__':unittest.main()
