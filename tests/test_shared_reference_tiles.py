"""Shared preparation keeps original maxima, budgets and mutable-input isolation."""
from copy import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from femto.planning.reference_tiles import bind,ResidualTileSnapshot
from femto.performance_detail import collect
from test_reference_residual_tiles import ResidualTileTests


class SharedResidualTests(unittest.TestCase):
    def envelope(self):return ResidualTileTests().envelope()

    def test_batches_keep_exact_values_and_logical_charges_with_one_preparation(self):
        reference=self.envelope();bind(reference)
        rng=np.random.default_rng(830)
        with collect() as detail:
            for batch_index in range(40):
                actual=copy(reference);old=self.envelope()
                for rect in [(0,0,99,99),*map(tuple,rng.integers(-130,230,(20,4)))]:
                    self.assertEqual(actual._residual_max(rect),old._residual_max(rect))
                    self.assertEqual(actual.sampled_pixels,old.sampled_pixels)
        counts=detail.snapshot()['rows']['route.residual_work']['counts']
        # Forty fresh reference reductions plus exactly one shared preparation.
        self.assertEqual(counts['tile_builds'],41)
        self.assertEqual(counts['shared_tile_hits'],39)
        self.assertEqual(reference.sampled_pixels,0)
        self.assertIsNone(reference._residual_tiles)

    def test_snapshot_cannot_be_changed_by_source_alias_or_writeable_flag(self):
        source=np.arange(10000,dtype=np.int32).reshape(100,100)
        alias=source.view();snapshot=ResidualTileSnapshot(source)
        expected=snapshot.values.copy();tiles=snapshot.get(snapshot.values)
        alias[:]=0
        np.testing.assert_array_equal(snapshot.values,expected)
        for values in (snapshot.values,tiles):
            with self.assertRaises(ValueError):values.setflags(write=True)
            with self.assertRaises(ValueError):values.view().setflags(write=True)
        self.assertIsNone(snapshot.get(source))

    def test_replaced_residual_does_not_reuse_snapshot_and_failed_batch_can_retry(self):
        reference=self.envelope();bind(reference)
        batch=copy(reference);batch._residual_max((0,0,99,99))
        retry=copy(reference);retry.sampled_pixels=100000000
        with self.assertRaisesRegex(ValueError,'超过预算'):retry._residual_max((0,0,99,99))
        self.assertEqual(copy(reference)._residual_max((0,0,99,99)),batch._residual_max((0,0,99,99)))
        replaced=copy(reference);replaced.residual=np.full((100,100),4567,np.int32)
        self.assertEqual(replaced._residual_max((0,0,99,99)),.004567)
        self.assertNotEqual(copy(reference)._residual_max((0,0,99,99)),.004567)

    def test_snapshot_memory_limit_and_allocation_failure_keep_original_path(self):
        large=SimpleNamespace(residual=np.empty((2049,2048),np.int32),_shared_residual_tiles=None)
        with patch('femto.planning.reference_tiles.ResidualTileSnapshot') as constructor:
            bind(large);constructor.assert_not_called()
        reference=self.envelope();original=reference.residual
        with patch('femto.planning.reference_tiles.ResidualTileSnapshot',side_effect=MemoryError):bind(reference)
        self.assertIs(reference.residual,original)
        self.assertIsNone(reference._shared_residual_tiles)
        self.assertEqual(reference._residual_max((0,0,99,99)),self.envelope()._residual_max((0,0,99,99)))


if __name__=='__main__':unittest.main()
