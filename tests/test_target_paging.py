"""Bounded pages must advance the same target and survive independent replay."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import unittest

import numpy as np

import test_target_surface as fixtures
from femto.planning import target_surface as target, target_paging as paging


class PagingTests(unittest.TestCase):
    def setUp(self):
        self.fixture_case=fixtures.TargetSurfaceTests();self.fixture_case.setUp()
        self.addCleanup(self.fixture_case.doCleanups)
        mask=np.zeros((100,100),bool);mask[49:52,49:52]=True
        model,region=self.fixture_case.fixture(mask=mask)
        self.full,self.revision,self.evidence=self.fixture_case.build(model,region)
        self.first,_,_=self.fixture_case.build(model,region,budget=target.TargetBudget(max_units=8))

    def bundle(self):
        return paging.begin(self.first,self.revision,evidence=self.evidence,domain=None)

    def complete(self):
        result=self.bundle()
        for _ in range(20):
            if result.pages[-1].metadata['metrics']['stop_reason']=='complete':return result
            result=paging.resume(result,self.revision,evidence=self.evidence,domain=None,budget=target.TargetBudget(max_units=8))
        self.fail('bounded target did not finish')

    def test_pages_progress_without_deleting_old_units_or_changing_target(self):
        result=self.complete()
        self.assertGreater(len(result.pages),1)
        self.assertTrue(all(len(page.arrays['area_m2'])<=8 for page in result.pages))
        self.assertEqual(result.metadata['target_id'],self.full.metadata['target_id'])
        np.testing.assert_array_equal(result.pixel_status,self.full.arrays['pixel_status'])
        for name in target._LEAVES:
            np.testing.assert_array_equal(np.concatenate([p.arrays[name] for p in result.pages]),self.full.arrays[name])
        self.assertGreater(result.metadata['summary']['leaf_count'],8)
        self.assertEqual(result.metadata['summary']['unknown_by_reason']['budget_unprocessed'],0)

    def test_saved_pages_independently_replay_and_reject_corruption(self):
        result=self.complete();folder=Path(self.fixture_case.temp.name)/'pages'
        paging.write(folder,result)
        restored=paging.read(folder,self.revision,evidence=self.evidence,domain=None)
        self.assertEqual(restored.metadata,result.metadata)
        manifest=folder/'page_001'/target.ASSET
        with np.load(manifest) as archive:arrays={name:archive[name].copy() for name in archive.files}
        arrays['area_m2'][0]*=2
        np.savez_compressed(manifest,**arrays)
        with self.assertRaises(ValueError):
            paging.read(folder,self.revision,evidence=self.evidence,domain=None)

    def test_wrong_frontier_or_changed_revision_cannot_skip_unknown_pixels(self):
        result=self.complete();page=result.pages[1]
        header=deepcopy(page.metadata);header['metrics']['page_start_group']+=1
        forged=replace(page,metadata=header)
        with self.assertRaises(ValueError):paging._seal((result.pages[0],forged),self.revision)
        other=deepcopy(self.revision);other['sha256']='0'*64
        with self.assertRaises(ValueError):paging.resume(result,other,evidence=self.evidence,domain=None)

    def test_cancellation_and_zero_progress_preserve_previous_pages(self):
        result=self.bundle();before=deepcopy(result.metadata)
        def stop():raise RuntimeError('cancelled')
        with self.assertRaisesRegex(RuntimeError,'cancelled'):
            paging.resume(result,self.revision,evidence=self.evidence,domain=None,check=stop)
        with self.assertRaisesRegex(ValueError,'没有推进'):
            paging.resume(result,self.revision,evidence=self.evidence,domain=None,budget=target.TargetBudget(max_units=0))
        self.assertEqual(result.metadata,before)
