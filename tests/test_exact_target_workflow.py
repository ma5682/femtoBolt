"""Versioned exact target calculation, independent archives and derived routes."""
from copy import copy,deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from test_coverage_workflow import fixture
from femto.planning import (target_reconstruction as exact, target_surface as target,
    planning_revision as revisions, coverage_workflow as coverage, coverage_certificate,
    region_project, surface_workspace)


class ExactTargetWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.base=fixture(Path(cls.temp.name)/'base')

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def setUp(self):
        self.temp_case=tempfile.TemporaryDirectory();self.addCleanup(self.temp_case.cleanup)
        self.root=Path(self.temp_case.name)
        model,region,plan,certificate,ledger=self.base
        self.model=copy(model);self.model.store=copy(model.store)
        self.region=deepcopy(region);self.model.store.regions=[self.region]
        self.plan=coverage.attach(self.model,self.region,plan,certificate,ledger)
        self.before=deepcopy(self.plan)

    def rebuild(self,**kwargs):
        result,ledger=exact.calculate(self.model,self.region,self.plan,self.root/'exact',**kwargs)
        self.model.target_ledger=ledger;self.model.coverage_certificate=None;self.model.coverage_display=None
        return result,ledger

    def test_real_partition_rebuild_keeps_inputs_and_route_and_archives_previous_claims(self):
        old_count=self.plan['planning_context']['target']['summary']['unlocated_pixel_count']
        result,ledger=self.rebuild(budget=target.TargetBudget(max_units=500))
        self.assertLess(ledger.summary['unlocated_pixel_count'],old_count)
        self.assertEqual(ledger.summary['unknown_by_reason']['budget_unprocessed'],0)
        self.assertGreater(len(ledger.metadata['construction_pages']),1)
        old,new=self.before['planning_context']['revision'],result['planning_context']['revision']
        self.assertNotEqual(old['target_id'],new['target_id'])
        self.assertEqual(revisions.changed(old,new),['algorithms'])
        self.assertEqual(new['inputs']['algorithms']['target'],revisions.EXACT_TARGET)
        self.assertEqual(self.plan,self.before)
        self.assertEqual(result['segments'],self.before['segments'])
        self.assertEqual(result['surface_trajectory']['ordered_items'],self.before['surface_trajectory']['ordered_items'])
        self.assertNotIn(coverage.KEY,result)
        self.assertTrue((self.root/'exact/previous/coverage_certificate.npz').is_file())
        self.assertFalse(exact.available(result))
        revisions.require_current(self.model,self.region,result['treatment_head'],new)
        params=result['surface_trajectory']['planner']['parameters']
        read=target.read(self.root/'exact',ledger.metadata,new,evidence=params['quality_evidence'],domain=params['effective_domain'])
        self.assertEqual(read.metadata,ledger.metadata)

    def test_rebuilt_target_coverage_and_portable_independent_open(self):
        result,ledger=self.rebuild()
        certificate,_=coverage.calculate(self.model,self.region,result,budget=coverage_certificate.CoverageBudget(max_depth=1))
        result=coverage.attach(self.model,self.region,result,certificate,ledger)
        path=region_project.save_project(self.model,self.region,self.root/'exact.zip',result)
        opened=region_project.open_project(path,workspace=self.root/'opened')
        self.assertEqual(opened.target_ledger.metadata,ledger.metadata)
        self.assertEqual(opened.coverage_certificate.metadata,certificate.metadata)
        self.assertEqual(opened.saved_plan['segments'],self.before['segments'])

    def test_cancel_and_zero_progress_preserve_old_plan_and_coverage(self):
        def stop():raise RuntimeError('cancelled')
        with self.assertRaisesRegex(RuntimeError,'cancelled'):self.rebuild(check=stop)
        self.assertFalse((self.root/'exact').exists())
        with self.assertRaisesRegex(ValueError,'未能推进'):self.rebuild(budget=target.TargetBudget(max_units=0))
        self.assertEqual(self.plan,self.before)
        self.assertFalse((self.root/'exact').exists())

    def test_cumulative_page_cap_stops_work_and_preserves_old_result(self):
        from femto.planning import target_paging as paging
        revision=deepcopy(self.plan['planning_context']['revision']['inputs'])
        revision['algorithms']['target']=revisions.EXACT_TARGET;revision=revisions.create(revision)
        inputs={k:self.model.target_ledger.arrays[k] for k in paging.SOURCE_NAMES}
        budget=target.TargetBudget(max_units=50)
        arrays,_=target._construct(inputs,revision,budget,lambda:None)
        cap=sum(a.nbytes for a in arrays.values())*3//2
        messages=[]
        def progress(message):
            # Original assets are already verified. One new page fits; two do not.
            target.MAX_ASSET_BYTES=cap;messages.append(message)
        with patch.object(target,'MAX_ASSET_BYTES',target.MAX_ASSET_BYTES):
            with self.assertRaisesRegex(ValueError,'累计数组'):
                exact.calculate(self.model,self.region,self.plan,self.root/'bounded',progress=progress,budget=budget)
        self.assertEqual(len(messages),2)
        self.assertEqual(self.plan,self.before)
        self.assertFalse((self.root/'bounded').exists())

    def test_strict_variable_and_fixed_refinement_preserve_exact_target_version(self):
        from femto.planning import (strict_workflow,variable_width_workflow,variable_footprint,
                                   refinement_workflow,coverage_refinement)
        result,ledger=self.rebuild()
        strict=strict_workflow.calculate(self.model,self.region,result)
        self.assertEqual(strict['planning_context']['revision']['target_id'],ledger.metadata['target_id'])
        radius=strict['treatment_head']['effective_treatment_width_mm']/2000
        profile=variable_footprint.nominal([dict(position=t,lower_half_width_m=radius,upper_half_width_m=radius)
            for t in (0.,1.)],name='exact partition',version='1')
        variable=variable_width_workflow.calculate(self.model,self.region,strict,profile)
        surface_workspace.target_for(self.model,variable,reproduce=True)
        restored=variable_width_workflow.restore(self.model,self.region,variable)
        original=strict_workflow.restore(self.model,self.region,restored)
        self.assertEqual(original['planning_context']['target'],ledger.metadata)
        certificate,_=coverage.calculate(self.model,self.region,original,budget=coverage_certificate.CoverageBudget(max_depth=1))
        original=coverage.attach(self.model,self.region,original,certificate,ledger)
        refined=refinement_workflow.calculate(self.model,self.region,original,
            budget=coverage_refinement.RefinementBudget(max_rounds=0))
        self.assertEqual(refined['planning_context']['revision']['inputs']['algorithms']['target'],revisions.EXACT_TARGET)
        restored=refinement_workflow.restore(self.model,self.region,refined)
        self.assertEqual(restored['planning_context']['target'],ledger.metadata)

    def test_unknown_target_version_and_rehashed_forged_pixel_status_rejected(self):
        self.assertFalse(revisions.supported_target_algorithms(dict(revisions.ALGORITHMS,target='unknown')))
        result,ledger=self.rebuild();revision=result['planning_context']['revision']
        arrays={k:v.copy() for k,v in ledger.arrays.items()}
        arrays['pixel_status'][arrays['mask']]=8
        forged=target._seal(arrays,revision,target.TargetBudget(),ledger.metadata['metrics'],
            ledger.metadata['quality_evidence_sha256'],ledger.metadata['effective_domain_sha256'],
            construction_pages=ledger.metadata['construction_pages'])
        params=result['surface_trajectory']['planner']['parameters']
        with self.assertRaises(ValueError):target.validate(forged,revision,evidence=params['quality_evidence'],domain=params['effective_domain'])


if __name__=='__main__':unittest.main()
