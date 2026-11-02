"""Continued targets use the whole surface and roundtrip through product archives."""
from copy import copy, deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from test_coverage_workflow import fixture
from femto.planning import target_surface as target, target_paging, target_continuation
from femto.planning import coverage_workflow as workflow, coverage_certificate as coverage
from femto.planning import surface_workspace, planning_state, region_project
from femto.planning.surface_trajectory import SurfaceTrajectory


class PagedTargetWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.base=fixture(Path(cls.temp.name)/'base')

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def setUp(self):
        self.case=tempfile.TemporaryDirectory();self.addCleanup(self.case.cleanup);self.root=Path(self.case.name)
        model,region,plan,self.original_certificate,self.full=self.base
        self.model=copy(model);self.model.store=copy(model.store);self.region=deepcopy(region)
        self.model.store.regions=[self.region];self.plan=deepcopy(plan)
        revision=self.plan['planning_context']['revision']
        inputs={key:self.full.arrays[key] for key in target_paging.SOURCE_NAMES}
        budget=target.TargetBudget(max_units=50)
        arrays,counts=target._construct(inputs,revision,budget,lambda:None)
        self.first=target._seal(arrays,revision,budget,counts,
            self.full.metadata['quality_evidence_sha256'],self.full.metadata['effective_domain_sha256'])
        self.plan['planning_context']['target']=self.first.metadata
        trajectory=replace(SurfaceTrajectory.from_dict(plan['surface_trajectory']),planning_context=self.plan['planning_context'])
        self.plan['surface_trajectory']=trajectory.to_dict()
        self.plan['plan_binding']=planning_state.binding(self.model.store.mesh,self.region,self.plan)
        self.model.target_ledger=self.first;self.model.coverage_certificate=None;self.model.coverage_display=None

    def continued(self):
        plan,ledger=target_continuation.calculate(self.model,self.region,self.plan,self.root/'continued')
        self.model.target_ledger=ledger
        return plan,ledger

    def test_continuation_preserves_route_and_old_evidence_and_replays_complete_target(self):
        before=deepcopy(self.plan);plan,ledger=self.continued()
        self.assertEqual(ledger.metadata['schema'],target.PAGED_SCHEMA)
        self.assertEqual(self.plan,before)
        self.assertEqual(plan['surface_trajectory']['ordered_items'],before['surface_trajectory']['ordered_items'])
        self.assertTrue((self.root/'continued/previous/plan.json').is_file())
        self.assertNotIn(workflow.KEY,plan)
        for key in target._DTYPES:np.testing.assert_array_equal(ledger.arrays[key],self.full.arrays[key])
        params=plan['surface_trajectory']['planner']['parameters'];revision=plan['planning_context']['revision']
        restored=target.read(self.root/'continued',ledger.metadata,revision,
            evidence=params['quality_evidence'],domain=params['effective_domain'])
        self.assertEqual(restored.metadata,ledger.metadata)
        surface_workspace.target_for(self.model,plan,reproduce=True)

    def test_whole_surface_coverage_matches_unpaged_result_across_page_boundary(self):
        plan,ledger=self.continued()
        certificate,_=workflow.calculate(self.model,self.region,plan,budget=coverage.CoverageBudget(max_depth=1))
        self.assertEqual(certificate.metadata['algorithm'],coverage.PAGED_ALGORITHM)
        for key in coverage.DTYPES:np.testing.assert_array_equal(certificate.arrays[key],self.original_certificate.arrays[key])
        plan=workflow.attach(self.model,self.region,plan,certificate,ledger)
        path=region_project.save_project(self.model,self.region,self.root/'paged.zip',plan)
        opened=region_project.open_project(path,workspace=self.root/'opened')
        self.assertEqual(opened.target_ledger.metadata,ledger.metadata)
        self.assertEqual(opened.coverage_certificate.metadata,certificate.metadata)
        self.assertEqual(opened.coverage_display.tasks.to_dict(),self.model.coverage_display.tasks.to_dict())

    def test_paged_same_input_recalculation_retains_confirmed_set_and_hint_roundtrips(self):
        plan,ledger=self.continued();budget=coverage.CoverageBudget(max_depth=1)
        first,_=workflow.calculate(self.model,self.region,plan,budget=budget)
        plan=workflow.attach(self.model,self.region,plan,first,ledger)
        second,_=workflow.calculate(self.model,self.region,plan,budget=budget)
        self.assertEqual(second.metadata['algorithm'],coverage.GROUPED_RECALCULATION_ALGORITHM)
        workflow.require_recalculation_retention(first,second,ledger)
        plan=workflow.attach(self.model,self.region,plan,second,ledger)
        path=region_project.save_project(self.model,self.region,self.root/'recalculated.zip',plan)
        opened=region_project.open_project(path,workspace=self.root/'recalculated_opened')
        self.assertEqual(opened.coverage_certificate.metadata,second.metadata)
        self.assertIsNotNone(opened.coverage_certificate.guide)

    def test_cancelled_continuation_does_not_publish_or_replace_current_plan(self):
        before=deepcopy(self.plan)
        def stop():raise RuntimeError('cancelled')
        with self.assertRaisesRegex(RuntimeError,'cancelled'):
            target_continuation.calculate(self.model,self.region,self.plan,self.root/'cancelled',check=stop)
        self.assertEqual(self.plan,before);self.assertFalse((self.root/'cancelled').exists())

    def test_rehashed_page_frontier_and_algorithm_relabelling_cannot_bypass_replay(self):
        from femto.planning.planning_revision import digest
        plan,ledger=self.continued();header=deepcopy(ledger.metadata)
        header['construction_pages'][1]['metrics']['page_start_group']+=1
        child=header['construction_pages'][1];child['sha256']=digest({k:v for k,v in child.items() if k!='sha256'})
        header['sha256']=digest({k:v for k,v in header.items() if k!='sha256'})
        with self.assertRaises(ValueError):target.validate_metadata(header,plan['planning_context']['revision'])
        metadata=deepcopy(self.original_certificate.metadata)
        metadata['algorithm']=coverage.PAGED_ALGORITHM;metadata['inputs']['algorithm']=coverage.PAGED_ALGORITHM
        metadata['coverage_computation_sha256']=digest(metadata['inputs'])
        metadata['sha256']=digest({k:v for k,v in metadata.items() if k!='sha256'})
        with self.assertRaises(ValueError):coverage.validate_metadata(metadata)

    def test_strict_and_variable_models_keep_page_provenance_and_restore_original_target(self):
        from femto.planning import strict_workflow,variable_width_workflow,variable_footprint
        plan,ledger=self.continued()
        strict=strict_workflow.calculate(self.model,self.region,plan)
        self.assertEqual(strict['planning_context']['target']['schema'],target.PAGED_SCHEMA)
        self.assertFalse(target_continuation.available(strict))
        radius=strict['treatment_head']['effective_treatment_width_mm']/2000
        profile=variable_footprint.nominal([dict(position=t,lower_half_width_m=radius,upper_half_width_m=radius)
                                           for t in (0.,1.)],name='paged model',version='1')
        variable=variable_width_workflow.calculate(self.model,self.region,strict,profile)
        surface_workspace.target_for(self.model,variable,reproduce=True)
        restored=variable_width_workflow.restore(self.model,self.region,variable)
        original=strict_workflow.restore(self.model,self.region,restored)
        self.assertEqual(original['planning_context']['target'],ledger.metadata)

    def test_fixed_refinement_and_restore_preserve_paged_target_and_coverage_assets(self):
        from femto.planning import refinement_workflow,coverage_refinement
        plan,ledger=self.continued()
        certificate,_=workflow.calculate(self.model,self.region,plan,budget=coverage.CoverageBudget(max_depth=1))
        plan=workflow.attach(self.model,self.region,plan,certificate,ledger)
        refined=refinement_workflow.calculate(self.model,self.region,plan,
            budget=coverage_refinement.RefinementBudget(max_rounds=0))
        surface_workspace.target_for(self.model,refined,reproduce=True)
        path=region_project.save_project(self.model,self.region,self.root/'refined.zip',refined)
        opened=region_project.open_project(path,workspace=self.root/'refined_opened')
        restored=refinement_workflow.restore(opened,opened.store.regions[0],opened.saved_plan)
        self.assertEqual(restored['planning_context']['target'],ledger.metadata)
        self.assertEqual(restored['coverage_evidence'],certificate.metadata)
