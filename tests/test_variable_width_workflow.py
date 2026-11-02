"""Full P5 replay, real bound checks, version invalidation and failure isolation."""
from copy import copy, deepcopy
from dataclasses import replace
from datetime import date
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from test_strict_workflow import strict_fixture
from test_variable_footprint import profile
from femto.planning import variable_width_workflow as workflow, variable_width_route as route, variable_footprint
from femto.planning import coverage_workflow, coverage_certificate, planning_state, planning_revision
from femto.planning import surface_workspace, region_project
from femto.planning.surface_trajectory import SurfaceTrajectory, validate_trajectory
from femto.planning.trajectory_export import load_trajectory


class VariableWidthWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.base=strict_fixture(Path(cls.temp.name)/'source')

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def setUp(self):
        self.case=tempfile.TemporaryDirectory();self.addCleanup(self.case.cleanup);self.root=Path(self.case.name)
        model,region,_,strict,_=self.base
        self.model=copy(model);self.model.store=copy(model.store)
        self.region=deepcopy(region);self.model.store.regions=[self.region];self.original=deepcopy(strict)
        self.model.reconstruction=self.root/'work';self.model.reconstruction.mkdir()
        self.radius=strict['treatment_head']['effective_treatment_width_mm']/2000
        self.profile=profile(self.radius)

    def apply(self, model=None):return workflow.calculate(self.model,self.region,self.original,model or self.profile)

    def cover(self, plan):
        certificate,ledger=coverage_workflow.calculate(self.model,self.region,plan,budget=coverage_certificate.CoverageBudget(max_depth=1))
        return coverage_workflow.attach(self.model,self.region,plan,certificate,ledger)

    def test_new_model_preserves_points_anchors_and_target_but_invalidates_coverage(self):
        old=deepcopy(self.original);arrays={k:v.copy() for k,v in self.model.target_ledger.arrays.items()}
        result=self.apply()
        self.assertEqual(old,self.original)
        self.assertEqual(result['surface_trajectory']['ordered_items'],old['surface_trajectory']['ordered_items'])
        self.assertEqual(result['planning_context']['revision']['target_id'],old['planning_context']['revision']['target_id'])
        self.assertNotEqual(result['planning_context']['revision']['sha256'],old['planning_context']['revision']['sha256'])
        self.assertNotIn('coverage_evidence',result);self.assertIsNone(self.model.coverage_certificate)
        for key,value in arrays.items():np.testing.assert_array_equal(value,self.model.target_ledger.arrays[key])
        planning_state.validate(result,self.model.mesh,self.region)
        planning_revision.require_current(self.model,self.region,result['treatment_head'],result['planning_context']['revision'])

    def test_every_connector_and_scan_edge_uses_its_own_upper_bound(self):
        result=self.apply();report=result['surface_trajectory']['gap_report']['variable_width']
        self.assertEqual(report['checked_edges'],report['accepted_edges'])
        self.assertGreater(report['connector_edges'],0)
        self.assertTrue(all(row['accepted'] and row['support_check']['status']=='passed' for row in report['records']))
        self.assertTrue(all(row['outer_check_width_mm']/2000 >= row['upper_half_width_m'] for row in report['records']))

    def test_old_certificate_cannot_attach_after_model_change(self):
        result=self.apply();forged=deepcopy(result);forged['coverage_evidence']=self.original['coverage_evidence']
        forged['plan_binding']=planning_state.binding(self.model.mesh,self.region,forged)
        with self.assertRaisesRegex(ValueError,'覆盖证据与当前路线'):planning_state.validate(forged,self.model.mesh,self.region)

    def test_peak_above_generation_envelope_blocks_and_keeps_every_cache(self):
        knots=deepcopy(self.profile['knots']);knots[1]['upper_half_width_m']=min(.05,self.radius*2)
        model=variable_footprint.nominal(knots,name='interior peak',version='1')
        cache=self.model.target_ledger,self.model.coverage_certificate,self.model.coverage_display
        with self.assertRaisesRegex(ValueError,'段内上界'):self.apply(model)
        for a,b in zip(cache,(self.model.target_ledger,self.model.coverage_certificate,self.model.coverage_display)):self.assertIs(a,b)
        reports=list((self.model.reconstruction/'variable_width_attempts').glob('*.json'))
        self.assertEqual(len(reports),1)
        self.assertEqual(json.loads(reports[0].read_text(encoding='utf-8'))['status'],'outer_bound_exceeds_generation_envelope')

    def test_discrete_samples_do_not_create_verified_route(self):
        with self.assertRaisesRegex(ValueError,'离散宽度'):self.apply(profile(self.radius,interpolation='samples_only'))

    def test_zero_inner_radius_cannot_confirm_any_area(self):
        result=self.cover(self.apply(profile(self.radius,zero=True)))
        certificate=self.model.coverage_certificate
        self.assertEqual(result['coverage_evidence']['summary']['confirmed_nominal_area_m2'],0)
        self.assertFalse(result['coverage_evidence']['summary']['original_target_coverage_complete'])
        self.assertEqual(len(certificate.arrays['seed_inner_radius']),0)

    def test_native_and_portable_replay_include_model_and_restore_two_levels(self):
        result=self.cover(self.apply())
        verified=[]
        native=surface_workspace.export(self.model,self.region,result,verified_sink=verified.append)
        self.assertEqual(len(verified),1)
        read=load_trajectory(native)
        self.assertEqual(read.planner['parameters']['variable_footprint'],self.profile)
        self.assertTrue((native/'variable_footprint.json').is_file());self.assertTrue((native/'variable_width_report.json').is_file())
        path=region_project.save_project(self.model,self.region,self.root/'model.region.zip',result,verified_assets=verified[0])
        opened=region_project.open_project(path,workspace=self.root/'opened')
        self.assertEqual(opened.saved_plan['surface_trajectory'],result['surface_trajectory'])
        self.assertEqual(opened.coverage_certificate.metadata,self.model.coverage_certificate.metadata)
        restored=workflow.restore(opened,opened.store.regions[0],opened.saved_plan)
        self.assertEqual(restored['surface_trajectory'],self.original['surface_trajectory']);self.assertNotIn('coverage_evidence',restored)
        from femto.planning.strict_workflow import restore
        ordinary=restore(opened,opened.store.regions[0],restored)
        self.assertEqual(ordinary['surface_trajectory']['planner']['algorithm_version'],'quality-constrained-surface-v3')

    def test_equal_constant_model_reproduces_fixed_ball_proof_arrays(self):
        model=variable_footprint.nominal([dict(position=t,lower_half_width_m=self.radius,upper_half_width_m=self.radius)
                                        for t in (0.,1.)],name='same fixed geometry',version='1')
        old=self.model.coverage_certificate
        self.cover(self.apply(model))
        for key,values in old.arrays.items():np.testing.assert_array_equal(values,self.model.coverage_certificate.arrays[key])

    def test_foreign_publication_receipt_cannot_skip_verification(self):
        result=self.cover(self.apply())
        from femto.planning.refinement_archive import VerifiedAssets,require_verified_assets
        with self.assertRaisesRegex(ValueError,'凭据'):
            require_verified_assets(VerifiedAssets('',object()),SurfaceTrajectory.from_dict(result['surface_trajectory']),
                                    self.model.target_ledger,self.model.coverage_certificate)

    def test_resealed_false_diagnostic_is_rejected_by_geometric_replay(self):
        result=SurfaceTrajectory.from_dict(self.apply()['surface_trajectory'])
        report=deepcopy(result.gap_report['variable_width']);report['records'][0]['residual_margin_mm']=999.
        fake=replace(result,gap_report=dict(result.gap_report,variable_width=report))
        mesh=surface_workspace.canonical_region_mesh(self.model.mesh,self.region)
        with self.assertRaisesRegex(ValueError,'无法重现'):validate_trajectory(fake,mesh=mesh,region=self.region)

    def test_cancel_before_install_preserves_caches(self):
        cache=self.model.target_ledger,self.model.coverage_certificate,self.model.coverage_display
        original_validate=planning_state.validate;calls=0
        def cancel(*args,**kwargs):
            nonlocal calls
            calls+=1
            if calls==2:raise RuntimeError('cancel before install')
            return original_validate(*args,**kwargs)
        with patch.object(planning_state,'validate',side_effect=cancel):
            with self.assertRaisesRegex(RuntimeError,'cancel before install'):self.apply()
        for a,b in zip(cache,(self.model.target_ledger,self.model.coverage_certificate,self.model.coverage_display)):self.assertIs(a,b)

    def test_expired_history_opens_readonly_but_cannot_save_or_recompute(self):
        expired=profile(self.radius,valid_until='2000-01-31')
        require=variable_footprint.require_current
        with patch.object(variable_footprint,'require_current',side_effect=lambda model:require(model,on=date(2000,1,1))):
            result=self.apply(expired)
            path=region_project.save_project(self.model,self.region,self.root/'expired.region.zip',result)
        opened=region_project.open_project(path,workspace=self.root/'opened')
        state=planning_state.state(opened.store.regions[0],opened.saved_plan)
        self.assertFalse(state['can_use']);self.assertIn('已过期',state['reason'])
        with self.assertRaisesRegex(ValueError,'已过期'):coverage_workflow.calculate(opened,opened.store.regions[0],opened.saved_plan)
        with self.assertRaisesRegex(ValueError,'已过期'):
            region_project.save_project(opened,opened.store.regions[0],self.root/'forbidden.region.zip',opened.saved_plan)
        restored=workflow.restore(opened,opened.store.regions[0],opened.saved_plan)
        self.assertEqual(restored['surface_trajectory']['planner']['algorithm_version'],'constructed-surface-route-v1')

    def test_model_expiring_during_save_cannot_be_published(self):
        expiring=profile(self.radius,valid_until='2000-01-31')
        require=variable_footprint.require_current
        def expire_before_commit(action):
            with patch.object(variable_footprint,'require_current',side_effect=require):return action()
        with patch.object(variable_footprint,'require_current',side_effect=lambda model:require(model,on=date(2000,1,1))):
            result=self.apply(expiring)
            with self.assertRaisesRegex(ValueError,'已过期'):
                surface_workspace.export(self.model,self.region,result,commit=expire_before_commit)
            with self.assertRaisesRegex(ValueError,'已过期'):
                region_project.save_project(self.model,self.region,self.root/'late.region.zip',result,commit=expire_before_commit)
        self.assertFalse((self.root/'late.region.zip').exists())
        self.assertFalse(list((self.model.reconstruction/'surface_paths').glob('trajectory_*')))
