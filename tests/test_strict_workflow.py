"""Opt-in, route-bound coverage, exact refinement and complete archive replay."""
from copy import copy,deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from test_coverage_workflow import fixture
from femto.planning import strict_workflow,strict_path,coverage_workflow,coverage_certificate
from femto.planning import refinement_workflow,coverage_refinement,refinement_archive,strict_refinement
from femto.planning import surface_workspace,region_project,planning_state,planning_revision
from femto.planning.surface_trajectory import SurfaceTrajectory,ScanSegment,validate_trajectory
from femto.planning.trajectory_export import load_trajectory


def strict_fixture(folder):
    model,region,plan,certificate,ledger=fixture(folder)
    model.store.regions=[region]
    before=coverage_workflow.attach(model,region,plan,certificate,ledger)
    original_cache=(model.target_ledger,model.coverage_certificate,model.coverage_display)
    strict=strict_workflow.calculate(model,region,before)
    proof,target=coverage_workflow.calculate(model,region,strict,budget=coverage_certificate.CoverageBudget(max_depth=1))
    strict=coverage_workflow.attach(model,region,strict,proof,target)
    return model,region,before,strict,original_cache


class StrictWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.base=strict_fixture(Path(cls.temp.name)/'source')

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def setUp(self):
        self.case=tempfile.TemporaryDirectory();self.addCleanup(self.case.cleanup);self.root=Path(self.case.name)
        model,region,before,strict,self.original_cache=self.base
        self.model=copy(model);self.model.store=copy(model.store)
        self.region=deepcopy(region);self.model.store.regions=[self.region]
        self.before,self.strict=deepcopy(before),deepcopy(strict)
        self.model.reconstruction=self.root/'work';self.model.reconstruction.mkdir()

    def original(self):
        self.model.target_ledger,self.model.coverage_certificate,self.model.coverage_display=self.original_cache

    def test_old_coverage_is_cleared_and_restore_requires_new_coverage(self):
        self.original()
        result=strict_workflow.calculate(self.model,self.region,self.before)
        self.assertNotIn(coverage_workflow.KEY,result)
        self.assertIsNone(self.model.coverage_certificate);self.assertIsNone(self.model.coverage_display)
        restored=strict_workflow.restore(self.model,self.region,result)
        self.assertEqual(restored['surface_trajectory'],self.before['surface_trajectory'])
        self.assertNotIn(coverage_workflow.KEY,restored)
        planning_state.validate(restored,self.model.store.mesh,self.region)

    def test_old_coverage_cannot_attach_to_reconstructed_route(self):
        forged=deepcopy(self.strict);forged[coverage_workflow.KEY]=self.before[coverage_workflow.KEY]
        forged['plan_binding']=planning_state.binding(self.model.store.mesh,self.region,forged)
        with self.assertRaisesRegex(ValueError,'覆盖证据与当前路线'):planning_state.validate(forged,self.model.store.mesh,self.region)

    def test_failure_keeps_active_caches_and_writes_separate_attempt(self):
        self.original();before=deepcopy(self.before)
        cache=(self.model.target_ledger,self.model.coverage_certificate,self.model.coverage_display)
        with self.assertRaisesRegex(ValueError,'检查记录'):
            strict_workflow.calculate(self.model,self.region,self.before,budget=strict_path.SupportBudget(max_face_checks=0))
        self.assertEqual(self.before,before)
        for old,new in zip(cache,(self.model.target_ledger,self.model.coverage_certificate,self.model.coverage_display)):
            self.assertIs(old,new)
        files=list((self.model.reconstruction/'strict_attempts').glob('*.json'));self.assertEqual(len(files),1)
        report=json.loads(files[0].read_text(encoding='utf-8'))
        self.assertEqual(report['stop_reason'],'support_face_budget')

    def test_cancel_after_candidate_validation_keeps_active_caches(self):
        self.original();cache=self.model.target_ledger,self.model.coverage_certificate,self.model.coverage_display
        calls=0
        validate=planning_state.validate
        def cancelled(*args,**kwargs):
            nonlocal calls
            calls+=1
            if calls==2:raise RuntimeError('cancel before install')
            return validate(*args,**kwargs)
        with patch.object(planning_state,'validate',side_effect=cancelled):
            with self.assertRaisesRegex(RuntimeError,'cancel before install'):
                strict_workflow.calculate(self.model,self.region,self.before)
        for old,new in zip(cache,(self.model.target_ledger,self.model.coverage_certificate,self.model.coverage_display)):
            self.assertIs(old,new)

    def test_same_target_arrays_and_new_route_binding(self):
        for key,value in self.original_cache[0].arrays.items():np.testing.assert_array_equal(value,self.model.target_ledger.arrays[key])
        old=SurfaceTrajectory.from_dict(self.before['surface_trajectory']);new=SurfaceTrajectory.from_dict(self.strict['surface_trajectory'])
        self.assertNotEqual(coverage_workflow.route_binding(old),coverage_workflow.route_binding(new))
        self.assertEqual(old.planning_context['revision']['target_id'],new.planning_context['revision']['target_id'])
        planning_revision.require_current(self.model,self.region,self.strict['treatment_head'],new.planning_context['revision'])

    def test_strict_native_and_portable_roundtrip_with_coverage(self):
        native=surface_workspace.export(self.model,self.region,self.strict)
        read=load_trajectory(native)
        self.assertEqual(read.to_dict()['ordered_items'],self.strict['surface_trajectory']['ordered_items'])
        self.assertTrue((native/'surface_support_report.json').is_file())
        header=(native/'trajectory.csv').read_text(encoding='utf-8').splitlines()[0]
        self.assertIn('surface_anchor_json',header)
        path=region_project.save_project(self.model,self.region,self.root/'strict.zip',self.strict)
        opened=region_project.open_project(path,workspace=self.root/'opened')
        self.assertEqual(opened.saved_plan['surface_trajectory'],self.strict['surface_trajectory'])
        self.assertEqual(opened.coverage_certificate.metadata,self.model.coverage_certificate.metadata)
        restored=strict_workflow.restore(opened,opened.store.regions[0],opened.saved_plan)
        self.assertEqual(restored['surface_trajectory'],self.before['surface_trajectory'])

    def test_strict_version_can_be_saved_before_independent_coverage(self):
        plan=deepcopy(self.strict);plan.pop(coverage_workflow.KEY)
        plan['plan_binding']=planning_state.binding(self.model.store.mesh,self.region,plan)
        path=region_project.save_project(self.model,self.region,self.root/'unverified.zip',plan)
        opened=region_project.open_project(path,workspace=self.root/'opened')
        self.assertNotIn(coverage_workflow.KEY,opened.saved_plan)
        self.assertIsNone(opened.coverage_certificate)
        self.assertEqual(opened.saved_plan['surface_trajectory'],plan['surface_trajectory'])

    def test_missing_native_construction_report_is_not_silently_accepted(self):
        native=surface_workspace.export(self.model,self.region,self.strict)
        (native/'surface_support_report.json').unlink()
        path=native/'trajectory_files.json';manifest=json.loads(path.read_text(encoding='utf-8'))
        manifest['files'].pop('surface_support_report.json');path.write_text(json.dumps(manifest),encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'严格构造报告'):load_trajectory(native)

    def test_strict_refinement_replays_all_trials_and_preserves_original_anchors(self):
        result=refinement_workflow.calculate(self.model,self.region,self.strict,
            budget=coverage_refinement.RefinementBudget(max_rounds=1))
        trajectory=SurfaceTrajectory.from_dict(result['surface_trajectory'])
        self.assertEqual(trajectory.planner['algorithm_version'],strict_path.REFINED_ALGORITHM)
        report=trajectory.gap_report['refinement']
        self.assertTrue(report['rounds'][0]['accepted'],report['rounds'][0]['status'])
        self.assertEqual(report['rounds'][0]['comparison']['lost_weight_units'],0)
        base=SurfaceTrajectory.from_dict(self.strict['surface_trajectory'])
        for old in (s for s in base.ordered_items if isinstance(s,ScanSegment)):
            anchors=[p[strict_path.ANCHOR] for p in old.points]
            self.assertTrue(any(s.block_index==old.block_index and [p[strict_path.ANCHOR] for p in s.points[:len(anchors)]]==anchors
                                for s in trajectory.ordered_items if isinstance(s,ScanSegment)))
        certificate=self.model.coverage_certificate
        refinement_archive.replay(certificate,trajectory,self.model.target_ledger)
        path=region_project.save_project(self.model,self.region,self.root/'refined.zip',result)
        opened=region_project.open_project(path,workspace=self.root/'opened')
        self.assertEqual(opened.saved_plan['surface_trajectory'],result['surface_trajectory'])
        strict=refinement_workflow.restore(opened,opened.store.regions[0],opened.saved_plan)
        self.assertEqual(strict['surface_trajectory'],self.strict['surface_trajectory'])
        self.assertEqual(strict[coverage_workflow.KEY],self.strict[coverage_workflow.KEY])
        ordinary=strict_workflow.restore(opened,opened.store.regions[0],strict)
        self.assertEqual(ordinary['surface_trajectory'],self.before['surface_trajectory'])

    def test_zero_round_strict_refinement_preserves_route_and_bound_coverage(self):
        result=refinement_workflow.calculate(self.model,self.region,self.strict,budget=coverage_refinement.RefinementBudget(max_rounds=0))
        self.assertEqual(result['surface_trajectory']['ordered_items'],self.strict['surface_trajectory']['ordered_items'])
        self.assertEqual(result[coverage_workflow.KEY]['summary'],self.strict[coverage_workflow.KEY]['summary'])
        self.assertNotEqual(result[coverage_workflow.KEY]['sha256'],self.strict[coverage_workflow.KEY]['sha256'])
        with self.assertRaises(ValueError):strict_workflow.restore(self.model,self.region,result)
        restored=refinement_workflow.restore(self.model,self.region,result)
        self.assertEqual(restored['surface_trajectory'],self.strict['surface_trajectory'])

    def test_interior_connection_keeps_exact_anchor_at_both_ends(self):
        from femto.planning.effective_domain import validate
        from femto.planning.mesh_slice_planner import _project
        base=SurfaceTrajectory.from_dict(self.strict['surface_trajectory']);params=base.planner['parameters']
        mesh=surface_workspace.canonical_region_mesh(self.model.store.mesh,self.region)
        allowed=validate(params['effective_domain'],params['quality_evidence'],params['effective_width_mm'],mesh,self.region)
        meter=coverage_refinement._Meter(coverage_refinement.RefinementBudget())
        meter.support_budget=strict_path.SupportBudget()
        meter.support=strict_path.engine_for(mesh,allowed,check=lambda:None,region=self.region)
        eligible=np.flatnonzero(allowed & (meter.support.groups==base.blocks['items'][0]['component_id']))
        centres=mesh.vertices[mesh.faces[eligible]].mean(axis=1)
        face=int(eligible[np.argmin(np.linalg.norm(centres[:,:2],axis=1))])
        triangle=mesh.vertices[mesh.faces[face]];centre=triangle.mean(axis=0)
        edges=np.roll(triangle,-1,axis=0)-triangle;direction=edges[np.argmax(np.linalg.norm(edges,axis=1))]
        direction/=np.linalg.norm(direction)
        def points(offsets,row):
            return [strict_path.construct_point(meter.support,dict(xyz_m=(centre+offset*direction).tolist(),
                triangle_index=face,geometry_support='measured_mesh',role='scan',row_index=row),params,meter.support_budget) for offset in offsets]
        left=points((-.0015,-.0005),0);right=points((.0005,.0015),1);chains=[(0,deepcopy(left))]
        link=strict_refinement._connect(chains,(0,right),base,mesh,self.region,allowed,meter,lambda:None)
        self.assertTrue(link['connected'],link)
        self.assertEqual(chains[0][1][:len(left)],left)
        self.assertEqual(link['samples'][-1][strict_path.ANCHOR],right[0][strict_path.ANCHOR])
        self.assertTrue(all(p['status']=='passed' for p in link['support_checks']))

    def test_changed_upstream_input_invalidates_both_versions(self):
        changed=deepcopy(self.region);changed['triangle_indices']=changed['triangle_indices'][:-1]
        with self.assertRaises(ValueError):strict_workflow.restore(self.model,changed,self.strict)
        self.original()
        with self.assertRaises(ValueError):strict_workflow.calculate(self.model,changed,self.before)

    def test_strict_presentation_does_not_reenter_expensive_history_validation(self):
        trajectory=deepcopy(self.strict['surface_trajectory'])
        trajectory['planner']['algorithm_version']=strict_path.REFINED_ALGORITHM
        trajectory['gap_report']['refinement']=dict(rounds=[],stop_reason='round_limit',
            base_trajectory=dict(statistics=trajectory['statistics']))
        with patch.object(SurfaceTrajectory,'from_dict',side_effect=AssertionError('presentation replayed history')):
            self.assertIn('严格表面路径',strict_path.summary_text(trajectory))
            self.assertIn('接纳 0 轮',coverage_refinement.summary_text(trajectory))
