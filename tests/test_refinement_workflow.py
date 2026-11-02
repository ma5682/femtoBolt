"""Full P3 archive replay, tamper refusal, revision lifecycle and rollback."""
from copy import copy, deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from test_coverage_workflow import fixture
from femto.planning import coverage_workflow, refinement_workflow, refinement_archive, coverage_refinement as refinement
from femto.planning import planning_state, planning_revision, region_project, surface_workspace
from femto.planning.surface_trajectory import SurfaceTrajectory
from femto.planning.trajectory_export import load_trajectory, export_trajectory


def refined_fixture(folder):
    model,region,original,certificate,ledger=fixture(folder)
    model.store.regions=[region]
    before=coverage_workflow.attach(model,region,original,certificate,ledger)
    result=refinement_workflow.calculate(model,region,before,budget=refinement.RefinementBudget(max_rounds=1))
    return model,region,before,result


class RefinementWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        cls.base=refined_fixture(Path(cls.temp.name)/'source')

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def setUp(self):
        self.case=tempfile.TemporaryDirectory();self.addCleanup(self.case.cleanup);self.root=Path(self.case.name)
        model,region,before,result=self.base
        self.model=copy(model);self.model.store=copy(model.store)
        self.region=deepcopy(region);self.model.store.regions=[self.region]
        self.before,self.result=deepcopy(before),deepcopy(result)

    def save(self):return region_project.save_project(self.model,self.region,self.root/'plan.zip',self.result)

    def open(self,path):return region_project.open_project(path,workspace=self.root/'opened')

    def rewrite(self,path,mutate):
        with zipfile.ZipFile(path) as z:files={n:z.read(n) for n in z.namelist()}
        data=json.loads(files['project.json']);mutate(data,files)
        data['files']={name:hashlib.sha256(value).hexdigest() for name,value in files.items() if name!='project.json'}
        files['project.json']=json.dumps(data,ensure_ascii=False).encode('utf-8')
        with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
            for name,value in files.items():z.writestr(name,value)

    def test_portable_replay_restores_full_history_and_can_restore_original(self):
        model=self.open(self.save())
        self.assertEqual(model.saved_plan['surface_trajectory'],self.result['surface_trajectory'])
        self.assertIsInstance(model.coverage_certificate,refinement.RefinedCertificate)
        self.assertEqual(len(model.coverage_certificate.history),2)
        result=refinement_workflow.restore(model,model.store.regions[0],model.saved_plan)
        self.assertEqual(result['surface_trajectory'],self.before['surface_trajectory'])
        self.assertEqual(result['coverage_evidence'],self.before['coverage_evidence'])
        planning_state.validate(result,model.store.mesh,model.store.regions[0])

    def test_native_and_relocated_portable_reexport_reproduce_v3_base(self):
        model=self.open(self.save())
        receipts=[]
        native=surface_workspace.export(model,model.store.regions[0],model.saved_plan,verified_sink=receipts.append)
        certificates=[]
        trajectory=load_trajectory(native,coverage_sink=certificates.append)
        self.assertEqual(trajectory.to_dict()['ordered_items'],self.result['surface_trajectory']['ordered_items'])
        self.assertEqual(certificates[0].metadata,self.model.coverage_certificate.metadata)
        self.assertIsInstance(certificates[0],refinement.RefinedCertificate)
        self.assertTrue((native/'refinement_0_coverage_certificate.npz').is_file())
        self.assertEqual(len(receipts),1)
        with patch('femto.planning.refinement_archive.replay',side_effect=AssertionError('same proof replayed twice')):
            path=region_project.save_project(model,model.store.regions[0],self.root/'checked_copy.zip',model.saved_plan,verified_assets=receipts[0])
        with patch('femto.planning.refinement_archive.replay',wraps=refinement_archive.replay) as replay:
            again=self.open(path)
            self.assertTrue(replay.called)
        self.assertEqual(again.coverage_certificate.metadata,model.coverage_certificate.metadata)

    def test_same_target_identity_but_upstream_changes_invalidate_new_revision(self):
        revision=self.result['planning_context']['revision']
        planning_revision.require_current(self.model,self.region,self.result['treatment_head'],revision)
        self.assertEqual(revision['target_id'],self.before['planning_context']['revision']['target_id'])
        changed=deepcopy(self.result['treatment_head']);changed['effective_treatment_width_mm']+=1
        with self.assertRaises(ValueError):planning_revision.require_current(self.model,self.region,changed,revision)
        region=deepcopy(self.region);region['triangle_indices']=region['triangle_indices'][:-1]
        with self.assertRaises(ValueError):planning_revision.require_current(self.model,region,self.result['treatment_head'],revision)

    def test_pure_metadata_and_portable_paths_do_not_invalidate(self):
        region=deepcopy(self.region);region.update(name='新名称',note='回放备注',source_mesh='relocated.ply')
        planning_revision.require_current(self.model,region,self.result['treatment_head'],self.result['planning_context']['revision'])
        trajectory=SurfaceTrajectory.from_dict(self.result['surface_trajectory'])
        moved=replace(trajectory,source=dict(trajectory.source,region_manifest='other/region.json'),data_layers={'display':'moved'})
        self.assertEqual(coverage_workflow.route_binding(trajectory),coverage_workflow.route_binding(moved))

    def test_missing_trial_cannot_be_hidden_by_rehashing_zip(self):
        path=self.save()
        self.rewrite(path,lambda data,files:files.pop('refinement_1_coverage_certificate.npz'))
        with self.assertRaisesRegex(ValueError,'补充规划证据'):self.open(path)

    def test_orphan_trial_is_rejected_on_a_legacy_route(self):
        names={'coverage_certificate.json','coverage_certificate.npz','refinement_0_coverage_certificate.json','refinement_0_coverage_certificate.npz'}
        with self.assertRaisesRegex(ValueError,'孤立'):
            refinement_archive.check_inventory(names,SurfaceTrajectory.from_dict(self.before['surface_trajectory']))

    def test_rehashed_gain_claim_must_reproduce(self):
        path=self.save()
        def mutate(data,files):
            plan=data['plan'];report=plan['surface_trajectory']['gap_report']['refinement']
            report['rounds'][0]['comparison']['gained_nominal_area_m2']+=.00001
            plan['plan_binding']=planning_state.binding(self.model.store.mesh,self.region,plan)
        self.rewrite(path,mutate)
        with self.assertRaisesRegex(ValueError,'无法完整重现'):self.open(path)

    def test_missing_cache_and_metadata_only_native_export_cannot_publish(self):
        self.model.coverage_certificate=None
        with self.assertRaisesRegex(ValueError,'资产缺失'):self.save()
        self.assertFalse((self.root/'plan.zip').exists())
        with self.assertRaisesRegex(ValueError,'完整覆盖证据'):
            export_trajectory(SurfaceTrajectory.from_dict(self.result['surface_trajectory']),self.root/'native',include_visuals=False)

    def test_failed_replay_keeps_existing_saved_version(self):
        path=self.save();before=path.read_bytes();revision=self.model.revision_id
        with patch('femto.planning.refinement_archive.replay',side_effect=RuntimeError('cancelled replay')):
            with self.assertRaisesRegex(RuntimeError,'cancelled replay'):self.save()
        self.assertEqual(before,path.read_bytes());self.assertEqual(revision,self.model.revision_id)

    def test_failed_display_keeps_all_prior_caches(self):
        previous=(self.model.coverage_certificate,self.model.target_ledger,self.model.coverage_display)
        with patch('femto.planning.coverage_display.build',side_effect=RuntimeError('display cancelled')):
            with self.assertRaisesRegex(RuntimeError,'display cancelled'):
                refinement_workflow.restore(self.model,self.region,self.result)
        for old,new in zip(previous,(self.model.coverage_certificate,self.model.target_ledger,self.model.coverage_display)):
            self.assertIs(old,new)

    def test_new_route_without_coverage_is_invalid_and_cannot_reset_budgets(self):
        plan=deepcopy(self.result);plan.pop('coverage_evidence')
        plan['plan_binding']=planning_state.binding(self.model.store.mesh,self.region,plan)
        with self.assertRaisesRegex(ValueError,'覆盖证据'):planning_state.validate(plan,self.model.store.mesh,self.region)
        with self.assertRaisesRegex(ValueError,'恢复补充前'):
            refinement_workflow.calculate(self.model,self.region,self.result)

    def test_serialized_or_changed_receipt_cannot_bypass_replay(self):
        route=SurfaceTrajectory.from_dict(self.result['surface_trajectory'])
        ledger=self.model.target_ledger;certificate=self.model.coverage_certificate
        receipt=refinement_archive.issue_verified_assets(route,ledger,certificate)
        for invalid in ({'binding_sha256':receipt.binding_sha256},replace(receipt,issuer=object())):
            with self.assertRaisesRegex(ValueError,'凭据'):
                refinement_archive.require_verified_assets(invalid,route,ledger,certificate)
        changed=deepcopy(route);changed.gap_report['refinement']['stop_reason']='low_gain_limit'
        with self.assertRaisesRegex(ValueError,'凭据'):
            refinement_archive.require_verified_assets(receipt,changed,ledger,certificate)

    def test_receipt_rechecks_actual_arrays_against_immutable_metadata(self):
        route=SurfaceTrajectory.from_dict(self.result['surface_trajectory'])
        ledger=self.model.target_ledger;certificate=self.model.coverage_certificate
        receipt=refinement_archive.issue_verified_assets(route,ledger,certificate)
        arrays=dict(ledger.arrays);arrays['area_m2']=arrays['area_m2'].copy();arrays['area_m2'][0]*=2
        with self.assertRaisesRegex(ValueError,'目标数组'):
            refinement_archive.require_verified_assets(receipt,route,replace(ledger,arrays=arrays),certificate)
