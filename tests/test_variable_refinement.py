"""P5 positive admission, whole-chain rebinding and independently replayed history."""
from copy import copy,deepcopy
from pathlib import Path
import tempfile
import unittest
import os
import shutil
import json
from unittest.mock import patch
from datetime import date
from dataclasses import replace

import numpy as np

from test_strict_workflow import strict_fixture
from femto.planning import variable_footprint,variable_width_workflow,variable_refinement_workflow as workflow
from femto.planning import variable_width_refinement as refinement,coverage_workflow,coverage_certificate,refinement_archive
from femto.planning import planning_state,planning_revision,surface_workspace,region_project
from femto.planning.surface_trajectory import SurfaceTrajectory,validate_trajectory
from femto.planning.trajectory_export import load_trajectory


class VariableRefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.base=strict_fixture(Path(cls.temp.name)/'source')
        base,region,_,strict,_=cls.base
        radius=strict['treatment_head']['effective_treatment_width_mm']/2000
        profile=variable_footprint.nominal([dict(position=t,lower_half_width_m=radius*.9,upper_half_width_m=upper)
            for t,upper in ((0.,radius*.95),(.5,radius),(1.,radius*.95))],
            name='synthetic nonconstant upper width',version='1',valid_until='2099-12-31')
        plan=variable_width_workflow.calculate(base,region,strict,profile)
        certificate,ledger=coverage_workflow.calculate(base,region,plan,budget=coverage_certificate.CoverageBudget(max_depth=1))
        cls.variable=coverage_workflow.attach(base,region,plan,certificate,ledger)
        cls.final_model=copy(base);cls.final_model.store=copy(base.store)
        cls.final_plan=workflow.calculate(cls.final_model,region,cls.variable,
                                        budget=refinement.fixed.RefinementBudget(max_rounds=1))

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def setUp(self):
        self.case=tempfile.TemporaryDirectory();self.addCleanup(self.case.cleanup);self.root=Path(self.case.name)
        base,self.region,_,strict,_=self.base
        self.model=copy(base);self.model.store=copy(base.store);self.region=deepcopy(self.region)
        self.model.store.regions=[self.region];self.model.reconstruction=self.root/'work';self.model.reconstruction.mkdir()
        self.strict=deepcopy(strict)
        self.before=deepcopy(self.variable)

    def install_final(self):
        for name in ('target_ledger','coverage_certificate','coverage_display'):
            setattr(self.model,name,getattr(self.final_model,name))
        return deepcopy(self.final_plan)

    def test_actual_positive_gain_preserves_original_target_and_replays_history(self):
        before=deepcopy(self.before);arrays=dict(self.model.target_ledger.arrays)
        result=self.install_final()
        self.assertEqual(self.before,before)
        self.assertEqual(result['surface_trajectory']['planner']['algorithm_version'],refinement.ALGORITHM)
        report=result['surface_trajectory']['gap_report']['refinement']
        accepted=[r for r in report['rounds'] if r['accepted']]
        self.assertTrue(accepted)
        self.assertTrue(all(r['comparison']['lost_weight_units']==0 and r['comparison']['new_scan_witness_weight_units']>0 for r in accepted))
        self.assertGreater(result['surface_trajectory']['statistics']['point_count'],before['surface_trajectory']['statistics']['point_count'])
        for key,value in arrays.items():np.testing.assert_array_equal(value,self.model.target_ledger.arrays[key])
        planning_state.validate(result,self.model.mesh,self.region)
        planning_revision.require_current(self.model,self.region,result['treatment_head'],result['planning_context']['revision'])
        trajectory=SurfaceTrajectory.from_dict(result['surface_trajectory'])
        proof=refinement_archive.replay(self.model.coverage_certificate,trajectory,self.model.target_ledger)
        self.assertIs(proof,self.model.coverage_certificate)
        restored=workflow.restore(self.model,self.region,result)
        self.assertEqual(restored,self.before)

    def test_native_portable_replay_and_two_level_restoration(self):
        result=self.install_final();receipts=[]
        native=surface_workspace.export(self.model,self.region,result,verified_sink=receipts.append)
        self.assertEqual(len(receipts),1)
        loaded=load_trajectory(native)
        from femto.planning.refinement_validation import semantic_payload
        expected=SurfaceTrajectory.from_dict(result['surface_trajectory'])
        self.assertEqual(planning_revision.digest(semantic_payload(loaded)),planning_revision.digest(semantic_payload(expected)))
        path=region_project.save_project(self.model,self.region,self.root/'refined.region.zip',result,verified_assets=receipts[0])
        opened=region_project.open_project(path,workspace=self.root/'opened')
        self.assertEqual(opened.saved_plan['surface_trajectory'],result['surface_trajectory'])
        first=workflow.restore(opened,opened.store.regions[0],opened.saved_plan)
        self.assertEqual(planning_revision.digest(first['surface_trajectory']),planning_revision.digest(self.before['surface_trajectory']))
        self.assertEqual(first['coverage_evidence'],self.before['coverage_evidence'])
        self.assertEqual(first['planning_context'],self.before['planning_context'])
        second=variable_width_workflow.restore(opened,opened.store.regions[0],first)
        self.assertEqual(second['surface_trajectory'],self.strict['surface_trajectory'])
        evidence=os.environ.get('FEMTO_ITERATION_EVIDENCE')
        if evidence:
            output=Path(evidence);output.mkdir(parents=True,exist_ok=True)
            shutil.copytree(native,output/'native')
            shutil.copyfile(path,output/'refined.region.zip')
            (output/'summary.json').write_text(json.dumps(dict(
                algorithm=refinement.ALGORITHM,refinement=result['surface_trajectory']['gap_report']['refinement'],
                native_and_portable_independent_replay=True,two_level_restoration=True),ensure_ascii=False,indent=2),encoding='utf-8')
        history_file=next(native.glob('refinement_0_*.json'))
        missing=history_file.with_suffix('.missing');history_file.rename(missing)
        try:
            with self.assertRaises((ValueError,FileNotFoundError)):load_trajectory(native)
        finally:missing.rename(history_file)

    def test_resealed_false_width_diagnostic_is_rejected_geometrically(self):
        result=SurfaceTrajectory.from_dict(self.final_plan['surface_trajectory'])
        report=deepcopy(result.gap_report['variable_width']);report['records'][0]['residual_margin_mm']=999.
        fake=replace(result,gap_report=dict(result.gap_report,variable_width=report))
        mesh=surface_workspace.canonical_region_mesh(self.model.mesh,self.region)
        with self.assertRaises(ValueError):validate_trajectory(fake,mesh=mesh,region=self.region)

    def test_resealed_variable_width_gain_claim_requires_full_replay(self):
        trajectory=SurfaceTrajectory.from_dict(self.final_plan['surface_trajectory'])
        report=deepcopy(trajectory.gap_report['refinement'])
        report['rounds'][0]['comparison']['gained_nominal_area_m2']+=.00001
        fake=replace(trajectory,gap_report=dict(trajectory.gap_report,refinement=report))
        validate_trajectory(fake)
        self.assertEqual(coverage_workflow.route_binding(fake),coverage_workflow.route_binding(trajectory))
        with self.assertRaisesRegex(ValueError,'无法完整重现'):
            refinement_archive.replay(self.final_model.coverage_certificate,fake,self.final_model.target_ledger)

    def test_expired_refined_history_restores_but_cannot_publish(self):
        result=self.install_final();require=variable_footprint.require_current
        with patch.object(variable_footprint,'require_current',side_effect=lambda model:require(model,on=date(2100,1,1))):
            self.assertFalse(planning_state.state(self.region,result)['can_use'])
            with self.assertRaisesRegex(ValueError,'已过期'):
                surface_workspace.export(self.model,self.region,result)
            with self.assertRaisesRegex(ValueError,'已过期'):
                region_project.save_project(self.model,self.region,self.root/'forbidden.zip',result)
            restored=workflow.restore(self.model,self.region,result)
            self.assertEqual(restored,self.before)
        self.assertFalse((self.root/'forbidden.zip').exists())

    def test_cancel_and_model_expiry_before_install_preserve_all_caches(self):
        cache=self.model.target_ledger,self.model.coverage_certificate,self.model.coverage_display
        require=variable_footprint.require_current;calls=0
        def expire(model):
            nonlocal calls
            calls+=1
            return require(model,on=date(2026,1,1) if calls==1 else date(2100,1,1))
        trajectory=SurfaceTrajectory.from_dict(self.final_plan['surface_trajectory'])
        proposal=(trajectory,self.final_model.target_ledger,self.final_model.coverage_certificate)
        with patch.object(refinement,'refine',return_value=proposal),patch.object(variable_footprint,'require_current',side_effect=expire):
            with self.assertRaisesRegex(ValueError,'已过期'):workflow.calculate(self.model,self.region,self.before)
        for a,b in zip(cache,(self.model.target_ledger,self.model.coverage_certificate,self.model.coverage_display)):self.assertIs(a,b)
        def cancel(*args,**kwargs):raise RuntimeError('cancelled proposal')
        with patch.object(refinement,'refine',side_effect=cancel):
            with self.assertRaisesRegex(RuntimeError,'cancelled proposal'):workflow.calculate(self.model,self.region,self.before)
        for a,b in zip(cache,(self.model.target_ledger,self.model.coverage_certificate,self.model.coverage_display)):self.assertIs(a,b)

    def test_budget_failure_keeps_base_and_records_failure_without_publishing(self):
        cache=self.model.target_ledger,self.model.coverage_certificate,self.model.coverage_display
        with self.assertRaisesRegex(ValueError,'预算'):
            workflow.calculate(self.model,self.region,self.before,budget=refinement.fixed.RefinementBudget(max_edge_checks=1,max_rounds=1))
        for a,b in zip(cache,(self.model.target_ledger,self.model.coverage_certificate,self.model.coverage_display)):self.assertIs(a,b)
        self.assertEqual(len(list((self.model.reconstruction/'variable_refinement_attempts').glob('*.json'))),1)

    def test_nonconstant_width_rebinds_old_edges_and_checks_interior_peak(self):
        old=SurfaceTrajectory.from_dict(self.before['surface_trajectory'])
        new=SurfaceTrajectory.from_dict(self.final_plan['surface_trajectory'])
        radius=self.strict['treatment_head']['effective_treatment_width_mm']/2000
        model=variable_footprint.nominal([dict(position=t,lower_half_width_m=r,upper_half_width_m=radius)
            for t,r in ((0.,radius),(0.5,radius/2),(1.,radius))],name='variable interior',version='1')
        before=refinement.bounds(old,model);after=refinement.bounds(new,model)
        self.assertGreater(len(after),len(before))
        # The admitted candidate can be a new disconnected chain. Prove the
        # reparameterization rule separately by extending one existing chain.
        items=list(old.ordered_items);index=next(i for i,s in enumerate(items) if isinstance(s,refinement.ScanSegment))
        points=deepcopy(items[index].points);last=deepcopy(points[-1]);last['xyz_m']=list(np.asarray(last['xyz_m'])+[.001,0.,0.])
        last['role']='scan';points.append(last);items[index]=replace(items[index],points=points)
        changed=replace(old,ordered_items=items)
        rebound=refinement.bounds(changed,model)
        old_rows=[r for r in before if r['segment_index']==items[index].segment_index]
        new_rows=[r for r in rebound if r['segment_index']==items[index].segment_index]
        self.assertTrue(any(a['interval']!=b['interval'] and a['lower_half_width_m']!=b['lower_half_width_m']
                            for a,b in zip(old_rows,new_rows)))
        self.assertTrue(any(row['role']=='connector' for row in after))
        peak=deepcopy(model['knots']);peak[1]['upper_half_width_m']=radius*2
        peak_model=variable_footprint.nominal(peak,name='interior upper peak',version='1')
        with self.assertRaisesRegex(ValueError,'段内上界'):refinement.bounds(new,peak_model)
        seeds=refinement.scan_endpoints(new,self.model.target_ledger.arrays['region_faces'])
        rows={(r['segment_index'],r['edge_index']):r for r in new.gap_report['variable_width']['records']}
        self.assertTrue(seeds)
        self.assertTrue(all(rows[s['segment_index'],s['edge_index']]['role']!='connector' for s in seeds))

    def test_zero_inner_width_cannot_admit_a_supplement(self):
        radius=self.strict['treatment_head']['effective_treatment_width_mm']/2000
        zero=variable_footprint.nominal([dict(position=t,lower_half_width_m=0.,upper_half_width_m=radius)
            for t in (0.,1.)],name='zero lower width',version='1')
        strict=variable_width_workflow.restore(self.model,self.region,self.before)
        plan=variable_width_workflow.calculate(self.model,self.region,strict,zero)
        certificate,ledger=coverage_workflow.calculate(self.model,self.region,plan,budget=coverage_certificate.CoverageBudget(max_depth=1))
        plan=coverage_workflow.attach(self.model,self.region,plan,certificate,ledger)
        cache=self.model.target_ledger,self.model.coverage_certificate,self.model.coverage_display
        with self.assertRaisesRegex(ValueError,'未获得'):
            workflow.calculate(self.model,self.region,plan,budget=refinement.fixed.RefinementBudget(max_rounds=1))
        for a,b in zip(cache,(self.model.target_ledger,self.model.coverage_certificate,self.model.coverage_display)):self.assertIs(a,b)


if __name__=='__main__':unittest.main()
