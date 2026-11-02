"""Bounded candidate screening must retain whole-edge and coverage admission."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import trimesh

from test_coverage_workflow import fixture
from femto.annotation.selection_intent import encode_runs, digest
from femto.planning import coverage_refinement as refinement, coverage_partition, refinement_archive
from femto.planning import refinement_candidates_v2 as candidates, quality_constrained as quality
from femto.planning import surface_workspace, effective_domain
from femto.planning.coverage_domain import CoverageDomain
from femto.planning.surface_trajectory import SurfaceTrajectory, ScanSegment, validate_trajectory


def evidence(classes):
    record = dict(schema=quality.EVIDENCE_SCHEMA, image_size=[100,100],
        view_reference=dict(image_size=[100,100], undistorted_intrinsic=[[100.,0.,50.],[0.,100.,50.],[0.,0.,1.]]),
        intent_sha256=digest(classes>0), classes_runs=encode_runs(classes),
        residual_um_runs=encode_runs(np.zeros_like(classes, dtype=np.int32)),
        residual_semantics='ceil_abs_mesh_minus_raw_reference_depth_times_ray_norm_um', depth_gate_mm=10.,
        registration_uncertainty_mm=None, calibration_uncertainty_mm=None,
        physical_accuracy_verified=False, execution_ready=False)
    record['sha256'] = quality._hash(record)
    return record


class ReferenceScreeningTests(unittest.TestCase):
    def test_diagnostics_distinguish_selection_quality_and_both_without_changing_gate(self):
        for values, reason in (([0,1],'outside_selection'), ([7,1],'reference_quality_unconfirmed'),
                               ([0,7],'outside_selection_and_quality'), ([1,1],'accepted')):
            with self.subTest(reason=reason):
                classes=np.ones((100,100),np.uint8);classes[50,49:51]=values
                gate=quality.ReferenceEnvelope(evidence(classes),5.)
                plain=gate.probe(np.array([0.,0.,.5]),.001)
                detailed=gate.probe(np.array([0.,0.,.5]),.001,details=True)
                self.assertEqual(plain[:2],detailed[:2])
                self.assertEqual(detailed[2]['reason'],reason)
                self.assertEqual(detailed[0],reason=='accepted')
                self.assertEqual(detailed[2]['outside_selection_pixels'],values.count(0))
                self.assertEqual(detailed[2]['quality_unconfirmed_pixels'],values.count(7))

    def test_projection_and_residual_budget_fail_closed(self):
        gate=quality.ReferenceEnvelope(evidence(np.ones((100,100),np.uint8)),5.)
        for point, reason in (([1.,0.,.5],'footprint_outside_image'), ([0.,0.,0.],'outer_projection_unavailable')):
            passed, _, detail=gate.probe(np.array(point),.001,details=True)
            self.assertFalse(passed);self.assertEqual(detail['reason'],reason)
        gate.sampled_pixels=99_999_999
        with self.assertRaisesRegex(ValueError,'超过预算'):gate.probe(np.array([0.,0.,.5]),.001)

    def test_passing_reference_probe_cannot_admit_a_chord_above_the_mesh(self):
        mesh=trimesh.Trimesh(vertices=[[-.05,-.05,.5],[.05,-.05,.5],[.05,.05,.5],[-.05,.05,.5]],
                             faces=[[0,1,2],[0,2,3]],process=False)
        record=evidence(np.ones((100,100),np.uint8))
        points=[dict(xyz_m=[x,0.,.501],triangle_index=0,role='scan',row_index=0) for x in (-.001,.001)]
        self.assertTrue(quality.ReferenceEnvelope(record,1.).probe(np.array([0.,0.,.501]),.001)[0])
        diagnostics=[]
        okay,reasons,_=quality.edge_checks([ScanSegment(0,0,points,0)],mesh,record,1.,diagnostics=diagnostics)
        self.assertFalse(okay.any());self.assertEqual(reasons,['chord_not_certified_on_mesh'])
        self.assertEqual(diagnostics[0]['reason'],reasons[0]);self.assertFalse(diagnostics[0]['accepted'])


class ScreenedCandidatesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        cls.model,cls.region,cls.plan,cls.certificate,cls.ledger=fixture(Path(cls.temp.name)/'source')
        cls.base=SurfaceTrajectory.from_dict(cls.plan['surface_trajectory'])
        cls.mesh=surface_workspace.canonical_region_mesh(cls.model.store.mesh,cls.region)
        cls.budget=refinement.RefinementBudget(max_rounds=1)
        cls.result,cls.target,cls.proof=refinement.refine(cls.base,cls.ledger,cls.certificate,cls.mesh,cls.region,
            budget=cls.budget,candidate_policy=deepcopy(candidates.POLICY))
        cls.domain=CoverageDomain._checked_target(cls.ledger,cls.base.planning_context['revision'])
        params=cls.base.planner['parameters']
        cls.allowed=effective_domain.validate(params['effective_domain'],params['quality_evidence'],
                                             params['effective_width_mm'],cls.mesh,cls.region)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_positive_gain_has_new_scan_witness_and_preserves_original_set_and_arrays(self):
        report=self.result.gap_report['refinement'];round_=report['rounds'][0]
        self.assertTrue(round_['accepted'],round_['status'])
        self.assertEqual(round_['comparison']['lost_weight_units'],0)
        self.assertGreater(round_['comparison']['new_scan_witness_gain_m2']*1e6,self.budget.min_gain_mm2)
        comparison=coverage_partition.compare(self.certificate,self.proof,self.domain.areas)
        self.assertTrue(comparison['retained_subset']);self.assertEqual(comparison['lost_weight_units'],0)
        for name,value in self.ledger.arrays.items():np.testing.assert_array_equal(value,self.target.arrays[name])
        self.assertEqual(report['base_trajectory'],self.base.to_dict())
        self.assertEqual(self.proof.history[0].metadata,self.certificate.metadata)
        self.assertLessEqual(self.result.statistics['total_length_m'],report['max_total_length_m'])
        validate_trajectory(self.result,mesh=self.mesh,region=self.region)

    def test_version_and_exact_policy_are_bound_and_legacy_remains_distinct(self):
        legacy=refinement.algorithms_for(self.base,self.certificate,self.budget)
        current=refinement.algorithms_for(self.base,self.certificate,self.budget,candidates.POLICY)
        self.assertEqual(legacy['refinement']['schema'],refinement.SCHEMA)
        self.assertEqual(current['refinement']['schema'],refinement.OPTIMIZED_SCHEMA)
        self.assertNotEqual(legacy,current)
        self.assertTrue(refinement.checked_algorithms(legacy));self.assertTrue(refinement.checked_algorithms(current))
        for change in ({'max_faces_per_round':257}, {'max_faces_per_round':256.0},
                       {'screening_charged_to_edge_budget':1}, {'connection_length_policy':'unlimited'}):
            with self.subTest(change=change):
                altered=deepcopy(current);altered['refinement']['candidate_policy'].update(change)
                self.assertFalse(refinement.checked_algorithms(altered))
        altered=deepcopy(self.result);altered.gap_report['refinement'].pop('candidate_policy')
        with self.assertRaisesRegex(ValueError,'策略缺失|不同版本'):validate_trajectory(altered)

    def test_screening_consumes_shared_edge_budget_and_leaves_complete_original(self):
        route,_,proof=refinement.refine(self.base,self.ledger,self.certificate,self.mesh,self.region,
            budget=replace(self.budget,max_edge_checks=1),candidate_policy=candidates.POLICY)
        report=route.gap_report['refinement'];search=report['rounds'][0]['candidate_search']
        self.assertEqual(report['stop_reason'],'edge_budget');self.assertEqual(report['edge_checks'],1)
        self.assertEqual(search['reference_checks'],1)
        self.assertEqual(search['unexamined_ranked_faces'],search['ranked_faces']-search['screened_faces'])
        self.assertEqual(route.to_dict()['ordered_items'],self.base.to_dict()['ordered_items'])
        self.assertEqual(len(proof.history),1)
        self.assertTrue(coverage_partition.compare(self.certificate,proof,self.domain.areas)['retained_subset'])

    def test_zero_coverage_budget_cannot_replace_or_union_the_original_certificate(self):
        route,_,proof=refinement.refine(self.base,self.ledger,self.certificate,self.mesh,self.region,
            budget=replace(self.budget,coverage_seconds=0),candidate_policy=candidates.POLICY)
        round_=route.gap_report['refinement']['rounds'][0]
        self.assertEqual(round_['status'],'coverage_regression');self.assertFalse(round_['accepted'])
        self.assertGreater(round_['comparison']['lost_weight_units'],0)
        self.assertEqual(route.to_dict()['ordered_items'],self.base.to_dict()['ordered_items'])
        for name,values in self.certificate.arrays.items():np.testing.assert_array_equal(values,proof.arrays[name])
        self.assertEqual(len(proof.history),2)

    def test_cancel_during_reference_screening_propagates_without_mutating_inputs(self):
        arrays={key:value.copy() for key,value in self.ledger.arrays.items()}
        with patch.object(quality.ReferenceEnvelope,'probe',side_effect=RuntimeError('cancel during screening')):
            with self.assertRaisesRegex(RuntimeError,'cancel during screening'):
                refinement.refine(self.base,self.ledger,self.certificate,self.mesh,self.region,
                    budget=self.budget,candidate_policy=candidates.POLICY)
        for key,value in arrays.items():np.testing.assert_array_equal(value,self.ledger.arrays[key])
        self.assertEqual(self.base.to_dict(),self.plan['surface_trajectory'])

    def test_acquisition_connectivity_construction_and_budget_unknowns_are_not_candidates(self):
        for code in (2,4,5,6,7):
            arrays=dict(self.certificate.arrays);arrays['status']=np.full(len(arrays['status']),code,np.uint8)
            meter=refinement._Meter(self.budget);meter.candidate_policy=candidates.POLICY
            recipes=candidates.select(self.base,self.domain,SimpleNamespace(arrays=arrays),self.mesh,self.region,
                                      self.allowed,set(),meter,lambda:None)
            self.assertEqual(recipes,[]);self.assertEqual(meter.edge_checks,0)

    def test_long_connection_is_left_as_a_gap_before_expensive_search(self):
        first=self.result.gap_report['refinement']['rounds'][0]['candidates'][0]
        meter=refinement._Meter(self.budget);meter.candidate_policy=candidates.POLICY
        chains=refinement._chains(self.base)
        with patch.object(refinement._Graph,'__init__',side_effect=AssertionError('unnecessary graph search')):
            link=refinement._connect(chains,(first['recipe']['block_index'],first['points']),self.result,
                self.mesh,self.region,self.allowed,meter,lambda:None)
        self.assertFalse(link['connected']);self.assertEqual(link['reason'],'connection_length_budget')
        self.assertGreater(link['connection_budget']['endpoint_lower_bound_length_m'],link['connection_budget']['maximum_m'])
        self.assertEqual(link['samples'],[]);self.assertEqual(meter.edge_checks,0)
        self.assertEqual(len(chains),len(refinement._chains(self.base))+1)

    def test_search_caps_and_diagnostics_are_replayed_and_tampering_is_refused(self):
        round_=self.result.gap_report['refinement']['rounds'][0];search=round_['candidate_search']
        self.assertLessEqual(search['screened_faces'],candidates.POLICY['max_faces_per_round'])
        self.assertLessEqual(search['positions_considered'],search['screened_faces']*candidates.POLICY['max_positions_per_face'])
        self.assertLessEqual(len(search['rejection_examples']),candidates.POLICY['max_rejection_examples'])
        self.assertFalse(search['screening_is_admission']);self.assertEqual(len(round_['candidates']),4)
        refinement_archive.replay(self.proof,self.result,self.target)
        altered=deepcopy(self.result)
        altered.gap_report['refinement']['rounds'][0]['candidate_search']['reference_checks']+=1
        with self.assertRaisesRegex(ValueError,'无法完整重现'):refinement_archive.replay(self.proof,altered,self.target)
        altered=deepcopy(self.result)
        altered.gap_report['refinement']['rounds'][0]['candidates'][0]['quality_diagnostics'][0]['residual_margin_mm']+=1
        with self.assertRaisesRegex(ValueError,'无法完整重现'):refinement_archive.replay(self.proof,altered,self.target)

    def test_installed_summary_shows_diagnostics_without_replaying_proofs_on_ui_thread(self):
        with patch.object(SurfaceTrajectory,'from_dict',side_effect=AssertionError('presentation replayed proof')):
            text=refinement.summary_text(self.result.to_dict())
        self.assertIn('候选筛查',text);self.assertIn('作用范围越出工作区域',text);self.assertIn('保留断点',text)


if __name__=='__main__':unittest.main()
