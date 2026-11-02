"""P3 geometry, exact common-partition accounting and bounded refusal cases."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from test_coverage_workflow import fixture
from femto.planning import coverage_partition as partition, coverage_refinement as refinement
from femto.planning import coverage_workflow, refinement_candidates, surface_workspace, planning_revision
from femto.planning.coverage_domain import CoverageDomain
from femto.planning.surface_trajectory import SurfaceTrajectory, ScanSegment, validate_trajectory


def cells(rows):
    values = np.array(rows, dtype=np.int64).reshape(-1, 4)
    return SimpleNamespace(arrays=dict(zip(('root','depth','code','status'), values.T)))


class CommonPartitionTests(unittest.TestCase):
    def test_refining_a_confirmed_cell_does_not_invent_gain(self):
        old=cells([(0,0,0,1)]);new=cells([(0,1,i,1) for i in range(4)])
        result=partition.compare(old,new,np.array([8.]))
        self.assertEqual(result['retained_nominal_area_m2'],8.)
        self.assertEqual(result['gained_weight_units'],0)
        self.assertEqual(result['lost_weight_units'],0)

    def test_independent_quarter_truth_with_mixed_depth(self):
        old=cells([(0,1,i,1 if i==0 else 3) for i in range(4)])
        new=cells([(0,1,0,1)]+[(0,2,i,1) for i in range(4,8)]+[(0,1,i,3) for i in (2,3)])
        result=partition.compare(old,new,np.array([8.]),new_witnesses=np.ones(7,bool))
        self.assertEqual(result['retained_nominal_area_m2'],2.)
        self.assertEqual(result['gained_nominal_area_m2'],2.)
        self.assertEqual(result['new_scan_witness_gain_m2'],2.)
        self.assertTrue(result['retained_subset'])

    def test_larger_total_still_detects_loss_elsewhere(self):
        old=cells([(0,1,i,1 if i==0 else 3) for i in range(4)])
        new=cells([(0,1,i,3 if i==0 else 1) for i in range(4)])
        result=partition.compare(old,new,np.array([4.]))
        self.assertEqual(result['gained_nominal_area_m2'],3.)
        self.assertEqual(result['lost_nominal_area_m2'],1.)
        self.assertFalse(result['retained_subset'])

    def test_distinct_roots_are_never_cancelled_against_one_another(self):
        result=partition.compare(cells([(0,0,0,1),(1,0,0,3)]),cells([(0,0,0,3),(1,0,0,1)]),np.array([1.,1.]))
        self.assertFalse(result['retained_subset'])
        self.assertEqual(result['lost_weight_units'],partition.UNITS)

    def test_no_gain_credit_for_old_scan_witnesses(self):
        result=partition.compare(cells([(0,0,0,3)]),cells([(0,0,0,1)]),np.array([2.]),new_witnesses=np.zeros(1,bool))
        self.assertEqual(result['gained_nominal_area_m2'],2.)
        self.assertEqual(result['new_scan_witness_gain_m2'],0.)

    def test_missing_duplicate_and_invalid_prefixes_are_rejected(self):
        for rows in ([(0,1,0,1)],[(0,0,0,1),(0,0,0,1)],[(0,9,0,1)],[(0,1,4,1)]):
            with self.subTest(rows=rows),self.assertRaises(ValueError):partition.intervals(cells(rows),1)

    def test_tiny_area_cannot_hide_set_loss(self):
        result=partition.compare(cells([(0,0,0,1)]),cells([(0,0,0,3)]),np.array([np.nextafter(0.,1.)]))
        self.assertFalse(result['retained_subset'])
        self.assertGreater(result['lost_weight_units'],0)

    def test_comparison_can_be_cancelled(self):
        def stop():raise RuntimeError('cancelled')
        with self.assertRaisesRegex(RuntimeError,'cancelled'):
            partition.compare(cells([(0,0,0,1)]),cells([(0,0,0,1)]),np.array([1.]),stop)


class RefinementKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        cls.model,cls.region,cls.plan,cls.certificate,cls.ledger=fixture(Path(cls.temp.name)/'source')
        cls.base=SurfaceTrajectory.from_dict(cls.plan['surface_trajectory'])
        cls.mesh=surface_workspace.canonical_region_mesh(cls.model.store.mesh,cls.region)
        cls.budget=refinement.RefinementBudget(max_rounds=1)
        cls.result,cls.target,cls.proof=refinement.refine(cls.base,cls.ledger,cls.certificate,cls.mesh,cls.region,budget=cls.budget)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def run_with(self,**values):
        return refinement.refine(self.base,self.ledger,self.certificate,self.mesh,self.region,
            budget=replace(self.budget,**values))

    def test_positive_case_adds_new_scan_witnesses_without_losing_old_coverage(self):
        rounds=self.result.gap_report['refinement']['rounds']
        self.assertTrue(rounds[0]['accepted'],rounds[0]['status'])
        self.assertGreater(rounds[0]['comparison']['new_scan_witness_gain_m2'],0)
        self.assertEqual(rounds[0]['comparison']['lost_weight_units'],0)
        self.assertGreater(self.result.statistics['point_count'],self.base.statistics['point_count'])
        validate_trajectory(self.result,mesh=self.mesh,region=self.region)

    def test_original_scan_sequences_and_evidence_are_preserved(self):
        self.assertEqual(self.result.gap_report['refinement']['base_trajectory'],self.base.to_dict())
        self.assertEqual(self.proof.history[0].metadata,self.certificate.metadata)
        def points(item):return [{k:v for k,v in p.items() if k!='route_index'} for p in item.points]
        final=[s for s in self.result.ordered_items if isinstance(s,ScanSegment)]
        for old in (s for s in self.base.ordered_items if isinstance(s,ScanSegment)):
            self.assertTrue(any(s.block_index==old.block_index and points(s)[:len(old.points)]==points(old) for s in final))

    def test_same_original_target_new_computation_and_unknown_error(self):
        a=self.base.planning_context['revision'];b=self.result.planning_context['revision']
        self.assertEqual(a['target_id'],b['target_id']);self.assertNotEqual(a['sha256'],b['sha256'])
        for name,value in self.ledger.arrays.items():np.testing.assert_array_equal(value,self.target.arrays[name])
        self.assertIsNone(self.result.gap_report['refinement']['registration_uncertainty_mm'])
        self.assertFalse(self.result.to_dict()['execution_ready'])
        planning_revision.require_current(self.model,self.region,self.plan['treatment_head'],b)

    def test_every_proposed_edge_reproduces_existing_v3_predicate(self):
        from femto.planning.quality_constrained import edge_checks
        from femto.planning.effective_domain import validate
        params=self.base.planner['parameters']
        allowed=validate(params['effective_domain'],params['quality_evidence'],params['effective_width_mm'],self.mesh,self.region)
        records=self.result.gap_report['refinement']['rounds'][0]['candidates']
        rejected=0
        for record in records:
            if record['status']!='checked':continue
            segment=ScanSegment(0,record['points'][0]['row_index'],record['points'],record['recipe']['block_index'])
            okay,reasons,margins=edge_checks([segment],self.mesh,params['quality_evidence'],params['effective_width_mm'],allowed_faces=allowed)
            self.assertEqual(record['edge_checks'],[dict(accepted=bool(a),reason=b,residual_margin_mm=float(c)) for a,b,c in zip(okay,reasons,margins)])
            rejected+=int((~okay).sum())
        self.assertGreater(rejected,0)

    def test_non_spacing_unknown_reasons_never_become_candidates(self):
        domain=CoverageDomain._checked_target(self.ledger,self.base.planning_context['revision'])
        arrays=dict(self.certificate.arrays)
        for code in (2,4,5,6,7):
            arrays['status']=np.full(len(arrays['status']),code,np.uint8)
            rows=refinement_candidates.ranked_faces(domain,SimpleNamespace(arrays=arrays),self.mesh,self.region,
                self.base.blocks,np.ones(len(self.mesh.faces),bool),set())
            self.assertEqual(rows,[])

    def test_all_new_connectors_have_saved_per_edge_pass_records(self):
        from femto.planning.effective_domain import validate
        from femto.planning.mesh_slice_planner import _project
        from femto.planning.selection_blocks import selection_face_assignment
        from femto.planning.quality_constrained import edge_checks
        params=self.base.planner['parameters']
        allowed=validate(params['effective_domain'],params['quality_evidence'],params['effective_width_mm'],self.mesh,self.region)
        vertices,faces=np.asarray(self.mesh.vertices),np.asarray(self.mesh.faces)
        _,_,_,owners=selection_face_assignment(self.region,vertices,faces)
        eligible=np.flatnonzero(allowed & (owners==self.base.blocks['items'][0]['component_id']))
        centers=vertices[faces[eligible]].mean(axis=1)
        face=int(eligible[np.argmin(np.linalg.norm(centers[:,:2],axis=1))])
        triangle=vertices[faces[face]];center=triangle.mean(axis=0)
        edges=np.roll(triangle,-1,axis=0)-triangle
        direction=edges[np.argmax(np.linalg.norm(edges,axis=1))]
        direction=direction/np.linalg.norm(direction)
        reference=params['quality_evidence']['view_reference']
        intrinsic=np.asarray(reference['undistorted_intrinsic'])
        size=np.asarray(params['quality_evidence']['image_size'])
        def points(offsets,row):
            return [dict(xyz_m=p.tolist(),triangle_index=face,pixel=_project(p,intrinsic,size),
                geometry_support='measured_mesh',role='scan',row_index=row)
                for p in (center+offset*direction for offset in offsets)]
        # Two short interior segments have a real admissible connector, even
        # when the coverage-driven fixture legitimately leaves all gaps open.
        left=points((-.0015,-.0005),0);right=points((.0005,.0015),1)
        chains=[(0,deepcopy(left))]
        positive=refinement._connect(chains,(0,right),self.base,self.mesh,self.region,
            allowed,refinement._Meter(self.budget),lambda:None)
        self.assertTrue(positive['connected'],positive['reason'])
        self.assertEqual(len(chains),1)
        self.assertEqual(positive['samples'][-1]['xyz_m'],right[0]['xyz_m'])
        connector=ScanSegment(0,1,[left[-1]]+positive['samples'],0)
        flags,reasons,margins=edge_checks([connector],self.mesh,params['quality_evidence'],
            params['effective_width_mm'],allowed_faces=allowed)
        self.assertEqual(positive['edge_checks'],[dict(accepted=bool(a),reason=b,residual_margin_mm=float(c))
            for a,b,c in zip(flags,reasons,margins)])
        links=[link for record in self.result.gap_report['refinement']['rounds'][0]['candidates'] for link in record['connections']]
        for link in [positive]+links:
            self.assertEqual(len(link['samples']),len(link['edge_checks']))
            self.assertTrue(all(d['accepted'] and d['reason']=='accepted' for d in link['edge_checks']))

    def test_other_work_block_never_invokes_a_transfer_search(self):
        points=deepcopy(next(s.points for s in self.base.ordered_items if isinstance(s,ScanSegment))[:2])
        chains=[(1,deepcopy(points))]
        with patch.object(refinement._Graph,'__init__',side_effect=AssertionError('cross-block search')):
            link=refinement._connect(chains,(0,points),self.base,self.mesh,self.region,
                np.ones(len(self.mesh.faces),bool),refinement._Meter(self.budget),lambda:None)
        self.assertFalse(link['connected']);self.assertEqual(link['samples'],[])
        self.assertEqual([block for block,_ in chains],[0,1])

    def test_zero_edge_budget_retains_original_and_records_stop(self):
        route,target,cert=self.run_with(max_edge_checks=0)
        self.assertEqual(route.to_dict()['ordered_items'],self.base.to_dict()['ordered_items'])
        self.assertEqual(route.gap_report['refinement']['stop_reason'],'edge_budget')
        self.assertEqual(route.gap_report['refinement']['edge_checks'],0)
        self.assertEqual(len(cert.history),1)

    def test_zero_length_budget_never_installs_candidate(self):
        route,_,_=self.run_with(max_added_length_m=0)
        self.assertEqual(route.gap_report['refinement']['stop_reason'],'length_budget')
        self.assertEqual(route.to_dict()['ordered_items'],self.base.to_dict()['ordered_items'])

    def test_unfinished_coverage_cannot_displace_existing_confirmation(self):
        route,_,cert=self.run_with(coverage_seconds=0)
        self.assertEqual(route.gap_report['refinement']['rounds'][0]['status'],'coverage_regression')
        self.assertEqual(route.to_dict()['ordered_items'],self.base.to_dict()['ordered_items'])
        self.assertEqual(cert.summary['confirmed_nominal_area_m2'],self.certificate.summary['confirmed_nominal_area_m2'])

    def test_same_saved_stopping_frontier_reproduces_all_records(self):
        route,target,cert=refinement.refine(self.base,self.ledger,self.certificate,self.mesh,self.region,budget=self.budget,
            frontiers=[c.metadata['metrics'] for c in self.proof.history[1:]])
        self.assertEqual(route.to_dict(),self.result.to_dict())
        self.assertEqual(target.metadata,self.target.metadata)
        self.assertEqual(cert.metadata,self.proof.metadata)

    def test_point_limit_and_iteration_zero_keep_limits_explicit(self):
        route,_,_=self.run_with(max_output_points=self.base.statistics['point_count'])
        self.assertEqual(route.gap_report['refinement']['stop_reason'],'point_budget')
        route,_,cert=self.run_with(max_rounds=0)
        self.assertEqual(route.gap_report['refinement']['rounds'],[])
        self.assertEqual(len(cert.history),1)

    def test_invalid_budgets_and_nested_refinement_are_rejected(self):
        for values in (dict(max_rounds=3),dict(max_edge_checks=True),dict(coverage_seconds=float('nan')),dict(low_gain_rounds=0)):
            with self.assertRaises(ValueError):refinement.RefinementBudget(**values).checked()
        with self.assertRaises(ValueError):
            refinement.refine(self.result,self.target,self.proof,self.mesh,self.region)

    def test_cancellation_never_changes_original(self):
        original=deepcopy(self.base.to_dict())
        def stop():raise RuntimeError('cancelled')
        with self.assertRaisesRegex(RuntimeError,'cancelled'):
            refinement.refine(self.base,self.ledger,self.certificate,self.mesh,self.region,check=stop)
        self.assertEqual(original,self.base.to_dict())
