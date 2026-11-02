"""Lazy links remain untrusted until whole-edge certification and final replay."""
from copy import deepcopy
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import open3d as o3d

from test_fixed_rgb_experience import make_model, K
from test_mesh_slice_planner import grid_mesh
from femto.planning import continuous_quality as continuous, quality_constrained as quality
from femto.planning import surface_workspace as workspace, effective_domain, planning_revision
from femto.planning.surface_trajectory import SurfaceTrajectory


class LazySurfaceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        mesh=grid_mesh(width=.12,height=.08,cell=.005)
        self.model=make_model(self.temp.name,o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(mesh.vertices),o3d.utility.Vector3iVector(mesh.faces)))
        self.model.view_reference['undistorted_intrinsic']=K.tolist()
        self.model.store.origin_reference=dict(kind='initial_color_camera',provenance='synthetic_test')

    def inputs(self,split=False):
        mask=self.model.projection.valid.copy()
        mask[47:51,47:51]=False
        if split:mask[:,52:55]=False
        region=self.model.region([[30,30],[70,30],[70,70],[30,70]],'test','',
            selection=dict(mask=mask,method='work_only'))
        mesh=workspace.canonical_region_mesh(self.model.mesh,region)
        record=quality.evidence(self.model,region)
        domain=effective_domain.build(mesh,region,record,5)
        manifest=workspace.snapshot(self.model,region,lambda:None)
        candidate=continuous.candidate(manifest,row_spacing_mm=5)
        return candidate,mesh,region,record,domain

    def test_lazy_matches_eager_measured_routes_for_holes_and_separate_blocks(self):
        for split in (False,True):
            with self.subTest(split=split):
                candidate,mesh,region,record,domain=self.inputs(split)
                original=deepcopy(region)
                eager=continuous.replan(candidate,mesh,region,record,5,domain=domain,routing=continuous.ROUTING)
                lazy=continuous.replan(candidate,mesh,region,record,5,domain=domain)
                continuous.validate(eager,mesh,region);continuous.validate(lazy,mesh,region)
                self.assertEqual([i.to_dict() for i in eager.ordered_items],[i.to_dict() for i in lazy.ordered_items])
                self.assertEqual(eager.statistics,lazy.statistics)
                self.assertLessEqual(lazy.gap_report['reconnection']['graph_checks'],eager.gap_report['reconnection']['graph_checks'])
                self.assertEqual(region,original)
                if split:self.assertGreater(lazy.statistics['inter_block_gap_count'],0)
                for plan in (lazy,eager):
                    reopened=SurfaceTrajectory.from_dict(plan.to_dict())
                    continuous.validate(reopened,mesh,region)

    def test_no_optimistic_edge_can_be_emitted_when_certification_fails(self):
        candidate,mesh,region,record,domain=self.inputs()
        # Retained scans are still checked normally; reject every graph pair.
        with patch.object(continuous.VerifiedSurfaceGraph,'_admit_pairs',return_value=None):
            lazy=continuous.replan(candidate,mesh,region,record,5,domain=domain)
        report=lazy.gap_report['reconnection']
        self.assertEqual(report['added_connector_edges'],0)
        self.assertEqual(report['before_chains'],report['after_chains'])
        self.assertTrue(all(not link['connected'] for link in report['connections']))
        continuous.validate(lazy,mesh,region)

    def test_budget_and_cancel_fail_without_returning_a_plan(self):
        candidate,mesh,region,record,domain=self.inputs()
        with patch.object(continuous,'BUDGET',5):
            with self.assertRaisesRegex(ValueError,'超过预算'):
                continuous.replan(candidate,mesh,region,record,5,domain=domain)
        def cancel():raise RuntimeError('operator cancelled')
        with self.assertRaisesRegex(RuntimeError,'operator cancelled'):
            continuous.replan(candidate,mesh,region,record,5,domain=domain,check=cancel)

    def test_ordered_routing_preserves_scans_and_replays_checked_decisions(self):
        candidate,mesh,region,record,domain=self.inputs(True)
        ordered=continuous.replan(candidate,mesh,region,record,5,domain=domain,routing=continuous.ORDERED_ROUTING)
        continuous.validate(ordered,mesh,region)
        reopened=SurfaceTrajectory.from_dict(ordered.to_dict())
        continuous.validate(reopened,mesh,region)
        decision=ordered.gap_report['reconnection']['ordering']
        self.assertLessEqual(decision['selected_objective'][0],decision['baseline_objective'][0])
        self.assertLessEqual(decision['selected_objective'][1],decision['baseline_objective'][1])
        bad=deepcopy(ordered)
        bad.gap_report['reconnection']['ordering']['visits'][1]=decision['visits'][0]
        with self.assertRaises(ValueError):continuous.validate(bad,mesh,region)
        bad=deepcopy(ordered)
        bad.gap_report['reconnection']['ordering']['baseline_objective'][1]+=1
        with self.assertRaises(ValueError):continuous.validate(bad,mesh,region)

    def test_version_binding_and_legacy_policy_remain_explicit(self):
        self.assertTrue(planning_revision.supported_target_algorithms(planning_revision.LEGACY_ALGORITHMS))
        self.assertEqual(planning_revision.ALGORITHMS['routing'],continuous.LAZY_ROUTING)
        candidate,mesh,region,record,domain=self.inputs()
        lazy=continuous.replan(candidate,mesh,region,record,5,domain=domain)
        bad=deepcopy(lazy)
        bad.gap_report['reconnection']['method']=continuous.ROUTING
        with self.assertRaisesRegex(ValueError,'寻路证据'):
            continuous.validate(bad,mesh,region)
        from dataclasses import replace
        mismatched=replace(lazy,planning_context=dict(revision=dict(inputs=dict(
            algorithms=planning_revision.LEGACY_ALGORITHMS))))
        with self.assertRaisesRegex(ValueError,'计算修订'):
            continuous.validate(mismatched,mesh,region)
