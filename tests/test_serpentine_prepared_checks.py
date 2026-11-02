"""Prepared graph inputs retain complete predicates, batch budgets and replay."""
from copy import deepcopy
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np
import open3d as o3d
import trimesh

from test_fixed_rgb_experience import make_model, K
from test_mesh_slice_planner import grid_mesh
from femto.annotation.fixed_rgb_geometry import GeometryBudget, GeometryTaskCancelled
from femto.planning import quality_constrained as quality, surface_workspace, planning_state
from femto.planning.surface_trajectory import ScanSegment
from femto.planning.treatment_head import head_parameters


class PreparedQualityChecksTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.model = make_model(self.temp.name)
        self.model.view_reference['undistorted_intrinsic'] = K.tolist()
        self.mask = self.model.projection.valid.copy()
        self.mesh = trimesh.Trimesh(vertices=np.asarray(self.model.mesh.vertices),
                                   faces=np.asarray(self.model.mesh.triangles), process=False)
        self.allowed = np.ones(len(self.mesh.faces), bool)

    def record(self):
        region = self.model.region([[10,10],[90,10],[90,90],[10,90]], 'synthetic', '',
                                   selection=dict(mask=self.mask, method='work_only'))
        return quality.evidence(self.model, region)

    def segment(self, a=(-.026,.05,.5), b=(-.024,.05,.5)):
        points = [dict(xyz_m=list(p), triangle_index=0 if p[0] >= p[1] else 1) for p in (a,b)]
        return ScanSegment(0, 0, points, 0)

    def assert_same(self, actual, expected):
        np.testing.assert_array_equal(actual[0], expected[0])
        self.assertEqual(actual[1], expected[1])
        np.testing.assert_array_equal(actual[2], expected[2])

    def test_repeated_batches_match_fresh_checks_for_width_holes_and_excluded_faces(self):
        self.mask[48:53,48:53] = False
        record = self.record()
        rng = np.random.default_rng(4513)
        starts = np.c_[rng.uniform(-.19,.19,(70,2)), np.full(70,.5)]
        ends = starts.copy(); ends[:,:2] += rng.uniform(-.008,.008,(70,2))
        segments = [self.segment(a,b) for a,b in zip(starts,ends)]
        segments += [self.segment((-.10,0,.5),(.10,0,.5)), self.segment()]
        for width in (1.,5.,30.):
            for allowed in (self.allowed, np.array([False,True])):
                prepared = quality._prepare_edge_checks(self.mesh, record, width, lambda:None, allowed)
                decisions = []
                for start in range(0,len(segments),7):
                    batch = segments[start:start+7]
                    expected = quality.edge_checks(batch,self.mesh,record,width,allowed_faces=allowed)
                    self.assert_same(prepared(batch), expected)
                    self.assert_same(prepared(batch), expected)
                    decisions.extend(expected[0])
                self.assertTrue(any(decisions))
                self.assertFalse(all(decisions))

    def test_prepared_inputs_are_isolated_and_new_requests_validate_their_own_evidence(self):
        record = self.record();segments = [self.segment()]
        prepared = quality._prepare_edge_checks(self.mesh,record,1.,lambda:None,self.allowed)
        expected = prepared(segments)
        self.assertTrue(expected[0].all())
        self.mesh.vertices[:] += 1.
        self.allowed[:] = False
        record['sha256'] = '0'*64
        self.assert_same(prepared(segments), expected)
        with self.assertRaisesRegex(ValueError,'质量约束证据'):
            quality._prepare_edge_checks(self.mesh,record,1.,lambda:None,self.allowed)
        with self.assertRaisesRegex(ValueError,'质量约束证据'):
            quality.edge_checks(segments,self.mesh,record,1.)

    def test_residual_budget_is_per_batch_and_cannot_be_bypassed_within_a_batch(self):
        prepared = quality._prepare_edge_checks(self.mesh,self.record(),1.,lambda:None,self.allowed)
        probe = quality.ReferenceEnvelope.probe
        def expensive_probe(reference,*args,**kwargs):
            # Charge work without allocating a giant synthetic reference image.
            reference.sampled_pixels += 60_000_000
            return probe(reference,*args,**kwargs)
        with patch.object(quality.ReferenceEnvelope,'probe',expensive_probe):
            self.assertTrue(prepared([self.segment()])[0].all())
            self.assertTrue(prepared([self.segment()])[0].all())
            with self.assertRaisesRegex(ValueError,'局部残差邻域检查超过预算'):
                prepared([self.segment(),self.segment()])
            self.assertTrue(prepared([self.segment()])[0].all())

    def test_edge_count_budget_and_cancellation_still_apply_to_every_call(self):
        cancel = threading.Event()
        prepared = quality._prepare_edge_checks(self.mesh,self.record(),1.,GeometryBudget(cancel).check,self.allowed)
        with self.assertRaisesRegex(ValueError,'质量约束路线超过检查预算'):
            prepared([self.segment()] * 200001)
        cancel.set()
        with self.assertRaises(GeometryTaskCancelled):prepared([self.segment()])
        with self.assertRaises(GeometryTaskCancelled):
            quality._prepare_edge_checks(self.mesh,self.record(),1.,GeometryBudget(cancel).check,self.allowed)

    def test_invalid_width_and_domain_do_not_reuse_previous_acceptance(self):
        record = self.record()
        for width in (True,0.,101.,float('nan')):
            with self.subTest(width=width), self.assertRaises(ValueError):
                quality._prepare_edge_checks(self.mesh,record,width,lambda:None,self.allowed)
        for allowed in (np.zeros(2,bool), np.ones(3,bool)):
            prepared = quality._prepare_edge_checks(self.mesh,record,1.,lambda:None,allowed)
            with self.assertRaisesRegex(ValueError,'有效区域没有可规划网格面'):
                prepared([self.segment()])


class PreparedSerpentineTests(unittest.TestCase):
    def test_disjoint_selection_and_hole_keep_identical_motion_and_rejection_reports(self):
        with tempfile.TemporaryDirectory() as folder:
            mesh = grid_mesh(width=.12,height=.08,cell=.005)
            model = make_model(folder,o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),
                o3d.utility.Vector3iVector(mesh.faces)))
            model.view_reference['undistorted_intrinsic'] = K.tolist()
            model.store.origin_reference = dict(kind='initial_color_camera',provenance='synthetic_test')
            mask = model.projection.valid.copy();mask[:,48:53] = False;mask[46:49,41:44] = False
            region = model.region([[30,30],[70,30],[70,70],[30,70]],'synthetic','',
                                  selection=dict(mask=mask,method='work_only'))
            original = deepcopy(region)
            head = head_parameters('direct_row_spacing',row_spacing_mm=7)
            # Reference path performs fresh preparation on every original batch.
            def fresh(mesh,record,width,check,allowed):
                return lambda segments:quality.edge_checks(segments,mesh,record,width,check,allowed_faces=allowed)
            with patch.object(quality,'_prepare_edge_checks',fresh):
                before = surface_workspace.generate(model,region,head)
            after = surface_workspace.generate(model,region,head)
            for key in ('ordered_items','statistics','gap_report','spacing_report','row_report','blocks','planner'):
                self.assertEqual(before['surface_trajectory'][key],after['surface_trajectory'][key],key)
            self.assertEqual(before['segments'],after['segments'])
            self.assertEqual(before['chain_breaks'],after['chain_breaks'])
            self.assertEqual(region,original)
            self.assertTrue(any(gap['inter_block'] for gap in after['chain_breaks']))
            self.assertTrue(all(not gap['transition_planned'] for gap in after['chain_breaks']))
            planning_state.validate(after,model.mesh,region)


if __name__=='__main__':unittest.main()
