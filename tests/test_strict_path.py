"""P4 constructed support, full constraint replay and hostile route records."""
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction as Q
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from test_coverage_workflow import fixture
from femto.geometry.surface_support import MeshSupport
from femto.planning import strict_path as strict, strict_validation, quality_constrained as quality
from femto.planning import surface_workspace, planning_revision
from femto.planning.surface_trajectory import SurfaceTrajectory, ScanSegment, validate_trajectory


class ConstructionTests(unittest.TestCase):
    def test_edge_and_vertex_ids_cannot_use_booleans_as_canonical_integers(self):
        engine=MeshSupport([[0,0,0],[1,0,0],[0,1,0]],[[0,1,2]])
        edge=engine.point(0,[Q(1,2),Q(1,2),Q(0)])
        with self.assertRaises(ValueError):engine.resolve(dict(edge,edge=[False,True]))
        vertex=engine.point(0,[Q(0),Q(1),Q(0)])
        with self.assertRaises(ValueError):engine.resolve(dict(vertex,vertex=True))

    def test_fold_requires_real_edge_split_and_preserves_exact_endpoints(self):
        engine = MeshSupport([[0,0,1],[1,0,1],[0,1,1],[0,-1,2]], [[0,1,2],[1,0,3]])
        params = dict(sample_step_mm=1000, quality_evidence=dict(image_size=[100,100],
            view_reference=dict(undistorted_intrinsic=[[1,0,50],[0,1,50],[0,0,1]])))
        template = dict(xyz_m=[0,0,1],triangle_index=0,geometry_support='measured_mesh',role='scan',row_index=0)
        left = strict.point_from_anchor(engine, engine.point(0,[Q(1,2),Q(1,2)-Q(1,10**8),Q(1,10**8)]),template,params)
        right = strict.point_from_anchor(engine, engine.point(1,[Q(1,2)-Q(1,10**8),Q(1,2),Q(1,10**8)]),template,params)
        self.assertEqual(engine.chord(left[strict.ANCHOR],right[strict.ANCHOR]).status,'rejected')
        points, recipe = strict.construct_edge(engine,left,right,params,strict.SupportBudget())
        self.assertEqual(recipe['method'],'shared_edge_split')
        self.assertEqual(points[0],left);self.assertEqual(points[-1],right)
        self.assertEqual(recipe['crossing']['edge'],[0,1])
        self.assertTrue(all(engine.chord(a[strict.ANCHOR],b[strict.ANCHOR]).status=='passed'
                            for a,b in zip(points,points[1:])))

    def test_support_cannot_borrow_a_triangle_from_another_work_block(self):
        vertices = [[0,0,0],[1,0,0],[0,1,0],[2,0,0],[2,1,0],[3,0,0]]
        faces = [[0,1,2],[1,3,4],[3,5,4]]
        unscoped = MeshSupport(vertices,faces)
        left=unscoped.point(0,[Q(1,2),Q(1,2),Q(0)])
        right=unscoped.point(2,[Q(1,2),Q(1,2),Q(0)])
        self.assertEqual(unscoped.chord(left,right).status,'passed')
        scoped=MeshSupport(vertices,faces,groups=np.array([0,1,0]))
        self.assertEqual(scoped.chord(left,right).status,'rejected')
        crossing=MeshSupport(vertices,faces,groups=np.array([0,0,1]))
        self.assertEqual(crossing.chord(left,right).reason,'different_work_blocks')


class StrictRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        cls.model,cls.region,cls.plan,cls.certificate,cls.ledger=fixture(Path(cls.temp.name)/'source')
        cls.base=SurfaceTrajectory.from_dict(cls.plan['surface_trajectory'])
        cls.mesh=surface_workspace.canonical_region_mesh(cls.model.store.mesh,cls.region)
        cls.result=strict.reconstruct(cls.base,cls.mesh,cls.region)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_full_replay_preserves_original_route_and_target(self):
        validate_trajectory(self.result,mesh=self.mesh,region=self.region)
        self.assertEqual(self.base.to_dict(),self.plan['surface_trajectory'])
        self.assertEqual(self.result.gap_report['surface_support']['base_trajectory'],self.base.to_dict())
        self.assertEqual(self.result.planning_context['revision']['target_id'],self.base.planning_context['revision']['target_id'])
        self.assertNotEqual(self.result.planning_context['revision']['sha256'],self.base.planning_context['revision']['sha256'])
        self.assertTrue(planning_revision.supported_target_algorithms(self.result.planning_context['revision']['inputs']['algorithms']))
        self.assertTrue(all(strict.ANCHOR in p for s in self.result.ordered_items if isinstance(s,ScanSegment) for p in s.points))

    def test_zero_work_budget_blocks_without_mutating_base(self):
        before=self.base.to_dict()
        with self.assertRaises(quality.PlanningConstraintBlocked) as caught:
            strict.reconstruct(self.base,self.mesh,self.region,budget=strict.SupportBudget(max_face_checks=0))
        self.assertEqual(caught.exception.report['stop_reason'],'support_face_budget')
        self.assertEqual(self.base.to_dict(),before)

    def test_cancel_propagates_and_does_not_become_numerical_unknown(self):
        class Cancelled(Exception):pass
        def check():raise Cancelled()
        with self.assertRaises(Cancelled):strict.reconstruct(self.base,self.mesh,self.region,check=check)

    def test_projection_runs_all_quality_checks_and_cannot_override_refusal(self):
        called=[]
        def refuse(segments,*args,**kwargs):
            called.append(segments)
            self.assertIsInstance(kwargs['support_checker'],strict.Checks)
            self.assertTrue(all(strict.ANCHOR in p for s in segments for p in s.points))
            count=sum(len(s.points)-1 for s in segments)
            kwargs['support_checker'].decisions=[dict(status='not_checked',reason='earlier_quality_constraint')]*count
            return np.zeros(count,bool),['mesh_boundary_clearance']*count,np.zeros(count)
        with patch.object(quality,'edge_checks',side_effect=refuse):
            with self.assertRaises(quality.PlanningConstraintBlocked) as caught:
                strict.reconstruct(self.base,self.mesh,self.region)
        self.assertEqual(len(called),1)
        self.assertGreater(caught.exception.report['edge_outcomes']['quality_rejected'],0)

    def test_changed_anchor_and_resealed_report_cannot_create_support(self):
        report=deepcopy(self.result.gap_report['surface_support'])
        point=report['records'][0]['points'][0]
        point[strict.ANCHOR]['barycentric']=[[1,1],[0,1],[0,1]]
        # Rebuild all visible statistics and route hashes around the forgery.
        forged=strict.route_from_report(self.base,report)
        with self.assertRaises(ValueError):validate_trajectory(forged,mesh=self.mesh,region=self.region)

    def test_missing_proof_cannot_be_replaced_by_an_accepted_label(self):
        report=deepcopy(self.result.gap_report['surface_support'])
        report['records'][0]['support_checks'][0]['status']='unconfirmed'
        with self.assertRaisesRegex(ValueError,'未经证明'):strict.route_from_report(self.base,report)

    def test_report_cannot_change_work_block_or_original_edge_order(self):
        for field in ('source_edge','block_index'):
            report=deepcopy(self.result.gap_report['surface_support'])
            report['records'][0][field]+=1
            with self.assertRaises((ValueError,IndexError)):
                forged=strict.route_from_report(self.base,report)
                validate_trajectory(forged,mesh=self.mesh,region=self.region)

    def test_malformed_rational_and_false_metrics_fail_structural_validation(self):
        changed=deepcopy(self.result)
        changed.ordered_items[0].points[0][strict.ANCHOR]['barycentric'][0]=[1,0]
        with self.assertRaises(ValueError):strict_validation.validate(changed)

    def test_resealed_revision_cannot_substitute_a_different_base_binding(self):
        context=deepcopy(self.result.planning_context)
        inputs=context['revision']['inputs']
        inputs['algorithms']['support']['base_route_sha256']='0'*64
        context['revision']=planning_revision.create(inputs)
        context['target']['computation_sha256']=context['revision']['sha256']
        context['target']['sha256']=planning_revision.digest({k:v for k,v in context['target'].items() if k!='sha256'})
        forged=replace(self.result,planning_context=context)
        with self.assertRaisesRegex(ValueError,'原目标或计算修订'):validate_trajectory(forged)
        with self.assertRaises(ValueError):validate_trajectory(replace(self.result,planning_context=None))
        report=deepcopy(self.result.gap_report['surface_support']);report['edge_outcomes']['accepted']+=1
        changed=replace(self.result,gap_report=dict(self.result.gap_report,surface_support=report))
        with self.assertRaises(ValueError):strict_validation.validate(changed)

    def test_legacy_v3_remains_valid_with_unchanged_algorithm(self):
        validate_trajectory(self.base,mesh=self.mesh,region=self.region)
        self.assertEqual(self.base.planner['algorithm_version'],'quality-constrained-surface-v3')
        self.assertNotIn('surface_support',self.base.planner['parameters'])

    def test_standalone_mesh_validation_uses_bound_selection_for_block_limits(self):
        validate_trajectory(self.result,mesh=self.mesh)
