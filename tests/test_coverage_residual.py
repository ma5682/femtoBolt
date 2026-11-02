"""V4 residual classifications never inherit old claims or spend off-budget."""
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction as Q
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from weakref import ref

import numpy as np

from femto.geometry import exact_surface as exact
from femto.planning import coverage_certificate as coverage, coverage_partition, footprint_model
from femto.planning import coverage_reproof, coverage_residual, coverage_refinement as refinement
from femto.planning import refinement_candidates_v2, refinement_archive, coverage_workflow
from femto.planning.coverage_domain import CoverageDomain
from femto.planning.planning_revision import digest
from femto.planning.surface_trajectory import SurfaceTrajectory, validate_trajectory
from test_coverage_certificate import triangle_domain, seed, reseal


class ResidualCoverageTests(unittest.TestCase):
    def setUp(self):
        self.domain=triangle_domain();self.records=[seed()]
        self.model=footprint_model.nominal(125.)
        self.budget=coverage.CoverageBudget(max_depth=3)
        self.old=coverage.evaluate(self.domain,self.records,self.model,budget=self.budget,algorithm=coverage.CURRENT_ALGORITHM)

    def evaluate(self,*,records=None,guide=None,budget=None,domain=None,frontier=None):
        return coverage.evaluate(self.domain if domain is None else domain,self.records if records is None else records,self.model,
            budget=self.budget if budget is None else budget,guide=self.old if guide is None else guide,
            algorithm=coverage.RESIDUAL_ALGORITHM,_frontier=frontier)

    def prepared(self,records=None):
        engine=coverage_residual.ResidualEngine(self.domain,self.records if records is None else records,self.model,
            self.budget,lambda:None,None,guide=self.old)
        engine.seed();self.assertTrue(engine._prepare());engine._pixel_bounds();self.assertTrue(engine._reprove())
        engine._outside_faces();engine._outside_pixels()
        return engine

    def test_partial_root_exclusions_keep_every_current_confirmation(self):
        self.assertIn(1,self.old.arrays['status']);self.assertIn(3,self.old.arrays['status'])
        engine=self.prepared();before,_=engine.leaf_arrays();engine._outside_leaves();after,_=engine.leaf_arrays()
        self.assertGreater(engine.meter.counts['residual_outside_leaves'],0)
        np.testing.assert_array_equal(before['status']==1,after['status']==1)
        np.testing.assert_array_equal(before['witness'],after['witness'])
        self.assertTrue(np.all(after['root']==0))
        self.assertGreater(np.count_nonzero(after['status']==3),np.count_nonzero(before['status']==3))

    def test_full_new_ball_gain_preserves_the_original_set_and_replays(self):
        records=self.records+[seed((.125,0,0))]
        actual=self.evaluate(records=records)
        compared=coverage_partition.compare(self.old,actual,self.domain.areas)
        self.assertEqual(compared['lost_weight_units'],0);self.assertGreater(compared['gained_weight_units'],0)
        self.assertEqual(actual.metadata['algorithm'],coverage.RESIDUAL_ALGORITHM)
        coverage.validate(actual,self.domain,records,self.model,guide=self.old)

    def test_new_ball_over_old_outside_leaf_blocks_inherited_exclusion(self):
        number=int(np.flatnonzero(self.old.arrays['status']==3)[-1])
        bary=exact.descend(exact.IDENTITY,int(self.old.arrays['depth'][number]),int(self.old.arrays['code'][number]))
        triangle=[exact.xyz(p,self.domain.triangle(0)) for p in bary]
        center=[float(sum(p[a] for p in triangle)/3) for a in range(3)]
        engine=self.prepared(self.records+[seed(center)])
        engine._outside_leaves()
        self.assertEqual(engine.nodes[number].status,7)

    def test_rehashed_old_outside_flags_are_never_evidence(self):
        arrays={k:v.copy() for k,v in self.old.arrays.items()}
        arrays['status'][arrays['status']!=1]=3
        false_hint=reseal(self.old,arrays=arrays)
        expected=self.evaluate();actual=self.evaluate(guide=false_hint)
        for key in actual.arrays:np.testing.assert_array_equal(expected.arrays[key],actual.arrays[key])

    def test_all_outside_leaves_match_independent_exact_triangle_distance(self):
        records=self.records+[seed((.125,0,0))]
        actual=self.evaluate(records=records);a=actual.arrays
        balls=[exact.anchor_ball(r['xyz_m'],self.domain.triangle(0),.0625) for r in records]
        count=0
        for number in np.flatnonzero(a['status']==3):
            bary=exact.descend(exact.IDENTITY,int(a['depth'][number]),int(a['code'][number]))
            triangle=tuple(exact.xyz(p,self.domain.triangle(0)) for p in bary)
            for _,center,radius in balls:self.assertGreater(coverage._distance2(center,triangle),Q(radius)**2)
            count+=1
        self.assertGreater(count,0)

    def test_reproved_and_new_witnesses_match_fraction_containment_and_shared_subedges(self):
        from test_exact_recomputation import edge_reference,orientation
        records=self.records+[seed((.125,0,0))];actual=self.evaluate(records=records);a=actual.arrays
        cells=[exact.descend(exact.IDENTITY,int(d),int(c)) for d,c in zip(a['depth'],a['code'])]
        balls=[exact.anchor_ball(r['xyz_m'],self.domain.triangle(0),.0625) for r in records]
        for leaf,seed_id,parent in zip(a['proof_leaf'],a['proof_seed'],a['proof_parent']):
            bary,center,radius=balls[int(a['seed_input_index'][seed_id])]
            for point in cells[leaf]:
                xyz=exact.xyz(point,self.domain.triangle(0))
                self.assertLessEqual(sum((x-y)**2 for x,y in zip(xyz,center)),Q(radius)**2)
            if parent<0:
                signs=[orientation(x,y,bary) for x,y in zip(cells[leaf],cells[leaf][1:]+cells[leaf][:1])]
                self.assertTrue(all(v>=0 for v in signs) or all(v<=0 for v in signs))
            else:
                edges=[]
                for triangle in (cells[leaf],cells[a['proof_leaf'][parent]]):
                    edges.append([edge_reference(0,(0,1,2),x,y) for x,y in zip(triangle,triangle[1:]+triangle[:1])])
                self.assertTrue(any(x and y and x[0]==y[0] and min(x[2],y[2])>max(x[1],y[1]) for x in edges[0] for y in edges[1]))

    def test_positive_common_vertex_shortcut_requires_same_direction_and_nonzero_overlap(self):
        from test_exact_recomputation import edge_reference
        a=((1,0,0),(0,1,0),(0,0,1));p=a[0]
        for b,expected in ((((1,0,0),(1,1,0),(2,1,1)),True),
                           (((1,0,0),(2,-1,0),(2,0,-1)),False),
                           (((1,0,0),(2,-1,1),(2,1,-1)),False)):
            self.assertEqual(coverage_residual._positive_shared_vertex(a,b,p),expected)
            if expected:
                edges=[]
                for triangle in (a,b):
                    points=tuple(tuple(Q(x,sum(point)) for x in point) for point in triangle)
                    edges.append([edge_reference(0,(0,1,2),x,y) for x,y in zip(points,points[1:]+points[:1])])
                self.assertTrue(any(x and y and x[0]==y[0] and min(x[2],y[2])>max(x[1],y[1]) for x in edges[0] for y in edges[1]))

    def test_rehashed_parent_cannot_connect_separate_source_layers(self):
        domain=CoverageDomain.mesh([[0,0,0],[.01,0,0],[0,.01,0],
                                   [0,0,.001],[.01,0,.001],[0,.01,.001]],[[0,1,2],[3,4,5]])
        old=coverage.evaluate(domain,self.records,self.model,algorithm=coverage.CURRENT_ALGORITHM)
        arrays={k:v.copy() for k,v in old.arrays.items()}
        arrays['status'][1]=1;arrays['witness'][1]=1
        for key,value in (('proof_leaf',1),('proof_seed',0),('proof_parent',0)):
            arrays[key]=np.append(arrays[key],np.int64(value))
        actual=self.evaluate(domain=domain,guide=reseal(old,arrays=arrays))
        self.assertEqual(actual.arrays['status'][1],7)
        self.assertEqual(actual.metadata['metrics']['hint_connectivity_failures'],1)

    def test_interrupted_pair_reservation_installs_no_partial_batch(self):
        engine=self.prepared(self.records+[seed((.125,0,0))])
        before,_=engine.leaf_arrays()
        engine.meter.budget=replace(self.budget,max_comparisons=engine.meter.counts['comparisons']+1)
        with self.assertRaises(coverage._BudgetStop):engine._outside_leaves()
        self.assertEqual(engine.meter.counts['comparisons'],engine.meter.budget.max_comparisons)
        self.assertEqual(engine.meter.counts['residual_batches_completed'],0)
        after,_=engine.leaf_arrays()
        for key in ('root','depth','code','status','witness'):np.testing.assert_array_equal(before[key],after[key])

    def test_partial_residual_time_frontiers_replay_exactly(self):
        stages=[];method=coverage_residual.ResidualEngine._outside_leaves
        def observe(engine):
            touch=engine.meter.touch
            def track(kind=None):
                stages.append(engine.meter.steps)
                return touch(kind)
            with patch.object(engine.meter,'touch',track):return method(engine)
        with patch.object(coverage_residual.ResidualEngine,'_outside_leaves',observe):self.evaluate()
        self.assertGreater(len(stages),2)
        for stop in sorted(set((stages[0],stages[len(stages)//2],stages[-1]))):
            with self.subTest(stop=stop):
                actual=self.evaluate(frontier=dict(stop_reason='time_budget',steps=stop))
                self.assertEqual(actual.metadata['metrics']['stop_reason'],'time_budget')
                self.assertEqual(coverage_partition.compare(self.old,actual,self.domain.areas)['lost_weight_units'],0)
                coverage_partition.intervals(actual,len(self.domain.areas))
                coverage.validate(actual,self.domain,self.records,self.model,guide=self.old)

    def test_all_budget_caps_preserve_target_weights_and_replay(self):
        for limits in (dict(seconds=0),dict(max_comparisons=0),dict(max_adjacencies=0),dict(max_proofs=0),
                       dict(max_seeds=0),dict(max_added_cells=0)):
            with self.subTest(limits=limits):
                actual=self.evaluate(budget=replace(self.budget,**limits))
                coverage_partition.intervals(actual,len(self.domain.areas))
                self.assertEqual(actual.summary['located_nominal_area_m2'],self.old.summary['located_nominal_area_m2'])
                self.assertFalse(actual.summary['original_target_coverage_complete'])
                coverage.validate(actual,self.domain,self.records,self.model,guide=self.old)

    def test_cancellation_during_residuals_propagates_without_partial_claims(self):
        engine=self.prepared();before,_=engine.leaf_arrays()
        def cancel():raise RuntimeError('cancelled by test')
        engine.meter.check=cancel
        with self.assertRaisesRegex(RuntimeError,'cancelled'):
            engine.meter.steps=((engine.meter.steps+31)//32)*32
            engine._outside_leaves()
        after,_=engine.leaf_arrays()
        for key in ('status','witness'):np.testing.assert_array_equal(before[key],after[key])

    def test_interrupted_integer_link_keeps_the_current_reproved_witness(self):
        engine=self.prepared();number=int(engine.old_confirmed[0]);witness=engine.nodes[number].witness
        with patch.object(engine,'_link',side_effect=coverage._BudgetStop),self.assertRaises(coverage._BudgetStop):
            engine._visit(number)
        arrays,remap=engine.leaf_arrays()
        self.assertEqual(arrays['status'][remap[number]],1)
        self.assertEqual(arrays['witness'][remap[number]],witness)

    def test_unconstructible_current_root_cannot_gain_an_exclusion(self):
        engine=self.prepared();before,_=engine.leaf_arrays()
        with patch.object(engine,'_integer_root_cache',return_value=None):engine._outside_leaves()
        after,_=engine.leaf_arrays()
        np.testing.assert_array_equal(before['status'],after['status'])
        self.assertGreater(engine.meter.counts['residual_unconstructed_roots'],0)

    def test_lazy_and_eager_partition_serialization_agree_after_exclusions(self):
        engine=self.prepared();engine._outside_leaves()
        actual,remap=engine.leaf_arrays();expected,mapping=coverage._Engine.leaf_arrays(engine)
        for key in actual:np.testing.assert_array_equal(actual[key],expected[key])
        for key,value in mapping.items():self.assertEqual(remap[key],value)

    def test_no_engine_reference_cycle_is_introduced(self):
        engine=self.prepared();engine._outside_leaves();weak=ref(engine);del engine
        self.assertIsNone(weak())

    def test_policy_and_algorithm_versions_cannot_be_cross_wired(self):
        actual=self.evaluate();metadata=deepcopy(actual.metadata)
        metadata['algorithm']=metadata['inputs']['algorithm']=coverage.REPROOF_ALGORITHM
        metadata['coverage_computation_sha256']=digest(metadata['inputs'])
        forged=reseal(actual,metadata=metadata)
        with self.assertRaisesRegex(ValueError,'策略版本'):coverage.validate_metadata(forged.metadata)
        for change in ({'residual_leaf_batch':128.0},{'residual_root_batch':1024},{'residual_install':'copy_old_state'}):
            with self.subTest(change=change),self.assertRaises(ValueError):
                coverage.reproof_algorithm_for_policy(dict(coverage_residual.POLICY,**change))

    def test_new_model_or_target_cannot_reuse_the_guide(self):
        with self.assertRaisesRegex(ValueError,'作用模型'):
            coverage.evaluate(self.domain,self.records,footprint_model.nominal(250.),
                algorithm=coverage.RESIDUAL_ALGORITHM,guide=self.old)
        with self.assertRaisesRegex(ValueError,'原目标'):self.evaluate(domain=triangle_domain(.25))

    def test_v3_evaluation_is_unchanged_after_running_v4(self):
        def old():return coverage.evaluate(self.domain,self.records,self.model,budget=self.budget,
            algorithm=coverage.REPROOF_ALGORITHM,guide=self.old)
        before=old();self.evaluate();after=old()
        self.assertEqual(before.metadata,after.metadata)
        for key in before.arrays:np.testing.assert_array_equal(before.arrays[key],after.arrays[key])

    def test_current_pixel_candidates_and_residual_boxes_match_exact_truth(self):
        import open3d as o3d
        import test_target_surface
        from test_mesh_slice_planner import grid_mesh
        from femto.planning import target_surface,surface_workspace
        from femto.planning.effective_domain import build
        fixture=test_target_surface.TargetSurfaceTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        mesh=grid_mesh(width=.03,height=.03,cell=.005)
        native=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),o3d.utility.Vector3iVector(mesh.faces))
        mask=np.zeros((100,100),bool);mask[47:54,47:54]=True
        source,region=fixture.fixture(mesh=native,mask=mask);target,revision,evidence=fixture.build(source,region)
        quality=build(surface_workspace.canonical_region_mesh(source.mesh,region),region,evidence,5)
        target=target_surface.with_quality(target,revision,region,evidence,quality)
        domain=CoverageDomain.target(target,revision,evidence=evidence,effective_domain=quality)
        def endpoint(root):
            face=int(domain.source_faces[root]);tri=[exact.xyz(p,domain.triangle(face)) for p in domain.root(int(root))]
            return seed([float(sum(p[a] for p in tri)/3) for a in range(3)],face)
        eligible=np.flatnonzero(domain.quality==0);records=[endpoint(eligible[0])]
        model=footprint_model.nominal(5.);budget=coverage.CoverageBudget(max_depth=2)
        old=coverage.evaluate(domain,records,model,budget=budget,algorithm=coverage.CURRENT_ALGORITHM)
        records.append(endpoint(eligible[-1]))
        actual=coverage.evaluate(domain,records,model,budget=budget,algorithm=coverage.RESIDUAL_ALGORITHM,guide=old)
        self.assertGreater(actual.metadata['metrics']['residual_current_root_candidates'],0)
        self.assertGreater(actual.metadata['metrics']['residual_leaves_rechecked'],0)
        self.assertEqual(coverage_partition.compare(old,actual,domain.areas)['lost_weight_units'],0)
        balls=[exact.anchor_ball(r['xyz_m'],domain.triangle(r['source_face']),.0025) for r in records]
        a=actual.arrays
        for number in np.flatnonzero(a['status']==3):
            root=int(a['root'][number]);face=int(domain.source_faces[root])
            bary=exact.descend(domain.root(root),int(a['depth'][number]),int(a['code'][number]))
            triangle=tuple(exact.xyz(p,domain.triangle(face)) for p in bary)
            for ball in balls:
                if ball is not None:self.assertGreater(coverage._distance2(ball[1],triangle),Q(ball[2])**2)
        coverage.validate(actual,domain,records,model,guide=old)


class ResidualWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_coverage_workflow import fixture
        cls.temp=tempfile.TemporaryDirectory();cls.addClassCleanup(cls.temp.cleanup)
        cls.model,cls.region,cls.plan,cls.certificate,cls.ledger=fixture(Path(cls.temp.name)/'source')
        cls.base=SurfaceTrajectory.from_dict(cls.plan['surface_trajectory'])
        cls.mesh=refinement_archive.local_mesh(cls.ledger)
        cls.budget=refinement.RefinementBudget(max_rounds=1,candidates_per_round=1)
        cls.result,cls.target,cls.proof=refinement.refine(cls.base,cls.ledger,cls.certificate,cls.mesh,cls.region,
            budget=cls.budget,candidate_policy=deepcopy(refinement_candidates_v2.POLICY),coverage_policy=deepcopy(coverage_residual.POLICY))

    def test_explicit_schema_preserves_route_and_rejects_mixed_policy(self):
        for policy,schema in ((coverage_reproof.POLICY,refinement.REPROOF_SCHEMA),(coverage_residual.POLICY,refinement.RESIDUAL_SCHEMA)):
            values=refinement.algorithms_for(self.base,self.certificate,self.budget,refinement_candidates_v2.POLICY,policy)
            self.assertEqual(values['route'],refinement.ALGORITHM)
            self.assertEqual(values['refinement']['schema'],schema);self.assertTrue(refinement.checked_algorithms(values))
            values['refinement']['schema']=refinement.RESIDUAL_SCHEMA if schema==refinement.REPROOF_SCHEMA else refinement.REPROOF_SCHEMA
            self.assertFalse(refinement.checked_algorithms(values))

    def test_accepted_route_retains_arrays_and_positive_new_witness(self):
        report=self.result.gap_report['refinement'];self.assertEqual(report['schema'],refinement.RESIDUAL_SCHEMA)
        self.assertTrue(report['rounds'][0]['accepted'],report['rounds'][0]['status'])
        self.assertEqual(report['rounds'][0]['comparison']['lost_weight_units'],0)
        self.assertGreater(report['rounds'][0]['comparison']['new_scan_witness_gain_m2'],0)
        for key in self.ledger.arrays:np.testing.assert_array_equal(self.ledger.arrays[key],self.target.arrays[key])
        validate_trajectory(self.result,mesh=self.mesh,region=self.region)

    def test_complete_round_replay_uses_the_saved_v4_policy(self):
        refinement_archive.replay(self.proof,self.result,self.target)

    def test_verified_ordinary_v4_summary_does_not_repeat_route_validation_on_ui_thread(self):
        with patch.object(SurfaceTrajectory,'from_dict',side_effect=AssertionError('label triggered route replay')):
            text=refinement.summary_text(self.result.to_dict())
        self.assertIn('补充规划',text);self.assertIn('新增',text)

    def test_v3_route_and_certificate_replay_remain_available(self):
        trajectory,target,proof=refinement.refine(self.base,self.ledger,self.certificate,self.mesh,self.region,
            budget=self.budget,candidate_policy=deepcopy(refinement_candidates_v2.POLICY),coverage_policy=deepcopy(coverage_reproof.POLICY))
        self.assertEqual(trajectory.gap_report['refinement']['schema'],refinement.REPROOF_SCHEMA)
        refinement_archive.replay(proof,trajectory,target)


if __name__=='__main__':unittest.main()
