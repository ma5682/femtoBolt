import unittest
from types import SimpleNamespace
from copy import deepcopy
import numpy as np

from femto.planning import coverage_certificate as coverage
from femto.planning.coverage_grouped import GroupedRecalculationEngine, identical_boxes
from femto.planning.coverage_residual import ResidualEngine, _ResidualPartition


class GroupedBoundsTests(unittest.TestCase):
    def engine(self, cls, limit=1000000, check=lambda:None):
        engine=object.__new__(cls)
        points=np.repeat(np.array([[0.,0.,0.],[1.,0.,0.],[2.,0.,0.],[4.,0.,0.]]),5,axis=0)
        count=len(points)
        engine.domain=SimpleNamespace(ledger=True,areas=np.ones(count))
        engine.prism_bounds=(points.copy(),points.copy())
        engine.nodes=_ResidualPartition(np.arange(count),np.zeros(count,'u1'),np.zeros(count,'u4'),np.zeros(count,'u1'))
        engine.meter=coverage._Meter(coverage.CoverageBudget(seconds=30,max_comparisons=limit),check,None)
        engine.meter.counts.update(root_group_ball_comparisons=0,root_group_candidate_pairs_pruned=0,pixel_ball_comparisons=0,pixel_bounds_duplicates_removed=0)
        engine._box_groups=lambda roots:{'all':list(roots)}
        engine._candidates_in_box=lambda key:[0,1]
        engine.centers_low=engine.centers_high=np.array([[0.,0.,0.],[1.,0.,0.]])
        engine.seeds=[dict(inner=1.),dict(inner=0.5)]
        engine._root_candidates={}
        return engine

    def test_exact_shared_bounds_preserve_all_candidate_lists_and_boundary_contact(self):
        old=self.engine(ResidualEngine);new=self.engine(GroupedRecalculationEngine)
        old._outside_pixels();new._outside_pixels()
        np.testing.assert_array_equal(old.nodes.root_status,new.nodes.root_status)
        self.assertEqual(old._root_candidates,new._root_candidates)
        self.assertTrue(np.all(new.nodes.root_status[:10]==7))
        self.assertTrue(np.all(new.nodes.root_status[10:]==3))
        self.assertEqual(new.meter.counts['pixel_bounds_duplicates_removed'],16)
        self.assertLess(new.meter.counts['comparisons'],old.meter.counts['comparisons'])

    def test_randomized_outward_boxes_match_full_pair_reference(self):
        rng=np.random.default_rng(20261016)
        for _ in range(12):
            low=np.repeat(rng.uniform(-3,3,(5,3)),4,axis=0)
            high=low+np.repeat(rng.uniform(0,1,(5,3)),4,axis=0)
            old=self.engine(ResidualEngine);new=self.engine(GroupedRecalculationEngine)
            for engine in (old,new):engine.prism_bounds=(low,high)
            old._outside_pixels();new._outside_pixels()
            np.testing.assert_array_equal(old.nodes.root_status,new.nodes.root_status)
            self.assertEqual(old._root_candidates,new._root_candidates)

    def test_nearby_bounds_and_opposite_zero_bits_are_not_merged(self):
        low=np.zeros((4,3));low[1,0]=np.nextafter(0.,1.);low[2,0]=-0.
        representatives,members=identical_boxes([0,1,2,3],(low,low),lambda:None)
        self.assertEqual(representatives,[0,1,2]);self.assertEqual(members[0],[0,3])

    def test_partial_batch_never_installs_an_unchecked_exclusion(self):
        engine=self.engine(GroupedRecalculationEngine,limit=3)
        with self.assertRaises(coverage._BudgetStop):engine._outside_pixels()
        self.assertTrue(np.all(engine.nodes.root_status==7));self.assertFalse(engine._root_candidates)
        self.assertEqual(engine.meter.counts['comparisons'],3)

    def test_cancel_propagates_without_installing_results(self):
        def cancel():raise RuntimeError('cancelled')
        engine=self.engine(GroupedRecalculationEngine,check=cancel)
        with self.assertRaisesRegex(RuntimeError,'cancelled'):engine._outside_pixels()
        self.assertTrue(np.all(engine.nodes.root_status==7))

    def test_current_confirmations_and_quality_exclusions_are_preserved(self):
        engine=self.engine(GroupedRecalculationEngine)
        engine.nodes[15].status=1;engine.nodes[15].witness=0
        engine.nodes.root_status[16]=2
        engine._outside_pixels()
        self.assertEqual(engine.nodes[15].status,1);self.assertEqual(engine.nodes[16].status,2)

class GroupedWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import test_paged_target_workflow as paged
        paged.PagedTargetWorkflowTests.setUpClass()
        cls.addClassCleanup(paged.PagedTargetWorkflowTests.tearDownClass)
        case=paged.PagedTargetWorkflowTests()
        case.setUp();cls.addClassCleanup(case.doCleanups)
        cls.case=case;cls.plan,cls.ledger=case.continued()
        from femto.planning import coverage_workflow as workflow
        from femto.planning.surface_trajectory import SurfaceTrajectory
        cls.domain,cls.seeds,cls.model,cls.route=workflow.bundle(SurfaceTrajectory.from_dict(cls.plan['surface_trajectory']),cls.ledger)
        cls.first=coverage.evaluate(cls.domain,cls.seeds,cls.model,route_binding=cls.route,
            algorithm=coverage.PAGED_ALGORITHM,budget=coverage.CoverageBudget(max_depth=1))

    def calculate(self, algorithm, **budget):
        return coverage.evaluate(self.domain,self.seeds,self.model,route_binding=self.route,guide=self.first,
            algorithm=algorithm,budget=coverage.CoverageBudget(seconds=30,max_depth=1,**budget))

    def test_legacy_and_grouped_replay_and_retention(self):
        from femto.planning import coverage_workflow as workflow
        for algorithm in (coverage.PAGED_RECALCULATION_ALGORITHM,coverage.GROUPED_RECALCULATION_ALGORITHM):
            with self.subTest(algorithm=algorithm):
                certificate=self.calculate(algorithm)
                coverage.validate(certificate,self.domain,self.seeds,self.model,route_binding=self.route)
                self.assertEqual(workflow.require_recalculation_retention(self.first,certificate,self.ledger)['lost_weight_units'],0)
                self.assertEqual(certificate.guide.metadata,self.first.metadata)
                self.assertFalse(hasattr(certificate.guide,'guide'))

    def test_budget_frontier_and_rehashed_policy_cannot_relabel_legacy(self):
        from test_coverage_certificate import reseal
        from femto.planning.planning_revision import digest
        certificate=self.calculate(coverage.GROUPED_RECALCULATION_ALGORITHM,max_comparisons=1)
        coverage.validate(certificate,self.domain,self.seeds,self.model,route_binding=self.route)
        self.assertLessEqual(certificate.metadata['metrics']['comparisons'],1)
        metadata=deepcopy(certificate.metadata)
        from femto.planning.coverage_recalculation import POLICY
        metadata['inputs']['search_policy']=POLICY
        metadata['coverage_computation_sha256']=digest(metadata['inputs'])
        forged=reseal(certificate,metadata=metadata)
        with self.assertRaises(ValueError):coverage.validate_metadata(forged.metadata)

    def test_time_frontier_is_reproducible(self):
        full=self.calculate(coverage.GROUPED_RECALCULATION_ALGORITHM)
        certificate=coverage.evaluate(self.domain,self.seeds,self.model,route_binding=self.route,guide=self.first,
            algorithm=coverage.GROUPED_RECALCULATION_ALGORITHM,budget=coverage.CoverageBudget(seconds=30,max_depth=1),
            _frontier=dict(stop_reason='time_budget',steps=max(1,full.metadata['metrics']['steps']-10)))
        self.assertEqual(certificate.metadata['metrics']['stop_reason'],'time_budget')
        coverage.validate(certificate,self.domain,self.seeds,self.model,route_binding=self.route)

    def test_portable_roundtrip_and_flat_hint(self):
        from femto.planning import coverage_workflow as workflow,region_project
        certificate=self.calculate(coverage.GROUPED_RECALCULATION_ALGORITHM)
        case=self.case
        plan=workflow.attach(case.model,case.region,self.plan,certificate,self.ledger)
        path=region_project.save_project(case.model,case.region,case.root/'grouped.zip',plan)
        opened=region_project.open_project(path,workspace=case.root/'grouped_opened')
        self.assertEqual(opened.coverage_certificate.metadata,certificate.metadata)
        self.assertEqual(opened.coverage_certificate.guide.metadata,self.first.metadata)
        self.assertEqual(opened.saved_plan['surface_trajectory']['ordered_items'],self.plan['surface_trajectory']['ordered_items'])

if __name__=='__main__':unittest.main()
