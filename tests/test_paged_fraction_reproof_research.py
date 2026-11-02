"""Independent replay and isolation for paged fractional research proofs."""
from copy import deepcopy
import unittest

import numpy as np

import test_paged_target_workflow as paged
from femto.planning import (auto_fraction_supplement_research as auto_supplement, coverage_certificate as coverage, coverage_workflow,
    footprint_model, fraction_coverage, fraction_footprint, scan_fraction_seeds)
from femto.planning.planning_revision import digest
from femto.planning.surface_trajectory import ScanSegment, SurfaceTrajectory


class PagedFractionReproofTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        paged.PagedTargetWorkflowTests.setUpClass()
        cls.addClassCleanup(paged.PagedTargetWorkflowTests.tearDownClass)
        case = paged.PagedTargetWorkflowTests()
        case.setUp(); cls.addClassCleanup(case.doCleanups)
        cls.case = case
        cls.plan, cls.ledger = case.continued()
        cls.trajectory = SurfaceTrajectory.from_dict(cls.plan['surface_trajectory'])
        cls.domain, cls.endpoints, cls.width, cls.route = coverage_workflow.bundle(cls.trajectory, cls.ledger)
        requests = []
        for segment in cls.trajectory.ordered_items:
            if not isinstance(segment, ScanSegment):continue
            for i, (a, b) in enumerate(zip(segment.points, segment.points[1:])):
                if b.get('role') != 'connector' and a['xyz_m'] != b['xyz_m']:
                    requests.append((segment.segment_index, i, 1, 2))
                    break
            if requests:break
        if not requests:raise AssertionError('fixture has no nonzero scan edge')
        cls.requests = requests
        positions, _ = scan_fraction_seeds.propose(cls.trajectory, cls.ledger.arrays['region_faces'], requests)
        if not positions:raise AssertionError('fixture did not yield fractional seeds')
        cls.seeds = cls.endpoints + positions
        cls.fraction_model = fraction_footprint.nominal(cls.width['effective_width_mm'])
        cls.budget = coverage.CoverageBudget(max_depth=1, seconds=30)
        cls.guide = coverage.evaluate(cls.domain, cls.endpoints, cls.width, route_binding=cls.route,
                                      algorithm=coverage.PAGED_ALGORITHM, budget=cls.budget)
        cls.proof = coverage.evaluate(cls.domain, cls.seeds, cls.fraction_model, route_binding=cls.route,
            algorithm=coverage.RESEARCH_PAGED_FRACTION_REPROOF_ALGORITHM, budget=cls.budget, guide=cls.guide)

    def test_paged_fractional_proof_independently_replays(self):
        self.assertGreater(len(self.ledger.metadata['construction_pages']), 1)
        coverage.validate(self.guide, self.domain, self.endpoints, self.width, route_binding=self.route)
        coverage.validate(self.proof, self.domain, self.seeds, self.fraction_model,
                          route_binding=self.route, guide=self.guide)
        self.assertEqual(self.proof.metadata['inputs']['guide_sha256'], self.guide.metadata['sha256'])
        self.assertFalse(self.proof.summary['execution_ready'])
        self.assertNotIn(coverage.RESEARCH_PAGED_FRACTION_REPROOF_ALGORITHM, coverage.P2_ALGORITHMS)

    def test_guided_workflow_selects_paged_version_and_replays(self):
        proof, proposal = fraction_coverage.guided_calculate(self.case.model, self.case.region, self.plan,
            self.guide, self.plan, self.requests, budget=self.budget)
        self.assertEqual(proof.metadata['algorithm'], coverage.RESEARCH_PAGED_FRACTION_REPROOF_ALGORITHM)
        self.assertEqual(fraction_coverage.guided_replay(self.case.model, self.case.region, self.plan,
            self.guide, self.plan, self.requests, proof), proposal)
        with self.assertRaises(ValueError):
            fraction_coverage.guided_replay(self.case.model, self.case.region, self.plan,
                self.guide, self.plan, [], proof)

    def test_unguided_paged_research_replays_but_product_binding_rejects_it(self):
        proof, proposal = fraction_coverage.calculate(self.case.model, self.case.region, self.plan,
            self.requests, budget=self.budget)
        self.assertEqual(proof.metadata['algorithm'], coverage.PAGED_ALGORITHM)
        self.assertEqual(fraction_coverage.replay(self.case.model, self.case.region, self.plan,
            self.requests, proof), proposal)
        with self.assertRaises(ValueError):
            coverage_workflow.check_binding(dict(self.plan, coverage_evidence=proof.metadata))

    def test_automatic_supplement_proposal_replays_and_rejects_bad_proof(self):
        result,proof=auto_supplement.propose(self.case.model,self.case.region,self.plan,self.guide,
            self.plan,self.guide,budget=self.budget)
        self.assertTrue(result['comparison']['retained_subset'])
        self.assertEqual(result['selected_edges'],[])
        self.assertFalse(result['installed_route'])
        self.assertEqual(proof.metadata['algorithm'],coverage.RESEARCH_PAGED_FRACTION_REPROOF_ALGORITHM)
        bad=coverage.Certificate(dict(self.guide.metadata,sha256='0'*64),self.guide.arrays)
        with self.assertRaises(ValueError):
            auto_supplement.propose(self.case.model,self.case.region,self.plan,bad,self.plan,self.guide,budget=self.budget)

    def test_rejects_wrong_width_seed_guide_and_old_version(self):
        with self.assertRaises(ValueError):
            coverage.evaluate(self.domain, self.seeds, self.fraction_model, route_binding=self.route,
                algorithm=coverage.RESEARCH_FRACTION_REPROOF_ALGORITHM, budget=self.budget, guide=self.guide)
        with self.assertRaises(ValueError):
            coverage.evaluate(self.domain, self.seeds, footprint_model.nominal(self.width['effective_width_mm']),
                route_binding=self.route, algorithm=coverage.RESEARCH_PAGED_FRACTION_REPROOF_ALGORITHM,
                budget=self.budget, guide=self.guide)
        with self.assertRaises(ValueError):
            coverage.validate(self.proof, self.domain, self.endpoints, self.fraction_model,
                              route_binding=self.route, guide=self.guide)
        with self.assertRaises(ValueError):
            coverage.validate(self.proof, self.domain, self.seeds, self.fraction_model, route_binding=self.route)

    def test_rehashed_policy_and_target_tampering_rejected(self):
        metadata = deepcopy(self.proof.metadata)
        metadata['inputs']['search_policy'].pop('paged_target')
        metadata['coverage_computation_sha256'] = digest(metadata['inputs'])
        metadata['sha256'] = digest({k:v for k,v in metadata.items() if k != 'sha256'})
        with self.assertRaises(ValueError):coverage.validate_metadata(metadata)
        metadata = deepcopy(self.proof.metadata)
        metadata['inputs']['domain']['target_page_units'][0] += 1
        metadata['coverage_computation_sha256'] = digest(metadata['inputs'])
        metadata['sha256'] = digest({k:v for k,v in metadata.items() if k != 'sha256'})
        forged = coverage.Certificate(metadata, self.proof.arrays)
        with self.assertRaises(ValueError):
            coverage.validate(forged, self.domain, self.seeds, self.fraction_model,
                              route_binding=self.route, guide=self.guide)

    def test_unpaged_domain_cannot_use_paged_version(self):
        domain = coverage.CoverageDomain.mesh([[0,0,0],[1,0,0],[0,1,0]], [[0,1,2]])
        with self.assertRaises(ValueError):
            coverage.evaluate(domain, [], self.fraction_model, algorithm=coverage.RESEARCH_PAGED_FRACTION_REPROOF_ALGORITHM,
                              budget=self.budget, guide=self.guide)


if __name__ == '__main__':unittest.main()
