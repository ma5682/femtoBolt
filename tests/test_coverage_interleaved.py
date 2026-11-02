"""P3 scheduling must preserve proofs through every interleaved stop boundary."""
from copy import copy, deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from femto.geometry import exact_surface as exact
from femto.planning import coverage_certificate as coverage, coverage_partition, footprint_model
from femto.planning import coverage_interleaved, coverage_residual, coverage_refinement as refinement
from femto.planning import coverage_workflow, refinement_workflow, refinement_archive, region_project
from femto.planning.coverage_domain import CoverageDomain
from femto.planning.planning_revision import digest
from femto.planning.surface_trajectory import SurfaceTrajectory
from test_coverage_certificate import triangle_domain, seed, reseal
from test_coverage_workflow import fixture


class InterleavedCoverageTests(unittest.TestCase):
    def setUp(self):
        self.domain = triangle_domain(); self.model = footprint_model.nominal(125.)
        self.old_records = [seed()]; self.records = self.old_records+[seed((.125,0,0))]
        self.budget = coverage.CoverageBudget(max_depth=3)
        self.old = coverage.evaluate(self.domain, self.old_records, self.model,
            budget=self.budget, algorithm=coverage.CURRENT_ALGORITHM)

    def evaluate(self, *, guide=None, budget=None, frontier=None):
        return coverage.evaluate(self.domain, self.records, self.model, budget=budget or self.budget,
            algorithm=coverage.INTERLEAVED_ALGORITHM, guide=self.old if guide is None else guide,
            _frontier=frontier)

    def replay(self, result, guide=None):
        return coverage.validate(result, self.domain, self.records, self.model, guide=self.old if guide is None else guide)

    def test_new_scan_gain_preserves_original_set_and_replays(self):
        result = self.evaluate()
        compared = coverage_partition.compare(self.old, result, self.domain.areas)
        self.assertEqual(compared['lost_weight_units'], 0)
        self.assertGreater(compared['gained_weight_units'], 0)
        self.assertGreater(result.metadata['metrics']['interleaved_search_batches'], 0)
        self.assertGreater(result.metadata['metrics']['interleaved_residual_batches'], 0)
        self.replay(result)

    def test_no_confirmation_is_inherited_when_budget_stops_before_reproof(self):
        result = self.evaluate(budget=replace(self.budget, max_comparisons=0))
        self.assertEqual(result.summary['confirmed_leaf_count'], 0)
        coverage_partition.intervals(result, len(self.domain.areas))
        self.replay(result)

    def test_saved_outside_flags_cannot_exclude_current_new_balls(self):
        arrays = {key: a.copy() for key, a in self.old.arrays.items()}
        arrays['status'][arrays['status'] != 1] = 3
        hint = reseal(self.old, arrays=arrays)
        expected, actual = self.evaluate(), self.evaluate(guide=hint)
        for key in expected.arrays:np.testing.assert_array_equal(expected.arrays[key], actual.arrays[key])
        self.replay(actual, hint)

    def test_every_outside_result_matches_independent_exact_ball_distance(self):
        result = self.evaluate(); arrays = result.arrays
        balls = [exact.anchor_ball(r['xyz_m'], self.domain.triangle(0), .0625) for r in self.records]
        for index in np.flatnonzero(arrays['status'] == 3):
            bary = exact.descend(exact.IDENTITY, int(arrays['depth'][index]), int(arrays['code'][index]))
            triangle = tuple(exact.xyz(p, self.domain.triangle(0)) for p in bary)
            for _, center, radius in balls:
                self.assertGreater(coverage._distance2(center, triangle), exact.Q(radius)**2)

    def test_stopping_steps_keep_full_original_weights_and_replay(self):
        full = self.evaluate(); steps = full.metadata['metrics']['steps']
        for stop in (0, 1, 32, 128, steps//3, steps//2, steps-1):
            with self.subTest(stop=stop):
                result = self.evaluate(frontier=dict(stop_reason='time_budget', steps=stop))
                coverage_partition.intervals(result, len(self.domain.areas))
                self.replay(result)
                if result.metadata['metrics']['retained_leaves_rechecked'] == self.old.summary['confirmed_leaf_count']:
                    self.assertEqual(coverage_partition.compare(self.old, result, self.domain.areas)['lost_weight_units'], 0)

    def test_each_cumulative_work_limit_is_respected(self):
        for kind in ('comparisons', 'adjacencies', 'proofs'):
            with self.subTest(kind=kind):
                result = self.evaluate(budget=replace(self.budget, **{'max_'+kind: 12}))
                self.assertLessEqual(result.metadata['metrics'][kind], 12)
                coverage_partition.intervals(result, len(self.domain.areas))
                self.replay(result)

    def test_policy_and_algorithm_cannot_be_mixed_by_rehashing(self):
        result = self.evaluate(); metadata = deepcopy(result.metadata)
        metadata['inputs']['search_policy'] = deepcopy(coverage_residual.POLICY)
        metadata['coverage_computation_sha256'] = digest(metadata['inputs'])
        bad = reseal(result, metadata=metadata)
        with self.assertRaisesRegex(ValueError, '策略版本'):
            self.replay(bad)

    def test_legacy_v4_replay_stays_identical_after_v5_runs(self):
        before = coverage.evaluate(self.domain, self.records, self.model, guide=self.old,
            budget=self.budget, algorithm=coverage.RESIDUAL_ALGORITHM)
        self.evaluate()
        after = coverage.evaluate(self.domain, self.records, self.model, guide=self.old,
            budget=self.budget, algorithm=coverage.RESIDUAL_ALGORITHM)
        self.assertEqual(before.metadata, after.metadata)
        for key in before.arrays:np.testing.assert_array_equal(before.arrays[key], after.arrays[key])
        coverage.validate(before, self.domain, self.records, self.model, guide=self.old)

    def test_new_seed_groups_are_visited_in_turn(self):
        vertices, faces = [], []
        for offset in (0., .5):
            for index in range(20):
                x = offset+index*.001; first = len(vertices)
                vertices.extend(((x,0,0),(x+.0005,0,0),(x,.0005,0)))
                faces.append((first,first+1,first+2))
        domain = CoverageDomain.mesh(vertices, faces)
        old = coverage.evaluate(domain, [], self.model, algorithm=coverage.CURRENT_ALGORITHM)
        records = [seed((0,0,0)), dict(source_face=20, xyz_m=[.5,0,0], role='scan')]
        engine = coverage_interleaved.InterleavedEngine(domain, records, self.model,
            self.budget, lambda:None, None, guide=old)
        engine.seed(); self.assertTrue(engine._prepare()); self.assertTrue(engine._reprove())
        order = engine._new_order()
        self.assertEqual(len(order), 40)
        self.assertTrue(np.all(order[:8] < 20))
        self.assertTrue(np.all(order[8:16] >= 20))
        self.assertEqual(len(np.unique(order)), 40)


class InterleavedProductTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(); cls.addClassCleanup(cls.temp.cleanup)
        cls.base = fixture(Path(cls.temp.name)/'source')

    def test_recalculated_base_with_v5_addition_replays_and_restores(self):
        model, region, original, old, ledger = self.base
        model = copy(model); model.store = copy(model.store)
        region = deepcopy(region); model.store.regions = [region]
        before = coverage_workflow.attach(model, region, deepcopy(original), old, ledger)
        p2, ledger = coverage_workflow.calculate(model, region, before, budget=coverage.CoverageBudget(max_depth=1))
        before = coverage_workflow.attach(model, region, before, p2, ledger)
        result = refinement_workflow.calculate(model, region, before,
            budget=refinement.RefinementBudget(max_rounds=1))
        report = result['surface_trajectory']['gap_report']['refinement']
        self.assertEqual(report['schema'], refinement.INTERLEAVED_SCHEMA)
        self.assertTrue(any(row['accepted'] for row in report['rounds']))
        for row in report['rounds']:
            if row['accepted']:self.assertEqual(row['comparison']['lost_weight_units'], 0)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = region_project.save_project(model, region, root/'plan.zip', result)
            opened = region_project.open_project(path, workspace=root/'opened')
            self.assertEqual(opened.coverage_certificate.metadata, model.coverage_certificate.metadata)
            restored = refinement_workflow.restore(opened, opened.store.regions[0], opened.saved_plan)
            self.assertEqual(restored['surface_trajectory'], before['surface_trajectory'])
            self.assertEqual(restored['coverage_evidence'], p2.metadata)


if __name__ == '__main__':unittest.main()
