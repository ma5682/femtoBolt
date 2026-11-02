"""Same-input P2 hints are finite inputs, never inherited coverage authority."""
from copy import copy, deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from femto.planning import coverage_archive as archive, coverage_certificate as coverage
from femto.planning import coverage_partition, coverage_recalculation, coverage_workflow as workflow
from femto.planning import footprint_model, region_project, refinement_workflow, coverage_refinement
from femto.planning.planning_revision import digest
from femto.planning.surface_trajectory import SurfaceTrajectory
from test_coverage_certificate import triangle_domain, seed, reseal
from test_coverage_workflow import fixture


class RetainedFirstCoverageTests(unittest.TestCase):
    def setUp(self):
        self.domain = triangle_domain(); self.records = [seed()]
        self.model = footprint_model.nominal(125.)
        self.budget = coverage.CoverageBudget(max_depth=3)
        self.old = coverage.evaluate(self.domain, self.records, self.model,
            budget=self.budget, algorithm=coverage.CURRENT_ALGORITHM)

    def evaluate(self, *, guide=None, budget=None, frontier=None, records=None):
        return coverage.evaluate(self.domain, self.records if records is None else records, self.model,
            budget=self.budget if budget is None else budget, algorithm=coverage.RECALCULATION_ALGORITHM,
            guide=self.old if guide is None else guide, _frontier=frontier)

    def replay(self, value):
        return coverage.validate(value, self.domain, self.records, self.model)

    def test_reproves_previous_set_before_remaining_search(self):
        value = self.evaluate()
        compared = coverage_partition.compare(self.old, value, self.domain.areas)
        self.assertEqual(compared['lost_weight_units'], 0)
        self.assertGreater(value.metadata['metrics']['retained_leaves_rechecked'], 0)
        self.assertEqual(value.metadata['metrics']['retained_leaves_rechecked'], self.old.summary['confirmed_leaf_count'])
        self.assertIs(self.replay(value), value)

    def test_zero_comparisons_cannot_copy_previous_confirmation(self):
        value = self.evaluate(budget=replace(self.budget, max_comparisons=0))
        self.assertEqual(value.summary['confirmed_leaf_count'], 0)
        self.assertGreater(coverage_partition.compare(self.old, value, self.domain.areas)['lost_weight_units'], 0)
        coverage_partition.intervals(value, len(self.domain.areas))
        self.replay(value)

    def test_reproved_frontier_allows_new_confirmation_on_an_adjacent_face(self):
        from femto.planning.coverage_domain import CoverageDomain
        domain = CoverageDomain.mesh([[0,0,0],[.01,0,0],[0,.01,0],[.01,.01,0]], [[0,1,2],[1,3,2]])
        old = coverage.evaluate(domain, self.records, self.model,
            algorithm=coverage.CURRENT_ALGORITHM, budget=replace(self.budget, max_comparisons=1))
        self.assertEqual(old.summary['confirmed_leaf_count'], 1)
        fresh = coverage.evaluate(domain, self.records, self.model,
            algorithm=coverage.RECALCULATION_ALGORITHM, guide=old, budget=self.budget)
        compared = coverage_partition.compare(old, fresh, domain.areas)
        self.assertEqual(compared['lost_weight_units'], 0)
        self.assertGreater(compared['gained_weight_units'], 0)
        self.assertEqual(fresh.summary['confirmed_leaf_count'], 2)
        coverage.validate(fresh, domain, self.records, self.model)

    def test_repeated_recalculation_has_one_flat_hint_and_four_files(self):
        previous = self.old
        with tempfile.TemporaryDirectory() as folder:
            for index in range(3):
                current = self.evaluate(guide=previous)
                self.assertIs(type(current.guide), coverage.Certificate)
                self.assertEqual(current.guide.metadata, previous.metadata)
                self.assertEqual(coverage_partition.compare(previous, current, self.domain.areas)['lost_weight_units'], 0)
                destination = Path(folder)/str(index)
                archive.save(destination, current, self.domain, self.records, self.model)
                self.assertEqual({p.name for p in destination.iterdir()}, archive.asset_names(current.metadata))
                self.assertEqual(len(list(destination.iterdir())), 4)
                previous = archive.read(destination, self.domain, self.records, self.model)
                self.assertEqual(previous.metadata, current.metadata)

    def test_missing_hint_cannot_be_read_as_an_independent_certificate(self):
        value = self.evaluate()
        with tempfile.TemporaryDirectory() as folder:
            archive.write_assets(folder, value)
            (Path(folder)/(archive.HINT_PREFIX+archive.ASSET)).unlink()
            with self.assertRaisesRegex(ValueError, '引导资产缺失'):
                archive.read(folder, self.domain, self.records, self.model)
        with self.assertRaisesRegex(ValueError, '单层引导资产'):
            self.replay(coverage.Certificate(value.metadata, value.arrays))

    def test_orphan_and_recursive_hints_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            archive.write_assets(folder, self.old)
            (Path(folder)/(archive.HINT_PREFIX+archive.MANIFEST)).write_text('{}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '孤立'):
                archive.load_assets(folder)
        value = self.evaluate()
        with tempfile.TemporaryDirectory() as folder:
            archive.write_assets(folder, value)
            (Path(folder)/(archive.HINT_PREFIX*2+archive.MANIFEST)).write_text('{}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '递归历史'):
                archive.load_assets(folder)

    def test_changed_seed_input_is_not_a_same_input_recalculation(self):
        with self.assertRaisesRegex(ValueError, '同一扫描种子'):
            self.evaluate(records=self.records+[seed((.125, 0, 0))])

    def test_changed_model_cannot_reuse_a_hint(self):
        with self.assertRaisesRegex(ValueError, '作用模型'):
            coverage.evaluate(self.domain, self.records, footprint_model.nominal(100.),
                guide=self.old, budget=self.budget, algorithm=coverage.RECALCULATION_ALGORITHM)

    def test_hint_outside_flags_are_recomputed(self):
        arrays = {key: a.copy() for key, a in self.old.arrays.items()}
        arrays['status'][arrays['status'] != 1] = 3
        altered = reseal(self.old, arrays=arrays)
        normal, changed = self.evaluate(), self.evaluate(guide=altered)
        for key in normal.arrays:np.testing.assert_array_equal(normal.arrays[key], changed.arrays[key])

    def test_rehashed_changed_proof_must_independently_replay(self):
        value = self.evaluate()
        arrays = {key: a.copy() for key, a in value.arrays.items()}
        arrays['proof_parent'][0] = 0
        altered = reseal(value, arrays=arrays)
        changed = coverage_recalculation.RecalculatedCertificate(altered.metadata, altered.arrays, value.guide)
        with self.assertRaisesRegex(ValueError, '无法重现'):
            self.replay(changed)

    def test_hint_hash_is_bound_even_when_output_metadata_is_resealed(self):
        value = self.evaluate()
        wrong = replace(value, guide=self.evaluate(budget=replace(self.budget, max_comparisons=0)).guide)
        # A different candidate output as hint changes the binding, even with
        # the same domain and input seeds.
        wrong = replace(wrong, guide=coverage_recalculation.flat_hint(value))
        with self.assertRaisesRegex(ValueError, '引导资产与计算绑定不一致'):
            self.replay(wrong)

    def test_pair_shares_existing_array_allocation_bound(self):
        value = self.evaluate()
        limit = sum(a.nbytes for c in (value, value.guide) for a in c.arrays.values())-1
        with patch.object(coverage, 'MAX_ARRAY_BYTES', limit):
            with self.assertRaisesRegex(ValueError, '合计超过'):
                coverage_recalculation.require_guide(value)

    def test_partial_stop_frontiers_reproduce_complete_target_partition(self):
        complete = self.evaluate()
        for steps in (0, 1, 16, 64, 128, complete.metadata['metrics']['steps']//2):
            with self.subTest(steps=steps):
                value = self.evaluate(frontier=dict(stop_reason='time_budget', steps=steps))
                coverage_partition.intervals(value, len(self.domain.areas))
                self.replay(value)


class RetainedFirstProductTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(); cls.addClassCleanup(cls.temp.cleanup)
        cls.base = fixture(Path(cls.temp.name)/'source')

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        model, region, original, old, ledger = self.base
        self.model = copy(model); self.model.store = copy(model.store)
        self.region, self.original = deepcopy(region), deepcopy(original)
        self.model.store.regions = [self.region]
        self.before = workflow.attach(self.model, self.region, self.original, old, ledger)
        self.certificate, self.ledger = workflow.calculate(self.model, self.region, self.before,
            budget=coverage.CoverageBudget(max_depth=1))
        self.plan = workflow.attach(self.model, self.region, self.before, self.certificate, self.ledger)

    def test_product_recalculation_keeps_original_route_and_target(self):
        self.assertEqual(self.certificate.metadata['algorithm'], coverage.RECALCULATION_ALGORITHM)
        self.assertEqual(self.plan['surface_trajectory'], self.before['surface_trajectory'])
        self.assertEqual(self.plan['planning_context'], self.before['planning_context'])
        self.assertEqual(workflow.require_recalculation_retention(self.base[3], self.certificate, self.ledger)['lost_weight_units'], 0)

    def test_portable_replay_can_recalculate_again_without_recursive_assets(self):
        path = region_project.save_project(self.model, self.region, self.root/'first.zip', self.plan)
        opened = region_project.open_project(path, workspace=self.root/'opened')
        proof, ledger = workflow.calculate(opened, opened.store.regions[0], opened.saved_plan,
            budget=coverage.CoverageBudget(max_depth=1))
        selected = workflow.attach(opened, opened.store.regions[0], opened.saved_plan, proof, ledger)
        again = region_project.save_project(opened, opened.store.regions[0], self.root/'second.zip', selected)
        final = region_project.open_project(again, workspace=self.root/'final')
        self.assertEqual(final.coverage_certificate.metadata, proof.metadata)
        self.assertIs(type(final.coverage_certificate.guide), coverage.Certificate)

    def test_portable_open_validates_geometry_once_and_replays_saved_proof(self):
        from femto.planning import surface_workspace
        path = region_project.save_project(self.model, self.region, self.root/'once.zip', self.plan)
        with patch.object(surface_workspace, 'validate', wraps=surface_workspace.validate) as geometry:
            with patch.object(coverage, 'validate', wraps=coverage.validate) as proof:
                opened = region_project.open_project(path, workspace=self.root/'once')
        geometry.assert_called_once()
        self.assertGreaterEqual(proof.call_count, 1)
        self.assertEqual(opened.coverage_certificate.metadata, self.certificate.metadata)
        for key, values in self.certificate.arrays.items():
            np.testing.assert_array_equal(opened.coverage_certificate.arrays[key], values)
        self.assertEqual(opened.target_ledger.metadata, self.ledger.metadata)

    def test_resealed_player_coordinates_still_require_full_geometry_validation(self):
        from femto.planning import planning_state, surface_workspace
        path = region_project.save_project(self.model, self.region, self.root/'original.zip', self.plan)
        with zipfile.ZipFile(path) as bundle:
            files = {name: bundle.read(name) for name in bundle.namelist()}
        data = json.loads(files['project.json'])
        data['plan']['segments'][0][0]['xyz_m'][0] += .01
        data['plan']['plan_binding'] = planning_state.binding(self.model.mesh, data['region'], data['plan'])
        planning_state.validate_binding(data['plan'], self.model.mesh, data['region'])
        files['project.json'] = json.dumps(data, ensure_ascii=False).encode('utf-8')
        changed = self.root/'resealed.zip'
        with zipfile.ZipFile(changed, 'w', zipfile.ZIP_DEFLATED) as bundle:
            for name, raw in files.items():bundle.writestr(name, raw)
        with patch.object(surface_workspace, 'validate', wraps=surface_workspace.validate) as geometry:
            with self.assertRaisesRegex(ValueError, '三维路线显示与原始轨迹不一致：segments'):
                region_project.open_project(changed, workspace=self.root/'rejected')
        geometry.assert_called_once()

    def test_no_addition_refinement_replays_and_restores_recalculated_base(self):
        result = refinement_workflow.calculate(self.model, self.region, self.plan,
            budget=coverage_refinement.RefinementBudget(max_rounds=0))
        path = region_project.save_project(self.model, self.region, self.root/'refined.zip', result)
        opened = region_project.open_project(path, workspace=self.root/'refined')
        restored = refinement_workflow.restore(opened, opened.store.regions[0], opened.saved_plan)
        self.assertEqual(restored['coverage_evidence'], self.plan['coverage_evidence'])
        self.assertEqual(restored['surface_trajectory'], self.plan['surface_trajectory'])
        self.assertIsInstance(opened.coverage_certificate, coverage_recalculation.RecalculatedCertificate)

    def test_native_replay_contains_and_verifies_single_hint(self):
        from femto.planning import surface_workspace
        from femto.planning.trajectory_export import load_trajectory
        native = surface_workspace.export(self.model, self.region, self.plan)
        proofs = []
        loaded = load_trajectory(native, coverage_sink=proofs.append)
        self.assertEqual(loaded.to_dict()['ordered_items'], self.plan['surface_trajectory']['ordered_items'])
        self.assertEqual(proofs[0].metadata, self.certificate.metadata)
        self.assertTrue((native/(archive.HINT_PREFIX+archive.ASSET)).is_file())


if __name__ == '__main__':unittest.main()
