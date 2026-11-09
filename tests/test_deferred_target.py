"""On-demand area geometry must preserve real route checks and archived motion."""
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import open3d as o3d

import test_surface_workspace as fixtures
import test_target_surface as target_fixtures
from femto.planning import (surface_workspace as workspace, target_surface as target,
                            target_preparation, planning_state, region_project,
                            coverage_workflow)
from femto.planning.save_validation import SaveValidationSession
from femto.planning.trajectory_export import load_trajectory


class DeferredTargetTests(unittest.TestCase):
    def fixture(self, split=False):
        case = fixtures.SurfaceWorkspaceTests()
        case.setUp()
        self.addCleanup(case.doCleanups)
        o3d.io.write_triangle_mesh(case.model.store.source, case.model.mesh)
        region = case.region(split)
        case.model.store.regions = [region]
        return case.model, region, case.head, Path(case.temp.name)

    def assert_same_route(self, left, right):
        for name in ('ordered_items', 'statistics', 'gap_report'):
            self.assertEqual(left['surface_trajectory'][name], right['surface_trajectory'][name], name)
        for name in ('segments', 'chain_breaks', 'coordinate_frame', 'unit', 'treatment_head'):
            self.assertEqual(left[name], right[name], name)

    def test_generation_skips_area_construction_and_keeps_connected_and_split_routes(self):
        for split in (False, True):
            with self.subTest(split=split):
                model, region, head, _ = self.fixture(split)
                full = workspace.generate(model, region, head)
                with patch.object(target, '_construct', side_effect=AssertionError('area calculation ran')):
                    pending = workspace.generate(model, region, head, defer_target=True)
                    planning_state.validate(pending, model.mesh, region)
                    ledger = workspace.target_for(model, pending, reproduce=True)
                self.assert_same_route(full, pending)
                self.assertEqual(full['planning_context']['revision'], pending['planning_context']['revision'])
                self.assertTrue(target.is_deferred(ledger.metadata))
                self.assertEqual(ledger.summary['leaf_count'], 0)
                self.assertEqual(ledger.metadata['metrics']['clip_count'], 0)
                self.assertIsNone(ledger.summary['total_target_area_m2'])
                self.assertEqual(ledger.summary['coverage_status'], 'not_evaluated')
                self.assertIn('尚未计算', target.summary_text(ledger.metadata))

    def test_native_portable_and_independent_reopen_preserve_motion_without_area_replay(self):
        model, region, head, root = self.fixture(True)
        original_source = Path(model.store.source).read_bytes()
        with patch.object(target, '_construct', side_effect=AssertionError('area calculation ran')):
            plan = workspace.generate(model, region, head, defer_target=True)
            with SaveValidationSession() as session:
                native = workspace.export(model, region, plan, target_session=session)
                archive = region_project.save_project(model, region, root/'route.zip', plan,
                                                       target_session=session)
            saved = load_trajectory(native)
            with patch.object(planning_state, 'validate', wraps=planning_state.validate) as validate:
                reopened = region_project.open_project(archive, workspace=root/'reopened')
            self.assertGreater(validate.call_count, 0)
        self.assertEqual(saved.to_dict()['ordered_items'], plan['surface_trajectory']['ordered_items'])
        self.assert_same_route(plan, reopened.saved_plan)
        self.assertTrue(target.is_deferred(reopened.target_ledger.metadata))
        self.assertEqual(Path(model.store.source).read_bytes(), original_source)

    def test_explicit_area_check_materializes_same_target_and_leaves_old_plan_unchanged(self):
        model, region, head, root = self.fixture()
        full = workspace.generate(model, region, head)
        expected = workspace.target_for(model, full)
        pending = workspace.generate(model, region, head, defer_target=True)
        before = deepcopy(pending)
        with self.assertRaisesRegex(ValueError, '目标面积尚未计算'):
            coverage_workflow.calculate(model, region, pending)
        prepared = target_preparation.prepare(model, region, pending, output=root/'analysis')
        self.assert_same_route(pending, prepared)
        self.assertEqual(before, pending)
        actual = workspace.target_for(model, prepared, reproduce=True)
        self.assertEqual(expected.metadata, actual.metadata)
        for name, array in expected.arrays.items():
            np.testing.assert_array_equal(array, actual.arrays[name], err_msg=name)
        self.assertTrue((root/'analysis'/'previous'/'plan.json').is_file())
        with patch.object(target, 'build', side_effect=AssertionError('computed target rebuilt')):
            self.assertIs(target_preparation.prepare(model, region, prepared), prepared)
        from femto.planning.coverage_certificate import CoverageBudget
        certificate, ledger = coverage_workflow.calculate(model, region, prepared,
                                                          budget=CoverageBudget(max_depth=1))
        covered = coverage_workflow.attach(model, region, prepared, certificate, ledger)
        archive = region_project.save_project(model, region, root/'covered.zip', covered)
        reopened = region_project.open_project(archive, workspace=root/'covered-opened')
        self.assert_same_route(pending, reopened.saved_plan)
        self.assertEqual(reopened.saved_plan['coverage_evidence'], certificate.metadata)

    def test_pending_pixels_cannot_be_rehashed_into_a_located_surface(self):
        model, region, head, _ = self.fixture()
        plan = workspace.generate(model, region, head, defer_target=True)
        ledger = workspace.target_for(model, plan)
        arrays = {name: value.copy() for name, value in ledger.arrays.items()}
        arrays['pixel_status'][arrays['mask']] = 1
        params = plan['surface_trajectory']['planner']['parameters']
        revision = plan['planning_context']['revision']
        forged = target._seal(arrays, revision, target.TargetBudget(0, 0, 0, 0.),
                              ledger.metadata['metrics'], params['quality_evidence']['sha256'],
                              params['effective_domain']['sha256'])
        for reproduce in (False, True):
            with self.subTest(reproduce=reproduce), self.assertRaisesRegex(ValueError, '待计算目标'):
                target.validate(forged, revision, evidence=params['quality_evidence'],
                                domain=params['effective_domain'], reproduce=reproduce)

    def test_explicit_strict_path_prepares_area_and_restores_the_original_motion(self):
        from femto.planning import strict_workflow
        model, region, head, _ = self.fixture()
        pending = workspace.generate(model, region, head, defer_target=True)
        strict = strict_workflow.calculate(model, region, pending)
        self.assertFalse(target.is_deferred(strict['planning_context']['target']))
        restored = strict_workflow.restore(model, region, strict)
        self.assert_same_route(pending, restored)

    def test_route_edits_and_changed_reference_depth_are_still_rejected(self):
        model, region, head, _ = self.fixture()
        plan = workspace.generate(model, region, head, defer_target=True)
        forged = deepcopy(plan)
        forged['segments'][0][0]['xyz_m'][0] += .01
        forged['plan_binding'] = planning_state.binding(model.mesh, region, forged)
        with self.assertRaises(ValueError):
            planning_state.validate(forged, model.mesh, region)
        model.projection.depth = model.projection.depth.copy()
        model.projection.depth[model.projection.valid] += 1
        with self.assertRaises(ValueError):
            workspace.export(model, region, plan)

    def test_cancelled_or_changed_input_during_area_check_preserves_pending_plan(self):
        model, region, head, root = self.fixture()
        plan = workspace.generate(model, region, head, defer_target=True)
        before = deepcopy(plan)
        def cancel():
            raise RuntimeError('cancelled')
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):
            target_preparation.prepare(model, region, plan, cancel, output=root/'cancelled')
        build = target.build
        def change(*args, **kwargs):
            result = build(*args, **kwargs)
            model.projection.depth = model.projection.depth.copy()
            model.projection.depth[model.projection.valid] += 1
            return result
        with patch.object(target, 'build', side_effect=change), self.assertRaises(ValueError):
            target_preparation.prepare(model, region, plan, output=root/'changed')
        self.assertEqual(before, plan)
        self.assertFalse((root/'cancelled').exists())
        self.assertFalse((root/'changed').exists())

    def test_pending_keeps_raw_missing_depth_and_source_pixel_reasons(self):
        case = target_fixtures.TargetSurfaceTests()
        case.setUp()
        self.addCleanup(case.doCleanups)
        mask = np.zeros((100, 100), bool)
        mask[49:52, 49:52] = True
        model, region = case.fixture(mask=mask)
        model.projection.depth[49, 49] = 0
        model.projection.depth[49, 50] = 65000
        model.projection.image_valid = np.ones(mask.shape, bool)
        model.projection.image_valid[50, 49] = False
        pending, revision, evidence = case.build(model, region, deferred=True)
        status = pending.arrays['pixel_status']
        self.assertEqual(status[49, 49], 2)
        self.assertEqual(status[49, 50], 3)
        self.assertEqual(status[50, 49], 4)
        self.assertEqual(status[50, 50], 9)
        target.validate(pending, revision, evidence=evidence, domain=None)


if __name__ == '__main__':
    unittest.main()
