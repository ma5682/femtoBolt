"""Same-operation preparation retains exact data, rejection and cancellation."""
from copy import deepcopy
from pathlib import Path
import tempfile
from unittest import TestCase
from unittest.mock import patch
import numpy as np

from femto.annotation import selection_intent as intent
from femto.planning import effective_domain as domain, evidence_preparation as preparation
from femto.planning import coverage_workflow as coverage, coverage_certificate
from femto.planning import surface_workspace, planning_state, coverage_domain
from femto.planning.surface_trajectory import SurfaceTrajectory
from test_coverage_workflow import fixture


class ClassPreparationTests(TestCase):
    def test_class_records_are_computed_fresh_and_track_changed_pixels(self):
        classes = np.ones((40, 40), np.int32)
        classes[3:5, 6:9] = 7
        expected = domain._class_records(classes, lambda: None)
        with preparation.operation(), patch.object(domain, 'missing_regions', wraps=domain.missing_regions) as build:
            first = domain._class_records(classes, lambda: None)
            first['quality_supported_mask_runs'][0] = 999
            first['missing_regions'][0]['pixel_count'] = 999
            again = domain._class_records(classes.copy(), lambda: None)
            self.assertEqual(again, expected)
            self.assertEqual(list(again['missing_regions'][0]), list(expected['missing_regions'][0]))
            self.assertEqual(build.call_count, 2)
            classes[0, 0] = 5
            self.assertNotEqual(domain._class_records(classes, lambda: None), expected)

    def test_policy_change_and_cancel_are_not_hidden_by_preparation(self):
        classes = np.ones((20, 20), np.int32)
        classes[:, ::2] = 7
        with preparation.operation():
            domain._class_records(classes, lambda: None)
            with patch.object(intent, 'MAX_RUNS', 1):
                with self.assertRaisesRegex(ValueError, 'fragmented'):
                    domain._class_records(classes, lambda: None)
            def cancel(): raise RuntimeError('cancelled')
            with self.assertRaisesRegex(RuntimeError, 'cancelled'):
                domain._class_records(classes, cancel)
        self.assertIsNone(preparation._state())

    def test_numpy_rle_matches_scalar_reference_including_integer_extremes(self):
        rng = np.random.default_rng(54)
        for value in (np.zeros((2,3), bool), rng.integers(0,8,(100,100),dtype=np.int32),
                      np.array([-(2**63), 2**63-1, 2**63-1], np.int64),
                      np.array([[1., 2.], [2., 4.]])[:, ::-1]):
            flat = np.ascontiguousarray(value).reshape(-1)
            if flat.dtype == bool: flat = flat.astype(np.uint8)
            change = np.flatnonzero(flat[1:] != flat[:-1]) + 1
            starts, ends = np.r_[0, change], np.r_[change, flat.size]
            expected = [int(v) for v in np.c_[flat[starts].astype(np.int64), ends-starts].reshape(-1)]
            actual = intent.encode_runs(value)
            self.assertEqual(actual, expected)
            self.assertTrue(all(type(item) is int for item in actual))


class CoverageOperationReuseTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.model, cls.region, cls.plan, cls.certificate, cls.ledger = fixture(Path(cls.temp.name)/'source')

    def test_calculate_parses_once_and_keeps_the_identical_certificate(self):
        with patch.object(SurfaceTrajectory, 'from_dict', wraps=SurfaceTrajectory.from_dict) as parse:
            certificate, ledger = coverage.calculate(self.model, self.region, self.plan,
                budget=coverage_certificate.CoverageBudget(max_depth=1))
        self.assertEqual(parse.call_count, 1)
        self.assertEqual(certificate.metadata, self.certificate.metadata)
        for name, value in certificate.arrays.items():
            np.testing.assert_array_equal(value, self.certificate.arrays[name])
        self.assertEqual(ledger.metadata, self.ledger.metadata)
        self.assertIsNone(preparation._state())

    def test_rebound_route_changed_after_validation_is_rejected(self):
        plan = deepcopy(self.plan)
        original = surface_workspace.target_for
        def mutate(*args, **kwargs):
            ledger = original(*args, **kwargs)
            plan['surface_trajectory']['ordered_items'][0]['points'][0]['xyz_m'][0] += .001
            plan['plan_binding'] = planning_state.binding(self.model.mesh, self.region, plan)
            return ledger
        with patch.object(surface_workspace, 'target_for', side_effect=mutate):
            with self.assertRaisesRegex(ValueError, '期间发生变化'):
                coverage.calculate(self.model, self.region, plan)
        self.assertIsNone(preparation._state())

    def test_factory_and_default_bindings_agree_with_original_public_contracts(self):
        original = coverage_domain.from_plan(self.model, self.region, self.plan)
        actual = coverage_domain.from_plan(self.model, self.region, self.plan,
                                            binding_factory=coverage.route_binding)
        self.assertEqual(original[0].binding, actual[0].binding)
        self.assertEqual(original[1:3], actual[1:3])
        self.assertEqual(original[3]['schema'], 'femto-accepted-v3-coverage-source-v1')
        self.assertEqual(actual[3], coverage.route_binding(SurfaceTrajectory.from_dict(self.plan['surface_trajectory'])))

    def test_binding_factory_cannot_mutate_the_checked_object(self):
        def changed(trajectory):
            from dataclasses import replace
            segment = trajectory.ordered_items[0]
            points = deepcopy(segment.points)
            points[0]['xyz_m'][0] += .001
            trajectory.ordered_items[0] = replace(segment, points=points)
            return coverage.route_binding(trajectory)
        with self.assertRaisesRegex(ValueError, '期间发生变化'):
            coverage_domain.from_plan(self.model, self.region, deepcopy(self.plan), binding_factory=changed)
