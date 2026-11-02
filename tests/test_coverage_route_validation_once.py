"""A coverage request runs one complete surface check without trusting a cache."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from femto.planning import coverage_domain, planning_state, surface_workspace
from test_coverage_workflow import fixture


class CoverageRouteValidationOnceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.model, cls.region, cls.plan, _, _ = fixture(Path(cls.temp.name)/'source')

    def test_one_geometry_validation_preserves_domain_and_seeds(self):
        original = surface_workspace.validate
        with patch.object(surface_workspace, 'validate', wraps=original) as validations:
            domain, seeds, model, route = coverage_domain.from_plan(self.model, self.region, self.plan)
        self.assertEqual(validations.call_count, 1)
        self.assertEqual(seeds, coverage_domain.scan_endpoints(
            original(self.plan, self.model.mesh, self.region), domain.ledger.arrays['region_faces']))
        self.assertEqual(model['effective_width_mm'], self.plan['treatment_head']['effective_treatment_width_mm'])
        self.assertTrue(route['accepted_scan_edges_only'])
        self.assertEqual(domain.binding['target_sha256'], self.plan['planning_context']['target']['sha256'])
        self.assertIsNone(planning_state.validate(self.plan, self.model.mesh, self.region))

    def test_changed_route_during_validation_is_rejected_even_if_rebound(self):
        plan = deepcopy(self.plan)
        original = surface_workspace.validate
        def mutate_after_check(*args, **kwargs):
            trajectory = original(*args, **kwargs)
            plan['surface_trajectory']['ordered_items'][0]['points'][0]['xyz_m'][0] += .001
            plan['plan_binding'] = planning_state.binding(self.model.mesh, self.region, plan)
            return trajectory
        with patch.object(surface_workspace, 'validate', side_effect=mutate_after_check):
            with self.assertRaisesRegex(ValueError, '期间发生变化'):
                coverage_domain.from_plan(self.model, self.region, plan)

    def test_each_request_rechecks_geometry_and_can_cancel(self):
        coverage_domain.from_plan(self.model, self.region, self.plan)
        bad = deepcopy(self.plan)
        bad['surface_trajectory']['ordered_items'][0]['points'][0]['xyz_m'][0] += .01
        bad['plan_binding'] = planning_state.binding(self.model.mesh, self.region, bad)
        with self.assertRaises(ValueError):
            coverage_domain.from_plan(self.model, self.region, bad)
        def cancelled():raise RuntimeError('coverage cancelled')
        with self.assertRaisesRegex(RuntimeError, 'coverage cancelled'):
            coverage_domain.from_plan(self.model, self.region, self.plan, check=cancelled)

    def test_existing_ignored_trajectory_metadata_remains_accepted(self):
        plan = deepcopy(self.plan)
        plan['surface_trajectory']['legacy_extra'] = {'note': 'retained'}
        plan['plan_binding'] = planning_state.binding(self.model.mesh, self.region, plan)
        # Preserve the prior validator's handling of additional legacy fields.
        surface_workspace.validate(plan, self.model.mesh, self.region)
        domain, _, _, _ = coverage_domain.from_plan(self.model, self.region, plan)
        self.assertEqual(domain.binding['target_sha256'], plan['planning_context']['target']['sha256'])


if __name__ == '__main__':
    unittest.main()
