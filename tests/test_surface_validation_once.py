"""Single-pass workspace validation retains all independent rejection predicates."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np

from femto.planning import surface_workspace as workspace
from femto.planning import surface_trajectory as trajectory_api
from femto.planning.planning_state import binding, validate
import test_surface_workspace as workspace_fixture


class SurfaceValidationOnceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = workspace_fixture.SurfaceWorkspaceTests()
        cls.fixture.setUp()
        cls.addClassCleanup(cls.fixture.doCleanups)
        cls.model = cls.fixture.model
        cls.region = cls.fixture.region()
        cls.plan = workspace.generate(cls.model, cls.region, cls.fixture.head)
        cls.local = workspace.canonical_region_mesh(cls.model.mesh, cls.region)

    def test_single_complete_pass_keeps_the_same_player_projection(self):
        baseline = trajectory_api.to_player_plan(self.plan['surface_trajectory'])
        original = trajectory_api.validate_trajectory
        with patch.object(trajectory_api, 'validate_trajectory', wraps=original) as checks:
            result = workspace.validate(self.plan, self.model.mesh, self.region)
        full_input_calls = [call for call in checks.call_args_list
                            if call.args[0].planning_context is not None]
        self.assertEqual(len(full_input_calls), 1)
        self.assertIsNotNone(full_input_calls[0].kwargs['mesh'])
        self.assertIs(full_input_calls[0].kwargs['region'], self.region)
        self.assertEqual(trajectory_api.to_player_plan(result), baseline)
        self.assertEqual(result.to_dict(), self.plan['surface_trajectory'])

    def test_full_parse_rejects_points_off_the_supplied_mesh(self):
        # The serialized route remains structurally valid: only independent
        # physical support verification can reject this displaced local mesh.
        payload = deepcopy(self.plan['surface_trajectory'])
        trajectory_api.SurfaceTrajectory.from_dict(payload)
        moved = self.local.copy()
        moved.vertices = np.asarray(moved.vertices) + [0., 0., .02]
        with self.assertRaises(ValueError):
            trajectory_api.SurfaceTrajectory.from_dict(payload, mesh=moved, region=self.region)

    def test_result_digest_and_quality_report_still_reject_tampering(self):
        for mutate in (
            lambda value: value.update(result_sha256='0' * 64),
            lambda value: value['gap_report']['quality_constraint'].update(target_completion='complete'),
        ):
            payload = deepcopy(self.plan['surface_trajectory'])
            mutate(payload)
            with self.assertRaises(ValueError):
                trajectory_api.SurfaceTrajectory.from_dict(payload, mesh=self.local, region=self.region)

    def test_changed_region_is_rejected_by_full_parse_and_workspace(self):
        region = deepcopy(self.region)
        region['annotation_id'] += '-changed'
        with self.assertRaises(ValueError):
            trajectory_api.SurfaceTrajectory.from_dict(self.plan['surface_trajectory'], mesh=self.local, region=region)
        with self.assertRaises(ValueError):
            workspace.validate(self.plan, self.model.mesh, region)

    def test_player_tamper_is_rejected_even_with_a_recomputed_plan_hash(self):
        plan = deepcopy(self.plan)
        plan['segments'] = deepcopy(plan['segments'])
        plan['segments'][0][0]['xyz_m'][0] += .01
        plan['plan_binding'] = binding(self.model.mesh, self.region, plan)
        with self.assertRaisesRegex(ValueError, '显示与原始轨迹不一致'):
            validate(plan, self.model.mesh, self.region)

    def test_cancel_reaches_the_single_complete_validation(self):
        def cancel():
            raise RuntimeError('single-pass validation cancelled')
        with self.assertRaisesRegex(RuntimeError, 'single-pass validation cancelled'):
            trajectory_api.SurfaceTrajectory.from_dict(self.plan['surface_trajectory'],
                mesh=self.local, region=self.region, check=cancel)
        with self.assertRaisesRegex(RuntimeError, 'single-pass validation cancelled'):
            workspace.validate(self.plan, self.model.mesh, self.region, check=cancel)
        original = trajectory_api._point_on_triangle
        geometry_calls = []
        def observe_geometry(*args, **kwargs):
            geometry_calls.append(True)
            return original(*args, **kwargs)
        def cancel_after_geometry():
            if geometry_calls:
                cancel()
        with patch.object(trajectory_api, '_point_on_triangle', side_effect=observe_geometry):
            with self.assertRaisesRegex(RuntimeError, 'single-pass validation cancelled'):
                workspace.validate(self.plan, self.model.mesh, self.region, check=cancel_after_geometry)
        self.assertTrue(geometry_calls)

    def test_second_validation_never_reuses_a_previous_acceptance(self):
        plan = deepcopy(self.plan)
        workspace.validate(plan, self.model.mesh, self.region)
        plan['surface_trajectory']['gap_report']['quality_constraint']['target_completion'] = 'complete'
        plan['plan_binding'] = binding(self.model.mesh, self.region, plan)
        with self.assertRaises(ValueError):
            validate(plan, self.model.mesh, self.region)


if __name__ == '__main__':
    unittest.main()
