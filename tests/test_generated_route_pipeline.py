"""One generated object receives full final checks before its display projection."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from unittest import TestCase, main
from unittest.mock import patch
import numpy as np

import test_surface_workspace as fixture_module
from femto.planning import surface_workspace as workspace, planning_state, continuous_quality
from femto.planning.surface_trajectory import SurfaceTrajectory
from femto.planning.evidence_preparation import _state


class GeneratedRoutePipelineTests(TestCase):
    def setUp(self):
        self.fixture = fixture_module.SurfaceWorkspaceTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.model = self.fixture.model
        self.head = self.fixture.head

    def test_checked_object_matches_old_pipeline_without_parsing_or_preview_validation(self):
        region = self.fixture.region(split=True)
        old = workspace.generate(self.model, region, self.head, defer_target=True)
        planning_state.validate(old, self.model.mesh, region)
        manifest = Path(old['surface_trajectory']['source']['region_manifest'])
        original = workspace.validate_trajectory
        calls = []
        def checked(*args, **kwargs):
            calls.append(kwargs.get('mesh'))
            return original(*args, **kwargs)
        with patch.object(workspace, 'snapshot', return_value=manifest), \
             patch.object(workspace, 'to_player_plan', side_effect=AssertionError('duplicate preview validation')), \
             patch.object(SurfaceTrajectory, 'from_dict', side_effect=AssertionError('generated object reparsed')), \
             patch.object(workspace, 'validate_trajectory', side_effect=checked):
            current = workspace.generate_checked(self.model, region, self.head, defer_target=True)
        self.assertEqual(current, old)
        self.assertEqual(len(calls), 1)
        self.assertIsNotNone(calls[0])
        self.assertIsNone(planning_state.validate(current, self.model.mesh, region))
        self.assertIsNone(_state())

    def test_final_budget_is_created_at_validation_and_cancellation_clears_scope(self):
        region = self.fixture.region()
        called = []
        def factory():
            called.append('new final budget')
            def stop():
                raise RuntimeError('cancel final validation')
            return stop
        with self.assertRaisesRegex(RuntimeError, 'cancel final validation'):
            workspace.generate_checked(self.model, region, self.head,
                                       validation_check_factory=factory, defer_target=True)
        self.assertEqual(called, ['new final budget'])
        self.assertIsNone(_state())
        plan = workspace.generate_checked(self.model, region, self.head, defer_target=True)
        planning_state.validate(plan, self.model.mesh, region)

    def test_generated_geometry_is_still_checked_against_measured_triangles(self):
        region = self.fixture.region()
        original = continuous_quality.replan
        def moved(*args, **kwargs):
            result = original(*args, **kwargs)
            result.ordered_items[0].points[1]['xyz_m'][2] += .02
            return result
        with patch.object(continuous_quality, 'replan', side_effect=moved), self.assertRaises(ValueError):
            workspace.generate_checked(self.model, region, self.head, defer_target=True)

    def test_nested_mutation_during_final_validation_cannot_inherit_acceptance(self):
        region = self.fixture.region()
        original = continuous_quality.replan
        objects = []
        def captured(*args, **kwargs):
            result = original(*args, **kwargs)
            objects.append(result)
            return result
        def factory():
            calls = [0]
            def mutate():
                calls[0] += 1
                if calls[0] == 2:
                    objects[0].planner['parameters']['nested_mutation_probe'] = 'changed'
            return mutate
        with patch.object(continuous_quality, 'replan', side_effect=captured), \
             self.assertRaisesRegex(ValueError, '变化'):
            workspace.generate_checked(self.model, region, self.head,
                                       validation_check_factory=factory, defer_target=True)
        self.assertIsNone(_state())

    def test_later_validation_rechecks_modified_content_even_with_new_binding(self):
        region = self.fixture.region()
        plan = workspace.generate_checked(self.model, region, self.head, defer_target=True)
        bad = deepcopy(plan)
        bad['segments'][0][1]['xyz_m'][2] += .02
        bad['plan_binding'] = planning_state.binding(self.model.mesh, region, bad)
        with self.assertRaises(ValueError):
            planning_state.validate(bad, self.model.mesh, region)
        self.assertIsNone(planning_state.validate(plan, self.model.mesh, region))

    def test_reference_changed_between_generation_and_validation_is_rejected(self):
        region = self.fixture.region()
        def factory():
            self.model.projection.depth = self.model.projection.depth.copy()
            self.model.projection.depth[self.model.projection.valid] += 1
            return lambda: None
        with self.assertRaisesRegex(ValueError, '输入已变化'):
            workspace.generate_checked(self.model, region, self.head,
                                       validation_check_factory=factory, defer_target=True)

    def test_workspace_width_contract_is_not_replaced_by_point_validation(self):
        region = self.fixture.region()
        original = continuous_quality.replan
        def wrong_width(*args, **kwargs):
            result = original(*args, **kwargs)
            result.planner['parameters']['effective_width_mm'] += 1
            return result
        with patch.object(continuous_quality, 'replan', side_effect=wrong_width), self.assertRaises(ValueError):
            workspace.generate_checked(self.model, region, self.head, defer_target=True)

    def test_last_binding_callback_cannot_change_mesh_or_selection_after_checks(self):
        for what in ('mesh', 'selection'):
            with self.subTest(changed=what):
                region = self.fixture.region()
                vertices = np.asarray(self.model.mesh.vertices).copy()
                calls = []
                original = planning_state.validate_binding
                def binding_with_last_callback(plan, mesh, current_region, check):
                    def last_check():
                        check()
                        calls.append('check')
                        if len(calls) == 2:
                            # validate_binding has already checked the complete
                            # plan binding. This is its final cancellation hook.
                            if what == 'mesh':
                                np.asarray(mesh.vertices)[0, 0] += .001
                            else:
                                current_region['triangle_indices'] = current_region['triangle_indices'][:-1]
                    return original(plan, mesh, current_region, check=last_check)
                try:
                    with patch.object(planning_state, 'validate_binding', side_effect=binding_with_last_callback), \
                         self.assertRaisesRegex(ValueError, '输入已变化'):
                        workspace.generate_checked(self.model, region, self.head, defer_target=True)
                    self.assertEqual(calls, ['check', 'check'])
                finally:
                    np.asarray(self.model.mesh.vertices)[:] = vertices

    def test_last_callback_replacing_item_storage_is_checked_against_fresh_content(self):
        region = self.fixture.region()
        trajectories = []
        replan = continuous_quality.replan
        binding = planning_state.validate_binding
        calls = []
        def capture(*args, **kwargs):
            trajectory = replan(*args, **kwargs)
            trajectories.append(trajectory)
            return trajectory
        def binding_with_mutation(plan, mesh, current_region, check):
            def last_check():
                check()
                calls.append('check')
                if len(calls) == 2:
                    trajectory = trajectories[0]
                    first = trajectory.ordered_items[0]
                    points = deepcopy(first.points)
                    points[0]['xyz_m'][2] += .01
                    # Replace a container rather than modify a shared leaf:
                    # hashing the original to_dict payload would miss this.
                    trajectory.ordered_items[0] = replace(first, points=points)
            return binding(plan, mesh, current_region, check=last_check)
        with patch.object(continuous_quality, 'replan', side_effect=capture), \
             patch.object(planning_state, 'validate_binding', side_effect=binding_with_mutation), \
             self.assertRaisesRegex(ValueError, '路线在生成核验期间发生变化'):
            workspace.generate_checked(self.model, region, self.head, defer_target=True)
        self.assertEqual(calls, ['check', 'check'])


if __name__ == '__main__':
    main()
