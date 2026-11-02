"""Reduced references retain their declared sampling through final save/replay."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import open3d as o3d

from test_fixed_rgb_experience import make_model, K
from test_mesh_slice_planner import grid_mesh
from femto.planning import region_project, surface_workspace
from femto.planning.planning_resolution import create_reduced_model, profile
from femto.planning.planning_state import validate
from femto.planning.surface_trajectory import SurfaceTrajectory, to_player_plan
from femto.planning.treatment_head import head_parameters
from femto.planning.trajectory_export import load_trajectory


class ReducedRouteSamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        mesh = grid_mesh(width=.12, height=.08, cell=.005)
        original = make_model(cls.root, o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(mesh.vertices), o3d.utility.Vector3iVector(mesh.faces)))
        original.view_reference['undistorted_intrinsic'] = K.tolist()
        original.store.origin_reference = dict(kind='initial_color_camera', provenance='synthetic_test')
        o3d.io.write_triangle_mesh(original.store.source, original.mesh)
        cls.model = create_reduced_model(original, cls.root/'reduced', stride=4, route_sample_step_mm=3.)
        mask = cls.model.projection.valid.copy()
        height, width = mask.shape
        cls.region = cls.model.region([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]],
            'synthetic reduced region', '', selection=dict(mask=mask, method='work_only'))
        cls.plan = surface_workspace.generate(cls.model, cls.region,
            head_parameters('direct_row_spacing', row_spacing_mm=7))

    def test_final_route_saves_and_reproduces_declared_sampling(self):
        params = self.plan['surface_trajectory']['planner']['parameters']
        self.assertEqual(params['sample_step_mm'], 3.)
        self.assertTrue(params['preserve_face_crossings'])
        validate(self.plan, self.model.mesh, self.region)
        native = surface_workspace.export(self.model, self.region, self.plan)
        self.assertEqual(to_player_plan(load_trajectory(native, reproduce=True))['segments'], self.plan['segments'])
        portable = region_project.save_project(self.model, self.region, self.root/'result.region.zip', self.plan,
            allow_replace=False, catalog_root=self.root/'catalogue')
        reopened = region_project.open_project(portable, workspace=self.root/'opened')
        self.assertEqual(reopened.saved_plan['segments'], self.plan['segments'])
        self.assertEqual(reopened.view_reference, self.model.view_reference)
        np.testing.assert_array_equal(np.asarray(reopened.mesh.triangles), np.asarray(self.model.mesh.triangles))

    def test_route_parameter_cannot_disagree_with_bound_reference(self):
        forged = deepcopy(self.plan['surface_trajectory'])
        forged['planner']['parameters']['sample_step_mm'] = 5.
        with self.assertRaisesRegex(ValueError, '采样间隔'):
            SurfaceTrajectory.from_dict(forged)

    def test_standard_candidate_keeps_original_frozen_sampling(self):
        from femto.planning.mesh_slice_planner import _parameters
        with self.assertRaisesRegex(ValueError, 'sample_step_mm is frozen'):
            _parameters(dict(sample_step_mm=3.))

    def test_rehashed_reference_cannot_change_existing_route_sampling(self):
        from femto.planning.planning_revision import create, digest
        forged = deepcopy(self.plan['surface_trajectory'])
        context = forged['planning_context']
        inputs = deepcopy(context['revision']['inputs'])
        reduced = inputs['calibration']['view_reference']['derivation']
        reduced.update(route_sample_step_mm=5., profile=profile(4, 5.))
        revised = create(inputs)
        context['revision'] = revised
        target = context['target']
        target.update(computation_sha256=revised['sha256'], target_id=revised['target_id'])
        target['sha256'] = digest({key: value for key, value in target.items() if key != 'sha256'})
        with self.assertRaisesRegex(ValueError, '采样间隔'):
            SurfaceTrajectory.from_dict(forged)


if __name__ == '__main__':
    unittest.main()
