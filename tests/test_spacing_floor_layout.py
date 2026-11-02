"""Independent geometry replay and both persistence formats for v9 layout."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
import open3d as o3d
from test_fixed_rgb_experience import make_model,K
from test_mesh_slice_planner import grid_mesh
from femto.planning import planning_revision,surface_workspace as workspace
from femto.planning.continuous_quality import DEEP_RELOCATION_ROUTING,RELOCATION_ROUTING
from femto.planning.fixed_width_layout import FLOOR_PATCH_VERSION,OPTIMIZED_PATCH_VERSION,request
from femto.planning.region_project import open_project,save_project
from femto.planning.trajectory_export import load_trajectory
from femto.planning.treatment_head import head_parameters


class SpacingFloorLayoutTests(unittest.TestCase):
    def test_explicit_floor_bounds_and_revision_registration(self):
        for value in (.6,.65,.7,.75):
            spec=request(5,20,version=FLOOR_PATCH_VERSION,minimum_metric_fraction=value)
            self.assertEqual(spec['minimum_metric_fraction'],value)
            self.assertTrue(planning_revision.supported_target_algorithms(dict(planning_revision.FLOOR_DEEP_RELOCATION_ALGORITHMS,minimum_metric_fraction=value)))
        for value in (True,.49,1.01,float('nan')):
            with self.subTest(value=value),self.assertRaises(ValueError):request(5,20,version=FLOOR_PATCH_VERSION,minimum_metric_fraction=value)
            self.assertFalse(planning_revision.supported_target_algorithms(dict(planning_revision.FLOOR_DEEP_RELOCATION_ALGORITHMS,minimum_metric_fraction=value)))
        with self.assertRaises(ValueError):request(5,20,version=OPTIMIZED_PATCH_VERSION,minimum_metric_fraction=.65)

    def test_registered_layout_geometry_tamper_and_roundtrips(self):
        with tempfile.TemporaryDirectory() as folder:
            mesh=grid_mesh(width=.12,height=.08,cell=.005)
            model=make_model(folder,o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),o3d.utility.Vector3iVector(mesh.faces)))
            model.view_reference['undistorted_intrinsic']=K.tolist()
            model.store.origin_reference=dict(kind='initial_color_camera',provenance='synthetic_test')
            o3d.io.write_triangle_mesh(str(model.store.source),model.mesh)
            region=model.region([[30,30],[70,30],[70,70],[30,70]],'test','',selection=dict(mask=model.projection.valid.copy(),method='work_only'))
            head=head_parameters('fixed_width_auto',effective_treatment_width_mm=5,overlap_percent=20)
            with self.assertRaisesRegex(ValueError,'深度换序'):
                workspace.generate(model,region,head,layout_algorithm=FLOOR_PATCH_VERSION,routing_algorithm=RELOCATION_ROUTING)
            plan=workspace.generate(model,region,head,layout_algorithm=FLOOR_PATCH_VERSION,routing_algorithm=DEEP_RELOCATION_ROUTING)
            self.assertEqual(plan['planning_context']['revision']['inputs']['algorithms'],planning_revision.FLOOR_DEEP_RELOCATION_ALGORITHMS)
            workspace.validate(plan,model.mesh,region)
            decisions=[p['alignment_optimization'] for f in plan['surface_trajectory']['planner']['parameters']['fixed_width_layout']['fragments']
                       for p in f['patches'] if p.get('alignment_optimization')]
            self.assertTrue(decisions)
            for decision in decisions:
                self.assertEqual(decision['minimum_fraction'],.75)
                if decision.get('metric_step_min_m') is not None:
                    self.assertGreaterEqual(decision['metric_step_min_m'],decision['effective_minimum_metric_step_m']-2e-12)
            broken=deepcopy(plan)
            fragment=next(f for f in broken['surface_trajectory']['planner']['parameters']['fixed_width_layout']['fragments']
                          if any(p.get('alignment_optimization') for p in f['patches']))
            patch=next(p for p in fragment['patches'] if p.get('alignment_optimization'))
            patch['alignment_optimization']['minimum_fraction']=.5
            with self.assertRaises(ValueError):workspace.validate(broken,model.mesh,region)
            native=workspace.export(model,region,plan)
            self.assertEqual(load_trajectory(native).to_dict()['ordered_items'],plan['surface_trajectory']['ordered_items'])
            archive=save_project(model,region,Path(folder)/'floor.region.zip',plan,catalog_root=Path(folder)/'catalog')
            reopened=open_project(archive,workspace=Path(folder)/'reopened')
            self.assertEqual(reopened.saved_plan['surface_trajectory']['ordered_items'],plan['surface_trajectory']['ordered_items'])
            self.assertFalse(reopened.saved_plan['execution_ready'])
            tuned=workspace.generate(model,region,head,layout_algorithm=FLOOR_PATCH_VERSION,routing_algorithm=DEEP_RELOCATION_ROUTING,minimum_metric_fraction=.65)
            self.assertNotEqual(tuned['planning_context']['revision']['sha256'],plan['planning_context']['revision']['sha256'])
            self.assertEqual(tuned['planning_context']['revision']['target_id'],plan['planning_context']['revision']['target_id'])
            self.assertEqual(tuned['planning_context']['revision']['inputs']['algorithms']['minimum_metric_fraction'],.65)
            broken=deepcopy(tuned);broken['surface_trajectory']['planner']['parameters']['fixed_width_layout']['request']['minimum_metric_fraction']=.7
            with self.assertRaisesRegex(ValueError,'计算修订'):workspace.validate(broken,model.mesh,region)
            tuned_native=workspace.export(model,region,tuned)
            self.assertEqual(load_trajectory(tuned_native).to_dict()['ordered_items'],tuned['surface_trajectory']['ordered_items'])


if __name__=='__main__':unittest.main()
