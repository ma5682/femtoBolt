"""Adaptive layout is separately versioned, reproducible, and geometry checked."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
import open3d as o3d
from test_fixed_rgb_experience import make_model,K
from test_mesh_slice_planner import grid_mesh
from femto.planning import planning_revision,surface_workspace as ws
from femto.planning.continuous_quality import UNIFIED_FLOOR_ROUTING,FLOOR_DETOUR_ROUTING
from femto.planning.fixed_width_layout import ADAPTIVE_FLOOR_PATCH_VERSION,validate as validate_layout
from femto.planning.region_project import open_project,save_project
from femto.planning.trajectory_export import load_trajectory
from femto.planning.treatment_head import head_parameters


class AdaptiveFloorLayoutTests(unittest.TestCase):
    def test_layout_binding_geometry_tamper_and_native_portable_replay(self):
        with tempfile.TemporaryDirectory() as folder:
            mesh=grid_mesh(width=.12,height=.08,cell=.005)
            model=make_model(folder,o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),o3d.utility.Vector3iVector(mesh.faces)))
            model.view_reference['undistorted_intrinsic']=K.tolist()
            model.store.origin_reference=dict(kind='initial_color_camera',provenance='synthetic_test')
            o3d.io.write_triangle_mesh(str(model.store.source),model.mesh)
            region=model.region([[30,30],[70,30],[70,70],[30,70]],'test','',selection=dict(mask=model.projection.valid.copy(),method='work_only'))
            head=head_parameters('fixed_width_auto',effective_treatment_width_mm=5,overlap_percent=20)
            for routing,fraction in ((FLOOR_DETOUR_ROUTING,None),(UNIFIED_FLOOR_ROUTING,.65)):
                with self.assertRaises(ValueError):ws.generate(model,region,head,layout_algorithm=ADAPTIVE_FLOOR_PATCH_VERSION,routing_algorithm=routing,minimum_metric_fraction=fraction)
            plan=ws.generate(model,region,head,layout_algorithm=ADAPTIVE_FLOOR_PATCH_VERSION,routing_algorithm=UNIFIED_FLOOR_ROUTING)
            algorithms=plan['planning_context']['revision']['inputs']['algorithms']
            self.assertEqual(algorithms,planning_revision.ADAPTIVE_UNIFIED_FLOOR_ALGORITHMS)
            ws.validate(plan,model.mesh,region)
            params=plan['surface_trajectory']['planner']['parameters'];layout=params['fixed_width_layout']
            self.assertEqual(layout['request']['version'],ADAPTIVE_FLOOR_PATCH_VERSION)
            choices=[p for f in layout['fragments'] for p in f['patches'] if p.get('adaptive_spacing')]
            self.assertTrue(choices)
            for part in choices:
                choice=part['adaptive_spacing'];optim=part['alignment_optimization']
                self.assertGreaterEqual(choice['minimum_fraction'],.65);self.assertLessEqual(choice['minimum_fraction'],.8)
                self.assertEqual(optim['minimum_fraction'],choice['minimum_fraction'])
                if optim.get('metric_step_min_m') is not None:
                    self.assertGreaterEqual(optim['metric_step_min_m'],optim['effective_minimum_metric_step_m']-2e-12)
                    self.assertLessEqual(optim['metric_step_max_m'],head['row_spacing_mm']/1000+2e-12)
            modified=deepcopy(layout);next(p for f in modified['fragments'] for p in f['patches'] if p.get('adaptive_spacing'))['adaptive_spacing']['minimum_fraction']=.99
            with self.assertRaises(ValueError):validate_layout(modified,ws.canonical_region_mesh(model.mesh,region),region,params,lambda:None)
            native=ws.export(model,region,plan)
            self.assertEqual(load_trajectory(native).to_dict()['ordered_items'],plan['surface_trajectory']['ordered_items'])
            archive=save_project(model,region,Path(folder)/'adaptive.region.zip',plan,catalog_root=Path(folder)/'catalog')
            reopened=open_project(archive,workspace=Path(folder)/'opened')
            self.assertEqual(reopened.saved_plan['surface_trajectory']['ordered_items'],plan['surface_trajectory']['ordered_items'])
            self.assertFalse(reopened.saved_plan['execution_ready'])


if __name__=='__main__':unittest.main()
