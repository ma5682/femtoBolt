"""The bounded floor mode is reproducible and isolated from legacy routing."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import open3d as o3d
from test_fixed_rgb_experience import make_model,K
from test_mesh_slice_planner import grid_mesh
from femto.planning import planning_revision,surface_workspace as ws,verified_relocation_research as relocation
from femto.planning.continuous_quality import FLOOR_DETOUR_ROUTING,DEEP_RELOCATION_ROUTING
from femto.planning.fixed_width_layout import FLOOR_PATCH_VERSION,OPTIMIZED_PATCH_VERSION
from femto.planning.region_project import open_project,save_project
from femto.planning.trajectory_export import load_trajectory
from femto.planning.treatment_head import head_parameters


class FloorDetourTests(unittest.TestCase):
    def test_geometry_fallback_binding_and_both_archives(self):
        with tempfile.TemporaryDirectory() as folder:
            mesh=grid_mesh(width=.12,height=.08,cell=.005)
            model=make_model(folder,o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),o3d.utility.Vector3iVector(mesh.faces)))
            model.view_reference['undistorted_intrinsic']=K.tolist()
            model.store.origin_reference=dict(kind='initial_color_camera',provenance='synthetic_test')
            o3d.io.write_triangle_mesh(str(model.store.source),model.mesh)
            region=model.region([[30,30],[70,30],[70,70],[30,70]],'test','',selection=dict(mask=model.projection.valid.copy(),method='work_only'))
            head=head_parameters('fixed_width_auto',effective_treatment_width_mm=5,overlap_percent=20)
            for layout,fraction in ((OPTIMIZED_PATCH_VERSION,None),(FLOOR_PATCH_VERSION,.7)):
                with self.assertRaises(ValueError):ws.generate(model,region,head,layout_algorithm=layout,routing_algorithm=FLOOR_DETOUR_ROUTING,minimum_metric_fraction=fraction)
            plan=ws.generate(model,region,head,layout_algorithm=FLOOR_PATCH_VERSION,routing_algorithm=FLOOR_DETOUR_ROUTING)
            self.assertEqual(plan['planning_context']['revision']['inputs']['algorithms'],planning_revision.BOUNDED_FLOOR_ALGORITHMS)
            ws.validate(plan,model.mesh,region)
            record=plan['surface_trajectory']['gap_report']['reconnection']['ordering']['relocation_floor']
            self.assertEqual(record['version'],relocation.LIMITED_DETOUR_VERSION)
            for mutation in ('missing','limits','routing','fraction'):
                broken=deepcopy(plan);params=broken['surface_trajectory']['planner']['parameters']
                stage=broken['surface_trajectory']['gap_report']['reconnection']['ordering']
                if mutation=='missing':stage.pop('relocation_floor')
                elif mutation=='limits':stage['relocation_floor']['motion_limits']['connector_length_m']+=.1
                elif mutation=='routing':params['continuous_routing']=DEEP_RELOCATION_ROUTING
                else:params['fixed_width_layout']['request']['minimum_metric_fraction']=.7
                with self.subTest(mutation=mutation),self.assertRaises(ValueError):ws.validate(broken,model.mesh,region)
            native=ws.export(model,region,plan)
            self.assertEqual(load_trajectory(native).to_dict()['ordered_items'],plan['surface_trajectory']['ordered_items'])
            archive=save_project(model,region,Path(folder)/'floor-detour.region.zip',plan,catalog_root=Path(folder)/'catalog')
            reopened=open_project(archive,workspace=Path(folder)/'opened')
            self.assertEqual(reopened.saved_plan['surface_trajectory']['ordered_items'],plan['surface_trajectory']['ordered_items'])
            self.assertFalse(reopened.saved_plan['execution_ready'])
            original=relocation.propose
            def bounded(*args,**kwargs):
                if kwargs.get('objective_policy')=='detour_first_with_limits':
                    raise ValueError('连续路线整边搜索超过预算，请缩小选区')
                return original(*args,**kwargs)
            with patch.object(relocation,'propose',side_effect=bounded):
                fallback=ws.generate(model,region,head,layout_algorithm=FLOOR_PATCH_VERSION,routing_algorithm=FLOOR_DETOUR_ROUTING)
            stage=fallback['surface_trajectory']['gap_report']['reconnection']['ordering']['relocation_floor']
            self.assertEqual(stage['skipped'],'geometry_budget');self.assertFalse(stage['adopted'])
            ws.validate(fallback,model.mesh,region)


if __name__=='__main__':unittest.main()
