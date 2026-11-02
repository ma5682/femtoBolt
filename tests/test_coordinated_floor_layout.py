"""Seam direction coordination is versioned separately from adaptive v10."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import open3d as o3d
from test_fixed_rgb_experience import make_model,K
from test_mesh_slice_planner import grid_mesh
from femto.planning import planning_revision,surface_workspace as ws
from femto.planning.continuous_quality import UNIFIED_FLOOR_ROUTING
from femto.planning.fixed_width_layout import COORDINATED_FLOOR_PATCH_VERSION,ADAPTIVE_FLOOR_PATCH_VERSION,validate as validate_layout
from femto.planning.region_project import open_project,save_project
from femto.planning.trajectory_export import load_trajectory
from femto.planning.treatment_head import head_parameters


class CoordinatedFloorLayoutTests(unittest.TestCase):
    def test_shared_fold_can_align_direction_with_checked_geometry_and_no_new_gaps(self):
        with tempfile.TemporaryDirectory() as folder:
            mesh=grid_mesh(width=.04,height=.02,cell=.0025);vertices=np.asarray(mesh.vertices).copy()
            tilt=np.tan(np.radians(40));right=vertices[:,0]>0
            vertices[right,1]+=.1*vertices[right,0];vertices[right,2]+=tilt*vertices[right,0];mesh.vertices=vertices
            legacy=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),o3d.utility.Vector3iVector(mesh.faces))
            ray=(np.arange(100)-K[0,2])/K[0,0];z=np.where(ray>0,.5/(1-tilt*ray),.5)
            model=make_model(folder,legacy,depth=np.tile(np.rint(z*1000).astype(np.uint16),(100,1)))
            model.view_reference['undistorted_intrinsic']=K.tolist()
            model.store.origin_reference=dict(kind='initial_color_camera',provenance='synthetic_test')
            o3d.io.write_triangle_mesh(str(model.store.source),model.mesh)
            region=model.region([[30,30],[70,30],[70,70],[30,70]],'test','',selection=dict(mask=model.projection.valid.copy(),method='work_only'))
            head=head_parameters('fixed_width_auto',effective_treatment_width_mm=3.5,overlap_percent=20)
            baseline=ws.generate(model,region,head,layout_algorithm=ADAPTIVE_FLOOR_PATCH_VERSION,routing_algorithm=UNIFIED_FLOOR_ROUTING)
            plan=ws.generate(model,region,head,layout_algorithm=COORDINATED_FLOOR_PATCH_VERSION,routing_algorithm=UNIFIED_FLOOR_ROUTING)
            fresh=ws.validate(plan,model.mesh,region)
            patches=[p for f in fresh.planner['parameters']['fixed_width_layout']['fragments'] for p in f['patches']]
            accepted=[p for p in patches if p.get('seam_direction',{}).get('adopted')]
            self.assertTrue(accepted)
            self.assertGreaterEqual(accepted[0]['seam_direction']['after_alignment'],np.cos(np.radians(10)))
            self.assertLessEqual(len(accepted[0]['positions_m']),accepted[0]['seam_direction']['baseline_rows'])
            self.assertLessEqual(fresh.statistics['gap_count'],baseline['surface_trajectory']['statistics']['gap_count'])
            self.assertEqual(plan['treatment_head'],baseline['treatment_head'])
            self.assertEqual(plan['planning_context']['revision']['target_id'],baseline['planning_context']['revision']['target_id'])

    def test_accepted_direction_cannot_add_rows_after_adaptive_alignment(self):
        from femto.planning import coordinated_seam_research as direction
        with tempfile.TemporaryDirectory() as folder:
            mesh=grid_mesh(width=.12,height=.08,cell=.005)
            model=make_model(folder,o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),o3d.utility.Vector3iVector(mesh.faces)))
            model.view_reference['undistorted_intrinsic']=K.tolist()
            model.store.origin_reference=dict(kind='initial_color_camera',provenance='synthetic_test')
            o3d.io.write_triangle_mesh(str(model.store.source),model.mesh)
            region=model.region([[30,30],[70,30],[70,70],[30,70]],'test','',selection=dict(mask=model.projection.valid.copy(),method='work_only'))
            head=head_parameters('fixed_width_auto',effective_treatment_width_mm=5,overlap_percent=20)
            def valid_candidate(vertices,faces,patch,frame,previous,check):
                return deepcopy(frame),dict(version=direction.VERSION,semantics='test_known_valid_direction')
            with patch.object(direction,'propose',side_effect=valid_candidate):
                plan=ws.generate(model,region,head,layout_algorithm=COORDINATED_FLOOR_PATCH_VERSION,routing_algorithm=UNIFIED_FLOOR_ROUTING)
                ws.validate(plan,model.mesh,region)
            layout=plan['surface_trajectory']['planner']['parameters']['fixed_width_layout']
            adopted=[p for f in layout['fragments'] for p in f['patches'] if p.get('seam_direction',{}).get('adopted')]
            self.assertTrue(adopted)
            for patch_record in adopted:
                self.assertLessEqual(len(patch_record['positions_m']),patch_record['seam_direction']['baseline_rows'])
                self.assertEqual(patch_record['alignment_optimization']['extra_fraction'],0.)

    def test_version_geometry_decision_tamper_and_replay(self):
        with tempfile.TemporaryDirectory() as folder:
            mesh=grid_mesh(width=.12,height=.08,cell=.005)
            model=make_model(folder,o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),o3d.utility.Vector3iVector(mesh.faces)))
            model.view_reference['undistorted_intrinsic']=K.tolist()
            model.store.origin_reference=dict(kind='initial_color_camera',provenance='synthetic_test')
            o3d.io.write_triangle_mesh(str(model.store.source),model.mesh)
            region=model.region([[30,30],[70,30],[70,70],[30,70]],'test','',selection=dict(mask=model.projection.valid.copy(),method='work_only'))
            head=head_parameters('fixed_width_auto',effective_treatment_width_mm=5,overlap_percent=20)
            plan=ws.generate(model,region,head,layout_algorithm=COORDINATED_FLOOR_PATCH_VERSION,routing_algorithm=UNIFIED_FLOOR_ROUTING)
            self.assertEqual(plan['planning_context']['revision']['inputs']['algorithms'],planning_revision.COORDINATED_UNIFIED_FLOOR_ALGORITHMS)
            ws.validate(plan,model.mesh,region)
            params=plan['surface_trajectory']['planner']['parameters'];layout=params['fixed_width_layout']
            choices=[p for f in layout['fragments'] for p in f['patches'] if p.get('seam_direction')]
            self.assertTrue(choices)
            for patch in choices:
                self.assertEqual(patch['seam_direction']['version'],'research-shared-seam-direction-proposal-v1')
                self.assertIn('adaptive_spacing',patch);self.assertIn('adaptive_seams',patch)
            broken=deepcopy(layout);next(p for f in broken['fragments'] for p in f['patches'] if p.get('seam_direction'))['seam_direction']['adopted']=True
            with self.assertRaises(ValueError):validate_layout(broken,ws.canonical_region_mesh(model.mesh,region),region,params,lambda:None)
            native=ws.export(model,region,plan)
            self.assertEqual(load_trajectory(native).to_dict()['ordered_items'],plan['surface_trajectory']['ordered_items'])
            archive=save_project(model,region,Path(folder)/'coordinated.region.zip',plan,catalog_root=Path(folder)/'catalog')
            reopened=open_project(archive,workspace=Path(folder)/'opened')
            self.assertEqual(reopened.saved_plan['surface_trajectory']['ordered_items'],plan['surface_trajectory']['ordered_items'])
            self.assertFalse(reopened.saved_plan['execution_ready'])


if __name__=='__main__':unittest.main()
