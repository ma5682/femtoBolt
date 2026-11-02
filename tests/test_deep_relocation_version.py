"""The additional relocation is versioned, replayed, and safely falls back."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import open3d as o3d
from test_fixed_rgb_experience import make_model, K
from test_mesh_slice_planner import grid_mesh
from femto.planning import planning_revision, surface_workspace as workspace, verified_relocation_research as relocation
from femto.planning.continuous_quality import DEEP_RELOCATION_ROUTING, RELOCATION_ROUTING
from femto.planning.fixed_width_layout import OPTIMIZED_PATCH_VERSION
from femto.planning.region_project import save_project, open_project
from femto.planning.treatment_head import head_parameters

class DeepRelocationTests(unittest.TestCase):
    def test_budget_fallback_replay_tamper_and_portable_archive(self):
        with tempfile.TemporaryDirectory() as folder:
            mesh=grid_mesh(width=.12,height=.08,cell=.005)
            model=make_model(folder,o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),
                                                             o3d.utility.Vector3iVector(mesh.faces)))
            model.view_reference['undistorted_intrinsic']=K.tolist()
            model.store.origin_reference=dict(kind='initial_color_camera',provenance='synthetic_test')
            o3d.io.write_triangle_mesh(str(model.store.source),model.mesh)
            region=model.region([[30,30],[70,30],[70,70],[30,70]],'test','',
                                selection=dict(mask=model.projection.valid.copy(),method='work_only'))
            head=head_parameters('fixed_width_auto',effective_treatment_width_mm=5,overlap_percent=20)
            baseline=workspace.generate(model,region,head,layout_algorithm=OPTIMIZED_PATCH_VERSION,routing_algorithm=RELOCATION_ROUTING)
            original=relocation.propose
            def fail_extra(*args,**kwargs):
                if kwargs['attempts']==256:raise ValueError('连续路线整边搜索超过预算，请缩小选区')
                return original(*args,**kwargs)
            with patch.object(relocation,'propose',side_effect=fail_extra):
                plan=workspace.generate(model,region,head,layout_algorithm=OPTIMIZED_PATCH_VERSION,routing_algorithm=DEEP_RELOCATION_ROUTING)
            self.assertEqual(plan['planning_context']['revision']['inputs']['algorithms'],planning_revision.DEEP_RELOCATION_PATCH_LAYOUT_ALGORITHMS)
            deep=plan['surface_trajectory']['gap_report']['reconnection']['ordering']['relocation_deep']
            self.assertEqual(deep['skipped'],'geometry_budget');self.assertFalse(deep['adopted'])
            self.assertEqual(plan['surface_trajectory']['ordered_items'],baseline['surface_trajectory']['ordered_items'])
            self.assertEqual(plan['surface_trajectory']['statistics'],baseline['surface_trajectory']['statistics'])
            workspace.validate(plan,model.mesh,region)
            bad=deepcopy(plan);bad['surface_trajectory']['gap_report']['reconnection']['ordering']['relocation_deep']['skipped']='segment_budget'
            with self.assertRaises(ValueError):workspace.validate(bad,model.mesh,region)
            archive=save_project(model,region,Path(folder)/'deep.region.zip',plan,catalog_root=Path(folder)/'catalog')
            opened=open_project(archive,workspace=Path(folder)/'reopened')
            self.assertEqual(opened.saved_plan['surface_trajectory']['ordered_items'],plan['surface_trajectory']['ordered_items'])
            self.assertFalse(opened.saved_plan['execution_ready'])

if __name__=='__main__':unittest.main()
