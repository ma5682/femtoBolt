"""Independent replay of the explicitly requested research layout version."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import open3d as o3d
from test_fixed_rgb_experience import make_model,K
from test_mesh_slice_planner import grid_mesh
from femto.planning import planning_revision,surface_workspace as workspace
from femto.planning.continuous_quality import BRIDGE_REORDER_ROUTING,JOINT_REORDER_ROUTING,RELOCATION_ROUTING
from femto.planning.fixed_width_layout import ADAPTIVE_PATCH_VERSION,MINIMAL_PATCH_VERSION,validate as validate_layout
from femto.planning.region_project import save_project,open_project
from femto.planning.treatment_head import head_parameters


class AdaptiveLayoutResearchTests(unittest.TestCase):
    def test_minimal_research_version_matches_zero_budget_and_roundtrips(self):
        from femto.planning.trajectory_export import load_trajectory
        from femto.planning.surface_trajectory import to_player_plan
        with tempfile.TemporaryDirectory() as folder:
            mesh=grid_mesh(width=.12,height=.08,cell=.005)
            model=make_model(folder,o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),o3d.utility.Vector3iVector(mesh.faces)))
            model.view_reference['undistorted_intrinsic']=K.tolist()
            model.store.origin_reference=dict(kind='initial_color_camera',provenance='synthetic_test')
            o3d.io.write_triangle_mesh(str(model.store.source),model.mesh)
            region=model.region([[30,30],[70,30],[70,70],[30,70]],'test','',selection=dict(mask=model.projection.valid.copy(),method='work_only'))
            head=head_parameters('fixed_width_auto',effective_treatment_width_mm=5,overlap_percent=20)
            with self.assertRaisesRegex(ValueError,'移动换序'):
                workspace.generate(model,region,head,layout_algorithm=MINIMAL_PATCH_VERSION,routing_algorithm=BRIDGE_REORDER_ROUTING)
            with patch('femto.planning.patch_slice_layout.OPTIMIZED_EXTRA_FRACTION',0.):
                expected=workspace.generate(model,region,head,layout_algorithm=ADAPTIVE_PATCH_VERSION,routing_algorithm=RELOCATION_ROUTING)
            plan=workspace.generate(model,region,head,layout_algorithm=MINIMAL_PATCH_VERSION,routing_algorithm=RELOCATION_ROUTING)
            self.assertEqual(plan['planning_context']['revision']['inputs']['algorithms'],planning_revision.RESEARCH_MINIMAL_RELOCATION_ALGORITHMS)
            self.assertEqual(plan['surface_trajectory']['ordered_items'],expected['surface_trajectory']['ordered_items'])
            self.assertEqual(plan['surface_trajectory']['statistics'],expected['surface_trajectory']['statistics'])
            trajectory=workspace.validate(plan,model.mesh,region)
            self.assertEqual(trajectory.planner['parameters']['fixed_width_layout']['request']['effective_width_mm'],5)
            self.assertFalse(plan['execution_ready'])
            native=workspace.export(model,region,plan)
            self.assertEqual(to_player_plan(load_trajectory(native,reproduce=True))['segments'],plan['segments'])
            path=save_project(model,region,Path(folder)/'minimal.region.zip',plan,catalog_root=Path(folder)/'catalog')
            reopened=open_project(path,workspace=Path(folder)/'opened')
            self.assertEqual(reopened.saved_plan['segments'],plan['segments'])
            self.assertEqual(reopened.saved_plan['planning_context']['revision']['inputs']['algorithms'],planning_revision.RESEARCH_MINIMAL_RELOCATION_ALGORITHMS)
            direct=workspace.generate(model,region,head,layout_algorithm=MINIMAL_PATCH_VERSION,
                routing_algorithm=RELOCATION_ROUTING,target_algorithm=planning_revision.EXACT_TARGET)
            expected_algorithms=dict(planning_revision.RESEARCH_MINIMAL_RELOCATION_ALGORITHMS,target=planning_revision.EXACT_TARGET)
            self.assertEqual(direct['planning_context']['revision']['inputs']['algorithms'],expected_algorithms)
            self.assertEqual(direct['surface_trajectory']['ordered_items'],plan['surface_trajectory']['ordered_items'])
            workspace.validate(direct,model.mesh,region)

    def test_adaptive_layout_with_relocation_has_independent_revision_and_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            mesh=grid_mesh(width=.12,height=.08,cell=.005)
            model=make_model(folder,o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),o3d.utility.Vector3iVector(mesh.faces)))
            model.view_reference['undistorted_intrinsic']=K.tolist()
            model.store.origin_reference=dict(kind='initial_color_camera',provenance='synthetic_test')
            o3d.io.write_triangle_mesh(str(model.store.source),model.mesh)
            region=model.region([[30,30],[70,30],[70,70],[30,70]],'test','',selection=dict(mask=model.projection.valid.copy(),method='work_only'))
            head=head_parameters('fixed_width_auto',effective_treatment_width_mm=5,overlap_percent=20)
            plan=workspace.generate(model,region,head,layout_algorithm=ADAPTIVE_PATCH_VERSION,routing_algorithm=RELOCATION_ROUTING)
            self.assertEqual(plan['planning_context']['revision']['inputs']['algorithms'],planning_revision.RESEARCH_ADAPTIVE_RELOCATION_ALGORITHMS)
            trajectory=workspace.validate(plan,model.mesh,region)
            self.assertEqual(trajectory.planner['parameters']['fixed_width_layout']['request']['effective_width_mm'],5)
            self.assertEqual(trajectory.planner['parameters']['continuous_routing'],RELOCATION_ROUTING)
            self.assertFalse(plan['execution_ready'])
            path=save_project(model,region,Path(folder)/'adaptive_relocation.region.zip',plan,catalog_root=Path(folder)/'catalog')
            reopened=open_project(path,workspace=Path(folder)/'opened')
            self.assertEqual(reopened.saved_plan['segments'],plan['segments'])
            self.assertEqual(reopened.saved_plan['planning_context']['revision']['inputs']['algorithms'],planning_revision.RESEARCH_ADAPTIVE_RELOCATION_ALGORITHMS)
            expected=dict(planning_revision.RESEARCH_ADAPTIVE_RELOCATION_ALGORITHMS,target=planning_revision.EXACT_TARGET)
            self.assertTrue(planning_revision.supported_target_algorithms(expected))
            direct=workspace.generate(model,region,head,layout_algorithm=ADAPTIVE_PATCH_VERSION,
                routing_algorithm=RELOCATION_ROUTING,target_algorithm=planning_revision.EXACT_TARGET)
            self.assertEqual(direct['planning_context']['revision']['inputs']['algorithms'],expected)
            self.assertEqual(direct['surface_trajectory']['ordered_items'],plan['surface_trajectory']['ordered_items'])
            workspace.validate(direct,model.mesh,region)

    def test_direct_exact_target_matches_reconstruction_and_roundtrips(self):
        from femto.planning import target_reconstruction,target_surface,target_continuation
        with tempfile.TemporaryDirectory() as folder:
            mesh=grid_mesh(width=.12,height=.08,cell=.005)
            model=make_model(folder,o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),o3d.utility.Vector3iVector(mesh.faces)))
            model.view_reference['undistorted_intrinsic']=K.tolist()
            model.store.origin_reference=dict(kind='initial_color_camera',provenance='synthetic_test')
            o3d.io.write_triangle_mesh(str(model.store.source),model.mesh)
            region=model.region([[30,30],[70,30],[70,70],[30,70]],'test','',selection=dict(mask=model.projection.valid.copy(),method='work_only'))
            head=head_parameters('fixed_width_auto',effective_treatment_width_mm=5,overlap_percent=20)
            with self.assertRaisesRegex(ValueError,'仅用于指定的研究排布'):
                workspace.generate(model,region,head,target_algorithm=planning_revision.EXACT_TARGET)
            with self.assertRaisesRegex(ValueError,'仅用于指定的研究排布'):
                workspace.generate(model,region,head,layout_algorithm=ADAPTIVE_PATCH_VERSION,
                    routing_algorithm=BRIDGE_REORDER_ROUTING,target_algorithm='unknown')
            legacy=workspace.generate(model,region,head,layout_algorithm=ADAPTIVE_PATCH_VERSION,routing_algorithm=BRIDGE_REORDER_ROUTING)
            _,rebuilt=target_reconstruction.calculate(model,region,legacy,Path(folder)/'rebuilt')
            direct=workspace.generate(model,region,head,layout_algorithm=ADAPTIVE_PATCH_VERSION,
                routing_algorithm=BRIDGE_REORDER_ROUTING,target_algorithm=planning_revision.EXACT_TARGET)
            revision=direct['planning_context']['revision']
            self.assertEqual(revision['inputs']['algorithms']['target'],planning_revision.EXACT_TARGET)
            self.assertEqual(direct['surface_trajectory']['ordered_items'],legacy['surface_trajectory']['ordered_items'])
            actual=workspace.target_for(model,direct,reproduce=True)
            self.assertEqual(actual.summary,rebuilt.summary)
            self.assertEqual(actual.metadata,rebuilt.metadata)
            for name in actual.arrays:np.testing.assert_array_equal(actual.arrays[name],rebuilt.arrays[name])
            workspace.validate(direct,model.mesh,region)
            path=save_project(model,region,Path(folder)/'direct.region.zip',direct,catalog_root=Path(folder)/'direct_catalog')
            reopened=open_project(path,workspace=Path(folder)/'direct_opened')
            self.assertEqual(reopened.saved_plan['segments'],direct['segments'])
            self.assertEqual(reopened.saved_plan['planning_context']['target'],actual.metadata)
            self.assertFalse(direct['execution_ready'])
            original_build=target_surface.build
            def bounded_build(*args,**kwargs):
                kwargs['budget']=target_surface.TargetBudget(max_units=8)
                return original_build(*args,**kwargs)
            with patch.object(target_surface,'build',side_effect=bounded_build):
                partial=workspace.generate(model,region,head,layout_algorithm=ADAPTIVE_PATCH_VERSION,
                    routing_algorithm=BRIDGE_REORDER_ROUTING,target_algorithm=planning_revision.EXACT_TARGET)
            self.assertEqual(partial['planning_context']['target']['metrics']['stop_reason'],'unit_budget')
            self.assertGreater(partial['planning_context']['target']['summary']['unknown_by_reason']['budget_unprocessed'],0)
            self.assertTrue(target_continuation.available(partial))
            workspace.target_for(model,partial,reproduce=True)

    def test_explicit_version_independent_layout_replay_and_portable_archive(self):
        with tempfile.TemporaryDirectory() as folder:
            mesh=grid_mesh(width=.12,height=.08,cell=.005)
            model=make_model(folder,o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),o3d.utility.Vector3iVector(mesh.faces)))
            model.view_reference['undistorted_intrinsic']=K.tolist()
            model.store.origin_reference=dict(kind='initial_color_camera',provenance='synthetic_test')
            o3d.io.write_triangle_mesh(str(model.store.source),model.mesh)
            region=model.region([[30,30],[70,30],[70,70],[30,70]],'test','',selection=dict(mask=model.projection.valid.copy(),method='work_only'))
            head=head_parameters('fixed_width_auto',effective_treatment_width_mm=5,overlap_percent=20)
            with self.assertRaisesRegex(ValueError,'两步联合排序'):
                workspace.generate(model,region,head,layout_algorithm=ADAPTIVE_PATCH_VERSION,routing_algorithm=JOINT_REORDER_ROUTING)
            plan=workspace.generate(model,region,head,layout_algorithm=ADAPTIVE_PATCH_VERSION,routing_algorithm=BRIDGE_REORDER_ROUTING)
            self.assertEqual(plan['planning_context']['revision']['inputs']['algorithms'],planning_revision.RESEARCH_ADAPTIVE_PATCH_ALGORITHMS)
            trajectory=workspace.validate(plan,model.mesh,region)
            params=trajectory.planner['parameters'];layout=params['fixed_width_layout']
            self.assertEqual(layout['request']['effective_width_mm'],5)
            decisions=[d for fragment in layout['fragments'] for d in fragment['adaptive_cut_proposals']]
            self.assertTrue(decisions);self.assertFalse(plan['execution_ready'])
            bad=deepcopy(layout);bad['fragments'][0]['adaptive_cut_proposals'][0]['adopted']=True
            canonical=workspace.canonical_region_mesh(model.mesh,region)
            with self.assertRaisesRegex(ValueError,'独立重算'):
                validate_layout(bad,canonical,region,params,lambda:None)
            portable=save_project(model,region,Path(folder)/'adaptive.region.zip',plan,catalog_root=Path(folder)/'catalog')
            reopened=open_project(portable,workspace=Path(folder)/'opened')
            self.assertEqual(reopened.saved_plan['segments'],plan['segments'])
            self.assertEqual(reopened.saved_plan['planning_context']['revision']['inputs']['algorithms'],planning_revision.RESEARCH_ADAPTIVE_PATCH_ALGORITHMS)


if __name__=='__main__':unittest.main()
