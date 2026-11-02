"""Main workflow uses the native continuous planner, with identical archived motion."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import open3d as o3d

from test_fixed_rgb_experience import make_model, K
from test_mesh_slice_planner import grid_mesh
from femto.planning import surface_workspace as workspace
from femto.planning.treatment_head import head_parameters
from femto.planning.planning_state import validate, state, parameters
from femto.planning.trajectory_export import load_trajectory
from femto.planning.surface_trajectory import to_player_plan


class SurfaceWorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        mesh=grid_mesh(width=.12,height=.08,cell=.005)
        legacy=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),
                                        o3d.utility.Vector3iVector(mesh.faces))
        self.model=make_model(self.temp.name,legacy)
        self.model.view_reference['undistorted_intrinsic']=K.tolist()
        self.model.store.origin_reference={'kind':'initial_color_camera','provenance':'synthetic_test'}
        self.head=head_parameters('direct_row_spacing',row_spacing_mm=7)

    def region(self, split=False):
        mask=self.model.projection.valid.copy()
        if split:mask[:,48:53]=False
        return self.model.region([[30,30],[70,30],[70,70],[30,70]],'合成选区','',
            selection={'mask':mask,'method':'work_only'})

    def test_connected_rows_and_archive_keep_exact_motion_without_manual_save(self):
        region=self.region()
        before=copy.deepcopy(region)
        plan=workspace.generate(self.model,region,self.head)
        self.assertEqual(region,before)
        self.assertTrue(self.model.store.dirty)
        self.assertFalse((self.model.reconstruction/'annotations').exists())
        self.assertEqual(len(plan['segments']),1)
        self.assertGreater(plan['surface_trajectory']['gap_report']['quality_constraint']['rejected_edges'],0)
        validate(plan,self.model.mesh,region)
        self.assertTrue(state(region,plan,request=parameters(plan))['can_use'])
        with mock.patch('femto.planning.mesh_slice_planner.plan_mesh_slice',side_effect=AssertionError('replanned on save')):
            folder=workspace.export(self.model,region,plan)
        trajectory=load_trajectory(folder,reproduce=False)
        self.assertEqual(to_player_plan(trajectory)['segments'],plan['segments'])
        self.assertTrue((folder/'treatment_head.json').is_file())
        from femto.planning.trajectory_history import history_rows
        rows=history_rows(self.model.reconstruction)
        self.assertTrue(any(row['source']==folder/'trajectory.json' for row in rows))
        from femto.planning.path_presentation import render_diagram
        self.assertGreater(render_diagram(self.model.rgb,self.model.projection.ids,region,plan).width,0)
        self.assertIn('几何覆盖估算',workspace.report_text(plan,self.model.mesh,region))
        edited=copy.deepcopy(region);edited['triangle_indices']=edited['triangle_indices'][1:]
        self.assertFalse(state(edited,plan)['can_use'])
        bad=copy.deepcopy(plan);bad['segments'][0][0]['xyz_m'][0]+=.01
        with self.assertRaises(ValueError):validate(bad,self.model.mesh,region)

    def test_quality_archive_reproduces_exact_result(self):
        region=self.region()
        plan=workspace.generate(self.model,region,self.head)
        folder=workspace.export(self.model,region,plan)
        self.assertEqual(to_player_plan(load_trajectory(folder))['segments'],plan['segments'])

    def test_new_quality_evidence_invalidates_save_without_changing_region(self):
        region=self.region();before=copy.deepcopy(region)
        plan=workspace.generate(self.model,region,self.head)
        self.model.projection.depth=self.model.projection.depth.copy()
        self.model.projection.depth[self.model.projection.valid]=501
        with self.assertRaisesRegex(ValueError,'质量证据已变化'):
            workspace.export(self.model,region,plan)
        self.assertEqual(region,before)

    def test_width_change_or_forged_completion_cannot_validate(self):
        region=self.region();plan=workspace.generate(self.model,region,self.head)
        from femto.planning.planning_state import binding
        bad=copy.deepcopy(plan)
        bad['treatment_head']=head_parameters('direct_row_spacing',row_spacing_mm=7,effective_treatment_width_mm=10)
        bad['plan_binding']=binding(self.model.mesh,region,bad)
        with self.assertRaisesRegex(ValueError,'覆盖宽度'):validate(bad,self.model.mesh,region)
        bad=copy.deepcopy(plan)
        bad['surface_trajectory']['gap_report']['quality_constraint']['target_completion']='complete'
        bad['plan_binding']=binding(self.model.mesh,region,bad)
        with self.assertRaises(ValueError):validate(bad,self.model.mesh,region)

    def test_mesh_only_reference_pixels_are_retained_as_pending_intent(self):
        from femto.annotation.fixed_rgb_geometry import FixedProjection
        region=self.region();before=copy.deepcopy(region)
        depth=self.model.projection.depth.copy();depth[50,50]=0
        self.model.projection=FixedProjection(self.model.mesh,K,depth)
        plan=workspace.generate(self.model,region,self.head)
        report=plan['surface_trajectory']['gap_report']['quality_constraint']
        self.assertEqual(report['pending_pixels'],1)
        self.assertTrue(report['rescan_required'])
        self.assertNotEqual(plan['surface_trajectory']['statistics']['coverage_status'],'complete')
        self.assertEqual(region,before)

    def test_legacy_v44_archive_remains_exact_and_is_not_upgraded(self):
        from femto.planning.mesh_slice_planner import plan_mesh_slice
        from femto.planning.surface_trajectory import ORDERED_SELECTION_ALGORITHM
        from femto.planning.planning_state import binding,region_signature
        region=self.region()
        manifest=workspace.snapshot(self.model,region,lambda:None)
        trajectory=plan_mesh_slice(manifest,row_spacing_mm=7,connect_rows=True,block_planning=True,
            selection_blocks=True,selection_algorithm=ORDERED_SELECTION_ALGORITHM)
        old=to_player_plan(trajectory)
        old.update(schema=workspace.SCHEMA,region_id=region['annotation_id'],region_signature=region_signature(region),
            coordinate_frame='world_rdf',unit='metre',treatment_head=self.head,surface_trajectory=trajectory.to_dict())
        old['plan_binding']=binding(self.model.mesh,region,old)
        before=copy.deepcopy(old)
        folder=workspace.export(self.model,region,old)
        reopened=load_trajectory(folder)
        self.assertEqual(reopened.planner['algorithm_version'],ORDERED_SELECTION_ALGORITHM)
        self.assertEqual(to_player_plan(reopened)['segments'],old['segments'])
        self.assertEqual(old,before)

    def test_disjoint_work_areas_have_connected_rows_but_no_fabricated_transfer(self):
        region=self.region(split=True)
        plan=workspace.generate(self.model,region,self.head)
        self.assertEqual(plan['blocks']['summary']['block_count'],2)
        self.assertEqual(len(plan['segments']),2)
        self.assertEqual(sum(gap['inter_block'] for gap in plan['chain_breaks']),1)
        self.assertTrue(all(not gap['transition_planned'] for gap in plan['chain_breaks']))
        runs=list(workspace.display_runs(plan))
        self.assertTrue(any(role=='scan' for role,_ in runs))
        self.assertIn('质量与宽度约束',workspace.continuity_text(plan))
        validate(plan,self.model.mesh,region)

    def test_portable_project_keeps_native_route_without_original_snapshot(self):
        from femto.planning.region_project import save_project,open_project
        region=self.region()
        plan=workspace.generate(self.model,region,self.head)
        o3d.io.write_triangle_mesh(self.model.store.source,self.model.mesh)
        target=Path(self.temp.name)/'plan.region.zip'
        save_project(self.model,region,target,plan)
        inputs=self.model.reconstruction/'planning_inputs'
        inputs.rename(inputs.with_name('unavailable_original_inputs'))
        reopened=open_project(target,workspace=Path(self.temp.name)/'reopened')
        self.assertEqual(reopened.saved_plan['segments'],plan['segments'])
        validate(reopened.saved_plan,reopened.mesh,reopened.store.regions[0])
        folder=workspace.export(reopened,reopened.store.regions[0],reopened.saved_plan)
        self.assertEqual(to_player_plan(load_trajectory(folder,reproduce=False))['segments'],plan['segments'])


if __name__=='__main__':unittest.main()
