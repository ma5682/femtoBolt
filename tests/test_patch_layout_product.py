"""Full candidate, clipping, graph, persistence-contract checks for local patches."""
from copy import deepcopy
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import open3d as o3d
from test_fixed_rgb_experience import make_model,K
from test_mesh_slice_planner import grid_mesh
from femto.planning import surface_workspace as workspace,planning_revision
from femto.planning.treatment_head import head_parameters
from femto.planning.fixed_width_layout import PATCH_VERSION,STABLE_PATCH_VERSION,REPAIR_PATCH_VERSION,OPTIMIZED_PATCH_VERSION,validate as validate_layout


class PatchProductTests(unittest.TestCase):
    def test_relocation_budget_fallback_retains_verified_v10_route(self):
        from femto.planning.continuous_quality import RELOCATION_ROUTING,BRIDGE_REORDER_ROUTING
        with tempfile.TemporaryDirectory() as folder:
            mesh=grid_mesh(width=.12,height=.08,cell=.005)
            model=make_model(folder,o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),
                             o3d.utility.Vector3iVector(mesh.faces)))
            model.view_reference['undistorted_intrinsic']=K.tolist()
            model.store.origin_reference=dict(kind='initial_color_camera',provenance='synthetic_test')
            region=model.region([[30,30],[70,30],[70,70],[30,70]],'test','',
                                selection=dict(mask=model.projection.valid.copy(),method='work_only'))
            head=head_parameters('fixed_width_auto',effective_treatment_width_mm=5,overlap_percent=20)
            old=workspace.generate(model,region,head,layout_algorithm=OPTIMIZED_PATCH_VERSION,routing_algorithm=BRIDGE_REORDER_ROUTING)
            with patch('femto.planning.verified_relocation_research.propose',
                       side_effect=ValueError('连续路线整边搜索超过预算，请缩小选区')):
                plan=workspace.generate(model,region,head,layout_algorithm=OPTIMIZED_PATCH_VERSION,routing_algorithm=RELOCATION_ROUTING)
            decision=plan['surface_trajectory']['gap_report']['reconnection']['ordering']['relocation']
            self.assertEqual(decision['skipped'],'geometry_budget');self.assertFalse(decision['adopted'])
            self.assertEqual(plan['surface_trajectory']['ordered_items'],old['surface_trajectory']['ordered_items'])
            self.assertEqual(plan['surface_trajectory']['statistics'],old['surface_trajectory']['statistics'])
            workspace.validate(plan,model.mesh,region)

    def test_relocation_version_rechecks_its_record_and_roundtrips(self):
        from pathlib import Path
        from femto.planning.continuous_quality import RELOCATION_ROUTING
        from femto.planning.region_project import save_project,open_project
        from femto.planning.trajectory_export import load_trajectory
        from femto.planning.surface_trajectory import to_player_plan
        from femto.planning.verified_relocation_research import VERSION
        with tempfile.TemporaryDirectory() as folder:
            mesh=grid_mesh(width=.12,height=.08,cell=.005)
            model=make_model(folder,o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),
                             o3d.utility.Vector3iVector(mesh.faces)))
            model.view_reference['undistorted_intrinsic']=K.tolist()
            model.store.origin_reference=dict(kind='initial_color_camera',provenance='synthetic_test')
            o3d.io.write_triangle_mesh(str(model.store.source),model.mesh)
            yy,xx=np.indices((100,100));mask=model.projection.valid.copy();rng=np.random.default_rng(6)
            for _ in range(4):
                cx,cy=rng.integers(42,59),rng.integers(42,59);rx,ry=rng.integers(2,6),rng.integers(2,6)
                mask &= ~((abs(xx-cx)<=rx)&(abs(yy-cy)<=ry))
            region=model.region([[25,25],[75,25],[75,75],[25,75]],'test','',selection=dict(mask=mask,method='work_only'))
            head=head_parameters('fixed_width_auto',effective_treatment_width_mm=5,overlap_percent=20)
            plan=workspace.generate(model,region,head,layout_algorithm=OPTIMIZED_PATCH_VERSION,routing_algorithm=RELOCATION_ROUTING)
            self.assertEqual(plan['planning_context']['revision']['inputs']['algorithms'],planning_revision.RELOCATION_PATCH_LAYOUT_ALGORITHMS)
            decision=plan['surface_trajectory']['gap_report']['reconnection']['ordering']['relocation']
            self.assertEqual(decision['version'],VERSION)
            workspace.validate(plan,model.mesh,region)
            bad=deepcopy(plan);bad['surface_trajectory']['gap_report']['reconnection']['ordering']['relocation']['version']='bad'
            with self.assertRaises(ValueError):workspace.validate(bad,model.mesh,region)
            bad=deepcopy(plan);bad['surface_trajectory']['gap_report']['reconnection']['ordering']['relocation']['baseline_visits'][0][1]=True
            with self.assertRaises(ValueError):workspace.validate(bad,model.mesh,region)
            if 'motion_limits' in decision:
                bad=deepcopy(plan)
                bad['surface_trajectory']['gap_report']['reconnection']['ordering']['relocation']['motion_limits']['connector_length_m']+=1
                with self.assertRaises(ValueError):workspace.validate(bad,model.mesh,region)
            native=workspace.export(model,region,plan)
            self.assertEqual(to_player_plan(load_trajectory(native,reproduce=True))['segments'],plan['segments'])
            portable=save_project(model,region,Path(folder)/'relocation.region.zip',plan,catalog_root=Path(folder)/'catalog')
            reopened=open_project(portable,workspace=Path(folder)/'reopened')
            self.assertEqual(reopened.saved_plan['segments'],plan['segments'])
            self.assertFalse(plan['execution_ready'])

    def test_bridge_reorder_adoption_survives_repair_and_archive(self):
        from pathlib import Path
        from femto.planning.continuous_quality import BRIDGE_REORDER_ROUTING
        from femto.planning.region_project import save_project,open_project
        from femto.planning.trajectory_export import load_trajectory
        from femto.planning.surface_trajectory import to_player_plan
        from femto.planning.bridge_joint_incremental import VERSION
        with tempfile.TemporaryDirectory() as folder:
            mesh=grid_mesh(width=.12,height=.08,cell=.005)
            model=make_model(folder,o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),
                             o3d.utility.Vector3iVector(mesh.faces)))
            model.view_reference['undistorted_intrinsic']=K.tolist()
            model.store.origin_reference=dict(kind='initial_color_camera',provenance='synthetic_test')
            o3d.io.write_triangle_mesh(str(model.store.source),model.mesh)
            yy,xx=np.indices((100,100));mask=model.projection.valid.copy();rng=np.random.default_rng(6)
            for _ in range(4):
                cx,cy=rng.integers(42,59),rng.integers(42,59);rx,ry=rng.integers(2,6),rng.integers(2,6)
                mask &= ~((abs(xx-cx)<=rx)&(abs(yy-cy)<=ry))
            region=model.region([[25,25],[75,25],[75,75],[25,75]],'test','',selection=dict(mask=mask,method='work_only'))
            head=head_parameters('fixed_width_auto',effective_treatment_width_mm=5,overlap_percent=20)
            plan=workspace.generate(model,region,head,layout_algorithm=OPTIMIZED_PATCH_VERSION,routing_algorithm=BRIDGE_REORDER_ROUTING)
            self.assertEqual(plan['planning_context']['revision']['inputs']['algorithms'],planning_revision.BRIDGE_REORDER_PATCH_LAYOUT_ALGORITHMS)
            decision=plan['surface_trajectory']['gap_report']['reconnection']['ordering']['joint_reorder']
            self.assertEqual(decision['version'],VERSION);self.assertTrue(decision['adopted'])
            workspace.validate(plan,model.mesh,region)
            bad=deepcopy(plan);bad['surface_trajectory']['gap_report']['reconnection']['ordering']['joint_reorder']['steps'][0]['first']=0
            with self.assertRaises(ValueError):workspace.validate(bad,model.mesh,region)
            native=workspace.export(model,region,plan)
            self.assertEqual(to_player_plan(load_trajectory(native,reproduce=True))['segments'],plan['segments'])
            portable=save_project(model,region,Path(folder)/'bridge.region.zip',plan,catalog_root=Path(folder)/'catalog')
            reopened=open_project(portable,workspace=Path(folder)/'reopened')
            self.assertEqual(reopened.saved_plan['segments'],plan['segments'])
            self.assertFalse(plan['execution_ready'])

    def test_adopted_joint_reorder_survives_repair_and_archive(self):
        from pathlib import Path
        from femto.planning.continuous_quality import JOINT_REORDER_ROUTING
        from femto.planning.region_project import save_project,open_project
        from femto.planning.trajectory_export import load_trajectory
        from femto.planning.surface_trajectory import to_player_plan
        with tempfile.TemporaryDirectory() as folder:
            mesh=grid_mesh(width=.12,height=.08,cell=.005)
            model=make_model(folder,o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),
                             o3d.utility.Vector3iVector(mesh.faces)))
            model.view_reference['undistorted_intrinsic']=K.tolist()
            model.store.origin_reference=dict(kind='initial_color_camera',provenance='synthetic_test')
            o3d.io.write_triangle_mesh(str(model.store.source),model.mesh)
            yy,xx=np.indices((100,100));mask=model.projection.valid.copy();rng=np.random.default_rng(6)
            for _ in range(4):
                cx,cy=rng.integers(42,59),rng.integers(42,59);rx,ry=rng.integers(2,6),rng.integers(2,6)
                mask &= ~((abs(xx-cx)<=rx)&(abs(yy-cy)<=ry))
            region=model.region([[25,25],[75,25],[75,75],[25,75]],'test','',
                                selection=dict(mask=mask,method='work_only'))
            head=head_parameters('fixed_width_auto',effective_treatment_width_mm=5,overlap_percent=20)
            plan=workspace.generate(model,region,head,layout_algorithm=OPTIMIZED_PATCH_VERSION,routing_algorithm=JOINT_REORDER_ROUTING)
            decision=plan['surface_trajectory']['gap_report']['reconnection']['ordering']['joint_reorder']
            self.assertTrue(decision['adopted']);self.assertTrue(decision['steps'])
            with patch('femto.planning.bounded_joint_reorder.MAX_SCORE_CACHE_ENTRIES',0):
                without_cache=workspace.generate(model,region,head,layout_algorithm=OPTIMIZED_PATCH_VERSION,routing_algorithm=JOINT_REORDER_ROUTING)
            for field in ('ordered_items','statistics','gap_report','spacing_report','row_report','blocks','planner'):
                self.assertEqual(plan['surface_trajectory'].get(field),without_cache['surface_trajectory'].get(field))
            workspace.validate(plan,model.mesh,region)
            bad=deepcopy(plan)
            bad['surface_trajectory']['gap_report']['reconnection']['ordering']['joint_reorder']['steps'][0]['first']=0
            with self.assertRaises(ValueError):workspace.validate(bad,model.mesh,region)
            bad=deepcopy(plan)
            altered=bad['surface_trajectory']['gap_report']['reconnection']['ordering']['joint_reorder']['baseline_connections']
            next(link for link in altered if link['connected'])['samples'][-1]['xyz_m'][0]+=1.
            with self.assertRaises(ValueError):workspace.validate(bad,model.mesh,region)
            native=workspace.export(model,region,plan)
            self.assertEqual(to_player_plan(load_trajectory(native,reproduce=True))['segments'],plan['segments'])
            portable=save_project(model,region,Path(folder)/'joint.region.zip',plan,catalog_root=Path(folder)/'catalog')
            reopened=open_project(portable,workspace=Path(folder)/'reopened')
            self.assertEqual(reopened.saved_plan['segments'],plan['segments'])

    def test_joint_reorder_version_rechecks_its_record(self):
        from femto.planning.continuous_quality import JOINT_REORDER_ROUTING
        from femto.planning.bounded_joint_reorder import VERSION
        with tempfile.TemporaryDirectory() as folder:
            mesh=grid_mesh(width=.12,height=.08,cell=.005)
            model=make_model(folder,o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),
                             o3d.utility.Vector3iVector(mesh.faces)))
            model.view_reference['undistorted_intrinsic']=K.tolist()
            model.store.origin_reference=dict(kind='initial_color_camera',provenance='synthetic_test')
            region=model.region([[30,30],[70,30],[70,70],[30,70]],'test','',
                                selection=dict(mask=model.projection.valid.copy(),method='work_only'))
            head=head_parameters('fixed_width_auto',effective_treatment_width_mm=5,overlap_percent=20)
            plan=workspace.generate(model,region,head,layout_algorithm=OPTIMIZED_PATCH_VERSION,routing_algorithm=JOINT_REORDER_ROUTING)
            self.assertEqual(plan['planning_context']['revision']['inputs']['algorithms'],planning_revision.JOINT_REORDER_PATCH_LAYOUT_ALGORITHMS)
            trajectory=workspace.validate(plan,model.mesh,region)
            self.assertEqual(trajectory.planner['parameters']['continuous_routing'],JOINT_REORDER_ROUTING)
            decision=plan['surface_trajectory']['gap_report']['reconnection']['ordering']['joint_reorder']
            self.assertEqual(decision['version'],VERSION)
            bad=deepcopy(plan);bad['surface_trajectory']['gap_report']['reconnection']['ordering']['joint_reorder']['version']='bad'
            with self.assertRaises(ValueError):workspace.validate(bad,model.mesh,region)
            self.assertFalse(plan['execution_ready'])

    def test_short_detour_version_rechecks_its_record(self):
        from femto.planning.continuous_quality import SHORT_DETOUR_ROUTING
        from femto.planning.endpoint_preserving_detour import SHORT_VERSION
        with tempfile.TemporaryDirectory() as folder:
            mesh=grid_mesh(width=.12,height=.08,cell=.005)
            model=make_model(folder,o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),
                             o3d.utility.Vector3iVector(mesh.faces)))
            model.view_reference['undistorted_intrinsic']=K.tolist()
            model.store.origin_reference=dict(kind='initial_color_camera',provenance='synthetic_test')
            region=model.region([[30,30],[70,30],[70,70],[30,70]],'test','',
                                selection=dict(mask=model.projection.valid.copy(),method='work_only'))
            head=head_parameters('fixed_width_auto',effective_treatment_width_mm=5,overlap_percent=20)
            plan=workspace.generate(model,region,head,layout_algorithm=OPTIMIZED_PATCH_VERSION,routing_algorithm=SHORT_DETOUR_ROUTING)
            self.assertEqual(plan['planning_context']['revision']['inputs']['algorithms'],planning_revision.SHORT_DETOUR_PATCH_LAYOUT_ALGORITHMS)
            trajectory=workspace.validate(plan,model.mesh,region)
            self.assertEqual(trajectory.planner['parameters']['continuous_routing'],SHORT_DETOUR_ROUTING)
            decision=plan['surface_trajectory']['gap_report']['reconnection']['ordering']['preserved_detour']
            self.assertEqual(decision['version'],SHORT_VERSION)
            bad=deepcopy(plan);bad['surface_trajectory']['gap_report']['reconnection']['ordering']['preserved_detour']['version']='experimental-preserved-endpoint-detour-v1'
            with self.assertRaises(ValueError):workspace.validate(bad,model.mesh,region)
            self.assertFalse(plan['execution_ready'])

    def test_preserved_detour_version_rechecks_its_record(self):
        from femto.planning.continuous_quality import PRESERVED_DETOUR_ROUTING
        with tempfile.TemporaryDirectory() as folder:
            mesh=grid_mesh(width=.12,height=.08,cell=.005)
            model=make_model(folder,o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),
                             o3d.utility.Vector3iVector(mesh.faces)))
            model.view_reference['undistorted_intrinsic']=K.tolist()
            model.store.origin_reference=dict(kind='initial_color_camera',provenance='synthetic_test')
            region=model.region([[30,30],[70,30],[70,70],[30,70]],'test','',
                                selection=dict(mask=model.projection.valid.copy(),method='work_only'))
            head=head_parameters('fixed_width_auto',effective_treatment_width_mm=5,overlap_percent=20)
            plan=workspace.generate(model,region,head,layout_algorithm=OPTIMIZED_PATCH_VERSION,routing_algorithm=PRESERVED_DETOUR_ROUTING)
            self.assertEqual(plan['planning_context']['revision']['inputs']['algorithms'],planning_revision.PRESERVED_DETOUR_PATCH_LAYOUT_ALGORITHMS)
            trajectory=workspace.validate(plan,model.mesh,region)
            self.assertEqual(trajectory.planner['parameters']['continuous_routing'],PRESERVED_DETOUR_ROUTING)
            bad=deepcopy(plan);decision=bad['surface_trajectory']['gap_report']['reconnection']['ordering']['preserved_detour']
            decision['baseline']['detours']+=1
            with self.assertRaises(ValueError):workspace.validate(bad,model.mesh,region)
            self.assertFalse(plan['execution_ready'])

    def test_foldback_layout_is_versioned_and_rechecks_record(self):
        from femto.planning.continuous_quality import SEAM_BALANCED_ROUTING
        from femto.planning.fixed_width_layout import FOLDBACK_PATCH_VERSION
        from femto.planning.foldback_partition import VERSION as FOLDBACK_VERSION
        with tempfile.TemporaryDirectory() as folder:
            mesh=grid_mesh(width=.12,height=.08,cell=.005)
            model=make_model(folder,o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),
                             o3d.utility.Vector3iVector(mesh.faces)))
            model.view_reference['undistorted_intrinsic']=K.tolist()
            model.store.origin_reference=dict(kind='initial_color_camera',provenance='synthetic_test')
            region=model.region([[30,30],[70,30],[70,70],[30,70]],'test','',
                                selection=dict(mask=model.projection.valid.copy(),method='work_only'))
            head=head_parameters('fixed_width_auto',effective_treatment_width_mm=5,overlap_percent=20)
            with self.assertRaisesRegex(ValueError,'需要接缝联合排序'):
                workspace.generate(model,region,head,layout_algorithm=FOLDBACK_PATCH_VERSION)
            plan=workspace.generate(model,region,head,layout_algorithm=FOLDBACK_PATCH_VERSION,routing_algorithm=SEAM_BALANCED_ROUTING)
            self.assertEqual(plan['planning_context']['revision']['inputs']['algorithms'],planning_revision.FOLDBACK_PATCH_LAYOUT_ALGORITHMS)
            trajectory=workspace.validate(plan,model.mesh,region)
            layout=trajectory.planner['parameters']['fixed_width_layout']
            self.assertTrue(layout['fragments'])
            self.assertTrue(all(f['foldback_split_version']==FOLDBACK_VERSION for f in layout['fragments']))
            bad=deepcopy(plan);bad['surface_trajectory']['planner']['parameters']['fixed_width_layout']['fragments'][0]['foldback_split_version']='changed'
            with self.assertRaises(ValueError):workspace.validate(bad,model.mesh,region)
            with patch('femto.planning.foldback_partition.MAX_FACES',0):
                with self.assertRaisesRegex(ValueError,'面数超过预算'):
                    workspace.generate(model,region,head,layout_algorithm=FOLDBACK_PATCH_VERSION,routing_algorithm=SEAM_BALANCED_ROUTING)

    def test_seam_balanced_version_validates_and_detects_tampering(self):
        from femto.planning.continuous_quality import SEAM_BALANCED_ROUTING
        with tempfile.TemporaryDirectory() as folder:
            mesh=grid_mesh(width=.12,height=.08,cell=.005)
            model=make_model(folder,o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),
                             o3d.utility.Vector3iVector(mesh.faces)))
            model.view_reference['undistorted_intrinsic']=K.tolist()
            model.store.origin_reference=dict(kind='initial_color_camera',provenance='synthetic_test')
            region=model.region([[30,30],[70,30],[70,70],[30,70]],'test','',
                                selection=dict(mask=model.projection.valid.copy(),method='work_only'))
            head=head_parameters('fixed_width_auto',effective_treatment_width_mm=5,overlap_percent=20)
            plan=workspace.generate(model,region,head,layout_algorithm=OPTIMIZED_PATCH_VERSION,routing_algorithm=SEAM_BALANCED_ROUTING)
            self.assertEqual(plan['planning_context']['revision']['inputs']['algorithms'],planning_revision.SEAM_BALANCED_PATCH_LAYOUT_ALGORITHMS)
            trajectory=workspace.validate(plan,model.mesh,region)
            self.assertEqual(trajectory.planner['parameters']['continuous_routing'],SEAM_BALANCED_ROUTING)
            bad=deepcopy(plan);decision=bad['surface_trajectory']['gap_report']['reconnection']['ordering']['improvement']
            decision['balanced']['baseline']['detours']+=1
            with self.assertRaises(ValueError):workspace.validate(bad,model.mesh,region)
            self.assertFalse(plan['execution_ready'])

    def test_seam_routing_is_versioned_and_checks_records(self):
        from femto.planning.continuous_quality import SEAM_ROUTING
        with tempfile.TemporaryDirectory() as folder:
            mesh=grid_mesh(width=.12,height=.08,cell=.005)
            model=make_model(folder,o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),
                             o3d.utility.Vector3iVector(mesh.faces)))
            model.view_reference['undistorted_intrinsic']=K.tolist()
            model.store.origin_reference=dict(kind='initial_color_camera',provenance='synthetic_test')
            region=model.region([[30,30],[70,30],[70,70],[30,70]],'test','',
                                selection=dict(mask=model.projection.valid.copy(),method='work_only'))
            head=head_parameters('fixed_width_auto',effective_treatment_width_mm=5,overlap_percent=20)
            plan=workspace.generate(model,region,head,layout_algorithm=OPTIMIZED_PATCH_VERSION,routing_algorithm=SEAM_ROUTING)
            self.assertEqual(plan['planning_context']['revision']['inputs']['algorithms'],planning_revision.SEAM_PATCH_LAYOUT_ALGORITHMS)
            trajectory=workspace.validate(plan,model.mesh,region)
            self.assertEqual(trajectory.planner['parameters']['continuous_routing'],SEAM_ROUTING)
            bad=deepcopy(plan);decision=bad['surface_trajectory']['gap_report']['reconnection']['ordering']['improvement']
            decision['baseline']['detours']+=1
            with self.assertRaises(ValueError):workspace.validate(bad,model.mesh,region)
            self.assertFalse(plan['execution_ready'])

    def test_seam_routing_budget_fallback_preserves_verified_route(self):
        from femto.planning.continuous_quality import SEAM_ROUTING
        with tempfile.TemporaryDirectory() as folder:
            mesh=grid_mesh(width=.12,height=.08,cell=.005)
            model=make_model(folder,o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),
                             o3d.utility.Vector3iVector(mesh.faces)))
            model.view_reference['undistorted_intrinsic']=K.tolist()
            model.store.origin_reference=dict(kind='initial_color_camera',provenance='synthetic_test')
            region=model.region([[30,30],[70,30],[70,70],[30,70]],'test','',
                                selection=dict(mask=model.projection.valid.copy(),method='work_only'))
            head=head_parameters('fixed_width_auto',effective_treatment_width_mm=5,overlap_percent=20)
            with patch('femto.planning.seam_visit_proposal.propose',
                       side_effect=ValueError('连续路线整边搜索超过预算，请缩小选区')):
                plan=workspace.generate(model,region,head,layout_algorithm=OPTIMIZED_PATCH_VERSION,routing_algorithm=SEAM_ROUTING)
                trajectory=workspace.validate(plan,model.mesh,region)
                decision=trajectory.gap_report['reconnection']['ordering']['improvement']
                self.assertEqual(decision['reason'],'geometry_budget');self.assertFalse(decision['adopted'])

    def test_seam_routing_improves_fold_without_losing_scan_edges(self):
        from collections import Counter
        from femto.planning.continuous_quality import BALANCED_ROUTING,SEAM_ROUTING
        def edges(plan):
            return Counter(tuple(sorted((tuple(a['xyz_m']),tuple(b['xyz_m']))))
                for segment in plan['surface_trajectory']['ordered_items'] if segment['kind']=='scan_segment'
                for a,b in zip(segment['points'],segment['points'][1:]) if b.get('role')!='connector')
        for angle in (55,70):
            with self.subTest(angle=angle),tempfile.TemporaryDirectory() as folder:
                mesh=grid_mesh(width=.12,height=.08,cell=.005);slope=np.tan(np.deg2rad(angle))
                mesh.vertices[:,2]+=np.maximum(mesh.vertices[:,0],0)*slope
                ray=(np.arange(100)-K[0,2])/K[0,0];values=.5/(1-np.maximum(ray,0)*slope)
                depth=np.broadcast_to(np.round(values*1000),(100,100)).astype(np.uint16).copy()
                model=make_model(folder,o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),
                                 o3d.utility.Vector3iVector(mesh.faces)),depth=depth)
                model.view_reference['undistorted_intrinsic']=K.tolist()
                model.store.origin_reference=dict(kind='initial_color_camera',provenance='synthetic_test')
                region=model.region([[30,30],[70,30],[70,70],[30,70]],'test','',
                                    selection=dict(mask=model.projection.valid.copy(),method='work_only'))
                head=head_parameters('fixed_width_auto',effective_treatment_width_mm=5,overlap_percent=20)
                prior=workspace.generate(model,region,head,layout_algorithm=OPTIMIZED_PATCH_VERSION,routing_algorithm=BALANCED_ROUTING)
                actual=workspace.generate(model,region,head,layout_algorithm=OPTIMIZED_PATCH_VERSION,routing_algorithm=SEAM_ROUTING)
                workspace.validate(actual,model.mesh,region)
                a=prior['surface_trajectory']['statistics'];b=actual['surface_trajectory']['statistics']
                self.assertEqual(edges(prior),edges(actual))
                self.assertEqual(prior['surface_trajectory']['planner']['parameters']['fixed_width_layout'],
                                 actual['surface_trajectory']['planner']['parameters']['fixed_width_layout'])
                self.assertLess(b['connector_length_m'],a['connector_length_m'])
                self.assertLessEqual(b['connected_detours'],a['connected_detours'])
                self.assertLessEqual(b['revisit_connector_length_m'],a['revisit_connector_length_m'])
                self.assertEqual(b['gap_count']-b['inter_block_gap_count'],0)

    def test_balanced_routing_versions_and_checks_metrics(self):
        from femto.planning.continuous_quality import BALANCED_ROUTING
        with tempfile.TemporaryDirectory() as folder:
            mesh=grid_mesh(width=.12,height=.08,cell=.005)
            model=make_model(folder,o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),
                             o3d.utility.Vector3iVector(mesh.faces)))
            model.view_reference['undistorted_intrinsic']=K.tolist()
            model.store.origin_reference=dict(kind='initial_color_camera',provenance='synthetic_test')
            region=model.region([[30,30],[70,30],[70,70],[30,70]],'test','',
                                selection=dict(mask=model.projection.valid.copy(),method='work_only'))
            head=head_parameters('fixed_width_auto',effective_treatment_width_mm=5,overlap_percent=20)
            plan=workspace.generate(model,region,head,layout_algorithm=OPTIMIZED_PATCH_VERSION,routing_algorithm=BALANCED_ROUTING)
            self.assertEqual(plan['planning_context']['revision']['inputs']['algorithms'],planning_revision.BALANCED_PATCH_LAYOUT_ALGORITHMS)
            trajectory=workspace.validate(plan,model.mesh,region)
            self.assertEqual(trajectory.planner['parameters']['continuous_routing'],BALANCED_ROUTING)
            bad=deepcopy(plan);record=bad['surface_trajectory']['gap_report']['reconnection']['ordering']['improvement']
            record['baseline']['detours']+=1
            with self.assertRaises(ValueError):workspace.validate(bad,model.mesh,region)
            self.assertFalse(plan['execution_ready'])

    def test_balanced_routing_budget_fallback_preserves_verified_route(self):
        from femto.planning.continuous_quality import BALANCED_ROUTING
        with tempfile.TemporaryDirectory() as folder:
            mesh=grid_mesh(width=.12,height=.08,cell=.005)
            model=make_model(folder,o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),
                             o3d.utility.Vector3iVector(mesh.faces)))
            model.view_reference['undistorted_intrinsic']=K.tolist()
            model.store.origin_reference=dict(kind='initial_color_camera',provenance='synthetic_test')
            region=model.region([[30,30],[70,30],[70,70],[30,70]],'test','',
                                selection=dict(mask=model.projection.valid.copy(),method='work_only'))
            head=head_parameters('fixed_width_auto',effective_treatment_width_mm=5,overlap_percent=20)
            with patch('femto.planning.balanced_visit_improvement.propose',
                       side_effect=ValueError('连续路线整边搜索超过预算，请缩小选区')):
                plan=workspace.generate(model,region,head,layout_algorithm=OPTIMIZED_PATCH_VERSION,routing_algorithm=BALANCED_ROUTING)
                trajectory=workspace.validate(plan,model.mesh,region)
                decision=trajectory.gap_report['reconnection']['ordering']['improvement']
                self.assertEqual(decision['skipped'],'geometry_budget');self.assertFalse(decision['adopted'])

    def test_support_routing_recomputes_endpoint_decisions(self):
        from femto.planning.continuous_quality import SUPPORT_ROUTING
        with tempfile.TemporaryDirectory() as folder:
            mesh=grid_mesh(width=.12,height=.08,cell=.005)
            model=make_model(folder,o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),
                             o3d.utility.Vector3iVector(mesh.faces)))
            model.view_reference['undistorted_intrinsic']=K.tolist()
            model.store.origin_reference=dict(kind='initial_color_camera',provenance='synthetic_test')
            region=model.region([[30,30],[70,30],[70,70],[30,70]],'test','',
                                selection=dict(mask=model.projection.valid.copy(),method='work_only'))
            head=head_parameters('fixed_width_auto',effective_treatment_width_mm=5,overlap_percent=20)
            plan=workspace.generate(model,region,head,layout_algorithm=OPTIMIZED_PATCH_VERSION,routing_algorithm=SUPPORT_ROUTING)
            self.assertEqual(plan['planning_context']['revision']['inputs']['algorithms'],planning_revision.SUPPORT_PATCH_LAYOUT_ALGORITHMS)
            trajectory=workspace.validate(plan,model.mesh,region)
            self.assertEqual(trajectory.planner['parameters']['continuous_routing'],SUPPORT_ROUTING)
            decision=trajectory.gap_report['reconnection']['ordering']['endpoint_support']
            self.assertEqual(decision['candidate_count'],len(decision['adopted_links'])+len(decision['rejected_links']))
            bad=deepcopy(plan);bad['surface_trajectory']['gap_report']['reconnection']['ordering']['endpoint_support']['candidate_count']+=1
            with self.assertRaises(ValueError):workspace.validate(bad,model.mesh,region)

    def test_improved_routing_is_versioned_and_preserves_original_layout(self):
        from femto.planning.continuous_quality import IMPROVED_ROUTING
        with tempfile.TemporaryDirectory() as folder:
            mesh=grid_mesh(width=.12,height=.08,cell=.005)
            model=make_model(folder,o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),
                             o3d.utility.Vector3iVector(mesh.faces)))
            model.view_reference['undistorted_intrinsic']=K.tolist()
            model.store.origin_reference=dict(kind='initial_color_camera',provenance='synthetic_test')
            region=model.region([[30,30],[70,30],[70,70],[30,70]],'test','',
                                selection=dict(mask=model.projection.valid.copy(),method='work_only'))
            head=head_parameters('fixed_width_auto',effective_treatment_width_mm=5,overlap_percent=20)
            plan=workspace.generate(model,region,head,layout_algorithm=OPTIMIZED_PATCH_VERSION,routing_algorithm=IMPROVED_ROUTING)
            self.assertEqual(plan['planning_context']['revision']['inputs']['algorithms'],planning_revision.IMPROVED_PATCH_LAYOUT_ALGORITHMS)
            trajectory=workspace.validate(plan,model.mesh,region)
            self.assertEqual(trajectory.planner['parameters']['continuous_routing'],IMPROVED_ROUTING)
            self.assertIn('improvement',trajectory.gap_report['reconnection']['ordering'])
            self.assertFalse(plan['treatment_head']['execution_ready'])
            bad=deepcopy(plan);bad['surface_trajectory']['gap_report']['reconnection']['ordering']['improvement']['version']='invalid'
            with self.assertRaises(ValueError):workspace.validate(bad,model.mesh,region)
            with self.assertRaises(ValueError):workspace.generate(model,region,head,routing_algorithm=IMPROVED_ROUTING)

    def test_improved_routing_keeps_verified_original_when_search_budget_ends(self):
        from femto.planning.continuous_quality import IMPROVED_ROUTING
        with tempfile.TemporaryDirectory() as folder:
            mesh=grid_mesh(width=.12,height=.08,cell=.005)
            model=make_model(folder,o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),
                             o3d.utility.Vector3iVector(mesh.faces)))
            model.view_reference['undistorted_intrinsic']=K.tolist()
            model.store.origin_reference=dict(kind='initial_color_camera',provenance='synthetic_test')
            region=model.region([[30,30],[70,30],[70,70],[30,70]],'test','',
                                selection=dict(mask=model.projection.valid.copy(),method='work_only'))
            head=head_parameters('fixed_width_auto',effective_treatment_width_mm=5,overlap_percent=20)
            with patch('femto.planning.verified_visit_improvement.improve',
                       side_effect=ValueError('连续路线整边搜索超过预算，请缩小选区')):
                plan=workspace.generate(model,region,head,layout_algorithm=OPTIMIZED_PATCH_VERSION,routing_algorithm=IMPROVED_ROUTING)
                trajectory=workspace.validate(plan,model.mesh,region)
                decision=trajectory.gap_report['reconnection']['ordering']['improvement']
                self.assertEqual(decision['skipped'],'geometry_budget');self.assertFalse(decision['adopted'])
                self.assertEqual(trajectory.gap_report['reconnection']['connections'],decision['baseline_connections'])

    def test_budgeted_layout_falls_back_only_for_declared_search_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            mesh=grid_mesh(width=.12,height=.08,cell=.005)
            model=make_model(folder,o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),
                             o3d.utility.Vector3iVector(mesh.faces)))
            model.view_reference['undistorted_intrinsic']=K.tolist()
            model.store.origin_reference=dict(kind='initial_color_camera',provenance='synthetic_test')
            region=model.region([[30,30],[70,30],[70,70],[30,70]],'test','',
                                selection=dict(mask=model.projection.valid.copy(),method='work_only'))
            head=head_parameters('fixed_width_auto',effective_treatment_width_mm=5,overlap_percent=20)
            with patch('femto.planning.metric_row_optimizer.positions_with_slack',
                       side_effect=ValueError('局部优化状态转移超过预算')):
                plan=workspace.generate(model,region,head,layout_algorithm=OPTIMIZED_PATCH_VERSION)
                trajectory=workspace.validate(plan,model.mesh,region)
                layout=trajectory.planner['parameters']['fixed_width_layout']
                parts=[p for f in layout['fragments'] for p in f['patches'] if p.get('alignment_optimization')]
                self.assertTrue(parts)
                self.assertTrue(all(p['alignment_optimization']['strategy']=='prior_greedy_budget_fallback' for p in parts))
            with patch('femto.planning.metric_row_optimizer.positions_with_slack',side_effect=ValueError('unexpected data failure')):
                with self.assertRaisesRegex(ValueError,'unexpected data failure'):
                    workspace.generate(model,region,head,layout_algorithm=OPTIMIZED_PATCH_VERSION)

    def test_budgeted_global_layout_is_explicit_version_and_independently_recomputed(self):
        with tempfile.TemporaryDirectory() as folder:
            mesh=grid_mesh(width=.12,height=.08,cell=.005)
            model=make_model(folder,o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),
                             o3d.utility.Vector3iVector(mesh.faces)))
            model.view_reference['undistorted_intrinsic']=K.tolist()
            model.store.origin_reference=dict(kind='initial_color_camera',provenance='synthetic_test')
            region=model.region([[30,30],[70,30],[70,70],[30,70]],'test','',
                                selection=dict(mask=model.projection.valid.copy(),method='work_only'))
            head=head_parameters('fixed_width_auto',effective_treatment_width_mm=5,overlap_percent=20)
            plan=workspace.generate(model,region,head,layout_algorithm=OPTIMIZED_PATCH_VERSION)
            self.assertEqual(plan['planning_context']['revision']['inputs']['algorithms'],planning_revision.OPTIMIZED_PATCH_LAYOUT_ALGORITHMS)
            trajectory=workspace.validate(plan,model.mesh,region)
            layout=trajectory.planner['parameters']['fixed_width_layout']
            self.assertEqual(layout['request']['version'],OPTIMIZED_PATCH_VERSION)
            self.assertFalse(plan['treatment_head']['execution_ready'])
            self.assertFalse(plan['treatment_head']['physical_accuracy_verified'])
            parts=[p for f in layout['fragments'] for p in f['patches'] if p.get('alignment_optimization')]
            self.assertTrue(parts)
            self.assertTrue(all(p['alignment_optimization']['extra_fraction']==.2 for p in parts))
            bad=deepcopy(layout);part=next(p for f in bad['fragments'] for p in f['patches'] if p.get('alignment_optimization'))
            part['alignment_optimization']['extra_fraction']=.1
            with self.assertRaisesRegex(ValueError,'独立重算'):
                validate_layout(bad,workspace.canonical_region_mesh(model.mesh,region),region,trajectory.planner['parameters'],lambda:None)
            with self.assertRaises(ValueError):
                workspace.generate(model,region,head_parameters('direct_row_spacing',row_spacing_mm=5),layout_algorithm=OPTIMIZED_PATCH_VERSION)

    def test_repair_variant_without_problem_is_saved_as_noop(self):
        with tempfile.TemporaryDirectory() as folder:
            mesh=grid_mesh(width=.12,height=.08,cell=.005)
            model=make_model(folder,o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),
                             o3d.utility.Vector3iVector(mesh.faces)))
            model.view_reference['undistorted_intrinsic']=K.tolist()
            model.store.origin_reference=dict(kind='initial_color_camera',provenance='synthetic_test')
            region=model.region([[30,30],[70,30],[70,70],[30,70]],'test','',
                                selection=dict(mask=model.projection.valid.copy(),method='work_only'))
            head=head_parameters('fixed_width_auto',effective_treatment_width_mm=5,overlap_percent=20)
            capture=planning_revision.capture
            def selected(*args,**kwargs):
                kwargs.setdefault('algorithms',planning_revision.REPAIR_PATCH_LAYOUT_ALGORITHMS)
                return capture(*args,**kwargs)
            with patch.object(planning_revision,'capture',side_effect=selected):plan=workspace.generate(model,region,head)
            trajectory=workspace.validate(plan,model.mesh,region)
            self.assertEqual(trajectory.planner['parameters']['fixed_width_layout']['request']['version'],REPAIR_PATCH_VERSION)
            self.assertIsNone(trajectory.gap_report['row_repair']['selected'])
            bad=deepcopy(trajectory);bad.gap_report['row_repair']['original_internal_gaps']+=1
            from femto.planning.repair_workflow import validate as validate_repair
            with self.assertRaises(ValueError):validate_repair(bad,workspace.canonical_region_mesh(model.mesh,region),region)

    def test_stable_patch_variant_is_recomputed_and_not_default(self):
        with tempfile.TemporaryDirectory() as folder:
            mesh=grid_mesh(width=.12,height=.08,cell=.005)
            model=make_model(folder,o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),
                             o3d.utility.Vector3iVector(mesh.faces)))
            model.view_reference['undistorted_intrinsic']=K.tolist()
            model.store.origin_reference=dict(kind='initial_color_camera',provenance='synthetic_test')
            region=model.region([[30,30],[70,30],[70,70],[30,70]],'test','',
                                selection=dict(mask=model.projection.valid.copy(),method='work_only'))
            head=head_parameters('fixed_width_auto',effective_treatment_width_mm=5,overlap_percent=20)
            capture=planning_revision.capture
            def selected(*args,**kwargs):
                kwargs.setdefault('algorithms',planning_revision.STABLE_PATCH_LAYOUT_ALGORITHMS)
                return capture(*args,**kwargs)
            with patch.object(planning_revision,'capture',side_effect=selected):plan=workspace.generate(model,region,head)
            trajectory=workspace.validate(plan,model.mesh,region)
            layout=trajectory.planner['parameters']['fixed_width_layout']
            self.assertEqual(layout['request']['version'],STABLE_PATCH_VERSION)
            self.assertTrue(any('prior_positions_m' in f for f in layout['fragments']))
            bad=deepcopy(layout);bad['fragments'][0]['prior_positions_m'].append(.123)
            with self.assertRaisesRegex(ValueError,'独立重算'):
                validate_layout(bad,workspace.canonical_region_mesh(model.mesh,region),region,trajectory.planner['parameters'],lambda:None)

    def test_default_keeps_previous_layout_and_uses_verified_visit_order(self):
        with tempfile.TemporaryDirectory() as folder:
            mesh=grid_mesh(width=.12,height=.08,cell=.005)
            model=make_model(folder,o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),
                             o3d.utility.Vector3iVector(mesh.faces)))
            model.view_reference['undistorted_intrinsic']=K.tolist()
            model.store.origin_reference=dict(kind='initial_color_camera',provenance='synthetic_test')
            region=model.region([[30,30],[70,30],[70,70],[30,70]],'test','',
                                selection=dict(mask=model.projection.valid.copy(),method='work_only'))
            head=head_parameters('fixed_width_auto',effective_treatment_width_mm=5,overlap_percent=20)
            plan=workspace.generate(model,region,head)
            self.assertEqual(plan['planning_context']['revision']['inputs']['algorithms'],planning_revision.ORDERED_LAYOUT_ALGORITHMS)
            workspace.validate(plan,model.mesh,region)

    def test_planar_hole_and_fold_keep_original_geometry_checks_and_recompute(self):
        for shape in ('plane','hole','fold'):
            with self.subTest(shape=shape),tempfile.TemporaryDirectory() as folder:
                mesh=grid_mesh(width=.12,height=.08,cell=.005)
                depth=None
                if shape=='fold':
                    slope=np.tan(np.deg2rad(55.))
                    mesh.vertices[:,2]+=(np.maximum(mesh.vertices[:,0],0)*slope)
                    ray=(np.arange(100)-K[0,2])/K[0,0]
                    values=.5/(1-np.maximum(ray,0)*slope)
                    depth=np.broadcast_to(np.round(values*1000),(100,100)).astype(np.uint16).copy()
                model=make_model(folder,o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),
                                o3d.utility.Vector3iVector(mesh.faces)),depth=depth)
                model.view_reference['undistorted_intrinsic']=K.tolist()
                model.store.origin_reference=dict(kind='initial_color_camera',provenance='synthetic_test')
                mask=model.projection.valid.copy()
                if shape=='hole':mask[48:52,48:52]=False
                region=model.region([[30,30],[70,30],[70,70],[30,70]],'test','',selection=dict(mask=mask,method='work_only'))
                head=head_parameters('fixed_width_auto',effective_treatment_width_mm=5,overlap_percent=20)
                capture=planning_revision.capture
                def selected(*args,**kwargs):
                    kwargs.setdefault('algorithms',planning_revision.PATCH_LAYOUT_ALGORITHMS)
                    return capture(*args,**kwargs)
                with patch.object(planning_revision,'capture',side_effect=selected):
                    plan=workspace.generate(model,region,head)
                self.assertEqual(plan['planning_context']['revision']['inputs']['algorithms'],planning_revision.PATCH_LAYOUT_ALGORITHMS)
                trajectory=workspace.validate(plan,model.mesh,region)
                parameters=trajectory.planner['parameters'];layout=parameters['fixed_width_layout']
                self.assertEqual(layout['request']['version'],PATCH_VERSION)
                self.assertFalse(plan['treatment_head']['execution_ready'])
                patches=[p for f in layout['fragments'] for p in f['patches']]
                if shape=='fold':self.assertGreater(len(patches),1)
                canonical=workspace.canonical_region_mesh(model.mesh,region)
                bad=deepcopy(layout)
                bad['fragments'][0]['patches'][0]['face_ids'].pop()
                with self.assertRaisesRegex(ValueError,'独立重算'):
                    validate_layout(bad,canonical,region,parameters,lambda:None)
                bad=deepcopy(layout)
                p=next(p for f in bad['fragments'] for p in f['patches'] if p.get('positions_m'))
                p['positions_m'][0]+=.001
                with self.assertRaisesRegex(ValueError,'独立重算'):
                    validate_layout(bad,canonical,region,parameters,lambda:None)
