"""Versioned research shifts retain bindings and round-trip geometry."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import open3d as o3d
from test_fixed_rgb_experience import make_model, K
from test_mesh_slice_planner import grid_mesh
from femto.planning import planning_revision, surface_workspace
from femto.planning.continuous_quality import RELOCATION_ROUTING, DEEP_RELOCATION_ROUTING, BRIDGE_REORDER_ROUTING
from femto.planning.fixed_width_layout import SHIFT_PATCH_VERSION, OPTIMIZED_PATCH_VERSION, Layout, request
from femto.planning.region_project import open_project, save_project
from femto.planning.surface_trajectory import validate_trajectory
from femto.planning.treatment_head import head_parameters


class ShiftLayoutResearchTests(unittest.TestCase):
    def test_shift_limits_and_nonresearch_rules(self):
        record = dict(block_index=0, fragment_face_sha256='a'*64, patch_seed_face=0,
                      position_m=.01, shift_m=.00015)
        with patch('femto.planning.quality_constrained.ReferenceEnvelope'):
            Layout(request(5, 20, version=SHIFT_PATCH_VERSION), {}, row_shifts=[record])
            for value in (0., .0003, True, float('nan')):
                bad = dict(record, shift_m=value)
                with self.subTest(value=value), self.assertRaises(ValueError):
                    Layout(request(5, 20, version=SHIFT_PATCH_VERSION), {}, row_shifts=[bad])
            with self.assertRaises(ValueError):
                Layout(request(5, 20, version=OPTIMIZED_PATCH_VERSION), {}, row_shifts=[record])

    def test_versioned_shift_binding_and_portable_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            mesh = grid_mesh(width=.12, height=.08, cell=.005)
            model = make_model(folder, o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),
                                                                  o3d.utility.Vector3iVector(mesh.faces)))
            model.view_reference['undistorted_intrinsic'] = K.tolist()
            model.store.origin_reference = dict(kind='initial_color_camera', provenance='synthetic_test')
            o3d.io.write_triangle_mesh(str(model.store.source), model.mesh)
            region = model.region([[30,30],[70,30],[70,70],[30,70]], 'test', '',
                                  selection=dict(mask=model.projection.valid.copy(), method='work_only'))
            head = head_parameters('fixed_width_auto', effective_treatment_width_mm=5, overlap_percent=20)
            with self.assertRaisesRegex(ValueError, '直接精确目标'):
                surface_workspace.generate(model, region, head, layout_algorithm=OPTIMIZED_PATCH_VERSION,
                    routing_algorithm=BRIDGE_REORDER_ROUTING, target_algorithm=planning_revision.EXACT_TARGET)
            old = surface_workspace.generate(model, region, head, layout_algorithm=OPTIMIZED_PATCH_VERSION,
                                             routing_algorithm=RELOCATION_ROUTING)
            exact_old = surface_workspace.generate(model, region, head, layout_algorithm=OPTIMIZED_PATCH_VERSION,
                routing_algorithm=RELOCATION_ROUTING, target_algorithm=planning_revision.EXACT_TARGET)
            self.assertEqual(exact_old['surface_trajectory']['ordered_items'], old['surface_trajectory']['ordered_items'])
            self.assertEqual(exact_old['surface_trajectory']['statistics'], old['surface_trajectory']['statistics'])
            old_revision, exact_revision = old['planning_context']['revision'], exact_old['planning_context']['revision']
            self.assertEqual(planning_revision.changed(old_revision, exact_revision), ['algorithms'])
            self.assertEqual(exact_revision['inputs']['algorithms']['target'], planning_revision.EXACT_TARGET)
            exact_ledger = surface_workspace.target_for(model, exact_old, reproduce=True)
            exact_old_archive = save_project(model, region, Path(folder)/'exact-v11.region.zip', exact_old,
                                             catalog_root=Path(folder)/'catalog')
            reopened_old = open_project(exact_old_archive, workspace=Path(folder)/'v11-exact-reopened')
            self.assertEqual(reopened_old.target_ledger.metadata, exact_ledger.metadata)
            self.assertEqual(reopened_old.saved_plan['surface_trajectory']['ordered_items'],
                             exact_old['surface_trajectory']['ordered_items'])
            fragments = old['surface_trajectory']['planner']['parameters']['fixed_width_layout']['fragments']
            fragment, area = next((f, p) for f in fragments for p in f['patches'] if len(p.get('positions_m', [])) >= 3)
            shift = dict(block_index=fragment['block_index'], fragment_face_sha256=fragment['face_indices_sha256'],
                         patch_seed_face=area['seed_face'], position_m=area['positions_m'][1], shift_m=.00005)
            with self.assertRaisesRegex(ValueError, '移动换序'):
                surface_workspace.generate(model, region, head, layout_algorithm=SHIFT_PATCH_VERSION,
                                           routing_algorithm=BRIDGE_REORDER_ROUTING, research_row_shifts=[shift])
            with self.assertRaisesRegex(ValueError, '仅可用于'):
                surface_workspace.generate(model, region, head, research_row_shifts=[shift])
            with self.assertRaisesRegex(ValueError, '直接精确目标'):
                surface_workspace.generate(model,region,head,layout_algorithm=SHIFT_PATCH_VERSION,
                    routing_algorithm=DEEP_RELOCATION_ROUTING,research_row_shifts=[shift],
                    target_algorithm=planning_revision.EXACT_TARGET)
            plan = surface_workspace.generate(model, region, head, layout_algorithm=SHIFT_PATCH_VERSION,
                routing_algorithm=RELOCATION_ROUTING, research_row_shifts=[shift])
            algorithms = plan['planning_context']['revision']['inputs']['algorithms']
            self.assertEqual(algorithms['row_shifts_sha256'], planning_revision.digest([shift]))
            self.assertTrue(planning_revision.supported_target_algorithms(algorithms))
            self.assertFalse(planning_revision.supported_target_algorithms(dict(algorithms,row_shifts_sha256='bad')))
            trajectory = surface_workspace.validate(plan, model.mesh, region)
            damaged = deepcopy(trajectory)
            damaged.planner['parameters']['fixed_width_layout']['row_shifts'][0]['shift_m'] *= 2
            with self.assertRaisesRegex(ValueError, '计算修订'):
                validate_trajectory(damaged, mesh=model.mesh, region=region)
            archive = save_project(model, region, Path(folder)/'research.region.zip', plan,
                                   catalog_root=Path(folder)/'catalog')
            reopened = open_project(archive, workspace=Path(folder)/'reopened')
            self.assertEqual(reopened.saved_plan['surface_trajectory']['ordered_items'],
                             plan['surface_trajectory']['ordered_items'])
            self.assertFalse(reopened.saved_plan['execution_ready'])
            deep = surface_workspace.generate(model, region, head, layout_algorithm=SHIFT_PATCH_VERSION,
                routing_algorithm=DEEP_RELOCATION_ROUTING, research_row_shifts=[shift])
            deep_algorithms = deep['planning_context']['revision']['inputs']['algorithms']
            self.assertEqual(deep_algorithms,dict(planning_revision.RESEARCH_SHIFT_DEEP_RELOCATION_ALGORITHMS,
                                                row_shifts_sha256=planning_revision.digest([shift])))
            self.assertTrue(planning_revision.supported_target_algorithms(deep_algorithms))
            self.assertFalse(planning_revision.supported_target_algorithms(dict(deep_algorithms,row_shifts_sha256='bad')))
            surface_workspace.validate(deep,model.mesh,region)
            damaged = deepcopy(deep)
            del damaged['surface_trajectory']['gap_report']['reconnection']['ordering']['relocation_deep']
            with self.assertRaises(ValueError):surface_workspace.validate(damaged,model.mesh,region)
            deep_archive=save_project(model,region,Path(folder)/'deep-shift.region.zip',deep,
                                     catalog_root=Path(folder)/'catalog')
            reopened_deep=open_project(deep_archive,workspace=Path(folder)/'deep-shift-reopened')
            self.assertEqual(reopened_deep.saved_plan['surface_trajectory']['ordered_items'],deep['surface_trajectory']['ordered_items'])
            self.assertEqual(reopened_deep.saved_plan['planning_context']['revision']['inputs']['algorithms'],deep_algorithms)
            exact = surface_workspace.generate(model, region, head, layout_algorithm=SHIFT_PATCH_VERSION,
                routing_algorithm=RELOCATION_ROUTING, research_row_shifts=[shift],
                target_algorithm=planning_revision.EXACT_TARGET)
            self.assertEqual(exact['surface_trajectory']['ordered_items'], plan['surface_trajectory']['ordered_items'])
            self.assertEqual(exact['surface_trajectory']['statistics'], plan['surface_trajectory']['statistics'])
            exact_algorithms = exact['planning_context']['revision']['inputs']['algorithms']
            self.assertEqual(exact_algorithms['target'], planning_revision.EXACT_TARGET)
            self.assertEqual(exact_algorithms['row_shifts_sha256'], planning_revision.digest([shift]))
            self.assertTrue(planning_revision.supported_target_algorithms(exact_algorithms))
            ledger = surface_workspace.target_for(model, exact, reproduce=True)
            exact_archive = save_project(model, region, Path(folder)/'exact-research.region.zip', exact,
                                         catalog_root=Path(folder)/'catalog')
            reopened_exact = open_project(exact_archive, workspace=Path(folder)/'exact-reopened')
            self.assertEqual(reopened_exact.saved_plan['surface_trajectory']['ordered_items'],
                             exact['surface_trajectory']['ordered_items'])
            self.assertEqual(reopened_exact.target_ledger.metadata, ledger.metadata)
            self.assertFalse(reopened_exact.saved_plan['execution_ready'])


if __name__ == '__main__': unittest.main()
