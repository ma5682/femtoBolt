"""Operation-local reuse preserves files, geometry and independent validation."""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

import numpy as np
import open3d as o3d
import trimesh

from femto.annotation.scene_annotations import region_mesh, mesh_digest
from femto.geometry import region_reuse as reuse
from femto.planning import mesh_slice_planner as planner, surface_workspace as workspace
import test_surface_workspace as workspace_fixture


class RegionGeometryReuseTests(unittest.TestCase):
    def setUp(self):
        fixture = workspace_fixture.SurfaceWorkspaceTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        self.model, self.region = fixture.model, fixture.region()

    def test_region_and_canonical_reuse_preserve_full_precision_and_face_order(self):
        source = mesh_digest(self.model.mesh)
        plain = region_mesh(self.model.mesh, self.region['triangle_indices'])
        with reuse.operation():
            first = region_mesh(self.model.mesh, self.region['triangle_indices'])
            second = region_mesh(self.model.mesh, self.region['triangle_indices'])
            self.assertIs(first, second)
            np.testing.assert_array_equal(first.vertices, plain.vertices)
            np.testing.assert_array_equal(first.triangles, plain.triangles)
            a = workspace.canonical_region_mesh(self.model.mesh, self.region)
            self.assertIs(a, workspace.canonical_region_mesh(self.model.mesh, self.region))
            np.testing.assert_array_equal(a.vertices, np.asarray(plain.vertices).astype(np.float32).astype(float))
            np.testing.assert_array_equal(a.faces, plain.triangles)
        self.assertEqual(mesh_digest(self.model.mesh), source)
        self.assertIsNot(a, workspace.canonical_region_mesh(self.model.mesh, self.region))

    def test_source_attributes_selection_and_cached_mutation_invalidate(self):
        with reuse.operation():
            old = region_mesh(self.model.mesh, self.region['triangle_indices'])
            np.asarray(old.vertices)[0, 0] += .1
            fixed = region_mesh(self.model.mesh, self.region['triangle_indices'])
            self.assertIsNot(old, fixed)
            np.asarray(self.model.mesh.vertices)[0, 0] += .02
            changed = region_mesh(self.model.mesh, self.region['triangle_indices'])
            self.assertIsNot(fixed, changed)
            self.model.mesh.paint_uniform_color([.2, .3, .4])
            colored = region_mesh(self.model.mesh, self.region['triangle_indices'])
            self.assertIsNot(changed, colored)
            self.assertTrue(colored.has_vertex_colors())
            self.model.mesh.triangle_material_ids = o3d.utility.IntVector(
                np.full(len(self.model.mesh.triangles), 7, dtype=np.int32))
            material = region_mesh(self.model.mesh, self.region['triangle_indices'])
            self.assertIsNot(colored, material)
            np.asarray(material.triangle_material_ids)[:] = 11
            restored = region_mesh(self.model.mesh, self.region['triangle_indices'])
            np.testing.assert_array_equal(restored.triangle_material_ids, np.full(len(material.triangle_material_ids), 7))
            smaller = region_mesh(self.model.mesh, self.region['triangle_indices'][1:])
            self.assertEqual(len(smaller.triangles), len(colored.triangles)-1)
            a = workspace.canonical_region_mesh(self.model.mesh, self.region)
            a.vertices[0, 0] += .05
            self.assertIsNot(a, workspace.canonical_region_mesh(self.model.mesh, self.region))

    def test_memory_limit_and_exception_reset_always_fall_back(self):
        with patch.object(reuse, 'MAX_BYTES', 0), reuse.operation():
            a = region_mesh(self.model.mesh, self.region['triangle_indices'])
            self.assertIsNot(a, region_mesh(self.model.mesh, self.region['triangle_indices']))
            self.assertEqual(reuse._active.get(), {})
        with self.assertRaisesRegex(RuntimeError, 'cancel'):
            with reuse.operation():
                with reuse.operation():
                    region_mesh(self.model.mesh, self.region['triangle_indices'])
                    raise RuntimeError('cancel')
        self.assertIsNone(reuse._active.get())

    def test_index_types_keep_original_numpy_selection_rules(self):
        with reuse.operation():
            region_mesh(self.model.mesh, [0])
            with self.assertRaises(IndexError):
                region_mesh(self.model.mesh, [0.])
            region_mesh(self.model.mesh, [1, 0])
            mask = [True, False] + [False] * (len(self.model.mesh.triangles)-2)
            selected = region_mesh(self.model.mesh, mask)
            self.assertEqual(len(selected.triangles), 1)
            with self.assertRaises(IndexError):
                region_mesh(self.model.mesh, (1, 0))

    def test_adjacency_tables_take_the_original_extraction_path(self):
        with reuse.operation():
            first = region_mesh(self.model.mesh, self.region['triangle_indices'])
            first.compute_adjacency_list()
            plain = region_mesh(self.model.mesh, self.region['triangle_indices'])
            self.assertIsNot(first, plain)
            self.assertFalse(plain.has_adjacency_list())
            self.model.mesh.compute_adjacency_list()
            with_table = region_mesh(self.model.mesh, self.region['triangle_indices'])
            self.assertIsNot(plain, with_table)
            self.assertTrue(with_table.adjacency_list)

    def test_saved_ply_is_complete_and_reused_only_after_hash_verification(self):
        with reuse.operation():
            manifest = workspace.snapshot(self.model, self.region, lambda: None, include_texture=False)
            data = json.loads(manifest.read_text(encoding='utf-8'))
            ply = manifest.parent / data['files']['mesh_file']
            disk = trimesh.load(ply, force='mesh', process=False)
            with patch.object(planner.trimesh, 'load', side_effect=AssertionError('decoded twice')):
                _, _, cached = planner._load_input(manifest)
            np.testing.assert_array_equal(cached.vertices, disk.vertices)
            np.testing.assert_array_equal(cached.faces, disk.faces)
            self.assertTrue(all((manifest.parent / name).is_file() for name in data['files'].values()))
            ply.write_bytes(ply.read_bytes() + b'changed')
            with self.assertRaisesRegex(ValueError, 'SHA-256'):
                planner._load_input(manifest)
        with self.assertRaisesRegex(ValueError, 'SHA-256'):
            planner._load_input(manifest)

    def test_cached_ply_mutation_forces_disk_read_and_manifest_is_rechecked(self):
        with reuse.operation():
            manifest = workspace.snapshot(self.model, self.region, lambda: None, include_texture=False)
            _, _, cached = planner._load_input(manifest)
            cached.vertices[0, 0] += .2
            original_load = planner.trimesh.load
            with patch.object(planner.trimesh, 'load', wraps=original_load) as loads:
                _, _, repaired = planner._load_input(manifest)
            self.assertEqual(loads.call_count, 1)
            self.assertFalse(np.array_equal(repaired.vertices, cached.vertices))
            data = json.loads(manifest.read_text(encoding='utf-8'))
            data['unit'] = 'millimetre'
            manifest.write_text(json.dumps(data), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'metres'):
                planner._load_input(manifest)

    def test_generate_and_final_validate_share_geometry_but_not_a_verdict(self):
        from femto.planning.planning_state import validate, binding
        with reuse.operation():
            plan = workspace.generate(self.model, self.region, self.fixture.head)
            local = workspace.canonical_region_mesh(self.model.mesh, self.region)
            validate(plan, self.model.mesh, self.region)
            self.assertIs(local, workspace.canonical_region_mesh(self.model.mesh, self.region))
            bad = deepcopy(plan)
            bad['surface_trajectory']['gap_report']['quality_constraint']['target_completion'] = 'complete'
            bad['plan_binding'] = binding(self.model.mesh, self.region, bad)
            with self.assertRaises(ValueError):
                validate(bad, self.model.mesh, self.region)
        self.assertIsNone(reuse._active.get())


if __name__ == '__main__':
    unittest.main()
