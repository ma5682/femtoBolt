"""Internal geometry snapshots defer appearance; final exports retain texture."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import open3d as o3d
from PIL import Image

import test_surface_workspace as fixture
from femto.annotation.region_texture import textured_region
from femto.planning import surface_workspace as workspace
from femto.planning.trajectory_export import load_trajectory
from femto.planning.surface_trajectory import to_player_plan


class PlanningSnapshotTextureTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture.SurfaceWorkspaceTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.model = self.fixture.model
        self.model.store.source = str(self.model.reconstruction/'scene_tsdf_mesh_m.ply')
        o3d.io.write_triangle_mesh(self.model.store.source, self.model.mesh)
        np.save(self.model.reconstruction/'mesh_triangle_uv.npy',
            np.zeros((len(self.model.mesh.triangles), 3, 2)))
        Image.new('RGB', (16, 16), (100, 150, 180)).save(self.model.reconstruction/'texture_multiview.png')
        self.region = self.fixture.region()

    def test_geometry_bytes_match_full_snapshot_and_source_texture_is_untouched(self):
        original = (self.model.reconstruction/'texture_multiview.png').read_bytes()
        full = workspace.snapshot(self.model, self.region, lambda: None)
        with patch('femto.annotation.region_texture.textured_region', side_effect=AssertionError('atlas decoded')):
            lean = workspace.snapshot(self.model, self.region, lambda: None, include_texture=False)
        a, b = (json.loads(path.read_text(encoding='utf-8')) for path in (full, lean))
        for name in ('mesh_file', 'point_cloud_file', 'boundary_file'):
            self.assertEqual((full.parent/a['files'][name]).read_bytes(), (lean.parent/b['files'][name]).read_bytes())
        self.assertEqual(a['region'], b['region'])
        self.assertEqual(a['geometry_sha256'], b['geometry_sha256'])
        self.assertEqual(a['appearance'], 'source_rgb_atlas_with_original_corner_uv')
        self.assertEqual(b['texture_metadata']['reason'], 'internal_planning_snapshot_texture_deferred_to_export')
        self.assertEqual((self.model.reconstruction/'texture_multiview.png').read_bytes(), original)

    def test_generation_defers_texture_but_final_export_and_replay_preserve_it(self):
        with patch('femto.annotation.region_texture.textured_region', side_effect=AssertionError('atlas decoded')):
            plan = workspace.generate(self.model, self.region, self.fixture.head)
        with patch('femto.annotation.region_texture.textured_region', wraps=textured_region) as export_texture:
            folder = workspace.export(self.model, self.region, plan)
        self.assertTrue(export_texture.called)
        trajectory = load_trajectory(folder, reproduce=True)
        self.assertEqual(to_player_plan(trajectory)['segments'], plan['segments'])
        manifest = json.loads(Path(trajectory.source['region_manifest']).read_text(encoding='utf-8'))
        self.assertEqual(manifest['appearance'], 'source_rgb_atlas_with_original_corner_uv')

    def test_absent_texture_keeps_the_original_manifest_bytes(self):
        for name in ('mesh_triangle_uv.npy', 'texture_multiview.png'):
            (self.model.reconstruction/name).unlink()
        full = workspace.snapshot(self.model, self.region, lambda: None)
        with patch('femto.annotation.region_texture.textured_region', side_effect=AssertionError('atlas decoded')):
            lean = workspace.snapshot(self.model, self.region, lambda: None, include_texture=False)
        self.assertEqual(full.read_bytes(), lean.read_bytes())
        manifest = json.loads(full.read_text(encoding='utf-8'))
        for name in manifest['files'].values():
            self.assertEqual((full.parent/name).read_bytes(), (lean.parent/name).read_bytes())


if __name__ == '__main__':unittest.main()
