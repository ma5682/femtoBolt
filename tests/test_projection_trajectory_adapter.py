import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import open3d as o3d
import trimesh

from tests import test_serpentine_path as serpentine_fixtures
from femto.annotation.scene_annotations import mesh_digest, region_mesh
from femto.planning.route_motion import MotionPlayer
from femto.planning.route_optimization import optimized_serpentine_plan
from femto.planning.planning_state import trajectory_binding
from femto.planning.surface_spacing import surface_spacing_plan
from femto.planning.surface_trajectory import (
    _point_on_triangle, _relocate_faces, canonical_json_bytes, from_projection_plan, to_player_plan,
)
from femto.planning.trajectory_export import export_trajectory, load_trajectory


def fixture(tmp):
    vertices = np.asarray([[0, 0, .5], [.01, 0, .5], [.01, .01, .5], [0, .01, .5]], float)
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], int)
    legacy = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),
                                      o3d.utility.Vector3iVector(faces.astype(np.int32)))
    region = {
        'annotation_id': 'adapter-region', 'triangle_indices': [0, 1],
        'screen_polygon': [[0, 0], [20, 0], [20, 20], [0, 20]],
        'work_screen_polygon': [[0, 0], [20, 0], [20, 20], [0, 20]],
        'forbidden_screen_polygon': [], 'exclusion_polygons': [],
        'selection_method': 'work_only',
        'view_reference': {
            'undistorted_intrinsic': [[100, 0, 10], [0, 100, 10], [0, 0, 1]],
            'image_size': [20, 20], 'rgb_sha256': '0' * 64,
            'depth_sha256': '1' * 64, 'reference_frame_id': 0,
        },
    }
    mesh_path = Path(tmp) / 'region_1_mesh_m.ply'
    mesh_path.write_bytes(trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(file_type='ply'))
    geometry_sha256 = mesh_digest(legacy)
    manifest = {
        'schema': 'femto-region-export-v1', 'region': region,
        'geometry_sha256': geometry_sha256, 'coordinate_frame': 'world_rdf', 'unit': 'metre',
        'origin_reference': {'kind': 'initial_color_camera', 'reference_frame_id': 0},
        'files': {'mesh_file': mesh_path.name},
        'file_sha256': {mesh_path.name: hashlib.sha256(mesh_path.read_bytes()).hexdigest()},
    }
    manifest_path = Path(tmp) / 'region_1.json'
    manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
    point = lambda xyz, pixel, triangle, kind='scan': {
        'xyz_m': xyz, 'pixel': pixel, 'triangle_index': triangle,
        'motion_kind': kind, 'normal': [0, 0, 1],
    }
    plan = {
        'schema': 'femto-region-coverage-path-v4',
        'segments': [
            [point([.006, .002, .5], [4, 4], 0),
             point([.008, .003, .5], [8, 5], 0, 'connector')],
            [point([.002, .006, .5], [4, 14], 0),
             point([.003, .008, .5], [6, 16], 1)],
        ],
        'chain_breaks': [{'from_chain': 0, 'to_chain': 1,
                          'reason': 'no_path_in_allowed_surface_graph',
                          'explanation': 'surface graph disconnected',
                          'transition_planned': False, 'transition_time_s': None}],
        'plan_binding': {'algorithm_version': 'serpentine-v4', 'result_sha256': 'old-plan-hash'},
    }
    store = SimpleNamespace(mesh=legacy, origin_reference=manifest['origin_reference'],
                            digest=geometry_sha256, data_layers=None)
    return plan, region, store, manifest_path


class ProjectionTrajectoryAdapterTests(unittest.TestCase):
    def test_mapping_relocation_playback_and_old_plan_immutability(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan, region, store, manifest = fixture(tmp)
            before = canonical_json_bytes(plan)
            old_json = Path(tmp) / 'path.json';old_csv = Path(tmp) / 'path.csv'
            old_json.write_bytes(before);old_csv.write_bytes(b'segment,x,y,z\n0,0,0,0.5\n')
            file_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in (old_json, old_csv)}
            trajectory = from_projection_plan(plan, region, store, manifest)
            self.assertEqual(canonical_json_bytes(plan), before)
            self.assertEqual({path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in (old_json, old_csv)}, file_hashes)
            points = [point for item in trajectory.ordered_items if hasattr(item, 'points')
                      for point in item.points]
            self.assertEqual([point['triangle_index'] for point in points], [0, 0, 1, 1])
            self.assertEqual(points[2]['face_index_source'], 'relocated_geometrically')
            self.assertEqual(points[1]['role'], 'connector')
            self.assertNotIn('"normal"', canonical_json_bytes(trajectory.to_dict()).decode('utf-8'))
            binding = trajectory_binding(trajectory)
            self.assertEqual(binding['algorithm_version'], 'projection-adapter-v1')
            self.assertFalse(binding['execution_ready'])
            player_plan = to_player_plan(trajectory)
            self.assertEqual(player_plan['segments'][0][1]['motion_kind'], 'connector')
            player = MotionPlayer(player_plan)
            player.play(); player.advance(10)
            self.assertTrue(player.at_break)
            output = export_trajectory(trajectory, Path(tmp) / 'results', include_visuals=False)
            self.assertEqual(load_trajectory(output).to_dict(), trajectory.to_dict())

    def test_rebuilt_mesh_matches_export_and_outside_point_rejects_whole_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan, region, store, manifest = fixture(tmp)
            exported = from_projection_plan(plan, region, store, manifest)
            rebuilt = from_projection_plan(plan, region, store)
            exported_faces = [point['triangle_index'] for item in exported.ordered_items if hasattr(item, 'points')
                              for point in item.points]
            rebuilt_faces = [point['triangle_index'] for item in rebuilt.ordered_items if hasattr(item, 'points')
                             for point in item.points]
            self.assertEqual(exported_faces, rebuilt_faces)
            self.assertEqual(rebuilt.source['bound_region_mesh']['source'], 'rebuilt_from_source_mesh')
            shared = copy.deepcopy(plan)
            shared['segments'][0][0].update(xyz_m=[0, 0, .5], triangle_index=99)
            shared_export = from_projection_plan(shared, region, store, manifest)
            shared_rebuild = from_projection_plan(shared, region, store)
            self.assertEqual(shared_export.ordered_items[0].points[0]['triangle_index'], 0)
            self.assertEqual(shared_rebuild.ordered_items[0].points[0]['triangle_index'], 0)
            damaged = copy.deepcopy(plan)
            damaged['segments'][0][0]['xyz_m'] = [.2, .2, .5]
            with self.assertRaisesRegex(ValueError, 'cannot be bound'):
                from_projection_plan(damaged, region, store, manifest)


def real_planner_fixture(root):
    """Real fixed-RGB planners on a sub-region, so global and local face indices differ."""
    root = Path(root)
    mesh, store, _, projection = serpentine_fixtures.SerpentineTests().setup_surface(root)
    store.origin_reference = {'kind': 'initial_color_camera', 'reference_frame_id': 0,
                              'session_id': 'synthetic-fixture'}
    faces = np.asarray(mesh.triangles)
    centers = np.asarray(mesh.vertices)[faces].mean(axis=1) * 1000
    ids = [int(index) for index in np.flatnonzero(centers[:, 0] > 20)]
    region = store.add_region(dict(
        triangle_indices=ids, screen_polygon=[[20, 0], [80, 0], [80, 80], [20, 80]],
        exclusion_polygons=[[[40, 0], [52, 0], [52, 80], [40, 80]]],
        view_reference=dict(undistorted_intrinsic=projection.k.tolist(), image_size=[85, 85],
                            rgb_sha256='0' * 64, depth_sha256='1' * 64,
                            reference_frame_id=0, session_id='synthetic-fixture')))
    return mesh, store, region, projection


class RealPlannerAdapterTests(unittest.TestCase):
    def adapt_and_check(self, plan, region, store, expected_algorithm):
        mapping = {global_id: local for local, global_id
                   in enumerate(sorted(set(region['triangle_indices'])))}
        before = canonical_json_bytes(plan)
        trajectory = from_projection_plan(plan, region, store)
        self.assertEqual(canonical_json_bytes(plan), before)
        self.assertEqual(trajectory.planner['kind'], 'projection_fixed_rgb')
        self.assertEqual(trajectory.planner['parameters']['source_algorithm_version'], expected_algorithm)
        self.assertEqual(trajectory.source['bound_region_mesh']['source'], 'rebuilt_from_source_mesh')
        segments = [item for item in trajectory.ordered_items if hasattr(item, 'points')]
        gaps = [item for item in trajectory.ordered_items if not hasattr(item, 'points')]
        self.assertEqual(len(segments), len(plan['segments']))
        self.assertEqual(len(gaps), len(plan['segments']) - 1)
        self.assertTrue(all(gap.reason in ('row_turn', 'unsupported_geometry') for gap in gaps))
        old_points = [point for chain in plan['segments'] for point in chain]
        new_points = [point for segment in segments for point in segment.points]
        self.assertEqual([point['xyz_m'] for point in new_points],
                         [[float(value) for value in point['xyz_m']] for point in old_points])
        self.assertEqual([point['triangle_index'] for point in new_points],
                         [mapping[point['triangle_index']] for point in old_points])
        self.assertTrue(all(point['face_index_source'] == 'mapped_from_source_mesh_index'
                            for point in new_points))
        self.assertEqual(trajectory.gap_report['relocated_point_count'], 0)
        self.assertEqual([point['role'] for point in new_points],
                         [point.get('motion_kind', 'scan') for point in old_points])
        # Independent geometric relocation agrees with the mapped index, or the point sits on a
        # shared edge where both faces support it and relocation picks the smaller index.
        local = region_mesh(store.mesh, region['triangle_indices'])
        local_mesh = trimesh.Trimesh(vertices=np.asarray(local.vertices),
                                     faces=np.asarray(local.triangles), process=False)
        vertices = np.asarray(local_mesh.vertices, float)
        faces = np.asarray(local_mesh.faces)
        relocated = _relocate_faces([np.asarray(point['xyz_m'], float) for point in new_points], local_mesh)
        for point, found in zip(new_points, relocated):
            self.assertIsNotNone(found)
            self.assertLessEqual(found, point['triangle_index'])
            self.assertTrue(_point_on_triangle(point['xyz_m'], vertices, faces[found]))
            self.assertTrue(_point_on_triangle(point['xyz_m'], vertices, faces[point['triangle_index']]))
        player = MotionPlayer(to_player_plan(trajectory))
        player.play(); player.advance(1e9)
        self.assertEqual(player.at_break, len(segments) > 1)
        self.assertEqual(player.finished, len(segments) == 1)
        for _ in gaps:
            self.assertTrue(player.play()); player.advance(1e9)
        self.assertTrue(player.finished)
        return trajectory

    def test_serpentine_v4_and_surface_spacing_plans_adapt_map_and_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mesh, store, region, projection = real_planner_fixture(root)
            v4 = optimized_serpentine_plan(projection, mesh, region, 4, 20)
            spacing = surface_spacing_plan(projection, mesh, region, 4, 1, 2, 0, 1)
            self.assertGreater(len(v4['segments']), 1)
            self.assertGreater(len(spacing['segments']), 1)
            rebuilt = {'v4': self.adapt_and_check(v4, region, store, 'serpentine-v4'),
                       'spacing': self.adapt_and_check(spacing, region, store, 'mesh-surface-spacing-v1')}
            saved = Path(store.save(root / 'annotations'))
            if saved.is_dir():
                saved = saved / 'annotations.json'
            data = json.loads(saved.read_text(encoding='utf-8-sig'))
            record = next(item for item in data['regions'] if item['annotation_id'] == region['annotation_id'])
            manifest = saved.parent / record['region_manifest_file']
            for name, plan in (('v4', v4), ('spacing', spacing)):
                exported = from_projection_plan(plan, region, store, manifest)
                self.assertEqual(exported.source['bound_region_mesh']['source'], 'exported_ply')
                self.assertEqual([item.to_dict() for item in exported.ordered_items],
                                 [item.to_dict() for item in rebuilt[name].ordered_items])
                folder = export_trajectory(exported, root / ('results_' + name), include_visuals=False)
                self.assertEqual(load_trajectory(folder).to_dict(), exported.to_dict())


if __name__ == '__main__':
    unittest.main()
