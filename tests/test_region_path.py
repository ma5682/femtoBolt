import csv
import json
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
from femto.planning.region_path import raster_path, export_path, signature, load_path, path_summary


class Projection:
    def __init__(self):
        self.ids = np.zeros((3, 8), dtype=int)
        self.valid = np.ones((3, 8), dtype=bool)
        self.jump = False

    def pixel(self, x, y):
        return dict(triangle_index=int(self.ids[y, x]),
                    point_world_m=[x * .001, y * .001, .6 if self.jump and x >= 4 else .5])


class RegionPathTests(unittest.TestCase):
    def setUp(self):
        self.projection = Projection()
        self.mesh = SimpleNamespace(vertices=np.array([[0., 0, .5], [.1, 0, .5], [0, .1, .5],
                                                       [0, 0, .6], [.1, 0, .6], [0, .1, .6]]),
                                    triangles=np.array([[0, 1, 2], [3, 4, 5]]))
        self.region = dict(annotation_id='region-a', triangle_indices=[0], name='测试区域')

    def plan(self, **kwargs):
        return raster_path(self.projection, self.mesh, self.region, row_step_px=1, **kwargs)

    def test_rows_alternate_without_connectors(self):
        plan = self.plan()
        self.assertEqual(len(plan['segments']), 3)
        self.assertEqual(plan['segments'][0][0]['pixel'], [0, 0])
        self.assertEqual(plan['segments'][1][0]['pixel'], [7, 1])
        self.assertFalse(plan['execution_ready'])
        np.testing.assert_allclose(plan['segments'][0][0]['normal'], [0, 0, 1])

    def test_holes_and_unselected_faces_split(self):
        self.projection.valid[:, 3] = False
        self.projection.ids[:, 4] = 1
        plan = self.plan()
        self.assertEqual(len(plan['segments']), 6)
        for segment in plan['segments']:
            xs = [p['pixel'][0] for p in segment]
            self.assertTrue(all(x < 3 for x in xs) or all(x > 4 for x in xs))

    def test_depth_discontinuity_splits(self):
        self.projection.jump = True
        self.assertEqual(len(self.plan()['segments']), 6)

    def test_disconnected_mesh_faces_split_even_when_close(self):
        self.region['triangle_indices'] = [0, 1]
        self.projection.ids[:, 4:] = 1
        self.assertEqual(len(self.plan()['segments']), 6)

    def test_empty_and_cancellation(self):
        self.projection.valid[:] = False
        with self.assertRaises(ValueError):self.plan()
        self.projection.valid[:] = True
        def cancel():raise RuntimeError('cancelled')
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):self.plan(check=cancel)

    def test_export_preserves_coordinates_and_source(self):
        plan = self.plan()
        store = SimpleNamespace(source='scene.ply', digest='source-hash',
                                origin_reference={'frame': 0}, local_context={'scope': 'local'})
        with tempfile.TemporaryDirectory() as folder:
            result = export_path(plan, store, folder)
            payload = json.loads((result/'path.json').read_text(encoding='utf-8'))
            self.assertEqual(payload['segments'], plan['segments'])
            self.assertEqual(payload['geometry_sha256'], 'source-hash')
            self.assertEqual(payload['origin_reference'], {'frame': 0})
            with (result/'path.csv').open(encoding='utf-8', newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 24)
            self.assertAlmostEqual(float(rows[7]['segment_length_m']), .007)
            self.assertEqual(float(rows[8]['segment_length_m']), 0.)
            self.assertEqual(float(rows[0]['z_m']), .5)
        before = signature(self.region)
        self.region['triangle_indices'] = [1]
        self.assertNotEqual(before, signature(self.region))

    def test_real_projection_preserves_surface_and_pixel_mapping(self):
        import open3d as o3d
        from femto.annotation.fixed_rgb_geometry import FixedProjection
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector([[-.5, -.5, .5], [.5, -.5, .5],
                                                   [.5, .5, .5], [-.5, .5, .5]])
        mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]])
        projection = FixedProjection(mesh, np.array([[100., 0, 5], [0, 100., 5], [0, 0, 1]]),
                                     np.full((10, 10), 500, dtype=np.uint16))
        region = dict(annotation_id='real', triangle_indices=[0, 1])
        plan = raster_path(projection, mesh, region, row_step_px=2)
        self.assertEqual(len(plan['segments']), 5)
        for segment in plan['segments']:
            self.assertEqual(len(segment), 10)
            xyz = np.asarray([sample['xyz_m'] for sample in segment])
            np.testing.assert_allclose(xyz[:, 2], .5)
            np.testing.assert_allclose(projection.project(xyz), [p['pixel'] for p in segment], atol=1e-6)

    def test_statistics_exclude_transition_distances(self):
        summary = path_summary(self.plan())
        self.assertEqual(summary['sample_count'], 24)
        self.assertEqual(summary['segment_count'], 3)
        self.assertAlmostEqual(summary['total_segment_length_m'], .021)

    def test_load_checks_source_region_and_sample_geometry(self):
        plan = self.plan()
        store = SimpleNamespace(source='scene.ply', digest='hash', mesh=self.mesh,
                                origin_reference={'frame': 0}, local_context=None)
        with tempfile.TemporaryDirectory() as folder:
            path = export_path(plan, store, folder)/'path.json'
            original = path.read_text(encoding='utf-8')
            self.assertEqual(load_path(path, store, self.projection, self.region), plan)
            for field, value in [('geometry_sha256', 'wrong'), ('unit', 'mm'),
                                 ('origin_reference', {'frame': 2}), ('region_signature', 'wrong'),
                                 ('execution_ready', True), ('row_step_px', 0), ('max_step_mm', None)]:
                with self.subTest(field=field):
                    payload = json.loads(original);payload[field] = value
                    path.write_text(json.dumps(payload), encoding='utf-8')
                    with self.assertRaises(ValueError):load_path(path, store, self.projection, self.region)
            payload = json.loads(original)
            payload['segments'][0][0]['xyz_m'][0] += .001
            path.write_text(json.dumps(payload), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'samples'):
                load_path(path, store, self.projection, self.region)
            path.write_text(original, encoding='utf-8')
            self.projection.valid[:, 3] = False
            with self.assertRaisesRegex(ValueError, 'samples'):
                load_path(path, store, self.projection, self.region)


if __name__ == '__main__':unittest.main()
