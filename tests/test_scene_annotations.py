import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import open3d as o3d
import trimesh
from femto.annotation.scene_annotations import AnnotationStore, distance_m, visible_region, region_mesh


class AnnotationTests(unittest.TestCase):
    def setUp(self):
        self.mesh = o3d.geometry.TriangleMesh()
        self.mesh.vertices = o3d.utility.Vector3dVector([[-.5,-.5,0],[.5,-.5,0],[0,.5,0],[-.5,-.5,.4],[.5,-.5,.4],[0,.5,.4]])
        self.mesh.triangles = o3d.utility.Vector3iVector([[0,1,2],[3,4,5]])
        self.points = [dict(point_world_m=[0,0,0],triangle_index=0),dict(point_world_m=[.003,.004,0],triangle_index=0)]

    def test_distance_in_metres(self):
        self.assertAlmostEqual(distance_m(*self.points), .005)

    def test_visible_region_excludes_occluded_back_surface(self):
        ids = visible_region(self.mesh, [[10,10],[90,10],[90,90],[10,90]], np.eye(4),np.eye(4),100,100)
        self.assertEqual(ids, [0])
        result = region_mesh(self.mesh, ids)
        self.assertEqual(len(result.triangles),1)
        self.assertEqual(len(result.vertices),3)
        self.assertEqual(len(self.mesh.triangles),2)

    def test_empty_and_incomplete_polygon(self):
        self.assertEqual(visible_region(self.mesh,[[0,0],[5,0],[5,5]],np.eye(4),np.eye(4),100,100),[])
        with self.assertRaises(ValueError):
            visible_region(self.mesh,[[0,0]],np.eye(4),np.eye(4),100,100)

    def test_roundtrip_exports_and_mismatch_are_atomic(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AnnotationStore(self.mesh, Path(tmp)/'source.ply')
            store.points = self.points
            store.regions = [dict(triangle_indices=[0])]
            path = store.save(Path(tmp)/'中文标注')
            self.assertNotEqual(path, store.save(path.parent.parent))
            saved_mesh = trimesh.load(path.parent/'region_1_mesh_m.ply',process=False)
            self.assertEqual(len(saved_mesh.faces),1)
            glb = trimesh.load(path.parent/'region_1_mesh_m.glb',force='scene',process=False)
            preview = glb.to_geometry()
            expected = np.asarray(saved_mesh.vertices)*[1,-1,-1]
            np.testing.assert_allclose(preview.vertices,expected,atol=1e-7)
            self.assertEqual(len(preview.faces),len(saved_mesh.faces))
            manifest=json.loads((path.parent/'region_1.json').read_text(encoding='utf-8'))
            self.assertEqual(manifest['unit'],'metre')
            self.assertEqual(manifest['geometry_sha256'],store.digest)
            np.testing.assert_allclose(np.asarray(manifest['W0_from_glb_scene'])[:3,:3] @
                                       preview.vertices.T,saved_mesh.vertices.T,atol=1e-7)
            self.assertTrue((path.parent/'region_1_boundary_m.csv').is_file())
            saved_cloud = trimesh.load(path.parent/'region_1_vertices_m.ply',process=False)
            self.assertEqual(len(saved_cloud.vertices),3)
            loaded = AnnotationStore(self.mesh,store.source)
            data = loaded.load(path)
            self.assertEqual([p['point_world_m'] for p in loaded.points],
                             [p['point_world_m'] for p in self.points])
            self.assertEqual([p['distance_to_origin_m'] for p in loaded.points], [0., .005])
            saved_points = list(loaded.points)
            self.assertAlmostEqual(data['measurements'][0]['distance_m'],.005)
            self.assertEqual(data['scale_validation'],'not_physically_validated')
            data['geometry_sha256'] = 'wrong'
            path.write_text(json.dumps(data),encoding='utf-8')
            with self.assertRaises(ValueError): loaded.load(path)
            self.assertEqual(loaded.points,saved_points)

    def test_bad_region_rejected_without_partial_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AnnotationStore(self.mesh,Path(tmp)/'source.ply')
            path = store.save(tmp)
            data = json.loads(path.read_text())
            data['regions'] = [dict(triangle_indices=[-1])]
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError): store.load(path)
            self.assertEqual(store.regions,[])
