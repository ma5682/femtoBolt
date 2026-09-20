import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import open3d as o3d
from scene_annotations import AnnotationStore
from fixed_rgb_geometry import FixedProjection,polygon_contains
from region_coverage import coverage_plan
from region_path import export_path,load_path,raster_path
from region_exclusions import allowed_region


class ExclusionTests(unittest.TestCase):
    def setUp(self):
        vertices=[[x*.002,y*.002,.5] for y in range(21) for x in range(21)]
        faces=[]
        for y in range(20):
            for x in range(20):
                a=y*21+x;faces.extend([[a,a+1,a+22],[a,a+22,a+21]])
        self.mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),o3d.utility.Vector3iVector(faces))
        self.projection=FixedProjection(self.mesh,np.array([[500.,0,0],[0,500.,0],[0,0,1]]),np.full((50,50),500,np.uint16))
        self.region=dict(annotation_id='r',triangle_indices=list(range(800)),screen_polygon=[[0,0],[40,0],[40,40],[0,40]],
            exclusion_polygons=[[[16,16],[24,16],[24,24],[16,24]]])

    def test_paths_avoid_forbidden_surface_and_replay(self):
        plan=coverage_plan(self.projection,self.mesh,self.region,4,20,boundary_margin_mm=0)
        removed=set(plan['exclusions']['removed_triangle_indices'])
        self.assertTrue(removed)
        for segment in plan['segments']:
            self.assertFalse(polygon_contains(np.array([p['pixel'] for p in segment]),self.region['exclusion_polygons'][0]).any())
            self.assertFalse(any(p['triangle_index'] in removed for p in segment))
        store=SimpleNamespace(mesh=self.mesh,source='mesh.ply',digest='hash',origin_reference={},local_context=None)
        with tempfile.TemporaryDirectory() as tmp:
            folder=export_path(plan,store,Path(tmp),self.region)
            self.assertEqual(load_path(folder/'path.json',store,self.projection,self.region)['segments'],plan['segments'])
            import trimesh
            scene=trimesh.load(folder/'region_with_path.glb',force='scene',process=False)
            self.assertIn('forbidden_region_W0',scene.graph.nodes)

    def test_save_load_undo_invalidates_signature(self):
        from region_path import signature
        with tempfile.TemporaryDirectory() as tmp:
            store=AnnotationStore(self.mesh,Path(tmp)/'mesh.ply')
            r=store.add_region(dict(self.region,exclusion_polygons=[]));before=signature(r)
            store.set_exclusions(0,self.region['exclusion_polygons'])
            self.assertNotEqual(signature(store.regions[0]),before)
            path=store.save(Path(tmp)/'saved')
            other=AnnotationStore(self.mesh,store.source);other.load(path)
            self.assertEqual(other.regions[0]['exclusion_polygons'],self.region['exclusion_polygons'])
            store.undo();self.assertEqual(signature(store.regions[0]),before)

    def test_full_exclusion_and_legacy_path_refused(self):
        region=dict(self.region,exclusion_polygons=[self.region['screen_polygon']])
        with self.assertRaisesRegex(ValueError,'全部'):allowed_region(self.projection,self.mesh,region)
        with self.assertRaisesRegex(ValueError,'禁止'):raster_path(self.projection,self.mesh,self.region)


if __name__=='__main__':unittest.main()
