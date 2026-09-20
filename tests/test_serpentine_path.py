import tempfile
import unittest
from pathlib import Path
import numpy as np
import open3d as o3d
from fixed_rgb_geometry import FixedProjection
from scene_annotations import AnnotationStore
from serpentine_path import serpentine_plan
from region_path import export_path,load_path


class SerpentineTests(unittest.TestCase):
    def setup_surface(self, root):
        vertices=[[x*.004,y*.004,.5] for y in range(21) for x in range(21)]
        faces=[]
        for y in range(20):
            for x in range(20):
                a=y*21+x;faces.extend([[a,a+1,a+22],[a,a+22,a+21]])
        mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),o3d.utility.Vector3iVector(faces))
        source=root/'mesh.ply';o3d.io.write_triangle_mesh(str(source),mesh)
        store=AnnotationStore(mesh,source)
        region=store.add_region(dict(triangle_indices=list(range(len(faces))),screen_polygon=[[0,0],[80,0],[80,80],[0,80]]))
        projection=FixedProjection(mesh,np.diag([500.,500.,1.]),np.full((85,85),500,np.uint16))
        return mesh,store,region,projection

    def test_horizontal_alternating_continuous_and_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);mesh,store,region,projection=self.setup_surface(root)
            plan=serpentine_plan(projection,mesh,region,8,20)
            self.assertEqual(len(plan['segments']),1)
            chain=plan['segments'][0];directions=[];last_y=None
            for a,b in zip(chain,chain[1:]):
                if b['motion_kind']=='scan':
                    self.assertEqual(a['pixel'][1],b['pixel'][1])
                    if b['pixel'][1]!=last_y:
                        directions.append(np.sign(b['pixel'][0]-a['pixel'][0]));last_y=b['pixel'][1]
                self.assertLessEqual(np.linalg.norm(np.subtract(a['pixel'],b['pixel'])),1.01)
            self.assertGreater(len(directions),3)
            self.assertEqual(directions,[1 if i%2==0 else -1 for i in range(len(directions))])
            folder=export_path(plan,store,root/'exports')
            loaded=load_path(folder/'path.json',store,projection,region)
            self.assertEqual(loaded['segments'],plan['segments'])

    def test_forbidden_detour_and_disconnected_break(self):
        with tempfile.TemporaryDirectory() as tmp:
            mesh,store,region,projection=self.setup_surface(Path(tmp))
            region=dict(region,exclusion_polygons=[[[30,30],[45,30],[45,45],[30,45]]])
            plan=serpentine_plan(projection,mesh,region,4,20,boundary_margin_mm=0)
            self.assertEqual(len(plan['segments']),1)
            for sample in plan['segments'][0]:
                x,y=sample['pixel'];self.assertFalse(30<=x<=45 and 30<=y<=45)
            region['exclusion_polygons']=[[[30,0],[45,0],[45,80],[30,80]]]
            plan=serpentine_plan(projection,mesh,region,4,20,boundary_margin_mm=0)
            self.assertGreater(len(plan['segments']),1)
            for chain in plan['segments']:
                xs=[p['pixel'][0] for p in chain]
                self.assertTrue(max(xs)<30 or min(xs)>45)


if __name__=='__main__':unittest.main()
