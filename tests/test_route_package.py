import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import open3d as o3d
import trimesh
from region_path import export_path,signature
from export_history import scan_exports


class PackageTests(unittest.TestCase):
    def test_share_bundle_has_diagram_geometry_and_unchanged_path(self):
        mesh=o3d.geometry.TriangleMesh()
        mesh.vertices=o3d.utility.Vector3dVector([[0,0,.5],[.1,0,.5],[0,.1,.5]])
        mesh.triangles=o3d.utility.Vector3iVector([[0,1,2]])
        region={'annotation_id':'r','name':'区域A','triangle_indices':[0]}
        plan={'schema':'femto-region-raster-path-v1','row_step_px':10,'region_id':'r',
              'region_signature':signature(region),'segments':[[
                  {'xyz_m':[.01,.01,.5],'normal':[0,0,1],'pixel':[10,10]},
                  {'xyz_m':[.05,.01,.5],'normal':[0,0,1],'pixel':[50,10]}]]}
        store=SimpleNamespace(mesh=mesh,source='source.ply',digest='digest',origin_reference={},local_context=None)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            folder=export_path(plan,store,root/'region_paths',region,
                presentation=(np.full((80,80,3),100,np.uint8),np.zeros((80,80),int)))
            payload=json.loads((folder/'path.json').read_text(encoding='utf-8'))
            self.assertEqual(payload['segments'],plan['segments'])
            with zipfile.ZipFile(folder/'route_package.zip') as archive:
                self.assertEqual(set(archive.namelist()),{'region_with_path.glb','path.json','path.csv','README.txt','route_diagram.png'})
                self.assertEqual(archive.read('path.json'),(folder/'path.json').read_bytes())
            scene=trimesh.load(folder/'region_with_path.glb',force='scene',process=False)
            self.assertEqual(len(scene.geometry),5)
            for name,point in [('segment_starts_W0',[.01,.01,.5]),('segment_ends_W0',[.05,.01,.5])]:
                transform,geometry=scene.graph[name]
                np.testing.assert_allclose(scene.geometry[geometry].bounds.mean(axis=0),point,atol=1e-7)
                np.testing.assert_allclose(transform,np.diag([1,-1,-1,1]))
            row=scan_exports(root)[0]
            self.assertEqual(row['name'],'区域A')
            self.assertTrue(row['diagram'].exists());self.assertTrue(row['archive'].exists())


if __name__=='__main__':unittest.main()
