import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import open3d as o3d
import trimesh
from metric_region_path import metric_path
from region_path import export_path, load_path


def grid(tilt=0., hole=False):
    vertices=[[x*.001,y*.001,.5+y*.001*tilt] for y in range(11) for x in range(11)]
    faces=[]
    for y in range(10):
        for x in range(10):
            if hole and 3<=x<7 and 3<=y<7:continue
            a=y*11+x;faces.extend([[a,a+1,a+12],[a,a+12,a+11]])
    mesh=o3d.geometry.TriangleMesh()
    mesh.vertices=o3d.utility.Vector3dVector(vertices)
    mesh.triangles=o3d.utility.Vector3iVector(faces)
    return mesh


class Projection:
    def project(self,points):return np.asarray(points)[:,:2]*1000


class MetricPathTests(unittest.TestCase):
    def plan(self,mesh,spacing=2.,step=.4,check=lambda:None):
        region=dict(annotation_id='r',triangle_indices=list(range(len(mesh.triangles))))
        return metric_path(Projection(),mesh,region,spacing,step,check),region

    def test_plane_spacing_and_samples(self):
        plan,_=self.plan(grid())
        self.assertEqual(len(plan['segments']),5)
        rows=[]
        for segment in plan['segments']:
            xyz=np.asarray([s['xyz_m'] for s in segment])
            rows.append(xyz[0,1]);np.testing.assert_allclose(xyz[:,1],xyz[0,1],atol=1e-10)
            np.testing.assert_allclose(xyz[:,2],.5)
            self.assertLessEqual(np.linalg.norm(np.diff(xyz,axis=0),axis=1).max(),.0004+1e-10)
        np.testing.assert_allclose(sorted(rows),[.001,.003,.005,.007,.009],atol=1e-10)

    def test_tilt_uses_surface_distance_not_projected_y(self):
        plan,_=self.plan(grid(tilt=1.))
        rows=sorted(segment[0]['xyz_m'][1] for segment in plan['segments'])
        np.testing.assert_allclose(np.diff(rows)*np.sqrt(2),.002,atol=1e-10)
        for segment in plan['segments']:
            xyz=np.asarray([s['xyz_m'] for s in segment])
            np.testing.assert_allclose(xyz[:,2],.5+xyz[:,1])

    def test_hole_is_not_bridged(self):
        plan,_=self.plan(grid(hole=True))
        for segment in plan['segments']:
            xyz=np.asarray([s['xyz_m'] for s in segment])
            mid=(xyz[:-1]+xyz[1:])/2
            inside=(mid[:,0]>.003+1e-10)&(mid[:,0]<.007-1e-10)&(mid[:,1]>.003+1e-10)&(mid[:,1]<.007-1e-10)
            self.assertFalse(inside.any())

    def test_export_roundtrip_scene_and_metric_plan(self):
        mesh=grid();plan,region=self.plan(mesh)
        store=SimpleNamespace(mesh=mesh,source='source.ply',digest='hash',origin_reference={'frame':0},local_context=None)
        with tempfile.TemporaryDirectory() as root:
            folder=export_path(plan,store,root,region)
            self.assertEqual(load_path(folder/'path.json',store,Projection(),region),plan)
            payload=json.loads((folder/'path.json').read_text(encoding='utf-8'))
            scene=trimesh.load(folder/'region_with_path.glb',force='scene')
            self.assertEqual(len(scene.geometry),5)
            transform,geom=scene.graph['selected_region_W0']
            np.testing.assert_allclose(transform,np.diag([1,-1,-1,1]))
            exported=scene.geometry[geom]
            np.testing.assert_allclose(exported.bounds,np.asarray(mesh.vertices)[[0,-1]],atol=1e-7)
            self.assertEqual(len(exported.faces),len(mesh.triangles))
            self.assertEqual(payload['visualization']['path_display_radius_m'],.00015)
            path_transform,path_geom=scene.graph['path_display_W0']
            np.testing.assert_allclose(path_transform,transform)
            self.assertGreater(len(scene.geometry[path_geom].faces),0)
            payload['spacing_mm']=3.
            (folder/'path.json').write_text(json.dumps(payload),encoding='utf-8')
            with self.assertRaises(ValueError):load_path(folder/'path.json',store,Projection(),region)

    def test_invalid_and_cancelled(self):
        for value in (0,float('nan'),True,101):
            with self.assertRaises(ValueError):self.plan(grid(),spacing=value)
        def cancel():raise RuntimeError('cancelled')
        with self.assertRaisesRegex(RuntimeError,'cancelled'):self.plan(grid(),check=cancel)

    def test_disconnected_islands_have_independent_paths(self):
        mesh=grid();other=grid();other.translate([.1,0,0]);mesh+=other
        plan,_=self.plan(mesh)
        self.assertEqual(len(plan['segments']),10)
        for segment in plan['segments']:
            xs=np.asarray([sample['xyz_m'][0] for sample in segment])
            self.assertLess(xs.max()-xs.min(),.011)

    def test_closed_component_is_rejected(self):
        mesh=o3d.geometry.TriangleMesh.create_sphere(radius=.01,resolution=5)
        mesh.translate([0,0,.5])
        with self.assertRaisesRegex(ValueError,'no boundary'):self.plan(mesh)


if __name__=='__main__':unittest.main()
