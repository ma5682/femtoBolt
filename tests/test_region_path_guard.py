import json
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import open3d as o3d
from region_coverage import coverage_plan
from region_path_guard import constrain_path,coverage_report_text
from region_path import export_path,load_path


class GuardTests(unittest.TestCase):
    def setUp(self):
        vertices=[[x*.001,y*.001,.5] for y in range(21) for x in range(21)]
        faces=[]
        for y in range(20):
            for x in range(20):
                a=y*21+x;faces.extend([[a,a+1,a+22],[a,a+22,a+21]])
        self.mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),o3d.utility.Vector3iVector(faces))
        self.region={'annotation_id':'r','triangle_indices':list(range(len(faces)))}
        self.projection=SimpleNamespace(project=lambda p:np.asarray(p)[:,:2]*1000)

    def test_width_clearance_retains_interior_and_splits_edges(self):
        plan=coverage_plan(self.projection,self.mesh,self.region,4,20,boundary_margin_mm=1)
        self.assertGreater(plan['boundary_check']['rejected_edges'],0)
        for segment in plan['segments']:
            for p in segment:
                x,y,_=p['xyz_m']
                self.assertGreaterEqual(min(x,y,.02-x,.02-y),.003-1e-9)
        self.assertIn('边缘遗漏',coverage_report_text(plan))
        self.assertFalse(plan['boundary_check']['contact_footprint_verified'])

    def test_too_wide_footprint_rejected_not_silently_relaxed(self):
        with self.assertRaisesRegex(ValueError,'无可用路线'):
            coverage_plan(self.projection,self.mesh,self.region,30,20,boundary_margin_mm=0)

    def test_hole_boundary_excludes_nearby_sweep(self):
        xyz=np.asarray(self.mesh.vertices)[np.asarray(self.mesh.triangles)].mean(axis=1)
        keep=~((xyz[:,0]>.008)&(xyz[:,0]<.012)&(xyz[:,1]>.008)&(xyz[:,1]<.012))
        self.region['triangle_indices']=np.flatnonzero(keep).tolist()
        def point(x,y):return dict(xyz_m=[x,y,.5],pixel=[x*1000,y*1000],triangle_index=0)
        plan={'segments':[[point(.007,.009),point(.007,.011)], [point(.003,.005),point(.004,.005)]]}
        report=constrain_path(self.projection,self.mesh,self.region,plan,4,0)
        self.assertEqual(report['retained_edges'],1)
        self.assertEqual(len(plan['segments']),1)

    def test_polygon_footprint_and_occlusion_gate(self):
        self.region['screen_polygon']=[[0,0],[15,0],[15,20],[0,20]]
        self.projection.k=np.diag([500.,500.,1.])
        self.projection.visible=lambda xyz,check:np.asarray(xyz)[:,1]<.015
        plan=coverage_plan(self.projection,self.mesh,self.region,4,20,boundary_margin_mm=0)
        self.assertTrue(plan['boundary_check']['polygon_footprint_checked'])
        for segment in plan['segments']:
            self.assertTrue(all(p['xyz_m'][1]<.015 for p in segment))
            self.assertTrue(all(p['pixel'][0]<15 for p in segment))

    def test_guarded_export_reload_and_report_tamper(self):
        plan=coverage_plan(self.projection,self.mesh,self.region,4,20,boundary_margin_mm=1)
        store=SimpleNamespace(mesh=self.mesh,source='mesh.ply',digest='hash',origin_reference={},local_context=None)
        with tempfile.TemporaryDirectory() as tmp:
            folder=export_path(plan,store,Path(tmp))
            self.assertTrue((folder/'coverage_report.txt').is_file())
            reread=load_path(folder/'path.json',store,self.projection,self.region)
            self.assertEqual(plan['segments'],reread['segments'])
            data=json.loads((folder/'path.json').read_text());data['boundary_check']['retained_edges']=999
            (folder/'path.json').write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'report mismatch'):load_path(folder/'path.json',store,self.projection,self.region)

    def test_margin_validation_and_cancel(self):
        for value in (-1,float('nan'),True):
            with self.assertRaises(ValueError):coverage_plan(self.projection,self.mesh,self.region,4,20,boundary_margin_mm=value)
        def cancel():raise RuntimeError('cancel')
        with self.assertRaisesRegex(RuntimeError,'cancel'):
            constrain_path(self.projection,self.mesh,self.region,{'segments':[]},4,0,cancel)


if __name__=='__main__':unittest.main()
