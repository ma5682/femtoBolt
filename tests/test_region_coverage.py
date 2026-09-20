import unittest
import numpy as np
import open3d as o3d
from types import SimpleNamespace
from femto.planning.region_coverage import coverage_plan,evaluate_coverage
from femto.planning.path_presentation import motion_position,render_diagram


class CoverageTests(unittest.TestCase):
    def setUp(self):
        self.mesh=o3d.geometry.TriangleMesh()
        vertices=[[x*.001,y*.001,.5] for y in range(11) for x in range(11)]
        faces=[]
        for y in range(10):
            for x in range(10):
                a=y*11+x;faces.extend([[a,a+1,a+12],[a,a+12,a+11]])
        self.mesh.vertices=o3d.utility.Vector3dVector(vertices)
        self.mesh.triangles=o3d.utility.Vector3iVector(faces)
        self.region={'annotation_id':'r','triangle_indices':list(range(len(faces)))}
        self.projection=SimpleNamespace(project=lambda xyz:np.asarray(xyz)[:,:2]*1000)

    def test_coverage_parameters_area_and_supplementation(self):
        plan=coverage_plan(self.projection,self.mesh,self.region,4.,25.)
        self.assertAlmostEqual(plan['spacing_mm'],3.)
        report=plan['coverage']
        self.assertAlmostEqual(report['target_mesh_area_m2'],.0001)
        self.assertGreaterEqual(report['sampled_coverage_percent'],report['initial_sampled_coverage_percent'])
        self.assertLessEqual(report['sampled_coverage_percent'],100.+1e-8)
        self.assertFalse(report['full_surface_coverage_verified'])

    def test_nearby_disconnected_surface_does_not_gain_coverage(self):
        other=o3d.geometry.TriangleMesh(self.mesh);other.translate([0,0,.0001])
        both=self.mesh+other
        plan=coverage_plan(self.projection,self.mesh,self.region,4.,25.)
        region={'triangle_indices':list(range(len(both.triangles)))}
        report=evaluate_coverage(both,region,plan,4.)
        self.assertEqual(sum(report['covered_samples_per_face'][200:]),0)

    def test_motion_uses_metric_distance_and_no_gap_connector(self):
        segment=[{'xyz_m':[0,0,0],'pixel':[0,0]}, {'xyz_m':[.02,0,0],'pixel':[100,0]}]
        plan={'segments':[segment,segment]}
        self.assertEqual(motion_position(plan,.5),(0,[50.,0.],'moving'))
        self.assertEqual(motion_position(plan,1.25),(0,None,'gap'))
        self.assertEqual(motion_position(plan,3.)[2],'finished')

    def test_invalid_parameter_and_cancellation(self):
        with self.assertRaises(ValueError):coverage_plan(self.projection,self.mesh,self.region,0,20)
        def cancel():raise RuntimeError('cancelled')
        with self.assertRaisesRegex(RuntimeError,'cancelled'):
            coverage_plan(self.projection,self.mesh,self.region,4,20,cancel)

    def test_region_smaller_than_spacing_gets_supplementary_path(self):
        self.mesh.scale(.1,center=[0,0,.5])
        plan=coverage_plan(self.projection,self.mesh,self.region,10.,20.)
        self.assertTrue(plan['segments'])
        self.assertGreater(plan['coverage']['supplementary_segments'],0)
        self.assertEqual(plan['coverage']['initial_sampled_coverage_percent'],0.)
        self.assertGreater(plan['coverage']['sampled_coverage_percent'],0.)

    def test_export_and_reload_coverage_plan(self):
        import tempfile
        import zipfile
        from pathlib import Path
        from femto.planning.region_path import export_path,load_path
        plan=coverage_plan(self.projection,self.mesh,self.region,4,25)
        store=SimpleNamespace(mesh=self.mesh,source='mesh.ply',digest='hash',origin_reference={},local_context=None)
        with tempfile.TemporaryDirectory() as folder:
            output=export_path(plan,store,folder,self.region,
                presentation=(np.zeros((11,11,3),np.uint8),np.zeros((11,11),int)))
            reread=load_path(output/'path.json',store,self.projection,self.region)
            self.assertEqual(reread['coverage'],plan['coverage'])
            self.assertTrue((output/'coverage_diagram.png').is_file())
            with zipfile.ZipFile(output/'route_package.zip') as archive:
                self.assertIn('coverage_diagram.png',archive.namelist())


if __name__=='__main__':unittest.main()
