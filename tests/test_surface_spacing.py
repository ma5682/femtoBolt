import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from tests import test_coverage_inspection as fixtures
from femto.planning.surface_spacing import surface_spacing_plan
from femto.planning.surface_spacing_report import spacing_report,show_spacing_report
from femto.planning.planning_state import validate,parameters,state
from femto.planning.region_path import export_path,load_path,path_summary
from femto.annotation.fixed_rgb_geometry import polygon_contains
from femto.planning.region_path_guard import constrain_path

class SurfaceSpacingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.mesh,self.store,self.region,self.projection=fixtures.InspectionTests().surface(self.root)
    def tearDown(self):self.tmp.cleanup()
    def plan(self,**kwargs):return surface_spacing_plan(self.projection,self.mesh,self.region,spacing_mm=4,sample_step_mm=1,width_mm=2,boundary_margin_mm=0,tolerance_mm=1,**kwargs)
    def test_plane_retained_levels_and_distances(self):
        p=self.plan();self.assertEqual(parameters(p)['kind'],'surface_spacing')
        rows=p['spacing_report']['measurements'];self.assertTrue(rows)
        # Independent exact Euclidean distance to all source segments on this flat
        # fixture bounds any graph surface distance, even when trusted-face removal
        # changes the seed set and produces non-horizontal contours.
        from femto.planning.region_coverage import point_segment_distance
        for row in rows[::7]:
            sources=[c for c in p['segments'] if abs(c[0]['surface_level_m']-row['preceding_level_m'])<1e-9]
            exact=min(float(point_segment_distance(np.asarray([row['xyz_m']]),np.asarray(a['xyz_m']),np.asarray(b['xyz_m']))[0])*1000 for c in sources for a,b in zip(c,c[1:]))
            self.assertLessEqual(row['lower_bound_mm'],exact+1e-6)
            self.assertGreaterEqual(row['surface_graph_distance_mm'],exact-1e-6)
        self.assertFalse(p['physical_accuracy_verified']);self.assertIsNone(p['surface_mode']['fallback_mode'])
        self.assertEqual(path_summary(p)['connector_length_m'],0)
        self.assertTrue(all(not b['transition_planned'] and b['transition_time_s'] is None for b in p['chain_breaks']))
        self.assertTrue(all(v['motion_kind']=='scan' for c in p['segments'] for v in c))
    def test_holes_forbidden_and_disconnected_never_bridge(self):
        for kind,keep in [('hole',lambda c:~((c[:,0]>28)&(c[:,0]<52)&(c[:,1]>28)&(c[:,1]<52))),('split',lambda c:(c[:,0]<28)|(c[:,0]>52)),('forbidden',None)]:
            folder=self.root/kind;folder.mkdir();m,s,r,proj=fixtures.InspectionTests().surface(folder,keep)
            if kind=='forbidden':r=dict(r,exclusion_polygons=[[[30,30],[45,30],[45,45],[30,45]]])
            plan=surface_spacing_plan(proj,m,r,4,1,2,0,1)
            effective=dict(r,triangle_indices=plan['surface_mode']['trusted_triangle_indices'])
            again=copy.deepcopy(plan);guard=constrain_path(proj,m,effective,again,2,0)
            self.assertEqual(guard['rejected_edges'],0);self.assertEqual(again['segments'],plan['segments'])
            for c in plan['segments']:
                pts=np.array([p['pixel'] for p in c]);mid=(pts[:-1]+pts[1:])/2
                if kind=='forbidden':self.assertFalse(polygon_contains(mid,r['exclusion_polygons'][0]).any())
                else:self.assertFalse(((mid[:,0]>28)&(mid[:,0]<52)&((mid[:,1]>28)&(mid[:,1]<52) if kind=='hole' else True)).any())
    def test_missing_preceding_level_not_skipped(self):
        p=self.plan();levels=sorted(set(c[0]['surface_level_m'] for c in p['segments']));p['segments']=[c for c in p['segments'] if c[0]['surface_level_m']!=levels[1]]
        r=spacing_report(self.mesh,self.region,p,4,1)
        missing=[x for x in r['measurements'] if x['level_m']==levels[2]]
        self.assertTrue(missing);self.assertTrue(all(x['status']=='unmeasurable' and x['surface_graph_distance_mm'] is None for x in missing))
    def test_no_geometry_and_cancel_do_not_fallback(self):
        with patch.object(self.projection,'visible',side_effect=lambda points,**kw:np.zeros(len(points),bool)):
            with self.assertRaisesRegex(ValueError,'不会回退RGB'):self.plan()
        def cancel():raise RuntimeError('cancelled')
        with self.assertRaisesRegex(RuntimeError,'cancelled'):self.plan(check=cancel)
        with self.assertRaises(ValueError):surface_spacing_plan(self.projection,self.mesh,self.region,spacing_mm=float('nan'))
    def test_cancel_at_measurement_stage_keeps_input_untouched(self):
        before=copy.deepcopy(self.region);stages=[]
        def progress(stage):
            stages.append(stage)
            if stage=='测量实际保留路线的表面间距':raise RuntimeError('cancelled before report')
        with self.assertRaisesRegex(RuntimeError,'cancelled before report'):self.plan(progress=progress)
        self.assertEqual(self.region,before);self.assertEqual(len(stages),5)
    def test_binding_export_and_tamper(self):
        p=self.plan();validate(p,self.mesh,self.region)
        renamed=dict(self.region,name='Renamed');validate(p,self.mesh,renamed)
        changed=dict(self.region,exclusion_polygons=[[[30,30],[45,30],[45,45],[30,45]]])
        with self.assertRaises(ValueError):validate(p,self.mesh,changed)
        folder=export_path(p,self.store,self.root/'exports',self.region,(np.zeros((85,85,3),np.uint8),self.projection.ids))
        self.assertEqual(load_path(folder/'path.json',self.store,self.projection,self.region),p)
        import trimesh
        glb=trimesh.load(folder/'region_with_path.glb',force='scene')
        self.assertEqual(glb.metadata['surface_mode']['algorithm'],'mesh-surface-spacing-v1');self.assertFalse(glb.metadata['animation_included'])
        report=json.loads((folder/'surface_spacing_report.json').read_text(encoding='utf-8'))
        self.assertEqual(report['plan_binding'],p['plan_binding']);self.assertEqual(report['report'],p['spacing_report'])
        for key in ('spacing_report','surface_mode'):
            payload=json.loads((folder/'path.json').read_text(encoding='utf-8'));payload[key]['schema']='tampered'
            file=self.root/(key+'.json');file.write_text(json.dumps(payload),encoding='utf-8')
            with self.assertRaises(ValueError):load_path(file,self.store,self.projection,self.region)
