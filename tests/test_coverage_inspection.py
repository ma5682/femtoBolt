import copy,json,tempfile,unittest,hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import open3d as o3d
from tests import test_serpentine_path as fixtures
from scene_annotations import AnnotationStore
from fixed_rgb_geometry import FixedProjection,polygon_contains
from coverage_inspection import inspected_serpentine_plan
from coverage_inspector import render_map,crop_bounds,show_inspector
from region_path import export_path,load_path
from planning_state import validate,parameters,state

class InspectionTests(unittest.TestCase):
    def surface(self,root,keep=None):
        mesh,_,_,_=fixtures.SerpentineTests().setup_surface(root)
        if keep:
            faces=np.asarray(mesh.triangles);centers=np.asarray(mesh.vertices)[faces].mean(axis=1)*1000
            mesh.triangles=o3d.utility.Vector3iVector(faces[keep(centers)])
        path=root/'mesh.ply';o3d.io.write_triangle_mesh(str(path),mesh)
        store=AnnotationStore(mesh,path)
        region=store.add_region(dict(triangle_indices=list(range(len(mesh.triangles))),screen_polygon=[[0,0],[80,0],[80,80],[0,80]],exclusion_polygons=[]))
        projection=FixedProjection(mesh,np.diag([500.,500.,1.]),np.full((85,85),500,np.uint16))
        return mesh,store,region,projection

    def test_hole_and_denominator_no_fabricated_area(self):
        with tempfile.TemporaryDirectory() as tmp:
            mesh,store,region,p=self.surface(Path(tmp),lambda c:~((c[:,0]>28)&(c[:,0]<52)&(c[:,1]>28)&(c[:,1]<52)))
            plan=inspected_serpentine_plan(p,mesh,region,4,20)
            report=plan['coverage_inspection'];samples=report['samples']
            holes=[o for o in report['omissions'] if o['reason']=='no_geometry' and o['bbox_pixel'][0]<=40<=o['bbox_pixel'][2]]
            self.assertTrue(holes)
            for entry in holes:
                self.assertIsNone(entry['sampled_area_m2']);self.assertIsNone(entry['focus_xyz_m']);self.assertFalse(entry['in_denominator'])
            self.assertIsNone(report['denominator']['unknown_geometry_area_m2'])
            self.assertAlmostEqual(sum(s['represented_area_m2'] for s in samples if s['eligible']),plan['coverage']['target_mesh_area_m2'])
            self.assertAlmostEqual(sum(s['represented_area_m2'] for s in samples if s['covered']),plan['coverage']['sampled_covered_area_m2'])
            missing=[i for o in report['omissions'] for i in o['sample_indices']]
            self.assertEqual(sorted(missing),[i for i,s in enumerate(samples) if s['eligible'] and not s['covered']])
            for sample in samples:
                if sample['covered']:self.assertFalse(28<sample['pixel'][0]<52 and 28<sample['pixel'][1]<52)
            image=render_map(np.zeros((85,85,3),np.uint8),plan,item=holes[0],zoom=True,header=False)
            box=crop_bounds(report,holes[0]);self.assertEqual(image.size,(box[2]-box[0],box[3]-box[1]))

    def test_forbidden_whole_footprint_and_export_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);mesh,store,region,p=self.surface(root)
            region=dict(region,exclusion_polygons=[[[30,30],[45,30],[45,45],[30,45]]])
            plan=inspected_serpentine_plan(p,mesh,region,4,20,boundary_margin_mm=1)
            self.assertTrue(plan['boundary_check']['accepted_footprint_checked'])
            # Independent flat-plane footprint witness at every endpoint and midpoint.
            centers=[]
            for chain in plan['segments']:
                points=np.asarray([s['xyz_m'] for s in chain]);centers.extend(points);centers.extend((points[:-1]+points[1:])/2)
            angles=np.arange(32)*2*np.pi/32
            circle=np.c_[np.cos(angles),np.sin(angles),np.zeros(32)]*.003
            cloud=(np.asarray(centers)[:,None,:]+circle).reshape(-1,3);uv=p.project(cloud)
            self.assertTrue(polygon_contains(uv,region['screen_polygon']).all())
            self.assertFalse(polygon_contains(uv,region['exclusion_polygons'][0]).any())
            self.assertTrue(any(o['reason']=='forbidden' for o in plan['coverage_inspection']['omissions']))
            for sample in plan['coverage_inspection']['samples']:
                if sample['covered']:self.assertFalse(polygon_contains(np.asarray([sample['pixel']]),region['exclusion_polygons'][0])[0])
            folder=export_path(plan,store,root/'exports',region,(np.zeros((85,85,3),np.uint8),p.ids))
            self.assertEqual(load_path(folder/'path.json',store,p,region),plan)
            payload=json.loads((folder/'path.json').read_text(encoding='utf-8'))
            for name in ('coverage_inspection.json','coverage_omissions.csv','coverage_diagram.png'):
                self.assertEqual(payload['files'][name],hashlib.sha256((folder/name).read_bytes()).hexdigest())
            for key in ('samples','omissions','denominator','layers'):
                tampered=copy.deepcopy(plan);tampered['coverage_inspection'][key]=None
                with self.subTest(key=key),self.assertRaises(ValueError):validate(tampered,mesh,region)
            validate(plan,mesh,dict(region,name='rename only'))
            self.assertFalse(state(region,plan,request=dict(kind='serpentine_v2',values={}))['can_use'])

    def test_narrow_channel_is_not_bridged(self):
        with tempfile.TemporaryDirectory() as tmp:
            mesh,store,region,p=self.surface(Path(tmp),lambda c:(c[:,0]<28)|(c[:,0]>52)|((c[:,1]>36)&(c[:,1]<44)))
            plan=inspected_serpentine_plan(p,mesh,region,8,20)
            self.assertGreater(len(plan['segments']),1)
            for chain in plan['segments']:
                for sample in chain:self.assertFalse(28<sample['pixel'][0]<52)
            report=plan['coverage_inspection']
            channel=[s for s in report['samples'] if s['eligible'] and 30<s['pixel'][0]<50 and 36<s['pixel'][1]<44]
            self.assertTrue(channel)
            self.assertTrue(all(not s['covered'] for s in channel))
            self.assertTrue(any(s['reason'] in ('boundary_margin','boundary_clearance') for s in channel))

    def test_unseeded_disconnected_component_and_unknown_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            mesh,store,region,p=self.surface(Path(tmp),lambda c:(c[:,0]<40)|((c[:,0]>64)&(c[:,1]<8)))
            plan=inspected_serpentine_plan(p,mesh,region,8,20)
            samples=plan['coverage_inspection']['samples']
            island=[s for s in samples if s['eligible'] and s['pixel'][0]>64]
            self.assertTrue(island);self.assertTrue(all(not s['covered'] for s in island))
            self.assertTrue(any(s['reason']=='disconnected' for s in island))
            from coverage_inspection import EVIDENCE
            self.assertIn('未知',EVIDENCE['unknown']);self.assertIn('未知',EVIDENCE['algorithm_limit'])

    def test_selected_faces_and_roi_eligibility_are_distinct_from_unknown_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            mesh,store,region,p=self.surface(Path(tmp))
            centers=np.asarray(mesh.vertices)[np.asarray(mesh.triangles)].mean(axis=1)
            region=dict(region,triangle_indices=np.flatnonzero(centers[:,0]<.060).tolist(),screen_polygon=[[0,0],[70,0],[70,70],[0,70]])
            plan=inspected_serpentine_plan(p,mesh,region,8,20)
            report=plan['coverage_inspection']
            entries=[e for e in report['omissions'] if e['reason']=='selection_limit']
            self.assertTrue(entries)
            self.assertTrue(all(not e['in_denominator'] for e in entries))
            for sample in report['samples']:
                if sample['eligible']:self.assertTrue(polygon_contains(np.asarray([sample['pixel']]),region['screen_polygon'])[0])
                if sample['reason']=='route_spacing':self.assertGreater(sample['evidence']['route_distance_lower_m'],.004)
                if sample['reason']=='algorithm_limit':
                    self.assertLessEqual(sample['evidence']['nearest_route_node_distance_m'],.004+1e-12)
                    self.assertGreater(sample['evidence']['graph_distance_upper_m'],.004)

    def test_cancel_and_no_usable_footprint_do_not_publish(self):
        with tempfile.TemporaryDirectory() as tmp:
            mesh,store,region,p=self.surface(Path(tmp));old=inspected_serpentine_plan(p,mesh,region,8,20)
            before=copy.deepcopy(old)
            with self.assertRaises(ValueError):inspected_serpentine_plan(p,mesh,region,100,20,boundary_margin_mm=50)
            def cancel():raise RuntimeError('cancelled')
            with self.assertRaisesRegex(RuntimeError,'cancelled'):inspected_serpentine_plan(p,mesh,region,8,20,check=cancel)
            self.assertEqual(old,before);validate(old,mesh,region)

    def test_report_size_and_cancellation_fail_before_publish(self):
        from coverage_inspection import check_document_budget
        value={'samples':['bounded report']*100}
        before=copy.deepcopy(value)
        with self.assertRaisesRegex(ValueError,'可回读大小'):check_document_budget(value,limit=100)
        self.assertEqual(value,before)
        def cancel():raise RuntimeError('cancelled serialization')
        with self.assertRaisesRegex(RuntimeError,'cancelled serialization'):check_document_budget(value,cancel)

    def test_inspector_click_zoom_and_invalidation(self):
        import tkinter as tk
        import time
        with tempfile.TemporaryDirectory() as tmp:
            mesh,store,region,p=self.surface(Path(tmp));plan=inspected_serpentine_plan(p,mesh,region,8,20)
            root=tk.Tk();root.withdraw();current=[True];errors=[]
            root.report_callback_exception=lambda *args:errors.append(args)
            try:
                w=show_inspector(root,SimpleNamespace(rgb=np.zeros((85,85,3),np.uint8)),plan,lambda:current[0]);root.update()
                entry=plan['coverage_inspection']['omissions'][0]
                w.omission_tree.selection_set(entry['id']);w.omission_tree.event_generate('<<TreeviewSelect>>');root.update()
                self.assertEqual(w.selected_omission,entry);self.assertEqual(w.focus_bbox,crop_bounds(plan['coverage_inspection'],entry))
                self.assertEqual(w.plan_binding,plan['plan_binding'])
                current[0]=False;w.locate_omission(entry['id']);self.assertFalse(w.winfo_exists())
                root.update();self.assertFalse(errors)
            finally:root.destroy()

if __name__=='__main__':unittest.main()
