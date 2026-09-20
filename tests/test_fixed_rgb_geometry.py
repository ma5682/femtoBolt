import gc
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock
from types import SimpleNamespace
import tkinter as tk
import numpy as np
import open3d as o3d
from fixed_rgb_geometry import FixedProjection, FixedRGBModel
from scene_annotations import AnnotationStore
from local_capture import LocalCapture


def plane(z=.5):
    mesh=o3d.geometry.TriangleMesh()
    mesh.vertices=o3d.utility.Vector3dVector([[-.5,-.5,z],[.5,-.5,z],[.5,.5,z],[-.5,.5,z]])
    mesh.triangles=o3d.utility.Vector3iVector([[0,1,2],[0,2,3]])
    return mesh


class FixedProjectionTests(unittest.TestCase):
    def test_cloud_render_uses_nearest_vertex_and_radial_distance(self):
        from fixed_rgb_geometry import render_vertex_cloud
        view=dict(uv=np.array([[1.,1.],[1.,1.],[2.,1.]]),z=np.array([.7,.5,.6]),
                  distance_mm=np.array([700.,500.,650.]),limits=(500.,700.))
        image,mask=render_vertex_cloud(view,4,3,1,[0,0],True)
        self.assertEqual(int(mask.sum()),2)
        np.testing.assert_array_equal(image[1,1],[40,180,255])
        np.testing.assert_array_equal(image[0,0],[8,18,28])

    def test_selected_fill_keeps_holes_and_tracks_zoom(self):
        from fixed_rgb_geometry import selected_surface_mask,region_vertex_summary
        p=SimpleNamespace(ids=np.array([[0,-1],[1,2]]))
        np.testing.assert_array_equal(selected_surface_mask(p,[0,1],2,2,1,[0,0]),[[True,False],[True,False]])
        mask=selected_surface_mask(p,[0],5,5,2,[1,1])
        self.assertFalse(mask[0,0]);self.assertTrue(mask[1,1]);self.assertFalse(mask[3,3])
        summary=region_vertex_summary(plane(),[0,1])
        self.assertEqual(summary['vertex_count'],4)
        self.assertAlmostEqual(summary['minimum_distance_mm'],np.sqrt(.75)*1000)

    def setUp(self):
        self.mesh=plane();self.k=np.array([[100.,0,50],[0,100.,50],[0,0,1]])
        self.depth=np.full((100,100),500,np.uint16)

    def test_pixel_has_metric_w0_coordinates_and_origin_distance(self):
        projection=FixedProjection(self.mesh,self.k,self.depth)
        hit=projection.pixel(70,40)
        np.testing.assert_allclose(hit['point_world_m'],[.1,-.05,.5],atol=1e-6)
        store=AnnotationStore(self.mesh,'test_mesh.ply');point=store.add_point(hit)
        self.assertAlmostEqual(point['distance_to_origin_m'],np.linalg.norm([.1,-.05,.5]),places=6)
        np.testing.assert_allclose(projection.project([point['point_world_m']]),[[70,40]],atol=1e-5)

    def test_foreground_depth_conflict_and_outside_image_reject(self):
        self.depth[:]=300
        p=FixedProjection(self.mesh,self.k,self.depth)
        self.assertIsNone(p.pixel(50,50))
        self.assertIsNone(p.pixel(-1,50));self.assertIsNone(p.pixel(float('nan'),0))
        self.assertEqual(p.region([[0,0],[99,0],[99,99],[0,99]]),[])

    def test_missing_depth_does_not_create_geometry_but_other_view_mesh_can_support(self):
        self.depth[:]=0
        mesh=plane();mesh.triangles=o3d.utility.Vector3iVector([[0,1,2]])
        p=FixedProjection(mesh,self.k,self.depth)
        self.assertIsNone(p.pixel(20,80))
        self.assertIsNotNone(p.pixel(80,20))
        self.assertTrue(p.new_support[20,80])
        self.assertEqual(p.region([[0,0],[99,0],[99,99],[0,99]]),[0])

    def test_nearest_surface_occludes_back_triangles(self):
        mesh=plane()+plane(.7)
        p=FixedProjection(mesh,self.k,self.depth)
        ids=p.region([[0,0],[99,0],[99,99],[0,99]])
        self.assertEqual(ids,[0,1])
        np.testing.assert_array_equal(p.visible([[0,0,.5],[0,0,.7],[0,0,-.5]]),[True,False,False])

    def test_roundtrip_keeps_reference_evidence_and_geometry(self):
        with tempfile.TemporaryDirectory() as td:
            p=FixedProjection(self.mesh,self.k,self.depth)
            model=FixedRGBModel.__new__(FixedRGBModel)
            model.store=AnnotationStore(self.mesh,Path(td)/'mesh.ply')
            model.view_reference={'rgb_sha256':'test'}
            model.store.add_point(dict(p.pixel(70,40),view_reference=model.view_reference),name='A')
            path=model.store.save(Path(td)/'annotations')
            raw=path.read_bytes();model.load(path)
            self.assertEqual(path.read_bytes(),raw)
            model.view_reference={'rgb_sha256':'changed'}
            with self.assertRaisesRegex(ValueError,'RGB'):model.load(path)


class EnvelopeTests(unittest.TestCase):
    def test_total_rotation_and_radius_boundaries(self):
        from local_capture import local_workspace
        def pose(degrees):
            t=np.eye(4);a=np.deg2rad(degrees)
            t[:3,:3]=[[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]]
            t[0,3]=.03
            return t
        self.assertTrue(local_workspace(pose(10))['within_limit'])
        self.assertFalse(local_workspace(pose(10.01))['within_limit'])
        t=pose(0);t[1,3]=.001
        self.assertFalse(local_workspace(t)['within_limit'])

    def test_outside_request_rejected_without_reference_change_and_return_can_capture(self):
        c=LocalCapture.__new__(LocalCapture)
        c.lock=threading.Lock();c.closing=threading.Event();c.paused=False;c.requested=10.
        c.accepted=[dict(T_WC=np.eye(4).tolist())];c.emit=mock.Mock()
        t=np.eye(4);t[0,3]=.031
        r=dict(state='tracking',T_WC=t.tolist(),step_m=0,step_deg=0)
        with mock.patch('local_capture.time.monotonic',return_value=10.5):
            self.assertFalse(c._select(r,9.9));self.assertEqual(c.requested,10.)
            self.assertFalse(c._select(r,10.1));self.assertIsNone(c.requested)
            self.assertFalse(r['local_workspace']['within_limit'])
            c.requested=10.;t[0,3]=.03;r['T_WC']=t.tolist()
            self.assertTrue(c._select(r,10.2))
            self.assertTrue(r['local_workspace']['within_limit'])
        np.testing.assert_array_equal(c.accepted[0]['T_WC'],np.eye(4))


class FixedViewerTests(unittest.TestCase):
    def test_zoom_pan_point_region_save_load_and_miss(self):
        gc.collect()
        from fixed_rgb_viewer import FixedRGBViewer
        with tempfile.TemporaryDirectory() as td:
            model=FixedRGBModel.__new__(FixedRGBModel);model.mesh=plane()
            model.reconstruction=Path(td)/'reconstruction';model.reconstruction.mkdir()
            model.rgb=np.full((100,100,3),150,np.uint8)
            model.projection=FixedProjection(model.mesh,np.array([[100.,0,50],[0,100,50],[0,0,1]]),np.full((100,100),500,np.uint16))
            model.store=AnnotationStore(model.mesh,model.reconstruction/'mesh.ply')
            model.view_reference={'reference_frame_id':0};model.max_radius_mm=0
            root=tk.Tk()
            try:
                with mock.patch.object(FixedRGBViewer,'work'):
                    v=FixedRGBViewer(root,model.reconstruction)
                v.loaded(model);v.mode.set('point');v.active_mode='point';root.update();v.draw()
                pixel=np.array([70.,40.]);xy=pixel*v.scale+v.offset
                v.click(SimpleNamespace(x=xy[0],y=xy[1]));self.assertEqual(len(model.store.points),1)
                before=model.store.points[0]['point_world_m'][:]
                v.wheel(SimpleNamespace(x=xy[0],y=xy[1],delta=120))
                np.testing.assert_allclose(v.image_pixel(*xy),pixel,atol=.2)
                v.pan_start(SimpleNamespace(x=0,y=0));v.pan_move(SimpleNamespace(x=20,y=10))
                xy=pixel*v.scale+v.offset;v.click(SimpleNamespace(x=xy[0],y=xy[1]))
                np.testing.assert_allclose(model.store.points[-1]['point_world_m'],before)
                # A background click clears the reading without adding a point.
                v.click(SimpleNamespace(x=-10000,y=-10000))
                self.assertIsNone(v.focus);self.assertEqual(len(model.store.points),2)
                path=model.store.save(Path(td)/'saved');model.load(path)
                v.refresh_items();self.assertEqual(v.items.size(),2)
                root.geometry('1000x720');root.update();root.update()
                # Planning controls are intentionally hidden in the capture viewer.
                visible=[widget for widget in v.widgets if widget.winfo_viewable()]
                self.assertTrue(visible)
                for widget in visible:self.assertGreaterEqual(widget.winfo_height(),widget.winfo_reqheight())
                v.side_canvas.yview_moveto(1);root.update()
                last=visible[-1]
                self.assertLessEqual(last.winfo_rooty()+last.winfo_height(),root.winfo_rooty()+720)
                with mock.patch.object(v,'work',side_effect=lambda fn,done:done(fn())):
                    v.polygon=[[0,0],[99,0],[99,99],[0,99]];v.finish_polygon()
                    self.assertEqual(len(model.store.regions),1)
                    self.assertNotIn('X:',v.readout.get())
                    v.name.set('Region A');v.edit()
                    self.assertEqual(model.store.regions[0]['name'],'Region A')
                    v.delete();self.assertEqual(model.store.regions,[])
                    v.undo();self.assertEqual(len(model.store.regions),1)
                    v.save()
                    saved=list((model.reconstruction/'annotations').glob('*/annotations.json'))
                    self.assertEqual(len(saved),1)
                    with mock.patch('fixed_rgb_viewer.filedialog.askopenfilename',return_value=str(saved[0])):v.load()
                    self.assertEqual(v.items.size(),3)
            finally:
                v.close()
            del v,root
            gc.collect()


if __name__=='__main__':unittest.main()
