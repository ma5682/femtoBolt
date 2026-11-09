"""Painted targets constrain reconstruction inputs and inspection geometry."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase, mock

import numpy as np

from femto.geometry.local_region import (load_reference, save_request, read_request,
    RegionDepthFilter, SCHEMA, LEGACY_SCHEMA)
from femto.io_utils import load_json_object, write_json_strict
from femto.viewer.vtk_scene import scene_snapshot
from test_region_first_reconstruction import dataset_fixture


class SelectedOnlyReconstructionTests(TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.capture,self.dataset=dataset_fixture(self.root,3)
        self.reference=load_reference(self.dataset)
        self.mask=np.zeros(self.reference.depth.shape,np.uint8)
        self.mask[35:75,50:95]=1;self.mask[45:57,60:72]=0
        self.mask[80:100,110:135]=1

    def test_zero_margin_is_default_and_does_not_build_distance_band(self):
        path=save_request(self.reference,self.mask,self.root/'request')
        _,mask,record=read_request(path)
        self.assertEqual(record['schema'],SCHEMA)
        self.assertEqual(record['support_margin_mm'],0)
        self.assertEqual(record['geometry_scope'],'selected_surface_only')
        with mock.patch('cv2.distanceTransform',side_effect=AssertionError('support band')):
            region=RegionDepthFilter(path,self.dataset,.25,2.21,self.root)
            selected=region.filter(self.reference.depth,np.eye(4),self.reference.k,0)
        np.testing.assert_array_equal(selected,np.where(mask,self.reference.depth,0))
        self.assertEqual(region.metrics[0]['integrated_pixels'],int(mask.sum()))

    def test_moving_frame_uses_pose_mapping_and_preserves_holes(self):
        path=save_request(self.reference,self.mask,self.root/'request')
        region=RegionDepthFilter(path,self.dataset,.25,2.21,self.root)
        pose=np.eye(4);pose[0,3]=.012
        selected=region.filter(self.reference.depth,pose,self.reference.k,1)
        y,x=np.nonzero(selected);z=selected[y,x]/1000.;k=self.reference.k
        points=np.c_[(x-k[0,2])*z/k[0,0],(y-k[1,2])*z/k[1,1],z]+pose[:3,3]
        uv=np.rint(points[:,:2]/points[:,2,None]*[k[0,0],k[1,1]]+k[:2,2]).astype(int)
        self.assertTrue(self.mask[uv[:,1],uv[:,0]].all())
        self.assertFalse(np.array_equal(selected>0,self.mask>0))
        region.reference.depth[40:65,50:85]=0
        other=region.filter(self.reference.depth,pose,k,2)
        self.assertLess(np.count_nonzero(other),len(x))
        np.testing.assert_array_equal(region.mask,self.mask)

    def test_legacy_margin_replays_and_versions_cannot_be_mixed(self):
        old=save_request(self.reference,self.mask,self.root/'old',40)
        self.assertEqual(read_request(old)[2]['schema'],LEGACY_SCHEMA)
        region=RegionDepthFilter(old,self.dataset,.25,2.21,self.root)
        self.assertGreater(np.count_nonzero(region.filter(self.reference.depth,np.eye(4),self.reference.k,0)),self.mask.sum())
        new=save_request(self.reference,self.mask,self.root/'new')
        record=load_json_object(new)
        for altered in (dict(record,schema=LEGACY_SCHEMA),dict(record,support_margin_mm=40),
                        dict(record,geometry_scope='selected_surface_and_observed_surroundings')):
            write_json_strict(new,altered)
            with self.assertRaises(ValueError):read_request(new)

    def test_tsdf_integrates_only_pose_mapped_selection(self):
        from femto.pipeline.scene_reconstruction import reconstruct_scene
        from femto.annotation.fixed_rgb_geometry import FixedRGBModel
        from femto.geometry.local_region import model_selection
        path=save_request(self.reference,self.mask,self.root/'request')
        report=reconstruct_scene(self.dataset,self.root/'model',depth_min_m=.25,depth_max_m=2.21,region_request=path)
        self.assertEqual(report['frame_count'],3)
        self.assertEqual(report['parameters']['mesh_method'],'tsdf')
        scope=report['reconstruction_scope']
        self.assertEqual(scope['support_margin_mm'],0)
        for frame in scope['frames']:
            self.assertLessEqual(frame['surface_input_pixels'],int(self.mask.sum()))
        np.testing.assert_array_equal(model_selection(FixedRGBModel(self.root/'model')),self.mask)


class SelectedOnlyDisplayTests(TestCase):
    def scene(self):
        from femto.annotation.selection_intent import build_selection_intent
        vertices=np.array([[0,0,1],[9,0,1],[0,9,1],[9,9,1],[40,40,1],[41,40,1],[40,41,1]],float)
        faces=np.array([[4,5,6],[0,1,2],[1,3,2]],int)
        model=SimpleNamespace(mesh=SimpleNamespace(vertices=vertices,triangles=faces,vertex_colors=np.ones_like(vertices)))
        mask=np.zeros((10,10),np.uint8);mask[1:6,1:6]=1;mask[3:5,3:5]=0;mask[7:9,7:9]=1
        region=dict(triangle_indices=[1,2],selection_intent=build_selection_intent(mask,method='work_only'),
                    view_reference=dict(undistorted_intrinsic=np.eye(3).tolist()))
        return model,mask,region

    def test_clipped_model_keeps_hole_disconnected_regions_and_original_face_ids(self):
        model,mask,region=self.scene();before=model.mesh.vertices.copy()
        points=[dict(xyz_m=[2.,2.,1.]),dict(xyz_m=[2.,3.,1.])]
        plan=dict(segments=[points])
        scene=scene_snapshot(model,region,plan,selection_only=True)
        self.assertTrue(len(scene.triangles))
        self.assertEqual(set(scene.source_triangle_ids),{1,2})
        self.assertLess(scene.vertices[:,:2].max(),9)
        for bary in ([1/3]*3,[.8,.1,.1],[.1,.8,.1],[.1,.1,.8]):
            p=np.einsum('j,ijk->ik',bary,scene.vertices[scene.triangles])
            uv=np.floor(p[:,:2]/p[:,2,None]+.5).astype(int)
            self.assertTrue(mask[uv[:,1],uv[:,0]].all())
        np.testing.assert_array_equal(scene.route_runs[0],[[2,2,1],[2,3,1]])
        np.testing.assert_array_equal(model.mesh.vertices,before)
        self.assertFalse(scene.source_triangle_ids.flags.writeable)

    def test_no_confirmed_selection_never_shows_all_geometry_or_old_route(self):
        model,_,_=self.scene()
        scene=scene_snapshot(model,plan=dict(segments=[[dict(xyz_m=[0,0,1]),dict(xyz_m=[1,1,1])]]),selection_only=True)
        self.assertEqual(len(scene.vertices),0)
        self.assertEqual(len(scene.triangles),0)
        self.assertEqual(scene.route_runs,())

    def test_legacy_triangle_region_compacts_vertices_and_preserves_pick_ids(self):
        model,_,_=self.scene()
        scene=scene_snapshot(model,dict(triangle_indices=[2]),selection_only=True)
        self.assertEqual(scene.source_triangle_ids.tolist(),[2])
        np.testing.assert_array_equal(scene.vertices[scene.triangles],model.mesh.vertices[model.mesh.triangles[[2]]])
        self.assertEqual(len(scene.vertices),3)

    def test_vtk_actor_receives_original_ids_after_cropping(self):
        from femto.app.qt_vtk_viewport import VTKViewport
        model,_,_=self.scene();actor=mock.Mock()
        view=SimpleNamespace(_check_thread=lambda:None,renderer=mock.Mock(),picker=mock.Mock(),
            _actors=[],_actor=mock.Mock(return_value=actor),_render=lambda:None)
        VTKViewport.set_scene(view,model,dict(triangle_indices=[2]),reset_camera=False)
        self.assertEqual(view._actor.call_args_list[0].kwargs['ids'].tolist(),[2])
