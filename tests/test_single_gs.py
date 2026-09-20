from pathlib import Path
import tempfile
import unittest
from unittest import mock
import numpy as np
from PIL import Image
from io_utils import load_json as read_json, write_json
from scene_single_gs import prepare, seed_arrays, resolve_source
from scene_single_gs_viewer import FixedViewport, load_viewport, SingleViewer, native_render_size


class SingleGsTests(unittest.TestCase):
    def test_native_render_preserves_source_detail_with_bounded_pixel_count(self):
        self.assertEqual(native_render_size(1920,1080),(1920,1080))
        self.assertEqual(native_render_size(3840,2160),(1920,1080))
        self.assertEqual(native_render_size(640,480),(640,480))
        self.assertEqual(native_render_size(1080,1920),(608,1080))
        with self.assertRaises(ValueError): native_render_size(0,1080)

    def test_region_buttons_missing_save_reload_and_corrupt(self):
        with tempfile.TemporaryDirectory() as td:
            view=object.__new__(SingleViewer)
            view.output=Path(td); view.language='zh'; view.messagebox=mock.Mock(); view.status=mock.Mock()
            view.meta=dict(model_sha256='model',source_fingerprint='source',source_photo='photo')
            view.view=FixedViewport(2,.5,.5); view.dirty=False
            original=view.view
            for language in ('zh','en'):
                view.language=language; view.load()
                self.assertIs(view.view,original)
                self.assertFalse(view.dirty)
            self.assertEqual(view.messagebox.showinfo.call_count,2)
            view.messagebox.showerror.assert_not_called()
            view.save(); saved=(view.output/'viewport.json').read_bytes()
            view.reset(); view.load()
            self.assertEqual(view.view.zoom,2)
            with mock.patch.object(Path,'replace',side_effect=PermissionError): view.save()
            self.assertEqual((view.output/'viewport.json').read_bytes(),saved)
            self.assertFalse(list(view.output.glob('*.tmp')))
            before=view.view
            for content in ('{', '[]', '{"model_sha256":"model","source_fingerprint":"source","viewport":{"zoom":null,"cx":0.5,"cy":0.5}}'):
                (view.output/'viewport.json').write_text(content)
                view.messagebox.reset_mock(); view.load()
                view.messagebox.showerror.assert_called_once()
                self.assertIs(view.view,before)
            write_json(view.output/'viewport.json',dict(model_sha256='different',source_fingerprint='source'))
            view.load(); view.messagebox.showwarning.assert_called_once()
            self.assertIs(view.view,before)

    def test_full_hd_render_budget_is_explicit_and_keeps_multiview_default(self):
        from scene_gsplat_viewer import plan_render_budget
        options=dict(free_cuda_bytes=6*1024**3,total_cuda_bytes=8*1024**3)
        default=plan_render_budget(220000,1920,1080,**options)
        full=plan_render_budget(220000,1920,1080,max_pixels=1920*1080,**options)
        self.assertLess(default['effective_resolution'][0],1920)
        self.assertEqual(full['effective_resolution'],[1920,1080])
        self.assertFalse(full['temporary_sampling'])

    def test_completed_model_resolves_exact_selected_frame_and_rejects_multiview(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); model=root/'model'; model.mkdir(); dataset=root/'keyframe'; dataset.mkdir()
            write_json(model/'demo_summary.json',dict(frame_count=1,fixed_camera=True,source_dataset=str(dataset)))
            write_json(dataset/'manifest.json',dict(schema='femto-full-scene-rgbd-v1',status='derived_fixed_camera_keyframe',
                frame_count=1,parameters=dict(fixed_camera=True,depth_min_mm=250,depth_max_mm=2210)))
            (dataset/'frames.jsonl').write_text('{"rgb":"rgb/selected.jpg","depth":"depth/selected.png","source_frame":7}\n')
            source,manifest,lineage=resolve_source(model)
            self.assertEqual(source,dataset)
            self.assertEqual(manifest['files']['color'],'rgb/selected.jpg')
            self.assertEqual(lineage['selected_source_frame'],7)
            self.assertEqual(lineage['source_model'],str(model))
            write_json(model/'demo_summary.json',dict(frame_count=90,fixed_camera=False,source_dataset=str(dataset)))
            with self.assertRaises(ValueError): resolve_source(model)

    def test_zoom_preserves_anchor_and_camera_projection(self):
        K=np.array([[500.,0,400],[0,500,300],[0,0,1]])
        view=FixedViewport()
        np.testing.assert_allclose(view.intrinsic(K,800,600,800,600),K)
        anchor=view.source_point(.3,.4)
        view.zoom_at(2,.3,.4)
        np.testing.assert_allclose(view.source_point(.3,.4),anchor)
        pixel=np.array([anchor[0]*800,anchor[1]*600,1])
        ray=np.linalg.inv(K)@pixel
        projected=view.intrinsic(K,800,600,800,600)@ray
        np.testing.assert_allclose(projected[:2]/projected[2],[240,240],atol=1e-4)

    def test_box_zoom_respects_aspect_bounds_and_reset(self):
        view=FixedViewport(); view.select(.25,.25,.75,.75)
        self.assertEqual(view.zoom,2)
        self.assertEqual((view.cx,view.cy),(.5,.5))
        view.select(.95,.95,.7,.7)
        self.assertLessEqual(view.cx+.5/view.zoom,1)
        view.zoom_at(1e6); self.assertEqual(view.zoom,16)
        view.zoom_at(1e-6); self.assertEqual(vars(view),dict(zoom=1.,cx=.5,cy=.5))

    def test_seed_count_and_measured_geometry(self):
        depth=np.full((24,32),1000,np.uint16); depth[:3]=0
        rgb=np.full((24,32,3),127,np.uint8)
        K=np.array([[30.,0,16],[0,30,12],[0,0,1]])
        for cap in (1,50,100,1000):
            points,colors,scales,stride=seed_arrays(depth,rgb,K,cap)
            self.assertLessEqual(len(points),cap)
            np.testing.assert_allclose(points[:,2],1)
            self.assertTrue((scales>0).all())
            if len(points):
                uv=(points@K.T)[:,:2]
                np.testing.assert_allclose(uv/stride,np.round(uv/stride),atol=1e-6)

    def test_viewport_source_and_model_binding(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'viewport.json'; meta=dict(model_sha256='a',source_fingerprint='b')
            write_json(path,dict(**meta,viewport=dict(zoom=2,cx=.5,cy=.5)))
            self.assertEqual(load_viewport(path,meta).zoom,2)
            with self.assertRaises(ValueError): load_viewport(path,dict(meta,model_sha256='c'))
            with self.assertRaises(ValueError): load_viewport(path,dict(meta,source_fingerprint='c'))

    def test_prepare_preserves_source_and_no_external_pose(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); photo=root/'photo'; photo.mkdir(); (photo/'calibration').mkdir()
            Image.fromarray(np.full((24,32,3),127,np.uint8)).save(photo/'rgb.png')
            Image.fromarray(np.full((24,32),1000,np.uint16)).save(photo/'depth.png')
            write_json(photo/'manifest.json',dict(schema='femto-rgbd-photo-v1',alignment='depth_to_color',
                files=dict(color='rgb.png',depth='depth.png'),image=dict(depth_range_m=[.2,2.])))
            write_json(photo/'calibration'/'color_intrinsic.json',dict(width=32,height=24,fx=30,fy=30,cx=16,cy=12))
            write_json(photo/'calibration'/'distortion.json',dict(rgb=dict(model='BROWN_CONRADY_K6',
                **{k:0 for k in ('k1','k2','k3','k4','k5','k6','p1','p2')})))
            before={p:p.read_bytes() for p in photo.rglob('*') if p.is_file()}
            out=prepare(photo,root/'derived'); meta=read_json(out/'single_input.json')
            self.assertIsNone(meta['T_WC']); self.assertFalse(meta['external_pose_valid'])
            self.assertEqual(before,{p:p.read_bytes() for p in before})
            with self.assertRaises(FileExistsError): prepare(photo,out)
            with self.assertRaises(ValueError): prepare(photo,photo/'derived')


if __name__=='__main__': unittest.main()
