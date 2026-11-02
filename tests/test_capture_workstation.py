"""Display-only workflow, camera freshness and rendering regressions."""
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import unittest

import numpy as np

from femto.app.capture_workstation import workflow_progress, STATUS_COLORS
from femto.app.capture_viewport import current_camera_pose, grid_interval, render_scene, read_review_surface
from femto.app.capture_viewport import model_framing


class CaptureViewportTests(unittest.TestCase):
    def test_main_extent_framing_resists_sparse_distant_points_without_mutation(self):
        x,y = np.meshgrid(np.linspace(-.3,.3,40),np.linspace(-.2,.2,40))
        body = np.column_stack((x.ravel(),y.ravel(),np.full(x.size,.5)))
        points = np.vstack((body,[[-2.,-1.,2.]]))
        before = points.copy()
        center,span = model_framing(points)
        np.testing.assert_allclose(center[:2],[0.,0.],atol=.02)
        self.assertLess(span,.8)
        self.assertGreater(span,.6)
        np.testing.assert_array_equal(points,before)

    def test_saved_mesh_rgb_is_loaded_and_visible_over_point_cloud(self):
        vertices = np.array([[-.1,-.1,.5],[.1,-.1,.5],[0,.1,.5]])
        colors = np.tile([1.,0.,0.],(3,1))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'reconstruction'/'scene_tsdf_mesh_m.ply'
            path.parent.mkdir();path.touch()
            mesh = SimpleNamespace(vertices=vertices,triangles=np.array([[0,1,2]]),vertex_colors=colors)
            with mock.patch('femto.geometry.fusion.read_geometry_safe',return_value=mesh):
                surface = read_review_surface(tmp)
        image = np.asarray(render_scene(400,400,vertices,colors,center=[0,0,.5],span=.4,surface=surface))
        self.assertGreater(image[200,200,0],240)
        self.assertLess(image[200,200,1],5)
        surface['colors'] = None
        fallback = np.asarray(render_scene(400,400,vertices,colors,center=[0,0,.5],span=.4,surface=surface))
        self.assertGreater(fallback[200,200,2],fallback[200,200,0])

    def test_step_completion_follows_evidence_not_a_selected_menu(self):
        def progress(state,count=0,result=None,failure=None):
            return workflow_progress(state=state,count=count,active=True,result=result,failure=failure)
        self.assertEqual(progress('waiting'),['current','pending','pending','pending'])
        self.assertEqual(progress('tracking',1),['complete','current','pending','pending'])
        self.assertEqual(progress('processing',3),['complete','complete','current','pending'])
        self.assertEqual(progress('lost',3),['complete','blocked','pending','pending'])
        self.assertEqual(progress('origin_timeout',failure={}),['blocked','pending','pending','pending'])

    def test_stale_failed_or_finished_pose_never_draws_as_live_camera(self):
        record=dict(state='tracking',received_monotonic=10.,T_WC=np.eye(4).tolist())
        kwargs=dict(active=True,finishing=False,state='tracking',now=11.)
        np.testing.assert_array_equal(current_camera_pose(record,**kwargs),np.eye(4))
        for change in (dict(now=12.01),dict(active=False),dict(finishing=True),dict(state='uncertain')):
            self.assertIsNone(current_camera_pose(record,**(kwargs|change)))
        record['T_WC'][0][0]=99
        self.assertIsNone(current_camera_pose(record,**kwargs))

    def test_render_preserves_source_geometry_and_distinguishes_highlight(self):
        points=np.array([[-.1,-.1,.5],[.1,.1,.5]])
        colors=np.full((2,3),.5);before=points.copy();colours_before=colors.copy()
        image=render_scene(700,500,points,colors,center=[0,0,.5],span=.5,highlight=[False,True])
        np.testing.assert_array_equal(points,before);np.testing.assert_array_equal(colors,colours_before)
        self.assertEqual(image.size,(700,500))
        self.assertTrue(np.any(np.all(np.asarray(image)==[6,214,160],axis=2)))
        self.assertAlmostEqual(grid_interval(.4),.05)

    def test_locked_xyz_projection_ignores_pan(self):
        points=np.array([[-.1,-.04,.5],[.06,.08,.6],[.02,-.09,.4]])
        colors=np.full((3,3),.5)
        base=np.asarray(render_scene(700,500,points,colors,center=[0,0,.5],span=.5,zoom=1.7))
        shifted=np.asarray(render_scene(700,500,points,colors,center=[0,0,.5],span=.5,
                                        zoom=1.7,pan=(180,-95)))
        np.testing.assert_array_equal(shifted,base)

    def test_mesh_budget_samples_original_faces_without_filling_holes(self):
        vertices=np.array([[0,0,.5],[.1,0,.5],[0,.1,.5]])
        faces=np.tile([0,1,2],(240001,1))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'reconstruction'/'scene_tsdf_mesh_m.ply'
            path.parent.mkdir();path.touch()
            with mock.patch('femto.geometry.fusion.read_geometry_safe',return_value=SimpleNamespace(vertices=vertices,triangles=faces)):
                surface=read_review_surface(tmp)
        self.assertTrue(surface['sampled']);self.assertLessEqual(len(surface['triangles']),12000)
        np.testing.assert_array_equal(surface['triangles'][0],vertices)
        surface['triangles'][0]=100
        self.assertEqual(vertices[0,0],0)




if __name__=='__main__':unittest.main()
