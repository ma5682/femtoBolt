"""Captured RGB reaches VTK without a selection tint or geometry mutation."""
from types import MethodType, SimpleNamespace
import unittest

import numpy as np

from femto.app.qt_vtk_viewport import VTKViewport


def numeric_viewport():
    # Exercise real VTK actors and mappers without opening a native Qt window.
    from vtkmodules.vtkRenderingCore import vtkRenderer, vtkCellPicker
    view = SimpleNamespace(renderer=vtkRenderer(), picker=vtkCellPicker(), _actors=[],
                           _check_thread=lambda: None, _render=lambda: None)
    view._actor = MethodType(VTKViewport._actor, view)
    view.set_snapshot = MethodType(VTKViewport.set_snapshot, view)
    return view


class CapturedColorTests(unittest.TestCase):
    def setUp(self):
        try:
            from vtkmodules.util.numpy_support import vtk_to_numpy
        except ImportError:
            self.skipTest('VTK is not installed')
        self.to_numpy = vtk_to_numpy
        self.mesh = SimpleNamespace(
            vertices=np.array([[0., 0., 1.], [1., 0., 1.], [0., 1., 1.]]),
            triangles=np.array([[0, 1, 2]]),
            vertex_colors=np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]))
        self.model = SimpleNamespace(mesh=self.mesh)
        self.region = {'triangle_indices': [0]}
        self.view = numeric_viewport()

    def test_captured_rgb_is_displayed_without_full_region_overlay(self):
        before = self.mesh.vertex_colors.copy()
        VTKViewport.set_scene(self.view, self.model, self.region, reset_camera=False)
        self.assertEqual(len(self.view._actors), 1)
        mapper = self.view._actors[0].GetMapper()
        rgb = self.to_numpy(mapper.GetInput().GetPointData().GetScalars())
        np.testing.assert_array_equal(rgb, [[255, 0, 0], [0, 255, 0], [0, 0, 255]])
        np.testing.assert_array_equal(self.to_numpy(mapper.GetInput().GetPoints().GetData()), self.mesh.vertices)
        np.testing.assert_array_equal(self.mesh.vertex_colors, before)
        self.assertIsNone(self.view._region_actor)
        self.assertEqual(self.view.picker.GetPickList().GetNumberOfItems(), 1)

    def test_coverage_toggle_preserves_rgb_and_restores_yellow_route(self):
        plan = {'segments': [[{'xyz_m': [0., 0., 1.]}, {'xyz_m': [1., 0., 1.]}]]}
        VTKViewport.set_scene(self.view, self.model, self.region, plan, reset_camera=False)
        mesh_actor = self.view._actors[0]
        before = self.to_numpy(mesh_actor.GetMapper().GetInput().GetPointData().GetScalars()).copy()
        display = SimpleNamespace(triangles=self.mesh.vertices[self.mesh.triangles], status=np.array([1]))
        VTKViewport.set_coverage(self.view, display)
        self.assertEqual(self.view._route_actor.GetVisibility(), 0)
        self.assertIsNotNone(self.view._coverage_actor)
        VTKViewport.set_coverage(self.view, None)
        self.assertEqual(self.view._route_actor.GetVisibility(), 1)
        self.assertEqual(self.view._route_actor.GetProperty().GetColor(), (1., .8, .25))
        self.assertIsNone(self.view._coverage_actor)
        np.testing.assert_array_equal(before, self.to_numpy(mesh_actor.GetMapper().GetInput().GetPointData().GetScalars()))

    def test_missing_rgb_keeps_neutral_fallback_without_inventing_colors(self):
        self.mesh.vertex_colors = []
        VTKViewport.set_scene(self.view, self.model, self.region, reset_camera=False)
        actor = self.view._actors[0]
        self.assertEqual(len(self.view._actors), 1)
        self.assertEqual(actor.GetMapper().GetScalarVisibility(), 0)
        self.assertIsNone(actor.GetMapper().GetInput().GetPointData().GetScalars())
        self.assertEqual(actor.GetProperty().GetColor(), (.7, .76, .8))
