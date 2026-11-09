"""View switching is lazy, recoverable and independent of annotation state."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import unittest
from unittest.mock import patch
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QApplication, QWidget
from femto.app.qt_planning_canvas import PlanningCanvas
from femto.app.qt_scene_panel import ScenePanel


class FakeViewport(QWidget):
    picked=Signal(dict)
    def __init__(self,parent=None):
        super().__init__(parent);self.calls=[];self.closed=False
    def set_scene(self,*args,**kwargs):self.calls.append((args,kwargs))
    def reset_camera(self):pass
    def shutdown(self):self.closed=True


class ScenePanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])

    def test_lazy_switch_keeps_camera_for_layer_change_and_releases_resources(self):
        canvas=PlanningCanvas();panel=ScenePanel(canvas)
        self.addCleanup(panel.close)
        model=object();region=object();plan=object()
        with patch('femto.app.qt_scene_panel.create_viewport',side_effect=FakeViewport) as factory:
            panel.set_scene(model,region,plan);factory.assert_not_called()
            panel.mode.setCurrentIndex(1);viewport=panel.viewport
            self.assertIs(panel.stack.currentWidget(),viewport)
            self.assertTrue(viewport.calls[-1][1]['reset_camera'])
            panel.set_scene(model,region,plan);self.assertEqual(len(viewport.calls),1)
            panel.set_scene(model,region,None)
            self.assertFalse(viewport.calls[-1][1]['reset_camera'])
            viewport.picked.emit(dict(triangle_index=7,point_world_m=[.1,.2,.3]))
            self.assertIn('0.100000',panel.notice.text())
            panel.show_reference();self.assertIs(panel.stack.currentWidget(),canvas)
            panel.shutdown();self.assertTrue(viewport.closed)
            self.assertIsNone(panel.viewport)

    def test_failed_three_dimensional_view_can_retry_without_replacing_reference(self):
        canvas=PlanningCanvas();panel=ScenePanel(canvas);self.addCleanup(panel.close)
        panel.set_scene(object())
        with patch('femto.app.qt_scene_panel.create_viewport',side_effect=RuntimeError('missing driver')):
            panel.mode.setCurrentIndex(1)
        self.assertIs(panel.stack.currentWidget(),canvas)
        self.assertIn('missing driver',panel.notice.text())
        with patch('femto.app.qt_scene_panel.create_viewport',side_effect=FakeViewport):panel.retry.click()
        self.assertIs(panel.stack.currentWidget(),panel.viewport)
        with patch.object(panel.viewport,'reset_camera',side_effect=RuntimeError('render failed')):
            panel.reset.click()
        self.assertIs(panel.stack.currentWidget(),canvas)
        self.assertIn('render failed',panel.notice.text())
        panel.shutdown()

    def test_review_switches_sources_without_showing_hidden_reference_over_three_dimensional_view(self):
        from test_qt_planning import QtPlanningTests
        from femto.app.qt_review import ReviewPage
        fixture=QtPlanningTests();fixture.setUpClass();fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        page=ReviewPage();self.addCleanup(page.close);self.addCleanup(page.shutdown)
        page.install_model(fixture.model,fixture.model.reconstruction)
        with patch('femto.app.qt_scene_panel.create_viewport',side_effect=FakeViewport):
            page.scene_panel.mode.setCurrentIndex(1)
            self.assertEqual(page.layers.itemText(1),'已确认工作区域')
            page.install_model(fixture.model,fixture.model.reconstruction)
            self.assertIs(page.scene_panel.stack.currentWidget(),page.scene_panel.viewport)
            self.assertTrue(page.canvas.isHidden())
            page.scene_panel.show_reference()
            self.assertEqual(page.layers.itemText(1),'工作区域与禁区')
        with patch('femto.app.qt_scene_panel.create_viewport',side_effect=FakeViewport):
            fixture.page.scene_panel.mode.setCurrentIndex(1)
            self.assertFalse(fixture.page.tools.isEnabled())
            self.assertTrue(fixture.page.tools.isHidden())
            self.assertTrue(fixture.page.support.isHidden())
            self.assertTrue(fixture.page.undo_button.isHidden())
            self.assertFalse(fixture.page.canvas.editable)
            self.assertTrue(fixture.page.inspector_scroll.isHidden())
            # The current workbench keeps its next-step controls in the 3D view.
            self.assertFalse(fixture.page.action_dock.isHidden())
            self.assertTrue(fixture.page.scene_panel.notice.isHidden())
            fixture.page._error='检查未完成'
            fixture.page.refresh()
            self.assertFalse(fixture.page.action_dock.isHidden())
            self.assertFalse(fixture.page.primary.isHidden())
            fixture.page._error=''
            fixture.page.scene_panel.show_reference()
            self.assertTrue(fixture.page.tools.isEnabled())
            self.assertFalse(fixture.page.tools.isHidden())
            self.assertTrue(fixture.page.canvas.editable)
            self.assertFalse(fixture.page.inspector_scroll.isHidden())
            self.assertFalse(fixture.page.action_dock.isHidden())
            self.assertIn('请先涂抹',fixture.page.support_hint.text())

    def test_switch_verifies_selection_without_advancing_and_removes_stale_overlay(self):
        from test_qt_planning import QtPlanningTests
        fixture=QtPlanningTests();fixture.setUpClass();fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        page=fixture.page
        page.canvas.work[:]=fixture.model.projection.valid
        page._selection_changed()
        with patch('femto.app.qt_scene_panel.create_viewport',side_effect=FakeViewport):
            page.scene_panel.mode.setCurrentIndex(1)
            fixture.wait_task()
            self.assertEqual(page._stage,0)
            self.assertFalse(page.selection_dirty)
            self.assertIsNotNone(page.region)
            self.assertIs(page.scene_panel.viewport.calls[-1][0][1],page.region)
            page.scene_panel.show_reference()
            page.canvas.work[:]=0;page._selection_changed()
            page.scene_panel.mode.setCurrentIndex(1)
            self.assertIsNone(page.scene_panel.viewport.calls[-1][0][1])

    def test_fixed_view_mouse_routes_to_pan_not_rotation(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from femto.app.qt_vtk_viewport import VTKViewport
        view=SimpleNamespace(widget=Mock(),style=Mock(),picker=Mock(),renderer=Mock())
        view.widget.GetEventPosition.return_value=(10,10)
        view.picker.Pick.return_value=False
        VTKViewport._pressed(view,None,None)
        VTKViewport._released(view,None,None)
        view.style.OnMiddleButtonDown.assert_called_once()
        view.style.OnMiddleButtonUp.assert_called_once()
        view.style.OnLeftButtonDown.assert_not_called()

    def test_new_region_refits_view_and_capture_editor_cannot_add_surroundings(self):
        from femto.app.qt_region_capture import CaptureRegionEditor
        editor=CaptureRegionEditor();self.addCleanup(editor.close)
        self.assertEqual(editor.margin.value(),0)
        editor.margin.setValue(40)
        self.assertEqual(editor.margin.value(),0)
        self.assertTrue(editor.margin.isHidden())
        panel=ScenePanel(PlanningCanvas());self.addCleanup(panel.close)
        with patch('femto.app.qt_scene_panel.create_viewport',side_effect=FakeViewport):
            model=object();panel.set_scene(model,object());panel.mode.setCurrentIndex(1)
            panel.set_scene(model,object())
            self.assertTrue(panel.viewport.calls[-1][1]['reset_camera'])
            self.assertIn('仅显示',panel.notice.text())


if __name__=='__main__':unittest.main()
