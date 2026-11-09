"""A region-first capture hands off to its filtered model and route controls."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtGui import QImage, QColor
from PySide6.QtWidgets import QLineEdit, QMessageBox, QPushButton

from femto.app.qt_main import WorkstationWindow
import test_qt_planning as planning_fixture
from test_qt_scene_panel import FakeViewport


class RouteOnlyHandoffTests(unittest.TestCase):
    def setUp(self):
        self.fixture = planning_fixture.QtPlanningTests()
        self.fixture.setUpClass(); self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.mark(split=True)
        self.model = self.fixture.page.model
        self.model.store.local_context = {'reconstruction_scope': {'schema': 'test-region-first'}}
        self.page = self.fixture.page
        viewport = patch('femto.app.qt_scene_panel.create_viewport', side_effect=FakeViewport)
        viewport.start(); self.addCleanup(viewport.stop)

    def install(self):
        self.page.install_model(self.model, self.model.reconstruction, self.fixture.journal)
        self.fixture.app.processEvents()

    def test_filtered_model_opens_with_route_controls_and_readonly_reference(self):
        before = self.page.canvas.work & ~self.page.canvas.forbidden
        loaded = []
        self.page.loaded.connect(lambda _: loaded.append((self.page._stage, self.page.scene_panel.is_3d)))
        self.install()
        self.assertEqual(loaded, [(1, True)])
        self.assertEqual(self.page.title.text(), '三维蛇形路线规划')
        self.assertEqual(self.page.primary.text(), '生成路线')
        self.assertNotIn('详情 / 帮助',[button.text() for button in self.page.findChildren(QPushButton)])
        self.assertTrue(self.page.inspector_scroll.isVisible())
        self.assertTrue(self.page.head.isVisible())
        self.assertIs(self.page.scene_panel.viewport.calls[-1][0][1], self.page.region)
        for reference in (False, True):
            self.page.scene_panel.mode.setCurrentIndex(0 if reference else 1)
            self.assertFalse(self.page.canvas.editable)
            for control in (self.page.tools, self.page.undo_button, self.page.redo_button,
                            self.page.clear_button, self.page.resolution_toggle):
                self.assertTrue(control.isHidden())
            self.page.canvas.clear_work(); self.page.canvas.undo(); self.page.canvas.redo()
            np.testing.assert_array_equal(self.page.canvas.work, before)
        from femto.viewer.vtk_scene import scene_snapshot
        scene = scene_snapshot(self.page.model, self.page.region, selection_only=True)
        self.assertTrue(len(scene.triangles))
        self.assertTrue(set(scene.source_triangle_ids) <= set(self.page.region['triangle_indices']))

    def test_back_and_workflow_return_cannot_reopen_selection_editor(self):
        self.install()
        self.page.set_stage(0)
        self.assertEqual(self.page._stage, 1)
        returned = []
        self.page.back_requested.connect(lambda: returned.append(True))
        self.page.go_back()
        self.assertEqual(returned, [True])
        window = WorkstationWindow(connect_camera=False)
        self.addCleanup(window.close)
        window.planning.install_model(self.model, self.model.reconstruction)
        window.pages.setCurrentWidget(window.planning)
        available = window.workflow_navigation._local_destinations(window.planning)
        self.assertIn(4, available); self.assertNotIn(3, available)
        self.assertTrue(window.workflow_navigation._show_local(window.planning, 3))
        self.assertEqual(window.planning._stage, 1)

    def test_pending_old_draft_can_recheck_without_editing_or_bypassing_geometry(self):
        self.install()
        self.page.selection_dirty = True
        self.page.set_stage(0)
        self.assertEqual(self.page._stage, 1)
        self.assertEqual(self.page.primary.text(), '检查模型并继续')
        self.assertFalse(self.page.canvas.editable)
        self.page.primary_action(); self.fixture.wait_task()
        self.assertFalse(self.page.selection_dirty)
        self.assertEqual(self.page.primary.text(), '生成路线')

    def test_render_failure_keeps_route_controls_and_readonly_fallback(self):
        with patch('femto.app.qt_scene_panel.create_viewport', side_effect=RuntimeError('test renderer unavailable')):
            self.install()
        self.assertIs(self.page.scene_panel.stack.currentWidget(), self.page.canvas)
        self.assertIn('test renderer unavailable', self.page.scene_panel.notice.text())
        self.assertTrue(self.page.scene_panel.retry.isVisible())
        self.assertTrue(self.page.head.isVisible())
        self.assertFalse(self.page.canvas.editable)
        self.page.scene_panel.retry.click()
        self.assertIs(self.page.scene_panel.stack.currentWidget(), self.page.scene_panel.viewport)

    def wait(self, runner):
        loop = QEventLoop(); timer = QTimer()
        timer.setSingleShot(True); timer.timeout.connect(loop.quit)
        runner.finished.connect(loop.quit)
        try:
            timer.start(15000)
            if runner.busy: loop.exec()
        finally:
            runner.finished.disconnect(loop.quit); timer.stop()
        self.assertFalse(runner.busy)

    def exercise_handoff(self, outcome, *, previous_model=False, retry=False, unrelated_preview=False):
        window = WorkstationWindow(connect_camera=False)
        window.show()
        released = threading.Event()
        runner = window.planning.runner
        path = Path(self.model.reconstruction).resolve()
        if previous_model:
            window.planning.install_model(self.model, path / 'previous')
            previous_source=window._planning_source
        window.capture.session.result=path.parent/'another-capture' if unrelated_preview else path.parent
        window.capture.scan_roi_preview=QImage(80,60,QImage.Format_RGB888)
        window.capture.scan_roi_preview.fill(QColor('#208580'))
        image_key=window.capture.scan_roi_preview.cacheKey()
        calls=[]

        def load(*args, **kwargs):
            calls.append(args[0])
            args[2]('正在读取当前参考来源')
            if not released.wait(10): raise RuntimeError('test worker timed out')
            if outcome == 'failure' and len(calls)==1: raise ValueError('test source failed')
            return self.model

        try:
            with patch('femto.app.qt_planning.load_source', side_effect=load), \
                 patch('femto.planning.planning_drafts.read',return_value=None), \
                 patch.object(QMessageBox, 'warning') as warning:
                self.assertTrue(window.open_planning(path))
                page=window.planning
                self.assertIs(window.pages.currentWidget(),page)
                self.assertEqual(page.loading_source,path)
                self.assertIs(page.body_stack.currentWidget(),page.preparation)
                self.assertFalse(page.primary.isEnabled())
                self.assertFalse(page.canvas.editable)
                self.assertFalse(page.planning_body.isEnabled())
                self.assertEqual(page.workflow.current_step,4)
                self.assertEqual(page.title.text(),'模型准备')
                self.assertEqual(page.preparation.reference.image.isNull(),unrelated_preview)
                self.assertEqual(page.preparation.findChildren(QLineEdit),[])
                self.assertEqual(page.preparation.title.text(),'正在准备三维模型…')
                self.assertTrue(page.status_label.isHidden())
                self.assertNotIn('详情 / 帮助',[button.text() for button in page.findChildren(QPushButton)])
                self.assertFalse(window.capture_button.isEnabled())
                if outcome == 'cancel':
                    page.preparation.cancel.click()
                    self.assertTrue(page.preparation.isVisible())
                    self.assertFalse(page.preparation.cancel.isEnabled())
                released.set(); self.wait(runner)
                if outcome == 'success':
                    self.assertIs(window.pages.currentWidget(),page)
                    self.assertIs(page.body_stack.currentWidget(),page.planning_body)
                    self.assertIsNone(page.loading_source)
                    self.assertEqual(page._stage, 1)
                    self.assertTrue(page.scene_panel.is_3d)
                    self.assertTrue(page.head.isEnabled())
                    self.assertTrue(page.primary.isEnabled())
                    self.assertEqual(window._planning_source, path)
                elif outcome == 'cancel':
                    self.assertIs(window.pages.currentWidget(), window.capture)
                    self.assertIsNone(page.loading_source)
                    self.assertEqual(window._planning_source,previous_source if previous_model else None)
                    self.assertIs(page.model,self.model if previous_model else None)
                    self.assertNotIn(window.planning, window._pending_sources)
                else:
                    self.assertIs(window.pages.currentWidget(),page)
                    self.assertEqual(page.preparation.state,'failed')
                    self.assertTrue(page.preparation.retry.isVisible())
                    self.assertFalse(page.primary.isEnabled())
                    self.assertFalse(window.capture_button.isEnabled())
                    self.assertEqual(window.workflow_navigation.destinations(page)[0],set())
                    if retry:
                        page.preparation.retry.click();self.wait(runner)
                        self.assertIsNone(page.loading_source)
                        self.assertEqual(calls,[path,path])
                        self.assertEqual(window._planning_source,path)
                        self.assertTrue(page.head.isEnabled())
                    else:
                        page.preparation.cancel.click()
                        self.assertIs(window.pages.currentWidget(),window.capture)
                        self.assertEqual(window._planning_source,previous_source if previous_model else None)
                self.assertEqual(warning.call_count,0)
                self.assertTrue(window.capture_button.isEnabled())
                self.assertEqual(window.capture.scan_roi_preview.cacheKey(),image_key)
                self.assertFalse(page.preparation.spinner.timer.isActive())
        finally:
            released.set()
            if runner.busy: runner.cancel(); self.wait(runner)
            window.close()

    def test_loading_enters_destination_with_locked_preview_then_reveals_model(self):
        self.exercise_handoff('success')

    def test_failed_handoff_stays_inline_until_explicit_return(self):
        self.exercise_handoff('failure')

    def test_cancelled_handoff_keeps_capture_and_discards_late_result(self):
        self.exercise_handoff('cancel')

    def test_inline_retry_loads_same_source_and_unlocks_controls(self):
        self.exercise_handoff('failure',retry=True)

    def test_loading_another_source_hides_old_model_and_cancel_preserves_it(self):
        self.exercise_handoff('cancel',previous_model=True)

    def test_failed_new_source_never_exposes_previous_route_controls(self):
        self.exercise_handoff('failure',previous_model=True)

    def test_unrelated_capture_image_is_never_used_as_loading_preview(self):
        self.exercise_handoff('cancel',unrelated_preview=True)

    def test_direct_project_load_returns_to_previous_planning_page_on_failure(self):
        window=WorkstationWindow(connect_camera=False)
        self.addCleanup(window.close)
        source=Path(self.model.reconstruction)
        window.planning.install_model(self.model,source)
        window.pages.setCurrentWidget(window.planning)
        with patch('femto.app.qt_planning.load_source',side_effect=ValueError('missing source')):
            self.assertTrue(window.planning.load_source(source/'missing'))
            self.wait(window.planning.runner)
        self.assertIn('missing source',window.planning._failure_details)
        with patch.object(window.planning.quality,'request',wraps=window.planning.quality.request) as request:
            window.planning.preparation.cancel.click()
            request.assert_called()
        self.assertIs(window.pages.currentWidget(),window.planning)
        self.assertIs(window.planning.model,self.model)
        self.assertEqual(window.planning.source,source)
        self.assertTrue(window.planning.head.isEnabled())

    def test_preparation_stages_follow_work_and_ignore_progress_after_cancel(self):
        from femto.app.qt_planning_loading import PlanningPreparation,STAGE_MESSAGES
        preparation=PlanningPreparation()
        self.addCleanup(preparation.close)
        self.addCleanup(preparation.stop)
        preparation.start()
        self.assertTrue(preparation.spinner.timer.isActive())
        self.assertEqual(preparation.title.text(),'正在准备三维模型…')
        preparation.progress(STAGE_MESSAGES[1])
        self.assertEqual(preparation.stage,1)
        preparation.progress('内部处理阶段信息')
        self.assertEqual(preparation.title.text(),'正在准备三维模型…')
        self.assertEqual(preparation.last_progress,'内部处理阶段信息')
        preparation.cancelling()
        preparation.progress(STAGE_MESSAGES[2])
        self.assertEqual(preparation.stage,1)
        self.assertIn('取消',preparation.title.text())
        preparation.start()
        self.assertEqual(preparation.stage,0)
        self.assertEqual(preparation.title.text(),'正在准备三维模型…')
