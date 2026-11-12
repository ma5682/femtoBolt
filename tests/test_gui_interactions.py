"""User-visible recovery and editing contracts without camera hardware."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import time
import unittest
from unittest.mock import patch

import numpy as np
from PySide6.QtCore import QPointF, QPoint
from PySide6.QtGui import QCloseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

from femto.app.qt_capture import CapturePage, CaptureState, REASONS
from femto.app.qt_main import WorkstationWindow
from femto.app.qt_planning_canvas import PlanningCanvas
from femto.app.qt_history import HistoryDialog
import test_qt_planning as planning_fixture


class InteractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QApplication.instance() or QApplication([])

    def wait(self, condition):
        deadline=time.monotonic()+15
        while not condition() and time.monotonic()<deadline:QTest.qWait(10)
        self.assertTrue(condition())

    def capture(self):
        page=CapturePage(connect_camera=False)
        self.addCleanup(page.close)
        self.addCleanup(page.shutdown)
        return page

    def planning(self):
        fixture=planning_fixture.QtPlanningTests()
        fixture.setUpClass();fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        return fixture

    def test_new_capture_keeps_previous_evidence_and_can_start_again(self):
        page=self.capture()
        with TemporaryDirectory() as folder:
            evidence=Path(folder)/'raw.json';evidence.write_text('evidence')
            page.session=CaptureState(session_id='previous',state='ready',result=Path(folder))
            page.refresh()
            self.assertFalse(page.new_button.isHidden())
            page.new_button.click()
            self.assertIsNone(page.session.result)
            self.assertEqual(evidence.read_text(),'evidence')
            page.choose_scan_roi()
            with patch.object(page,'live_frame_valid',return_value=True),patch.object(page,'send') as send:
                page.primary_action()
            send.assert_not_called()
            self.assertFalse(page.session.active)
            from test_scan_roi import frame
            from femto.capture.scan_roi import prepare
            initial=frame()
            page.host.latest_frame=initial
            page.scan_roi_rect=[.125,.125,.875,.875]
            page.scan_roi_size=[64,64]
            page.device_info['depth_range_m']=[.25,2.21]
            expected_roi=prepare(page.scan_roi_rect,initial,*page._depth_limits())
            with patch.object(page,'live_frame_valid',return_value=True),patch.object(page,'send',return_value=True) as send:
                page.primary_action()
            send.assert_called_once()
            self.assertEqual(send.call_args.args,('local_start',))
            self.assertEqual(set(send.call_args.kwargs),{'initial_frame','scan_roi','region_first'})
            self.assertIs(send.call_args.kwargs['region_first'],True)
            self.assertEqual(send.call_args.kwargs['scan_roi'],expected_roi)
            sent_frame=send.call_args.kwargs['initial_frame']
            self.assertIsNot(sent_frame,initial)
            self.assertEqual(sent_frame.frame_index,initial.frame_index)
            np.testing.assert_array_equal(sent_frame.color_bgr,initial.color_bgr)
            np.testing.assert_array_equal(sent_frame.depth_mm,initial.depth_mm)
            self.assertEqual(evidence.read_text(),'evidence')
            self.assertTrue(page.session.active)
            self.assertNotEqual(page.session.session_id,'previous')
            page.session.active=False

    def test_workspace_exceeded_instruction_and_cancelled_feedback(self):
        page=self.capture()
        page.session=CaptureState(active=True,count=3,state='tracking',record={
            'state':'tracking','received_monotonic':time.monotonic(),
            'local_workspace':{'within_limit':False}})
        with patch.object(page,'live_frame_valid',return_value=True):page.refresh()
        self.assertEqual(page.guidance.text(),REASONS['outside_local_workspace'])
        self.assertFalse(page.primary.isEnabled())
        page.session.active=False;page.session.state='cancelled';page.session.source=Path('saved_raw')
        page.refresh()
        self.assertIn('已取消',page.state_title.text())
        self.assertIn('saved_raw',page.guidance.text())

    def test_failed_model_has_in_place_retry(self):
        page=self.capture()
        page.session=CaptureState(state='review_load_error',result=Path('previous'))
        page.refresh()
        self.assertFalse(page.retry_button.isHidden())
        with patch.object(page,'load_review') as load:page.retry_button.click()
        load.assert_called_once()

    def test_close_wait_or_cancel_then_close(self):
        window=WorkstationWindow(connect_camera=False)
        try:
            window.capture.session.active=True
            with patch.object(QMessageBox,'question',return_value=QMessageBox.No) as question:
                event=QCloseEvent();window.closeEvent(event)
            question.assert_called_once();self.assertFalse(event.isAccepted())
            with patch.object(QMessageBox,'question',return_value=QMessageBox.Yes),patch.object(window.capture,'finish') as finish:
                event=QCloseEvent();window.closeEvent(event)
            finish.assert_called_once_with(False)
            self.assertTrue(window._close_timer.isActive())
            window.capture.session.active=False
            with patch.object(QMessageBox,'question',side_effect=AssertionError('repeat prompt')):
                event=QCloseEvent();window.closeEvent(event)
            self.assertTrue(event.isAccepted());self.assertFalse(window._close_timer.isActive())
        finally:
            window.capture.session.active=False
            window._close_timer.stop();window.planning.shutdown();window.capture.shutdown();window.close()

    def test_undo_redo_clear_and_mouse_zoom_anchor(self):
        canvas=PlanningCanvas();canvas.resize(800,600)
        canvas.set_model(SimpleNamespace(rgb=np.zeros((100,120,3),np.uint8)))
        canvas._before=(canvas.work.copy(),canvas.forbidden.copy())
        canvas.work[20:40,50:70]=1;canvas.forbidden[25:28,55:58]=1;canvas.finish_stroke()
        selected=canvas.work.copy();forbidden=canvas.forbidden.copy()
        canvas.clear();self.assertFalse(canvas.work.any());self.assertFalse(canvas.forbidden.any())
        canvas.undo();np.testing.assert_array_equal(canvas.work,selected)
        np.testing.assert_array_equal(canvas.forbidden,forbidden)
        canvas.redo();self.assertFalse(canvas.work.any())
        canvas.undo()
        position=QPointF(520,220);before=canvas.image_point(position)
        canvas.wheelEvent(SimpleNamespace(position=lambda:position,angleDelta=lambda:QPoint(0,120)))
        np.testing.assert_allclose(canvas.image_point(position),before,atol=1e-10)
        canvas.fit_selection()
        np.testing.assert_allclose(canvas.image_point(QPointF(400,300)),(60,30),atol=1e-10)
        canvas.undo()
        canvas._before=(canvas.work.copy(),canvas.forbidden.copy());canvas.work[0,0]=1;canvas.finish_stroke()
        self.assertFalse(canvas._redo)
        canvas.close()

    def test_draft_failure_on_close_keeps_capture_available(self):
        window=WorkstationWindow(connect_camera=False)
        try:
            with patch.object(window.planning,'can_leave',return_value=True), \
                 patch.object(window.planning,'shutdown',return_value=False), \
                 patch.object(window.planning,'journal',SimpleNamespace(status=lambda:'error')), \
                 patch.object(window.capture,'shutdown') as stop, \
                 patch.object(QMessageBox,'warning') as warning:
                event=QCloseEvent();window.closeEvent(event)
            self.assertFalse(event.isAccepted());self.assertTrue(window.isEnabled())
            stop.assert_not_called();warning.assert_called_once()
        finally:window.planning.shutdown();window.capture.shutdown();window.close()

    def test_parameter_edit_preserves_route_but_blocks_stale_save(self):
        f=self.planning();f.mark()
        from femto.planning.treatment_head import head_parameters
        f.page.head.set_record(head_parameters('direct_row_spacing',row_spacing_mm=5))
        f.page.generate();f.wait_task()
        page=f.page;plan=page.plan
        self.assertFalse(page.parameters_button.isChecked())
        page.head.fields['row_spacing_mm'].setText('7')
        self.assertIs(page.plan,plan);self.assertFalse(page.usable())
        self.assertIn('旧路线',page.warning.text())
        page.reviewed.setChecked(True)
        with patch('femto.app.qt_planning.surface_workspace.export') as export:page.save_plan()
        export.assert_not_called()
        page.head.fields['row_spacing_mm'].setText('')
        self.assertIs(page.plan,plan);self.assertFalse(page.usable())
        page.head.fields['row_spacing_mm'].setText('5')
        self.assertIs(page.plan,plan);self.assertTrue(page.usable())
        self.assertFalse(page.reviewed.isChecked())

    def test_failed_draft_blocks_source_switch_and_can_retry(self):
        f=self.planning();f.mark();f.journal.wait()
        page=f.page;model=page.model;source=page.source
        with patch('femto.planning.planning_drafts.tempfile.NamedTemporaryFile',side_effect=OSError('disk full')), \
             patch.object(QMessageBox,'question',return_value=QMessageBox.Yes), \
             patch('femto.app.qt_planning.load_source') as load:
            page.load_source(Path(f.temp.name)/'another')
            self.wait(lambda:not page.busy)
            load.assert_not_called()
            page._refresh_draft_status()
            self.assertIn('保存失败',page.draft_label.text())
            self.assertFalse(page.retry_draft_button.isHidden())
        self.assertIs(page.model,model);self.assertEqual(page.source,source)
        # The preparation failure page owns navigation until the user returns.
        self.assertTrue(page.preparation.cancel.isVisible())
        page.preparation.cancel.click()
        self.assertIsNone(page.loading_source)
        self.assertTrue(page.retry_draft_button.isVisible())
        self.assertTrue(page.retry_draft_button.isEnabled())
        page.retry_draft_button.click();f.journal.wait();page._refresh_draft_status()
        self.assertEqual(page.draft_label.text(),'草稿已保存')

    def test_preparation_retry_resaves_failed_draft_before_switching_source(self):
        from femto.app.qt_planning_io import clone_model
        from femto.planning.planning_drafts import Journal, read
        f=self.planning();f.mark()
        page=f.page;model=page.model;source=page.source
        page.draft_timer.stop();page.checkpoint();f.journal.wait()
        page._new_source_coarse=False
        target=Path(f.temp.name)/'another'
        replacement=clone_model(model,freeze_inputs=True)
        next_journal=Journal(target,Path(f.temp.name)/'next-drafts')
        loaded=[];page.loaded.connect(loaded.append)
        with patch.object(QMessageBox,'question',return_value=QMessageBox.Yes) as question, \
             patch('femto.app.qt_planning.load_source',return_value=replacement) as load, \
             patch('femto.app.qt_planning_resolution.journal_for',return_value=next_journal):
            with patch('femto.planning.planning_drafts.tempfile.NamedTemporaryFile',side_effect=OSError('disk full')):
                self.assertTrue(page.load_source(target,restore_draft=False))
                self.wait(lambda:not page.busy)
                self.assertEqual(f.journal.status(),'error');load.assert_not_called()
                self.assertIs(page.model,model);self.assertEqual(page.source,source)
                sequence=f.journal.sequence
                self.assertTrue(page.preparation.retry.isVisible())
                page.preparation.retry.click();self.wait(lambda:not page.busy)
                self.assertGreater(f.journal.sequence,sequence)
                self.assertEqual(f.journal.status(),'error');load.assert_not_called()
                self.assertEqual(loaded,[])
            # Once storage is available, retry the visible action. The old
            # draft must reach disk before the destination can be installed.
            page.preparation.retry.click();self.wait(lambda:not page.busy)
            self.assertFalse(page._error,page.status_label.text())
            self.assertEqual(f.journal.status(),'saved')
            self.assertEqual(read(f.journal.path,model)['regions'],model.store.regions)
            load.assert_called_once()
            question.assert_called_once()
        self.assertIsNone(page.loading_source)
        self.assertEqual(page.source,target.resolve())
        self.assertIs(page.journal,next_journal)
        self.assertEqual(loaded,[str(target.resolve())])

    def test_history_diagnostic_buttons_work_for_non_openable_record(self):
        with TemporaryDirectory() as folder:
            report=Path(folder)/'report.json';report.write_text('{"error":"test"}')
            dialog=HistoryDialog(root=folder)
            try:
                QTest.qWait(30);self.wait(lambda:not dialog.busy)
                dialog.current=lambda:dict(report=str(report),source=folder,file_status='损坏')
                dialog._selection_changed()
                self.assertFalse(dialog.open_button.isEnabled())
                self.assertTrue(dialog.report_button.isEnabled())
                with patch.object(dialog,'_show_dialog') as show:dialog.report_button.click()
                show.assert_called_once()
                with patch('femto.app.qt_history.open_path') as opened:dialog.folder_button.click()
                opened.assert_called_once_with(folder)
            finally:dialog.close()


if __name__=='__main__':unittest.main()
