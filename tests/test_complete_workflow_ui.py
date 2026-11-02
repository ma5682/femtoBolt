"""One task through capture-boundary replay and real plan/save/history/retry.

Only camera transport, finished capture preview and its initial model adapter
are synthetic. Geometry, archives, history readers and version saves are real.
"""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from copy import deepcopy
import json
from pathlib import Path
import time
import unittest
from unittest.mock import patch

import numpy as np
from PySide6.QtCore import Qt,QEventLoop,QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication,QMessageBox

from femto.app.qt_main import WorkstationWindow,apply_theme
from femto.app.qt_history import HistoryDialog
from femto.hardware.orbbec_rgbd import AlignedFrame
from femto.dataset.task_context import save_task
import test_qt_planning as fixture


class CompleteWorkflowUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QApplication.instance() or QApplication([]);apply_theme(cls.app)

    def setUp(self):
        self.fixture=fixture.QtPlanningTests();self.fixture.setUpClass();self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups);self.fixture.page.hide()
        self.root=Path(self.fixture.temp.name);self.outputs=self.root/'outputs'
        for target,value in [('femto.planning.region_project.OUTPUTS_ROOT',self.outputs),
                             ('femto.app.qt_planning.OUTPUTS_ROOT',self.outputs),
                             ('femto.planning.planning_drafts.ROOT',self.root/'drafts')]:
            modifier=patch(target,value);modifier.start();self.addCleanup(modifier.stop)
        self.window=WorkstationWindow(connect_camera=False);self.window.resize(1120,740);self.window.show()
        self.addCleanup(self.close_window)
        self.artifacts=Path(os.environ['FEMTO_UI_AUDIT_DIR']).resolve() if os.environ.get('FEMTO_UI_AUDIT_DIR') else None
        if self.artifacts:self.artifacts.mkdir(parents=True,exist_ok=False)
        self.events=[]

    def close_window(self):
        self.window.capture.session.active=False
        self.window.close();self.app.processEvents()

    def wait(self,predicate):
        loop=QEventLoop();poll=QTimer();deadline=QTimer()
        poll.setInterval(10);deadline.setSingleShot(True)
        poll.timeout.connect(lambda:loop.quit() if predicate() else None);deadline.timeout.connect(loop.quit)
        poll.start();deadline.start(60000)
        if not predicate():loop.exec()
        poll.stop();deadline.stop();self.assertTrue(predicate(),'UI flow did not finish')
        self.app.processEvents()

    def shot(self,label,widget=None):
        page=self.window.planning
        self.events.append(dict(stage=label,scope='synthetic camera boundary; real downstream persistence',
            inspector=dict(viewport=page.inspector_scroll.viewport().width(),panel=page.inspector_scroll.widget().width(),
                tabs=page.tabs.width(),active_page=page.tabs.currentWidget().width(),summary=page.save_summary.width(),
                tab_index=page.tabs.currentIndex(),page_minima=[w.minimumSizeHint().width() for w in page.pages],
                panel_minimum=page.inspector_scroll.widget().minimumSizeHint().width())))
        if self.artifacts:
            self.events[-1]['wide_widgets']=[dict(type=type(w).__name__,name=w.objectName(),
                minimum=w.minimumSizeHint().width(),text=w.text()[:90] if callable(getattr(w,'text',None)) else '')
                for w in page.inspector_scroll.widget().findChildren(__import__('PySide6.QtWidgets',fromlist=['QWidget']).QWidget)
                if w.isVisible() and w.minimumSizeHint().width()>400]
        if self.artifacts:
            (widget or self.window).grab().save(str(self.artifacts/(label+'.png')))
            (self.artifacts/'trace.json').write_text(json.dumps(self.events,ensure_ascii=False,indent=2),encoding='utf-8')

    def test_new_capture_plan_save_history_edit_failure_and_retry_same_task(self):
        window=self.window;capture=window.capture;model=self.fixture.model
        self.assertTrue(window.new_task());task_id=capture.task_context['task_id'];self.shot('01_new_task')
        capture.choose_scan_roi()
        capture.task_name.setText('UI-chain-synthetic')
        frame=AlignedFrame(color_bgr=model.rgb[...,::-1].copy(),depth_mm=model.projection.depth.copy(),
            timestamp_us=123,system_timestamp_us=123,frame_index=0,color_timestamp_us=123,
            color_system_timestamp_us=123,depth_timestamp_us=123,depth_system_timestamp_us=123)
        capture.host.latest_frame=frame;capture.host.latest_received=time.monotonic()
        capture.device_info['depth_range_m']=[.25,2.21]
        with patch.object(capture,'live_frame_valid',return_value=True),patch.object(capture,'send',return_value=True) as send:
            capture.choose_scan_roi()
            capture._roi_changed([.125,.125,.875,.875]);self.assertTrue(capture.primary.isEnabled())
            QTest.mouseClick(capture.primary,Qt.LeftButton)
            self.assertEqual(send.call_args.args,('local_start',))
            np.testing.assert_array_equal(send.call_args.kwargs['initial_frame'].depth_mm,frame.depth_mm)
            session=capture.session.session_id
            capture.handle_event('local_pose',dict(session_id=session,accepted=3,record=dict(
                state='tracking',frame_id=2,received_monotonic=time.monotonic())))
            capture.refresh();QTest.mouseClick(capture.finish_button,Qt.LeftButton)
            model.task_context=deepcopy(capture.task_context);save_task(self.root,model.task_context)
            points=np.asarray(model.mesh.vertices).copy()
            preview=dict(clouds=dict(final=(points,np.ones_like(points)),initial=(points,np.ones_like(points))),
                surface=None,quality={},frame_count=3,task_context=model.task_context,
                scan_roi_quality=dict(status='cannot_determine',pending_pixels=1),scan_roi_preview=None)
            with patch('femto.app.qt_capture.read_review',return_value=preview):
                capture.handle_event('local_finished',dict(session_id=session,result=str(self.root)))
                self.wait(lambda:not capture.review_task.busy)
            self.assertTrue(capture.review_ready());self.shot('02_capture_boundary_replayed')
            with patch('femto.app.qt_capture.QMessageBox.question',return_value=QMessageBox.Yes),\
                    patch('femto.app.qt_planning.load_source',return_value=model):
                QTest.mouseClick(capture.primary,Qt.LeftButton)
                self.wait(lambda:not window.planning.busy)
        page=window.planning
        self.assertIs(window.pages.currentWidget(),page)
        self.assertEqual(page.model.task_context['task_id'],task_id)
        page.canvas.work[:]=page.model.projection.valid;page.canvas.rebuild_layers();page._selection_changed()
        QTest.mouseClick(page.primary,Qt.LeftButton);self.wait(lambda:not page.busy)
        self.shot('03_region_committed')
        QTest.mouseClick(page.primary,Qt.LeftButton);self.wait(lambda:not page.busy)
        self.assertTrue(page.usable());QTest.mouseClick(page.primary,Qt.LeftButton)
        page.reviewed.setChecked(True);self.shot('04_plan_reviewed')
        QTest.mouseClick(page.primary,Qt.LeftButton);self.wait(lambda:not page.busy)
        self.assertFalse(page._error,page.status_label.text());first=page.archive/'planning.region.zip';original=first.read_bytes()
        self.shot('05_first_version_saved')
        history=HistoryDialog(root=self.root);history.show();self.addCleanup(history.close)
        history.show_all_plans();self.wait(lambda:not history.busy and bool(history.records))
        history.planning_requested.connect(lambda source:window._open_planning_from_history(history,source))
        history.record_tree.setCurrentItem(history.record_tree.topLevelItem(0))
        QTest.mouseClick(history.open_button,Qt.LeftButton);self.wait(lambda:not window.review.busy)
        self.assertIsNotNone(window.review.model);self.assertFalse(window.review.canvas.editable)
        self.assertEqual(window.review.model.task_context['task_id'],task_id);self.shot('06_history_readonly')
        QTest.mouseClick(window.review.edit,Qt.LeftButton);self.wait(lambda:not page.busy)
        self.assertIsNone(page.archive);page.note.setText('v2 audit note');page._metadata_changed()
        page.set_stage(2);page.reviewed.setChecked(True)
        with patch('femto.planning.region_project._pack_assets',side_effect=OSError('injected disk write failure')):
            QTest.mouseClick(page.primary,Qt.LeftButton);self.wait(lambda:not page.busy)
        self.assertIn('当前方案仍保留',page._error);self.assertTrue(page.usable());self.assertEqual(first.read_bytes(),original)
        self.assertNotIn('诊断记录：',page.status_label.text())
        self.assertLess(page.status_label.height(),120)
        self.app.processEvents()
        self.assertLessEqual(page.inspector_scroll.widget().width(),page.inspector_scroll.viewport().width())
        with patch('femto.app.qt_planning.show_text') as details:
            page.show_details()
        self.assertIn('已发布的三维路线包',details.call_args.args[2])
        self.assertIn('injected disk write failure',details.call_args.args[2])
        self.shot('07_partial_failure_preserved')
        QTest.mouseClick(page.primary,Qt.LeftButton);self.wait(lambda:not page.busy)
        self.assertFalse(page._error,page.status_label.text());second=page.archive/'planning.region.zip'
        import zipfile
        with zipfile.ZipFile(first) as archive:a=json.loads(archive.read('project.json'))
        with zipfile.ZipFile(second) as archive:b=json.loads(archive.read('project.json'))
        self.assertEqual(first.read_bytes(),original);self.assertNotEqual(a['revision_id'],b['revision_id'])
        self.assertEqual(b['parent_revision_id'],a['revision_id']);self.assertEqual(a['task_context'],b['task_context'])
        self.shot('08_retry_new_version_saved')
