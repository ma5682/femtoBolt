"""Finished local captures navigate to model/route review without auto-confirming."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch
import numpy as np
from PySide6.QtCore import QEventLoop,QTimer
from PySide6.QtWidgets import QApplication
from femto.app.qt_capture import CapturePage,CaptureState


class AutoRouteTests(TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.page=CapturePage(connect_camera=False)
        self.addCleanup(self.page.close);self.addCleanup(self.page.shutdown)
        self.record=dict(confirmed_for_planning=False,reconstruction_scope={'schema':'test'})
        points=np.array([[0.,0.,.5],[.01,.01,.5]])
        self.data=dict(clouds={'final':(points,np.ones_like(points))},surface=None,
                       quality={},scan_roi_quality=None,scan_roi_preview=None,
                       acquisition_completion=self.record,frame_count=1)
        self.requested=[]
        self.page.planning_preview_requested.connect(self.requested.append)

    def wait(self):
        loop=QEventLoop();deadline=QTimer();deadline.setSingleShot(True)
        deadline.timeout.connect(loop.quit);self.page.review_task.finished.connect(loop.quit)
        try:
            deadline.start(10000)
            if self.page.review_task.busy:loop.exec()
        finally:
            deadline.stop();self.page.review_task.finished.disconnect(loop.quit)
        self.assertFalse(self.page.review_task.busy)

    def finish(self,count=1):
        self.page.session=CaptureState(session_id='current',active=True,state='processing',count=count)
        self.data['frame_count']=count
        self.page.handle_event('local_finished',dict(session_id='current',result=str(self.root),
                               acquisition_completion=self.record))

    def test_single_and_multiframe_finish_navigate_once_without_confirmation(self):
        for count in (1,3):
            with self.subTest(count=count),patch('femto.app.qt_capture.read_review',return_value=self.data), \
                 patch('femto.dataset.acquisition_completion.confirm_for_planning') as confirm:
                self.requested.clear();self.finish(count);self.wait()
                self.assertEqual(self.requested,[str(self.root/'reconstruction')])
                self.assertEqual(self.page.session.count,count)
                self.assertFalse(self.page.session.acquisition_completion['confirmed_for_planning'])
                confirm.assert_not_called()
                self.page.load_review();self.wait()
                self.assertEqual(len(self.requested),1)

    def test_failed_read_stays_retryable_and_does_not_navigate(self):
        with patch('femto.app.qt_capture.read_review',side_effect=ValueError('test read failed')):
            self.finish();self.wait()
        self.assertEqual(self.requested,[])
        self.assertEqual(self.page.session.state,'review_load_error')
        with patch('femto.app.qt_capture.read_review',return_value=self.data):
            self.page.load_review();self.wait()
        self.assertEqual(self.requested,[str(self.root/'reconstruction')])

    def test_historical_review_does_not_automatically_navigate(self):
        self.page.session=CaptureState(session_id='history',result=self.root,state='ready')
        self.page._history_replay=True
        self.page._auto_plan_result=self.root
        with patch('femto.app.qt_capture.read_review',return_value=self.data):
            self.page.load_review();self.wait()
        self.assertEqual(self.requested,[])

    def test_return_action_uses_route_preview_without_writing_confirmation(self):
        with patch('femto.app.qt_capture.read_review',return_value=self.data), \
             patch('femto.dataset.acquisition_completion.confirm_for_planning') as confirm:
            self.finish();self.wait();self.requested.clear()
            self.page._workflow_view=1
            self.page.primary_action();self.wait()
            self.assertEqual(self.requested,[str(self.root/'reconstruction')])
            self.assertEqual(self.page.primary.text(),'进入路线规划')
            confirm.assert_not_called()
