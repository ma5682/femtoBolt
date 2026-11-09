"""Selection submission must survive the live 100 ms UI refresh."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from femto.app.qt_capture import CapturePage


class SelectionSubmitClickTests(TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])

    def setUp(self):
        self.page=page=CapturePage(connect_camera=False);page.timer.stop();page.resize(1200,850)
        page.session.begin();sid=page.session.session_id
        reference=SimpleNamespace(rgb=np.zeros((240,320,3),np.uint8),
                                  binding={'session_id':sid},frame_count=1)
        page.handle_event('local_region_ready',dict(session_id=sid,reference=reference,source='fixture'))
        self.local=SimpleNamespace(select_region=Mock(return_value=True))
        page.host.local_capture=self.local
        page.show();self.app.processEvents()
        self.addCleanup(self.close_page)

    def close_page(self):
        self.page.session.active=False;self.page.host.local_capture=None
        self.page.shutdown();self.page.close()

    def paint(self):
        canvas=self.page.region_editor.canvas
        QTest.mouseClick(canvas,Qt.LeftButton,pos=canvas.rect().center())
        self.assertTrue(canvas.work.any());self.assertTrue(self.page.primary.isEnabled())

    def test_mouse_press_survives_multiple_polls_then_submits_once(self):
        self.paint();button=self.page.primary
        expected=self.page.region_editor.canvas.work.copy()
        QTest.mousePress(button,Qt.LeftButton,pos=button.rect().center())
        self.assertTrue(button.isDown())
        for _ in range(3):
            self.page.poll()
            self.assertTrue(button.isDown(),'UI refresh discarded the pending click')
            self.assertTrue(button.isEnabled())
        QTest.mouseRelease(button,Qt.LeftButton,pos=button.rect().center())
        self.local.select_region.assert_called_once()
        np.testing.assert_array_equal(self.local.select_region.call_args.args[0],expected)
        self.assertEqual(self.page.session.state,'processing')
        self.assertIn('正在重建',button.text());self.assertFalse(button.isEnabled())
        self.page.poll();QTest.mouseClick(button,Qt.LeftButton,pos=button.rect().center())
        self.local.select_region.assert_called_once()

    def test_keyboard_press_survives_poll_and_empty_selection_stays_disabled(self):
        button=self.page.primary
        self.assertFalse(button.isEnabled());self.page.poll();self.assertFalse(button.isEnabled())
        self.paint();button.setFocus()
        QTest.keyPress(button,Qt.Key_Space);self.assertTrue(button.isDown())
        self.page.poll();self.assertTrue(button.isDown())
        QTest.keyRelease(button,Qt.Key_Space)
        self.local.select_region.assert_called_once()

    def test_rejected_submission_retains_paint_and_next_press_can_retry(self):
        self.paint();button=self.page.primary
        expected=self.page.region_editor.canvas.work.copy()
        self.local.select_region.side_effect=[False,True]
        QTest.mouseClick(button,Qt.LeftButton,pos=button.rect().center())
        self.assertEqual(self.page.session.state,'awaiting_region')
        self.assertIn('选区尚未提交',self.page.region_editor.note.text())
        np.testing.assert_array_equal(self.page.region_editor.canvas.work,expected)
        self.page.poll();self.assertTrue(button.isEnabled())
        QTest.mouseClick(button,Qt.LeftButton,pos=button.rect().center())
        self.assertEqual(self.page.session.state,'processing')
        self.assertEqual(self.local.select_region.call_count,2)
