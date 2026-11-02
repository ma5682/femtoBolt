"""History opens complete only after the selected data was installed."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from copy import copy
from pathlib import Path
import threading
import time
import unittest
from unittest.mock import patch

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QTreeWidgetItem

from femto.app.qt_history import HistoryDialog
from femto.app.qt_main import WorkstationWindow
import test_qt_planning as fixtures


class HistoryNavigationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.fixture = fixtures.QtPlanningTests()
        self.fixture.setUpClass(); self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.window = WorkstationWindow(connect_camera=False)
        self.old_source = Path(self.fixture.temp.name)/'old.region.zip'
        self.window.review.install_model(self.fixture.model, self.old_source)
        self.window.pages.setCurrentWidget(self.window.review)
        self.dialog = HistoryDialog(self.window, root=self.fixture.temp.name)
        self.dialog.show()
        self.wait(lambda: not self.dialog.busy)
        self.release = threading.Event()
        self.addCleanup(self.finish)
        # Catalogue and asset validity are covered by test_qt_history. These
        # tests drive the real asynchronous workspace read after delegation.
        self.target = Path(self.fixture.temp.name)/'next.region.zip'
        self.row = dict(key='next', name='待核验记录', result=str(self.target.parent),
                        source=None, state='model', count=1, time=0)
        item = QTreeWidgetItem(['', '', '', self.row['name']])
        item.setData(0, Qt.UserRole, self.row)
        self.dialog.session_tree.addTopLevelItem(item)
        self.dialog.session_tree.setCurrentItem(item)
        self.model = copy(self.fixture.model)
        self.model.task_context = dict(name='新记录')

    def wait(self, predicate):
        deadline = time.monotonic()+10
        while time.monotonic() < deadline:
            self.app.processEvents()
            if predicate(): return
            QTest.qWait(10)
        self.assertTrue(predicate())

    def finish(self):
        self.release.set()
        self.wait(lambda: not self.window.review.busy)
        self.dialog.close()
        self.window.close()
        self.app.processEvents()

    def read_later(self, cancel, progress):
        progress('核验模型与来源…')
        if not self.release.wait(5): raise TimeoutError('test release not signalled')
        return self.model

    def open(self, reader=None):
        def load(path, cancel, progress, **kwargs):
            self.assertEqual(path, self.target.resolve())
            return (reader or self.read_later)(cancel, progress)
        with patch('femto.app.qt_review.load_source', side_effect=load):
            self.assertTrue(self.window._open_planning_from_history(self.dialog, str(self.target)))
            self.wait(lambda: self.dialog.opening)
            # The patched callable remains captured by TaskRunner after start;
            # the worker has entered the reader before returning here.
            self.wait(lambda: self.dialog.status.text() == '核验模型与来源…')

    def test_success_waits_for_loaded_model_and_blocks_duplicate_open(self):
        self.open()
        self.assertTrue(self.dialog.isVisible())
        self.assertNotEqual(self.dialog.result(), QDialog.Accepted)
        self.assertFalse(self.dialog.tabs.isEnabled())
        self.assertFalse(self.dialog.search.isEnabled())
        self.assertFalse(self.dialog.open_button.isEnabled())
        self.assertTrue(self.dialog.cancel_button.isEnabled())
        self.assertIs(self.window.review.model, self.fixture.model)
        self.assertFalse(self.window._open_planning_from_history(self.dialog, 'other'))
        self.release.set(); self.wait(lambda: not self.dialog.opening)
        self.assertEqual(self.dialog.result(), QDialog.Accepted)
        self.assertIs(self.window.review.model, self.model)
        self.assertEqual(self.window.review.source, self.target.resolve())

    def test_failure_keeps_selection_and_original_model_then_retry_succeeds(self):
        def fail(cancel, progress):
            self.read_later(cancel, progress)
            raise FileNotFoundError('缺失参考深度')
        self.open(fail)
        self.release.set(); self.wait(lambda: not self.dialog.opening)
        self.assertTrue(self.dialog.isVisible())
        self.assertEqual(self.dialog.current(), self.row)
        self.assertTrue(self.dialog.search.isEnabled())
        self.assertIn('缺失参考深度', self.dialog.status.text())
        self.assertIn('重试', self.dialog.status.text())
        self.assertIs(self.window.review.model, self.fixture.model)
        self.assertEqual(self.window.review.source, self.old_source.resolve())
        self.release.clear(); self.open()
        self.release.set(); self.wait(lambda: not self.dialog.opening)
        self.assertEqual(self.dialog.result(), QDialog.Accepted)

    def test_cancel_rejects_late_success_and_keeps_picker_for_retry(self):
        self.open()
        self.dialog.cancel_button.click()
        self.assertFalse(self.dialog.cancel_button.isEnabled())
        self.release.set(); self.wait(lambda: not self.dialog.opening)
        self.assertIs(self.window.review.model, self.fixture.model)
        self.assertTrue(self.dialog.isVisible())
        self.assertNotEqual(self.dialog.result(), QDialog.Accepted)
        self.assertIn('取消', self.dialog.status.text())
        self.assertTrue(self.dialog.search.isEnabled())

    def test_close_during_load_waits_for_cancel_and_disconnects_callbacks(self):
        self.open()
        self.dialog.close()
        self.assertTrue(self.dialog.isVisible())
        self.assertTrue(self.window.review.runner._cancel.is_set())
        self.release.set(); self.wait(lambda: not self.dialog.isVisible())
        self.assertIs(self.window.review.model, self.fixture.model)
        message = self.dialog.status.text()
        self.window.review.runner.progress.emit('下一次读取进度')
        self.window.review.loaded.emit(str(self.target))
        self.window.review.runner.finished.emit()
        self.assertEqual(self.dialog.status.text(), message)
        self.assertNotEqual(self.dialog.result(), QDialog.Accepted)
