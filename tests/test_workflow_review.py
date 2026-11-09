"""Saved workflow roundtrips, immutable versions and same-source replay."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')

from copy import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
from unittest.mock import patch
import unittest
import zipfile

import numpy as np
from PySide6.QtCore import Qt, QEventLoop, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

from femto.app.qt_main import WorkstationWindow
from femto.app.qt_review import ReviewPage
from femto.app.qt_history import HistoryDialog
from femto.app.qt_planning_io import verified_capture_source
from femto.dataset.task_context import new_task,read_task,save_task
import test_qt_planning as fixture_module


class WorkflowReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])

    def fixture(self):
        f=fixture_module.QtPlanningTests();f.setUpClass();f.setUp()
        self.addCleanup(f.doCleanups)
        return f

    def wait(self,predicate):
        # Match QApplication.exec: repeated qWait can starve Python readers.
        # Keep the original 90-second deadline and completion assertion.
        loop=QEventLoop();poll=QTimer();deadline=QTimer()
        poll.setInterval(20);deadline.setSingleShot(True);deadline.setTimerType(Qt.PreciseTimer)
        poll.timeout.connect(lambda:loop.quit() if predicate() else None)
        deadline.timeout.connect(loop.quit)
        try:
            poll.start();deadline.start(90000)
            if not predicate():loop.exec()
        finally:poll.stop();deadline.stop()
        self.assertTrue(predicate())

    def metadata(self,path):
        with zipfile.ZipFile(path) as archive:return json.loads(archive.read('project.json'))

    def test_model_layer_keeps_visible_return_to_saved_selection(self):
        f=self.fixture();f.mark(split=True)
        review=ReviewPage();review.show();self.addCleanup(review.close)
        review.install_model(f.page.model,f.page.source)
        self.assertIsNotNone(review.region);self.assertIsNone(review.plan)
        review.layers.setCurrentIndex(0);self.app.processEvents()
        # All three model/selection/route states share one active workflow
        # stage. Its disabled stage button cannot replace this local control.
        self.assertTrue(review.layers.isVisible())
        self.assertTrue(review.fit.isVisible())
        review.layers.setCurrentIndex(1)
        self.assertTrue(review.canvas.show_regions)
        self.assertFalse(review.canvas.editable)

    def test_geometry_overlay_from_3d_opens_reference_without_changing_selection(self):
        from test_qt_scene_panel import FakeViewport
        f=self.fixture();f.mark(split=True)
        review=ReviewPage();review.show();self.addCleanup(review.close)
        review.install_model(f.page.model,f.page.source)
        before=review.canvas.work.copy()
        with patch('femto.app.qt_scene_panel.create_viewport',side_effect=FakeViewport):
            review.scene_panel.mode.setCurrentIndex(1)
            self.assertTrue(review.scene_panel.is_3d)
            review.quality_toggle.click()
        self.assertFalse(review.scene_panel.is_3d)
        self.assertTrue(review.canvas.support)
        np.testing.assert_array_equal(review.canvas.work,before)

    def test_header_tracks_pending_record_and_restores_committed_source_on_failure(self):
        f=self.fixture()
        window=WorkstationWindow(connect_camera=False)
        released=threading.Event()
        try:
            model=copy(f.model)
            model.task_context=dict(new_task(),name='已核验任务甲')
            first=Path(f.temp.name)/'record-a.zip'
            window.review.install_model(model,first)
            window.pages.setCurrentWidget(window.review)
            self.assertIn('已核验任务甲',window.subtitle.text())
            second=Path(f.temp.name)/'missing-b.zip'
            def fail_read(*args,**kwargs):
                released.wait(5)
                raise FileNotFoundError('测试记录缺失')
            with patch('femto.app.qt_review.load_source',side_effect=fail_read):
                self.assertTrue(window.open_review(second))
                self.assertIn('正在核验',window.project_badge.text())
                self.assertIn('missing-b.zip',window.subtitle.text())
                self.assertNotIn('已核验任务甲',window.subtitle.text())
                released.set()
                self.wait(lambda:not window.review.busy)
            self.assertIs(window.review.model,model)
            self.assertEqual(window.review.source,first.resolve())
            self.assertIn('已核验任务甲',window.subtitle.text())
            self.assertNotIn('部位：',window.subtitle.text())
            self.assertIn('只读',window.project_badge.text())
            self.assertEqual(window.project_badge.toolTip(),str(first.resolve()))
            self.assertIn('记录未切换',window.review.status.text())
        finally:
            released.set()
            self.wait(lambda:not window.review.busy)
            window.close()

    def test_save_review_edit_new_version_and_find_both_in_history(self):
        f=self.fixture();page=f.page;root=Path(f.temp.name)
        f.mark(split=True)
        page.model.task_context=dict(new_task(),name='任务甲')
        page.generate();f.wait_task()
        page.set_stage(2);page.reviewed.setChecked(True)
        with patch('femto.app.qt_planning.OUTPUTS_ROOT',root/'outputs'):
            page.save_plan();f.wait_task()
        first=page.archive/'planning.region.zip';original=first.read_bytes();first_data=self.metadata(first)
        self.assertEqual(first_data['task_context']['name'],'任务甲')
        self.assertTrue(page.saved_actions.isVisible())
        requested=[];page.review_requested.connect(requested.append);page.view_saved()
        self.assertEqual(requested,[str(first)])
        review=ReviewPage();review.show()
        try:
            review.load_source(first);self.wait(lambda:not review.busy)
            self.assertIsNotNone(review.model)
            self.assertFalse(review.canvas.editable)
            self.assertEqual(review.plan['segments'],page.plan['segments'])
            mask=review.canvas.work.copy()
            QTest.mouseClick(review.canvas,Qt.LeftButton,pos=review.canvas.rect().center())
            np.testing.assert_array_equal(review.canvas.work,mask)
            review.layers.setCurrentIndex(0)
            self.assertFalse(review.canvas.show_regions);self.assertFalse(review.canvas.show_plan)
            self.assertIn(first_data['revision_id'],review.summary.text())
            edits=[];review.edit_requested.connect(edits.append);review.edit.click()
            self.assertEqual(edits,[str(first)])
        finally:review.shutdown();review.close()
        page.load_source(first,force=True,restore_draft=False,new_version=True);f.wait_task()
        self.assertTrue(page.dirty);self.assertIsNone(page.archive)
        page.note.setText('第二版备注');page._metadata_changed()
        page.reviewed.setChecked(True)
        with patch('femto.app.qt_planning.OUTPUTS_ROOT',root/'outputs'):
            page.save_plan();f.wait_task()
        second=page.archive/'planning.region.zip';second_data=self.metadata(second)
        self.assertEqual(first.read_bytes(),original)
        self.assertNotEqual(second_data['revision_id'],first_data['revision_id'])
        self.assertEqual(second_data['parent_revision_id'],first_data['revision_id'])
        self.assertEqual(second_data['project_id'],first_data['project_id'])
        self.assertEqual(second_data['task_context'],first_data['task_context'])
        with patch('femto.app.qt_planning.QFileDialog.getSaveFileName',return_value=(str(first),'')),patch.object(QMessageBox,'information'):
            page.save_as()
        self.assertEqual(first.read_bytes(),original)
        history=HistoryDialog(root=root)
        try:
            history.show_all_plans()
            self.wait(lambda:not history.busy and len(history.records)==2)
            revisions={r.get('revision_id') for r in history.records}
            self.assertEqual(revisions,{first_data['revision_id'],second_data['revision_id']})
        finally:history.close()

    def test_unrelated_capture_is_never_shown_for_unresolved_archive(self):
        f=self.fixture()
        window=WorkstationWindow(connect_camera=False)
        try:
            f.model.capture_source=None
            window.review.install_model(f.model,Path(f.temp.name)/'A.region.zip')
            window.capture.session.result=Path(f.temp.name)/'B'
            window.pages.setCurrentWidget(window.review)
            with patch.object(window.capture,'restore_review_for_source',return_value=False),patch.object(QMessageBox,'information'):
                self.assertFalse(window.show_capture())
            self.assertIs(window.pages.currentWidget(),window.review)
            self.assertTrue(window.new_task())
            self.assertIs(window.pages.currentWidget(),window.capture)
            self.assertIsNone(window.capture.session.result)
        finally:window.review.shutdown();window.planning.shutdown();window.capture.shutdown();window.close()

    def test_capture_link_requires_complete_matching_reference(self):
        f=self.fixture();root=Path(f.temp.name)/'original'/'reconstruction';root.mkdir(parents=True)
        (root/'reconstruction_report.json').write_text('{}')
        portable=copy(f.model);portable.store=copy(f.model.store)
        portable.store.source=str(Path(f.temp.name)/'cache'/'mesh.ply')
        portable.source_lineage={'source_mesh':str(root/'scene_tsdf_mesh_m.ply')}
        wrong=copy(f.model);wrong.view_reference=dict(f.model.view_reference,wrong_source=True)
        with patch('femto.annotation.fixed_rgb_geometry.FixedRGBModel',return_value=wrong):
            self.assertIsNone(verified_capture_source(portable))
        self.assertEqual(portable.capture_source_verification['status'],'mismatch')
        self.assertIn('不一致',portable.capture_source_verification['message'])
        with patch('femto.annotation.fixed_rgb_geometry.FixedRGBModel',return_value=f.model):
            self.assertEqual(verified_capture_source(portable),root)
        self.assertEqual(portable.capture_source_verification['status'],'verified')
        self.assertEqual(portable.capture_source_verification['issues'],[])

    def test_source_verification_does_not_hide_cancellation_or_invent_a_capture(self):
        from femto.annotation.fixed_rgb_geometry import GeometryTaskCancelled
        f=self.fixture()
        self.assertIsNone(verified_capture_source(f.model))
        self.assertEqual(f.model.capture_source_verification['status'],'not_recorded')
        cancelled=threading.Event();cancelled.set()
        with self.assertRaises(GeometryTaskCancelled):verified_capture_source(f.model,cancelled)

    def test_task_sidecar_is_bound_to_its_capture(self):
        with TemporaryDirectory() as folder:
            a=Path(folder)/'a';b=Path(folder)/'b';a.mkdir();b.mkdir()
            task=dict(new_task(),name='测试任务')
            save_task(a,task);self.assertEqual(read_task(a),task)
            (b/'task_context.json').write_bytes((a/'task_context.json').read_bytes())
            with self.assertRaises(ValueError):read_task(b)
            self.assertEqual(save_task(a,dict(task,name='不得覆盖原信息')),task)
            envelope=json.loads((a/'task_context.json').read_text(encoding='utf-8'))
            envelope['schema']='unsupported-link'
            (a/'task_context.json').write_text(json.dumps(envelope),encoding='utf-8')
            with self.assertRaises(ValueError):read_task(a)

    def test_changed_saved_revision_is_not_silently_edited(self):
        f=self.fixture();page=f.page;previous=page.source
        incoming=copy(f.model);incoming.revision_id='changed'
        with patch('femto.app.qt_planning.load_source',return_value=incoming):
            page.load_source(Path(f.temp.name)/'other.zip',force=True,restore_draft=False,
                             new_version=True,expected_revision='reviewed')
            self.wait(lambda:not page.busy)
        self.assertEqual(page.source,previous)
        self.assertIs(page.model,f.model)
        self.assertIn('记录已变更',page._error)


if __name__=='__main__':unittest.main()
