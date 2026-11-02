"""Quality feedback uses measured evidence, preserves history, and rejects stale UI results."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
from PySide6.QtCore import Qt, QPoint
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from test_fixed_rgb_experience import make_model, quad, ring
from femto.app.region_quality import assess_selection, require_geometry, rescan_context
from femto.app.qt_planning import PlanningPage
from femto.app.qt_planning_canvas import PlanningCanvas
from femto.app.qt_region_quality import RegionQualityPanel
from femto.app.qt_review import ReviewPage
from femto.app.qt_main import WorkstationWindow
from femto.app.qt_planning_io import extract_region
from femto.dataset.task_context import new_task


def task_context():
    return new_task()


class QualityEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup)

    def test_reference_missing_mesh_is_pending_and_not_counted_as_consistent(self):
        depth=np.full((100,100),500,np.uint16);depth[40:60,40:60]=0
        model=make_model(self.temp.name,depth=depth)
        work=model.projection.valid.astype(np.uint8);before=deepcopy(model.view_reference)
        result=assess_selection(model,work,np.zeros_like(work))
        self.assertEqual(result['status'],'checked')
        self.assertEqual(result['counts'][7],400)
        self.assertEqual(result['pending'],400)
        self.assertTrue(np.all(result['classes'][40:60,40:60]==7))
        self.assertLess(result['ratio'],1)
        self.assertEqual(model.view_reference,before)
        self.assertFalse(model.store.regions)

    def test_actual_hole_and_depth_conflict_are_not_green(self):
        depth=np.full((100,100),500,np.uint16);depth[20:30,20:30]=300
        model=make_model(self.temp.name,ring(),depth)
        result=assess_selection(model,np.ones((100,100),np.uint8),np.zeros((100,100),np.uint8))
        self.assertEqual(result['status'],'checked')
        self.assertEqual(result['classes'][50,50],2)
        self.assertEqual(result['classes'][25,25],5)
        self.assertGreater(result['counts'][2],0)
        self.assertGreater(result['counts'][5],0)
        self.assertEqual(result['supported']+result['pending'],10000)

    def test_exclusions_holes_and_empty_work_do_not_expand_selection(self):
        model=make_model(self.temp.name)
        work=np.zeros((100,100),np.uint8);work[20:40,20:40]=1;work[25:30,25:30]=0
        forbidden=np.zeros_like(work);forbidden[20:22,20:40]=1
        result=assess_selection(model,work,forbidden)
        self.assertEqual(result['selected'],335)
        self.assertFalse(result['rgba'][forbidden.astype(bool)].any())
        self.assertEqual(result['classes'][26,26],0)
        self.assertEqual(assess_selection(model,np.zeros_like(work),forbidden)['status'],'empty')

    def test_missing_or_corrupt_evidence_is_unknown_not_success(self):
        model=make_model(self.temp.name);work=np.ones((100,100),np.uint8);zero=np.zeros_like(work)
        for field,value in [('k',np.zeros((3,3))),('depth',np.zeros((1,1))),('reasons',np.full((100,100),99))]:
            with self.subTest(field=field),patch.object(model.projection,field,value):
                result=assess_selection(model,work,zero)
                self.assertEqual(result['status'],'unavailable');self.assertIsNone(result['ratio'])
                self.assertIsNone(result['rgba'])
                with self.assertRaises(ValueError):require_geometry(result)
        with patch.object(model,'view_reference',None):
            self.assertEqual(assess_selection(model,work,zero)['status'],'unavailable')

    def test_cancellation_and_zero_consistent_support_block_new_geometry(self):
        model=make_model(self.temp.name,depth=np.zeros((100,100),np.uint16))
        work=model.projection.valid.astype(np.uint8)
        result=assess_selection(model,work,np.zeros_like(work))
        self.assertEqual(result['supported'],0)
        with self.assertRaises(ValueError):require_geometry(result)
        from femto.annotation.fixed_rgb_geometry import GeometryTaskCancelled
        cancel=threading.Event();cancel.set()
        with self.assertRaises(GeometryTaskCancelled):assess_selection(model,work,np.zeros_like(work),cancel)

    def test_out_of_range_and_source_pixel_missing_stay_unconfirmed(self):
        model=make_model(self.temp.name,quad(z=3))
        work=np.ones((100,100),np.uint8)
        result=assess_selection(model,work,np.zeros_like(work))
        self.assertGreater(result['counts'][4],0)
        self.assertEqual(result['supported'],0)
        model.projection.image_valid=np.zeros_like(work,dtype=bool)
        result=assess_selection(model,work,np.zeros_like(work))
        self.assertEqual(result['counts'][6],10000)

    def test_rescan_requires_valid_task_but_no_anatomy(self):
        task=task_context()
        self.assertEqual(rescan_context(SimpleNamespace(task_context=task)),task)
        for invalid in (None, dict(task,task_id=''), dict(task,schema='femto-ui-task-v1')):
            self.assertIsNone(rescan_context(SimpleNamespace(task_context=invalid)))


class QualityInteractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.model=make_model(self.temp.name)
        self.model.task_context=task_context()

    def wait(self,predicate):
        deadline=time.monotonic()+20
        while not predicate() and time.monotonic()<deadline:
            self.app.processEvents();time.sleep(.01)
        self.assertTrue(predicate(),'Qt operation timed out')

    def page(self):
        page=PlanningPage();page.resize(1280,850);page.show()
        page.install_model(self.model,self.model.reconstruction)
        self.addCleanup(lambda:self.close(page));return page

    def close(self,page):
        if isinstance(page,WorkstationWindow):
            self.wait(page.planning.shutdown);self.wait(page.review.shutdown);self.wait(page.capture.shutdown)
        else:self.wait(page.shutdown)
        page.close();self.app.processEvents()

    def test_paint_and_undo_recompute_counts_and_clear_old_overlay_immediately(self):
        page=self.page();self.wait(lambda:page.quality.result is not None)
        page.canvas._before=(page.canvas.work.copy(),page.canvas.forbidden.copy())
        page.canvas.work[20:40,20:40]=1;page.canvas.rebuild_layers();page.canvas.finish_stroke()
        self.assertIsNone(page.quality.result)
        self.assertIsNone(page.canvas._quality_image)
        self.wait(lambda:page.quality.result is not None)
        self.assertEqual(page.quality.result['selected'],400)
        page.canvas.undo();self.assertIsNone(page.quality.result)
        self.wait(lambda:page.quality.result is not None)
        self.assertEqual(page.quality.result['status'],'empty')
        page.canvas.redo();self.wait(lambda:page.quality.result is not None)
        self.assertEqual(page.quality.result['selected'],400)
        QTest.mousePress(page.canvas,Qt.LeftButton,pos=page.canvas.rect().center())
        self.assertIsNone(page.quality.result)
        QTest.mouseRelease(page.canvas,Qt.LeftButton,pos=page.canvas.rect().center())
        self.wait(lambda:page.quality.result is not None)
        QTest.mouseClick(page.canvas,Qt.LeftButton,pos=page.canvas.rect().center())
        self.wait(lambda:page.quality.result is not None)

    def test_cancelled_old_analysis_cannot_replace_new_selection_or_model(self):
        page=self.page();self.wait(lambda:page.quality.result is not None)
        page.canvas.work[20:40,20:40]=1
        entered=threading.Event();release=threading.Event()
        real=assess_selection;calls=[]
        def slow(model,work,forbidden,cancel):
            calls.append(model)
            if len(calls)==1:
                entered.set();release.wait(5)
            return real(model,work,forbidden)  # Deliberately ignores cancellation.
        with patch('femto.app.qt_region_quality.assess_selection',side_effect=slow):
            page.quality.request();self.wait(entered.is_set)
            model2=make_model(Path(self.temp.name)/'other')
            page.install_model(model2,model2.reconstruction)
            page.canvas.work[30:40,30:40]=1;page._selection_changed()
            release.set();self.wait(lambda:page.quality.result is not None)
        self.assertIs(page.model,model2)
        self.assertEqual(page.quality.result['selected'],100)
        self.assertEqual(page.quality.result['status'],'checked')

    def test_all_missing_area_does_not_confirm_region_and_keeps_missing_overlay(self):
        page=self.page()
        page.canvas.work[:]=0;page.canvas.work[0:4,0:4]=1;page._selection_changed()
        page.commit_region();self.wait(lambda:not page.busy)
        self.assertIsNone(page.region);self.assertTrue(page.selection_dirty)
        self.assertIn('没有可核验',page.status_label.text())
        self.assertEqual(page.quality.result['supported'],0)
        self.assertIsNotNone(page.canvas._quality_image)

    def test_partial_selection_exposes_missing_pixels_and_focus_preserves_intent(self):
        page=self.page();page.canvas.work[:]=1;page._selection_changed()
        page.commit_region();self.wait(lambda:not page.busy)
        self.assertIsNotNone(page.region);self.assertGreater(page.quality.result['pending'],0)
        old=page.canvas.work.copy();region=deepcopy(page.region)
        page.quality.focus_pending()
        self.assertTrue(page.canvas.support);self.assertEqual(page._stage,0)
        np.testing.assert_array_equal(page.canvas.work,old);self.assertEqual(page.region,region)
        self.assertIn('像素几何待确认',page.status_label.text())

    def test_legacy_region_readonly_quality_does_not_write_original_fields(self):
        work=self.model.projection.valid.astype(np.uint8)
        candidate,region=extract_region(self.model,work,np.zeros_like(work),'old','',threading.Event())
        region.pop('selection_support',None)
        before=deepcopy(candidate.store.regions)
        page=ReviewPage();self.addCleanup(lambda:self.close(page))
        page.install_model(candidate,candidate.reconstruction)
        self.wait(lambda:page.quality.result is not None)
        self.assertEqual(page.quality.result['status'],'checked')
        self.assertEqual(candidate.store.regions,before)
        self.assertFalse(page.canvas.editable)
        page.quality.focus_pending()
        self.assertEqual(candidate.store.regions,before)

    def test_rescan_preserves_task_and_history_without_reusing_geometry(self):
        window=WorkstationWindow(connect_camera=False);self.addCleanup(lambda:self.close(window))
        window.review.install_model(self.model,self.model.reconstruction)
        window.capture.clouds={'old':'geometry'}
        evidence=Path(self.temp.name)/'original';evidence.write_bytes(b'unchanged')
        task=deepcopy(self.model.task_context)
        self.assertTrue(window.rescan_region(window.review))
        self.assertEqual(window.capture.task_context,task)
        self.assertFalse(window.capture.clouds);self.assertIsNone(window.capture.session.result)
        self.assertFalse(window.capture.session.active)
        self.assertIs(window.review.model,self.model)
        self.assertEqual(evidence.read_bytes(),b'unchanged')
        self.model.task_context=None
        self.assertIsNone(rescan_context(self.model));self.assertFalse(window.rescan_region(window.review))

    def test_save_as_with_route_cannot_bypass_unavailable_evidence(self):
        page=self.page();page.canvas.work[:]=self.model.projection.valid
        page._selection_changed();page.commit_region();self.wait(lambda:not page.busy)
        target=Path(self.temp.name)/'must-not-exist.region.zip'
        page.plan=dict(spacing_mm=5,segments=[],chain_breaks=[],surface_trajectory=dict(
            statistics=dict(block_count=1,planned_block_count=1,coverage_status='complete',geometry_unconfirmed_pixel_count=0)))
        with patch.object(page,'usable',return_value=True),patch.object(page.model.projection,'k',np.zeros((3,3))),patch(
                'femto.app.qt_planning.QFileDialog.getSaveFileName',return_value=(str(target),'')),patch(
                'femto.planning.region_project.save_project') as save:
            page.save_as();self.wait(lambda:not page.busy)
            save.assert_not_called()
        self.assertFalse(target.exists());self.assertIn('标定不可用',page.status_label.text())


if __name__=='__main__':unittest.main()
