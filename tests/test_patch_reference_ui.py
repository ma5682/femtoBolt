"""Local single/multi-view sources share one intent-preserving planning adapter."""
from pathlib import Path
import threading
import unittest
from unittest.mock import patch
import numpy as np
import test_qt_planning as fixture

from femto.app.qt_planning_resolution import prepare_new_source, journal_for, start_creation
from femto.app.qt_planning_io import masks_from_region
from femto.planning.patch_reference import SCHEMA
from femto.planning.treatment_head import head_parameters


class PatchReferenceUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):fixture.QtPlanningTests.setUpClass()

    def setUp(self):
        self.case=fixture.QtPlanningTests();self.case.setUp();self.addCleanup(self.case.doCleanups)
        self.page=self.case.page;self.source=self.case.model;self.root=Path(self.case.temp.name)
        scoped=patch('femto.planning.planning_drafts.ROOT',self.root/'migration_drafts')
        scoped.start();self.addCleanup(scoped.stop)
        self.source.store.local_context={'reconstruction_scope':{'schema':'synthetic_test'}}
        self.work=np.zeros(self.source.rgb.shape[:2],np.uint8);self.work[25:50,20:82]=1
        self.forbidden=np.zeros_like(self.work);self.forbidden[30:45,48:54]=1
        self.source.pre_reconstruction_selection=self.work.copy()

    def test_standard_draft_preserves_pixels_head_and_old_file(self):
        self.page.journal=journal_for(self.page.source,self.source)
        self.page.canvas.work[:]=self.work;self.page.canvas.forbidden[:]=self.forbidden
        self.page.head.set_record(head_parameters('fixed_width_auto',effective_treatment_width_mm=12.,overlap_percent=25.))
        request=self.page.request();self.page.dirty=True;self.page.checkpoint();self.page.journal.wait()
        old=self.page.journal.path.read_bytes()
        result=prepare_new_source(self.source,self.page.source,enabled=True,restore_draft=True,
                                  check=lambda:None,progress=lambda text:None)
        np.testing.assert_array_equal(result.pre_reconstruction_selection,self.work)
        np.testing.assert_array_equal(result.pre_reconstruction_forbidden,self.forbidden)
        self.assertEqual(result.planning_parameters,request)
        self.assertEqual(self.page.journal.path.read_bytes(),old)
        self.assertNotEqual(journal_for(self.page.source,result).path,self.page.journal.path)
        self.assertNotIn('derivation',self.source.view_reference)

    def test_explicit_new_version_keeps_selected_and_erased_pixels(self):
        self.page.canvas.work[:]=self.work;self.page.canvas.forbidden[:]=self.forbidden
        self.page.head.set_record(head_parameters('fixed_width_auto',effective_treatment_width_mm=12.,overlap_percent=25.))
        head=self.page.head.record()
        self.assertTrue(start_creation(self.page,new_version=True));self.case.wait_task()
        self.assertEqual(self.page.model.view_reference['derivation']['schema'],SCHEMA)
        work,forbidden=masks_from_region(self.page.region,self.work.shape)
        np.testing.assert_array_equal(work,self.work & ~self.forbidden)
        np.testing.assert_array_equal(forbidden,self.forbidden)
        self.assertEqual(self.page.head.record(),head);self.assertEqual(self.page._stage,1)

    def test_cancelled_local_conversion_leaves_original_model_installed(self):
        entered,release=threading.Event(),threading.Event()
        def create(*args,**kwargs):
            entered.set()
            if not release.wait(3):raise RuntimeError('Test did not release worker')
            kwargs['check']()
            raise AssertionError('Cancellation was not checked')
        with patch('femto.planning.patch_reference.create',side_effect=create):
            self.assertTrue(start_creation(self.page,new_version=True));self.assertTrue(entered.wait(2))
            self.page.runner.cancel();release.set();self.case.wait_task()
        self.assertIs(self.page.model,self.source)
        self.assertNotIn('derivation',self.source.view_reference)

    def test_busy_fixed_selection_tab_rollback_does_not_reenter(self):
        from unittest.mock import PropertyMock
        self.page._stage=0
        with patch.object(type(self.page),'busy',new_callable=PropertyMock,return_value=True):
            self.page.tabs.setCurrentIndex(1)
            self.assertEqual(self.page.tabs.currentIndex(),0)
            self.assertEqual(self.page._stage,0)
        self.page._stage=1
        self.page.tabs.setCurrentIndex(0)
        self.page._stage_selected(0)
        self.assertEqual(self.page.tabs.currentIndex(),1)


if __name__=='__main__':unittest.main()
