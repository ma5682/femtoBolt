"""Coarse references are default only for new planning, never legacy evidence."""
from copy import deepcopy
from pathlib import Path
import json
import threading
import unittest
from unittest.mock import patch

import numpy as np
from PySide6.QtCore import QEventLoop,QTimer

import test_qt_planning as fixture
from femto.app.qt_planning_io import clone_model
from femto.app.qt_planning_resolution import (prepare_new_source,journal_for,derivation,DERIVATION_SCHEMA)


class CoarsePlanningEntryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):fixture.QtPlanningTests.setUpClass()

    def setUp(self):
        self.case=fixture.QtPlanningTests();self.case.setUp();self.addCleanup(self.case.doCleanups)
        self.page=self.case.page;self.model=self.case.model;self.root=Path(self.case.temp.name)
        for name,value in (('femto.app.qt_planning.OUTPUTS_ROOT',self.root/'outputs'),
                           ('femto.planning.planning_drafts.ROOT',self.root/'drafts')):
            change=patch(name,value);change.start();self.addCleanup(change.stop)

    def reduced(self):
        model=clone_model(self.model,freeze_inputs=True)
        model.store.points=[];model.store.regions=[];model.store._history=[];model.saved_plan=None
        model.view_reference['derivation']=dict(schema=DERIVATION_SCHEMA,profile='stride8-original-mesh-step3mm-v1')
        return model

    def camera_sized(self):
        # Only the source-selection adapter is exercised with a mocked geometry
        # constructor below. Full RGB-D geometry is verified in the real replay.
        model=clone_model(self.model,freeze_inputs=True)
        model.rgb=np.zeros((320,320,3),np.uint8)
        return model

    def prepare(self,model=None,**kwargs):
        return prepare_new_source(model or self.model,self.model.reconstruction,
            enabled=kwargs.get('enabled',True),restore_draft=kwargs.get('restore_draft',True),
            check=lambda:None,progress=lambda text:None)

    def wait(self):
        loop=QEventLoop();timer=QTimer();timer.setSingleShot(True);timer.timeout.connect(loop.quit)
        self.page.runner.finished.connect(loop.quit)
        try:
            timer.start(10000)
            if self.page.busy:loop.exec()
        finally:
            timer.stop();self.page.runner.finished.disconnect(loop.quit)
        self.assertFalse(self.page.busy)

    def test_fresh_empty_source_uses_verified_coarse_constructor(self):
        reduced=self.reduced();view=deepcopy(self.model.view_reference)
        with patch('femto.app.qt_planning_resolution.create_reduced_model',return_value=reduced) as create:
            self.assertIs(self.prepare(self.camera_sized()),reduced)
        create.assert_called_once()
        self.assertEqual(self.model.view_reference,view)
        self.assertEqual(self.model.store.regions,[])

    def test_existing_selection_route_history_or_derivation_remains_unchanged(self):
        for kind in ('regions','points','_history','saved_plan','derived'):
            model=clone_model(self.model,freeze_inputs=True)
            if kind=='saved_plan':model.saved_plan={'old':True}
            elif kind=='derived':model.view_reference['derivation']=dict(schema=DERIVATION_SCHEMA)
            else:setattr(model.store,kind,[{'old':True}])
            with self.subTest(kind=kind),patch('femto.app.qt_planning_resolution.create_reduced_model') as create:
                self.assertIs(self.prepare(model),model);create.assert_not_called()

    def test_standard_draft_is_never_silently_moved_to_a_coarse_reference(self):
        journal=journal_for(self.model.reconstruction,self.model)
        journal.path.parent.mkdir(parents=True,exist_ok=True)
        journal.path.write_bytes(b'old standard draft is retained even if not readable')
        with patch('femto.app.qt_planning_resolution.create_reduced_model') as create:
            self.assertIs(self.prepare(),self.model);create.assert_not_called()
        self.assertEqual(journal.path.read_bytes(),b'old standard draft is retained even if not readable')
        with patch('femto.app.qt_planning_resolution.create_reduced_model',return_value=self.reduced()) as create:
            self.prepare(self.camera_sized(),restore_draft=False);create.assert_called_once()

    def test_explicit_standard_preference_skips_coarse_creation(self):
        with patch('femto.app.qt_planning_resolution.create_reduced_model') as create:
            self.assertIs(self.prepare(enabled=False),self.model);create.assert_not_called()
        from femto.app.qt_planning import PlanningPage
        blank=PlanningPage();self.addCleanup(blank.close)
        blank.planning_mode.setCurrentIndex(1)
        self.assertFalse(blank._new_source_coarse)

    def test_selected_source_uses_a_new_blank_version_without_mutating_the_old_roi(self):
        self.case.mark(split=True)
        old_model=self.page.model;old_region=deepcopy(self.page.region)
        old_source=self.page.source;old_journal=self.page.journal
        self.page.checkpoint();old_journal.wait_saved()
        old_bytes=old_journal.path.read_bytes()
        reduced=self.reduced();snapshots=[]
        def create(source,*args,**kwargs):
            snapshots.append(source)
            self.assertEqual(source.store.regions,[]);self.assertEqual(source.store.points,[])
            self.assertEqual(source.store._history,[]);self.assertIsNone(source.saved_plan)
            return reduced
        with patch('femto.app.qt_planning_resolution.create_reduced_model',side_effect=create):
            self.assertTrue(self.page.create_coarse_planning_version());self.wait()
        self.assertTrue(snapshots)
        self.assertEqual(old_model.store.regions[0],old_region)
        before=json.loads(old_bytes);after=json.loads(old_journal.path.read_bytes())
        before.pop('created_at');after.pop('created_at')
        self.assertEqual(before,after)
        self.assertEqual(self.page.source,old_source)
        self.assertIs(self.page.model,reduced);self.assertIsNone(self.page.region);self.assertIsNone(self.page.plan)
        self.assertFalse(self.page.canvas.work.any())
        self.assertIsNot(self.page.journal,old_journal)
        self.assertIn('粗分区',self.page.resolution_label.text())

    def test_failure_keeps_existing_model_region_and_reference_mode(self):
        self.case.mark();model=self.page.model;region=deepcopy(self.page.region)
        with patch('femto.app.qt_planning_resolution.create_reduced_model',side_effect=ValueError('coarse failed')):
            self.assertTrue(self.page.create_coarse_planning_version());self.wait()
        self.assertIs(self.page.model,model);self.assertEqual(self.page.region,region)
        self.assertIsNone(derivation(model))
        self.assertIn('coarse failed',self.page._error)

    def test_mode_change_cannot_reinterpret_a_selected_source(self):
        self.case.mark();region=deepcopy(self.page.region)
        with patch.object(self.page,'create_reduced_planning_model') as create:
            self.page.planning_mode.setCurrentIndex(0)
        create.assert_not_called()
        self.assertEqual(self.page.region,region)
        self.assertEqual(self.page.planning_mode.currentIndex(),1)

    def test_cancelled_new_version_keeps_the_old_selection(self):
        self.case.mark();model=self.page.model;region=deepcopy(self.page.region)
        entered,release=threading.Event(),threading.Event()
        def create(*args,**kwargs):
            entered.set()
            if not release.wait(5):raise RuntimeError('test did not release worker')
            return self.reduced()
        with patch('femto.app.qt_planning_resolution.create_reduced_model',side_effect=create):
            self.assertTrue(self.page.create_coarse_planning_version());self.assertTrue(entered.wait(2))
            self.page.runner.cancel();release.set();self.wait()
        self.assertIs(self.page.model,model);self.assertEqual(self.page.region,region)
        self.assertIsNone(derivation(model))

    def test_load_source_defaults_to_coarse_but_restores_legacy_modes(self):
        reduced=self.reduced()
        with patch('femto.app.qt_planning.load_source',return_value=self.camera_sized()), \
             patch('femto.app.qt_planning_resolution.create_reduced_model',return_value=reduced):
            self.assertTrue(self.page.load_source(self.model.reconstruction,force=True,restore_draft=False))
            self.wait()
        self.assertIs(self.page.model,reduced)
        self.assertEqual(self.page.planning_mode.currentIndex(),0)
        self.assertFalse(self.page._error)

    def test_small_reference_remains_standard_without_automatic_sampling(self):
        with patch('femto.app.qt_planning_resolution.create_reduced_model') as create:
            self.assertIs(self.prepare(self.model),self.model);create.assert_not_called()

    def test_blank_coarse_reference_can_reload_the_real_standard_source(self):
        reduced=self.reduced()
        self.page.install_model(reduced,self.model.reconstruction)
        with patch('femto.app.qt_planning.load_source',return_value=self.model), \
             patch('femto.app.qt_planning_resolution.create_reduced_model') as create:
            self.page.planning_mode.setCurrentIndex(1);self.wait()
        self.assertIs(self.page.model,self.model);self.assertFalse(self.page._new_source_coarse)
        self.assertEqual(self.page.planning_mode.currentIndex(),1);create.assert_not_called()

    def test_uncommitted_brush_history_blocks_mode_switch_even_after_erasing(self):
        self.page.canvas._undo.append(('old brush record',))
        self.page.refresh()
        self.assertFalse(self.page.planning_mode.isEnabled())
        with patch.object(self.page,'create_reduced_planning_model') as create:
            self.page.planning_mode.setCurrentIndex(0)
        create.assert_not_called();self.assertIsNone(derivation(self.page.model))


if __name__=='__main__':unittest.main()
