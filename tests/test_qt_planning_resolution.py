"""Explicit reduced-model UI transitions; the geometry backend is mocked."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from copy import deepcopy
from pathlib import Path
import threading
import unittest
from unittest.mock import patch

from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QMessageBox

from femto.app.qt_planning_io import clone_model
from femto.app.qt_planning_resolution import can_create, derivation, journal_for, DERIVATION_SCHEMA
import test_qt_planning as planning_fixture


class PlanningResolutionUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        planning_fixture.QtPlanningTests.setUpClass()

    def setUp(self):
        self.fixture = planning_fixture.QtPlanningTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.page = self.fixture.page
        self.model = self.fixture.model
        self.root = Path(self.fixture.temp.name)
        for name, value in (
            ('femto.app.qt_planning.OUTPUTS_ROOT', self.root / 'outputs'),
            ('femto.planning.planning_drafts.ROOT', self.root / 'drafts'),
        ):
            change = patch(name, value)
            change.start()
            self.addCleanup(change.stop)

    def reduced_model(self):
        model = clone_model(self.model, freeze_inputs=True)
        model.view_reference['derivation'] = dict(schema=DERIVATION_SCHEMA, profile='stride4-mesh50-v1')
        return model

    def wait_worker(self):
        loop = QEventLoop()
        deadline = QTimer()
        deadline.setSingleShot(True)
        deadline.timeout.connect(loop.quit)
        self.page.runner.finished.connect(loop.quit)
        try:
            deadline.start(5000)
            if self.page.busy:
                loop.exec()
        finally:
            deadline.stop()
            self.page.runner.finished.disconnect(loop.quit)
        self.assertFalse(self.page.busy, 'Reduced-model UI worker did not finish')

    def test_only_blank_standard_model_can_be_reduced(self):
        self.assertTrue(can_create(self.page))
        self.assertTrue(self.page.reduce_model_button.isEnabled())
        for attr, value in (('region', {}), ('plan', {})):
            setattr(self.page, attr, value)
            self.assertFalse(can_create(self.page), attr)
            setattr(self.page, attr, None)
        for name in ('work', 'forbidden'):
            getattr(self.page.canvas, name)[0, 0] = 1
            self.assertFalse(can_create(self.page), name)
            getattr(self.page.canvas, name)[0, 0] = 0
        for name in ('points', 'regions', '_history'):
            getattr(self.model.store, name).append({})
            self.assertFalse(can_create(self.page), name)
            getattr(self.model.store, name).clear()
        for name in ('_undo', '_redo'):
            getattr(self.page.canvas, name).append(())
            self.assertFalse(can_create(self.page), name)
            getattr(self.page.canvas, name).clear()
        self.page.runner.busy = True
        self.assertFalse(can_create(self.page))
        self.page.runner.busy = False
        self.model.view_reference['derivation'] = dict(schema=DERIVATION_SCHEMA)
        self.assertFalse(can_create(self.page))
        self.model.view_reference['derivation'] = dict(schema='unrelated-reference-v1')
        self.assertTrue(can_create(self.page))

    def test_success_keeps_source_and_isolates_draft_profile(self):
        source = self.page.source
        previous_journal = self.page.journal
        previous_journal.path.parent.mkdir(parents=True, exist_ok=True)
        previous_journal.path.write_text('original draft', encoding='utf-8')
        original_reference = deepcopy(self.model.view_reference)
        reduced = self.reduced_model()
        self.page.head.fields['row_spacing_mm'].setText('7')
        self.page.dirty = False
        self.page.draft_timer.stop()
        with patch('femto.app.qt_planning_resolution.create_reduced_model', return_value=reduced) as create:
            self.assertTrue(self.page.create_reduced_planning_model())
            self.assertFalse(self.page.reduce_model_button.isEnabled())
            self.wait_worker()
        self.assertFalse(self.page._error, self.page._error)
        self.assertIs(self.page.model, reduced)
        self.assertEqual(self.page.source, source)
        self.assertEqual(self.model.view_reference, original_reference)
        self.assertEqual(self.page.head.record()['row_spacing_mm'], 7)
        self.assertEqual(previous_journal.path.read_text(encoding='utf-8'), 'original draft')
        self.assertNotEqual(self.page.journal.path, previous_journal.path)
        self.assertEqual(self.page.journal.path, journal_for(source, reduced).path)
        self.assertEqual(self.page._stage, 0)
        self.assertIsNone(self.page.region)
        self.assertIsNone(self.page.plan)
        self.assertIsNone(self.page.archive)
        self.assertFalse(self.page.reduce_model_button.isEnabled())
        self.assertIn('简化', self.page.resolution_label.text())
        self.assertIn('保留原三维网格和原始数据', self.page.status_label.text())
        self.assertIsNot(create.call_args.args[0], self.model)
        self.assertEqual(create.call_args.args[1].parent, self.root / 'outputs' / 'planning_models')
        self.assertFalse(create.call_args.args[1].exists())

    def test_backend_failure_retains_current_model_source_and_journal(self):
        journal = self.page.journal
        source = self.page.source
        reference = deepcopy(self.model.view_reference)
        with patch('femto.app.qt_planning_resolution.create_reduced_model', side_effect=ValueError('synthetic reduction failure')):
            self.assertTrue(self.page.create_reduced_planning_model())
            self.wait_worker()
        self.assertIs(self.page.model, self.model)
        self.assertIs(self.page.journal, journal)
        self.assertEqual(self.page.source, source)
        self.assertEqual(self.model.view_reference, reference)
        self.assertIn('synthetic reduction failure', self.page._error)
        self.assertTrue(self.page.reduce_model_button.isEnabled())

    def test_cancelled_success_is_not_installed(self):
        entered, release = threading.Event(), threading.Event()
        reduced = self.reduced_model()
        def create(*args, **kwargs):
            entered.set()
            if not release.wait(3):
                raise RuntimeError('Test did not release worker')
            return reduced
        with patch('femto.app.qt_planning_resolution.create_reduced_model', side_effect=create):
            self.assertTrue(self.page.create_reduced_planning_model())
            self.assertTrue(entered.wait(2))
            self.page.runner.cancel()
            release.set()
            self.wait_worker()
        self.assertIs(self.page.model, self.model)
        self.assertIsNone(derivation(self.page.model))

    def test_saved_reduced_model_is_recognized_without_reduction(self):
        reduced = self.reduced_model()
        source = self.root / 'saved.region.zip'
        journal = journal_for(source, reduced)
        with patch('femto.app.qt_planning_resolution.create_reduced_model') as create:
            self.page.install_model(reduced, source, journal)
            self.assertFalse(self.page.create_reduced_planning_model())
            create.assert_not_called()
        self.assertIn('简化', self.page.resolution_label.text())
        self.assertIn('保留原三维网格和原始数据', self.page.source_label.text())
        self.assertFalse(self.page.reduce_model_button.isEnabled())
        self.assertEqual(journal_for(source, reduced, journal.path.parent).path, journal.path)

    def test_recreating_the_same_profile_offers_its_existing_draft(self):
        source = self.page.source
        standard_journal = self.page.journal
        reduced = self.reduced_model()
        reduced_journal = journal_for(source, reduced)
        self.page.install_model(reduced, source, reduced_journal)
        self.page.canvas.work[40:50, 40:50] = 1
        self.page._selection_changed()
        self.page.checkpoint()
        reduced_journal.wait_saved()
        saved_bytes = reduced_journal.path.read_bytes()
        self.page.draft_timer.stop()
        self.page.install_model(self.model, source, standard_journal)
        with patch('femto.app.qt_planning_resolution.create_reduced_model', return_value=reduced), \
                patch('PySide6.QtWidgets.QMessageBox.question', return_value=QMessageBox.Yes) as question:
            self.assertTrue(self.page.create_reduced_planning_model())
            self.wait_worker()
        question.assert_called_once()
        self.assertEqual(self.page.source, source)
        self.assertEqual(self.page.journal.path, reduced_journal.path)
        self.assertTrue(self.page.canvas.work[40:50, 40:50].all())
        self.assertTrue(self.page.selection_dirty)
        self.assertTrue(self.page.dirty)
        self.assertIsNone(self.page.plan)
        self.assertEqual(reduced_journal.path.read_bytes(), saved_bytes)


if __name__ == '__main__':
    unittest.main()
