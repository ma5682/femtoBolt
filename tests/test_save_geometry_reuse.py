"""A real Qt save reuses geometry only inside its own background request."""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from threading import get_ident
import os
import unittest
from unittest.mock import patch

import numpy as np
from PySide6.QtCore import QEventLoop, QTimer

import test_qt_planning as fixture
from femto.geometry import region_reuse
from femto.planning import region_project, surface_workspace, target_surface
from femto.planning.surface_trajectory import to_player_plan
from femto.planning.trajectory_export import load_trajectory


class SaveGeometryReuseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.QtPlanningTests.setUpClass()

    def setUp(self):
        self.case = fixture.QtPlanningTests()
        self.case.setUp()
        self.addCleanup(self.case.doCleanups)
        self.case.mark(split=True)
        self.page = self.case.page
        self.page.head.fields['row_spacing_mm'].setText('7')
        self.page.generate()
        self.case.wait_task()
        self.page.set_stage(2)
        self.page.reviewed.setChecked(True)
        self.main_thread = get_ident()
        self.scopes, self.releases, self.stages, self.sessions = [], [], [], []

    @contextmanager
    def observe(self):
        operation = region_reuse.operation
        native, portable = surface_workspace.export, region_project.save_project

        @contextmanager
        def scoped():
            before = region_reuse._active.get()
            try:
                with operation():
                    self.scopes.append((get_ident(), before, region_reuse._active.get()))
                    yield
            finally:
                self.releases.append((get_ident(), region_reuse._active.get()))

        def export(*args, **kwargs):
            state = region_reuse._active.get()
            session = kwargs.get('target_session')
            self.sessions.append(session)
            self.stages.append(('native', get_ident(), state))
            result = native(*args, **kwargs)
            self.stages.append(('native_published', get_ident(), state, result))
            return result

        def save(*args, **kwargs):
            self.sessions.append(kwargs.get('target_session'))
            self.stages.append(('portable', get_ident(), region_reuse._active.get()))
            return portable(*args, **kwargs)

        with patch.object(region_reuse, 'operation', scoped), \
                patch.object(surface_workspace, 'export', export), \
                patch.object(region_project, 'save_project', save):
            yield

    def finish(self, *, error=False):
        loop, timer = QEventLoop(), QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(loop.quit)
        self.page.runner.finished.connect(loop.quit)
        try:
            timer.start(60000)
            if self.page.busy:
                loop.exec()
        finally:
            timer.stop()
            self.page.runner.finished.disconnect(loop.quit)
        self.assertFalse(self.page.busy, 'Save worker did not finish')
        self.assertEqual(bool(self.page._error), error, self.page.status_label.text())

    def assert_released(self):
        self.assertIsNone(region_reuse._active.get())
        self.assertTrue(self.scopes)
        self.assertEqual(len(self.scopes), len(self.releases))
        for thread, before, state in self.scopes:
            self.assertNotEqual(thread, self.main_thread)
            self.assertIsNone(before)
            self.assertIsInstance(state, dict)
        for thread, after in self.releases:
            self.assertNotEqual(thread, self.main_thread)
            self.assertIsNone(after)
        for session in self.sessions:
            if session is not None:
                self.assertEqual(session._state, 'closed')

    def test_complete_save_shares_worker_scope_and_independent_readers_replay(self):
        expected = deepcopy(self.page.plan)
        expected_target = surface_workspace.target_for(self.page.model, expected, reproduce=False)
        with self.observe(), patch.object(target_surface, '_construct', wraps=target_surface._construct) as construct:
            self.page.save_plan()
            self.finish()
            self.assertEqual(construct.call_count, 1)
            self.assert_released()
            self.assertEqual(len(self.scopes), 1)
            self.assertIs(self.stages[0][2], self.stages[-1][2])
            self.assertTrue(self.scopes[0][2], 'Real save did not prepare any reusable geometry')
            self.assertIs(self.sessions[0], self.sessions[1])
            folder = self.page.archive
            native = load_trajectory(folder, reproduce=True)
            self.assertEqual(construct.call_count, 2)
            portable = region_project.open_project(folder/'planning.region.zip',
                workspace=Path(self.case.temp.name)/'independent_reopen')
            self.assertEqual(construct.call_count, 3)
            self.assertEqual(to_player_plan(native)['segments'], expected['segments'])
            self.assertEqual(portable.saved_plan['segments'], expected['segments'])
            for name, array in expected_target.arrays.items():
                self.assertEqual(portable.target_ledger.arrays[name].dtype, array.dtype)
                np.testing.assert_array_equal(portable.target_ledger.arrays[name], array)
            self.assertIsNone(region_reuse._active.get())
        self.assertEqual(self.page.plan, expected)

    def test_save_as_owns_a_fresh_worker_scope_and_reopens_full_plan(self):
        target = Path(self.case.temp.name)/'save_as.region.zip'
        expected = deepcopy(self.page.plan)
        with self.observe(), patch('femto.app.qt_planning.QFileDialog.getSaveFileName', return_value=(str(target), '')):
            self.page.save_as()
            self.finish()
        self.assert_released()
        self.assertEqual(len(self.scopes), 1)
        self.assertEqual([row[0] for row in self.stages], ['portable'])
        self.assertIs(self.stages[0][2], self.scopes[0][2])
        reopened = region_project.open_project(target, workspace=Path(self.case.temp.name)/'save_as_reopened')
        self.assertEqual(reopened.saved_plan['segments'], expected['segments'])
        self.assertIsNone(region_reuse._active.get())

    def _interrupt_and_retry(self, cancel):
        expected = deepcopy(self.page.plan)
        archive_before = self.page.archive
        original_pack = region_project._pack_assets
        interrupted = []
        def pack(root, check):
            interrupted.append(region_reuse._active.get())
            if cancel:
                self.page.runner.cancel()
                return original_pack(root, check)
            raise OSError('injected archive failure')
        with self.observe():
            with patch.object(region_project, '_pack_assets', pack):
                self.page.save_plan()
                # TaskRunner reports cancellation separately from an error.
                self.finish(error=not cancel)
            self.assertTrue(interrupted)
            self.assert_released()
            first_scope = self.scopes[0][2]
            published = [row[3] for row in self.stages if row[0] == 'native_published']
            self.assertEqual(len(published), 1)
            self.assertTrue((published[0]/'trajectory.json').is_file())
            self.assertFalse((published[0]/'planning.region.zip').exists())
            self.assertEqual(self.page.plan, expected)
            self.assertEqual(self.page.archive, archive_before)
            self.page.save_plan()
            self.finish()
            self.assert_released()
            self.assertEqual(len(self.scopes), 2)
            self.assertIsNot(self.scopes[1][2], first_scope)
            self.assertEqual(self.page.plan, expected)
            self.assertTrue((self.page.archive/'planning.region.zip').is_file())

    def test_archive_failure_releases_geometry_and_retry_uses_a_new_scope(self):
        self._interrupt_and_retry(False)

    def test_cancel_after_native_publish_releases_geometry_and_retry_uses_a_new_scope(self):
        self._interrupt_and_retry(True)

    def test_source_file_change_is_rejected_even_with_populated_geometry_cache(self):
        original_save = region_project.save_project
        tampered = []
        def save(model, *args, **kwargs):
            state = region_reuse._active.get()
            self.assertTrue(state)
            source = Path(model.store.source)
            raw, stat = source.read_bytes(), source.stat()
            try:
                source.write_bytes(raw+b'changed source bytes')
                os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
                tampered.append(True)
                return original_save(model, *args, **kwargs)
            finally:
                source.write_bytes(raw)
                os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        # Wrap the genuine portable path; no source digest or acceptance gate
        # is mocked. Only this test's temporary source file is changed.
        with patch.object(region_project, 'save_project', save), self.observe():
            self.page.save_plan()
            self.finish(error=True)
        self.assertTrue(tampered)
        self.assertIn('输入已变化', self.page.status_label.text())
        self.assert_released()
        published = [row[3] for row in self.stages if row[0] == 'native_published']
        self.assertEqual(len(published), 1)
        self.assertFalse((published[0]/'planning.region.zip').exists())


if __name__ == '__main__':
    unittest.main()
