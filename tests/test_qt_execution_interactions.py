"""Regression probes for existing simulation controls and record interactions."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import time
import unittest
from unittest.mock import patch

from PySide6.QtWidgets import QApplication, QFileDialog

from femto.app.qt_execution import ExecutionPage
from femto.execution.application import configurations
from femto.execution.history import ExecutionStore
from femto.execution.models import FrozenPlan, synthetic_plan
from femto.execution.state_machine import RunState
from femto.hardware.configuration import save_configuration
from femto.hardware.simulation.scenario import Scenario


class ExecutionInteractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def wait(self, predicate, seconds=8):
        deadline = time.monotonic()+seconds
        while time.monotonic() < deadline:
            self.app.processEvents()
            if predicate():
                return
            time.sleep(.005)
        self.fail('Simulation interaction did not finish within bounded timeout')

    def close_page(self, page):
        self.wait(page.shutdown)
        page.close()
        self.app.processEvents()

    def ready(self, page):
        page.mode.setCurrentIndex(1)
        page.freeze_button.click()
        self.wait(lambda: page._snapshot and page._snapshot['run_id'] and not page.busy)
        page.connect_button.click()
        self.wait(lambda: not page.busy and all(d['status']['connected'] for d in page._snapshot['devices']))
        page.check_button.click()
        self.wait(lambda: page._snapshot['state'] == 'ready_for_simulation' and not page.busy)

    def records(self, root, count):
        store = ExecutionStore(root)
        plan = synthetic_plan()
        for number in range(count):
            run_id = f'fixture-{number:03d}'
            store.create(run_id, dict(source='simulation', plan=plan.to_dict(),
                execution_ready=False, physical_accuracy_verified=False))
            store.append(run_id, 'fixture', 'test_fixture', {}, RunState.SUCCEEDED, len(plan.steps))
        return store

    def test_export_does_not_block_feedback_or_stop(self):
        with TemporaryDirectory() as root:
            page = ExecutionPage(outputs=root)
            entered, released = threading.Event(), threading.Event()
            original = ExecutionStore.export
            def slow_export(store, run, destination, *, check=lambda: None):
                entered.set()
                if not released.wait(6):
                    raise RuntimeError('Unreleased export probe')
                check()
                return original(store, run, destination, check=check)
            try:
                self.ready(page)
                page.start_button.click()
                self.wait(lambda: page._snapshot['state'] == 'running' and not page.busy)
                run = page._selected_run
                target = str(Path(root)/'events.json')
                with patch.object(ExecutionStore, 'export', slow_export), patch.object(
                        QFileDialog, 'getSaveFileName', return_value=(target, '')):
                    page.export_button.click()
                    self.wait(entered.is_set)
                    page._buttons()
                    self.assertTrue(page.stop_button.isEnabled(), 'Record IO must not disable stop')
                    observed = page._snapshot['devices'][0]['status']['timestamp_s']
                    self.wait(lambda: page._snapshot['devices'][0]['status']['timestamp_s'] > observed, seconds=2)
                    page.stop_button.click()
                    self.wait(lambda: page._snapshot['state'] == 'stopped', seconds=2)
                    self.assertFalse(released.is_set())
                    record = ExecutionStore(root).read(run, verify_sources=False)
                    stops = [e for e in record['events'] if e['kind'] == 'operator_stop']
                    self.assertEqual(len(stops), 1)
                    released.set()
                    self.wait(lambda: not page.busy)
                self.assertEqual(json.loads(Path(target).read_text(encoding='utf-8'))['run_id'], run)
            finally:
                released.set()
                self.close_page(page)

    def test_stop_interrupts_resume_source_verification(self):
        with TemporaryDirectory() as root:
            page = ExecutionPage(outputs=root)
            entered, released = threading.Event(), threading.Event()
            original = FrozenPlan.verify_assets
            def slow_verify(plan, check=None):
                entered.set()
                while not released.wait(.005):
                    if check:
                        check()
                return original(plan, check)
            try:
                self.ready(page)
                page.start_button.click()
                self.wait(lambda: page._snapshot['state'] == 'running' and not page.busy)
                page.pause_button.click()
                self.wait(lambda: page._snapshot['state'] == 'pause_confirmed' and not page.busy)
                with patch.object(FrozenPlan, 'verify_assets', slow_verify):
                    page.resume_button.click()
                    self.wait(entered.is_set)
                    page._buttons()
                    self.assertTrue(page.stop_button.isEnabled(), 'Source verification must remain cancellable by stop')
                    page.stop_button.click()
                    self.wait(lambda: page._snapshot['state'] == 'stopped', seconds=2)
                    self.assertFalse(released.is_set())
                    released.set()
                record = ExecutionStore(root).read(page._selected_run, verify_sources=False)
                requests = [e['payload']['request']['intent'] for e in record['events'] if e['kind'] == 'request_intent']
                self.assertNotIn('request_resume', requests)
            finally:
                released.set()
                self.close_page(page)

    def test_more_history_and_refresh_keep_the_selected_older_record(self):
        with TemporaryDirectory() as root:
            self.records(root, 201)
            page = ExecutionPage(outputs=root)
            try:
                page.history_refresh.click()
                self.wait(lambda: page.history_records.count() == 200 and not page.busy)
                self.assertTrue(page.history_more.isEnabled())
                page.history_more.click()
                self.wait(lambda: page.history_records.count() == 201 and not page.busy)
                self.assertFalse(page.history_more.isEnabled())
                page.history_records.setCurrentIndex(page.history_records.findData('fixture-000'))
                page.history_refresh.click()
                self.wait(lambda: not page.busy)
                self.assertEqual(page.history_records.currentData(), 'fixture-000')
                ids = [page.history_records.itemData(i) for i in range(page.history_records.count())]
                self.assertEqual(len(ids), len(set(ids)))
            finally:
                self.close_page(page)

    def test_export_uses_the_selected_record_instead_of_the_previous_replay(self):
        with TemporaryDirectory() as root:
            self.records(root, 3)
            page = ExecutionPage(outputs=root)
            try:
                page.history_refresh.click()
                self.wait(lambda: page.history_records.count() == 3 and not page.busy)
                page.history_records.setCurrentIndex(page.history_records.findData('fixture-000'))
                page.replay_button.click()
                self.wait(lambda: page._replay and page._selected_run == 'fixture-000' and not page.busy)
                page.history_records.setCurrentIndex(page.history_records.findData('fixture-001'))
                target = str(Path(root)/'selected.json')
                with patch.object(QFileDialog, 'getSaveFileName', return_value=(target, '')):
                    page.export_button.click()
                    self.wait(lambda: not page.busy)
                exported = json.loads(Path(target).read_text(encoding='utf-8'))
                self.assertEqual(exported['run_id'], 'fixture-001')
                page.history_refresh.click()
                self.wait(lambda: not page.busy)
                self.assertEqual(page.history_records.currentData(), 'fixture-001')
            finally:
                self.close_page(page)

    def test_multiple_configuration_files_load_together_and_failure_keeps_current_devices(self):
        with TemporaryDirectory() as root:
            values = [replace(c, revision='2') for c in configurations(Scenario())]
            paths = [str(save_configuration(c, outputs=root)) for c in values]
            page = ExecutionPage(outputs=root)
            try:
                page.command('configuration', paths)
                self.wait(lambda: not page.busy and '配置已读取' in page.status.text())
                self.assertNotIn('操作失败', page.status.text())
                page.connect_button.click()
                self.wait(lambda: page._snapshot and not page.busy)
                devices = page._snapshot['devices']
                self.assertEqual({d['configuration']['device_id'] for d in devices},
                                 {'simulation-arm', 'simulation-head'})
                self.assertTrue(all(d['configuration']['revision'] == '2' for d in devices))
                bindings = [d['status']['binding'] for d in devices]
                page.command('configuration', [paths[0], str(Path(root)/'missing.json')])
                # The worker can clear busy before Qt delivers its queued
                # notice. Wait for the visible result before asserting it.
                self.wait(lambda: not page.busy and '操作失败' in page.status.text())
                self.assertIn('操作失败', page.status.text())
                self.assertEqual([d['status']['binding'] for d in page._snapshot['devices']], bindings)
                self.assertTrue(all(d['status']['connected'] for d in page._snapshot['devices']))
            finally:
                self.close_page(page)

    def test_close_cancels_record_read_and_waits_for_its_worker(self):
        with TemporaryDirectory() as root:
            store = self.records(root, 1)
            page = ExecutionPage(outputs=root)
            entered, cancelled, released = threading.Event(), threading.Event(), threading.Event()
            original = ExecutionStore.read
            def slow_read(store, run, *, verify_sources=True, check=lambda: None):
                entered.set()
                while not released.wait(.005):
                    try:
                        check()
                    except RuntimeError:
                        cancelled.set()
                        raise
                return original(store, run, verify_sources=verify_sources, check=check)
            try:
                with patch.object(ExecutionStore, 'read', slow_read):
                    page.command('replay', 'fixture-000')
                    self.wait(entered.is_set)
                    self.assertFalse(page.shutdown())
                    self.wait(cancelled.is_set)
                    self.wait(page.shutdown)
                    self.assertFalse(page.command('start'))
                    released.set()
                self.assertEqual(store.read('fixture-000')['event_sequence'], 1)
                self.assertFalse(page._replay)
            finally:
                released.set()
                self.close_page(page)

    def test_return_from_replay_refreshes_current_device_status(self):
        with TemporaryDirectory() as root:
            page = ExecutionPage(outputs=root)
            try:
                self.ready(page)
                page.start_button.click()
                self.wait(lambda: page._snapshot['state'] == 'succeeded' and not page.busy)
                run = page._selected_run
                page.command('replay', run)
                self.wait(lambda: page._replay and not page.busy)
                observed = page._snapshot['devices'][0]['status']['timestamp_s']
                self.wait(lambda: json.loads(page.owner.take_snapshot())['devices'][0]['status']['timestamp_s'] > observed+.15)
                page.tabs.setCurrentIndex(0)
                self.assertFalse(page._replay, 'Returning to current devices must leave read-only replay mode')
                self.assertGreater(page._snapshot['devices'][0]['status']['timestamp_s'], observed)
                self.assertEqual(page._selected_run, run)
            finally:
                self.close_page(page)


if __name__ == '__main__':
    unittest.main()
