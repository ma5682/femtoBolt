"""Behavior regressions for simulation guidance, recovery and view context."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import time
import unittest
from unittest.mock import patch

from PySide6.QtWidgets import QApplication, QMessageBox

from femto.app.qt_execution import ExecutionPage
from femto.execution.history import ExecutionStore
from femto.execution.models import synthetic_plan
from femto.execution.state_machine import RunState
from femto.hardware.simulation.scenario import Scenario
from femto.io_utils import write_json_strict


class ExecutionWorkflowTests(unittest.TestCase):
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
        self.fail('Simulation UI did not reach the expected state within the bounded timeout')

    def close_page(self, page):
        self.wait(page.shutdown)
        page.close()
        self.app.processEvents()

    def freeze(self, page):
        page.mode.setCurrentIndex(1)
        page.freeze_button.click()
        self.wait(lambda: page._snapshot and page._snapshot['run_id'] and not page.busy)
        return page._snapshot['run_id']

    def ready(self, page):
        run = self.freeze(page)
        page.connect_button.click()
        self.wait(lambda: not page.busy and all(d['status']['connected'] for d in page._snapshot['devices']))
        page.check_button.click()
        self.wait(lambda: page._snapshot['state'] == 'ready_for_simulation' and not page.busy)
        return run

    def history_fixture(self, root):
        store = ExecutionStore(root)
        plan = synthetic_plan()
        store.create('old-history-run', dict(source='simulation', plan=plan.to_dict(),
            execution_ready=False, physical_accuracy_verified=False))
        store.append('old-history-run', 'fixture', 'test_fixture', {}, RunState.SUCCEEDED, len(plan.steps))
        return store

    def test_connection_must_precede_check_and_keeps_the_frozen_task(self):
        with TemporaryDirectory() as root:
            page = ExecutionPage(outputs=root)
            try:
                run = self.freeze(page)
                self.assertFalse(page.check_button.isEnabled())
                self.assertIn('请先连接', page.check_button.toolTip())
                self.assertIn('连接模拟设备', page.status.text())
                page.check_button.click()
                self.assertFalse(page.command('check'))
                self.assertEqual(page._snapshot['state'], 'unprepared')
                self.assertNotIn('preflight_started', [e['kind'] for e in ExecutionStore(root).read(run)['events']])
                page.connect_button.click()
                self.wait(lambda: page.check_button.isEnabled() and not page.busy)
                self.assertIn('下一步：模拟前置检查', page.status.text())
                page.check_button.click()
                self.wait(lambda: page._snapshot['state'] == 'ready_for_simulation' and not page.busy)
                self.assertEqual(page._snapshot['run_id'], run)
                self.assertEqual(len(ExecutionStore(root).runs()), 1)
            finally:
                self.close_page(page)

    def test_running_and_paused_tasks_use_stop_before_closing_instances(self):
        with TemporaryDirectory() as root:
            page = ExecutionPage(outputs=root)
            try:
                run = self.ready(page)
                self.assertFalse(page.reset_button.isEnabled())
                page.start_button.click()
                self.wait(lambda: page._snapshot['state'] == 'running' and not page.busy)
                with patch.object(QMessageBox, 'question') as question:
                    page.reset_button.click()
                    self.assertFalse(page.command('reset'))
                    question.assert_not_called()
                self.assertEqual(page._snapshot['run_id'], run)
                page.pause_button.click()
                self.wait(lambda: page._snapshot['state'] == 'pause_confirmed' and not page.busy)
                self.assertFalse(page.reset_button.isEnabled())
                self.assertIn('请求停止', page.reset_button.toolTip())
                self.assertTrue(page.owner.submit('reset'))
                self.wait(lambda: not page.busy and '请先请求停止' in page.status.text())
                self.assertEqual(page._snapshot['state'], 'pause_confirmed')
                self.assertEqual(page._snapshot['run_id'], run)
                page.stop_button.click()
                self.wait(lambda: page._snapshot['state'] == 'stop_requested')
                self.assertFalse(page.reset_button.isEnabled())
                self.wait(lambda: page._snapshot['state'] == 'stopped' and not page.busy)
                with patch.object(QMessageBox, 'question', return_value=QMessageBox.Yes) as question:
                    page.reset_button.click()
                    self.wait(lambda: page._snapshot is None and not page.busy)
                    question.assert_called_once()
                record = ExecutionStore(root).read(run)
                self.assertEqual(record['display_state'], 'stopped')
                self.assertNotIn('stop_result_unknown', [e['kind'] for e in record['events']])
            finally:
                self.close_page(page)

    def test_unknown_recovery_requires_confirmation_and_preserves_old_result(self):
        with TemporaryDirectory() as root:
            page = ExecutionPage(outputs=root)
            scenario_path = Path(root)/'unconfirmed.json'
            write_json_strict(scenario_path, Scenario(complete_delay_s=.08, request_timeout_s=.4,
                                                     confirm_stop=False).to_dict())
            try:
                page.command('scenario', str(scenario_path))
                self.wait(lambda: not page.busy)
                run = self.ready(page)
                page.start_button.click()
                self.wait(lambda: page._snapshot['state'] == 'running' and not page.busy)
                page.stop_button.click()
                self.wait(lambda: page._snapshot['state'] == 'result_unknown' and not page.busy)
                self.assertTrue(page._snapshot['resource_held'])
                self.assertTrue(page.reset_button.isEnabled())
                with patch.object(QMessageBox, 'question', return_value=QMessageBox.No):
                    page.reset_button.click()
                self.assertEqual(page._snapshot['run_id'], run)
                self.assertTrue(page._snapshot['resource_held'])
                with patch.object(QMessageBox, 'question', return_value=QMessageBox.Yes) as question:
                    page.reset_button.click()
                    self.wait(lambda: page._snapshot is None and not page.busy)
                    self.assertIn('未知结果不会改为已停止', question.call_args.args[2])
                self.assertEqual(ExecutionStore(root).read(run)['display_state'], 'result_unknown')
                self.assertNotEqual(self.freeze(page), run)
            finally:
                self.close_page(page)

    def test_history_connect_and_freeze_return_to_the_identified_live_task(self):
        with TemporaryDirectory() as root:
            store = self.history_fixture(root)
            before = store.read('old-history-run')
            page = ExecutionPage(outputs=root)
            page.resize(1120, 450)
            page.show()
            try:
                self.app.processEvents()
                scroll = page.tabs.widget(0).verticalScrollBar()
                self.assertGreater(scroll.maximum(), 0)
                scroll.setValue(scroll.maximum())
                page.history_refresh.click()
                self.wait(lambda: page.history_records.count() == 1 and not page.busy)
                page.replay_button.click()
                self.wait(lambda: page._replay and not page.busy)
                self.assertEqual(page.tabs.currentIndex(), 1)
                page.connect_button.click()
                self.wait(lambda: page._snapshot and not page.busy)
                self.assertFalse(page._replay)
                self.assertEqual(page.tabs.currentIndex(), 0)
                self.assertEqual(scroll.value(), 0)
                scroll.setValue(scroll.maximum())
                page.tabs.setCurrentIndex(1)
                run = self.freeze(page)
                self.assertEqual(page.tabs.currentIndex(), 0)
                self.assertEqual(scroll.value(), 0)
                self.assertNotEqual(run, 'old-history-run')
                self.assertIn(run[:12], page.state.text())
                self.assertIn(run, page.state.toolTip())
                self.assertEqual(store.read('old-history-run'), before)
            finally:
                self.close_page(page)

    def test_new_connection_invalidates_a_late_history_read(self):
        with TemporaryDirectory() as root:
            self.history_fixture(root)
            page = ExecutionPage(outputs=root)
            entered, released = threading.Event(), threading.Event()
            original = ExecutionStore.read
            def slow_read(store, run, *, verify_sources=True, check=lambda: None):
                entered.set()
                while not released.wait(.005):
                    check()
                return original(store, run, verify_sources=verify_sources, check=check)
            try:
                with patch.object(ExecutionStore, 'read', slow_read):
                    page.command('replay', 'old-history-run')
                    self.wait(entered.is_set)
                    self.assertTrue(page.command('connect'))
                    released.set()
                    self.wait(lambda: not page.busy and page._snapshot)
                self.assertFalse(page._replay)
                self.assertEqual(page.tabs.currentIndex(), 0)
                self.assertNotIn('old-history-run', page.replay_text.toPlainText())
                self.assertTrue(all(d['status']['connected'] for d in page._snapshot['devices']))
            finally:
                released.set()
                self.close_page(page)

    def test_heartbeat_keeps_device_scroll_and_selection_while_values_update(self):
        with TemporaryDirectory() as root:
            page = ExecutionPage(outputs=root)
            page.resize(1120, 620)
            page.show()
            try:
                page.connect_button.click()
                self.wait(lambda: page._snapshot and not page.busy)
                browser = page.devices
                cursor = browser.document().find('软件模拟治疗头')
                self.assertFalse(cursor.isNull())
                browser.setTextCursor(cursor)
                bar = browser.verticalScrollBar()
                self.assertGreater(bar.maximum(), 0)
                bar.setValue(bar.maximum())
                position = bar.value()
                observed = page._snapshot['devices'][0]['status']['timestamp_s']
                self.wait(lambda: page._snapshot['devices'][0]['status']['timestamp_s'] > observed+.2)
                self.assertEqual(bar.value(), position)
                self.assertEqual(browser.textCursor().selectedText(), '软件模拟治疗头')
                current = page._snapshot['devices'][0]['status']['timestamp_s']
                self.assertIn(f'{current:.3f} s', browser.toPlainText())
                devices = json.loads(json.dumps(page._snapshot['devices']))
                devices[0]['status']['timestamp_s'] = 10000.25
                browser.update_devices(devices)
                self.assertEqual(browser.textCursor().selectedText(), '软件模拟治疗头')
                self.assertEqual(bar.value(), position)
                self.assertIn('10000.250 s', browser.toPlainText())
            finally:
                self.close_page(page)

    def test_phase_guidance_updates_and_completion_does_not_erase_export_notice(self):
        with TemporaryDirectory() as root:
            page = ExecutionPage(outputs=root)
            try:
                run = self.ready(page)
                self.assertIn('检查通过', page.status.text())
                self.assertNotIn('连接模拟设备后', page.status.text())
                page.start_button.click()
                self.wait(lambda: page._snapshot['state'] == 'running' and not page.busy)
                self.assertIn('模拟运行中', page.status.text())
                page.pause_button.click()
                self.wait(lambda: page._snapshot['state'] == 'pause_confirmed' and not page.busy)
                self.assertIn('暂停已确认', page.status.text())
                page.resume_button.click()
                self.wait(lambda: page._snapshot['state'] == 'succeeded' and not page.busy)
                self.assertIn('全部软件步骤已确认', page.status.text())
                path = str(Path(root)/'events.json')
                page.command('export', dict(run_id=run, path=path))
                self.wait(lambda: not page.busy and '模拟记录已导出' in page.status.text())
                observed = page._snapshot['devices'][0]['status']['timestamp_s']
                self.wait(lambda: page._snapshot['devices'][0]['status']['timestamp_s'] > observed+.2)
                self.assertIn(path, page.status.text())
            finally:
                self.close_page(page)

    def test_failed_first_load_keeps_the_error_instead_of_initial_guidance(self):
        with TemporaryDirectory() as root:
            page = ExecutionPage(outputs=root)
            try:
                page.path.setText(str(Path(root)/'missing.region.zip'))
                page.preview_button.click()
                self.wait(lambda: '操作失败' in page.status.text() and not page.busy)
                self.assertIn('操作失败', page.status.text())
                self.assertFalse(page.start_button.isEnabled())
                self.assertFalse(ExecutionStore(root).path.exists())
            finally:
                self.close_page(page)


if __name__ == '__main__':
    unittest.main()
