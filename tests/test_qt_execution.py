"""UI drives the real simulation coordinator/storage; offscreen, no hardware."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import threading
import unittest
from unittest.mock import patch

from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QApplication, QMessageBox

from femto.app.qt_execution import ExecutionPage, SimulationOwner
from femto.app.qt_main import WorkstationWindow
from femto.execution.history import ExecutionStore
from femto.execution.models import synthetic_plan
from femto.hardware.configuration import DeviceConfiguration
from femto.hardware.interface_contract import DeviceKind
from femto.hardware.simulation.scenario import Scenario
from femto.io_utils import write_json_strict


class ExecutionUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QApplication.instance() or QApplication([])

    def wait(self, predicate, seconds=10):
        deadline=time.monotonic()+seconds
        while time.monotonic()<deadline:
            self.app.processEvents()
            if predicate():return
            time.sleep(.01)
        pages=[w for w in self.app.allWidgets() if isinstance(w,ExecutionPage)]
        self.fail('UI operation did not finish within bounded test timeout: '+str([
            (p.status.text(),(p._snapshot or {}).get('state'),p.owner.busy,
             [(e['kind'],e['payload'].get('reason')) for e in (p._snapshot or {}).get('events',[])[-8:]]) for p in pages]))

    def close_page(self,page):
        page.shutdown()
        self.wait(lambda:not page.owner.alive)
        page.close();self.app.processEvents()

    def test_construction_creates_no_database_or_owner_thread(self):
        with TemporaryDirectory() as root:
            page=ExecutionPage(outputs=root)
            try:
                self.assertFalse(ExecutionStore(root).path.exists())
                self.assertFalse(page.owner.alive)
                self.assertFalse(page.start_button.isEnabled())
                self.assertIn('无真实运动及超声输出',page.banner.text())
            finally:self.close_page(page)

    def test_idle_heartbeats_refresh_display_without_creating_events_or_database(self):
        with TemporaryDirectory() as root:
            page = ExecutionPage(outputs=root)
            try:
                page.connect_button.click()
                self.wait(lambda: page._snapshot and not page.busy)
                initial = page._snapshot['devices'][0]['status']['timestamp_s']
                self.wait(lambda: page._snapshot['devices'][0]['status']['timestamp_s'] > initial+.2, seconds=3)
                self.assertIsNone(page._snapshot['run_id'])
                self.assertEqual(page._snapshot['events'], [])
                self.assertFalse(ExecutionStore(root).path.exists())
                current = page._snapshot['devices'][0]['status']['timestamp_s']
                self.assertIn(f'{current:.3f} s', page.devices.toPlainText())
            finally: self.close_page(page)

    def test_synthetic_ui_run_then_new_page_read_only_replay(self):
        with TemporaryDirectory() as root:
            page=ExecutionPage(outputs=root)
            page.show()
            try:
                page.mode.setCurrentIndex(1);page.freeze_button.click()
                self.wait(lambda:page._snapshot and page._snapshot['run_id'])
                self.assertEqual(page._snapshot['state'],'unprepared')
                page.connect_button.click();self.wait(lambda:not page.busy and all(d['status']['connected'] for d in page._snapshot['devices']))
                page.check_button.click();self.wait(lambda:page._snapshot['state']=='ready_for_simulation')
                self.assertTrue(page.start_button.isEnabled())
                page.start_button.click();self.wait(lambda:page._snapshot['state']=='running')
                page.pause_button.click();self.wait(lambda:page._snapshot['state']=='pause_confirmed')
                self.assertTrue(page.resume_button.isEnabled())
                page.resume_button.click();self.wait(lambda:page._snapshot['state']=='succeeded')
                run=page._snapshot['run_id']
                self.assertEqual(page._snapshot['completed_steps'],4)
                self.assertFalse(page._snapshot['execution_ready'])
                self.assertIn('全部软件步骤已确认', page.status.text())
                self.assertIn('全部停止及输出关闭已确认',page.events.toPlainText())
                self.assertEqual(page.events.textCursor().position(),len(page.events.toPlainText()))
            finally:self.close_page(page)
            replay=ExecutionPage(outputs=root)
            try:
                replay.history_refresh.click();self.wait(lambda:replay.history_records.count()==1)
                replay.replay_button.click();self.wait(lambda:replay._replay)
                self.assertIn(run,replay.replay_text.toPlainText())
                self.assertFalse(replay.start_button.isEnabled())
                self.assertIsNone(replay.owner.coordinator)
            finally:self.close_page(replay)

    def test_failed_real_plan_does_not_fallback_to_synthetic(self):
        with TemporaryDirectory() as root:
            page=ExecutionPage(outputs=root)
            try:
                page.path.setText(str(Path(root)/'missing.region.zip'));page.freeze()
                self.wait(lambda:'操作失败' in page.status.text())
                self.assertFalse(ExecutionStore(root).path.exists())
                self.assertFalse(page.start_button.isEnabled())
                self.assertEqual(page.mode.currentIndex(),0)
            finally:self.close_page(page)

    def test_window_nav_and_close_chain_include_simulation(self):
        window=WorkstationWindow(connect_camera=False)
        try:
            self.assertTrue(window.show_devices())
            self.assertIs(window.pages.currentWidget(),window.execution)
            self.assertIn('不可执行',window.execution_badge.text())
            self.assertEqual(window.devices_button.property('state'),'active')
            with patch.object(window.execution,'can_leave',return_value=False),patch.object(QMessageBox,'information'):
                self.assertFalse(window.new_task())
                self.assertFalse(window.show_capture())
                self.assertFalse(window.open_review('missing'))
                self.assertFalse(window._capture_body_record(type('Dialog',(),{'status':window.subtitle})(),{},{}))
            event=QCloseEvent()
            with patch.object(window.execution,'shutdown',return_value=False) as shutdown:
                window.closeEvent(event)
            self.assertTrue(shutdown.called);self.assertFalse(event.isAccepted())
            self.assertTrue(window._close_timer.isActive())
        finally:
            window._close_timer.stop();window._close_pending=False
            window.capture.shutdown();window.planning.shutdown();window.review.shutdown()
            self.close_page(window.execution);window.close();self.app.processEvents()

    def test_cancel_loading_and_duplicate_freeze_do_not_create_a_task(self):
        with TemporaryDirectory() as root:
            page=ExecutionPage(outputs=root);entered=threading.Event();released=threading.Event()
            def slow(path,*,cancel,progress):
                entered.set();released.wait(3)
                if cancel.is_set():raise RuntimeError('cancelled')
                return synthetic_plan()
            try:
                page.path.setText('selected.region.zip')
                with patch('femto.app.qt_execution.load_saved_plan',side_effect=slow):
                    page.freeze();self.wait(entered.is_set)
                    self.assertFalse(page.freeze())
                    page.owner.cancel_loading();released.set()
                    self.wait(lambda:not page.busy and '取消' in page.status.text() or 'cancelled' in page.status.text())
                self.assertFalse(ExecutionStore(root).path.exists())
                self.assertFalse(page.start_button.isEnabled())
            finally:
                released.set();self.close_page(page)

    def test_unknown_stop_shutdown_finishes_with_record_and_without_new_commands(self):
        with TemporaryDirectory() as root:
            scenario=Scenario(confirm_stop=False,shutdown_timeout_s=.4)
            scenario_path=Path(root)/'scenario.json';write_json_strict(scenario_path,scenario.to_dict())
            page=ExecutionPage(outputs=root)
            try:
                page.command('scenario',str(scenario_path));self.wait(lambda:not page.busy and '配置已读取' in page.status.text())
                self.assertFalse(page._installation_details['scenario']['confirm_stop'])
                self.assertIn('停止完成未确认',page.details_text.toPlainText())
                page.mode.setCurrentIndex(1);page.freeze();self.wait(lambda:page._snapshot and page._snapshot['run_id'])
                page.command('connect');self.wait(lambda:not page.busy)
                page.command('check');self.wait(lambda:page._snapshot['can_start'])
                page.command('start');self.wait(lambda:page._snapshot['state']=='running')
                run=page._snapshot['run_id'];page.shutdown()
                self.assertFalse(page.command('start'))
                self.wait(lambda:not page.owner.alive)
                record=ExecutionStore(root).read(run)
                self.assertEqual(record['display_state'],'result_unknown')
                self.assertIn('simulation_instances_closed',[e['kind'] for e in record['events']])
            finally:self.close_page(page)

    def test_overflow_discards_queued_start_and_preserves_unknown_result(self):
        with TemporaryDirectory() as root:
            page=ExecutionPage(outputs=root)
            try:
                page.mode.setCurrentIndex(1);page.freeze();self.wait(lambda:page._snapshot and page._snapshot['run_id'])
                page.command('connect');self.wait(lambda:not page.busy)
                page.command('check');self.wait(lambda:page._snapshot['can_start'])
                page.owner._overflow.set();page.owner._wake.set()
                self.wait(lambda:page._snapshot['state'] in ('failed','result_unknown'))
                self.assertFalse(page.start_button.isEnabled())
                self.assertEqual(page._snapshot['completed_steps'],0)
                self.assertTrue(any(e['kind']=='critical_queue_gap' for e in page._snapshot['events']))
            finally:self.close_page(page)

    def test_delayed_replay_cannot_replace_new_selection_or_activate_controls(self):
        with TemporaryDirectory() as root:
            page=ExecutionPage(outputs=root)
            try:
                page.owner._replay_revision=2
                record=dict(view_revision=1,run_id='old',display_state='succeeded',source='simulation')
                page._show_replay(json.dumps(record))
                self.assertFalse(page._replay)
                record.update(view_revision=2,run_id='new')
                page._show_replay(json.dumps(record))
                self.assertTrue(page._replay);self.assertEqual(page._selected_run,'new')
                self.assertFalse(page.start_button.isEnabled());self.assertIsNone(page.owner.coordinator)
            finally:self.close_page(page)

    def test_configuration_save_keeps_multiple_device_revisions_separate(self):
        with TemporaryDirectory() as root:
            page=ExecutionPage(outputs=root)
            try:
                page.command('connect');self.wait(lambda:page._snapshot and not page.busy)
                page.command('save_config');self.wait(lambda:'已按修订保存' in page.status.text())
                self.assertTrue((Path(root)/'hardware_configuration/simulation-arm/1.json').is_file())
                self.assertTrue((Path(root)/'hardware_configuration/simulation-head/1.json').is_file())
            finally:self.close_page(page)

    def test_real_configuration_is_visible_but_never_falls_back_to_simulation(self):
        with TemporaryDirectory() as root:
            path=Path(root)/'real-config.json'
            write_json_strict(path,DeviceConfiguration('vendor-arm',DeviceKind.ARM,'real','arm','1').to_dict())
            page=ExecutionPage(outputs=root)
            try:
                page.command('configuration',str(path));self.wait(lambda:page._installation_details and
                    page._installation_details.get('configurations') and not page.busy)
                self.assertEqual(page._installation_details['configurations'][0]['adapter'],'real')
                self.assertIn('真实设备',page.details_text.toPlainText())
                page.command('connect');self.wait(lambda:'不会回退模拟' in page.status.text())
                self.assertIsNone(page.owner.coordinator)
                self.assertFalse(page.start_button.isEnabled())
                self.assertFalse(ExecutionStore(root).path.exists())
            finally:self.close_page(page)

    def test_successive_replay_switch_reset_and_new_load_keep_selection_isolated(self):
        with TemporaryDirectory() as root:
            page=ExecutionPage(outputs=root);runs=[]
            try:
                for _ in range(2):
                    page.mode.setCurrentIndex(1);page.freeze();self.wait(lambda:page._snapshot and page._snapshot['run_id'])
                    page.command('connect');self.wait(lambda:not page.busy)
                    page.command('check');self.wait(lambda:page._snapshot['can_start'])
                    page.command('start');self.wait(lambda:page._snapshot['state']=='succeeded')
                    runs.append(page._snapshot['run_id']);page.command('reset');self.wait(lambda:page._snapshot is None and not page.busy)
                    self.assertIsNone(page._selected_run);self.assertFalse(page.export_button.isEnabled())
                page.command('history');self.wait(lambda:page.history_records.count()==2)
                for run in runs:
                    page.command('replay',run);self.wait(lambda:page._replay and page._selected_run==run)
                    self.assertIn(run,page.replay_text.toPlainText());self.assertFalse(page.start_button.isEnabled())
                    self.assertIsNone(page.owner.coordinator)
                page.command('reset');self.wait(lambda:page._selected_run is None and not page.busy)
                page.freeze();self.wait(lambda:page._snapshot and page._snapshot['run_id'] not in runs)
                self.assertFalse(page._replay)
            finally:self.close_page(page)

    def test_cleanup_failure_still_reports_finished_and_preserves_unknown_record(self):
        with TemporaryDirectory() as root:
            page=ExecutionPage(outputs=root);finished=[]
            page.owner.finished.connect(lambda:finished.append(True))
            try:
                page.mode.setCurrentIndex(1);page.freeze();self.wait(lambda:page._snapshot and page._snapshot['run_id'])
                page.command('connect');self.wait(lambda:not page.busy)
                page.command('check');self.wait(lambda:page._snapshot['can_start'])
                coordinator=page.owner.coordinator;run=coordinator.run_id
                with patch.object(coordinator.by_id['simulation-arm'],'close',side_effect=OSError('scripted cleanup failure')):
                    page.shutdown();self.wait(lambda:not page.owner.alive and finished)
                self.assertTrue(page.owner._cleanup_failed)
                self.assertEqual(ExecutionStore(root).read(run)['display_state'],'result_unknown')
                self.assertFalse(page.owner.submit('start'))
                coordinator.close_instances()
            finally:self.close_page(page)


if __name__=='__main__':unittest.main()
