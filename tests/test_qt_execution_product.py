"""Visible simulation controls, saved-plan handoff and actionable recovery."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')

from dataclasses import replace
from contextlib import contextmanager
import json
from pathlib import Path
import threading
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from femto.app.execution_presentation import plan_text
from femto.app.qt_execution import ExecutionPage, _device_rows
from femto.app.qt_main import WorkstationWindow
from femto.dataset.installation_calibration import RDF_AXES, TransformRecord, save_record
from femto.execution.history import ExecutionStore
from femto.execution.models import AssetReference, canonical, synthetic_plan
from femto.hardware.simulation.scenario import Scenario
from femto.io_utils import sha256_file, write_json_strict


class SimulationProductTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])

    def wait(self,predicate,seconds=8):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            self.app.processEvents()
            if predicate():return
            time.sleep(.005)
        self.fail('Simulation did not reach the expected visible state')

    @contextmanager
    def workspace(self):
        with TemporaryDirectory() as root:
            self._workspace_pages=[]
            try:yield root
            finally:
                for page in reversed(self._workspace_pages):self.close_page(page)

    def page(self,root):
        page=ExecutionPage(outputs=root);page.resize(1120,620);page.show();self.app.processEvents()
        self._workspace_pages.append(page)
        return page

    def close_page(self,page):
        self.wait(page.shutdown);page.close();self.app.processEvents()

    def ready(self,page):
        page.mode.setCurrentIndex(1);page.freeze_button.click()
        self.wait(lambda:page._snapshot and page._snapshot['run_id'] and not page.busy)
        page.connect_button.click()
        self.wait(lambda:page.check_button.isEnabled() and not page.busy)
        page.check_button.click()
        self.wait(lambda:page._snapshot['state']=='ready_for_simulation' and not page.busy)

    def saved_fixture(self,root):
        source=Path(root)/'right-knee.region.zip';source.write_text('fixture archive',encoding='utf-8')
        original=synthetic_plan();data=original.public_plan
        data.update(source_path=str(source),revision_id='knee-saved-7',
            task_context=dict(name='右膝检查任务'))
        return source,replace(original,data_json=canonical(data),assets=(AssetReference(str(source),sha256_file(source)),))

    def test_stop_state_and_progress_stay_visible_in_all_tabs_and_scrolled_view(self):
        with self.workspace() as root:
            page=self.page(root);self.ready(page);page.start_button.click()
            self.wait(lambda:page._snapshot['state']=='running' and not page.busy)
            run=page._snapshot['run_id']
            for index in (0,1,2):
                page.tabs.setCurrentIndex(index);self.app.processEvents()
                if index==0:page.tabs.widget(0).verticalScrollBar().setValue(page.tabs.widget(0).verticalScrollBar().maximum())
                self.assertTrue(page.stop_button.isVisible())
                self.assertTrue(page.state.isVisible())
                self.assertTrue(page.progress.isVisible())
                self.assertTrue(page.stop_button.isEnabled())
                self.assertIn(run[:12],page.state.text())
                self.assertTrue(page.stop_button.visibleRegion().contains(page.stop_button.rect().center()))
            page.stop_button.click()
            self.wait(lambda:page._snapshot['state']=='stopped' and not page.busy)
            self.assertEqual(page.tabs.currentIndex(),0)
            self.assertFalse(page.stop_button.isEnabled())

    def test_primary_action_changes_with_phase_and_abandon_unlocks_plan_selection(self):
        with self.workspace() as root:
            page=self.page(root)
            self.assertEqual(page._primary_action,'preview_plan')
            page.mode.setCurrentIndex(1);page.freeze_button.click()
            self.wait(lambda:page._snapshot and page._snapshot['run_id'] and not page.busy)
            run=page._snapshot['run_id']
            self.assertEqual(page._primary_action,'connect')
            self.assertEqual(page.connect_button.property('role'),'primary')
            self.assertTrue(page.abandon_button.isEnabled())
            page.abandon_button.click()
            self.wait(lambda:page._snapshot['state']=='abandoned_unrun' and not page.busy)
            self.assertTrue(page.mode.isEnabled())
            page.mode.setCurrentIndex(0)
            self.assertTrue(page.path.isEnabled())
            self.assertEqual(ExecutionStore(root).read(run)['display_state'],'abandoned_unrun')
            page.mode.setCurrentIndex(1)
            page.freeze_button.click()
            self.wait(lambda:page._snapshot['run_id']!=run and not page.busy)
            page.connect_button.click();self.wait(lambda:page.check_button.isEnabled() and not page.busy)
            self.assertEqual(page._primary_action,'check')
            page.check_button.click();self.wait(lambda:page._snapshot['state']=='ready_for_simulation' and not page.busy)
            self.assertEqual(page._primary_action,'start')
            self.assertFalse(page.command('abandon'))
            self.assertFalse(page.abandon_button.isVisible())
            page.start_button.click();self.wait(lambda:page._snapshot['state']=='running' and not page.busy)
            page.pause_button.click();self.wait(lambda:page._snapshot['state']=='pause_confirmed' and not page.busy)
            self.assertEqual(page._primary_action,'resume')

    def test_saved_plan_is_verified_before_freezing_and_displays_recorded_context(self):
        with self.workspace() as root:
            source,frozen=self.saved_fixture(root);page=self.page(root)
            with patch('femto.app.qt_execution.load_saved_plan',return_value=frozen) as load:
                self.assertTrue(page.use_saved_plan(source,expected_revision='knee-saved-7',expected_result_sha256=frozen.public_plan['result_sha256']))
                self.wait(lambda:page._candidate_token and not page.busy)
                self.assertFalse(ExecutionStore(root).path.exists())
                self.assertIn('右膝检查任务',page.plan_context.text())
                for text in ('右膝检查任务','knee-saved-7'):self.assertIn(text,page.plan_context.text())
                self.assertEqual(page._primary_action,'freeze')
                previous_token=page._candidate_token;original_summary=page.plan_context.text()
                page.preview_button.click()
                self.wait(lambda:page._candidate_token and page._candidate_token!=previous_token and not page.busy)
                self.assertEqual(page.plan_context.text(),original_summary,'Rechecking the same source restores its summary')
                self.assertFalse(ExecutionStore(root).path.exists())
                page.freeze_button.click();self.wait(lambda:page._snapshot and page._snapshot['run_id'] and not page.busy)
                self.assertEqual(load.call_count,2,'Each explicit recheck verifies geometry once, then freezing reuses it')
                self.assertEqual(page._snapshot['plan_summary']['revision_id'],'knee-saved-7')
                self.assertFalse(page._snapshot['execution_ready'])
                self.assertEqual(sha256_file(source),frozen.assets[0].sha256)

    def test_changed_saved_version_and_asset_are_rejected_without_fallback(self):
        with self.workspace() as root:
            source,frozen=self.saved_fixture(root);page=self.page(root)
            with patch('femto.app.qt_execution.load_saved_plan',return_value=frozen):
                page.use_saved_plan(source,expected_revision='different-version')
                self.wait(lambda:page._action_error and not page.busy)
                self.assertFalse(ExecutionStore(root).path.exists())
                self.assertIsNone(page._candidate_token)
                self.assertTrue(page.exception_box.isVisible())
                self.assertIn('版本',page.exception_message.text())
                self.assertTrue(page.exception_retry.isVisible())
                self.assertIsNone(page._primary_action)
                page.use_saved_plan(source,expected_revision='knee-saved-7')
                self.wait(lambda:page._candidate_token and not page.busy)
                source.write_text('changed archive',encoding='utf-8')
                page.freeze_button.click()
                self.wait(lambda:page._action_error and not page.busy)
                self.assertFalse(ExecutionStore(root).path.exists())
                self.assertIsNone(page._snapshot['run_id'])
                self.assertEqual(page.mode.currentIndex(),0)

    def test_rechecking_handoff_keeps_reviewed_version_until_source_changes(self):
        with self.workspace() as root:
            source,frozen=self.saved_fixture(root);page=self.page(root)
            updated=replace(frozen,data_json=canonical(dict(frozen.public_plan,revision_id='knee-saved-8')))
            with patch('femto.app.qt_execution.load_saved_plan',return_value=frozen) as load:
                page.use_saved_plan(source,expected_revision='knee-saved-7',
                                    expected_result_sha256=frozen.public_plan['result_sha256'])
                self.wait(lambda:page._candidate_token and not page.busy)
                load.return_value=updated
                page.preview_button.click()
                self.wait(lambda:(page._candidate_token or page._action_error) and not page.busy)
                self.assertIsNone(page._candidate_token)
                self.assertIn('版本',page.exception_message.text())
                self.assertEqual(page._preview_inputs['expected_revision'],'knee-saved-7')
                self.assertFalse(ExecutionStore(root).path.exists())
                load.return_value=frozen
                page.exception_retry.click()
                self.wait(lambda:page._candidate_token and not page.busy)
                self.assertEqual(page._candidate_summary['revision_id'],'knee-saved-7')
                page.path.setText(str(Path(root)/'another-plan.region.zip'))
                self.wait(lambda:page.preview_button.isEnabled() and not page.busy)
                load.return_value=updated
                page.preview_button.click()
                self.wait(lambda:page._candidate_token and not page.busy)
                self.assertIsNone(page._preview_inputs['expected_revision'])
                self.assertEqual(page._candidate_summary['revision_id'],'knee-saved-8')

    def test_current_result_opens_and_exports_this_run_after_older_history_selection(self):
        with self.workspace() as root:
            page=self.page(root);page.mode.setCurrentIndex(1);page.freeze_button.click()
            self.wait(lambda:page._snapshot and page._snapshot['run_id'] and not page.busy)
            previous=page._snapshot['run_id']
            page.abandon_button.click()
            self.wait(lambda:page._snapshot['state']=='abandoned_unrun' and not page.busy)
            page.history_refresh.click();self.wait(lambda:page.history_records.count()==1 and not page.busy)
            page.replay_button.click();self.wait(lambda:page._replay and not page.busy)
            self.ready(page);page.start_button.click()
            self.wait(lambda:page._snapshot['state']=='succeeded' and not page.busy)
            current=page._snapshot['run_id']
            self.assertNotEqual(current,previous)
            self.assertEqual(page.history_records.currentData(),previous)
            self.assertEqual(page._primary_action,'result')
            self.assertTrue(page.result_button.isVisible() and page.result_button.isEnabled())
            page.result_button.click()
            self.wait(lambda:page._engineering_record and page._engineering_record['run_id']==current and not page.busy)
            self.assertEqual(page.tabs.currentIndex(),1)
            self.assertEqual(page.history_records.currentData(),current)
            self.assertIn('模拟步骤完成',page.replay_text.toPlainText())
            self.assertTrue(page.state.isVisible() and page.stop_button.isVisible())
            exported=Path(root)/'current-result.json'
            with patch.object(QFileDialog,'getSaveFileName',return_value=(str(exported),'模拟事件 (*.json)')):
                page.export_button.click()
            self.wait(lambda:exported.exists() and not page.busy)
            self.assertEqual(json.loads(exported.read_text(encoding='utf-8'))['run_id'],current)

    def test_missing_context_stays_unknown_and_irrelevant_device_fields_are_not_unknown(self):
        text=plan_text(dict(source_path='C:/saved/old.region.zip',revision_id='old-1',plan_source='unknown'))
        self.assertNotIn('左右：',text);self.assertNotIn('体位：',text);self.assertIn('任务：未知',text)
        with self.workspace() as root:
            page=self.page(root);page.connect_button.click();self.wait(lambda:page._snapshot and not page.busy)
            arm,head=page._snapshot['devices']
            self.assertEqual(dict(_device_rows(arm)[1])['输出关闭'],'不适用')
            self.assertEqual(dict(_device_rows(head)[1])['机械停止'],'不适用')
            self.assertEqual(dict(_device_rows(head)[1])['底座空间状态'],'不适用')
            self.assertNotIn('连接代次',page.devices.toPlainText())
            self.assertNotIn('ready',page.devices.toPlainText())
            self.assertIn('就绪',page.devices.toPlainText())

    def test_unknown_exception_is_actionable_and_close_preserves_original_unknown_result(self):
        with self.workspace() as root:
            page=self.page(root);scenario=Path(root)/'unknown.json'
            write_json_strict(scenario,Scenario(complete_delay_s=.08,request_timeout_s=.4,confirm_stop=False).to_dict())
            page.command('scenario',str(scenario));self.wait(lambda:not page.busy and page.tabs.currentIndex()==2)
            self.ready(page);page.start_button.click();self.wait(lambda:page._snapshot['state']=='running' and not page.busy)
            page.tabs.setCurrentIndex(2);page.stop_button.click()
            self.wait(lambda:page._snapshot['state']=='result_unknown' and not page.busy)
            run=page._snapshot['run_id'];self.assertTrue(page.exception_box.isVisible())
            for text in ('发生了什么','影响','下一步'):self.assertIn(text,page.exception_message.text())
            with patch.object(QMessageBox,'question',return_value=QMessageBox.No):page.exception_reset.click()
            self.assertEqual(page._snapshot['run_id'],run);self.assertTrue(page._snapshot['resource_held'])
            with patch.object(QMessageBox,'question',return_value=QMessageBox.Yes):page.exception_reset.click()
            self.wait(lambda:page._snapshot is None and not page.busy)
            self.assertEqual(ExecutionStore(root).read(run)['display_state'],'result_unknown')
            self.assertIn('历史记录保留',page.replay_text.toPlainText())
            self.assertIn('尚未核验方案',page.plan_context.text())
            self.assertIsNone(page._preview_inputs)

    def test_empty_history_and_saved_record_show_plain_summary_with_engineering_details(self):
        with self.workspace() as root:
            page=self.page(root);page.history_refresh.click();self.wait(lambda:not page.busy)
            self.assertIn('暂无模拟记录',page.replay_text.toPlainText())
            self.assertFalse(ExecutionStore(root).path.exists())
            source,frozen=self.saved_fixture(root)
            with patch('femto.app.qt_execution.load_saved_plan',return_value=frozen):
                page.use_saved_plan(source);self.wait(lambda:page._candidate_token and not page.busy)
                page.freeze_button.click();self.wait(lambda:page._snapshot and page._snapshot['run_id'] and not page.busy)
            page.abandon_button.click();self.wait(lambda:page._snapshot['state']=='abandoned_unrun' and not page.busy)
            page.history_refresh.click();self.wait(lambda:page.history_records.count()==1 and not page.busy)
            self.assertIn('右膝检查任务',page.history_records.currentText())
            page.replay_button.click();self.wait(lambda:page._replay and not page.busy)
            text=page.replay_text.toPlainText()
            for label in ('已放弃、未运行','右膝检查任务','已确认软件步骤'):self.assertIn(label,text)
            self.assertFalse(text.lstrip().startswith('{'))
            self.assertTrue(page.history_details_button.isEnabled())
            self.assertIn('recent_events',page._engineering_record)
            self.assertTrue(page.state.isVisible())

    def test_review_button_hands_off_saved_revision_without_auto_freeze(self):
        with self.workspace() as root:
            source,frozen=self.saved_fixture(root)
            with patch('femto.app.qt_main.ExecutionPage',side_effect=lambda:ExecutionPage(outputs=root)):
                window=WorkstationWindow(connect_camera=False)
            self._workspace_pages.append(window.execution)
            try:
                window.review.source=source
                window.review.model=SimpleNamespace(revision_id='knee-saved-7')
                window.review.plan=dict(surface_trajectory=frozen.public_plan)
                window.review.simulate.setEnabled(True)
                window.pages.setCurrentWidget(window.review)
                with patch('femto.app.qt_execution.load_saved_plan',return_value=frozen):
                    window.review.simulate.click()
                    self.wait(lambda:window.execution._candidate_token and not window.execution.busy)
                self.assertIs(window.pages.currentWidget(),window.execution)
                self.assertEqual(window.execution.path.text(),str(source))
                self.assertNotIn('体位：',window.execution.plan_context.text())
                self.assertFalse(ExecutionStore(root).path.exists())
                self.assertEqual(window.execution._preview_inputs['expected_revision'],'knee-saved-7')
                self.assertEqual(window.execution._preview_inputs['expected_result_sha256'],frozen.public_plan['result_sha256'])
            finally:
                window.review.model=None;window.review.plan=None
                window._close_timer.stop()
                self.wait(lambda:all((window.planning.shutdown(),window.review.shutdown(),window.capture.shutdown(),window.execution.shutdown())))
                window.close();self.app.processEvents()

    def test_cancelled_or_switched_candidate_never_installs_late_summary(self):
        with self.workspace() as root:
            source,frozen=self.saved_fixture(root);page=self.page(root)
            entered=threading.Event();released=threading.Event()
            def slow(path, *, cancel=None, progress=lambda text:None):
                entered.set()
                while not released.wait(.005):
                    if cancel.is_set():raise RuntimeError('Cancelled source verification')
                return frozen
            with patch('femto.app.qt_execution.load_saved_plan',side_effect=slow):
                page.use_saved_plan(source);self.wait(entered.is_set)
                page.cancel_reading()
                self.wait(lambda:not page.busy and page._action_error)
                released.set()
            self.assertIsNone(page._candidate_token)
            self.assertFalse(ExecutionStore(root).path.exists())
            with patch('femto.app.qt_execution.load_saved_plan',return_value=frozen):
                page.use_saved_plan(source);self.wait(lambda:page._candidate_token and not page.busy)
                page.path.setText(str(Path(root)/'different.region.zip'))
                self.wait(lambda:not page.busy)
                self.assertIsNone(page._candidate_token)
                self.assertFalse(page.freeze_button.isEnabled())

    def test_irrelevant_fixed_base_motion_is_not_applicable(self):
        with self.workspace() as root:
            page=self.page(root);scenario=Path(root)/'fixed.json'
            write_json_strict(scenario,Scenario(base_mode='fixed').to_dict())
            page.command('scenario',str(scenario));self.wait(lambda:not page.busy and page.tabs.currentIndex()==2)
            self.assertFalse(page.details_text.toPlainText().lstrip().startswith('{'))
            self.assertTrue(page.detail_engineering_button.isEnabled())
            page.connect_button.click();self.wait(lambda:page._snapshot and not page.busy)
            base=next(d for d in page._snapshot['devices'] if d['status']['binding']['kind']=='base')
            rows=dict(_device_rows(base)[1])
            self.assertEqual(rows['机械停止'],'不适用')
            self.assertEqual(rows['输出关闭'],'不适用')
            self.assertIn('固定',rows['底座空间状态'])

    def test_imported_installation_and_synthetic_gap_keep_provenance_and_units_visible(self):
        with self.workspace() as root:
            page=self.page(root)
            self.assertIn('尚未导入安装记录',page.details_text.toPlainText())
            page.mode.setCurrentIndex(1)
            self.assertFalse(page.path.isEnabled());self.assertFalse(page.choose.isEnabled())
            self.assertTrue(page.mode.isEnabled())
            evidence=Path(root)/'installation-evidence.txt'
            evidence.write_text('unverified measurement record',encoding='utf-8')
            record=TransformRecord('F','C',((1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1)),
                'imported-hand-eye','1','measured_import_unverified',('arm','camera'),('mount',),
                json.dumps(dict(files=[dict(path=str(evidence),sha256=sha256_file(evidence))])),
                'documented flange axes',RDF_AXES)
            source=save_record(record,outputs=Path(root)/'input')
            page.command('calibration',str(source))
            self.wait(lambda:page._installation_details and not page.busy)
            text=page.details_text.toPlainText()
            for label in ('导入安装记录','实测资格未验证','C → F','来源摘要核对通过','实时测距不可用'):
                self.assertIn(label,text)
            self.assertFalse(text.lstrip().startswith('{'))
            self.assertFalse(page._installation_details['evidence_status']['measurement_qualified'])
            page.gap_button.click()
            self.wait(lambda:'gap_demo' in page._installation_details and not page.busy)
            text=page.details_text.toPlainText()
            for label in ('独立合成间隙演示','0.200 m','参考坐标：C','已知合成平面'):
                self.assertIn(label,text)
            self.assertFalse(ExecutionStore(root).path.exists())
            gap=page._installation_details['gap_demo']
            self.assertFalse(gap['execution_ready']);self.assertFalse(gap['physical_accuracy_verified'])
            with patch('femto.app.qt_execution.show_text') as details:
                page.detail_engineering_button.click()
            exported=json.loads(details.call_args.args[2])
            self.assertEqual(exported['gap_demo']['distance_m'],.2)
            self.assertEqual(sha256_file(evidence),record.to_dict()['evidence']['files'][0]['sha256'])

    def test_non_plan_failure_does_not_offer_plan_retry_or_freeze_a_task(self):
        with self.workspace() as root:
            page=self.page(root);page.mode.setCurrentIndex(1)
            page.command('configuration',str(Path(root)/'missing-config.json'))
            self.wait(lambda:page._action_error and not page.busy)
            self.assertEqual(page._failure_action,'configuration')
            self.assertTrue(page.exception_box.isVisible())
            self.assertFalse(page.exception_retry.isVisible())
            self.assertIsNone(page._primary_action)
            self.assertFalse(page.retry_plan())
            self.assertFalse(ExecutionStore(root).path.exists())
            page.command('replay','missing-record')
            self.wait(lambda:page._failure_action=='replay' and not page.busy)
            self.assertFalse(page.exception_retry.isVisible())
            self.assertIsNone(page._primary_action)
            self.assertFalse(page.retry_plan())
            self.assertFalse(ExecutionStore(root).path.exists())

    def test_persistence_gap_recovery_action_does_not_hide_stop(self):
        with self.workspace() as root:
            page=self.page(root);self.ready(page)
            original=ExecutionStore.append
            def fail(store,run,event,kind,*args):
                if kind=='operator_start':raise OSError('disk full for product test')
                return original(store,run,event,kind,*args)
            with patch.object(ExecutionStore,'append',fail):
                page.start_button.click()
                self.wait(lambda:page._snapshot['persistence_gap'])
                self.assertTrue(page.exception_box.isVisible())
                self.assertTrue(page.exception_recover.isVisible())
                self.assertIn('后续步骤已阻止',page.exception_message.text())
                self.assertTrue(page.stop_button.isVisible())
                self.wait(lambda:page._snapshot['state'] in ('failed','result_unknown') and not page.busy)
            page.exception_recover.click()
            self.wait(lambda:not page._snapshot['persistence_gap'] and not page.busy)
            self.assertIn('模拟任务失败',page.exception_title.text())
            self.assertFalse(page.start_button.isEnabled())


if __name__=='__main__':unittest.main()
