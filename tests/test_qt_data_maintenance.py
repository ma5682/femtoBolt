import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest
from PySide6.QtCore import Qt,QTimer

from femto.app import qt_data_maintenance as ui
from femto import data_backup
from femto.data_access import DataLease


class QtDataMaintenanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        raw=self.root/'datasets/scene_test_local_test';raw.mkdir(parents=True)
        (raw/'manifest.json').write_text('{}');(raw/'rgb.bin').write_bytes(b'recorded')
        self.rec=self.root/'outputs/complete_demo_test/reconstruction';self.rec.mkdir(parents=True)
        (self.rec/'reconstruction_report.json').write_text(json.dumps(dict(source_dataset=str(raw))))
        self.dialog=ui.DataMaintenanceDialog(self.root);self.dialog.show();self.addCleanup(self.close)
        self.destination=self.root/'backup';self.dialog.path.setText(str(self.destination));self.app.processEvents()

    def close(self):
        self.dialog.close();self.wait();self.dialog.close();self.app.processEvents()

    def wait(self):
        deadline=time.monotonic()+10
        while self.dialog.runner.busy and time.monotonic()<deadline:
            self.app.processEvents();time.sleep(.002)
        self.app.processEvents();self.assertFalse(self.dialog.runner.busy)

    def click_run(self):
        QTest.mouseClick(self.dialog.run,Qt.LeftButton);self.wait()

    def test_real_backup_and_restore_failure_retry_via_widgets(self):
        self.click_run();self.assertIn('备份已保存',self.dialog.status.text());data_backup.verify(self.destination)
        self.dialog.operation.setCurrentIndex(self.dialog.operation.findData('restore'))
        self.dialog.inspect_data();self.wait()
        self.click_run();self.assertIn('未完成',self.dialog.status.text())
        self.assertTrue((self.rec/'reconstruction_report.json').is_file())
        for index in range(self.dialog.roots.topLevelItemCount()):
            item=self.dialog.roots.topLevelItem(index);item.setText(1,str(self.root/('restored_'+item.data(0,Qt.UserRole))))
        self.click_run();self.assertIn('恢复完成',self.dialog.status.text())
        self.assertIn('来源路径未重写',self.dialog.status.text())
        self.assertEqual((self.root/'restored_datasets/scene_test_local_test/rgb.bin').read_bytes(),b'recorded')

    def test_open_workstation_gate_keeps_failed_backup_retryable(self):
        with DataLease(self.root):
            self.click_run();self.assertIn('关闭',self.dialog.status.text());self.assertFalse(self.destination.exists())
        self.click_run();self.assertTrue(self.destination.is_dir())

    def test_existing_missing_is_explicit_and_not_claimed_complete(self):
        (self.rec/'reconstruction_report.json').write_text(json.dumps(dict(source_dataset=str(self.root/'missing'))))
        self.click_run();self.assertIn('缺失',self.dialog.status.text());self.assertFalse(self.destination.exists())
        self.dialog.allow_missing.setChecked(True);self.click_run()
        self.assertIn('原数据已有1项',self.dialog.status.text())
        self.assertFalse(data_backup.verify(self.destination)['reference_audit']['source_assets_complete'])

    def test_late_cancel_does_not_hide_published_backup(self):
        original=ui.backup_workspace
        def published(*args,**kwargs):
            result=original(*args,**kwargs);self.dialog.runner._cancel.set();return result
        with patch.object(ui,'backup_workspace',side_effect=published):self.click_run()
        self.assertIn('已发布结果仍保留',self.dialog.status.text());data_backup.verify(self.destination)

    def test_disk_preview_shows_capacity_without_deleting_draft(self):
        draft=self.rec/'.draft_test.tmp';draft.write_bytes(b'recovery evidence')
        QTest.mouseClick(self.dialog.disk_preview,Qt.LeftButton);self.wait()
        self.assertIn('未删除文件',self.dialog.status.text())
        self.assertIn('可用',self.dialog.status.text())
        self.assertTrue(self.dialog.detail_toggle.isChecked())
        self.assertEqual(draft.read_bytes(),b'recovery evidence')
        report=json.loads(self.dialog.details.toPlainText())
        self.assertEqual(report['cleanup']['recovery_review']['files'],1)
