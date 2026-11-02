"""Interrupting a real manifest write preserves the Qt plan and saved version."""
from copy import deepcopy
import unittest
from unittest.mock import patch

from PySide6.QtCore import QEventLoop, QTimer
import test_qt_planning as fixture
from femto.planning import region_project


class QtSaveInterruptionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):fixture.QtPlanningTests.setUpClass()

    def setUp(self):
        self.case=fixture.QtPlanningTests();self.case.setUp();self.addCleanup(self.case.doCleanups)
        self.case.mark();self.page=self.case.page
        self.page.generate();self.case.wait_task()
        self.page.set_stage(2);self.page.reviewed.setChecked(True)
        self.page.save_plan();self.case.wait_task()

    def interrupt(self, cancel):
        page=self.page;plan=page.plan;before=deepcopy(plan);archive=page.archive
        raw=(archive/'planning.region.zip').read_bytes();reviewed=page.reviewed.isChecked();dirty=page.dirty
        original=region_project._write_manifest;interrupted=[]
        def write(path,payload,check,**kwargs):
            def controlled():
                if any(p.stat().st_size for p in path.parent.glob('.project.json-*.tmp')):
                    interrupted.append(True)
                    if cancel:page.runner.cancel()
                    else:raise OSError('injected manifest disk failure')
                check()
            return original(path,payload,controlled,**kwargs)
        loop=QEventLoop();limit=QTimer();limit.setSingleShot(True)
        page.runner.finished.connect(loop.quit);limit.timeout.connect(loop.quit)
        try:
            with patch.object(region_project,'_write_manifest',side_effect=write):
                page.save_plan();limit.start(60000)
                if page.busy:loop.exec()
        finally:limit.stop();page.runner.finished.disconnect(loop.quit)
        self.assertFalse(page.busy);self.assertTrue(interrupted)
        self.assertIs(page.plan,plan);self.assertEqual(page.plan,before)
        self.assertEqual(page.archive,archive);self.assertEqual(page.dirty,dirty)
        self.assertEqual(page.reviewed.isChecked(),reviewed)
        self.assertEqual((archive/'planning.region.zip').read_bytes(),raw)
        if cancel:self.assertIn('任务已取消',page.status_label.text())
        else:self.assertIn('可编辑工程保存未完成',page.status_label.text())
        page.save_plan();self.case.wait_task()
        self.assertIs(page.plan,plan);self.assertEqual(page.plan,before)
        self.assertNotEqual(page.archive,archive)
        self.assertEqual((archive/'planning.region.zip').read_bytes(),raw)

    def test_mid_manifest_disk_failure_can_retry_without_losing_saved_plan(self):self.interrupt(False)

    def test_mid_manifest_cancel_can_retry_without_losing_saved_plan(self):self.interrupt(True)

    def test_save_reports_real_stages_and_event_loop_remains_responsive(self):
        stages=[];ticks=[]
        self.page.runner.progress.connect(stages.append)
        timer=QTimer();timer.setInterval(10);timer.timeout.connect(lambda:ticks.append(True))
        timer.start()
        try:
            self.page.save_plan();self.case.wait_task()
        finally:
            timer.stop();self.page.runner.progress.disconnect(stages.append)
        self.assertGreater(len(ticks), 5)
        self.assertTrue(any('2/4 重建并核验原目标' == text for text in stages), stages)
        self.assertTrue(any('3/4 复核本次保存的原目标验证结果' == text for text in stages), stages)
        phases=[int(text[0]) for text in stages if len(text)>3 and text[1:3]=='/4']
        self.assertEqual(sorted(set(phases)), [1,2,3,4])
        self.assertEqual(phases, sorted(phases))


if __name__=='__main__':unittest.main()
