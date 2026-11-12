"""Structured failure evidence reaches only its current Qt task and UI."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication

from femto import operation_diagnostics as diagnostics
from femto.app.qt_common import TaskRunner, TaskResultDiscarded
from femto.app.qt_failure_details import capture_failure_text, failure_text,brief_failure_text


class FailurePresentationTests(unittest.TestCase):
    def test_compact_notice_preserves_publication_fact_without_expanding_paths(self):
        cause=OSError('disk full')
        error=RuntimeError('wrapper with a very long published path')
        error.__cause__=cause
        summary=diagnostics.OperationSummary('failed','portable.save','RuntimeError',str(error),
            'diagnostic-directory',(diagnostics.RetainedArtifact('published-directory','三维路线包'),),None)
        brief=brief_failure_text(error,summary,has_model=True,current_task=True)
        self.assertIn('disk full',brief);self.assertIn('可编辑工程保存未完成',brief)
        self.assertNotIn('published-directory',brief);self.assertNotIn('diagnostic-directory',brief)
        full=failure_text(error,summary,retained_current='当前方案',next_steps=('重试',),current_task=True)
        self.assertIn('published-directory',full);self.assertIn('diagnostic-directory',full)

    def test_compact_notice_does_not_borrow_stale_publication_or_hide_discard(self):
        summary=diagnostics.OperationSummary('failed','portable.save','ValueError','old','old',
            (diagnostics.RetainedArtifact('old','三维路线包'),),None)
        brief=brief_failure_text(TaskResultDiscarded('changed'),summary,has_model=True)
        self.assertIn('结果未安装',brief);self.assertNotIn('已发布',brief)
    @classmethod
    def setUpClass(cls):cls.app = QApplication.instance() or QApplication([])

    def wait(self, runner):
        loop, timer = QEventLoop(), QTimer()
        timer.setSingleShot(True); timer.timeout.connect(loop.quit); runner.finished.connect(loop.quit)
        timer.start(5000)
        if runner.busy:loop.exec()
        timer.stop(); runner.finished.disconnect(loop.quit)
        self.assertFalse(runner.busy)

    def test_worker_preserves_original_error_and_current_stage_then_clears_next_task(self):
        runner, failures, summaries = TaskRunner(), [], []
        self.addCleanup(lambda: runner.shutdown(2000))
        error = ValueError('package failed')
        def work(cancel, progress):
            with diagnostics.CURRENT.get().stage('portable.archive'):raise error
        def failed(value):
            failures.append(value); summaries.append(runner.last_diagnostics)
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, FEMTO_DIAGNOSTICS_DIR=temp):
            runner.start(work, lambda value: self.fail('installed failed result'), failed)
            self.wait(runner)
            self.assertIs(failures[0], error)
            self.assertEqual(summaries[0].stage, 'portable.archive')
            text = failure_text(error, summaries[0], retained_current='当前方案', next_steps=('点击保存重试',))
            self.assertIn('打包可编辑工程', text); self.assertIn('点击保存重试', text)
            self.assertTrue(Path(summaries[0].directory).is_dir())
            runner.start(lambda cancel, progress: 42, lambda value: None, failed)
            self.assertIsNone(runner.last_diagnostics)
            self.wait(runner)
            self.assertEqual(runner.last_diagnostics.status, 'complete')
            self.assertIsNone(runner.last_diagnostics.stage)

    def test_discarded_result_cannot_claim_previous_failure_or_artifact(self):
        summary = diagnostics.OperationSummary('failed', 'portable.archive', 'ValueError', 'failed',
            'old-diagnostics', (diagnostics.RetainedArtifact('old-package', '路线包'),), None)
        text = failure_text(TaskResultDiscarded('inputs changed'), summary,
            retained_current='当前方案', next_steps=('重新计算',))
        self.assertIn('结果未安装', text)
        self.assertNotIn('old-package', text); self.assertNotIn('old-diagnostics', text)

    def test_cancelled_or_discarded_current_task_retains_explicit_publication_facts(self):
        from femto.annotation.fixed_rgb_geometry import GeometryTaskCancelled
        summary = diagnostics.OperationSummary('complete', None, None, None, 'current-diagnostics',
            (diagnostics.RetainedArtifact('current-package', '路线包'),), None)
        for error in (GeometryTaskCancelled('任务已取消'), TaskResultDiscarded('inputs changed')):
            text = failure_text(error, summary, retained_current='当前方案', next_steps=('重新核验',), current_task=True)
            self.assertIn('current-package', text)
            self.assertIn('未安装为当前方案', text)
            self.assertIn('原输入', text)

    def test_actual_delivery_after_publication_preserves_package_without_installing(self):
        for mode in ('cancel', 'changed-input'):
            runner, failures, installed = TaskRunner(), [], []
            current = ['first']
            with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, FEMTO_DIAGNOSTICS_DIR=temp):
                package = Path(temp)/'published'; package.mkdir()
                def work(cancel, progress):
                    diagnostics.mark_retained(package, '路线包')
                    if mode == 'cancel':runner.cancel()
                    else:current[0] = 'second'
                    return 'completed old request'
                def failed(error):
                    failures.append(failure_text(error, runner.last_diagnostics,
                        retained_current='当前方案', next_steps=('重新核验',), current_task=True))
                runner.start(work, installed.append, failed, revision='first', current_revision=lambda:current[0])
                self.wait(runner)
                self.assertEqual(installed, [])
                self.assertEqual(len(failures), 1)
                self.assertIn(str(package.resolve()), failures[0])
                self.assertIn('未安装为当前方案', failures[0])
                self.assertTrue(runner.shutdown(2000))

    def test_capture_existing_failure_facts_are_actionable_and_do_not_promote_residue(self):
        text = capture_failure_text({'error': 'worker exceeded memory', 'processing_report': 'report.json',
            'successful_model_published': True}, {'reconstruction': {'status': 'failed'},
            'worker_resources': {'failure_diagnostics_recorded': True, 'failure_diagnostics': 'worker-log'}}, 'raw-source')
        for value in ('模型重建', 'worker exceeded memory', 'raw-source', 'report.json', 'worker-log', '重新建立扫描参考', '独立核验'):
            self.assertIn(value, text)
        empty = capture_failure_text({'error': 'camera not ready'}, None, None)
        self.assertIn('尚无已记录', empty)
        self.assertIn('阶段未记录', empty)

    def test_capture_notice_does_not_hide_failure_when_range_needs_confirmation(self):
        from femto.app.qt_capture import CapturePage, CaptureState
        page = CapturePage(connect_camera=False)
        try:
            state = CaptureState(state='failed', failure={'error': 'specific failure'}, source=Path('raw-source'))
            page.roi_message = '需要确认部位'
            for stale in (False, True):
                with self.subTest(stale=stale):
                    title, message = page._capture_message(state, live=True, fresh=False, stale=stale)
                    self.assertIn('specific failure', message)
                    self.assertIn('下一步', message)
                    self.assertNotIn(page.roi_message, message)
                    self.assertEqual('相机画面已过期' in message, stale)
        finally:page.shutdown(); page.close()

    def test_next_planning_status_clears_previous_error_tooltip(self):
        from femto.app.qt_planning import PlanningPage
        page = PlanningPage()
        try:
            page._failure(ValueError('specific failure'))
            self.assertIn('specific failure', page.status_label.toolTip())
            self.assertIn('尚未载入', page.status_label.text())
            page._status('new operation')
            self.assertEqual(page.status_label.toolTip(), '')
        finally:page.shutdown(); page.close()


if __name__ == '__main__':unittest.main()
