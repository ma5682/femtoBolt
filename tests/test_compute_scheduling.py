"""Thread-local scheduling hints cannot change task semantics or affinity."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from contextlib import contextmanager
import threading
import unittest
from unittest.mock import Mock, patch

from femto.compute_scheduling import interactive_task_thread


def cpu(logical, *, group=0, efficiency=1, parked=False, allocated=False, own=False):
    return dict(logical=logical,group=group,efficiency=efficiency,parked=parked,allocated=allocated,own=own)


class ComputeSchedulingTests(unittest.TestCase):
    def backend(self, rows=None, affinity=(0,255)):
        return Mock(high_qos=Mock(return_value=True),topology=Mock(return_value=rows or [cpu(0),cpu(1,efficiency=0)]),
                    affinity=Mock(return_value=affinity),prefer=Mock(return_value='previous'),restore=Mock(return_value=True))

    def enter(self, backend):
        return patch('femto.compute_scheduling._WindowsThread',return_value=backend)

    def test_non_windows_does_not_load_windows_apis(self):
        with patch('femto.compute_scheduling.sys.platform','linux'),patch('femto.compute_scheduling._WindowsThread') as load:
            with interactive_task_thread() as report:self.assertFalse(report['high_qos'])
        load.assert_not_called()

    def test_unavailable_api_does_not_block_or_swallow_task_error(self):
        with patch('femto.compute_scheduling.sys.platform','win32'), \
                patch('femto.compute_scheduling._WindowsThread',side_effect=AttributeError('No API')):
            with self.assertRaisesRegex(ValueError,'original task error'):
                with interactive_task_thread():raise ValueError('original task error')

    def test_respects_existing_group_mask_parking_and_exclusive_allocation(self):
        rows=[cpu(0,efficiency=0),cpu(1,group=1),cpu(2),cpu(3,parked=True),
              cpu(4,allocated=True),cpu(5,allocated=True,own=True)]
        backend=self.backend(rows,(0,(1<<0)|(1<<3)|(1<<4)|(1<<5)))
        with patch('femto.compute_scheduling.sys.platform','win32'),self.enter(backend):
            with interactive_task_thread() as report:self.assertEqual(report['preferred_processor'],[0,5])
        backend.prefer.assert_called_once_with(0,5)
        backend.restore.assert_called_once_with('previous')

    def test_qos_failure_does_not_prevent_ideal_hint_and_restores_on_task_error(self):
        backend=self.backend();backend.high_qos.side_effect=OSError('Unsupported QoS')
        with patch('femto.compute_scheduling.sys.platform','win32'),self.enter(backend):
            with self.assertRaisesRegex(RuntimeError,'task failed'):
                with interactive_task_thread() as report:
                    self.assertFalse(report['high_qos']);raise RuntimeError('task failed')
        backend.prefer.assert_called_once();backend.restore.assert_called_once()

    def test_homogeneous_or_disallowed_fast_cores_leave_ideal_unchanged(self):
        for rows,affinity in (([cpu(0),cpu(1)],(0,3)),([cpu(0),cpu(1,efficiency=0)],(0,2)),
                              ([cpu(0),cpu(1,efficiency=0)],None)):
            with self.subTest(rows=rows,affinity=affinity):
                backend=self.backend(rows,affinity)
                with patch('femto.compute_scheduling.sys.platform','win32'),self.enter(backend):
                    with interactive_task_thread():pass
                backend.high_qos.assert_called_once();backend.prefer.assert_not_called();backend.restore.assert_not_called()

    def test_ideal_or_restore_failure_does_not_change_successful_task(self):
        for error in ('topology','prefer','restore'):
            with self.subTest(error=error):
                backend=self.backend();getattr(backend,error).side_effect=OSError('Unavailable')
                values=[]
                with patch('femto.compute_scheduling.sys.platform','win32'),self.enter(backend):
                    with interactive_task_thread():values.append('result')
                self.assertEqual(values,['result'])

    def test_task_runner_calls_hint_in_its_worker_only_and_keeps_error_delivery(self):
        from PySide6.QtCore import QEventLoop,QTimer
        from PySide6.QtWidgets import QApplication
        from femto.app.qt_common import TaskRunner
        app=QApplication.instance() or QApplication([])
        owner=threading.get_ident();events=[];errors=[]
        @contextmanager
        def hint():
            events.append(('enter',threading.get_ident()))
            try:yield {}
            finally:events.append(('exit',threading.get_ident()))
        def job(cancel,progress):
            events.append(('job',threading.get_ident()))
            raise ValueError('unchanged task failure')
        runner=TaskRunner();loop=QEventLoop();timer=QTimer();timer.setSingleShot(True)
        timer.timeout.connect(loop.quit);runner.finished.connect(loop.quit)
        with patch('femto.app.qt_common.interactive_task_thread',hint):
            self.assertTrue(runner.start(job,lambda value:self.fail('Unexpected success'),
                lambda error:errors.append((str(error),threading.get_ident()))))
            timer.start(3000)
            if runner.busy:loop.exec()
            timer.stop()
        self.assertFalse(runner.busy);self.assertTrue(runner.shutdown())
        self.assertEqual([name for name,_ in events],['enter','job','exit'])
        self.assertTrue(all(ident!=owner for _,ident in events))
        self.assertEqual(len({ident for _,ident in events}),1)
        self.assertEqual(errors,[('unchanged task failure',owner)])


if __name__=='__main__':unittest.main()
