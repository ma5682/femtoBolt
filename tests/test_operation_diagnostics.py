"""Diagnostics must survive failed workers without influencing calculations."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import gc
import weakref
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from femto import operation_diagnostics as diagnostics


class OperationDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def record(self, name):return diagnostics.record_operation(name, root=self.root/'diagnostics')

    def test_nested_timings_have_inclusive_and_exclusive_totals_without_arguments(self):
        marker = 'private-argument-must-not-be-serialized'
        @diagnostics.diagnosed('inner')
        def inner(value):return value
        with self.record('outer') as trace:
            self.assertEqual(inner(marker), marker)
        value = json.loads((trace.directory/'operation.json').read_text(encoding='utf-8'))
        self.assertEqual(value['status'], 'complete')
        self.assertEqual(value['stages']['inner']['calls'], 1)
        for row in value['stages'].values():
            self.assertGreaterEqual(row['self_seconds'], 0)
            self.assertLessEqual(row['self_seconds'], row['inclusive_seconds'])
        self.assertLessEqual(value['stages']['inner']['inclusive_seconds'], value['stages']['outer']['inclusive_seconds'])
        self.assertNotIn(marker, ''.join(p.read_text(encoding='utf-8') for p in trace.directory.iterdir()))

    def test_failure_keeps_full_traceback_and_original_exception(self):
        error = ValueError('meaningful calculation failure')
        with self.assertRaises(ValueError) as caught:
            with self.record('fails') as trace:raise error
        self.assertIs(caught.exception, error)
        value = json.loads((trace.directory/'operation.json').read_text(encoding='utf-8'))
        self.assertEqual(value['status'], 'failed')
        self.assertIn('meaningful calculation failure', (trace.directory/'error.log').read_text(encoding='utf-8'))
        self.assertIsNone(diagnostics.CURRENT.get())

    def test_repeated_runs_own_distinct_directories_and_leave_previous_bytes(self):
        with self.record('same') as first:pass
        before = {p.name:p.read_bytes() for p in first.directory.iterdir()}
        with self.record('same') as second:pass
        self.assertNotEqual(first.directory, second.directory)
        self.assertEqual(before, {p.name:p.read_bytes() for p in first.directory.iterdir()})

    def test_thread_contexts_do_not_mix_stage_events(self):
        barrier = threading.Barrier(2)
        def run(name):
            with self.record(name) as trace:
                barrier.wait(timeout=5)
                with trace.stage(name+'.child'):
                    self.assertIs(diagnostics.CURRENT.get(), trace)
            return json.loads((trace.directory/'operation.json').read_text(encoding='utf-8'))
        with ThreadPoolExecutor(max_workers=2) as pool:
            rows = list(pool.map(run, ('first','second')))
        for row in rows:self.assertEqual(set(row['stages']), {row['operation'], row['operation']+'.child'})
        self.assertIsNone(diagnostics.CURRENT.get())

    def test_event_cap_preserves_aggregate_counts_and_marks_omission(self):
        with patch.object(diagnostics, 'MAX_EVENTS', 4), self.record('many') as trace:
            for _ in range(12):
                with trace.stage('repeat'):pass
        value = json.loads((trace.directory/'operation.json').read_text(encoding='utf-8'))
        self.assertEqual(value['recorded_events'], 4)
        self.assertGreater(value['omitted_events'], 0)
        self.assertEqual(value['stages']['repeat']['calls'], 12)

    def test_recording_failure_cannot_replace_success_or_the_original_error(self):
        with patch.object(diagnostics, '_new_directory', side_effect=OSError('disk unavailable')):
            with self.record('success') as trace:result = 17
            self.assertEqual(result, 17); self.assertIn('disk unavailable', trace.recording_error)
            original = RuntimeError('original failure')
            with self.assertRaises(RuntimeError) as caught:
                with self.record('failure'):raise original
            self.assertIs(caught.exception, original)

    def test_cancel_is_distinct_from_failure(self):
        class TestCancelled(RuntimeError):pass
        with self.assertRaises(TestCancelled):
            with self.record('cancel') as trace:raise TestCancelled('requested')
        value = json.loads((trace.directory/'operation.json').read_text(encoding='utf-8'))
        self.assertEqual(value['status'], 'cancelled')
        self.assertEqual(value['stages']['cancel']['failed_calls'], 0)
        self.assertEqual(value['stages']['cancel']['stopped_calls'], 1)

    def test_deepest_failure_and_published_artifact_survive_wrapped_error(self):
        original = ValueError('archive failed')
        published = self.root/'published'
        published.mkdir()
        with self.assertRaises(RuntimeError):
            with self.record('qt_worker') as trace:
                diagnostics.mark_retained(published, '三维路线包')
                try:
                    with trace.stage('portable.save'):
                        with trace.stage('portable.archive'):raise original
                except ValueError as error:
                    raise RuntimeError('partial save') from error
        summary = trace.summary()
        self.assertEqual(summary.stage, 'portable.archive')
        self.assertEqual(summary.error_type, 'RuntimeError')
        self.assertEqual(summary.retained[0].path, str(published.resolve()))
        self.assertIsNone(trace._failed_exception)
        saved = json.loads((trace.directory/'operation.json').read_text(encoding='utf-8'))
        self.assertEqual(saved['failed_stage'], 'portable.archive')
        self.assertEqual(saved['retained_artifacts'][0]['kind'], '三维路线包')

    def test_caught_failure_does_not_label_success_or_later_error_as_old_stage(self):
        with self.record('outer') as success:
            try:
                with success.stage('handled'):raise ValueError('handled')
            except ValueError:pass
        self.assertIsNone(success.summary().stage)
        with self.assertRaises(TypeError):
            with self.record('outer') as trace:
                try:
                    with trace.stage('handled'):raise ValueError('handled')
                except ValueError:pass
                with trace.stage('new'):raise TypeError('new failure')
        self.assertEqual(trace.summary().stage, 'new')

    def test_in_memory_failure_stage_survives_unavailable_disk(self):
        with patch.object(diagnostics, '_new_directory', side_effect=OSError('disk unavailable')):
            with self.assertRaises(ValueError):
                with self.record('worker') as trace:
                    with trace.stage('portable.archive'):raise ValueError('failed')
        self.assertEqual(trace.summary().stage, 'portable.archive')
        self.assertIsNone(trace.summary().directory)
        self.assertIn('disk unavailable', trace.summary().recording_error)

    def test_successful_enclosing_stage_releases_handled_budget_exception_frames(self):
        class _BudgetStop(RuntimeError):pass
        class Payload:
            def __init__(self):self.buffer=bytearray(1024*1024)
        probes=[]
        def calculate(trace):
            payload=Payload();probes.append(weakref.ref(payload))
            with trace.stage('inner_budget'):raise _BudgetStop()
        with self.record('worker') as trace:
            with trace.stage('calculation'):
                try:calculate(trace)
                except _BudgetStop:pass
            gc.collect()
            self.assertIsNone(probes[0]())
            self.assertEqual(trace.totals['inner_budget']['stopped_calls'],1)

    def test_real_worker_failure_log_survives_temporary_source_cleanup(self):
        from femto.pipeline.local_comparison import supervise, StageStopped
        durable = self.root/'durable'
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, FEMTO_DIAGNOSTICS_DIR=str(durable)):
            log = Path(temp)/'child.log'
            with self.assertRaises(StageStopped) as caught:
                supervise([sys.executable, '-X', 'utf8', '-c', 'print("persistent worker message", flush=True); raise SystemExit(7)'],
                          log, timeout_s=10, memory_mb=1024)
            self.assertEqual(caught.exception.stats['returncode'], 7)
            self.assertTrue(caught.exception.stats['failure_diagnostics_recorded'])
            destination = Path(caught.exception.stats['failure_diagnostics'])
            self.assertTrue(destination.is_relative_to(durable))
            original_bytes = log.read_bytes()
        self.assertFalse(log.exists())
        self.assertEqual((destination/'worker.log').read_bytes(), original_bytes)
        value = json.loads((destination/'failure.json').read_text(encoding='utf-8'))
        self.assertEqual(value['source_sha256'], hashlib.sha256(original_bytes).hexdigest())
        self.assertFalse(value['command_and_environment_recorded'])

    def test_log_tail_is_bounded_and_cannot_claim_full_source_hash(self):
        log = self.root/'large.log'; original = b'header---the complete error tail'
        log.write_bytes(original)
        with patch.object(diagnostics, 'MAX_FAILURE_LOG_BYTES', 8):
            directory = diagnostics.retain_subprocess_failure(log, status='failed', stats={'returncode':1},
                error=RuntimeError('failure'), root=self.root/'durable')
        value = json.loads((directory/'failure.json').read_text(encoding='utf-8'))
        self.assertTrue(value['truncated']); self.assertEqual(value['stored_bytes'], 8)
        self.assertIsNone(value['source_sha256'])
        self.assertEqual((directory/'worker.log').read_bytes(), original[-8:])
        self.assertEqual(log.read_bytes(), original)

    def test_timing_records_do_not_enter_coverage_evidence(self):
        from femto.planning import coverage_certificate as coverage, footprint_model
        from test_coverage_certificate import triangle_domain, seed
        domain, model = triangle_domain(), footprint_model.nominal(500)
        before = coverage.evaluate(domain, [seed()], model, algorithm=coverage.CURRENT_ALGORITHM)
        with self.record('coverage'):
            after = coverage.evaluate(domain, [seed()], model, algorithm=coverage.CURRENT_ALGORITHM)
        self.assertEqual(before.metadata, after.metadata)
        self.assertNotIn('elapsed_seconds', str(after.metadata))


if __name__ == '__main__':unittest.main()
