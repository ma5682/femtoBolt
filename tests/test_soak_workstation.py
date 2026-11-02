"""Short failure-path tests; these never run reconstruction or real planning."""
from copy import deepcopy
import gc
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from tools.soak_workstation import (
    GIB, MIB, Measurements, NaturalGC, WeakObjects, WorkstationCycle, array_values, file_digest, gc_delta,
    linear_trend, memory_report, run_session,
)


class ObservedObject:
    pass


class SoakWorkstationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root/'soak'
        self.inputs = dict(files={'project.zip': 'input-hash'}, datasets={},
                           production_code={'femto/example.py': 'code-hash'}, tools={'runner': 'tool-hash'})

    def worker(self, folder, stage, tracker, cycle):
        observed = ObservedObject()
        tracker.watch('fixture', observed)
        stage('fixture_work', lambda: (folder/'asset.bin').write_bytes(b'unchanged result'))
        return dict(equivalence={'route': 'fixed', 'arrays': 'fixed'}, process='same test process')

    def run_short(self, worker=None, snapshot=None, **kwargs):
        settings = dict(duration_seconds=0, max_cycles=3, minimum_cycles=8,
                        idle_seconds=0, reserve_bytes=0, initial_cycle_bytes=0, warmup_cycles=1,
                        expected_stages=['fixture_work', 'cleanup_idle'])
        settings.update(kwargs)
        with patch('builtins.print'):
            return run_session(self.output, worker or self.worker, snapshot or (lambda: self.inputs),
                               **settings)

    def test_short_run_retains_every_cycle_and_never_claims_hour_qualification(self):
        with patch.object(gc, 'collect', side_effect=AssertionError('forced GC forbidden')), \
                patch.object(gc, 'disable', side_effect=AssertionError('GC disabling forbidden')):
            report = self.run_short()
        self.assertEqual(report['status'], 'stopped')
        self.assertEqual(report['stop_reason'], 'max_cycles')
        self.assertTrue(report['functional_checks_passed'])
        self.assertFalse(report['sustained_hour_qualified'])
        cycles = json.loads((self.output/'cycles.json').read_text())
        self.assertEqual(len(cycles), 3)
        for cycle, row in enumerate(cycles):
            self.assertEqual(row['status'], 'passed')
            self.assertEqual(row['weak_references']['alive'], 0)
            self.assertEqual((self.output/f'cycle_{cycle:04d}'/'asset.bin').read_bytes(), b'unchanged result')
        stages = json.loads((self.output/'stages.json').read_text())
        self.assertEqual([row['stage'] for row in stages], ['fixture_work', 'cleanup_idle']*3)
        self.assertTrue(all('gc' in row for row in stages))
        self.assertEqual(report['gc']['uncollectable'], 0)

    def test_failure_retains_partial_assets_and_running_stage_name(self):
        def fail(folder, stage, tracker, cycle):
            if cycle == 1:
                def bad_write():
                    (folder/'unfinished.bin').write_bytes(b'partial')
                    raise OSError('disk write failure')
                stage('portable_save', bad_write)
            return self.worker(folder, stage, tracker, cycle)
        report = self.run_short(fail)
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['completed_cycles'], 1)
        self.assertFalse(report['functional_checks_passed'])
        checkpoint = json.loads((self.output/'checkpoint.json').read_text())
        self.assertEqual(checkpoint['status'], 'failed')
        self.assertEqual(checkpoint['stage'], 'portable_save')
        self.assertEqual((self.output/'cycle_0001'/'unfinished.bin').read_bytes(), b'partial')
        self.assertIn('disk write failure', report['error'])

    def test_sampling_failure_is_not_a_completed_cycle(self):
        original = Measurements.run
        def broken_sampling(measurements, *args):
            value = original(measurements, *args)
            measurements.rows[-1]['sampling_errors'] = ['RSS access denied']
            return value
        with patch.object(Measurements, 'run', broken_sampling):
            report = self.run_short()
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['completed_cycles'], 0)
        self.assertFalse(report['functional_checks_passed'])
        self.assertFalse(report['sustained_hour_qualified'])
        state = json.loads((self.output/'checkpoint.json').read_text())
        self.assertEqual(state['measurement_status'], 'failed')
        self.assertIn('RSS access denied', state['error'])

    def test_missing_duplicate_or_out_of_order_stages_cannot_pass(self):
        for sequence in ([], ['fixture_work', 'fixture_work'], ['unexpected', 'fixture_work']):
            with self.subTest(sequence=sequence):
                self.output = self.root/('case_'+str(len(sequence))+str(sequence.count('fixture_work')))
                def worker(folder, stage, tracker, cycle):
                    for name in sequence:
                        stage(name, lambda: None)
                    return {'equivalence': {'same': True}}
                report = self.run_short(worker)
                self.assertEqual(report['completed_cycles'], 0)
                self.assertFalse(report['functional_checks_passed'])
                self.assertIn('Cycle stages', report['error'])

    def test_expected_stages_reject_invalid_contract_before_creation(self):
        with self.assertRaises(ValueError):
            self.run_short(expected_stages=[])
        self.assertFalse(self.output.exists())

    def test_completion_requires_five_post_warmup_cycles(self):
        report = self.run_short(max_cycles=10, minimum_cycles=1, warmup_cycles=4)
        self.assertEqual(report['required_cycles'], 9)
        self.assertEqual(report['completed_cycles'], 9)
        self.assertEqual(report['stop_reason'], 'requested_duration_and_cycles_completed')
        memory = json.loads((self.output/'memory.json').read_text())
        self.assertEqual(memory['post_warmup_cycles'], 5)
        self.assertFalse(report['sustained_hour_qualified'])

    def test_changed_later_result_fails_and_records_mismatched_digest(self):
        def changed(folder, stage, tracker, cycle):
            result = self.worker(folder, stage, tracker, cycle)
            if cycle == 1:
                result['equivalence']['arrays'] = 'changed'
            return result
        report = self.run_short(changed)
        self.assertEqual(report['completed_cycles'], 1)
        self.assertFalse(report['functional_checks_passed'])
        state = json.loads((self.output/'cycle_0001'/'state.json').read_text())
        self.assertEqual(state['result']['equivalence']['arrays'], 'changed')
        self.assertEqual(state['status'], 'failed')

    def test_disk_guard_stops_without_deleting_previous_cycles(self):
        free = [SimpleNamespace(free=8*GIB), SimpleNamespace(free=0)]
        with patch('tools.soak_workstation.shutil.disk_usage', side_effect=free):
            report = self.run_short(reserve_bytes=4*GIB, initial_cycle_bytes=256*MIB)
        self.assertEqual(report['status'], 'stopped')
        self.assertEqual(report['stop_reason'], 'disk_guard')
        self.assertEqual(report['completed_cycles'], 1)
        self.assertFalse(report['sustained_hour_qualified'])
        self.assertTrue((self.output/'cycle_0000'/'asset.bin').is_file())
        self.assertFalse((self.output/'cycle_0001').exists())

    def test_existing_output_is_never_resumed_or_overwritten(self):
        self.output.mkdir()
        checkpoint = self.output/'checkpoint.json'
        original = b'{"status":"running","stage":"portable_save"}'
        checkpoint.write_bytes(original)
        worker, snapshot = Mock(), Mock()
        with self.assertRaises(FileExistsError):
            self.run_short(worker, snapshot)
        worker.assert_not_called()
        snapshot.assert_not_called()
        self.assertEqual(checkpoint.read_bytes(), original)

    def test_source_and_code_changes_cannot_pass(self):
        for key, expected in (('files', 'source_unchanged'), ('production_code', 'code_unchanged')):
            with self.subTest(key=key):
                self.output = self.root/key
                after = deepcopy(self.inputs)
                after[key]['added'] = 'changed'
                report = self.run_short(snapshot=Mock(side_effect=[self.inputs, after]), max_cycles=1)
                self.assertEqual(report['status'], 'failed')
                self.assertFalse(report['inputs_and_code_unchanged'])
                self.assertFalse(report[expected])

    def test_final_hash_failure_is_explicit_and_keeps_assets(self):
        report = self.run_short(snapshot=Mock(side_effect=[self.inputs, OSError('source disappeared')]),
                                max_cycles=1)
        self.assertEqual(report['status'], 'failed')
        self.assertIn('Final input verification failed', report['error'])
        self.assertFalse(report['source_unchanged'])
        self.assertTrue((self.output/'cycle_0000'/'asset.bin').exists())

    def test_disabled_natural_gc_is_rejected_without_running_work(self):
        worker = Mock()
        callbacks = list(gc.callbacks)
        with patch.object(gc, 'isenabled', return_value=False):
            report = self.run_short(worker)
        self.assertEqual(report['status'], 'failed')
        self.assertIn('Natural GC is disabled', report['error'])
        worker.assert_not_called()
        self.assertEqual(gc.callbacks, callbacks)

    def test_gc_observer_records_duration_and_restores_callbacks(self):
        observer = NaturalGC()
        original = list(gc.callbacks)
        with observer, patch('tools.soak_workstation.time.perf_counter', side_effect=[1.0, 1.25]):
            before = observer.snapshot()
            observer.callback('start', {'generation': 2})
            observer.callback('stop', {'generation': 2, 'collected': 3, 'uncollectable': 0})
            delta = gc_delta(before, observer.snapshot())
        self.assertEqual(delta['seconds'], .25)
        self.assertEqual(delta['generations'], [0, 0, 1])
        self.assertEqual(delta['collected'], 3)
        self.assertEqual(gc.callbacks, original)

    def test_weak_observer_detects_aged_objects_without_retaining_them(self):
        tracker = WeakObjects()
        value = ObservedObject()
        tracker.watch('fixture', value)
        tracker.cycle = 2
        self.assertEqual(tracker.snapshot()['oldest_age'], 2)
        self.assertEqual(tracker.snapshot()['older_cycles'], 1)
        del value
        self.assertEqual(tracker.snapshot()['alive'], 0)

    def test_memory_trend_excludes_warmup_and_compares_same_stage(self):
        cycles = [dict(cycle=index, status='passed', elapsed_seconds=index*60,
                       cleanup_rss_bytes=(900 if index < 3 else 100+index)*MIB,
                       weak_references={'alive': 0}) for index in range(8)]
        stages = [dict(repetition=row['cycle'], status='passed', stage='history_open',
                       session_elapsed_seconds=row['elapsed_seconds'], rss_end_bytes=200*MIB)
                  for row in cycles]
        result = memory_report(cycles, stages, 3)
        self.assertTrue(result['enough_cycles'])
        self.assertEqual(result['cleanup']['min_rss_bytes'], 103*MIB)
        self.assertEqual(result['cleanup']['max_rss_bytes'], 107*MIB)
        self.assertAlmostEqual(result['cleanup']['slope_bytes_per_hour'], 60*MIB)
        self.assertEqual(result['same_stage_rss']['history_open']['slope_bytes_per_hour'], 0)
        self.assertIn('autocorrelated', result['cleanup']['note'])
        self.assertEqual(linear_trend([{'elapsed_seconds': 1, 'rss_bytes': 1}])['status'],
                         'insufficient_samples')

    def test_array_comparison_keeps_shape_dtype_and_bytes(self):
        array = np.arange(6, dtype=np.float64)
        original = array_values({'target': array})
        self.assertNotEqual(original, array_values({'target': array.reshape(2, 3)}))
        self.assertNotEqual(original, array_values({'target': array.astype(np.int64)}))
        self.assertNotEqual(original, array_values({'target': array+1}))

    def test_first_reconstruction_must_match_fixed_mesh_not_only_later_cycles(self):
        fixed = self.root/'fixed.ply'
        fixed.write_bytes(b'fixed mesh')
        worker = WorkstationCycle.__new__(WorkstationCycle)
        worker.expected_reconstruction = dict(mesh_sha256=file_digest(fixed), frame_count=5)
        worker.region_project = Mock()
        def rebuild(dataset, destination):
            destination.mkdir(parents=True)
            (destination/'scene_tsdf_mesh_m.ply').write_bytes(b'different mesh')
            return {'frame_count': 5}
        worker.reconstruction = (rebuild, self.root/'dataset', {})
        with self.assertRaisesRegex(AssertionError, 'fixed reference mesh'):
            worker(self.output, lambda name, function: function(), WeakObjects(), 0)
        worker.region_project.open_project.assert_not_called()

    def test_nonfinite_duration_is_rejected_before_output_creation(self):
        with self.assertRaises(ValueError):
            self.run_short(duration_seconds=float('nan'))
        self.assertFalse(self.output.exists())

    def test_sqlite_cycle_uses_only_isolated_metadata_and_restores_record(self):
        portable = self.root/'fixed.region.zip'
        portable.write_bytes(b'fixed source evidence')
        worker = WorkstationCycle.__new__(WorkstationCycle)
        worker.output = self.output
        row = dict(kind='区域工程 / 方案', name='example', project=portable,
                   project_id='project', revision_id='revision', parent_revision_id='parent')
        with patch('femto.planning.catalog.scan_saved_versions', return_value=[row]):
            result = worker.history(portable)
        self.assertTrue(result['hidden_restored'])
        self.assertEqual(result['indexed_count'], 1)
        self.assertEqual(result['version_count'], 1)
        self.assertTrue((self.output/'isolated'/'outputs'/'history_metadata'/'history.sqlite3').is_file())
        self.assertFalse((self.root/'outputs').exists())
        self.assertEqual(portable.read_bytes(), b'fixed source evidence')


if __name__ == '__main__':
    unittest.main()
