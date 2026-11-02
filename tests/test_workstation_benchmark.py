"""Benchmark errors must remain evidence rather than disappear from reports."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from tools.benchmark_workstation import Measurements, file_digest, inventory
from tools.audit_workstation_benchmark import audit, compare, main, STAGES


class WorkstationBenchmarkTests(unittest.TestCase):
    @staticmethod
    def write_plan(path, *, route=0, array=0, summary=1, dtype='float64',
                   shape=(2, 3), coverage=False):
        target = io.BytesIO()
        np.savez(target, positions=np.arange(6, dtype=dtype).reshape(shape) + array)
        document = dict(plan=dict(
            surface_trajectory=dict(ordered_items=[dict(position=[route, 0, 0])]),
            planning_context=dict(target=dict(summary=dict(area_m2=summary))),
            coverage_evidence={'status': 'verified'} if coverage else None))
        with zipfile.ZipFile(path, 'w') as archive:
            archive.writestr('project.json', json.dumps(document))
            archive.writestr('target_surface.npz', target.getvalue())

    def make_pair(self, root):
        project = root/'original.region.zip'
        self.write_plan(project, route=9, array=9, summary=9, coverage=True)
        baseline, optimized = root/'baseline', root/'optimized'
        for folder in (baseline, optimized):
            folder.mkdir()
            manifest = dict(status='passed', source_unchanged=True, code_unchanged=True,
                            repetitions=2, project=str(project), project_sha256=file_digest(project),
                            reconstruction_report_sha256='a'*64,
                            input_files={'dataset': {'rgb/0.png': 'b'*64}},
                            code_sha256={'femto/example.py': 'c'*64})
            if folder == optimized:
                manifest['code_sha256']['femto/example.py'] = 'd'*64
            (folder/'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            rows = [dict(repetition=r, stage=s, status='passed', sampling_errors=[])
                    for r in range(2) for s in STAGES]
            (folder/'stages.json').write_text(json.dumps(rows), encoding='utf-8')
            (folder/'equivalence.json').write_text(json.dumps([
                dict(repetition=r, saved_route_equal=True) for r in range(2)]), encoding='utf-8')
            for repetition in range(2):
                run = folder/f'run_{repetition}'
                (run/'reconstruction').mkdir(parents=True)
                (run/'reconstruction'/'scene_tsdf_mesh_m.ply').write_bytes(b'fixed mesh')
                self.write_plan(run/'result.region.zip')
        return baseline, optimized

    def test_cross_run_compares_outputs_and_allows_changed_code(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline, optimized = self.make_pair(Path(directory))
            report = compare(baseline, optimized)
            self.assertTrue(report['all_checks_passed'], report['errors'])
            self.assertEqual(len(report['round_comparisons']), 2)
            self.assertTrue(all(row['all_equal'] for row in report['round_comparisons']))
            self.assertEqual(report['changed_production_files'], ['femto/example.py'])
            # The saved original includes coverage and intentionally differs
            # from newly generated plans. Compare corresponding outputs only.
            self.assertNotEqual(report['baseline_audit']['original_project']['ordered_route_sha256'],
                                report['baseline_audit']['repetitions'][0]['ordered_route_sha256'])

    def test_cross_run_rejects_same_within_run_but_different_outputs(self):
        cases = [
            ('mesh', 'reconstructed_mesh_file_sha256', {}),
            ('route', 'ordered_route_sha256', {'route': 1}),
            ('array', 'target_arrays', {'array': 1}),
            ('dtype', 'target_arrays', {'dtype': 'int64'}),
            ('shape', 'target_arrays', {'shape': (3, 2)}),
            ('summary', 'target_summary', {'summary': 2}),
            ('coverage', 'has_coverage', {'coverage': True}),
        ]
        for name, field, changes in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                baseline, optimized = self.make_pair(Path(directory))
                for repetition in range(2):
                    run = optimized/f'run_{repetition}'
                    if name == 'mesh':
                        (run/'reconstruction'/'scene_tsdf_mesh_m.ply').write_bytes(b'changed mesh')
                    else:
                        self.write_plan(run/'result.region.zip', **changes)
                self.assertTrue(audit(optimized)['all_checks_passed'])
                report = compare(baseline, optimized)
                self.assertFalse(report['all_checks_passed'])
                self.assertTrue(all(not row['checks'][field] for row in report['round_comparisons']))
                if field == 'target_arrays':
                    self.assertEqual(report['round_comparisons'][0]['differing_target_arrays'],
                                     ['positions'])

    def test_cross_run_checks_later_repetitions(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline, optimized = self.make_pair(Path(directory))
            self.write_plan(optimized/'run_1'/'result.region.zip', array=1)
            report = compare(baseline, optimized)
            self.assertFalse(report['all_checks_passed'])
            self.assertTrue(report['round_comparisons'][0]['all_equal'])
            self.assertFalse(report['round_comparisons'][1]['checks']['target_arrays'])
            self.assertFalse(report['optimized_audit']['repetitions_equivalent'])

    def test_cross_run_requires_matching_input_hashes_and_complete_inventory(self):
        for key in ('project_sha256', 'reconstruction_report_sha256', 'input_files'):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                baseline, optimized = self.make_pair(Path(directory))
                manifest = json.loads((optimized/'manifest.json').read_text())
                manifest[key] = ({'dataset': {'rgb/0.png': 'b'*64, 'extra.png': 'e'*64}}
                                 if key == 'input_files' else 'e'*64)
                (optimized/'manifest.json').write_text(json.dumps(manifest))
                report = compare(baseline, optimized)
                self.assertFalse(report['all_checks_passed'])
                self.assertFalse(report['fixed_inputs'][key]['equal'])
                self.assertTrue(all(row['all_equal'] for row in report['round_comparisons']))
        with tempfile.TemporaryDirectory() as directory:
            baseline, optimized = self.make_pair(Path(directory))
            for folder in (baseline, optimized):
                manifest = json.loads((folder/'manifest.json').read_text())
                del manifest['input_files']
                (folder/'manifest.json').write_text(json.dumps(manifest))
            self.assertFalse(compare(baseline, optimized)['fixed_inputs']['input_files']['equal'])

    def test_cross_run_rejects_incomplete_and_mismatched_repetitions(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline, optimized = self.make_pair(Path(directory))
            rows = json.loads((optimized/'stages.json').read_text())
            (optimized/'stages.json').write_text(json.dumps(rows[:-1]))
            report = compare(baseline, optimized)
            self.assertFalse(report['all_checks_passed'])
            self.assertFalse(report['optimized_audit']['all_checks_passed'])
            manifest = json.loads((optimized/'manifest.json').read_text())
            manifest['repetitions'] = 1
            (optimized/'manifest.json').write_text(json.dumps(manifest))
            for name in ('stages.json', 'equivalence.json'):
                values = json.loads((optimized/name).read_text())
                (optimized/name).write_text(json.dumps([v for v in values if v['repetition'] == 0]))
            report = compare(baseline, optimized)
            self.assertFalse(report['all_checks_passed'])
            self.assertFalse(report['repetition_counts_equal'])
            self.assertFalse(report['round_comparisons'][1]['all_equal'])

    def test_cross_run_requires_unchanged_sources_code_and_declared_repetitions(self):
        for key in ('source_unchanged', 'code_unchanged'):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                baseline, optimized = self.make_pair(Path(directory))
                manifest = json.loads((optimized/'manifest.json').read_text())
                manifest[key] = False
                (optimized/'manifest.json').write_text(json.dumps(manifest))
                report = compare(baseline, optimized)
                self.assertFalse(report['all_checks_passed'])
                self.assertTrue(any(key in error for error in report['errors']))
        for name in ('stages.json', 'equivalence.json'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                baseline, optimized = self.make_pair(Path(directory))
                rows = json.loads((optimized/name).read_text())
                rows.append(dict(rows[0], repetition=99))
                (optimized/name).write_text(json.dumps(rows))
                report = compare(baseline, optimized)
                self.assertFalse(report['all_checks_passed'])
                self.assertTrue(any('undeclared repetition' in error for error in report['errors']))

    def test_comparison_cli_reports_missing_output_and_preserves_single_run_usage(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline, optimized = self.make_pair(Path(directory))
            with patch('builtins.print'):
                self.assertEqual(main([str(baseline)]), 0)
            self.assertTrue(json.loads((baseline/'audit.json').read_text())['all_checks_passed'])
            (optimized/'run_1'/'result.region.zip').unlink()
            report_path = optimized/'missing-output-comparison.json'
            with patch('builtins.print'):
                self.assertEqual(main([str(optimized), '--baseline', str(baseline),
                                       '--report', str(report_path)]), 1)
            report = json.loads(report_path.read_text())
            self.assertFalse(report['all_checks_passed'])
            self.assertIn('FileNotFoundError', report['optimized_audit']['errors'][0])

    def test_audit_rejects_missing_stage_and_changed_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'manifest.json').write_text(json.dumps(dict(status='passed', source_unchanged=True,
                code_unchanged=True, repetitions=2, project='source.zip')), encoding='utf-8')
            rows = [dict(repetition=r, stage=s, status='passed', sampling_errors=[])
                    for r in range(2) for s in STAGES]
            (root/'stages.json').write_text(json.dumps(rows[:-1]), encoding='utf-8')
            (root/'equivalence.json').write_text(json.dumps([
                dict(repetition=r, saved_route_equal=True) for r in range(2)]), encoding='utf-8')
            with patch('tools.audit_workstation_benchmark.plan_values', side_effect=lambda path: {}), \
                    patch('tools.audit_workstation_benchmark.file_digest', side_effect=['first', 'changed']):
                report = audit(root)
            self.assertFalse(report['all_checks_passed'])
            self.assertFalse(report['repetitions_equivalent'])
            self.assertEqual(len(report['errors']), 2)

    def test_failure_retains_success_and_memory_observation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            measurements = Measurements(root)
            self.assertEqual(measurements.run('ok', 0, lambda: 42), 42)
            def fail():
                raise ValueError('deliberate failed stage')
            with self.assertRaisesRegex(ValueError, 'deliberate failed stage'):
                measurements.run('failed', 0, fail)
            rows = json.loads((root/'stages.json').read_text(encoding='utf-8-sig'))
            self.assertEqual([row['status'] for row in rows], ['passed', 'failed'])
            self.assertIn('deliberate failed stage', rows[-1]['error'])
            for row in rows:
                self.assertGreaterEqual(row['seconds'], 0)
                self.assertGreater(row['sampled_rss_peak_bytes'], 0)
                self.assertEqual(row['sampling_errors'], [])

    def test_inventory_detects_change_removal_and_addition(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/'证据.bin'
            path.write_bytes(b'original')
            before = inventory(root)
            self.assertEqual(before['证据.bin'], file_digest(path))
            path.write_bytes(b'changed!')
            self.assertNotEqual(before, inventory(root))
            path.unlink()
            self.assertNotEqual(before, inventory(root))
            (root/'new.bin').write_bytes(b'original')
            self.assertNotEqual(before, inventory(root))


if __name__ == '__main__':
    unittest.main()
