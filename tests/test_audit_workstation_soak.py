"""Audit gates use small manifests, without running geometry or a soak."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import audit_workstation_soak as subject


class SoakAuditTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.generated = dict(ordered_route_sha256=subject.canonical_digest([]))
        self.cycles = [dict(cycle=i, status='passed', result=dict(
            equivalence=dict(generated=self.generated, reconstruction=None),
            assets=dict(portable=str(self.root/f'{i}.zip'), native=str(self.root/f'native{i}'))))
            for i in range(8)]
        self.manifest = dict(status='completed', sustained_hour_qualified=True,
            functional_checks_passed=True, source_unchanged=True, code_unchanged=True,
            forced_gc=False, active_cycle_seconds=3601, warmup_cycles=3,
            completed_cycles=8, expected_stages=['work', 'cleanup_idle'])
        self.stages = [dict(repetition=i, stage=name, status='passed', sampling_errors=[])
                       for i in range(8) for name in self.manifest['expected_stages']]
        self.inputs = dict(files={}, datasets={}, production_code={}, tools={})

    def run_audit(self, *, bad_portable=False, bad_native=False):
        data = dict(manifest=self.manifest, cycles=self.cycles, stages=self.stages,
                    inputs=self.inputs, inputs_after=self.inputs, checkpoint=self.cycles[-1])
        def read(path):
            if path.name == 'state.json':
                return self.cycles[int(path.parent.name.split('_')[1])]
            if path.name == 'trajectory.json':
                return dict(ordered_items=[])
            return data[path.stem]
        with patch.object(subject, 'ROOT', self.root), patch.object(subject, 'read', side_effect=read), \
                patch.object(subject, 'plan_values', return_value={} if bad_portable else self.generated), \
                patch.object(subject, 'archive_values', return_value=({}, {'evidence': 'fixed'})), \
                patch.object(subject, 'folder_values', return_value={} if bad_native else {'evidence': 'fixed'}):
            return subject.audit(self.root)

    def test_complete_run_passes_without_claiming_no_leaks(self):
        report = self.run_audit()
        self.assertTrue(report['all_checks_passed'])
        self.assertEqual(report['cycles_checked'], 8)
        self.assertIn('separately', report['memory_interpretation'])

    def test_short_or_incomplete_run_cannot_pass(self):
        for field, value in [('active_cycle_seconds', 3599), ('status', 'running'),
                             ('sustained_hour_qualified', False), ('completed_cycles', 9)]:
            with self.subTest(field=field):
                original = deepcopy(self.manifest)
                self.manifest[field] = value
                self.assertFalse(self.run_audit()['all_checks_passed'])
                self.manifest = original

    def test_missing_stage_and_sampler_error_cannot_pass(self):
        self.stages.pop()
        self.assertFalse(self.run_audit()['all_checks_passed'])
        self.stages[-1]['sampling_errors'] = ['sampler failed']
        self.assertTrue(any('measured stage' in error for error in self.run_audit()['errors']))

    def test_changed_retained_portable_or_native_evidence_is_rejected(self):
        self.assertFalse(self.run_audit(bad_portable=True)['all_checks_passed'])
        self.assertFalse(self.run_audit(bad_native=True)['all_checks_passed'])


if __name__ == '__main__':
    unittest.main()
