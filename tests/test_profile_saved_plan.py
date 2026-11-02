"""Small evidence checks; no geometry, real planning or performance workloads."""
from copy import deepcopy
import unittest

from tools.profile_saved_plan import STAGES, finalize_checks, measurement_errors


class ProfileSavedPlanTests(unittest.TestCase):
    def rows(self):
        return [dict(repetition=repetition, stage=stage, status='passed', sampling_errors=[])
                for repetition in range(2) for stage in STAGES]

    def test_complete_successful_measurements_pass(self):
        metadata = dict(status='passed', source_unchanged=True, code_unchanged=True)
        finalize_checks(metadata, [dict(all_checks_passed=True)]*2, self.rows(), 2)
        self.assertTrue(metadata['all_checks_passed'])
        self.assertTrue(metadata['measurement_checks_passed'])
        self.assertEqual(metadata['status'], 'passed')

    def test_sampler_failure_invalidates_otherwise_successful_run(self):
        rows = self.rows()
        rows[-1]['sampling_errors'] = ['AccessDenied while sampling RSS']
        metadata = dict(status='passed', source_unchanged=True, code_unchanged=True)
        finalize_checks(metadata, [dict(all_checks_passed=True)]*2, rows, 2)
        self.assertFalse(metadata['all_checks_passed'])
        self.assertFalse(metadata['measurement_checks_passed'])
        self.assertEqual(metadata['status'], 'failed')
        self.assertIn('RSS sampling failure', metadata['measurement_errors'][0])

    def test_missing_duplicate_extra_and_failed_stages_are_rejected(self):
        variants = []
        rows = self.rows()
        variants.append(rows[:-1])
        variants.append(rows+[dict(rows[0])])
        variants.append(rows+[dict(rows[0], repetition=2)])
        failed = deepcopy(rows)
        failed[-1]['status'] = 'failed'
        variants.append(failed)
        missing_sampling = deepcopy(rows)
        del missing_sampling[-1]['sampling_errors']
        variants.append(missing_sampling)
        for rows in variants:
            with self.subTest(rows=rows):
                self.assertTrue(measurement_errors(rows, 2))

    def test_final_source_code_and_readback_failures_cannot_keep_passed_status(self):
        for field in ('source_unchanged', 'code_unchanged', 'readback'):
            with self.subTest(field=field):
                metadata = dict(status='passed', source_unchanged=True, code_unchanged=True)
                equivalence = [dict(all_checks_passed=True), dict(all_checks_passed=True)]
                if field == 'readback':
                    equivalence[-1]['all_checks_passed'] = False
                else:
                    metadata[field] = False
                finalize_checks(metadata, equivalence, self.rows(), 2)
                self.assertFalse(metadata['all_checks_passed'])
                self.assertEqual(metadata['status'], 'failed')

    def test_existing_failure_detail_is_preserved(self):
        metadata = dict(status='failed', source_unchanged=True, code_unchanged=True,
                        error='original save failure', traceback='original traceback')
        finalize_checks(metadata, [], self.rows()[:2], 2)
        self.assertFalse(metadata['all_checks_passed'])
        self.assertEqual(metadata['error'], 'original save failure')
        self.assertEqual(metadata['traceback'], 'original traceback')


if __name__ == '__main__':
    unittest.main()
