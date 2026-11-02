import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest import mock

from femto.reports.local_comparison_status import comparison_source, progress_text


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


class ComparisonReportTests(unittest.TestCase):
    def test_ancestry_uses_original_recorded_input_and_rejects_cross_session(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original, refined, filtered, result = [root / s for s in ('original', 'refined', 'filtered', 'result')]
            local = dict(measured_only=True, session_id='same', reference_frame_id=0)
            write(original / 'manifest.json', dict(local_capture=local, pose=dict(source='online_rgbd_research_grade')))
            write(refined / 'manifest.json', dict(local_capture=local, refinement_source_dataset=str(original)))
            write(root / 'temporal.json', dict(source_dataset=str(refined)))
            write(filtered / 'manifest.json', dict(local_capture=dict(local, temporal_support_report=str(root / 'temporal.json'))))
            write(result / 'reconstruction/reconstruction_report.json', dict(source_dataset=str(filtered)))
            write(result / 'reconstruction/local_geometry.json', dict(session_id='same', reference_frame_id=0))
            self.assertEqual(comparison_source(result), original)
            write(original / 'manifest.json', dict(local_capture=dict(local, session_id='other'), pose=dict(source='online_rgbd_research_grade')))
            with self.assertRaisesRegex(ValueError, 'crosses'):
                comparison_source(result)
            write(result / 'publication.json', dict(status='cancelled'))
            with self.assertRaisesRegex(ValueError, 'not published'):
                comparison_source(result)






    def test_status_reports_reused_geometry_and_missing_evidence_without_fabricated_zero(self):
        report = dict(state='complete', variants=[dict(name='B', optimization=dict(status='not_adopted'), same_as='A')])
        self.assertIn('几何与 A', progress_text(report))
        self.assertIn('Not recorded', progress_text(report, 'en'))
        self.assertNotIn('0.0', progress_text(report))


if __name__ == '__main__':
    unittest.main()
