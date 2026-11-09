"""Real offline runner: source binding, validation, all-view fallback, cancellation."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from femto.io_utils import sha256_file
from femto.local.organized_comparison import compare,ComparisonCancelled,analytic_normal_comparison
from femto.local.organized_fixture import generate


class OrganizedComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def test_real_two_view_report_keeps_inputs_and_disjoint_evidence(self):
        source=generate(self.root/'source',width=160,height=120,frames=2)
        original={p:sha256_file(p) for p in source.rglob('*') if p.is_file()}
        report=compare(source,self.root/'comparison',scale=1.)
        self.assertEqual(report['state'],'complete');self.assertTrue(report['source_unchanged'])
        self.assertFalse(report['published']);self.assertFalse(report['default_pipeline_changed'])
        self.assertEqual(len(report['pairs']),1)
        self.assertEqual({r['method'] for r in report['pairs'][0]['methods']},
                         {'baseline_icp','projective','normal_space_icp'})
        for frame in report['frames']:
            with np.load(frame['artifact']) as evidence:
                fit=evidence['fit_mask'];held=evidence['held_out_mask'];valid=evidence['normal_valid']
                self.assertFalse(np.any(fit&held));self.assertTrue(fit.any());self.assertTrue(held.any())
                self.assertTrue(np.isnan(evidence['normals'][~valid]).all())
            self.assertEqual(sha256_file(frame['artifact']),frame['artifact_sha256'])
        self.assertEqual(original,{p:sha256_file(p) for p in source.rglob('*') if p.is_file()})
        self.assertFalse((self.root/'latest_complete_demo.txt').exists())

    def test_analytic_normal_report_has_known_truth_and_noise_stability(self):
        rows=analytic_normal_comparison(width=160,height=120)
        self.assertEqual({row['surface'] for row in rows},{'plane','sphere'})
        for case in rows:
            self.assertEqual(case['injected_noise_sigma_mm'],.2)
            for run in case['runs']:
                self.assertGreater(run['valid_normal_fraction'],.7)
                self.assertLess(run['integral_error_vs_truth']['p95_deg'],2.)
                self.assertIsNotNone(run['kd_error_on_integral_support']['p95_deg'])
            self.assertIsNotNone(case['runs'][1]['integral_noise_repeat_difference'])

    def test_all_methods_use_existing_approval_and_cannot_override_rejection(self):
        source=generate(self.root/'source',width=160,height=120,frames=2)
        with patch('femto.local.organized_comparison.approve',return_value=['existing_gate_rejected']) as gate:
            report=compare(source,self.root/'comparison',scale=1.)
        self.assertEqual(gate.call_count,3)
        for method in report['pairs'][0]['methods']:
            self.assertFalse(method['would_pass_existing_gate']);self.assertIn('existing_gate_rejected',method['reasons'])
        self.assertTrue(all(not row['would_adopt_all_views'] for row in report['all_view_decisions'].values()))

    def test_all_view_decision_rejects_mixture(self):
        source=generate(self.root/'source',width=160,height=120,frames=3)
        calls=[0]
        def pair(method,*args):
            calls[0]+=1
            return dict(method=method,would_pass_existing_gate=calls[0]<=3,reasons=[])
        with patch('femto.local.organized_comparison._pair',side_effect=pair):
            report=compare(source,self.root/'comparison',scale=1.)
        for row in report['all_view_decisions'].values():
            self.assertFalse(row['would_adopt_all_views'])
            self.assertTrue(row['failed_view_keeps_entire_original_pose_set'])
            self.assertFalse(row['published'])

    def test_single_view_skips_registration_and_output_cannot_touch_source(self):
        source=generate(self.root/'source',width=160,height=120,frames=1)
        report=compare(source,self.root/'comparison',scale=1.)
        self.assertEqual(report['pairs'],[])
        self.assertTrue(all(not row['would_adopt_all_views'] for row in report['all_view_decisions'].values()))
        with self.assertRaises(ValueError):compare(source,source/'bad')
        with self.assertRaises(FileExistsError):compare(source,self.root/'comparison')

    def test_cancellation_and_invalid_pose_do_not_publish_success(self):
        source=generate(self.root/'source',width=160,height=120,frames=2)
        count=[0]
        def check():
            count[0]+=1
            if count[0]==2:raise RuntimeError('cancelled by test')
        with self.assertRaises(ComparisonCancelled):compare(source,self.root/'cancelled',check=check)
        report=json.loads((self.root/'cancelled/comparison.json').read_text())
        self.assertEqual(report['state'],'cancelled');self.assertFalse(report['published'])
        pose=source/'pose/000001.json';record=json.loads(pose.read_text());record['pose_valid']=False
        pose.write_text(json.dumps(record))
        with self.assertRaises(ValueError):compare(source,self.root/'invalid')
        self.assertFalse((self.root/'invalid').exists())

    def test_changed_source_is_rejected_at_final_binding_check(self):
        source=generate(self.root/'source',width=160,height=120,frames=2)
        changed=[False]
        def progress(message):
            if message.startswith('projective:') and not changed[0]:
                pose=source/'pose/000001.json';pose.write_bytes(pose.read_bytes()+b' ');changed[0]=True
        with self.assertRaisesRegex(ValueError,'Source changed'):
            compare(source,self.root/'changed',scale=1.,progress=progress)
        report=json.loads((self.root/'changed/comparison.json').read_text())
        self.assertEqual(report['state'],'failed');self.assertFalse(report['source_unchanged'])


if __name__=='__main__':unittest.main()
