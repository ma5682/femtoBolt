import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
from tests.test_scale_validation import _build_reconstruction
from scene_quality_report import SPEC_SCHEMA, build_report, evaluate_field_spec, write_report, render_report, dump

class ResultQualityTests(unittest.TestCase):
    def trial(self,root,index,length=.3,kind='Orbbec Femto Bolt'):
        base=root/str(index)
        reconstruction=_build_reconstruction(base,np.array([[0,0,1],[length,0,1]]))
        raw=base/'dataset';raw.mkdir()
        dump(raw/'manifest.json',dict(camera=dict(device=kind),frame_count=1))
        (raw/'rgb').write_bytes(f'rgb{index}'.encode());(raw/'depth').write_bytes(f'depth{index}'.encode())
        (raw/'frames.jsonl').write_text(json.dumps(dict(rgb='rgb',depth='depth'))+'\n')
        return dict(reconstruction=str(reconstruction),geometry_source='fused_point_cloud',measurements=[dict(target_id='bed',point_a_world_m=[0,0,1],point_b_world_m=[length,0,1])])
    def spec(self,trials):
        return dict(schema=SPEC_SCHEMA,targets=[dict(id='bed',ground_truth_distance_mm=300,ground_truth_source='Test ruler fixture',absolute_tolerance_mm=2,repeatability_range_tolerance_mm=2)],trials=trials)
    def test_missing_reference_and_tolerance_stay_pending(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);trial=self.trial(root,0)
            spec=self.spec([trial]);spec['targets'][0]['ground_truth_distance_mm']=None
            result=evaluate_field_spec(spec,root)
            self.assertEqual(result['status'],'PENDING');self.assertIsNone(result['accepted'])
    def test_three_independent_trials_pass_scoped_distance_checks(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);trials=[self.trial(root,i,x) for i,x in enumerate([.3,.301,.2995])]
            result=evaluate_field_spec(self.spec(trials),root)
            self.assertEqual(result['status'],'PASS')
            self.assertAlmostEqual(result['targets'][0]['distance_range_mm'],1.5,places=3)
            self.assertTrue(all(r['source_geometry_sha256'] for r in result['measurements']))
    def test_out_of_tolerance_fails_even_if_repeats_missing(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);result=evaluate_field_spec(self.spec([self.trial(root,0,.31)]),root)
            self.assertEqual(result['status'],'FAIL');self.assertFalse(result['accepted'])
    def test_duplicate_raw_capture_cannot_fake_repeatability(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);trial=self.trial(root,0)
            with self.assertRaisesRegex(ValueError,'Repeated raw'):
                evaluate_field_spec(self.spec([trial,trial,trial]),root)
    def test_synthetic_not_allowed_as_physical_acceptance(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            with self.assertRaisesRegex(ValueError,'real Femto'):
                evaluate_field_spec(self.spec([self.trial(root,0,kind='synthetic_test_camera')]),root)
    def test_no_snap_in_empty_space_and_invalid_tolerance(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);trial=self.trial(root,0)
            trial['measurements'][0]['point_b_world_m']=[4,4,4]
            with self.assertRaisesRegex(ValueError,'No reconstructed surface'):
                evaluate_field_spec(self.spec([trial]),root)
            spec=self.spec([trial]);spec['targets'][0]['absolute_tolerance_mm']=-1
            with self.assertRaises(ValueError):evaluate_field_spec(spec,root)
    def test_unknown_metrics_pending_and_html_escapes_source(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);trial=self.trial(root,0)
            output,report=build_report(Path(trial['reconstruction']))
            self.assertIsNone(report['tracking']['cycle_translation_mm'])
            self.assertEqual(report['physical_acceptance']['status'],'PENDING')
            report['source_dataset']='<script>alert(1)</script>'
            self.assertNotIn('<script>',render_report(report))
            path=write_report(Path(trial['reconstruction']))
            self.assertTrue(path.exists());self.assertTrue((output/'field_acceptance_spec.json').exists())
    def test_acceptance_cannot_attach_to_unrelated_model(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);trials=[self.trial(root,i) for i in range(4)]
            path=root/'spec.json';dump(path,self.spec(trials[:3]))
            with self.assertRaisesRegex(ValueError,'do not include'):
                build_report(Path(trials[3]['reconstruction']),path)
    def test_failure_report_does_not_claim_reconstruction(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);dump(root/'pose_quality_report.json',dict(status='BLOCKED',error='lost',source_frame_count=8))
            _,report=build_report(root)
            self.assertEqual(report['pipeline_status'],'BLOCKED')
            self.assertEqual(report['tracking']['keyframes'],0)
            self.assertIsNone(report['physical_acceptance']['accepted'])
