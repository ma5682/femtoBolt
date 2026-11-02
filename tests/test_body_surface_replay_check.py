"""Acceptance reports distinguish replay from field evidence and preserve input files."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import subprocess
import unittest
from unittest.mock import patch

from tools.check_body_surface_replay import evidence_paths, fingerprints, make_report, run_worker, main, enumerate_devices


class BodySurfaceReplayChecks(unittest.TestCase):
    def write(self,path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value),encoding='utf-8');return path

    def test_declared_derived_dataset_can_reference_immutable_original_frames(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);original=root/'session';derived=original/'derived'/'local_posed';model=root/'result'/'reconstruction'
            self.write(original/'manifest.json',{'schema':'femto-full-scene-rgbd-v1'})
            self.write(derived/'manifest.json',{'source_dataset':str(original)})
            self.write(model/'reconstruction_report.json',{'source_dataset':str(derived)})
            self.write(original/'rgb'/'0.png',{'fixture':1});self.write(original/'depth'/'0.png',{'fixture':2})
            self.write(derived/'pose'/'0.json',{'fixture':3})
            row=dict(rgb='../../rgb/0.png',depth='../../depth/0.png',pose='pose/0.json')
            self.write(derived/'frames.jsonl',row)
            paths=evidence_paths(model);before=fingerprints(paths)
            self.assertIn(original/'rgb'/'0.png',paths)
            self.assertIsNone(before[str(model.parent/'task_context.json')])
            self.assertEqual(before,fingerprints(paths))
            self.write(original/'rgb'/'0.png',{'changed':True})
            self.assertNotEqual(before,fingerprints(paths))
            row['rgb']='../../../outside.png';self.write(derived/'frames.jsonl',row)
            with self.assertRaises(ValueError):evidence_paths(model)

    def test_successful_replay_never_certifies_field_accuracy_or_missing_body(self):
        rows=[dict(source='old.zip',replay=dict(status='passed',confirmed_body_available=False,
            review_required=True),native=dict(status='passed'))]
        report=make_report({'ready':True},{'status':'detected'},rows)
        self.assertEqual(report['automatic_status'],'passed')
        self.assertEqual(report['field_acceptance'],'not_run')
        self.assertFalse(report['new_capture_performed'])
        self.assertFalse(report['physical_accuracy_verified']);self.assertFalse(report['execution_ready'])
        self.assertFalse(report['records'][0]['replay']['confirmed_body_available'])
        self.assertTrue(report['pending'])

    def test_no_data_failed_environment_or_native_render_cannot_pass(self):
        rows=[dict(replay=dict(status='passed'),native=dict(status='failed'))]
        self.assertEqual(make_report({'ready':True},{'status':'detected'},rows)['automatic_status'],'failed')
        self.assertEqual(make_report({'ready':True},{'status':'detected'},[])['automatic_status'],'failed')
        rows[0]['native']['status']='passed'
        self.assertEqual(make_report({'ready':False},{'status':'detected'},rows)['automatic_status'],'failed')
        self.assertEqual(make_report({'ready':True},{'status':'failed'},rows)['automatic_status'],'failed')

    def test_worker_crash_timeout_or_invalid_payload_is_not_a_pass(self):
        for response in [SimpleNamespace(returncode=1,stdout='',stderr='crashed'),
                         SimpleNamespace(returncode=0,stdout='noise',stderr=''),
                         SimpleNamespace(returncode=0,stdout='BODY_REPLAY_RESULT=[]',stderr='')]:
            with patch('tools.check_body_surface_replay.subprocess.run',return_value=response):
                self.assertEqual(run_worker('devices')['status'],'failed')
        with patch('tools.check_body_surface_replay.subprocess.run',side_effect=subprocess.TimeoutExpired('probe',20)):
            self.assertEqual(run_worker('devices')['status'],'failed')

    def test_report_output_cannot_overwrite_input_or_previous_report(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'record';source.mkdir()
            with self.assertRaises(SystemExit) as error:
                main(['--source',str(source),'--output',str(source/'report')])
            self.assertEqual(error.exception.code,2)
            self.assertFalse((source/'report').exists())
            output=root/'old-report';output.mkdir();(output/'report.json').write_bytes(b'preserve')
            with self.assertRaises(FileExistsError):main(['--source',str(source),'--output',str(output)])
            self.assertEqual((output/'report.json').read_bytes(),b'preserve')

    def test_device_probe_only_enumerates_and_does_not_start_pipeline(self):
        info=SimpleNamespace(get_name=lambda:'fixture camera')
        device=SimpleNamespace(get_device_info=lambda:info)
        devices=SimpleNamespace(get_count=lambda:1,get_device_by_index=lambda index:device)
        with patch('femto.hardware.orbbec_rgbd.Context',return_value=SimpleNamespace(query_devices=lambda:devices)),patch(
                'femto.hardware.orbbec_rgbd.Pipeline',side_effect=AssertionError('must not start capture')):
            report=enumerate_devices()
        self.assertEqual(report['status'],'detected');self.assertFalse(report['stream_started'])


if __name__=='__main__':unittest.main()
