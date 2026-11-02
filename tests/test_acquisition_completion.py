"""Pending targets survive manual confirmation; changed/stage models cannot plan."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from femto.dataset.acquisition_completion import (
    SCHEMA,read_completion,confirm_for_planning,portable_completion,validate_portable_completion)
from femto.io_utils import write_json,sha256_file


class CompletionTests(unittest.TestCase):
    def test_auto_limit_reason_survives_confirmation_without_clearing_pending_quality(self):
        self.record['capture_end_reason'] = 'accepted_view_limit'
        write_json(self.path, self.record)
        confirmed = confirm_for_planning(self.root, self.binding)
        reference = dict(session_id='session', reference_frame_id=0)
        restored = validate_portable_completion(portable_completion(confirmed, 'mesh', reference), 'mesh', reference)
        self.assertEqual(restored['capture_end_reason'], 'accepted_view_limit')
        self.assertEqual(restored['assessment_status'], 'needs_review_or_acquisition')
        self.assertEqual(restored['feedback']['pending_pixels'], 17)
        self.assertFalse(restored['execution_ready'])

    def test_portable_quality_binds_geometry_reference_and_retains_pending_without_source_folder(self):
        confirmed=confirm_for_planning(self.root,self.binding)
        reference=dict(session_id='session',reference_frame_id=0,image_size=[100,100])
        packed=portable_completion(confirmed,'portable_geometry',reference)
        restored=validate_portable_completion(packed,'portable_geometry',reference)
        self.assertEqual(restored['feedback']['pending_pixels'],17)
        self.assertEqual(restored['end_status'],'manual_end_pending')
        with self.assertRaisesRegex(ValueError,'model/reference'):
            validate_portable_completion(packed,'other_geometry',reference)
        damaged=copy.deepcopy(packed);damaged['completion']['feedback']['pending_pixels']=0
        with self.assertRaisesRegex(ValueError,'content changed'):
            validate_portable_completion(damaged,'portable_geometry',reference)
        with self.assertRaisesRegex(ValueError,'model/reference'):
            portable_completion(self.record,'portable_geometry',reference)

    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)/'result'
        self.mesh=self.root/'reconstruction/scene_tsdf_mesh_m.ply'
        self.mesh.parent.mkdir(parents=True);self.mesh.write_bytes(b'fixture mesh version one')
        self.report=self.mesh.parent/'reconstruction_report.json'
        write_json(self.report,dict(local_capture=dict(session_id='session',reference_frame_id=0),
            outputs=dict(mesh=dict(ply=str(self.mesh)))))
        write_json(self.root/'demo_summary.json',dict(acquisition_completion_required=True))
        self.binding=dict(session_id='session',reference_frame_id=0,
                          model_sha256=sha256_file(self.mesh),stage_id='final',target_key='target')
        self.record=dict(schema=SCHEMA,binding=self.binding,assessment_status='needs_review_or_acquisition',
            end_status='manual_end_pending',confirmed_for_planning=False,
            feedback=dict(binding=self.binding,status='needs_review_or_acquisition',pending_pixels=17),
            reconstruction_report_sha256=sha256_file(self.report),
            execution_ready=False,physical_accuracy_verified=False)
        self.path=self.root/'acquisition_completion.json'
        write_json(self.path,self.record)
        write_json(self.root/'publication.json',dict(status='published',session_id='session',result=str(self.root)))

    def test_manual_confirmation_preserves_pending_and_quality(self):
        with self.assertRaisesRegex(ValueError,'先检查并确认'):
            read_completion(self.root,require_confirmed=True)
        confirmed=confirm_for_planning(self.root,self.binding)
        self.assertEqual(confirmed['end_status'],'manual_end_pending')
        self.assertEqual(confirmed['feedback']['pending_pixels'],17)
        self.assertEqual(confirmed['assessment_status'],'needs_review_or_acquisition')
        self.assertTrue(read_completion(self.mesh.parent,require_confirmed=True)['confirmed_for_planning'])
        self.assertFalse(confirmed['execution_ready'])

    def test_readonly_loader_can_review_unconfirmed_but_planning_stays_blocked(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from femto.app.qt_planning_io import load_source
        model=SimpleNamespace(task_context={'name':'test'},view_reference={})
        original=self.path.read_bytes()
        with patch('femto.planning.saved_source._load_source',return_value=model), patch(
                'femto.planning.saved_source.verified_capture_source',return_value=self.mesh.parent):
            with self.assertRaisesRegex(ValueError,'先检查并确认'):
                load_source(self.root,None,lambda _:None)
            loaded=load_source(self.root,None,lambda _:None,require_confirmed=False)
        self.assertFalse(loaded.acquisition_completion['confirmed_for_planning'])
        self.assertEqual(self.path.read_bytes(),original)

    def test_changed_model_or_wrong_requested_binding_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'另一个模型'):
            confirm_for_planning(self.root,dict(self.binding,session_id='different'))
        original=self.path.read_bytes()
        self.mesh.write_bytes(b'changed geometry')
        with self.assertRaisesRegex(ValueError,'版本不一致'):
            confirm_for_planning(self.root,self.binding)
        self.assertEqual(self.path.read_bytes(),original)

    def test_missing_new_receipt_and_stage_checkpoint_cannot_plan(self):
        self.path.unlink()
        with self.assertRaisesRegex(ValueError,'缺少最终'):
            read_completion(self.root,require_confirmed=True)
        report=json.loads(self.report.read_text())
        report['local_capture']['capture_scope']='stage_checkpoint';write_json(self.report,report)
        with self.assertRaisesRegex(ValueError,'阶段模型'):
            read_completion(self.root)

    def test_unknown_quality_and_legacy_do_not_turn_into_completed_targets(self):
        record=dict(self.record,assessment_status='cannot_determine',end_status='manual_end_unknown',
            feedback=dict(binding=self.binding,status='cannot_determine'))
        write_json(self.path,record)
        confirmed=confirm_for_planning(self.root,self.binding)
        self.assertEqual(confirmed['end_status'],'manual_end_unknown')
        write_json(self.root/'demo_summary.json',{})
        self.path.unlink()
        self.assertIsNone(read_completion(self.root,require_confirmed=True))

    def test_cancelled_or_unpublished_confirmation_keeps_old_bytes(self):
        original=self.path.read_bytes()
        def cancel():raise RuntimeError('cancelled')
        with self.assertRaisesRegex(RuntimeError,'cancelled'):
            confirm_for_planning(self.root,self.binding,check=cancel)
        self.assertEqual(self.path.read_bytes(),original)
        write_json(self.root/'publication.json',dict(status='pending',session_id='session',result=str(self.root)))
        with self.assertRaisesRegex(ValueError,'尚未成功发布'):
            confirm_for_planning(self.root,self.binding)
        self.assertEqual(self.path.read_bytes(),original)

    def test_forged_completion_status_and_nonboolean_confirmation_reject(self):
        record=copy.deepcopy(self.record);record['end_status']='reference_support_met'
        write_json(self.path,record)
        with self.assertRaisesRegex(ValueError,'不能改写'):
            read_completion(self.root)
        record=copy.deepcopy(self.record);record['confirmed_for_planning']=1
        write_json(self.path,record)
        with self.assertRaisesRegex(ValueError,'版本不一致'):
            read_completion(self.root)


if __name__=='__main__':unittest.main()
