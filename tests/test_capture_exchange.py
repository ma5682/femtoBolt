"""Manual supplement observations retain request-time task/model and raw receipts."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import threading
import unittest
import numpy as np
from femto.capture.capture_exchange import current_tasks,record_manual_observation


class CaptureExchangeTests(unittest.TestCase):
    def test_request_and_actual_gate_result_are_persisted_without_faking_task_resolution(self):
        with TemporaryDirectory() as directory:
            binding=dict(stage_id='one',session_id='session',reference_frame_id=0,
                target_key='b'*64,model_sha256='c'*64)
            feedback=dict(binding=binding,tasks=[dict(task_id='one:3')])
            manager=SimpleNamespace(lock=threading.Lock(),last_feedback=(feedback,None))
            capture=SimpleNamespace(checkpoints=manager,session=Path(directory),calibration={'sha256':'a'*64})
            for child in ('rgb','depth','derived'):(capture.session/child).mkdir()
            for child in ('rgb','depth'):(capture.session/child/'000004.png').write_bytes(b'committed fixture')
            task=current_tasks(capture)
            manager.last_feedback=({'binding':dict(binding,stage_id='two'), 'tasks':[]},None)
            record=dict(frame_id=4,session_id='session',reference_frame_id=0,calibration_sha256='a'*64,
                capture_timestamp_us=100,color_timestamp_us=100,depth_timestamp_us=100,
                T_WC=np.eye(4).tolist(),state='tracking',selected_for_fusion=True,
                quality={'usable':True},continuity={'current_frame_verified':True})
            record['raw_write_receipt']={key:record[key] for key in
                ('frame_id','capture_timestamp_us','color_timestamp_us','depth_timestamp_us')}
            result=record_manual_observation(capture,task,record)
            self.assertTrue(result['accepted_for_fusion'])
            self.assertEqual(result['source_binding']['stage_id'],'one')
            self.assertIsNone(result['target_improvement'])
            row=json.loads((capture.session/'derived/manual_supplement_exchanges.jsonl').read_text())
            self.assertEqual(row['observation']['requested_task_ids'],['one:3'])
            self.assertFalse(row['observation']['accepted_for_fusion'])
            self.assertEqual(row['verification']['frame_id'],4)
            record['selected_for_fusion']=False
            self.assertFalse(record_manual_observation(capture,task,record)['accepted_for_fusion'])

    def test_before_first_stage_has_no_assumed_tasks(self):
        self.assertIsNone(current_tasks(SimpleNamespace()))
        manager=SimpleNamespace(lock=threading.Lock(),last_feedback=None)
        self.assertIsNone(current_tasks(SimpleNamespace(checkpoints=manager)))


if __name__=='__main__':unittest.main()
