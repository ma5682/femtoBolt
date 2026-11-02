"""Provider return validation never certifies a frame before the internal gate."""
import copy
import unittest
import numpy as np
from femto.capture.scan_exchange import RESULT_SCHEMA,task_batch,validate_observation,verification


class ExchangeTests(unittest.TestCase):
    def setUp(self):
        self.cal='a'*64
        self.binding=dict(stage_id='stage',session_id='session',reference_frame_id=0,
                          target_key='b'*64,model_sha256='c'*64)
        self.task=task_batch(dict(binding=self.binding,tasks=[]),self.cal)
        self.observation=dict(schema=RESULT_SCHEMA,task_batch_id='stage',source_binding=self.binding,
            calibration_sha256=self.cal,coordinate_frame='world_rdf',unit='metre',transform_direction='T_W0_Ct',
            accepted_for_fusion=False,execution_status='received',T_W0_Ct=None,
            rgbd=dict(rgb='raw/rgb/0001.png',depth='raw/depth/0001.png',
                      capture_timestamp_us=10000,color_timestamp_us=10000,depth_timestamp_us=10001))

    def check(self,packet):
        return validate_observation(packet,self.task,current_binding=self.binding,calibration_sha256=self.cal)

    def test_valid_envelope_still_requires_existing_capture_gate(self):
        packet=self.check(self.observation)
        self.assertFalse(packet['accepted_for_fusion'])
        self.assertTrue(packet['requires_capture_gate'])
        self.assertIsNone(packet['T_W0_Ct'])

    def test_wrong_session_model_calibration_or_task_is_rejected(self):
        for key,value in [('task_batch_id','other'),('calibration_sha256','d'*64),('accepted_for_fusion',True)]:
            packet=copy.deepcopy(self.observation);packet[key]=value
            with self.assertRaises(ValueError):self.check(packet)
        packet=copy.deepcopy(self.observation);packet['source_binding']['model_sha256']='e'*64
        with self.assertRaises(ValueError):self.check(packet)

    def test_unsynchronized_or_invalid_pose_is_rejected(self):
        packet=copy.deepcopy(self.observation);packet['rgbd']['depth_timestamp_us']=20000
        with self.assertRaisesRegex(ValueError,'unsynchronized'):self.check(packet)
        packet=copy.deepcopy(self.observation);packet['T_W0_Ct']=np.zeros((4,4)).tolist()
        with self.assertRaisesRegex(ValueError,'pose'):self.check(packet)

    def test_verification_needs_matching_internal_record_and_preserves_rejection(self):
        record=dict(session_id='session',reference_frame_id=0,calibration_sha256=self.cal,
            **{k:v for k,v in self.observation['rgbd'].items() if k.endswith('_us')},
            frame_id=5,state='tracking',selected_for_fusion=True,T_WC=np.eye(4).tolist(),
            quality=dict(usable=True),continuity=dict(current_frame_verified=True))
        record['raw_write_receipt']={key:record[key] for key in
            ('frame_id','capture_timestamp_us','color_timestamp_us','depth_timestamp_us')}
        result=verification(self.task,self.observation,record,current_binding=self.binding,calibration_sha256=self.cal)
        self.assertTrue(result['accepted_for_fusion'])
        self.assertIsNone(result['target_improvement'])
        packet=copy.deepcopy(self.observation)
        packet['T_W0_Ct']=np.eye(4).tolist()
        packet['T_W0_Ct'][0][3]=.01
        with self.assertRaisesRegex(ValueError,'pose differs'):
            verification(self.task,packet,record,current_binding=self.binding,calibration_sha256=self.cal)
        packet['T_W0_Ct']=copy.deepcopy(record['T_WC'])
        self.assertTrue(verification(self.task,packet,record,current_binding=self.binding,
            calibration_sha256=self.cal)['accepted_for_fusion'])
        committed=record.pop('raw_write_receipt')
        self.assertFalse(verification(self.task,self.observation,record,
            current_binding=self.binding,calibration_sha256=self.cal)['accepted_for_fusion'])
        record['raw_write_receipt']=committed
        record['selected_for_fusion']=False
        result=verification(self.task,self.observation,record,current_binding=self.binding,calibration_sha256=self.cal)
        self.assertFalse(result['accepted_for_fusion'])
        record['reference_frame_id']=3
        with self.assertRaises(ValueError):
            verification(self.task,self.observation,record,current_binding=self.binding,calibration_sha256=self.cal)


if __name__=='__main__':unittest.main()
