"""Do not confuse gains, degraded support, unchanged targets or unknown evidence."""
import unittest
import numpy as np
from femto.capture.scan_feedback import build_feedback


class ScanFeedbackTests(unittest.TestCase):
    def test_unmapped_original_target_pixels_cannot_disappear_into_green_mapped_view(self):
        mapping=dict(raw_requested_pixels=10,represented_raw_pixels=8,
            unmapped_raw_target_pixels=2,raw_pixel_bounds=[0,0,5,2])
        report=build_feedback(np.ones((2,4),np.uint8),self.binding(),raw_mapping=mapping)
        self.assertNotEqual(report['status'],'reference_support_met')
        self.assertEqual(report['raw_target_mapping']['unmapped_raw_target_pixels'],2)
        self.assertEqual(report['tasks'][0]['pixel_frame'],'raw_aligned_rgb')
        self.assertIsNone(report['tasks'][0]['position_world_m'])
        with self.assertRaisesRegex(ValueError,'sampling account'):
            build_feedback(np.ones((2,4)),self.binding(),raw_mapping=dict(mapping,raw_requested_pixels=9))

    def binding(self,stage='one',target='same'):
        return dict(target_key=target,stage_id=stage,session_id='session',
                    reference_frame_id=0,model_sha256=stage)

    def test_reasons_and_2d_tasks_preserve_unknown_3d(self):
        report=build_feedback(np.array([[0,1,2,3],[4,5,6,7]],np.uint8),self.binding())
        self.assertEqual(report['requested_pixels'],7)
        self.assertEqual(report['pending_pixels'],6)
        self.assertEqual(len(report['tasks']),6)
        self.assertTrue(all(t['position_world_m'] is None and t['candidate_observation_pose'] is None
                            for t in report['tasks']))
        self.assertFalse(report['execution_ready'])
        self.assertEqual(report['tasks'][0]['action'],'inspect_reconstruction')

    def test_exact_comparison_counts_gains_and_regression_separately(self):
        old=np.array([[0,1,2],[3,1,7]],np.uint8)
        previous=build_feedback(old,self.binding())
        new=np.array([[0,5,1],[1,1,7]],np.uint8)
        report=build_feedback(new,self.binding('two'),(previous,old))
        self.assertEqual(report['comparison']['improved_pixels'],2)
        self.assertEqual(report['comparison']['degraded_pixels'],1)
        self.assertEqual(report['pending_pixels'],2)

    def test_reference_or_target_domain_change_cannot_masquerade_as_improvement(self):
        old=np.array([[1,3]],np.uint8);report=build_feedback(old,self.binding())
        with self.assertRaisesRegex(ValueError,'different original target'):
            build_feedback(old,self.binding('two','different'),(report,old))
        with self.assertRaisesRegex(ValueError,'domain changed'):
            build_feedback(np.array([[1,0]]),self.binding('two'),(report,old))

    def test_unknown_previous_diagnosis_does_not_fabricate_gain_or_no_gain(self):
        unknown=build_feedback(None,self.binding())
        report=build_feedback(np.array([[1,3]]),self.binding('two'),(unknown,None))
        self.assertIsNone(report['comparison'])
        self.assertEqual(report['comparison_unavailable_reason'],'previous_diagnosis_unavailable')
        self.assertEqual(report['stagnant_rounds'],0)
        self.assertIsNone(report['stop_reason'])
        with self.assertRaisesRegex(ValueError,'different original target'):
            build_feedback(np.ones((1,2)),self.binding('two','changed'),(unknown,None))

    def test_repeated_no_gain_stop_and_reference_support_status_are_distinct(self):
        values=np.array([[1,3]])
        report=build_feedback(values,self.binding())
        report=build_feedback(values,self.binding('two'),(report,values))
        report=build_feedback(values,self.binding('three'),(report,values))
        self.assertEqual(report['stagnant_rounds'],2)
        self.assertIsNotNone(report['stop_reason'])
        self.assertNotEqual(report['status'],'reference_support_met')
        self.assertEqual(build_feedback(None,self.binding())['status'],'cannot_determine')
        passed=build_feedback(np.ones((2,2),np.uint8),self.binding())
        self.assertEqual(passed['status'],'reference_support_met')
        self.assertFalse(passed['physical_accuracy_verified'])


if __name__=='__main__':unittest.main()
