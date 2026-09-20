from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib
import json
import unittest

import numpy as np
from PIL import Image

from depth_quality import (
    DepthQualityConfig,
    assess_and_fill_depth,
    confidence_acceptance_mask,
    process_depth_quality,
)


class DepthQualityTests(unittest.TestCase):
    def run_case(self, depth, *, rgb=None):
        return assess_and_fill_depth(depth, 250, 2210, rgb_bgr=rgb)

    def test_white_table_small_random_holes_are_repaired(self):
        depth=np.full((80,100),1000,np.uint16)
        depth[20,30]=0; depth[45:47,60:62]=0
        filled,mask,confidence,report=self.run_case(depth)
        self.assertEqual(report['policy'],'automatic_small_holes')
        self.assertEqual(report['repaired_pixels'],5)
        self.assertTrue(np.all(filled[mask]==1000))
        self.assertTrue(np.all(confidence[mask]==np.float32(.3)))

    def test_range_endpoints_and_observed_values_are_preserved(self):
        depth = np.full((100,100),1000,np.uint16)
        depth[10,10], depth[20,20] = 250, 2210
        depth[40,40], depth[60,60] = 249, 2211
        original = depth.copy()
        filled, invalid, confidence, report = self.run_case(depth)
        self.assertEqual(report['invalid_depth_pixels'],2)
        np.testing.assert_array_equal(depth,original)
        np.testing.assert_array_equal(filled[~invalid],original[~invalid])
        self.assertTrue(np.all(confidence[~invalid] == 1))

    def test_global_gate_does_not_misreport_small_hole_as_large(self):
        depth = np.full((100,100),1000,np.uint16)
        depth[:11] = 0
        depth[50,50] = 0
        filled, _, confidence, report = self.run_case(depth)
        self.assertEqual(report['policy'],'report_only')
        self.assertEqual(report['rejected_large_holes'],1)
        self.assertEqual(report['rejected_global_policy_regions'],1)
        self.assertEqual(filled[50,50],0)
        self.assertEqual(confidence[50,50],0)

    def test_small_depth_step_is_not_hidden_by_standard_deviation(self):
        depth = np.full((100,100),1000,np.uint16)
        depth[50,50] = 0
        depth[49,49] = 1080
        filled, _, _, report = self.run_case(depth)
        self.assertLess(report['components'][0]['depth_std_mm'],30)
        self.assertEqual(report['rejected_reflection_regions'],1)
        self.assertEqual(filled[50,50],0)

    def test_thresholds_cannot_enable_unsafe_policy(self):
        for kwargs in ({'local_repair_limit':.11}, {'global_repair_limit':.04},
                       {'maximum_depth_std_mm':float('nan')},
                       {'minimum_surrounding_valid_ratio':.5},
                       {'repaired_confidence':1.0}, {'max_small_hole_area':0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                DepthQualityConfig(**kwargs)
        with self.assertRaises(ValueError):
            self.run_case(np.empty((0,0),np.uint16))

    def test_many_separate_holes_use_global_pixel_coordinates(self):
        depth = np.full((240,320),1000,np.uint16)
        depth[5:-5:9,5:-5:9] = 0
        filled, mask, _, report = self.run_case(depth)
        self.assertEqual(report['repaired_pixels'],int(mask.sum()))
        self.assertTrue(np.all(filled == 1000))

    def test_local_policy_fills_only_supported_small_hole(self):
        depth = np.full((100,100),1000,np.uint16)
        depth[:4] = 0
        depth[50:52,50:52] = 0
        filled, _, _, report = self.run_case(depth)
        self.assertEqual(report['policy'],'local_high_confidence_only')
        self.assertEqual(report['repaired_pixels'],4)
        self.assertTrue(np.all(filled[:4] == 0))

    def test_black_object_small_smooth_hole_is_repaired(self):
        depth=np.full((60,80),1200,np.uint16); depth[25:27,30:32]=0
        rgb=np.full((60,80,3),8,np.uint8)
        filled,_,_,report=self.run_case(depth,rgb=rgb)
        self.assertTrue(np.all(filled[25:27,30:32]==1200))
        self.assertEqual(report['components'][0]['classification'],'A_small_continuous_hole')

    def test_phone_glass_depth_discontinuity_is_rejected(self):
        depth=np.full((80,100),800,np.uint16)
        depth[:,50:]=1600; depth[38:42,48:52]=0
        filled,_,confidence,report=self.run_case(depth)
        self.assertTrue(np.all(filled[38:42,48:52]==0))
        self.assertTrue(np.all(confidence[38:42,48:52]==0))
        self.assertEqual(report['rejected_reflection_regions'],1)

    def test_transparent_object_large_hole_is_never_filled(self):
        depth=np.full((100,120),1100,np.uint16); depth[25:55,40:75]=0
        filled,_,_,report=self.run_case(depth)
        self.assertTrue(np.all(filled[25:55,40:75]==0))
        self.assertEqual(report['rejected_large_holes'],1)

    def test_eight_neighbor_support_below_sixty_percent_is_rejected(self):
        depth=np.full((80,100),1000,np.uint16)
        depth[30:34,40:44]=0
        config=DepthQualityConfig(max_small_hole_area=64,max_hole_width=12,max_hole_height=12)
        filled,_,_,report=assess_and_fill_depth(depth,250,2210,config=config)
        self.assertTrue(np.all(filled[30:34,40:44]==0))
        self.assertEqual(report['rejected_low_support_regions'],1)
        self.assertLess(report['components'][0]['local_8_neighborhood_valid_ratio'],.60)

    def test_medical_shell_small_continuous_hole_is_repaired(self):
        yy,xx=np.indices((100,120)); depth=np.rint(900+xx*.05+yy*.03).astype(np.uint16)
        depth[50:52,60:62]=0
        filled,_,_,report=self.run_case(depth)
        self.assertTrue(np.all(filled[50:52,60:62]>0))
        self.assertLess(report['filled_invalid_rate'],report['original_invalid_rate'])

    def test_invalid_rate_policies_are_three_and_ten_percent(self):
        for rate,policy in ((.03,'automatic_small_holes'),(.04,'local_high_confidence_only'),(.10,'local_high_confidence_only'),(.11,'report_only')):
            depth=np.full((100,100),1000,np.uint16); depth.flat[:round(rate*depth.size)]=0
            *_,report=self.run_case(depth)
            self.assertEqual(report['policy'],policy)
            if rate>.10: self.assertEqual(report['repaired_pixels'],0)

    def test_component_average_cannot_authorize_unsupported_center(self):
        depth = np.full((80,100),1000,np.uint16)
        depth[30:33,40:43] = 0
        filled, _, _, report = self.run_case(depth)
        self.assertEqual(report['repaired_pixels'],0)
        self.assertTrue(np.all(filled[30:33,40:43] == 0))

    def test_wider_color_matching_surface_cannot_move_local_depth(self):
        depth = np.full((80,100),2000,np.uint16)
        depth[29:32,39:42] = 1000
        depth[30,40] = 0
        rgb = np.zeros((80,100,3),np.uint8)
        rgb[29:32,39:42] = 255
        rgb[30,40] = 0
        filled, _, _, report = self.run_case(depth,rgb=rgb)
        self.assertEqual(report['repaired_pixels'],1)
        self.assertEqual(filled[30,40],1000)

    def test_derived_outputs_preserve_original_uint16_evidence(self):
        with TemporaryDirectory() as td:
            depth=np.full((30,40),1000,np.uint16); depth[15,20]=0
            digest=hashlib.sha256(depth.tobytes()).hexdigest()
            filled,confidence,report,paths=process_depth_quality(depth,250,2210,Path(td)/'derived')
            self.assertEqual(hashlib.sha256(depth.tobytes()).hexdigest(),digest)
            self.assertEqual(np.asarray(Image.open(paths['original_depth'])).dtype,np.uint16)
            self.assertEqual(np.asarray(Image.open(paths['filled_depth'])).dtype,np.uint16)
            saved=np.load(paths['confidence'],allow_pickle=False)
            self.assertEqual(saved.dtype,np.float32)
            np.testing.assert_allclose(np.unique(saved),[.3,1.])
            payload=json.loads(Path(paths['quality_report']).read_text(encoding='utf-8'))
            for key in ('original_invalid_rate','repaired_pixels','repair_rate','rejected_large_holes','rejected_reflection_regions'):
                self.assertIn(key,payload)

    def test_statistics_only_skips_large_depth_artifacts(self):
        with TemporaryDirectory() as td:
            output=Path(td)/'derived';depth=np.full((30,40),1000,np.uint16);depth[15,20]=0
            filled,confidence,report,paths=process_depth_quality(depth,250,2210,output,keep_artifacts=False)
            self.assertEqual(set(paths),{'quality_report'})
            self.assertTrue(Path(paths['quality_report']).is_file())
            self.assertEqual([p.name for p in output.iterdir()],['depth_quality.json'])
            payload=json.loads(Path(paths['quality_report']).read_text(encoding='utf-8'))
            self.assertTrue(payload['statistics_only'])
            self.assertEqual(payload['outputs'],{})
            self.assertEqual(filled.dtype,np.uint16);self.assertEqual(confidence.dtype,np.float32)

    def test_tsdf_acceptance_retains_all_real_and_about_thirty_percent_repaired(self):
        confidence=np.full((100,100),.3,np.float32); confidence[:10]=1; confidence[-10:]=0
        accepted=confidence_acceptance_mask(confidence,3)
        self.assertTrue(accepted[:10].all()); self.assertFalse(accepted[-10:].any())
        self.assertTrue(.27 < accepted[10:-10].mean() < .33)


if __name__=='__main__': unittest.main()
