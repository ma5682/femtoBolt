from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase, mock
import json

import numpy as np

import femto.depth.depth_quality as depth_quality
from femto.depth.depth_quality import DepthQualityConfig, assess_and_fill_depth, process_depth_quality
from femto.depth.depth_error_check import depth_error_check
from femto.geometry.depth_confidence_cloud import build_confident_depth_cloud, write_confident_point_cloud
from femto.viewer.depth_debug_viewer import DepthDebugData, DepthDebugViewer, MODES, GREEN, YELLOW, RED


class DepthErrorDebugTests(TestCase):
    def source(self):
        depth = np.full((60,80),1000,np.uint16)
        depth[30,40] = 0
        return depth

    def test_original_comparison_detects_changes_and_histogram_counts(self):
        original = self.source(); filled = original.copy()
        filled[0,0] += 7
        _, report = depth_error_check(original,filled,250,2210)
        comparison = report['original_valid_comparison']
        self.assertFalse(report['original_integrity_passed'])
        self.assertEqual(comparison['max_error_mm'],7)
        self.assertAlmostEqual(comparison['mean_absolute_error_mm'],7/4799)
        self.assertEqual(sum(comparison['depth_difference_histogram']['counts']),4799)
        self.assertIsNone(report['repair_ground_truth_error_mm'])

    def test_fifty_mm_is_inclusive_but_fifty_one_is_rejected(self):
        original = self.source()
        for value,rejected in ((1050,False),(1051,True),(949,True)):
            filled = original.copy(); filled[30,40] = value
            mask, report = depth_error_check(original,filled,250,2210)
            self.assertEqual(bool(mask[30,40]),rejected)
            self.assertEqual(report['original_valid_comparison']['max_error_mm'],0)

    def test_whole_hole_rolled_back_and_excluded_from_cloud(self):
        original = self.source(); original[30:32,40:42] = 0
        original[29,39] = 1055  # passes 30mm std +60mm span, fails new 50mm check
        filled, invalid, confidence, report = assess_and_fill_depth(original,250,2210)
        self.assertEqual(report['repaired_pixels'],0)
        self.assertEqual(report['rejected_depth_error_regions'],1)
        self.assertTrue(np.all(filled[invalid]==0))
        self.assertTrue(np.all(confidence[invalid]==0))
        self.assertTrue(report['depth_error_check']['after_rejection']['passed'])
        self.assertEqual(sum(report['method_pixels'].values()),0)

    def test_fast_has_no_full_connected_components_or_ransac(self):
        original = self.source(); original[40:42,50:52] = 0
        with mock.patch.object(depth_quality.cv2,'connectedComponentsWithStats',side_effect=AssertionError('full CC')), \
             mock.patch.object(depth_quality,'_ransac_plane',side_effect=AssertionError('RANSAC')):
            filled, _, _, report = assess_and_fill_depth(original,250,2210,config=DepthQualityConfig(mode='fast'))
        self.assertEqual(report['repaired_pixels'],1)
        self.assertTrue(np.all(filled[40:42,50:52]==0))
        self.assertEqual(report['unanalysed_invalid_pixels'],4)

    def test_quality_runs_plane_cross_check_after_neighbor_gate(self):
        with mock.patch.object(depth_quality,'_ransac_plane',wraps=depth_quality._ransac_plane) as plane:
            *_, report = assess_and_fill_depth(self.source(),250,2210)
        self.assertEqual(plane.call_count,1)
        self.assertEqual(report['ransac_checked_regions'],1)

    def test_fast_still_enforces_global_ten_percent_gate(self):
        original = self.source(); original[:10] = 0
        *_, report = assess_and_fill_depth(original,250,2210,config=DepthQualityConfig(mode='fast'))
        self.assertEqual(report['policy'],'report_only')
        self.assertEqual(report['repaired_pixels'],0)

    def test_confidence_property_survives_ply_export_and_bad_depth_blocked(self):
        original = self.source()
        filled, _, confidence, _ = assess_and_fill_depth(original,250,2210)
        rgb = np.full((60,80,3),128,np.uint8)
        intr = SimpleNamespace(width=80,height=60,fx=75,fy=75,cx=40,cy=30)
        distortion = dict(model='BROWN_CONRADY_K6',**{k:0.0 for k in ('k1','k2','k3','k4','k5','k6','p1','p2')})
        cloud, weights = build_confident_depth_cloud(original,filled,confidence,rgb,intr,distortion,250,2210)
        self.assertEqual(weights.shape,(4800,))
        self.assertEqual(weights[30*80+40],np.float32(.3))
        with TemporaryDirectory() as td:
            path = Path(td)/'confidence.ply'
            write_confident_point_cloud(path,cloud,weights)
            header, binary = path.read_bytes().split(b'end_header\n',1)
            self.assertIn(b'property float confidence',header)
            dtype = np.dtype([(k,'<f8') for k in ('x','y','z')]+[(k,'u1') for k in ('red','green','blue')]+[('confidence','<f4')])
            vertices = np.frombuffer(binary,dtype=dtype)
            np.testing.assert_array_equal(vertices['confidence'],weights)
            np.testing.assert_allclose(vertices['z'],1.0)
        filled[30,40] = 1051
        with self.assertRaises(ValueError):
            build_confident_depth_cloud(original,filled,confidence,rgb,intr,distortion,250,2210)
        filled[30,40] = 1000; confidence[30,40] = 1
        with self.assertRaises(ValueError):
            build_confident_depth_cloud(original,filled,confidence,rgb,intr,distortion,250,2210)

    def test_debug_modes_colors_error_file_and_pixel_description(self):
        original = self.source(); original[0,0] = 0
        with TemporaryDirectory() as td:
            output = Path(td)/'derived'
            process_depth_quality(original,250,2210,output)
            data = DepthDebugData(output)
            rendered = np.asarray(data.render('Confidence Map'))
            np.testing.assert_array_equal(rendered[1,1],GREEN)
            np.testing.assert_array_equal(rendered[30,40],YELLOW)
            np.testing.assert_array_equal(rendered[0,0],RED)
            for mode in MODES:
                self.assertEqual(data.render(mode).size,(80,60))
            self.assertIn('无原始深度真值',data.describe_pixel(40,30))
            error = json.loads((output/'depth_error_check.json').read_text())
            self.assertTrue(error['after_rejection']['passed'])
            data.export_overview(output/'debug.png')
            self.assertTrue((output/'debug.png').is_file())

    def test_tk_viewer_switches_all_modes_and_zoom(self):
        import tkinter as tk
        with TemporaryDirectory() as td:
            output = Path(td)/'derived'
            process_depth_quality(self.source(),250,2210,output)
            root = tk.Tk(); root.withdraw()
            try:
                viewer = DepthDebugViewer(root,output)
                root.update()
                for mode in MODES:
                    viewer.set_mode(mode)
                    self.assertEqual(viewer.mode.get(),mode)
                    self.assertEqual(len(viewer.canvas.find_all()),1)
                viewer.zoom_to(1)
                self.assertEqual(viewer.photo.width(),80)
            finally:
                root.destroy()
