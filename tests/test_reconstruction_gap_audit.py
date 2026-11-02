"""Actual ray probes distinguish extraction gaps and successive filter losses."""
import unittest

import numpy as np
import open3d as o3d

from femto.local.reconstruction_gap_audit import GapProbe, STAGES


def surface():
    return o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(
        [[-1,-1,1],[1,-1,1],[0,1,1]]),o3d.utility.Vector3iVector([[0,1,2]]))


class GapAuditTests(unittest.TestCase):
    def probe(self):
        return GapProbe(np.array([[0,0]]),np.eye(3),np.array([[1000]],np.uint16))

    def test_each_filter_loss_and_extraction_gap_are_distinguished(self):
        empty=o3d.geometry.TriangleMesh()
        for last_hit in range(-1,4):
            probe=self.probe()
            for index,stage in enumerate(STAGES):probe(stage,surface() if index<=last_hit else empty)
            report,cause=probe.report()
            self.assertEqual(cause.tolist(),[last_hit+1])
            self.assertEqual(sum(report['counts'].values()),1)
            self.assertEqual(report['counts']['surface_present_in_replay'],0)

    def test_final_hit_is_never_reported_as_lost_even_if_a_previous_probe_missed(self):
        probe=self.probe()
        for index,stage in enumerate(STAGES):probe(stage,surface() if index in (0,4) else o3d.geometry.TriangleMesh())
        report,cause=probe.report()
        self.assertEqual(cause.tolist(),[255])
        self.assertEqual(report['stages'][-1]['reference_consistent_pixels'],1)

    def test_incomplete_out_of_order_and_invalid_inputs_are_rejected(self):
        probe=self.probe()
        with self.assertRaises(ValueError):probe.report()
        with self.assertRaises(ValueError):probe(STAGES[1],surface())
        with self.assertRaises(ValueError):GapProbe(np.array([[1,0]]),np.eye(3),np.array([[1000]]))

    def test_cancellation_is_observed_before_native_ray_allocation(self):
        def stop():raise RuntimeError('cancelled')
        probe=self.probe();probe.check=stop
        with self.assertRaisesRegex(RuntimeError,'cancelled'):probe(STAGES[0],surface())
        self.assertEqual(probe.depths,{})
