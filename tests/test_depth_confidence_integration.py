from __future__ import annotations

import unittest

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from femto.gsplat.scene_gsplat_metrics import metric_depth_loss_torch


@unittest.skipIf(torch is None, 'direct gsplat runtime test requires PyTorch')
class DepthConfidenceIntegrationTests(unittest.TestCase):
    def test_repaired_depth_has_lower_training_influence(self):
        prediction = torch.tensor([[0.9, 1.8]], dtype=torch.float32, requires_grad=True)
        target = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        confidence = torch.tensor([[1.0, 0.3]], dtype=torch.float32)
        loss, valid = metric_depth_loss_torch(prediction, target, beta_m=0.02, confidence=confidence)
        expected = (0.09 * 1.0 + 0.19 * 0.3) / 1.3
        self.assertAlmostEqual(float(loss.detach()), expected, places=6)
        self.assertTrue(bool(valid.all()))

    def test_unknown_depth_is_excluded(self):
        prediction = torch.tensor([[0.9, 5.0]], dtype=torch.float32)
        target = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        confidence = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        loss, valid = metric_depth_loss_torch(prediction, target, beta_m=0.02, confidence=confidence)
        self.assertAlmostEqual(float(loss), 0.09, places=6)
        np.testing.assert_array_equal(valid.numpy(), [[True, False]])


if __name__ == '__main__':
    unittest.main()
