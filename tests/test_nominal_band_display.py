import unittest
from types import SimpleNamespace
import numpy as np
from femto.planning.nominal_band_display import ribbons


class NominalBandDisplayTests(unittest.TestCase):
    def test_width_is_metric_not_constant_pixels_and_does_not_bridge_chains(self):
        mesh=SimpleNamespace(vertices=np.array([[-1,-1,1],[1,-1,1],[0,1,1]]),faces=np.array([[0,1,2]]))
        def point(x,z,role='scan'):
            return dict(xyz_m=[x,0,z],triangle_index=0,motion_kind=role)
        plan=dict(treatment_head=dict(effective_treatment_width_mm=10),segments=[
            [point(0,1),point(.1,1)], [point(0,2),point(.1,2)],
            [point(0,1),point(.2,1,'connector')]])
        quads=ribbons(plan,mesh,np.diag([100.,100.,1.]))
        self.assertEqual(len(quads),2)
        self.assertAlmostEqual(np.ptp(quads[0][:,1]),1.)
        self.assertAlmostEqual(np.ptp(quads[1][:,1]),.5)

    def test_degenerate_chords_do_not_emit_nonfinite_geometry(self):
        mesh=SimpleNamespace(vertices=np.zeros((3,3)),faces=np.array([[0,1,2]]))
        p=dict(xyz_m=[0,0,1],triangle_index=0)
        plan=dict(treatment_head=dict(effective_treatment_width_mm=5),segments=[[p,p]])
        self.assertEqual(ribbons(plan,mesh,np.eye(3)),[])
