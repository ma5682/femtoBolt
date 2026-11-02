"""Independent exact truth and alias/error-state checks for a single ball."""
from fractions import Fraction
import math
import unittest
import numpy as np
from femto.geometry import exact_surface as exact


class SingleBallTests(unittest.TestCase):
    def test_single_ball_matches_fraction_truth_at_boundaries_and_scales(self):
        rng=np.random.default_rng(11102026)
        for scale in (2.**-540,.001,1.,2.**500):
            for repeat in range(40):
                points=rng.integers(-4,5,(3,3))*scale
                center=rng.integers(-2,3,3)*scale
                radius=float(rng.integers(0,8))*scale
                triangle=tuple(exact.rational(p) for p in points);origin=exact.rational(center)
                expected=all(sum((a-b)**2 for a,b in zip(point,origin))<=Fraction(radius)**2 for point in triangle)
                inside,_=exact.contains_balls(triangle,(origin,),(radius,))
                self.assertEqual(bool(inside[0]),expected)
        triangle=tuple(exact.rational(p) for p in ((1.,0.,0.),(0.,1.,0.),(0.,0.,1.)))
        origin=exact.rational((0.,0.,0.))
        for radius,expected in ((math.nextafter(1.,0.),False),(1.,True),(math.nextafter(1.,math.inf),True)):
            inside,_=exact.contains_balls(triangle,(origin,),(radius,));self.assertEqual(bool(inside[0]),expected)

    def test_single_readonly_strided_shared_bounds_preserve_inputs_and_error_state(self):
        storage=np.zeros((6,6));storage[::2,::2]=((1,0,0),(0,1,0),(0,0,1));storage.setflags(write=False)
        corners=storage[::2,::2][None,:,:];center=np.zeros((1,3));center.setflags(write=False)
        radius=np.array([2.]);radius.setflags(write=False);before=storage.tobytes()
        prior=np.geterr()
        with np.errstate(all='raise'):
            inside,outside=exact._ball_intervals(corners,corners,center,center,radius)
            self.assertEqual((inside.tolist(),outside.tolist()),([True],[False]))
            self.assertTrue(all(v=='raise' for v in np.geterr().values()))
        self.assertEqual(np.geterr(),prior);self.assertEqual(storage.tobytes(),before)


if __name__=='__main__':unittest.main()
