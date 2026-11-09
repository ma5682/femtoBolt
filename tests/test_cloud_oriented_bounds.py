"""Directed envelopes must never hide any original floating-point witness."""
from fractions import Fraction
import unittest
import numpy as np
from femto.geometry.cloud_oriented_bounds import dot_interval,OrientedBounds
from femto.geometry.cloud_layer_batches import negative_bounds


class OrientedBoundsTests(unittest.TestCase):
    def test_dot_interval_encloses_exact_float_rational_expression(self):
        rng=np.random.default_rng(1130)
        for scale in (1e-160,1e-4,1.,1e6,1e150):
            a=rng.normal(size=(40,3))*scale;b=rng.normal(size=(40,3))*scale;n=rng.normal(size=(40,3))
            with np.errstate(all='ignore'):lo,hi=dot_interval(a,b,n)
            for i in range(len(a)):
                exact=sum((Fraction(float(a[i,j]))-Fraction(float(b[i,j])))*Fraction(float(n[i,j])) for j in range(3))
                self.assertLessEqual(Fraction(float(lo[i])),exact);self.assertGreaterEqual(Fraction(float(hi[i])),exact)

    def test_tilted_curved_missing_and_double_surfaces_preserve_every_witness(self):
        rng=np.random.default_rng(1131)
        xy=rng.uniform(-.04,.04,(700,2))
        plane=np.c_[xy,.5+xy[:,0]*.6+xy[:,1]*.3]
        normal=np.tile([-.6,-.3,1.],(len(plane),1));normal/=np.linalg.norm(normal,axis=1)[:,None]
        cases=[(plane,normal),(plane+np.c_[xy*0,.1*np.sin(xy[:,0]*60)**2],normal),
               (np.r_[plane,plane+[0,0,.006]],np.tile(normal,(2,1))),
               (rng.uniform(-.04,.04,(700,3)),rng.normal(size=(700,3)))]
        for points,normals in cases:
            for origin in (np.zeros(3),np.array([1e6,-1e6,1e6])):
                p=points+origin;negative=negative_bounds(p,normals,oriented=True)
                self.assertTrue(np.all(negative|~negative_bounds(p,normals)))
                for i in np.flatnonzero(negative):
                    delta=p-p[i];near=np.linalg.norm(delta,axis=1)<.01
                    self.assertFalse(np.any(np.abs(delta[near]@normals[i])>.004))

    def test_thresholds_extreme_normals_and_cancellation(self):
        for height in (np.nextafter(.004,0),.004,np.nextafter(.004,np.inf)):
            points=np.array([[0.,0.,.5],[0.,0.,.5+height]])
            for normal in ([0.,0.,1.],[0.,0.,2.],[1e307,1e307,1.]):
                normals=np.tile(normal,(2,1));result=negative_bounds(points,normals,oriented=True)
                for i in np.flatnonzero(result):
                    with np.errstate(all='ignore'):
                        self.assertFalse(np.any(np.abs((points-points[i])@normals[i])>.004))
        def cancel():raise InterruptedError('cancel')
        with self.assertRaises(InterruptedError):negative_bounds(points,normals,oriented=True,check=cancel)


if __name__=='__main__':unittest.main()
