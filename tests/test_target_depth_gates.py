"""Depth-gate scalar conversion preserves priority, ULP boundaries and errors."""
import unittest
import warnings
import numpy as np
from femto.planning.target_surface import _pixel_depth_reason


def reference(visible, limits, measured, gate):
    if any(((p[2][:,2]<limits['minimum_m']) | (p[2][:,2]>limits['maximum_m'])).any() for p in visible):return 3
    if any((np.abs(p[2][:,2]-measured)>gate).any() for p in visible):return 6
    return 0


def outcome(fn,columns,measured,mode):
    visible=[(0,None,np.column_stack((np.zeros(len(z)),np.zeros(len(z)),z)).astype(z.dtype),None) for z in columns]
    events=[]
    with np.errstate(**(mode if isinstance(mode,dict) else {'all':mode})),warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always'); previous=np.geterrcall();np.seterrcall(lambda text,flag:events.append((text,flag)))
        try:
            try:result=('value',fn(visible,dict(minimum_m=-1e300,maximum_m=1e300),measured,.01))
            except Exception as e:result=(type(e).__name__,str(e))
        finally:np.seterrcall(previous)
        return result,[str(w.message) for w in caught],events


class TargetDepthGatesTests(unittest.TestCase):
    def test_ulp_boundaries_and_error_modes(self):
        cases=[np.array([.49,.5,.51]),np.array([np.nextafter(.51,-np.inf),.51,np.nextafter(.51,np.inf)]),np.array([0.,1e-310,-1e-310]),np.array([1e300,-1e300,0.]),np.array([np.nan,np.inf,-np.inf])]
        for dtype in (np.float64,np.float32):
            for mode in ('ignore','warn','raise','call',{'over':'raise','invalid':'raise','under':'ignore'}):
                for z in cases:
                    with np.errstate(all='ignore'):z=z.astype(dtype)
                    for measured in (np.float64(.5),np.float64(1e300),np.float64(np.nan)):
                        with self.subTest(dtype=dtype,mode=mode,values=z):
                            self.assertEqual(outcome(reference,[z],measured,mode),outcome(_pixel_depth_reason,[z],measured,mode))

    def test_range_limits_keep_strict_ulp_comparisons(self):
        for edge in (.1,1.):
            for value in (np.nextafter(edge,-np.inf),edge,np.nextafter(edge,np.inf)):
                visible=[(0,None,np.array([[0.,0.,value]]),None)]
                limits=dict(minimum_m=.1,maximum_m=1.)
                self.assertEqual(reference(visible,limits,np.float64(value),.01),_pixel_depth_reason(visible,limits,np.float64(value),.01))

    def test_range_failure_precedes_residual_failure_in_any_polygon(self):
        polygons=[(0,None,np.array([[0.,0.,z]]),None) for z in (.9,1.1)]
        self.assertEqual(_pixel_depth_reason(polygons,dict(minimum_m=.1,maximum_m=1.),np.float64(.5),.01),3)

    def test_many_small_polygons_match_original_with_strided_views(self):
        rng=np.random.default_rng(20261104)
        for _ in range(1000):
            polygons=[(i,None,rng.normal(size=(int(rng.integers(3,8)),6))[:,::2],None) for i in range(int(rng.integers(0,5)))]
            limits=dict(minimum_m=-2.,maximum_m=2.); measured=np.float64(rng.normal())
            self.assertEqual(reference(polygons,limits,measured,.01),_pixel_depth_reason(polygons,limits,measured,.01))

if __name__=='__main__':unittest.main()
