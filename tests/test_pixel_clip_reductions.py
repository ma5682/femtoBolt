"""Direct ndarray reductions keep original clipping arithmetic and fallback."""
import unittest
from unittest.mock import patch

import numpy as np

from femto.geometry import pixel_clip as pixel


def original_clip(triangle, planes):
    points=np.eye(3)
    for plane in planes:
        distances=triangle @ plane
        errors=pixel.EPS*np.sum(np.abs(triangle*plane),axis=1)
        signs=points @ distances
        bounds=points @ errors+pixel.EPS*(np.abs(points) @ np.abs(distances))
        if np.any(np.abs(signs)<=bounds):return pixel._rational_clip(triangle,planes),True
        positive=np.count_nonzero(signs>0)
        if positive==len(points):continue
        if positive==0:return np.empty((0,3),float),False
        out=[]
        for index,a in enumerate(points):
            b=points[(index+1)%len(points)]
            da,db=signs[index],signs[(index+1)%len(points)]
            if da>0:out.append(a)
            if (da<0)!=(db<0):out.append(a+(da/(da-db))*(b-a))
        if len(out)<3:return np.empty((0,3),float),False
        points=np.asarray(out)
    return points,False


def original_area(poly):
    if len(poly)<3:return 0.
    shifted=poly-poly[0]
    return abs(float(np.sum(shifted[:-1,0]*shifted[1:,1]-shifted[:-1,1]*shifted[1:,0])))/2


class PixelReductionTests(unittest.TestCase):
    def test_finite_overflow_preserves_original_unknown_result_and_flag(self):
        triangle=np.array([[-1.,-1.,1.],[1.,-1.,1.],[0.,1.,1.]])*1e200
        planes=pixel.pixel_planes(np.diag([1e200,1e200,1.]),0,0)
        with np.errstate(all='ignore'):
            actual,flag=pixel.clip_triangle_planes(triangle,planes)
            expected,old_flag=original_clip(triangle,planes)
        self.assertEqual(flag,old_flag)
        self.assertEqual(actual.tobytes(),expected.tobytes())

    def test_random_clip_and_area_preserve_bytes_and_rational_fallback(self):
        rng=np.random.default_rng(20261020)
        for scale in (1e-12,1.,1e12):
            for _ in range(160):
                triangle=rng.uniform(-2,2,(3,3))
                triangle[:,2]=rng.uniform(.5,2,3)
                triangle*=scale
                intrinsic=np.array([[1.7,0,.2],[0,1.3,-.4],[0,0,1.]])
                x,y=rng.integers(-2,3,size=2)
                planes=pixel.pixel_planes(intrinsic,int(x),int(y))
                actual,actual_exact=pixel.clip_triangle_planes(triangle,planes)
                expected,expected_exact=original_clip(triangle,planes)
                self.assertEqual(actual_exact,expected_exact)
                self.assertEqual(actual.shape,expected.shape)
                self.assertEqual(actual.tobytes(),expected.tobytes())
                projected=pixel.project(actual @ triangle,intrinsic)
                self.assertEqual(np.float64(pixel.area2(projected)).tobytes(),
                                 np.float64(original_area(projected)).tobytes())

    def test_half_pixel_and_tiny_offsets_keep_exact_boundary_decisions(self):
        for offset in (0.,2**-48,-2**-48):
            triangle=np.array([[-.5+offset,-.5,1.],[.5,-.5,1.],[.5,.5,1.]])
            planes=pixel.pixel_planes(np.eye(3),0,0)
            actual,flag=pixel.clip_triangle_planes(triangle,planes)
            expected,old_flag=original_clip(triangle,planes)
            self.assertTrue(flag)
            self.assertEqual(flag,old_flag)
            self.assertEqual(actual.tobytes(),expected.tobytes())
        for polygon in (np.zeros((0,2)),np.zeros((3,2)),
                        np.array([[0.,0.],[2**-1000,0.],[0.,2**-1000]])):
            self.assertEqual(np.float64(pixel.area2(polygon)).tobytes(),
                             np.float64(original_area(polygon)).tobytes())

    def test_full_target_keeps_metadata_bytes_and_fixed_work_frontiers(self):
        import test_target_surface as fixtures
        from femto.planning import planning_revision as revisions,target_surface as target
        from femto.planning.quality_constrained import evidence
        case=fixtures.TargetSurfaceTests();case.setUp();self.addCleanup(case.doCleanups)
        mask=np.zeros((100,100),bool);mask[47:54,47:54]=True;mask[50,50]=False
        model,region=case.fixture(mask=mask)
        quality=evidence(model,region,allow_empty=True)
        for algorithm in (revisions.ALGORITHMS['target'],revisions.EXACT_TARGET):
            revision=revisions.capture(model,region,case.head,
                algorithms=dict(revisions.ALGORITHMS,target=algorithm))
            for budget in (target.TargetBudget(),target.TargetBudget(max_units=7),
                           target.TargetBudget(max_pair_checks=3)):
                with self.subTest(algorithm=algorithm,budget=budget):
                    actual=target.build(model,region,revision,evidence=quality,budget=budget)
                    with patch.object(target,'clip_triangle_planes',original_clip), \
                         patch.object(target,'area2',original_area):
                        expected=target.build(model,region,revision,evidence=quality,budget=budget)
                    self.assertEqual(actual.metadata,expected.metadata)
                    for key in actual.arrays:
                        self.assertEqual(actual.arrays[key].tobytes(),expected.arrays[key].tobytes(),key)
                    target.validate(actual,revision,evidence=quality,domain=None)


if __name__=='__main__':unittest.main()
