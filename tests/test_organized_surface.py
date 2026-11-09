"""Analytic normals, unknown boundaries, correspondence and pose contracts."""
import unittest
import cv2
import numpy as np

from femto.geometry.organized_surface import (point_map,integral_normals,edge_labels,normal_space_indices,
    MISSING_BOUNDARY,OCCLUDING,OCCLUDED,HIGH_CURVATURE)
from femto.local.organized_registration import correspondences,solve_projective


def plane(width=100,height=80):
    k=np.array([[160.,0,(width-1)/2],[0,160.,(height-1)/2],[0,0,1.]])
    y,x=np.indices((height,width),dtype=float)
    z=.7/(1-.2*(x-k[0,2])/k[0,0]+.1*(y-k[1,2])/k[1,1])
    return z*1000,k


def curved(width=160,height=120):
    depth,k=plane(width,height)
    y,x=np.indices(depth.shape,dtype=float)
    depth=700+30*np.sin(x/19)+25*np.cos(y/17)+15*np.sin((x+y)/13)
    return point_map(depth,k),k


class OrganizedSurfaceTests(unittest.TestCase):
    def test_integral_normals_analytic_plane_orientation_and_noise_stability(self):
        depth,k=plane();points=point_map(depth,k);original=points.copy()
        normals,valid=integral_normals(points)
        expected=np.array([.2,-.1,-1]);expected/=np.linalg.norm(expected)
        self.assertGreater(valid.mean(),.7)
        np.testing.assert_allclose(normals[valid],np.broadcast_to(expected,(valid.sum(),3)),atol=1e-9)
        noisy=depth+np.random.default_rng(42).normal(0,.2,depth.shape)
        estimated,good=integral_normals(point_map(noisy,k))
        angles=np.degrees(np.arccos(np.clip(estimated[good]@expected,-1,1)))
        self.assertLess(np.percentile(angles,95),1.)
        np.testing.assert_array_equal(points,original)

    def test_missing_depth_step_and_outside_range_do_not_become_normals(self):
        depth,k=plane();depth[20:30,20:30]=0;depth[:,60:]+=200
        depth[10,10]=np.nan;depth[10,11]=2500
        points=point_map(depth,k);normals,good=integral_normals(points)
        self.assertFalse(good[17:33,17:33].any())
        self.assertFalse(good[:,57:63].any())
        self.assertFalse(np.isfinite(points[10,10:12]).any())
        self.assertTrue(np.isnan(normals[~good]).all())
        self.assertTrue(good[40:60,20:40].all())

    def test_edge_labels_separate_hole_foreground_background_and_bend(self):
        depth=np.full((30,40),700.);depth[:,20:]=900.;depth[5:9,5:9]=0
        k=np.array([[100.,0,20],[0,100.,15],[0,0,1.]])
        points=point_map(depth,k)
        normals=np.broadcast_to([0.,0,-1],points.shape).copy()
        normals[15:,0:15]=[.7,0,-np.sqrt(.51)]
        labels=edge_labels(points,normals)
        self.assertTrue(np.all(labels[:,19]&OCCLUDING))
        self.assertTrue(np.all(labels[:,20]&OCCLUDED))
        self.assertTrue(labels[4,6]&MISSING_BOUNDARY)
        self.assertEqual(labels[6,6],0)  # Missing pixels are not measured geometry.
        self.assertTrue(labels[14,10]&HIGH_CURVATURE)
        self.assertFalse(np.any(labels[:,19:21]&HIGH_CURVATURE))
        np.testing.assert_array_equal(edge_labels(points),edge_labels(points))

    def test_normal_sampling_balances_directions_preserves_indices_and_rejects_invalid(self):
        normals=np.r_[np.tile([[0,0,-1.]],(90,1)),np.tile([[1.,0,0]],(10,1)),[[np.nan,0,0],[0,0,0]]]
        before=normals.copy();ids=normal_space_indices(normals,10)
        self.assertEqual(len(ids),10);self.assertEqual(int((ids>=90).sum()),5)
        self.assertEqual(len(np.unique(ids)),len(ids))
        np.testing.assert_array_equal(ids,normal_space_indices(normals,10))
        np.testing.assert_array_equal(before,normals)
        np.testing.assert_array_equal(normal_space_indices(normals,200),np.arange(100))
        self.assertEqual(len(normal_space_indices(np.zeros((5,3)),3)),0)

    def test_bad_calibration_and_parameters_rejected(self):
        depth,k=plane()
        bad=k.copy();bad[0,0]=0
        with self.assertRaises(ValueError):point_map(depth,bad)
        with self.assertRaises(ValueError):point_map(depth,k,2,1)
        with self.assertRaises(ValueError):integral_normals(point_map(depth,k),radius=0)
        with self.assertRaises(ValueError):edge_labels(point_map(depth,k),curvature_deg=np.nan)
        with self.assertRaises(ValueError):normal_space_indices(np.ones((5,3)),0)


class ProjectiveRegistrationTests(unittest.TestCase):
    def test_projective_recovers_known_pose_without_mutating_inputs(self):
        target,k=curved();normals,good=integral_normals(target)
        truth=np.eye(4);truth[:3,:3]=cv2.Rodrigues(np.array([.001,-.002,.001]))[0];truth[:3,3]=[.001,-.001,.001]
        source=(target[good]-truth[:3,3])@truth[:3,:3]
        original=source.copy()
        result,report=solve_projective(source,target,normals,k,np.eye(4))
        np.testing.assert_allclose(result,truth,atol=1e-6)
        np.testing.assert_array_equal(source,original)
        self.assertTrue(report['iterations'])

    def test_occlusion_missing_behind_camera_and_duplicate_votes_rejected(self):
        depth,k=plane();target=point_map(depth,k);normals,_=integral_normals(target)
        p=target[40,50].copy()
        source=np.array([p,p,p+[0,0,.1],[-1,0,-1],[np.nan,0,1]])
        ids,*_=correspondences(source,target,normals,k,np.eye(4))
        np.testing.assert_array_equal(ids,[0])
        target[40,50]=np.nan
        self.assertEqual(len(correspondences(source,target,normals,k,np.eye(4))[0]),0)

    def test_unconstrained_plane_and_insufficient_support_rejected(self):
        depth,k=plane();target=point_map(depth,k);normal,good=integral_normals(target)
        with self.assertRaisesRegex(ValueError,'degenerate'):
            solve_projective(target[good],target,normal,k,np.eye(4))
        with self.assertRaisesRegex(ValueError,'insufficient'):
            solve_projective(target[good][:2],target,normal,k,np.eye(4))
        bad=np.eye(4);bad[0,0]=2
        with self.assertRaises(ValueError):solve_projective(target[good],target,normal,k,bad)

    def test_cancellation_is_checked(self):
        target,k=curved();normal,good=integral_normals(target)
        def cancelled():raise RuntimeError('cancelled')
        with self.assertRaisesRegex(RuntimeError,'cancelled'):
            solve_projective(target[good],target,normal,k,np.eye(4),check=cancelled)


if __name__=='__main__':unittest.main()
