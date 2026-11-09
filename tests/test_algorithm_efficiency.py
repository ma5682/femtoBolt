"""Independent pre-change oracles for batch metrics, ROI and occlusion stops."""
from copy import deepcopy
from types import SimpleNamespace
from unittest import TestCase, mock

import numpy as np
import open3d as o3d

from femto.local import local_pose_refinement as refinement
from femto.geometry.local_region import RegionDepthFilter
from femto.planning import target_surface as target


def old_metrics(source, destination, transform):
    moved = deepcopy(source).transform(transform)
    tree = o3d.geometry.KDTreeFlann(destination)
    colors, other = np.asarray(moved.colors), np.asarray(destination.colors)
    has_colors = len(colors)==len(moved.points) and len(other)==len(destination.points)
    distances, errors = [], []
    for i, point in enumerate(moved.points):
        count, ids, d = tree.search_knn_vector_3d(point, 1)
        if count:
            distances.append(np.sqrt(d[0]))
            if d[0] <= .015**2 and has_colors:errors.append(np.mean(np.abs(colors[i]-other[ids[0]])))
    d = np.asarray(distances)
    return dict(median_m=float(np.median(d)) if len(d) else None,
                p90_m=float(np.percentile(d,90)) if len(d) else None,
                overlap=float(np.mean(d<=.015)) if len(d) else 0,
                color_error=float(np.median(errors)) if len(errors) else None)


def cloud(points, colors=None):
    result = o3d.geometry.PointCloud()
    result.points = o3d.utility.Vector3dVector(np.asarray(points).reshape(-1,3))
    if colors is not None:result.colors = o3d.utility.Vector3dVector(colors)
    return result


def old_filter(region, depth, pose, k, frame_id):
    ref=region.reference; output=np.zeros_like(depth)
    transform=np.linalg.inv(ref.pose)@pose; h,w=depth.shape; focal=min(k[0,0],k[1,1])
    for top in range(0,h,64):
        block=depth[top:top+64]
        y,x=np.nonzero((block>=region.minimum*1000)&(block<=region.maximum*1000));y=y+top
        z=depth[y,x].astype(float)/1000
        points=np.c_[(x-k[0,2])*z/k[0,0],(y-k[1,2])*z/k[1,1],z]
        points=points@transform[:3,:3].T+transform[:3,3]
        ids=np.flatnonzero(np.isfinite(points).all(axis=1)&(points[:,2]>0));p=points[ids]
        uv=np.rint(p[:,:2]/p[:,2,None]*[k[0,0],k[1,1]]+[k[0,2],k[1,2]]).astype(np.int64)
        inside=(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h);ids,uv=ids[inside],uv[inside]
        rz=ref.depth[uv[:,1],uv[:,0]].astype(float)/1000
        consistent=(rz>=region.minimum)&(rz<=region.maximum)&(np.abs(points[ids,2]-rz)<=.01)
        nearby=region.distance[uv[:,1],uv[:,0]]*points[ids,2]/focal<=region.record['support_margin_mm']/1000
        ids=ids[consistent&nearby];output[y[ids],x[ids]]=depth[y[ids],x[ids]]
    return output


def region_fixture(width=320,height=240):
    rng=np.random.default_rng(42)
    region=RegionDepthFilter.__new__(RegionDepthFilter)
    depth=rng.integers(0,2300,(height,width),dtype=np.uint16)
    k=np.array([[620.,0,width/2+.125],[0,619.,height/2+.375],[0,0,1.]])
    region.reference=SimpleNamespace(depth=depth,k=k,pose=np.eye(4))
    region.distance=rng.random(depth.shape).astype(np.float32)*80
    region.minimum,region.maximum=.25,2.21
    region.record={'support_margin_mm':40.};region.metrics=[]
    return region


def polygons_fixture(count=16):
    result=[]
    for i in range(count):
        x=i/count
        uv=np.array([[x,0],[x+.4/count,0],[x+.4/count,1],[x,1.]])
        xyz=np.c_[uv,np.ones(4)]
        result.append((i,None,xyz,uv))
    return result


class AlgorithmEfficiencyTests(TestCase):
    def test_batch_metrics_exact_scores_and_approval(self):
        rng=np.random.default_rng(241102)
        for count in (0,1,4000):
            points=rng.normal(0,.05,(count,3));points[:,2]+=.7
            colors=rng.random((count,3));source=cloud(points,colors)
            destination=cloud(points+rng.normal(0,.0005,(count,3)),colors)
            for colored in (True,False):
                if not colored:source.colors=o3d.utility.Vector3dVector()
                for angle in (0.,.003):
                    transform=np.eye(4);transform[:3,:3]=o3d.geometry.get_rotation_matrix_from_xyz([angle,-angle,angle])
                    transform[:3,3]=[.002,-.001,.0005]
                    old=old_metrics(source,destination,transform)
                    actual=refinement.metrics(source,destination,transform)
                    self.assertEqual(actual,old)
                    before={x:old for x in ('forward','reverse')};after={x:actual for x in before}
                    self.assertEqual(refinement.approve(np.eye(4),transform,np.linalg.inv(transform),before,after),
                                     refinement.approve(np.eye(4),transform,np.linalg.inv(transform),before,before))
            np.testing.assert_array_equal(np.asarray(source.points),points)

    def test_ties_duplicates_and_exact_gate_keep_legacy_colors(self):
        destination=cloud([[-.015,0,0],[.015,0,0],[0,.015,0],[0,.015,0]],
                          [[1,0,0],[0,1,0],[0,0,1],[1,1,1]])
        source=cloud([[0,0,0],[0,1e-17,0],[0,-1e-17,0]],np.zeros((3,3)))
        self.assertEqual(refinement.metrics(source,destination,np.eye(4)),old_metrics(source,destination,np.eye(4)))

    def test_reference_fast_path_and_actual_motion_match_old_pixels(self):
        region=region_fixture();depth=region.reference.depth.copy();depth.flat[::7]+=10
        for shift in (0.,1e-8,.024):
            pose=np.eye(4);pose[0,3]=shift
            expected=old_filter(region,depth,pose,region.reference.k,0)
            np.testing.assert_array_equal(region.filter(depth,pose,region.reference.k,0),expected)
        self.assertEqual(region.metrics[-1]['integrated_pixels'],np.count_nonzero(expected))
        # Allclose alone is insufficient to enable the shortcut.
        region.reference.pose[0,3]=1e-10
        np.testing.assert_array_equal(region.filter(depth,region.reference.pose,region.reference.k,0),
                                      old_filter(region,depth,region.reference.pose,region.reference.k,0))

    def test_occlusion_pairs_and_cancel_frontiers_preserved(self):
        polygons=polygons_fixture()
        triangles=np.zeros((len(polygons),3,3));calls=[]
        def run(limit):
            calls.clear()
            value=target._resolve_pixel_occlusion(polygons,0,0,0,{'pair_checks':0},
                target.TargetBudget(max_pair_checks=limit),triangles,np.eye(3),
                lambda a,b:False,lambda:calls.append('check'))
            return value,list(calls)
        for limit in (0,3,64,119,120,500):
            actual=run(limit)
            with mock.patch.object(target,'_pixel_box_hints',return_value=None):expected=run(limit)
            self.assertEqual(actual,expected)
        with mock.patch.object(target,'interior_overlap',wraps=target.interior_overlap) as predicate:
            run(500);self.assertEqual(predicate.call_count,0)
        # Touches, thin gaps, degenerate edges and non-axis-aligned triangles
        # retain the old uncertainty and exact shared-boundary rules.
        for scale in (1e-12,1.,1e12):
            for delta in (0.,1e-16,1e-6):
                polygons[1]=(1,None,polygons[1][2],polygons[0][3]*scale+[delta,0])
                actual=run(500)
                with mock.patch.object(target,'_pixel_box_hints',return_value=None):expected=run(500)
                self.assertEqual(actual,expected)
