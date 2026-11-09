"""Topology/order oracle and measured-cloud regression for sparse pixel surfaces."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import cv2
import numpy as np
import open3d as o3d

from femto.geometry.rgbd_frames import prepare_rgbd
from femto.geometry.single_frame_mesh import build_pixel_mesh, build_pixel_mesh_and_cloud
from femto.pipeline.local_single_surface import prepare_surface


def inputs(depth, *, distorted=False):
    h, w = depth.shape
    intrinsic = SimpleNamespace(width=w, height=h, fx=650., fy=640., cx=(w-1)/2, cy=(h-1)/2)
    distortion = dict(model='BROWN_CONRADY_K6', **dict.fromkeys(('k1','k2','p1','p2','k3','k4','k5','k6'), 0.))
    if distorted:
        distortion.update(k1=.03, k2=-.01, p1=.001, p2=-.002)
    color = np.random.default_rng(11).integers(0, 256, (h,w,3), dtype=np.uint8)
    pose = np.eye(4)
    pose[:3,:3] = cv2.Rodrigues(np.array([.08,-.05,.1]))[0]
    pose[:3,3] = [.01,-.02,.03]
    return color, intrinsic, distortion, pose


def scalar_faces(depth, intrinsic, distortion):
    """Independent full-image oracle in pixel IDs, including original group order."""
    h,w = depth.shape
    k = np.array([[intrinsic.fx,0,intrinsic.cx],[0,intrinsic.fy,intrinsic.cy],[0,0,1.]])
    coeff = np.array([distortion[key] for key in ('k1','k2','p1','p2','k3','k4','k5','k6')])
    y,x = np.nonzero(np.isfinite(depth)&(depth>=250)&(depth<=2210))
    rays = cv2.undistortPoints(np.c_[x,y].astype(float).reshape(-1,1,2),k,coeff).reshape(-1,2)
    points = {int(row*w+col):np.r_[ray*float(depth[row,col])/1000,float(depth[row,col])/1000]
              for row,col,ray in zip(y,x,rays)}
    groups = [[],[],[],[]]
    def cost(a,b):
        return abs(points[a][2]-points[b][2]) if a in points and b in points else np.inf
    for row in range(h-1):
        for col in range(w-1):
            a=row*w+col;b=a+1;c=a+w;d=c+1
            selected = ((2,(a,d,b)),(3,(a,c,d))) if cost(a,d)<cost(b,c) else ((0,(a,c,b)),(1,(b,c,d)))
            for group,face in selected:
                if not all(i in points for i in face):continue
                for i,j in zip(face,face[1:]+face[:1]):
                    limit=max(.010,.02*min(points[i][2],points[j][2]))
                    if abs(points[i][2]-points[j][2])>limit or np.linalg.norm(points[i]-points[j])>np.sqrt(2)*limit:
                        break
                else:groups[group].append(face)
    return np.array([face for group in groups for face in group],dtype=np.int64).reshape(-1,3)


class PixelSurfaceOptimizationTests(unittest.TestCase):
    def test_sparse_cells_keep_full_image_topology_order_and_uv(self):
        rng=np.random.default_rng(42)
        for distorted in (False,True):
            for _ in range(8):
                depth=rng.integers(500,514,(24,31)).astype(float)
                depth[rng.random(depth.shape)<.7]=0
                depth[8:11,10:14]=500  # Ensure one connected component.
                depth[2,3]=np.nan;depth[4,5]=np.inf;depth[3,4]=249;depth[5,6]=2211
                args=inputs(depth,distorted=distorted)
                original=depth.copy()
                mesh,uv,report=build_pixel_mesh(depth,*args)
                h,w=depth.shape
                pixels=np.rint((1-uv[:,1])*h-.5).astype(int)*w+np.rint(uv[:,0]*w-.5).astype(int)
                np.testing.assert_array_equal(pixels[np.asarray(mesh.triangles)],scalar_faces(depth,args[1],args[2]))
                np.testing.assert_array_equal(depth,original)
                self.assertEqual(report['candidate_triangles'],2*(h-1)*(w-1))
                self.assertLess(report['generated_candidate_triangles'],report['candidate_triangles'])

    def test_three_corner_cells_border_and_separated_components(self):
        depth=np.zeros((80,100),np.uint16)
        depth[0,0]=depth[0,1]=depth[1,0]=500  # Valid half-cell at image edge.
        depth[-3:,-4:]=800
        depth[40,50]=600  # Isolated observation remains outside the mesh.
        color,intr,dist,pose=inputs(depth)
        mesh,uv,report,cloud=build_pixel_mesh_and_cloud(depth,color,intr,dist,pose)
        self.assertEqual(len(cloud.points),int(np.count_nonzero(depth)))
        self.assertEqual(report['retained_points_without_triangle'],1)
        pixels=np.rint((1-uv[:,1])*80-.5).astype(int)*100+np.rint(uv[:,0]*100-.5).astype(int)
        np.testing.assert_array_equal(pixels[np.asarray(mesh.triangles)],scalar_faces(depth,intr,dist))
        self.assertLessEqual(report['generated_candidate_triangles'],26)
        for thin in (np.full((1,10),500),np.full((10,1),500),np.array([[500]])):
            with self.assertRaisesRegex(ValueError,'No pixel-adjacent triangle'):
                build_pixel_mesh(thin,*inputs(thin))

    def test_shared_cloud_keeps_pixel_order_color_pose_and_isolated_depth(self):
        depth=np.zeros((35,40),np.uint16)
        depth[4:20,8:29]=np.arange(21,dtype=np.uint16)+700
        depth[30,35]=2210
        color,intr,dist,pose=inputs(depth,distorted=True)
        mesh,uv,report,cloud=build_pixel_mesh_and_cloud(depth,color,intr,dist,pose)
        previous=build_pixel_mesh(depth,color,intr,dist,pose)
        for key in ('vertices','triangles','vertex_colors','vertex_normals'):
            np.testing.assert_array_equal(getattr(mesh,key),getattr(previous[0],key))
        np.testing.assert_array_equal(uv,previous[1])
        y,x=np.nonzero(depth)
        coeff=np.array([dist[key] for key in ('k1','k2','p1','p2','k3','k4','k5','k6')])
        k=np.array([[intr.fx,0,intr.cx],[0,intr.fy,intr.cy],[0,0,1.]])
        ray=cv2.undistortPoints(np.c_[x,y].astype(float).reshape(-1,1,2),k,coeff).reshape(-1,2)
        z=depth[y,x].astype(float)/1000
        np.testing.assert_allclose(cloud.points,np.c_[ray*z[:,None],z]@pose[:3,:3].T+pose[:3,3],rtol=0,atol=1e-15)
        np.testing.assert_array_equal(cloud.colors,color[y,x,::-1]/255.)

    def test_dense_and_sparse_traversals_match_on_strided_depth_arrays(self):
        for missing in (0,5,20,21,30):
            base=np.full((18,22),800.,dtype=float)
            depth=base[::2,::2]
            for y,x in np.ndindex(depth.shape):
                depth[y,x]+=((x+y)%3)
            interior=[(y,x) for y in range(1,8) for x in range(1,10)]
            for y,x in interior[:missing]:depth[y,x]=0
            args=inputs(depth)
            mesh,uv,_=build_pixel_mesh(depth,*args)
            h,w=depth.shape
            pixel_ids=np.rint((1-uv[:,1])*h-.5).astype(int)*w+np.rint(uv[:,0]*w-.5).astype(int)
            np.testing.assert_array_equal(pixel_ids[np.asarray(mesh.triangles)],scalar_faces(depth,args[1],args[2]))

    def test_local_surface_matches_old_rgbd_cloud_with_float_depth_roundoff(self):
        depth=np.full((50,70),731,np.uint16);color,intr,dist,pose=inputs(depth)
        k=np.array([[intr.fx,0,intr.cx],[0,intr.fy,intr.cy],[0,0,1.]])
        y,x=np.indices(depth.shape,dtype=np.float32)
        mask=np.zeros_like(depth,bool);mask[10:30,20:45]=True;mask[45,60]=True
        selected=np.where(mask,depth,0).astype(np.uint16)
        calls=[]
        def filter_depth(value,*args):
            calls.append(value.copy());return np.where(mask,value,0).astype(np.uint16)
        with TemporaryDirectory() as temp:
            cloud,mesh,initial,measured,report=prepare_surface(color,depth,(x,y,k),pose,.25,2.21,
                SimpleNamespace(filter=filter_depth),0,Path(temp))
            self.assertTrue((Path(temp)/'initial_observed_points_m.ply').exists())
        rgbd,calibrated,_,_=prepare_rgbd(color,selected,x,y,k,1.,.25,2.21)
        old=o3d.geometry.PointCloud.create_from_rgbd_image(rgbd,calibrated).transform(pose)
        # Open3D RGBD stores depth as float32 metres; shared mesh rays use float64.
        np.testing.assert_allclose(cloud.points,old.points,rtol=0,atol=1.2e-7)
        np.testing.assert_array_equal(cloud.colors,old.colors)
        np.testing.assert_array_equal(initial.points,np.asarray(cloud.points)[::16])
        np.testing.assert_array_equal(measured,selected)
        self.assertEqual(len(calls),1)
        self.assertEqual(report['tsdf_integrations'],0)


if __name__ == '__main__':
    unittest.main()
