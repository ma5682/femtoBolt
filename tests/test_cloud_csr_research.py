"""The experimental complete-neighborhood counter vs the native old query."""
import unittest
import numpy as np
import open3d as o3d
from femto.geometry.cloud_csr_research import CellRadiusCounter
from femto.performance_detail import collect


def counter(points,normals,**kwargs):
    cloud=o3d.geometry.PointCloud()
    cloud.points=o3d.utility.Vector3dVector(points)
    tree=o3d.geometry.KDTreeFlann(cloud)
    def legacy(indices):
        return sum(int(np.any(np.abs((points[np.asarray(tree.search_radius_vector_3d(points[i],.01)[1])]-points[i])@normals[i])>.004)) for i in indices)
    return CellRadiusCounter(points,normals,legacy,**kwargs),legacy


class CellCounterTests(unittest.TestCase):
    def test_random_curved_double_layers_nonunit_and_large_origins(self):
        rng=np.random.default_rng(1251)
        xy=rng.uniform(-.03,.03,(180,2))
        base=np.c_[xy,.5+.004*np.cos(xy[:,0]*50)]
        for p in (base,np.r_[base,base+[0,0,.006]],rng.uniform(-.03,.03,(400,3))):
            for origin in (0,np.array([1e6,-1e6,1e6])):
                points=p+origin;normals=rng.normal(size=points.shape)
                fast,old=counter(points,normals,max_pairs=97)
                ids=np.arange(len(points))
                self.assertEqual(fast(ids),old(ids))
                self.assertEqual(fast(ids[::-2]),old(ids[::-2]))

    def test_radius_height_and_bin_boundaries(self):
        for r in (.01,np.nextafter(.01,0),np.nextafter(.01,np.inf)):
            for h in (.004,np.nextafter(.004,0),np.nextafter(.004,np.inf)):
                for origin in (0.,.01,np.nextafter(.01,0),-.02,1e6):
                    p=np.array([[origin,0,0],[origin+r,0,0],[origin,h,0],[origin,0,.003]])
                    n=np.tile([1.,1.,1.],(len(p),1))
                    fast,old=counter(p,n,max_pairs=1)
                    self.assertEqual(fast(np.arange(len(p))),old(np.arange(len(p))))

    def test_dense_cell_bounded_buffers_and_cancellation(self):
        rng=np.random.default_rng(1271)
        p=rng.uniform(0,.001,(400,3));n=np.tile([0.,0.,1.],(len(p),1))
        fast,old=counter(p,n,max_pairs=37)
        with collect() as detail:self.assertEqual(fast(np.arange(30)),old(np.arange(30)))
        row=detail.snapshot()['rows']['cloud.csr_work']['counts']
        self.assertEqual(row['candidate_pairs'],30*len(p))
        self.assertGreater(row['pair_batches'],row['candidate_pairs']//37)
        fast.check=lambda:(_ for _ in ()).throw(InterruptedError('cancel'))
        with self.assertRaises(InterruptedError):fast(np.arange(30))


if __name__=='__main__':unittest.main()
