"""Independent eager/rectangle oracles for request-local routing preparation."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from femto.planning import surface_row_turns as turns, quality_constrained as quality
from test_reference_residual_tiles import ResidualTileTests


def original_rectangle(mid, radius, k, **unused):
    z=mid[2]
    if z<=radius:return None
    uv=mid[:2]/z*[k[0,0],k[1,1]]+k[:2,2]
    bound=max(k[0,0],k[1,1])*radius/(z-radius)*(1+np.linalg.norm(mid[:2])/z)
    low=np.floor(uv-bound-.5).astype(int);high=np.ceil(uv+bound+.5).astype(int)
    return int(low[0]),int(low[1]),int(high[0]),int(high[1])


class EagerLinks(turns.SurfaceLinks):
    def _state(self,p,q):return super()._state(p,q)


class CandidateReuseTests(unittest.TestCase):
    def fixture(self):
        rng=np.random.default_rng(942)
        vertices=np.c_[rng.uniform(1,18,(48,2)),np.ones(48)]
        faces=np.arange(48).reshape(-1,3)
        region=dict(view_reference=dict(undistorted_intrinsic=np.eye(3).tolist(),image_size=[20,20]),
                    exclusion_polygons=[[[8.,8.],[12.,8.],[12.,12.],[8.,12.]]])
        allowed=np.ones((20,20),bool);allowed[4:7,3:9]=False
        return SimpleNamespace(vertices=vertices,faces=faces),region,allowed

    def make(self,cls=turns.SurfaceLinks,check=lambda:None):
        mesh,region,allowed=self.fixture()
        return cls(mesh,region,None,check,range(len(mesh.faces)),allowed)

    def test_all_edge_states_equal_eager_with_holes_and_polygons(self):
        old=self.make(EagerLinks);new=self.make()
        self.assertIsInstance(new.edge_state,turns._LazyEdgeStates)
        self.assertLess(len(new.edge_state),len(new.edges))
        self.assertEqual(new.adjacency,old.adjacency)
        self.assertEqual({key:new.edge_state[key] for key in old.edge_state},old.edge_state)
        with self.assertRaises(KeyError):new.edge_state[(0,99999)]

    def test_snapshot_cannot_reuse_mutated_vertices_selection_or_calibration(self):
        old=self.make(EagerLinks);new=self.make()
        new.vertices[:]=0;new.allowed[:]=False;new.k[:]=0;new.polygons[:]=[]
        self.assertEqual({key:new.edge_state[key] for key in old.edge_state},old.edge_state)

    def test_cancellation_does_not_cache_partial_state_and_retry_is_fresh(self):
        stop=[False]
        def check():
            if stop[0]:raise RuntimeError('cancelled')
        router=self.make(check=check)
        key=next(k for k in router.edge_state.keys_allowed if k not in router.edge_state)
        stop[0]=True
        with self.assertRaisesRegex(RuntimeError,'cancelled'):router.edge_state[key]
        self.assertNotIn(key,router.edge_state)
        stop[0]=False
        self.assertEqual(router.edge_state[key],self.make(EagerLinks).edge_state[key])

    def test_half_pixel_corner_keeps_original_eager_budget_failure(self):
        mesh=SimpleNamespace(vertices=np.array([[1.,1.,1.],[1.5-1e-13,1.5,1.],[3.25,2.25,1.]]),faces=np.array([[0,1,2]]))
        region=dict(view_reference=dict(undistorted_intrinsic=np.eye(3).tolist(),image_size=[20,20]))
        for cls in (EagerLinks,turns.SurfaceLinks):
            with self.assertRaisesRegex(ValueError,'Projected segment is too long'):
                cls(mesh,region,None,lambda:None,[0],np.ones((20,20),bool))

    def test_allocation_failure_and_nondefault_error_policy_keep_eager_path(self):
        with patch.object(turns,'_LazyEdgeStates',side_effect=MemoryError):
            self.assertEqual(self.make().edge_state,self.make(EagerLinks).edge_state)
        with np.errstate(all='raise'):
            self.assertIs(type(self.make().edge_state),dict)

    def test_same_search_path_and_empty_or_disconnected_results(self):
        old=self.make(EagerLinks);new=self.make()
        for a,b in [(0,0),(0,1),(2,3),(10,15)]:
            points=[]
            for face in (a,b):
                points.append(dict(xyz_m=old.vertices[old.faces[face]].mean(axis=0).tolist(),triangle_index=face))
            for flags in ({},dict(ignore_forbidden=True,ignore_image=True,ignore_selection=True)):
                self.assertEqual(new._search(*points,**flags),old._search(*points,**flags))

    def test_projection_reuse_preserves_original_arithmetic_and_ulp_boundaries(self):
        rng=np.random.default_rng(433)
        k=np.array([[1143.9777571324541,0,953.869309717688],[0,1144.0402801856196,536.0229852407614],[0,0,1.]])
        points=rng.uniform([-.3,-.3,.05],[.3,.3,2.],(4000,3))
        radii=rng.uniform(0,.06,(4000,2))
        points=np.vstack([points,[[np.nextafter(x,d),0.,1.] for x in (.5,1.5,2.5) for d in (-np.inf,np.inf)]])
        radii=np.vstack([radii,np.zeros((6,2))])
        for point,pair in zip(points,radii):
            prepared=[]
            for radius in pair:
                self.assertEqual(quality._rectangle(point,radius,k,_projection=prepared),original_rectangle(point,radius,k))
        mid=np.array([0.,0.,.5]);prepared=[]
        self.assertIsNone(quality._rectangle(mid,.5,k,_projection=prepared))
        self.assertEqual(prepared,[])

    def test_probe_decisions_margins_details_and_budget_equal_original(self):
        current=ResidualTileTests().envelope();old=ResidualTileTests().envelope()
        rng=np.random.default_rng(770)
        for i in range(300):
            point=rng.uniform([-.08,-.08,.35],[.08,.08,.7]);half=float(rng.uniform(0,.02))
            expected=None
            with patch.object(quality,'_rectangle',original_rectangle):expected=old.probe(point,half,details=True)
            self.assertEqual(current.probe(point,half,details=True),expected)
            self.assertEqual(current.sampled_pixels,old.sampled_pixels)
        for reference in (old,current):reference.sampled_pixels=100000000
        with patch.object(quality,'_rectangle',original_rectangle):
            with self.assertRaisesRegex(ValueError,'超过预算'):old.probe(np.array([0.,0.,.5]),.005)
        with self.assertRaisesRegex(ValueError,'超过预算'):current.probe(np.array([0.,0.,.5]),.005)
        self.assertEqual(current.sampled_pixels,old.sampled_pixels)

if __name__=='__main__':unittest.main()
