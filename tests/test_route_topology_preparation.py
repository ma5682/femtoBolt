"""Exact topology, projection bytes and edge-check budgets for route preparation."""
from collections import defaultdict
from unittest import TestCase,main
from unittest.mock import patch
import numpy as np
from femto.geometry import triangle_components as components
from femto.planning import quality_constrained as quality, surface_trajectory as trajectory
import test_reference_residual_tiles as residual_fixture
import test_serpentine_prepared_checks as edge_fixture


def reference_components(raw):
    faces=np.asarray(raw,dtype=np.int64);edges=defaultdict(list)
    for fi,face in enumerate(faces):
        a,b,c=map(int,face[:3])
        for pair in ((a,b),(b,c),(c,a)):edges[tuple(sorted(pair))].append(fi)
    adjacent=[set() for _ in faces]
    for ids in edges.values():
        for a,b in zip(ids,ids[1:]):adjacent[a].add(b);adjacent[b].add(a)
    result=np.full(len(faces),-1,np.int64);component=0
    for start in range(len(faces)):
        if result[start]>=0:continue
        result[start]=component;stack=[start]
        while stack:
            for nxt in adjacent[stack.pop()]:
                if result[nxt]<0:result[nxt]=component;stack.append(nxt)
        component+=1
    return result


class RouteTopologyPreparationTests(TestCase):
    def test_integer_components_match_independent_graph_and_preserve_input(self):
        rng=np.random.default_rng(911)
        cases=[np.empty((0,3),int),np.array([[0,1,2],[2,3,4]]),
               np.array([[0,1,2],[1,0,3],[1,0,4],[7,7,7],[7,7,8]]),
               np.array([[-1,0,1],[0,-1,2]]),np.array([[0,1,2**31],[1,0,2**31+1]])]
        for n in (2,39,200,1000):
            for vertices in (12,100,6000):cases.append(rng.integers(0,vertices,(n,3)))
        for faces in cases:
            before=faces.copy();np.testing.assert_array_equal(trajectory.face_components(faces),reference_components(faces))
            np.testing.assert_array_equal(faces,before)
        long=np.column_stack((np.arange(5000),np.arange(1,5001),np.arange(2,5002)))
        np.testing.assert_array_equal(trajectory.face_components(long),np.zeros(5000,dtype=np.int64))

    def test_component_limits_and_allocation_failure_use_original_union(self):
        faces=np.array([[0,1,2],[2,1,3],[7,8,9],[8,7,10],[3,3,3]])
        expected=reference_components(faces)
        for name,value in (('MAX_WORK_BYTES',0),('MAX_ROUNDS',0)):
            with patch.object(components,name,value):np.testing.assert_array_equal(trajectory.face_components(faces),expected)
        with patch.object(components,'_labels',side_effect=MemoryError):
            np.testing.assert_array_equal(trajectory.face_components(faces),expected)

    def test_batch_centres_and_original_norms_are_byte_identical_at_ulp_edges(self):
        rng=np.random.default_rng(582)
        k=np.array([[1143.9777571324541,0,953.869309717688],[0,1144.0402801856196,536.0229852407614],[0,0,1.]])
        points=rng.uniform([-.3,-.3,.05],[.3,.3,2.],(4000,3))
        points=np.vstack((points,[[np.nextafter(x,d),-x,1.] for x in (.5,1.5,2.5) for d in (-np.inf,np.inf)]))
        uv,radial=quality._batch_projections(points,k,lambda:None)
        for i,point in enumerate(points):
            expected=point[:2]/point[2]*[k[0,0],k[1,1]]+k[:2,2]
            self.assertEqual(uv[i].tobytes(),expected.tobytes())
            self.assertEqual(radial[i].tobytes(),np.linalg.norm(point[:2]).tobytes())
            for radius in (0.,.005,np.nextafter(.005,np.inf)):
                self.assertEqual(quality._rectangle(point,radius,k,_projection=(uv[i],radial[i])),quality._rectangle(point,radius,k))

    def test_batch_projection_fallback_and_cancel_leave_inputs_unchanged(self):
        points=np.full((128,3),.5);k=np.eye(3);before=points.copy()
        for values,calibration in ((points[:1],k),(points.astype(np.float32),k),(points,k.astype(np.float32)),
                                   (np.zeros((128,3)),k),(points*1e7,k),(points,np.full((3,3),np.nan))):
            self.assertIsNone(quality._batch_projections(values,calibration,lambda:None))
        with np.errstate(all='raise'):self.assertIsNone(quality._batch_projections(points,k,lambda:None))
        with patch.object(quality.np,'empty',side_effect=MemoryError):self.assertIsNone(quality._batch_projections(points,k,lambda:None))
        for error in (RuntimeError('cancel'),MemoryError('check failed')):
            def check():raise error
            with self.assertRaises(type(error)):quality._batch_projections(points,k,check)
        np.testing.assert_array_equal(points,before)
        self.assertIsNotNone(quality._batch_projections(points,k,lambda:None))

    def test_projection_preparation_preserves_probe_details_margins_and_pixel_charges(self):
        current=residual_fixture.ResidualTileTests().envelope();original=residual_fixture.ResidualTileTests().envelope()
        rng=np.random.default_rng(789);points=rng.uniform([-.1,-.1,.35],[.1,.1,.7],(128,3));half=rng.uniform(0,.02,128)
        uv,radial=quality._batch_projections(points,current.k,lambda:None)
        for i,(point,radius) in enumerate(zip(points,half)):
            width=(.5,5.,20.)[i%3]
            expected=original.probe(point,radius,details=True,width_mm=width)
            self.assertEqual(current.probe(point,radius,details=True,width_mm=width,_projection=(uv[i],radial[i])),expected)
            self.assertEqual(current.sampled_pixels,original.sampled_pixels)
        current.sampled_pixels=original.sampled_pixels=100000000
        for reference,kwargs in ((original,{}),(current,dict(_projection=(uv[0],radial[0])))):
            with self.assertRaisesRegex(ValueError,'超过预算'):reference.probe(points[0],half[0],**kwargs)
        self.assertEqual(current.sampled_pixels,original.sampled_pixels)

    def test_whole_edges_and_diagnostics_match_forced_scalar_preparation(self):
        fixture=edge_fixture.PreparedQualityChecksTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        record=fixture.record();rng=np.random.default_rng(559)
        starts=np.c_[rng.uniform(-.19,.19,(128,2)),np.full(128,.5)]
        ends=starts.copy();ends[:,:2]+=rng.uniform(-.008,.008,(128,2))
        segments=[fixture.segment(a,b) for a,b in zip(starts,ends)]
        for allowed in (fixture.allowed,np.array([False,True])):
            diag=[];actual=quality.edge_checks(segments,fixture.mesh,record,5.,allowed_faces=allowed,diagnostics=diag)
            legacy=[]
            with patch.object(quality,'_batch_projections',return_value=None):
                expected=quality.edge_checks(segments,fixture.mesh,record,5.,allowed_faces=allowed,diagnostics=legacy)
            fixture.assert_same(actual,expected);self.assertEqual(diag,legacy)


if __name__=='__main__':main()
