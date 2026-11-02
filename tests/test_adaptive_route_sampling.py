"""Area/shape choice is deterministic, bound to geometry, and independently replayable."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import open3d as o3d
import trimesh

from test_fixed_rgb_experience import make_model, K
from test_mesh_slice_planner import grid_mesh
from femto.planning import region_project, surface_workspace
from femto.planning.planning_resolution import create_reduced_model, route_sample_step_mm
from femto.planning.route_sampling import POLICY, decision, validate
from femto.planning.surface_trajectory import SurfaceTrajectory, to_player_plan
from femto.planning.treatment_head import head_parameters
from femto.planning.trajectory_export import load_trajectory


def rectangle(width, height):
    return trimesh.Trimesh(vertices=[[0, 0, .5], [width, 0, .5],
                                    [width, height, .5], [0, height, .5]],
                           faces=[[0, 1, 2], [0, 2, 3]], process=False)


class SamplingDecisionTests(unittest.TestCase):
    def test_face_local_sampling_keeps_bends_and_bounds_each_edge(self):
        from femto.planning.mesh_slice_planner import _resample
        points=np.array([[0,0,0],[.003,0,0],[.003,.003,0],[.003,.003,.012]])
        result=_resample(points,[0,1,2],.005,preserve_face_crossings=True,face_local_spacing=True)
        xyz=np.array([p for p,_ in result])
        self.assertEqual(len(xyz),6)
        for p in points:self.assertTrue(np.any(np.all(xyz==p,axis=1)))
        self.assertTrue(np.all(np.linalg.norm(np.diff(xyz,axis=0),axis=1)<=.005+1e-15))
        self.assertEqual([face for _,face in result],[0,1,2,2,2,2])

    def test_size_selects_finer_small_regions_and_coarser_large_regions(self):
        results=[decision(rectangle(s,s))['sample_step_mm'] for s in (.012,.030,.045,.12)]
        self.assertEqual(results,[1.,2.,3.,5.])

    def test_equal_area_narrow_strip_is_finer(self):
        square=decision(rectangle(.1,.1));strip=decision(rectangle(1.,.01))
        self.assertAlmostEqual(square['area_mm2'],strip['area_mm2'],places=2)
        self.assertEqual(square['sample_step_mm'],5.)
        self.assertEqual(strip['sample_step_mm'],2.)

    def test_hole_boundaries_reduce_characteristic_width(self):
        mesh=grid_mesh(width=.1,height=.1,cell=.005)
        full=decision(mesh)
        centers=mesh.triangles_center
        perforated=mesh.submesh([np.flatnonzero((np.abs(centers[:,0])>.025)
                                             | (np.abs(centers[:,1])>.025))],append=True)
        self.assertLess(decision(perforated)['characteristic_width_mm'],full['characteristic_width_mm'])

    def test_decision_is_independent_of_face_density(self):
        mesh=rectangle(.1,.1)
        dense=mesh.subdivide().subdivide()
        self.assertAlmostEqual(decision(mesh)['area_mm2'],decision(dense)['area_mm2'])
        self.assertEqual(decision(mesh)['sample_step_mm'],decision(dense)['sample_step_mm'])

    def test_saved_choice_requires_same_geometry_and_version(self):
        mesh=rectangle(.12,.12);record=decision(mesh)
        self.assertEqual(validate(record,mesh),record)
        with self.assertRaisesRegex(ValueError,'网格不一致'):validate(record,rectangle(.1,.1))
        for key,value in [('policy','future'),('sample_step_mm',3.),('area_mm2',float('nan')),
                          ('changes_row_spacing',True),('preserve_face_crossings',1)]:
            with self.subTest(key=key),self.assertRaises(ValueError):validate(dict(record,**{key:value}))


class AdaptiveRouteReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.addClassCleanup(cls.temp.cleanup)
        cls.root=Path(cls.temp.name)
        mesh=grid_mesh(width=.12,height=.08,cell=.005)
        original=make_model(cls.root,o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(mesh.vertices),o3d.utility.Vector3iVector(mesh.faces)))
        original.view_reference['undistorted_intrinsic']=K.tolist()
        original.store.origin_reference=dict(kind='initial_color_camera',provenance='synthetic_test')
        o3d.io.write_triangle_mesh(original.store.source,original.mesh)
        cls.original_view=deepcopy(original.view_reference)
        cls.model=create_reduced_model(original,cls.root/'auto',stride=4,
            route_sample_step_mm=5.,sampling_policy=POLICY)
        assert original.view_reference==cls.original_view
        mask=cls.model.projection.valid.copy();height,width=mask.shape
        cls.region=cls.model.region([[0,0],[width-1,0],[width-1,height-1],[0,height-1]],
            'synthetic adaptive sampling','',selection=dict(mask=mask,method='work_only'))
        cls.head=head_parameters('from_effective_width',effective_treatment_width_mm=4.,overlap_percent=25.)
        cls.plan=surface_workspace.generate(cls.model,cls.region,cls.head)

    def test_head_row_spacing_full_geometry_and_face_crossings_retained(self):
        params=self.plan['surface_trajectory']['planner']['parameters']
        self.assertEqual(self.plan['treatment_head'],self.head)
        self.assertEqual(params['row_spacing_mm'],3.)
        self.assertTrue(params['preserve_face_crossings'])
        self.assertEqual(params['sample_step_mm'],5.)
        self.assertEqual(params['sampling_decision']['policy'],POLICY)
        surface_workspace.validate(self.plan,self.model.mesh,self.region)
        self.assertIn('沿线自动取点',surface_workspace.continuity_text(self.plan))

    def test_native_and_portable_independent_replay(self):
        native=surface_workspace.export(self.model,self.region,self.plan)
        self.assertEqual(to_player_plan(load_trajectory(native,reproduce=True))['segments'],self.plan['segments'])
        archive=region_project.save_project(self.model,self.region,self.root/'result.region.zip',self.plan,
            allow_replace=False,catalog_root=self.root/'catalogue')
        restored=region_project.open_project(archive,workspace=self.root/'opened')
        self.assertEqual(restored.saved_plan['surface_trajectory'],self.plan['surface_trajectory'])
        self.assertEqual(restored.view_reference,self.model.view_reference)
        np.testing.assert_array_equal(restored.display_rgb,self.model.display_rgb)
        np.testing.assert_array_equal(np.asarray(restored.mesh.vertices),np.asarray(self.model.mesh.vertices))

    def test_forged_step_and_missing_decision_rejected(self):
        for field,value in [('sample_step_mm',3.),('sampling_decision',None)]:
            forged=deepcopy(self.plan['surface_trajectory']);forged['planner']['parameters'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):SurfaceTrajectory.from_dict(forged)
        with self.assertRaises(ValueError):route_sample_step_mm(self.model.view_reference)


if __name__=='__main__':unittest.main()
