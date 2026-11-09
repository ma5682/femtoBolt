"""Versioned tangent sampling with whole-edge checks and archive replay."""
from copy import deepcopy
import tempfile
from pathlib import Path
from unittest import TestCase

import numpy as np
import open3d as o3d
from femto.planning.route_sampling import CURVATURE_POLICY, POLICY, decision, validate
from femto.planning.mesh_slice_planner import _resample
from femto.planning import surface_workspace, region_project
from femto.planning.planning_resolution import create_reduced_model, route_sample_step_mm
from femto.planning.treatment_head import head_parameters
from femto.planning.trajectory_export import load_trajectory
from femto.planning.surface_trajectory import to_player_plan
from test_fixed_rgb_experience import make_model, K
from test_adaptive_route_sampling import rectangle
from test_mesh_slice_planner import grid_mesh


class CurvatureSamplingTests(TestCase):
    def test_flat_sparse_bend_dense_boundary_dense_and_face_crossings(self):
        record=decision(rectangle(.045,.045),CURVATURE_POLICY)
        self.assertEqual(record['shape_step_mm'],3.)
        self.assertEqual(record['sample_step_mm'],5.)
        points=np.array([[0,0,0],[.02,0,0],[.02,.02,0],[.02,.04,0]])
        result=_resample(points,[0,1,2],.005,preserve_face_crossings=True,
                         face_local_spacing=True,sampling_decision=record)
        xyz=np.array([p for p,_ in result]);lengths=np.linalg.norm(np.diff(xyz,axis=0),axis=1)
        self.assertTrue(np.all(lengths<=.005+1e-12))
        # Both sides of the 90 degree bend are sampled at <= 1mm.
        for face in (0,1):
            samples=np.array([p for p,f in result if f==face])
            self.assertTrue(np.all(np.linalg.norm(np.diff(samples,axis=0),axis=1)<=.001+1e-12))
        smooth=np.array([p for p,f in result if f==2])
        self.assertGreater(np.linalg.norm(np.diff(smooth,axis=0),axis=1).max(),.003)
        self.assertLessEqual(lengths[-1],.001+1e-12)
        for point in points:self.assertTrue(np.any(np.all(xyz==point,axis=1)))
        self.assertEqual(result[-1][1],2)

    def test_version_binding_tampering_and_old_policy_retained(self):
        mesh=rectangle(.045,.045);record=decision(mesh,CURVATURE_POLICY)
        self.assertEqual(validate(record,mesh),record)
        for field,value in [('bend_step_mm',[5.,5.]),('endpoint_band_mm',0.),('sample_step_mm',3.)]:
            with self.assertRaises(ValueError):validate(dict(record,**{field:value}),mesh)
        self.assertEqual(decision(mesh)['policy'],POLICY)
        self.assertEqual(decision(mesh)['sample_step_mm'],3.)
        view={'derivation':dict(schema='reduced-planning-reference-v1',stride=4,
            profile='stride4-original-mesh-'+CURVATURE_POLICY,route_sample_step_mm=5.,sampling_policy=CURVATURE_POLICY)}
        with self.assertRaises(ValueError):route_sample_step_mm(view,decision=decision(mesh))

    def test_generation_save_native_portable_and_independent_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            mesh=grid_mesh(width=.12,height=.08,cell=.02)
            mesh.vertices[:,2]+=.005*np.abs(mesh.vertices[:,0]/.06)
            source=make_model(root,o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),
                                                          o3d.utility.Vector3iVector(mesh.faces)))
            source.view_reference['undistorted_intrinsic']=K.tolist()
            # Use this synthetic curved mesh's raycast depths, then quantize to
            # sensor millimetres. A constant plane depth would be conflicting
            # evidence and must fail the existing width/quality checks.
            from femto.annotation.fixed_rgb_geometry import FixedProjection
            depth=np.zeros((100,100),np.uint16)
            good=np.isfinite(source.projection.z)&(source.projection.z>0)
            depth[good]=np.rint(source.projection.z[good]*1000).astype(np.uint16)
            source.projection=FixedProjection(source.mesh,K,depth)
            source.store.origin_reference=dict(kind='initial_color_camera',provenance='synthetic_test')
            o3d.io.write_triangle_mesh(source.store.source,source.mesh)
            model=create_reduced_model(source,root/'adaptive',stride=4,
                route_sample_step_mm=5.,sampling_policy=CURVATURE_POLICY)
            mask=model.projection.valid.copy();h,w=mask.shape
            region=model.region([[0,0],[w-1,0],[w-1,h-1],[0,h-1]],'tangent test','',
                                selection=dict(mask=mask,method='work_only'))
            head=head_parameters('from_effective_width',effective_treatment_width_mm=4.,overlap_percent=25.)
            plan=surface_workspace.generate(model,region,head)
            params=plan['surface_trajectory']['planner']['parameters']
            self.assertEqual(params['sampling_decision']['policy'],CURVATURE_POLICY)
            self.assertEqual(params['row_spacing_mm'],3.)
            surface_workspace.validate(plan,model.mesh,region)
            native=surface_workspace.export(model,region,plan)
            self.assertEqual(to_player_plan(load_trajectory(native,reproduce=True))['segments'],plan['segments'])
            archive=region_project.save_project(model,region,root/'result.region.zip',plan,
                allow_replace=False,catalog_root=root/'catalogue')
            restored=region_project.open_project(archive,workspace=root/'restored')
            self.assertEqual(restored.saved_plan['surface_trajectory'],plan['surface_trajectory'])
            self.assertEqual(restored.view_reference,model.view_reference)
            from femto.geometry.local_region import model_selection
            self.assertIsNone(model_selection(restored))
            # Existing whole-edge proof rejects a displaced scan point.
            bad=deepcopy(plan)
            bad['surface_trajectory']['ordered_items'][0]['points'][0]['xyz_m'][2]+=.02
            with self.assertRaises(ValueError):surface_workspace.validate(bad,model.mesh,region)
