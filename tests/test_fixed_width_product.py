"""Product-mode checks on measured synthetic surfaces and saved head contracts."""
import tempfile
import unittest
from unittest.mock import patch
from copy import deepcopy
import numpy as np
import open3d as o3d
from test_fixed_rgb_experience import make_model, K
from test_mesh_slice_planner import grid_mesh
from femto.planning import surface_workspace as workspace
from femto.planning import planning_revision
from femto.planning.treatment_head import head_parameters, _inputs
from femto.planning.continuous_quality import DENSE_ROUTING


class FixedWidthProductTests(unittest.TestCase):
    def test_synthetic_shapes_keep_whole_edge_checks(self):
        for shape in ('rectangle', 'curved', 'concave', 'narrow', 'hole', 'multiple'):
            with self.subTest(shape=shape), tempfile.TemporaryDirectory() as folder:
                mesh=grid_mesh(width=.12,height=.08,cell=.005)
                if shape=='curved':
                    mesh.vertices[:,2]+=1.5*mesh.vertices[:,0]**2
                model=make_model(folder,o3d.geometry.TriangleMesh(
                    o3d.utility.Vector3dVector(mesh.vertices),o3d.utility.Vector3iVector(mesh.faces)))
                model.view_reference['undistorted_intrinsic']=K.tolist()
                model.store.origin_reference=dict(kind='initial_color_camera',provenance='synthetic_test')
                mask=model.projection.valid.copy()
                if shape=='concave': mask[:50,50:]=False
                if shape=='hole': mask[47:53,47:53]=False
                if shape=='multiple': mask[:,50:54]=False
                if shape=='narrow':
                    mask[:,:49]=False;mask[:,51:]=False
                region=model.region([[30,30],[70,30],[70,70],[30,70]],'test','',
                                    selection=dict(mask=mask,method='work_only'))
                head=head_parameters('fixed_width_auto',effective_treatment_width_mm=5,overlap_percent=20)
                if shape=='narrow':
                    with self.assertRaises(ValueError):workspace.generate(model,region,head)
                    continue
                # Explicitly retain the previous residual-v2 mode as a full
                # compatibility fixture after the default moves to patches.
                capture=planning_revision.capture
                def legacy_capture(*args,**kwargs):
                    kwargs.setdefault('algorithms',planning_revision.RESIDUAL_LAYOUT_ALGORITHMS)
                    return capture(*args,**kwargs)
                with patch.object(planning_revision,'capture',side_effect=legacy_capture):
                    plan=workspace.generate(model,region,head)
                trajectory=workspace.validate(plan,model.mesh,region)
                if shape=='rectangle':
                    from femto.planning.fixed_width_layout import validate
                    parameters=trajectory.planner['parameters']
                    tampered=deepcopy(parameters['fixed_width_layout'])
                    tampered['fragments'][0]['positions_m'][0]+=.001
                    with self.assertRaisesRegex(ValueError,'排布决策'):
                        validate(tampered,workspace.canonical_region_mesh(model.mesh,region),region,parameters,lambda:None)
                    missing=deepcopy(parameters['fixed_width_layout'])
                    missing['fragments'].pop()
                    with self.assertRaisesRegex(ValueError,'缺少表面片段'):
                        validate(missing,workspace.canonical_region_mesh(model.mesh,region),region,parameters,lambda:None)
                self.assertEqual(trajectory.planner['parameters']['effective_width_mm'],5)
                self.assertEqual(trajectory.planner['parameters']['continuous_routing'],DENSE_ROUTING)
                self.assertFalse(plan['treatment_head']['execution_ready'])
                if shape=='multiple':self.assertGreater(trajectory.statistics['inter_block_gap_count'],0)
                self.assertEqual(plan['treatment_head'],head_parameters(head['mode'],**_inputs(head)))
                # Coverage is a separate calculation: no completed scan layout
                # may silently turn unknown target cells into confirmed area.
                from femto.planning import coverage_workflow,coverage_certificate
                certificate,ledger=coverage_workflow.calculate(model,region,plan,
                    budget=coverage_certificate.CoverageBudget(max_depth=1,max_added_cells=3000))
                attached=coverage_workflow.attach(model,region,plan,certificate,ledger)
                self.assertEqual(attached['segments'],plan['segments'])
                self.assertFalse(certificate.metadata['execution_ready'])
                self.assertFalse(certificate.metadata['physical_accuracy_verified'])
                self.assertFalse(np.any(model.coverage_display.mask(1)&~mask))
                self.assertGreater(certificate.summary['original_cell_count'],0)
                self.assertAlmostEqual(certificate.summary['located_nominal_area_m2'],
                    certificate.summary['confirmed_nominal_area_m2']+certificate.summary['unconfirmed_nominal_area_m2'])


if __name__=='__main__':unittest.main()
