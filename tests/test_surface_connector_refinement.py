"""Same-corridor shortening, unchanged scans, and independent version replay."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import open3d as o3d

from femto.planning import surface_connector_refinement as module,scan_visit_order as order
from femto.planning.surface_trajectory import ScanSegment,_point_on_triangle
from femto.planning.endpoint_preserving_detour import _signature
from femto.planning.surface_link_optimization import optimize_surface_path


class SurfaceConnectorRefinementTests(unittest.TestCase):
    def setUp(self):
        self.mesh=SimpleNamespace(vertices=np.asarray([[0.,0.,.4],[.04,0.,.4],[.04,.04,.4],[0.,.04,.4]]),
                                  faces=np.asarray([[0,1,2],[0,2,3]]))
        def point(x,y,row,face,role='scan'):
            return dict(xyz_m=[x,y,.4],row_index=row,triangle_index=face,role=role,
                        pixel=[50+x*250,50+y*250],geometry_support='measured_mesh')
        self.segments=[ScanSegment(0,0,[point(.035,.003,0,0),point(.03,.005,0,0)],0),
                       ScanSegment(1,1,[point(.005,.03,1,1),point(.003,.035,1,1)],0)]
        self.visits=[(0,False),(1,False)]
        samples=[point(.03,.03,1,0,'connector'),point(.005,.03,1,1,'connector')]
        self.links=[dict(from_segment=0,to_segment=1,connected=True,added_edges=2,samples=samples,reason='verified_surface_path')]
        self.k=np.asarray([[100.,0.,50.],[0.,100.,50.],[0.,0.,1.]])

    def run_refine(self,checker=lambda edges:[True]*len(edges),check=lambda:None):
        return module.refine(self.mesh,self.segments,self.visits,self.links,.005,.005,self.k,np.asarray([100,100]),checker,check)

    def test_shorter_supported_path_retains_scans_and_endpoints(self):
        before=deepcopy((self.segments,self.visits,self.links))
        rebuilt,links,record=self.run_refine()
        self.assertTrue(record['adopted'])
        self.assertLess(record['selected']['connector_length_m'],record['baseline']['connector_length_m']-.005)
        self.assertTrue(all(record['selected'][k]<=v+1e-12 for k,v in record['baseline'].items()))
        self.assertEqual(_signature(rebuilt[0],lambda:None),_signature([(s.block_index,s.points) for s in self.segments],lambda:None))
        tail=self.segments[0].points[-1]['xyz_m']
        for point in links[0]['samples']:
            for p in (tail,point['xyz_m']):self.assertTrue(_point_on_triangle(p,self.mesh.vertices,self.mesh.faces[point['triangle_index']]))
            tail=point['xyz_m']
        self.assertEqual(tail,self.segments[1].points[0]['xyz_m'])
        self.assertEqual(before,(self.segments,self.visits,self.links))
        self.assertEqual(self.run_refine()[1:],(links,record))

    def test_rejection_budget_and_cancellation_cannot_publish_partial_link(self):
        _,links,record=self.run_refine(lambda edges:[False]*len(edges))
        self.assertFalse(record['adopted']);self.assertEqual(links,self.links)
        with patch.object(module,'MAX_CHECKS',1):
            _,links,record=self.run_refine()
        self.assertFalse(record['adopted']);self.assertEqual(links,self.links)
        self.assertEqual(record['attempts'][-1]['status'],'budget_exhausted');self.assertLessEqual(record['edge_checks'],1)
        def cancel():raise RuntimeError('cancelled')
        with self.assertRaisesRegex(RuntimeError,'cancelled'):self.run_refine(check=cancel)

    def test_sampling_budget_retains_complete_baseline_route(self):
        with patch.object(module,'POINT_BUDGET',6):
            rebuilt,links,record=self.run_refine()
        self.assertFalse(record['adopted']);self.assertEqual(links,self.links)
        self.assertEqual(record['attempts'][-1]['status'],'sample_budget_exhausted')
        self.assertLessEqual(sum(len(points) for _,points in rebuilt[0]),6)

    def test_discrete_fold_counts_include_handoffs_and_collapse_duplicates(self):
        points=[dict(xyz_m=p,role=r) for p,r in [([0.,0.,0.],'scan'),([1.,0.,0.],'scan'),
                ([1.,0.,0.],'connector'),([0.,0.,0.],'connector'),([0.,0.,0.],'scan'),([-1.,0.,0.],'scan')]]
        self.assertEqual(module.fold_counts([(0,points)]),dict(sharp_turns_120=1,reversals_150=1))
        self.assertEqual(module.fold_counts([(0,[dict(p,role='scan') for p in points])]),dict(sharp_turns_120=0,reversals_150=0))

    def test_retained_corridor_default_is_exactly_four_sweeps(self):
        xyz=np.asarray([self.segments[0].points[-1]['xyz_m']]+[x['xyz_m'] for x in self.links[0]['samples']])
        faces=[x['triangle_index'] for x in self.links[0]['samples']]
        a=optimize_surface_path(self.mesh.vertices,self.mesh.faces,xyz,faces,lambda a,b:True)
        b=optimize_surface_path(self.mesh.vertices,self.mesh.faces,xyz,faces,lambda a,b:True,sweeps=4)
        np.testing.assert_array_equal(a[0],b[0]);self.assertEqual(a[1],b[1])
        for value in (False,0,5,1.5):
            with self.subTest(value=value),self.assertRaises(ValueError):
                optimize_surface_path(self.mesh.vertices,self.mesh.faces,xyz,faces,lambda a,b:True,sweeps=value)

    def test_v16_shares_optional_search_budget_across_row_retries(self):
        from dataclasses import dataclass
        from femto.planning import repair_workflow,global_search_budget_research as budgets
        from femto.planning.algorithm_registry import REFINED_CONNECTOR_ROUTING
        @dataclass
        class Trajectory:
            statistics: dict
            gap_report: dict
        seen=[]
        def replan(*args,search_budget,**kwargs):
            seen.append(search_budget)
            for _ in range(200):search_budget.take('relocation')
            return Trajectory(dict(gap_count=2,inter_block_gap_count=0,
                                   unplannable_face_count=0,total_length_m=1.),{})
        with patch('femto.planning.continuous_quality.candidate',return_value=object()), \
             patch('femto.planning.continuous_quality.replan',side_effect=replan), \
             patch('femto.planning.isolated_row_repair.requests',return_value=[{'trial_shifts':[{'test':1},{'test':2}]}]):
            result=repair_workflow.generate(None,None,None,None,5,None,{},routing=REFINED_CONNECTOR_ROUTING)
        self.assertEqual(len(seen),3);self.assertTrue(all(item is seen[0] for item in seen))
        self.assertEqual(result.gap_report['row_repair']['search_budget']['total_used'],budgets.TOTAL)

    def test_version_binding_tampering_and_native_portable_replay(self):
        from test_fixed_rgb_experience import make_model,K
        from test_mesh_slice_planner import grid_mesh
        from femto.planning import surface_workspace as ws,planning_revision
        from femto.planning.algorithm_registry import REFINED_CONNECTOR_ROUTING
        from femto.planning.fixed_width_layout import COORDINATED_FLOOR_PATCH_VERSION
        from femto.planning.treatment_head import head_parameters
        from femto.planning.region_project import save_project,open_project
        from femto.planning.trajectory_export import load_trajectory
        with tempfile.TemporaryDirectory() as folder:
            mesh=grid_mesh(width=.12,height=.08,cell=.005)
            model=make_model(folder,o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),o3d.utility.Vector3iVector(mesh.faces)))
            model.view_reference['undistorted_intrinsic']=K.tolist()
            model.store.origin_reference=dict(kind='initial_color_camera',provenance='synthetic_test')
            o3d.io.write_triangle_mesh(str(model.store.source),model.mesh)
            region=model.region([[30,30],[70,30],[70,70],[30,70]],'test','',selection=dict(mask=model.projection.valid.copy(),method='work_only'))
            head=head_parameters('fixed_width_auto',effective_treatment_width_mm=5,overlap_percent=20)
            plan=ws.generate(model,region,head,layout_algorithm=COORDINATED_FLOOR_PATCH_VERSION,routing_algorithm=REFINED_CONNECTOR_ROUTING)
            self.assertEqual(plan['planning_context']['revision']['inputs']['algorithms'],planning_revision.REFINED_COORDINATED_FLOOR_ALGORITHMS)
            original_plan=deepcopy(plan)
            ws.validate(plan,model.mesh,region)
            self.assertEqual(plan,original_plan)
            for field in ('policy','edge_checks','baseline_connections','metric_type','attempt_order','attempt_truncation','row_repair'):
                broken=deepcopy(plan);record=broken['surface_trajectory']['gap_report']['reconnection']['connector_refinement']
                if field=='policy':record[field]['portal_sweeps']+=1
                elif field=='edge_checks':record[field]=True
                elif field=='baseline_connections':record[field]=[]
                elif field=='metric_type':record['baseline']['internal_gaps']=False
                elif field=='attempt_order':
                    self.assertGreater(len(record['attempts']),1)
                    record['attempts'][0],record['attempts'][1]=record['attempts'][1],record['attempts'][0]
                elif field=='attempt_truncation':record['attempts']=[]
                else:broken['surface_trajectory']['gap_report']['row_repair']['search_budget']['total_used']+=1
                with self.subTest(field=field),self.assertRaises(ValueError):ws.validate(broken,model.mesh,region)
            native=ws.export(model,region,plan)
            self.assertEqual(load_trajectory(native).to_dict()['ordered_items'],plan['surface_trajectory']['ordered_items'])
            archive=save_project(model,region,Path(folder)/'refined.region.zip',plan,catalog_root=Path(folder)/'catalog')
            opened=open_project(archive,workspace=Path(folder)/'opened')
            self.assertEqual(opened.saved_plan['surface_trajectory']['ordered_items'],plan['surface_trajectory']['ordered_items'])
            self.assertFalse(opened.saved_plan['execution_ready'])


if __name__=='__main__':unittest.main()
