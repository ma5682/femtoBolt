"""A single candidate budget is shared by every route-repair trial."""
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import open3d as o3d
from test_fixed_rgb_experience import make_model,K
from test_mesh_slice_planner import grid_mesh
from femto.planning import global_search_budget_research as budgets,repair_workflow
from femto.planning import planning_revision,surface_workspace as ws
from femto.planning.continuous_quality import GLOBAL_FLOOR_ROUTING
from femto.planning.fixed_width_layout import COORDINATED_FLOOR_PATCH_VERSION
from femto.planning.region_project import open_project,save_project
from femto.planning.trajectory_export import load_trajectory
from femto.planning.treatment_head import head_parameters


@dataclass
class _Trajectory:
    statistics: dict
    gap_report: dict


class GlobalRouteBudgetTests(unittest.TestCase):
    def test_independent_objects_limits_and_invalid_reports(self):
        a,b=budgets.SearchBudget(),budgets.SearchBudget()
        for _ in range(budgets.LIMITS['two_opt']):self.assertTrue(a.take('two_opt'))
        self.assertFalse(a.take('two_opt'))
        for _ in range(budgets.TOTAL):a.take('relocation')
        self.assertEqual(a.report()['total_used'],budgets.TOTAL)
        self.assertEqual(b.report()['total_used'],0)
        self.assertFalse(a.take('joint'))
        budgets.validate_report(a.report())
        for field in ('negative','count','extra','limit'):
            broken=deepcopy(a.report())
            if field=='negative':broken['counts']['joint']=-1
            elif field=='count':broken['total_used']-=1
            elif field=='extra':broken['counts']['unexpected']=1
            else:broken['limits']['joint']+=1
            with self.subTest(field=field),self.assertRaises(ValueError):budgets.validate_report(broken)

    def test_budget_is_shared_across_all_row_repair_trials(self):
        identities=[]
        def replan(*args,search_budget,**kwargs):
            identities.append(id(search_budget))
            for _ in range(200):search_budget.take('relocation')
            return _Trajectory(dict(gap_count=2,inter_block_gap_count=0,
                                    unplannable_face_count=0,total_length_m=1.),{})
        proposals=[{'trial_shifts':[{'test':1},{'test':2}]}]
        with patch('femto.planning.continuous_quality.candidate',return_value=object()), \
             patch('femto.planning.continuous_quality.replan',side_effect=replan), \
             patch('femto.planning.isolated_row_repair.requests',return_value=proposals):
            result=repair_workflow.generate(None,None,None,None,5,None,{},routing=GLOBAL_FLOOR_ROUTING)
        self.assertEqual(len(identities),3)
        self.assertEqual(len(set(identities)),1)
        self.assertEqual(result.gap_report['row_repair']['search_budget']['total_used'],budgets.TOTAL)
        self.assertEqual(len(result.gap_report['row_repair']['attempts']),2)

    def test_exhaustion_keeps_verified_inputs_and_old_record_rules(self):
        from test_budgeted_relocation_research import BudgetedRelocationTests
        from test_seam_visit_proposal import segment
        from test_verified_visit_improvement import Graph
        from femto.planning import budgeted_relocation_research as relocation,seam_visit_proposal as seam,scan_visit_order as order
        fixture=BudgetedRelocationTests('test_adopted_candidate_is_fully_replayed_and_bounded');fixture.setUp()
        budget=budgets.SearchBudget()
        for _ in range(budgets.TOTAL):budget.take('relocation')
        with patch.object(relocation,'_ranked',side_effect=AssertionError('exhausted search ranked candidates')):
            chains,links,added,record=relocation.propose(fixture.graph,fixture.segments,fixture.visits,
                fixture.links,.01,search=budget,**fixture.options)
        self.assertFalse(record['adopted']);self.assertEqual(record['tested'],0)
        self.assertEqual(links,fixture.links)
        self.assertEqual(relocation.validate_record(record,fixture.segments,fixture.visits,fixture.links,links,.01,
            **fixture.options),(chains,added))
        segments=[segment(0,(0,0,0),(1,0,0)),segment(1,(1,0,0),(2,0,0))]
        visits=[(0,False),(1,False)];graph=Graph(segments);links=order.connect(graph,segments,visits,lambda:None,100)
        layout={'fragments':[{'block_index':0,'patches':[{'first_row_index':0,'retained_row_count':1},
                                                          {'first_row_index':1,'retained_row_count':1}]}]}
        _,selected,_,decision=seam.propose(graph,segments,visits,links,layout,.01,lambda:None,100,search=budget)
        self.assertEqual(decision['reason'],'search_budget');self.assertEqual(selected,links)
        decision.update(baseline_visits=visits,baseline_connections=links)
        with self.assertRaises(ValueError):seam.validate_record(decision,segments,visits,links,selected,.01,lambda:None)
        seam.validate_record(decision,segments,visits,links,selected,.01,lambda:None,allow_search_budget=True)

    def test_new_limits_use_stricter_reconstructed_measurement(self):
        from femto.planning import unified_floor_routing as routing
        baseline={'detours':4,'connector_length_m':1.,'revisit_connector_m':.3}
        measured={'detours':3,'connector_length_m':1.002,'revisit_connector_m':.1}
        with patch('femto.planning.continuous_quality._relocation_context',return_value=(None,baseline)), \
             patch.object(routing,'_supported',return_value=True), \
             patch.object(routing.budgeted,'limits_for',return_value=measured):
            _,old=routing._context([],[],[],[],[],[],.01,lambda:None)
            _,new=routing._context([],[],[],[],[],[],.01,lambda:None,strict=True)
        self.assertEqual(old['detours'],4)
        self.assertEqual(new,measured)

    def test_shared_motion_limit_checks_actual_route_and_limits_both_stages(self):
        from test_endpoint_preserving_detour import PreservedDetourTests
        from femto.planning import unified_floor_routing as routing,endpoint_preserving_detour as detour,bridge_joint_incremental as joint
        fixture=PreservedDetourTests('test_candidate_preserves_edges_source_and_decreases_detours');fixture.setUp()
        limits=routing.global_motion_limits(fixture.segments,fixture.visits,fixture.links,.01,lambda:None,1000)
        routing.check_global_motion(fixture.segments,fixture.visits,fixture.links,.01,limits,lambda:None,1000)
        too_short=dict(limits,connector_length_m=0.)
        with self.assertRaisesRegex(ValueError,'共同运动代价限制'):
            routing.check_global_motion(fixture.segments,fixture.visits,fixture.links,.01,too_short,lambda:None,1000)
        for module in (detour,joint):
            _,links,_,decision=module.propose(fixture.graph,fixture.segments,fixture.visits,fixture.links,
                .01,lambda:None,1000,motion_limits=too_short)
            self.assertFalse(decision['adopted']);self.assertEqual(links,fixture.links)
        from test_verified_visit_improvement import Graph
        from femto.planning.scan_visit_order import connect
        visits=[(0,False),(2,False),(1,False),(3,False)]
        links=connect(Graph(fixture.segments,only_baseline=True),fixture.segments,visits,lambda:None,1000)
        gap_limits=routing.global_motion_limits(fixture.segments,visits,links,.01,lambda:None,1000)
        self.assertGreater(gap_limits['internal_gaps'],0)
        routing.check_global_motion(fixture.segments,visits,links,.01,gap_limits,lambda:None,1000)
        with self.assertRaisesRegex(ValueError,'共同运动代价限制'):
            routing.check_global_motion(fixture.segments,visits,links,.01,dict(gap_limits,internal_gaps=0),lambda:None,1000)

    def test_new_version_geometry_and_native_portable_replay(self):
        with tempfile.TemporaryDirectory() as folder:
            mesh=grid_mesh(width=.12,height=.08,cell=.005)
            model=make_model(folder,o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),o3d.utility.Vector3iVector(mesh.faces)))
            model.view_reference['undistorted_intrinsic']=K.tolist()
            model.store.origin_reference=dict(kind='initial_color_camera',provenance='synthetic_test')
            o3d.io.write_triangle_mesh(str(model.store.source),model.mesh)
            region=model.region([[30,30],[70,30],[70,70],[30,70]],'test','',selection=dict(mask=model.projection.valid.copy(),method='work_only'))
            head=head_parameters('fixed_width_auto',effective_treatment_width_mm=5,overlap_percent=20)
            plan=ws.generate(model,region,head,layout_algorithm=COORDINATED_FLOOR_PATCH_VERSION,routing_algorithm=GLOBAL_FLOOR_ROUTING)
            self.assertEqual(plan['planning_context']['revision']['inputs']['algorithms'],planning_revision.GLOBAL_COORDINATED_FLOOR_ALGORITHMS)
            ws.validate(plan,model.mesh,region)
            report=plan['surface_trajectory']['gap_report']
            budget=report['row_repair']['search_budget']
            budgets.validate_report(budget)
            self.assertLessEqual(budget['total_used'],budgets.TOTAL)
            self.assertEqual(budget['counts']['relocation'],report['reconnection']['ordering']['relocation_unified']['tested'])
            broken=deepcopy(plan)
            broken['surface_trajectory']['gap_report']['row_repair']['search_budget']['total_used']+=1
            with self.assertRaises(ValueError):ws.validate(broken,model.mesh,region)
            native=ws.export(model,region,plan)
            self.assertEqual(load_trajectory(native).to_dict()['ordered_items'],plan['surface_trajectory']['ordered_items'])
            archive=save_project(model,region,Path(folder)/'global.region.zip',plan,catalog_root=Path(folder)/'catalog')
            reopened=open_project(archive,workspace=Path(folder)/'opened')
            self.assertEqual(reopened.saved_plan['surface_trajectory']['ordered_items'],plan['surface_trajectory']['ordered_items'])
            self.assertFalse(reopened.saved_plan['execution_ready'])


if __name__=='__main__':unittest.main()
