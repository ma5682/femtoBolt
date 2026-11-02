import copy,tempfile,unittest,json
from pathlib import Path
from collections import Counter
from unittest import mock
import numpy as np
from tests import test_coverage_inspection as fixtures
from femto.planning.coverage_inspection import inspected_serpentine_plan
from femto.planning.route_optimization import optimized_serpentine_plan,Budget,coverage_not_worse
from femto.planning.route_quality import quality,show_quality
from femto.planning.planning_state import validate
from femto.planning.region_path import export_path,load_path

class QualityTests(unittest.TestCase):
    def setup_case(self,root):
        m,s,r,p=fixtures.InspectionTests().surface(root)
        r=dict(r,exclusion_polygons=[[[30,20],[46,20],[46,60],[30,60]]])
        return m,s,r,p
    def test_obstacle_improves_without_losing_scan_coverage_or_certified_edges(self):
        with tempfile.TemporaryDirectory() as tmp:
            m,s,r,p=self.setup_case(Path(tmp));old=inspected_serpentine_plan(p,m,r,4,20);new=optimized_serpentine_plan(p,m,r,4,20)
            report=new['route_quality'];self.assertEqual(report['status'],'improved')
            self.assertLess(report['after']['connector_length_m'],report['before']['connector_length_m']*.8)
            self.assertLess(report['after']['repeated_edge_length_m'],report['before']['repeated_edge_length_m'])
            self.assertTrue(coverage_not_worse(old['coverage_inspection'],new['coverage_inspection']))
            def edges(chain,scan=False):
                return [tuple(sorted((tuple(a['xyz_m']),tuple(b['xyz_m'])))) for a,b in zip(chain,chain[1:]) if not scan or b['motion_kind']=='scan']
            for a,b in zip(old['segments'],new['segments']):
                self.assertEqual(a[0]['xyz_m'],b[0]['xyz_m']);self.assertEqual(a[-1]['xyz_m'],b[-1]['xyz_m'])
                self.assertEqual(Counter(edges(a,True)),Counter(edges(b,True)))
                self.assertTrue(set(edges(b))<=set(edges(a)))
            self.assertEqual(new['boundary_check']['rejected_edges'],0)
            self.assertTrue(any(a['status']=='coverage_regression_rejected' and a['lost_covered_samples']>0 for a in report['attempts']))
            self.assertFalse(report['smoothing']['applied'])
    def test_holes_and_disconnected_surfaces_keep_constraints(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name,keep in [('hole',lambda c:~((c[:,0]>28)&(c[:,0]<52)&(c[:,1]>28)&(c[:,1]<52))),('split',lambda c:(c[:,0]<28)|(c[:,0]>52))]:
                root=Path(tmp)/name;root.mkdir();m,s,r,p=fixtures.InspectionTests().surface(root,keep)
                old=inspected_serpentine_plan(p,m,r,4,20);new=optimized_serpentine_plan(p,m,r,4,20)
                self.assertEqual(len(old['segments']),len(new['segments']))
                self.assertTrue(coverage_not_worse(old['coverage_inspection'],new['coverage_inspection']))
                for chain in new['segments']:
                    for point in chain:
                        x,y=point['pixel']
                        self.assertFalse(28<x<52 and (name=='split' or 28<y<52))

    def test_replay_and_quality_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);m,s,r,p=self.setup_case(root);plan=optimized_serpentine_plan(p,m,r,4,20)
            self.assertEqual(plan,optimized_serpentine_plan(p,m,r,4,20))
            folder=export_path(plan,s,root/'exports',r,(np.zeros((85,85,3),np.uint8),p.ids))
            self.assertEqual(plan,load_path(folder/'path.json',s,p,r))
            for name in ('route_quality.json','route_quality.txt','sharp_turns.csv'):self.assertTrue((folder/name).exists())
            changed=copy.deepcopy(plan);changed['route_quality']['after']['total_length_m']=0
            with self.assertRaises(ValueError):validate(changed,m,r)
    def test_budget_falls_back_and_cancel_propagates(self):
        with tempfile.TemporaryDirectory() as tmp:
            m,s,r,p=self.setup_case(Path(tmp));old=inspected_serpentine_plan(p,m,r,4,20)
            with mock.patch('femto.planning.route_optimization.Budget',side_effect=lambda check:Budget(check,0)):
                plan=optimized_serpentine_plan(p,m,r,4,20)
            self.assertEqual(plan['segments'],old['segments']);self.assertEqual(plan['route_quality']['status'],'baseline_retained')
            self.assertEqual(plan['route_quality']['attempts'][0]['status'],'budget_exceeded')
            with mock.patch('femto.planning.route_optimization.reorder_chain',side_effect=RuntimeError('cancelled')):
                with self.assertRaisesRegex(RuntimeError,'cancelled'):optimized_serpentine_plan(p,m,r,4,20)
    def test_verification_failure_keeps_old_polyline(self):
        with tempfile.TemporaryDirectory() as tmp:
            m,s,r,p=self.setup_case(Path(tmp));old=inspected_serpentine_plan(p,m,r,4,20)
            with mock.patch('femto.planning.route_optimization.inspect_coverage',side_effect=ValueError('validation failure')):
                plan=optimized_serpentine_plan(p,m,r,4,20)
            self.assertEqual(plan['segments'],old['segments']);self.assertEqual(plan['route_quality']['status'],'baseline_retained')
    def test_sharp_angle_and_exact_reverse_repeat(self):
        def pt(x,y):return dict(xyz_m=[x,y,0],pixel=[x,y])
        plan=dict(segments=[[pt(0,0),pt(1,0),pt(1,1),pt(1,0)]])
        metrics,turns=quality(plan)
        self.assertEqual(metrics['repeated_edge_length_m'],1)
        self.assertEqual([t['deflection_degrees'] for t in turns],[90.,180.])
        self.assertEqual([t['sample_index'] for t in turns],[1,2])
