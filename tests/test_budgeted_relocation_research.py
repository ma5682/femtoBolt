"""Changed-edge ranking is bounded and cannot bypass complete route replay."""
from copy import deepcopy
import random
import unittest
from unittest.mock import patch

from test_verified_visit_improvement import Graph,make_segment
from femto.planning import budgeted_relocation_research as module
from femto.planning import balanced_visit_improvement as balanced,scan_visit_order as order
from femto.planning.endpoint_preserving_detour import _signature


class BudgetedRelocationTests(unittest.TestCase):
    def setUp(self):
        self.segments=[make_segment(i,float(y),0 if i<5 else 1) for i,y in enumerate((0,4,1,3,2,0,1))]
        for segment in self.segments:
            for point in segment.points:point['triangle_index']=0
        self.visits=list(enumerate([False]*len(self.segments)));self.graph=Graph(self.segments)
        self.links=order.connect(self.graph,self.segments,self.visits,lambda:None,1000)
        baseline=order.reconstruct(self.segments,self.visits,self.links,preserve_endpoint_faces=True,budget=1000)
        metrics=balanced._metrics(baseline[0],.01,lambda:None)
        self.limits=dict(detours=metrics['detours'],revisit_connector_m=metrics['revisit_connector_m'],connector_length_m=baseline[2][1])
        self.options=dict(motion_limits=self.limits,budget=1000,total_attempts=128,per_round=32,rounds=4,full_per_round=8)

    def test_changed_edges_match_complete_visit_reconstruction(self):
        for seed in range(20):
            rng=random.Random(seed);middle=list(range(1,5));rng.shuffle(middle)
            visits=[(0,False)]+[(i,bool(rng.getrandbits(1))) for i in middle]+[(5,False),(6,bool(rng.getrandbits(1)))]
            original=set(zip(visits,visits[1:]))
            for _,i,j,reverse in module.legacy._candidates(self.segments,visits,lambda:None):
                remaining=visits[:i]+visits[i+1:];candidate=remaining[:j]+[(visits[i][0],reverse)]+remaining[j:]
                removed,added=module._changes(visits,i,j,reverse)
                expected=set(zip(candidate,candidate[1:]))
                self.assertEqual(set(removed),original-expected)
                self.assertEqual(set(added),expected-original)

    def test_adopted_candidate_is_fully_replayed_and_bounded(self):
        before=deepcopy((self.segments,self.visits,self.links))
        chains,links,added,decision=module.propose(self.graph,self.segments,self.visits,self.links,.01,**self.options)
        self.assertTrue(decision['adopted'])
        self.assertLessEqual(decision['tested'],128);self.assertLessEqual(decision['full_evaluations'],32)
        self.assertLessEqual(decision['selected_objective'][1],self.limits['connector_length_m']+1e-12)
        self.assertLessEqual(decision['selected']['detours'],self.limits['detours'])
        self.assertLessEqual(decision['selected']['revisit_connector_m'],self.limits['revisit_connector_m']+1e-12)
        replayed,replayed_added=module.validate_record(decision,self.segments,self.visits,self.links,links,.01,**self.options)
        self.assertEqual((replayed,replayed_added),(chains,added))
        initial=order.reconstruct(self.segments,self.visits,self.links,preserve_endpoint_faces=True,budget=1000)[0]
        self.assertEqual(_signature(chains,lambda:None),_signature(initial,lambda:None))
        self.assertEqual(before,(self.segments,self.visits,self.links))
        self.assertEqual(module.propose(self.graph,self.segments,self.visits,self.links,.01,**self.options)[3],decision)
        for field in ('budget','full_evaluations','motion_limits','visits'):
            broken=deepcopy(decision)
            if field=='budget':broken[field]['per_round']+=1
            elif field=='full_evaluations':broken[field]=decision['tested']+1
            elif field=='motion_limits':broken[field]['connector_length_m']+=1.
            else:broken[field][0]=[1,False]
            with self.subTest(field=field),self.assertRaises(ValueError):
                module.validate_record(broken,self.segments,self.visits,self.links,links,.01,**self.options)

    def test_disconnected_alternatives_and_stricter_limits(self):
        graph=Graph(self.segments,only_baseline=True)
        links=order.connect(graph,self.segments,self.visits,lambda:None,1000)
        _,selected,_,record=module.propose(graph,self.segments,self.visits,links,.01,**self.options)
        self.assertFalse(record['adopted']);self.assertEqual(selected,links)
        module.validate_record(record,self.segments,self.visits,links,selected,.01,**self.options)
        limits=dict(self.limits,connector_length_m=self.limits['connector_length_m']+.01)
        with self.assertRaisesRegex(ValueError,'超过允许的基准增量'):
            module.propose(self.graph,self.segments,self.visits,self.links,.01,**dict(self.options,motion_limits=limits))

    def test_batch_queue_preserves_full_scalar_decision(self):
        def scalar(segments,visits,scorer,count,check,cache):
            ranked,original=module._ranked(segments,visits,scorer,check,cache)
            return module._work_queue(ranked,count),original
        with patch.object(module,'_queue',side_effect=scalar):
            expected=module.propose(self.graph,self.segments,self.visits,self.links,.01,**self.options)
        actual=module.propose(self.graph,self.segments,self.visits,self.links,.01,**self.options)
        self.assertEqual(actual,expected)

    def test_tradeoff_requires_fewer_detours_or_both_other_metrics_nonworse(self):
        initial={'detours':2,'revisit_connector_m':.1}
        self.assertFalse(module._motion_improves({'detours':2,'revisit_connector_m':.099999},1.004,initial,1.))
        self.assertFalse(module._motion_improves({'detours':2,'revisit_connector_m':.100001},.99,initial,1.))
        self.assertTrue(module._motion_improves({'detours':2,'revisit_connector_m':.099},.99,initial,1.))
        self.assertTrue(module._motion_improves({'detours':1,'revisit_connector_m':.101},1.004,initial,1.))

    def test_new_policy_can_replay_more_rounds_without_changing_old_version(self):
        _,links,_,decision=module.propose(self.graph,self.segments,self.visits,self.links,.01,**self.options)
        legacy={k:deepcopy(v) for k,v in decision.items() if k not in ('budget','full_evaluations')}
        legacy['version']=module.legacy.LIMITED_DETOUR_VERSION
        pair=[{'source':1,'destination':1,'reverse':True},{'source':1,'destination':1,'reverse':False}]
        legacy['steps']=pair*2+legacy['steps'];legacy['tested']=max(legacy['tested'],len(legacy['steps']))
        self.assertGreater(len(legacy['steps']),4);self.assertLessEqual(len(legacy['steps']),8)
        with self.assertRaises(ValueError):
            module.legacy.validate_record(legacy,self.segments,self.visits,self.links,links,.01,
                motion_limits=self.limits,attempts=32,rounds=8,budget=1000,objective_policy='detour_first_with_limits')
        module.legacy.validate_extended_record(legacy,self.segments,self.visits,self.links,links,.01,
            motion_limits=self.limits,attempts=32,rounds=8,budget=1000)

    def test_unusual_scan_structures_cannot_enter_proxy_index(self):
        from femto.planning.unified_floor_routing import _supported
        self.assertTrue(_supported(self.segments))
        for case in ('single_point','connector','mixed_rows','invalid_coordinate'):
            segments=deepcopy(self.segments)
            if case=='single_point':segments[1].points.pop()
            elif case=='connector':segments[1].points[0]['role']='connector'
            elif case=='mixed_rows':segments[1].points[0]['row_index']=99
            else:segments[1].points[0]['xyz_m'][0]=float('nan')
            with self.subTest(case=case):self.assertFalse(_supported(segments))

    def test_preview_result_cannot_bypass_full_measurement(self):
        original=balanced._metrics;original_reconstruct=order.reconstruct;rebuilds=0
        def reconstruct(*args,**kwargs):
            nonlocal rebuilds
            rebuilds+=1
            return original_reconstruct(*args,**kwargs)
        def inconsistent(*args):
            value=original(*args)
            if rebuilds>=2:value['revisit_connector_m']+=1.
            return value
        with patch.object(order,'reconstruct',side_effect=reconstruct),patch.object(balanced,'_metrics',side_effect=inconsistent):
            with self.assertRaisesRegex(ValueError,'完整重建不一致'):
                module.propose(self.graph,self.segments,self.visits,self.links,.01,**self.options)


if __name__=='__main__':unittest.main()
