"""A shorter proposal may not increase detours or nominal connector revisits."""
from copy import deepcopy
from unittest.mock import patch
import unittest
from test_verified_visit_improvement import Graph,make_segment
from femto.planning.scan_visit_order import connect,reconstruct
from femto.planning.balanced_visit_improvement import propose,_metrics,validate_record


class BalancedVisitTests(unittest.TestCase):
    def setUp(self):
        self.segments=[make_segment(i,float(i)) for i in range(5)]
        self.graph=Graph(self.segments);self.visits=list(enumerate([False]*5))
        self.links=connect(self.graph,self.segments,self.visits,lambda:None,1000)
        self.full=[(0,False),(1,True),(2,False),(3,True),(4,False)]
        self.full_links=connect(self.graph,self.segments,self.full,lambda:None,1000)
        self.full_result=reconstruct(self.segments,self.full,self.full_links)
        self.trial=dict(adopted=True,rounds=[dict(first=1,last=1),dict(first=3,last=3)],
                        selected_objective=list(self.full_result[2]))

    def test_shorter_prefix_selected_when_last_round_adds_a_detour(self):
        before=deepcopy((self.segments,self.visits,self.links))
        metrics=[dict(detours=2,revisit_connector_m=2.),dict(detours=3,revisit_connector_m=1.),
                 dict(detours=2,revisit_connector_m=1.)]
        with patch('femto.planning.balanced_visit_improvement.improve',
                   return_value=(self.full_result[0],self.full_links,self.full_result[1],self.trial)), \
             patch('femto.planning.balanced_visit_improvement._metrics',side_effect=metrics):
            _,links,_,decision=propose(self.graph,self.segments,self.visits,self.links,.01,lambda:None,1000)
        self.assertTrue(decision['adopted']);self.assertEqual(len(decision['rounds']),1)
        self.assertEqual(decision['visits'],[[0,False],[1,True],[2,False],[3,False],[4,False]])
        self.assertLess(reconstruct(self.segments,decision['visits'],links)[2][1],
                        reconstruct(self.segments,self.visits,self.links)[2][1])
        self.assertEqual((self.segments,self.visits,self.links),before)

    def test_all_worse_candidates_leave_original_unchanged(self):
        metrics=[dict(detours=2,revisit_connector_m=2.),dict(detours=3,revisit_connector_m=1.),
                 dict(detours=2,revisit_connector_m=3.)]
        with patch('femto.planning.balanced_visit_improvement.improve',
                   return_value=(self.full_result[0],self.full_links,self.full_result[1],self.trial)), \
             patch('femto.planning.balanced_visit_improvement._metrics',side_effect=metrics):
            _,links,_,decision=propose(self.graph,self.segments,self.visits,self.links,.01,lambda:None,1000)
        self.assertFalse(decision['adopted']);self.assertEqual(links,self.links)

    def test_invalid_spacing_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'行距'):
            propose(self.graph,self.segments,self.visits,self.links,0.,lambda:None,1000)

    def test_metrics_match_original_item_conversion_without_mutating_chains(self):
        from femto.planning.quality_constrained import _items
        from femto.planning.selection_blocks import _mark_revisits,_walk_connectors
        from femto.planning.surface_trajectory import ScanSegment
        original=reconstruct(self.segments,self.visits,self.links)[0]
        original.append((1,deepcopy(self.segments[0].points)))
        saved=deepcopy(original);reference=deepcopy(original)
        _mark_revisits([p for _,p in reference],.01,lambda:None)
        segments=[x for x in _items(reference) if isinstance(x,ScanSegment)]
        _,detours,_,revisit=_walk_connectors(segments)
        self.assertEqual(_metrics(original,.01,lambda:None),dict(detours=detours,revisit_connector_m=revisit))
        self.assertEqual(original,saved)

    def test_metric_copy_preserves_nested_inputs_and_legacy_results(self):
        from femto.planning.selection_blocks import _mark_revisits,_walk_connectors
        original=reconstruct(self.segments,self.visits,self.links)[0]
        for _,points in original:
            for point in points:point['nested_test']={'pixels':[[1,2],[3,4]]}
        before=deepcopy(original);reference=deepcopy(original)
        _mark_revisits([points for _,points in reference],.01,lambda:None)
        marked=[]
        for index,(block,points) in enumerate(reference):
            points[0]['role']='scan';points[0].pop('revisit',None)
            from femto.planning.surface_trajectory import ScanSegment
            marked.append(ScanSegment(index,points[0]['row_index'],points,block))
        _,detours,_,revisit=_walk_connectors(marked)
        self.assertEqual(_metrics(original,.01,lambda:None),dict(detours=detours,revisit_connector_m=revisit))
        self.assertEqual(original,before)

    def test_saved_record_recomputes_metrics_and_rejects_tampering(self):
        _,final,_,decision=propose(self.graph,self.segments,self.visits,self.links,.01,lambda:None,1000)
        record=dict(decision,baseline_visits=[list(v) for v in self.visits],baseline_connections=self.links)
        validate_record(record,self.segments,record['baseline_visits'],self.links,final,.01,lambda:None)
        forged=deepcopy(record);forged['baseline']['detours']+=1
        with self.assertRaisesRegex(ValueError,'基准不能重算'):
            validate_record(forged,self.segments,record['baseline_visits'],self.links,final,.01,lambda:None)
