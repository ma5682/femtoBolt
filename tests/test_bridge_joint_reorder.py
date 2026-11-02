"""Bounded bridge proposals retain complete final route constraints."""
from copy import deepcopy
import unittest
from unittest.mock import patch
from test_verified_visit_improvement import Graph,make_segment
from femto.planning import bridge_joint_reorder as module
from femto.planning import bounded_joint_reorder as base
from femto.planning.scan_visit_order import connect,reconstruct


class BridgeJointReorderTests(unittest.TestCase):
    def setUp(self):
        self.segments=[make_segment(i,0.) for i in range(4)]
        for segment,row in zip(self.segments,(0,0,1,1)):
            for point in segment.points:point.update(row_index=row,triangle_index=0)
        self.visits=list(enumerate([False]*4));self.graph=Graph(self.segments)
        self.links=connect(self.graph,self.segments,self.visits,lambda:None,1000)

    def test_valid_proposal_replays_and_does_not_mutate_input(self):
        initial=deepcopy((self.segments,self.visits,self.links))
        baseline=reconstruct(self.segments,self.visits,self.links,preserve_endpoint_faces=True)
        chains,links,added,record=module.propose(self.graph,self.segments,self.visits,self.links,.01,lambda:None,1000)
        self.assertTrue(record['adopted']);self.assertEqual((self.segments,self.visits,self.links),initial)
        self.assertEqual(base._signature(chains,lambda:None),base._signature(baseline[0],lambda:None))
        self.assertEqual(module.validate_record(record,self.segments,self.visits,self.links,links,.01,lambda:None,1000),(chains,added))
        self.assertLessEqual(record['selected_objective'][1],baseline[2][1]+base.EPS)
        for field in ('history_revisit_m','edge_revisit_m','detours'):
            self.assertLessEqual(record['selected'][field],record['baseline'][field]+base.EPS)
        for field in ('version','steps','selected'):
            bad=deepcopy(record)
            if field=='version':bad[field]='wrong'
            elif field=='steps':bad[field][0]['first']=0
            else:bad[field]['detours']+=1
            with self.subTest(field=field),self.assertRaises(ValueError):
                module.validate_record(bad,self.segments,self.visits,self.links,links,.01,lambda:None,1000)
        bad=deepcopy(record);bad['steps']=[dict(first=1,last=1) for _ in range(4)]
        with self.assertRaises(ValueError):
            module.validate_record(bad,self.segments,self.visits,self.links,links,.01,lambda:None,1000)

    def test_unavailable_paths_and_budget_keep_original(self):
        graph=Graph(self.segments,only_baseline=True)
        _,links,_,record=module.propose(graph,self.segments,self.visits,self.links,.01,lambda:None,1000)
        self.assertFalse(record['adopted']);self.assertEqual(links,self.links)
        with patch.object(module,'connect',side_effect=ValueError('连续路线整边搜索超过预算，请缩小选区')):
            chains,links,added,record=module.propose(self.graph,self.segments,self.visits,self.links,.01,lambda:None,1000)
        self.assertFalse(record['adopted']);self.assertEqual(record['skipped'],'geometry_budget');self.assertEqual(links,self.links)
        self.assertEqual(module.validate_record(record,self.segments,self.visits,self.links,links,.01,lambda:None,1000),(chains,added))

    def test_cancellation_and_invalid_input(self):
        def cancel():raise RuntimeError('cancelled')
        with self.assertRaisesRegex(RuntimeError,'cancelled'):
            module.propose(self.graph,self.segments,self.visits,self.links,.01,cancel,1000)
        with self.assertRaises(ValueError):module.propose(self.graph,[],[],[],.01,lambda:None,1000)
