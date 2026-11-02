"""Reused unchanged links cannot mutate inputs or bypass point budgets."""
from copy import deepcopy
import unittest
from test_verified_visit_improvement import Graph,make_segment
from femto.planning import bridge_joint_incremental as module,bridge_joint_reorder as original
from femto.planning.scan_visit_order import connect


class CountGraph(Graph):
    def __init__(self,segments):super().__init__(segments);self.calls=0
    def route(self,*args,**kwargs):self.calls+=1;return super().route(*args,**kwargs)


class IncrementalBridgeTests(unittest.TestCase):
    def setUp(self):
        self.segments=[make_segment(i,float(i)) for i in range(8)]
        for segment in self.segments:
            for point in segment.points:point['triangle_index']=0
        self.visits=list(enumerate([False]*len(self.segments)))
        self.links=connect(Graph(self.segments),self.segments,self.visits,lambda:None,1000)

    def test_only_changed_oriented_links_are_searched(self):
        candidate=list(self.visits);candidate[3:5]=[(i,not r) for i,r in reversed(candidate[3:5])]
        original_links=deepcopy(self.links);reference=connect(Graph(self.segments),self.segments,candidate,lambda:None,1000)
        graph=CountGraph(self.segments)
        result=module._connect_reusing(graph,self.segments,candidate,self.visits,self.links,lambda:None,1000)
        self.assertEqual(result,reference);self.assertLess(graph.calls,len(candidate)-1)
        self.assertEqual(self.links,original_links)
        result[0]['samples'][0]['xyz_m'][0]=99.
        self.assertEqual(self.links,original_links)

    def test_private_borrowed_links_do_not_change_snapshot_or_budget(self):
        before=deepcopy(self.links)
        result=module._connect_reusing(CountGraph(self.segments),self.segments,self.visits,self.visits,
                                       self.links,lambda:None,1000,copy_unchanged=False)
        self.assertIs(result[0],self.links[0]);self.assertEqual(self.links,before)
        with self.assertRaisesRegex(ValueError,'超过预算'):
            module._connect_reusing(CountGraph(self.segments),self.segments,self.visits,self.visits,
                                    self.links,lambda:None,16,copy_unchanged=False)
        with self.assertRaises(ValueError):
            module._connect_reusing(CountGraph(self.segments),self.segments,self.visits,self.visits,
                                    self.links,lambda:None,1000,copy_unchanged=1)

    def test_unchanged_links_still_count_against_budget(self):
        graph=CountGraph(self.segments)
        with self.assertRaisesRegex(ValueError,'超过预算'):
            module._connect_reusing(graph,self.segments,self.visits,self.visits,self.links,lambda:None,16)
        self.assertEqual(graph.calls,0)
        corrupted=deepcopy(self.links);corrupted[0]['from_segment']=99
        with self.assertRaises(ValueError):module._connect_reusing(graph,self.segments,self.visits,self.visits,corrupted,lambda:None,1000)
        def cancel():raise RuntimeError('cancelled')
        with self.assertRaisesRegex(RuntimeError,'cancelled'):
            module._connect_reusing(graph,self.segments,self.visits,self.visits,self.links,cancel,1000)

    def test_proposal_preserves_result_and_replays(self):
        scans=[make_segment(i,0.) for i in range(4)]
        for segment,row in zip(scans,(0,0,1,1)):
            for point in segment.points:point.update(row_index=row,triangle_index=0)
        visits=list(enumerate([False]*4));links=connect(Graph(scans),scans,visits,lambda:None,1000)
        normal=original.propose(Graph(scans),scans,visits,links,.01,lambda:None,1000)
        fast=module.propose(Graph(scans),scans,visits,links,.01,lambda:None,1000)
        self.assertEqual(normal[:3],fast[:3])
        self.assertEqual(normal[3]['steps'],fast[3]['steps'])
        self.assertEqual(module.validate_record(fast[3],scans,visits,links,fast[1],.01,lambda:None,1000),(fast[0],fast[2]))
        self.assertTrue(fast[3]['adopted'])
        before=deepcopy((scans,links))
        fast[1][0]['samples'][0]['xyz_m'][0]=99.
        self.assertEqual((scans,links),before)

    def test_separate_blocks_keep_their_unconnected_boundary(self):
        scans=[make_segment(i,float(i),0 if i<3 else 1) for i in range(6)]
        for scan in scans:
            for point in scan.points:point['triangle_index']=0
        visits=list(enumerate([False]*len(scans)));links=connect(Graph(scans),scans,visits,lambda:None,1000)
        candidate=list(visits);candidate[1:3]=[(i,not reverse) for i,reverse in reversed(candidate[1:3])]
        result=module._connect_reusing(CountGraph(scans),scans,candidate,visits,links,lambda:None,1000)
        self.assertEqual(result,connect(Graph(scans),scans,candidate,lambda:None,1000))
        self.assertFalse(result[2]['connected']);self.assertEqual(result[2]['reason'],'separate_work_blocks')
