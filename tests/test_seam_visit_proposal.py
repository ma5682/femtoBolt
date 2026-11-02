"""Seam proposals preserve work-block starts and never bypass graph checks."""
from copy import deepcopy
from itertools import product
import unittest
from unittest.mock import patch
import numpy as np
from femto.planning import seam_visit_proposal as seam
from femto.planning.scan_visit_order import connect,oriented,reconstruct
from femto.planning.surface_trajectory import ScanSegment
from test_verified_visit_improvement import Graph


def segment(index,a,b,block=0):
    return ScanSegment(index,index,[dict(xyz_m=list(a),row_index=index,role='scan',triangle_index=0),
                                    dict(xyz_m=list(b),row_index=index,role='scan',triangle_index=0)],block)


class SeamVisitProposalTests(unittest.TestCase):
    def test_endpoint_matches_oriented_scan_without_mutating_points(self):
        scan=segment(0,(0,0,0),(1,0,0))
        scan.points.insert(1,dict(xyz_m=[.5,0,0],row_index=2,role='connector',
                                  revisit=True,extra={'nested':[1,2]}))
        scan.points[-1].update(row_index=3,revisit=True,extra={'nested':[3]})
        before=deepcopy(scan)
        for reverse,end in product((False,True),repeat=2):
            expected=oriented(scan,reverse).points[-1 if end else 0]
            actual=seam._endpoint(scan,reverse,end=end)
            self.assertEqual(actual,dict(xyz_m=expected['xyz_m'],row_index=expected['row_index']))
        self.assertEqual(scan,before)

    def test_group_directions_match_exhaustive_and_pin_each_block_start(self):
        segments=[segment(0,(0,0,0),(1,0,0)),segment(1,(0,1,0),(1,1,0)),
                  segment(2,(0,2,0),(1,2,0)),segment(3,(0,0,1),(1,0,1),1),
                  segment(4,(0,1,1),(1,1,1),1)]
        visits,cost=seam._directions(segments,list(range(5)),lambda:None)
        self.assertEqual(visits[0],(0,False));self.assertEqual(visits[3],(3,False))
        def value(directions):
            score=[0,0.]
            for i in (0,1,3):
                tail=segments[i].points[1-int(directions[i])];head=segments[i+1].points[int(directions[i+1])]
                score[0]+=int(tail['row_index']==head['row_index'])
                score[1]+=float(np.linalg.norm(np.asarray(tail['xyz_m'])-head['xyz_m']))
            return tuple(score)
        expected=min(value((False,a,b,False,c)) for a,b,c in product((False,True),repeat=3))
        self.assertEqual(cost,expected)
        self.assertEqual(value([r for _,r in visits]),expected)

    def test_operation_local_distance_cache_preserves_every_direction_decision(self):
        rng=np.random.default_rng(111);segments=[]
        for index in range(24):
            block=index//8;a=rng.normal(size=3);b=rng.normal(size=3)
            segments.append(segment(index,a,b,block))
        cache={};orders=[list(range(24))]
        for first in (1,3,6,9,12,17,20):
            order=list(range(24));order[first],order[first+1]=order[first+1],order[first];orders.append(order)
        for order in orders:
            expected=seam._directions(segments,order,lambda:None)
            actual=seam._directions(segments,order,lambda:None,_distance_cache=cache)
            self.assertEqual(actual,expected)
        with patch('numpy.linalg.norm',wraps=np.linalg.norm) as norm:
            for order in orders:seam._directions(segments,order,lambda:None,_distance_cache=cache)
        self.assertEqual(norm.call_count,0)
        self.assertTrue(cache)

    def test_unverified_coincident_seam_keeps_original(self):
        segments=[segment(0,(0,0,0),(1,0,0)),segment(1,(1,0,0),(2,0,0))]
        visits=[(0,False),(1,False)];links=connect(Graph(segments),segments,visits,lambda:None,100)
        layout={'fragments':[{'block_index':0,'patches':[{'first_row_index':0,'retained_row_count':1},
                                                          {'first_row_index':1,'retained_row_count':1}]}]}
        class NoSeam:
            def route(self,*args,**kwargs):return None
        original=deepcopy((segments,visits,links,layout))
        _,selected,_,decision=seam.propose(NoSeam(),segments,visits,links,layout,.01,lambda:None,100)
        self.assertFalse(decision['adopted']);self.assertEqual(decision['reason'],'no_verified_seams')
        self.assertEqual(selected,links);self.assertEqual((segments,visits,links,layout),original)

    def test_pair_budget_falls_back_without_removing_scans(self):
        segments=[segment(0,(0,0,0),(1,0,0)),segment(1,(1,0,0),(2,0,0))]
        visits=[(0,False),(1,False)];links=connect(Graph(segments),segments,visits,lambda:None,100)
        layout={'fragments':[{'block_index':0,'patches':[{'first_row_index':0,'retained_row_count':1},
                                                          {'first_row_index':1,'retained_row_count':1}]}]}
        with patch.object(seam,'MAX_PAIRS',0):
            chains,selected,_,decision=seam.propose(Graph(segments),segments,visits,links,layout,.01,lambda:None,100)
        self.assertEqual(decision['reason'],'pair_budget');self.assertEqual(selected,links)
        self.assertEqual(chains,reconstruct(segments,visits,links)[0])

    def test_nearby_endpoints_without_zero_length_graph_path_do_not_join(self):
        segments=[segment(0,(0,0,0),(1,0,0)),segment(1,(1+5e-9,0,0),(2,0,0))]
        visits=[(0,False),(1,False)];graph=Graph(segments);links=connect(graph,segments,visits,lambda:None,100)
        layout={'fragments':[{'block_index':0,'patches':[{'first_row_index':0,'retained_row_count':1},
                                                          {'first_row_index':1,'retained_row_count':1}]}]}
        _,selected,_,decision=seam.propose(graph,segments,visits,links,layout,.01,lambda:None,100)
        self.assertEqual(decision['reason'],'no_verified_seams');self.assertEqual(selected,links)

    def test_coincident_points_in_different_work_blocks_never_join(self):
        segments=[segment(0,(0,0,0),(1,0,0)),segment(1,(1,0,0),(2,0,0),1)]
        visits=[(0,False),(1,False)];graph=Graph(segments);links=connect(graph,segments,visits,lambda:None,100)
        layout={'fragments':[{'block_index':0,'patches':[{'first_row_index':0,'retained_row_count':1}]},
                             {'block_index':1,'patches':[{'first_row_index':1,'retained_row_count':1}]}]}
        _,selected,_,decision=seam.propose(graph,segments,visits,links,layout,.01,lambda:None,100)
        self.assertEqual(decision['reason'],'no_verified_seams');self.assertEqual(selected,links)

    def test_overlap_and_cancellation_rejected(self):
        layout={'fragments':[{'block_index':0,'patches':[{'first_row_index':0,'retained_row_count':2},
                                                          {'first_row_index':1,'retained_row_count':2}]}]}
        with self.assertRaisesRegex(ValueError,'不唯一'):seam._patch_rows(layout,lambda:None)
        def cancel():raise RuntimeError('cancelled')
        with self.assertRaisesRegex(RuntimeError,'cancelled'):seam._patch_rows(layout,cancel)

    def test_verified_seams_can_improve_and_record_rejects_tampering(self):
        segments=[segment(0,(0,0,0),(1,0,0)),segment(1,(4,0,0),(5,0,0)),
                  segment(2,(1,0,0),(2,0,0)),segment(3,(3,0,0),(4,0,0))]
        visits=[(i,False) for i in range(4)];graph=Graph(segments);links=connect(graph,segments,visits,lambda:None,100)
        layout={'fragments':[{'block_index':0,'patches':[{'first_row_index':0,'retained_row_count':2},
                                                          {'first_row_index':2,'retained_row_count':2}]}]}
        before=deepcopy((segments,visits,links,layout))
        _,selected,_,decision=seam.propose(graph,segments,visits,links,layout,.01,lambda:None,100)
        self.assertTrue(decision['adopted']);self.assertGreater(decision['matched_seams'],0)
        self.assertLess(decision['selected_objective'][1],decision['baseline_objective'][1])
        bound=dict(decision,baseline_visits=[list(v) for v in visits],baseline_connections=links)
        seam.validate_record(bound,segments,bound['baseline_visits'],links,selected,.01,lambda:None)
        forged=deepcopy(bound);forged['selected']['detours']+=1
        with self.assertRaises(ValueError):
            seam.validate_record(forged,segments,bound['baseline_visits'],links,selected,.01,lambda:None)
        self.assertEqual((segments,visits,links,layout),before)

    def test_worse_verified_connections_fall_back(self):
        segments=[segment(0,(0,0,0),(1,0,0)),segment(1,(4,0,0),(5,0,0)),
                  segment(2,(1,0,0),(2,0,0)),segment(3,(3,0,0),(4,0,0))]
        visits=[(i,False) for i in range(4)];links=connect(Graph(segments),segments,visits,lambda:None,100)
        layout={'fragments':[{'block_index':0,'patches':[{'first_row_index':0,'retained_row_count':2},
                                                          {'first_row_index':2,'retained_row_count':2}]}]}
        class LongGraph(Graph):
            def route(self,left,right,*,left_reverse,right_reverse):
                result=super().route(left,right,left_reverse=left_reverse,right_reverse=right_reverse)
                tail=self.segments[left].points[0 if left_reverse else -1]['xyz_m'];head=result[0]['xyz_m']
                if tail!=head:
                    detour=deepcopy(result[0]);detour['xyz_m']=[100.,100.,100.];result.insert(0,detour)
                return result
        _,selected,_,decision=seam.propose(LongGraph(segments),segments,visits,links,layout,.01,lambda:None,1000)
        self.assertFalse(decision['adopted']);self.assertEqual(decision['reason'],'no_nonworse_candidate')
        self.assertEqual(selected,links)
        bound=dict(decision,baseline_visits=[list(v) for v in visits],baseline_connections=links)
        seam.validate_record(bound,segments,bound['baseline_visits'],links,selected,.01,lambda:None)
