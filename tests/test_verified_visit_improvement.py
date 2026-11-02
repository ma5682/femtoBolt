"""The proposal never substitutes an unchecked or worse final connection."""
import unittest
from copy import deepcopy
from unittest.mock import patch

from femto.planning.surface_trajectory import ScanSegment
from femto.planning.scan_visit_order import connect, reconstruct
from femto.planning.verified_visit_improvement import improve


def make_segment(index, y, block=0):
    return ScanSegment(index,index,[dict(xyz_m=[0.,y,0.],role='scan',row_index=index),
                                    dict(xyz_m=[1.,y,0.],role='scan',row_index=index)],block)


class Graph:
    def __init__(self,segments,only_baseline=False):
        self.segments=segments;self.only_baseline=only_baseline

    def route(self,left,right,*,left_reverse,right_reverse):
        if self.only_baseline and (right != left+1 or left_reverse or right_reverse):return None
        point=deepcopy(self.segments[right].points[-1 if right_reverse else 0])
        point['role']='connector'
        return [point]


class VerifiedVisitImprovementTests(unittest.TestCase):
    def test_bounded_distance_cache_preserves_result_and_input(self):
        from femto.planning import verified_visit_improvement as module
        segments=[make_segment(i,y,0 if i<7 else 1)
                  for i,y in enumerate((0.,5.,2.,7.,1.,4.,3.,0.,3.,1.,2.))]
        graph=Graph(segments);visits=list(enumerate([False]*len(segments)))
        links=connect(graph,segments,visits,lambda:None,1000)
        before=deepcopy((segments,visits,links))
        with patch.object(module,'MAX_ENDPOINT_DISTANCES',0):
            uncached=improve(graph,segments,visits,links,lambda:None,1000)
        for limit in (1,20,200_000):
            with self.subTest(limit=limit),patch.object(module,'MAX_ENDPOINT_DISTANCES',limit):
                self.assertEqual(improve(graph,segments,visits,links,lambda:None,1000),uncached)
        self.assertEqual((segments,visits,links),before)

    def test_random_three_dimensional_inputs_match_uncached_decisions(self):
        import numpy as np
        from femto.planning import verified_visit_improvement as module
        for seed in range(5):
            rng=np.random.default_rng(seed)
            segments=[make_segment(i,float(i)) for i in range(24)]
            for segment in segments:
                for point in segment.points:point['xyz_m']=rng.normal(size=3).tolist()
            graph=Graph(segments);visits=list(enumerate([False]*len(segments)))
            links=connect(graph,segments,visits,lambda:None,10000)
            with patch.object(module,'MAX_ENDPOINT_DISTANCES',0):
                expected=improve(graph,segments,visits,links,lambda:None,10000)
            with self.subTest(seed=seed),patch.object(module,'MAX_ENDPOINT_DISTANCES',200_000):
                self.assertEqual(improve(graph,segments,visits,links,lambda:None,10000),expected)

    def test_shorter_verified_order_keeps_scans_and_work_blocks(self):
        segments=[make_segment(i,y) for i,y in enumerate((0.,4.,1.,3.,2.))]
        segments.append(make_segment(5,0.,1))
        graph=Graph(segments);visits=list(enumerate([False]*len(segments)))
        links=connect(graph,segments,visits,lambda:None,1000)
        before=deepcopy((segments,visits,links));old=reconstruct(segments,visits,links)[2]
        _,selected,_,decision=improve(graph,segments,visits,links,lambda:None,1000)
        self.assertTrue(decision['adopted'])
        self.assertEqual(decision['visits'][0],[0,False]);self.assertEqual(decision['visits'][-1],[5,False])
        self.assertEqual(sorted(v[0] for v in decision['visits']),list(range(len(segments))))
        result=reconstruct(segments,decision['visits'],selected)[2]
        self.assertEqual(result[0],old[0]);self.assertLess(result[1],old[1])
        self.assertEqual((segments,visits,links),before)
        self.assertEqual(improve(graph,segments,visits,links,lambda:None,1000)[3],decision)

    def test_unavailable_candidate_links_keep_original_route(self):
        segments=[make_segment(i,y) for i,y in enumerate((0.,4.,1.,3.,2.))]
        graph=Graph(segments,only_baseline=True);visits=list(enumerate([False]*len(segments)))
        links=connect(graph,segments,visits,lambda:None,1000)
        _,selected,_,decision=improve(graph,segments,visits,links,lambda:None,1000)
        self.assertFalse(decision['adopted']);self.assertEqual(selected,links)

    def test_final_reconnection_that_is_longer_reverts_to_original(self):
        segments=[make_segment(i,y) for i,y in enumerate((0.,4.,1.,3.,2.))]
        graph=Graph(segments);visits=list(enumerate([False]*len(segments)))
        links=connect(graph,segments,visits,lambda:None,1000)
        def long_reconnect(graph,segments,visits,check,budget):
            result=connect(graph,segments,visits,check,budget)
            point=deepcopy(result[0]['samples'][0]);point['xyz_m']=[100.,100.,100.]
            result[0]['samples'].insert(0,point);result[0]['added_edges']+=1
            return result
        with patch('femto.planning.verified_visit_improvement.connect',side_effect=long_reconnect):
            _,selected,_,decision=improve(graph,segments,visits,links,lambda:None,1000)
        self.assertTrue(decision['rounds']);self.assertFalse(decision['adopted'])
        self.assertEqual(selected,links)

    def test_cancellation_and_empty_input(self):
        with self.assertRaisesRegex(ValueError,'不能为空'):
            improve(None,[],[],[],lambda:None,1000)
        segments=[make_segment(i,float(i)) for i in range(3)]
        graph=Graph(segments);visits=list(enumerate([False]*len(segments)))
        links=connect(graph,segments,visits,lambda:None,1000)
        def cancel():raise RuntimeError('cancelled')
        with self.assertRaisesRegex(RuntimeError,'cancelled'):
            improve(graph,segments,visits,links,cancel,1000)
