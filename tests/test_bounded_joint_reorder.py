"""Joint reordering preserves anchors and rejects worse reconstructed routes."""
from copy import deepcopy
import unittest
from unittest.mock import patch

from test_verified_visit_improvement import Graph,make_segment
from femto.planning import bounded_joint_reorder as module
from femto.planning.endpoint_preserving_detour import _signature
from femto.planning.scan_visit_order import connect,reconstruct,oriented,_clone_points


class BoundedJointReorderTests(unittest.TestCase):
    def setUp(self):
        self.segments=[make_segment(i,0.) for i in range(4)]
        for segment,row in zip(self.segments,(0,0,1,1)):
            for point in segment.points:point.update(row_index=row,triangle_index=0)
        self.visits=list(enumerate([False]*4));self.graph=Graph(self.segments)
        self.links=connect(self.graph,self.segments,self.visits,lambda:None,1000)

    def test_accepted_route_keeps_scan_edges_and_all_quality_bounds(self):
        original=deepcopy((self.segments,self.visits,self.links))
        base=reconstruct(self.segments,self.visits,self.links)
        before=module._values(base[0],.01,lambda:None)
        chains,links,_,record=module.propose(self.graph,self.segments,self.visits,self.links,.01,lambda:None,1000)
        self.assertTrue(record['adopted'])
        self.assertEqual(_signature(chains,lambda:None),_signature(base[0],lambda:None))
        self.assertEqual((self.segments,self.visits,self.links),original)
        self.assertLessEqual(record['selected']['detours'],before['detours'])
        self.assertLessEqual(record['selected']['history_revisit_m'],before['history_revisit_m']+module.EPS)
        self.assertLessEqual(record['selected']['edge_revisit_m'],before['edge_revisit_m']+module.EPS)
        self.assertLessEqual(record['selected_objective'][1],base[2][1]+module.EPS)
        self.assertEqual(len(links),len(self.links))
        self.assertEqual(module.propose(self.graph,self.segments,self.visits,self.links,.01,lambda:None,1000)[3],record)

    def test_unavailable_paths_and_graph_budget_fall_back(self):
        graph=Graph(self.segments,only_baseline=True)
        _,links,_,record=module.propose(graph,self.segments,self.visits,self.links,.01,lambda:None,1000)
        self.assertFalse(record['adopted']);self.assertEqual(links,self.links)
        with patch.object(module,'connect',side_effect=ValueError('连续路线整边搜索超过预算，请缩小选区')):
            _,links,_,record=module.propose(self.graph,self.segments,self.visits,self.links,.01,lambda:None,1000)
        self.assertFalse(record['adopted']);self.assertEqual(record['skipped'],'geometry_budget');self.assertEqual(links,self.links)

    def test_anchors_are_budgeted_and_corrupt_links_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'超过预算'):
            module._full_anchors(self.segments,self.visits,self.links,lambda:None,10)
        bad=deepcopy(self.links);bad[0]['samples'][-1]['xyz_m']=[9.,9.,9.]
        with self.assertRaises(ValueError):module.propose(self.graph,self.segments,self.visits,bad,.01,lambda:None,1000)
        def cancel():raise RuntimeError('cancelled')
        with self.assertRaisesRegex(RuntimeError,'cancelled'):
            module.propose(self.graph,self.segments,self.visits,self.links,.01,cancel,1000)

    def test_full_anchors_match_original_reconstruction_and_do_not_alias(self):
        def original(segments,visits,links,budget):
            _,added,objective=reconstruct(segments,visits,links,lambda:None,
                preserve_endpoint_faces=True,budget=budget)
            first=oriented(segments[visits[0][0]],visits[0][1]);chains=[(first.block_index,first.points)]
            count=len(first.points)
            for (_, _),(right,reverse),link in zip(visits,visits[1:],links):
                following=oriented(segments[right],reverse)
                count+=len(following.points)+len(link['samples'])
                if count>budget:raise ValueError('连续路线采样点超过预算，请缩小选区')
                if link['connected']:
                    chains[-1][1].extend(_clone_points(link['samples']));chains[-1][1].extend(following.points)
                else:chains.append((following.block_index,following.points))
            return chains,added,objective
        for visits in (self.visits,[(0,False),(2,True),(1,False),(3,True)]):
            links=connect(self.graph,self.segments,visits,lambda:None,1000)
            for variant in ('normal','face_handoff','gap'):
                value=deepcopy(links)
                if variant=='face_handoff':value[0]['samples'][-1]['triangle_index']=9
                if variant=='gap':value[0].update(connected=False,added_edges=0,samples=[],reason='no_verified_graph_path')
                with self.subTest(visits=visits,variant=variant):
                    before=deepcopy((self.segments,value))
                    expected=original(self.segments,visits,value,1000)
                    actual=module._full_anchors(self.segments,visits,value,lambda:None,1000)
                    self.assertEqual(actual,expected)
                    actual[0][0][1][0]['xyz_m'][0]+=1
                    self.assertEqual((self.segments,value),before)
                    for budget in (1,5):
                        with self.assertRaises(ValueError):original(self.segments,visits,value,budget)
                        with self.assertRaises(ValueError):module._full_anchors(self.segments,visits,value,lambda:None,budget)

    def test_baseline_preserves_original_face_handoff(self):
        links=deepcopy(self.links);links[0]['samples'][-1]['triangle_index']=9
        baseline=reconstruct(self.segments,self.visits,links,preserve_endpoint_faces=True)
        with patch.object(module,'_ranked',return_value=[]):
            chains,returned,_,record=module.propose(self.graph,self.segments,self.visits,links,.01,lambda:None,1000)
        self.assertFalse(record['adopted']);self.assertEqual(returned,links)
        self.assertEqual(chains,baseline[0])
        self.assertEqual(record['baseline'],module._values(baseline[0],.01,lambda:None))

    def test_ranking_includes_neutral_and_detour_reducing_moves(self):
        candidates=module._ranked(self.segments,self.visits,self.links,lambda:None)
        self.assertLess(candidates[0][0],0)
        self.assertEqual(candidates[1][0],0)

    def test_budget_failure_after_a_round_discards_partial_steps(self):
        actual=module.connect;calls=0
        def limited(*args,**kwargs):
            nonlocal calls
            calls+=1
            if calls==7:raise ValueError('连续路线整边搜索超过预算，请缩小选区')
            return actual(*args,**kwargs)
        with patch.object(module,'connect',side_effect=limited):
            _,links,_,record=module.propose(self.graph,self.segments,self.visits,self.links,.01,lambda:None,1000)
        self.assertGreaterEqual(calls,7)
        self.assertFalse(record['adopted']);self.assertEqual(record['steps'],[])
        self.assertEqual(record['skipped'],'geometry_budget');self.assertEqual(links,self.links)
        restored,_=module.validate_record(record,self.segments,self.visits,self.links,links,.01,lambda:None,1000)
        self.assertEqual(restored,reconstruct(self.segments,self.visits,self.links,preserve_endpoint_faces=True)[0])

    def test_record_replay_and_tamper_rejection(self):
        chains,links,added,record=module.propose(self.graph,self.segments,self.visits,self.links,.01,lambda:None,1000)
        self.assertTrue(record['adopted'])
        self.assertEqual(module.validate_record(record,self.segments,self.visits,self.links,links,.01,lambda:None,1000),(chains,added))
        for field in ('version','baseline','baseline_objective','steps','visits','selected','selected_objective'):
            bad=deepcopy(record)
            if field=='version':bad[field]='bad'
            elif field in ('baseline','selected'):bad[field]['detours']+=1
            elif field in ('baseline_objective','selected_objective'):bad[field][1]+=1
            elif field=='steps':bad[field][0]['first']=0
            else:bad[field][1][0]=bad[field][0][0]
            with self.subTest(field=field),self.assertRaises(ValueError):
                module.validate_record(bad,self.segments,self.visits,self.links,links,.01,lambda:None,1000)
        bad=deepcopy(links);bad[0]['samples'][-1]['xyz_m']=[99.,0.,0.]
        with self.assertRaises(ValueError):module.validate_record(record,self.segments,self.visits,self.links,bad,.01,lambda:None,1000)
        bad=deepcopy(record);bad['baseline_objective'][0]=False
        with self.assertRaises(ValueError):module.validate_record(bad,self.segments,self.visits,self.links,links,.01,lambda:None,1000)
        bad=deepcopy(record);bad['visits'][0][0]=False
        with self.assertRaises(ValueError):module.validate_record(bad,self.segments,self.visits,self.links,links,.01,lambda:None,1000)

    def test_unadopted_record_requires_original_connections(self):
        graph=Graph(self.segments,only_baseline=True)
        chains,links,added,record=module.propose(graph,self.segments,self.visits,self.links,.01,lambda:None,1000)
        self.assertFalse(record['adopted'])
        self.assertEqual(module.validate_record(record,self.segments,self.visits,self.links,links,.01,lambda:None,1000),(chains,added))
        bad=deepcopy(links);bad[0]['reason']='tampered'
        with self.assertRaises(ValueError):module.validate_record(record,self.segments,self.visits,self.links,bad,.01,lambda:None,1000)

    def test_score_cache_matches_uncached_decisions_and_graph_calls(self):
        class CountGraph(Graph):
            def __init__(self,segments):super().__init__(segments);self.calls=0
            def route(self,*args,**kwargs):self.calls+=1;return super().route(*args,**kwargs)
        rows=[]
        for cap in (0,1,100):
            graph=CountGraph(self.segments)
            with patch.object(module,'MAX_SCORE_CACHE_ENTRIES',cap):
                result=module.propose(graph,self.segments,self.visits,self.links,.01,lambda:None,1000)
            rows.append((result,graph.calls))
        self.assertEqual(rows[0],rows[1]);self.assertEqual(rows[0],rows[2])

    def test_score_cache_rechecks_changes_and_does_not_share_results(self):
        index=module.ConnectorRevisitIndex(self.segments,.01)
        cache=module._ScoreCache(index,lambda:None)
        tail=deepcopy(self.segments[0].points[-1]);head=deepcopy(self.segments[1].points[0])
        samples=[dict(xyz_m=[.5,0.,0.],role='connector',row_index=head['row_index']),
                 dict(xyz_m=list(head['xyz_m']),role='connector',row_index=head['row_index'])]
        with patch.object(index,'measure',wraps=index.measure) as measured:
            expected=cache.measure('edge',tail,head,samples)
            first=cache.measure('edge',deepcopy(tail),deepcopy(head),deepcopy(samples))
            self.assertEqual(measured.call_count,1);self.assertEqual(first,expected)
            first['length_m']=999.
            self.assertEqual(cache.measure('edge',tail,head,samples),expected)
            changed=deepcopy(samples);changed[0]['xyz_m'][0]=1.0
            cache.measure('edge',tail,head,changed)
            self.assertEqual(measured.call_count,2)
            invalid=deepcopy(changed);invalid[0]['xyz_m'][0]=True
            with self.assertRaises(ValueError):cache.measure('edge',tail,head,invalid)

    def test_score_cache_enforces_bounds_and_cancellation(self):
        index=module.ConnectorRevisitIndex(self.segments,.01);tail=self.segments[0].points[-1];head=self.segments[1].points[0]
        samples=[dict(xyz_m=list(head['xyz_m']),role='connector',row_index=head['row_index'])]
        for field in ('MAX_SCORE_CACHE_ENTRIES','MAX_SCORE_CACHE_POINTS','MAX_SCORE_CACHE_BYTES'):
            with self.subTest(field=field),patch.object(module,field,0):
                cache=module._ScoreCache(index,lambda:None);cache.measure('edge',tail,head,samples)
                self.assertEqual(cache.entries,{})
        def cancel():raise RuntimeError('cancelled')
        with self.assertRaisesRegex(RuntimeError,'cancelled'):
            module._ScoreCache(index,cancel).measure('edge',tail,head,samples)
