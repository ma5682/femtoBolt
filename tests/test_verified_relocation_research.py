"""Relocation proposals retain scans, starts and checked motion limits."""
from copy import deepcopy
import random
import unittest
from unittest.mock import patch

from test_verified_visit_improvement import Graph, make_segment
from femto.planning import verified_relocation_research as module
from femto.planning.endpoint_preserving_detour import _signature
from femto.planning.scan_visit_order import connect
from femto.planning.supplement_two_opt_research import _anchored


class RelocationResearchTests(unittest.TestCase):
    def setUp(self):
        self.segments=[make_segment(i,float(y),0 if i<5 else 1)
                       for i,y in enumerate((0,4,1,3,2,0,1))]
        for segment in self.segments:
            for point in segment.points:point['triangle_index']=0
        self.graph=Graph(self.segments)
        self.visits=list(enumerate([False]*len(self.segments)))
        self.links=connect(self.graph,self.segments,self.visits,lambda:None,1000)
        initial=_anchored(self.segments,self.visits,self.links,lambda:None,1000)
        metrics=module.balanced_visit_improvement._metrics(initial[0],.01,lambda:None)
        self.limits=dict(detours=metrics['detours'],revisit_connector_m=metrics['revisit_connector_m'],
                         connector_length_m=initial[2][1])

    def test_relocation_preserves_scans_and_block_starts(self):
        before=deepcopy((self.segments,self.visits,self.links))
        original=_anchored(self.segments,self.visits,self.links,lambda:None,1000)
        chains,links,_,report=module.propose(self.graph,self.segments,self.visits,self.links,.01,
                                             motion_limits=self.limits,attempts=256,budget=1000)
        self.assertTrue(report['adopted'])
        self.assertEqual(report['visits'][0],[0,False])
        self.assertEqual(next(v for v in report['visits'] if v[0]>=5),[5,False])
        self.assertEqual(_signature(chains,lambda:None),_signature(original[0],lambda:None))
        self.assertLess(report['selected_objective'][1],report['baseline_objective'][1])
        self.assertLessEqual(report['selected']['detours'],report['baseline']['detours'])
        self.assertLessEqual(report['selected']['revisit_connector_m'],report['baseline']['revisit_connector_m'])
        self.assertEqual((self.segments,self.visits,self.links),before)
        links[0]['samples'][0]['xyz_m'][0]=99.
        self.assertEqual((self.segments,self.visits,self.links),before)

    def test_explicit_detour_priority_can_use_only_declared_length_allowance(self):
        self.visits=[(0,False),(2,True),(4,False),(3,True),(1,False),(5,False),(6,True)]
        self.links=connect(self.graph,self.segments,self.visits,lambda:None,1000)
        self.limits['connector_length_m']=module.scan_visit_order.anchored_objective(self.segments,self.visits,self.links,budget=1000)[1]
        before=deepcopy((self.segments,self.visits,self.links))
        candidates=module._candidates(self.segments,self.visits,lambda:None)
        chosen=None
        for proposal in reversed(candidates):
            _,i,j,reverse=proposal;value=self.visits[i];remaining=self.visits[:i]+self.visits[i+1:]
            visits=remaining[:j]+[(value[0],reverse)]+remaining[j:]
            links=connect(self.graph,self.segments,visits,lambda:None,1000)
            objective=module.scan_visit_order.anchored_objective(self.segments,visits,links,budget=1000)
            if not objective[0] and objective[1]>self.limits['connector_length_m']+1e-9:
                chosen=proposal;length=objective[1];break
        self.assertIsNotNone(chosen)
        limits=dict(detours=5,revisit_connector_m=.1,connector_length_m=length+1.)
        def metric_sequence():
            calls=0
            def metrics(*args):
                nonlocal calls
                calls+=1
                return dict(detours=5 if calls==1 else 4,revisit_connector_m=.05)
            return metrics
        with patch.object(module,'_candidates',return_value=[chosen]):
            with patch.object(module.balanced_visit_improvement,'_metrics',side_effect=metric_sequence()):
                _,_,_,pareto=module.propose(self.graph,self.segments,self.visits,self.links,.01,motion_limits=limits,attempts=1,rounds=1,budget=1000)
            self.assertFalse(pareto['adopted'])
            with patch.object(module.balanced_visit_improvement,'_metrics',side_effect=metric_sequence()):
                chains,links,_,record=module.propose(self.graph,self.segments,self.visits,self.links,.01,motion_limits=limits,
                    attempts=1,rounds=1,budget=1000,objective_policy='detour_first_with_limits')
            self.assertTrue(record['adopted']);self.assertEqual(record['version'],module.LIMITED_DETOUR_VERSION)
            self.assertGreater(record['selected_objective'][1],record['baseline_objective'][1])
            with patch.object(module.balanced_visit_improvement,'_metrics',side_effect=metric_sequence()):
                replayed,_=module.validate_record(record,self.segments,self.visits,self.links,links,.01,motion_limits=limits,
                    attempts=1,rounds=1,budget=1000,objective_policy='detour_first_with_limits')
            self.assertEqual(chains,replayed)
            with self.assertRaises(ValueError):module.validate_record(record,self.segments,self.visits,self.links,links,.01,motion_limits=limits,attempts=1,rounds=1,budget=1000)
            for strict in (dict(limits,connector_length_m=self.limits['connector_length_m']),dict(limits,revisit_connector_m=0.)):
                with patch.object(module.balanced_visit_improvement,'_metrics',side_effect=metric_sequence()):
                    _,_,_,refused=module.propose(self.graph,self.segments,self.visits,self.links,.01,motion_limits=strict,
                        attempts=1,rounds=1,budget=1000,objective_policy='detour_first_with_limits')
                self.assertFalse(refused['adopted'])
        self.assertEqual((self.segments,self.visits,self.links),before)
        links[0]['samples'][0]['xyz_m'][0]=99.
        self.assertEqual((self.segments,self.visits,self.links),before)

    def test_disconnected_alternatives_return_original(self):
        graph=Graph(self.segments,only_baseline=True)
        links=connect(graph,self.segments,self.visits,lambda:None,1000)
        _,selected,_,report=module.propose(graph,self.segments,self.visits,links,.01,motion_limits=self.limits,budget=1000)
        self.assertFalse(report['adopted'])
        self.assertEqual(selected,links)

    def test_shorter_route_is_rejected_when_revisit_increases(self):
        calls=0
        def increased_revisit(*args):
            nonlocal calls
            calls+=1
            return dict(detours=0,revisit_connector_m=0. if calls==1 else 1.)
        with patch.object(module.balanced_visit_improvement,'_metrics',side_effect=increased_revisit):
            _,selected,_,report=module.propose(self.graph,self.segments,self.visits,self.links,.01,motion_limits=self.limits,budget=1000)
        self.assertGreater(calls,1)
        self.assertFalse(report['adopted'])
        self.assertEqual(selected,self.links)

    def test_published_revisit_limit_overrides_reconstructed_baseline(self):
        calls=0
        def mismatch(*args):
            nonlocal calls
            calls+=1
            return dict(detours=0,revisit_connector_m=.2 if calls==1 else .1)
        limits=dict(self.limits,revisit_connector_m=0.)
        with patch.object(module.balanced_visit_improvement,'_metrics',side_effect=mismatch):
            _,selected,_,report=module.propose(self.graph,self.segments,self.visits,self.links,.01,motion_limits=limits,budget=1000)
        self.assertGreater(calls,1)
        self.assertFalse(report['adopted'])
        self.assertEqual(selected,self.links)

    def test_incremental_distance_matches_full_estimate(self):
        from femto.planning.scan_visit_order import checked_visits
        visits=[(0,False),(3,True),(1,False),(4,True),(2,False),(5,False),(6,True)]
        with patch.object(module,'MAX_DISTANCE_CACHE',0):
            without_cache=module._candidates(self.segments,visits,lambda:None)
        with patch.object(module,'MAX_DISTANCE_CACHE',10000):
            with_cache=module._candidates(self.segments,visits,lambda:None)
        self.assertEqual(with_cache,without_cache)
        for estimate,i,j,reverse in with_cache:
            value=visits[i];remaining=visits[:i]+visits[i+1:]
            proposed=remaining[:j]+[(value[0],reverse)]+remaining[j:]
            checked_visits(self.segments,proposed)
            self.assertAlmostEqual(estimate,module._estimate(self.segments,proposed),places=12)

    def test_distance_cache_reuses_rounds_and_invalidates_changed_endpoints(self):
        visits=[(0,False),(3,True),(1,False),(4,True),(2,False),(5,False),(6,True)]
        cache={};original_norm=module.np.linalg.norm
        with patch.object(module.np.linalg,'norm',wraps=original_norm) as measured:
            first=module._candidates(self.segments,visits,lambda:None,cache)
            calls=measured.call_count;self.assertGreater(calls,0)
            second=module._candidates(self.segments,visits,lambda:None,cache)
            self.assertEqual(measured.call_count,calls);self.assertEqual(first,second)
            self.segments[3].points[-1]['xyz_m'][0]+=.125
            changed=module._candidates(self.segments,visits,lambda:None,cache)
            self.assertGreater(measured.call_count,calls)
        self.assertEqual(changed,module._candidates(self.segments,visits,lambda:None))
        with patch.object(module,'MAX_DISTANCE_CACHE',0):
            self.assertEqual(changed,module._candidates(self.segments,visits,lambda:None,cache))
        def cancel():raise RuntimeError('cancelled')
        with self.assertRaisesRegex(RuntimeError,'cancelled'):
            module._candidates(self.segments,visits,cancel,cache)

    def test_round_cache_keeps_complete_relocation_result_identical(self):
        with patch.object(module,'MAX_DISTANCE_CACHE',0):
            before=module.propose(self.graph,self.segments,self.visits,self.links,.01,
                                  motion_limits=self.limits,attempts=32,rounds=2,budget=1000)
        after=module.propose(self.graph,self.segments,self.visits,self.links,.01,
                             motion_limits=self.limits,attempts=32,rounds=2,budget=1000)
        self.assertEqual(after,before)

    def test_supports_more_than_128_scans_with_bounded_search(self):
        segments=[make_segment(i,float(i)) for i in range(129)]
        for segment in segments:
            for point in segment.points:point['triangle_index']=0
        visits=list(enumerate([False]*len(segments)))
        graph=Graph(segments,only_baseline=True)
        links=connect(graph,segments,visits,lambda:None,10000)
        initial=module.scan_visit_order.reconstruct(segments,visits,links,preserve_endpoint_faces=True,budget=10000)
        metrics=module.balanced_visit_improvement._metrics(initial[0],.01,lambda:None)
        limits=dict(detours=metrics['detours'],revisit_connector_m=metrics['revisit_connector_m'],connector_length_m=initial[2][1])
        _,selected,_,report=module.propose(graph,segments,visits,links,.01,motion_limits=limits,attempts=1,rounds=1,budget=10000)
        self.assertEqual(report['tested'],1)
        self.assertFalse(report['adopted'])
        self.assertEqual(selected,links)

    def test_preview_matches_full_reconstruction_without_mutating_inputs(self):
        from femto.planning.scan_visit_order import anchored_objective,reconstruct
        for seed in range(5):
            rng=random.Random(seed)
            segments=deepcopy(self.segments)
            for segment in segments:
                segment.points[0]['revisit']=True
            middle=list(range(1,5));rng.shuffle(middle)
            visits=[(0,False)]+[(i,bool(rng.getrandbits(1))) for i in middle]+[(5,False),(6,bool(rng.getrandbits(1)))]
            links=connect(Graph(segments),segments,visits,lambda:None,1000)
            for i,link in enumerate(links):
                if link['connected']:
                    link['samples'][-1]['triangle_index']=i%2
                    link['samples'][-1]['revisit']=True
            before=deepcopy((segments,visits,links))
            anchored_objective(segments,visits,links,budget=1000)
            full=reconstruct(segments,visits,links,preserve_endpoint_faces=True,budget=1000)[0]
            expected=module.balanced_visit_improvement._metrics(full,.01,lambda:None)
            for limit in (0,1,10000):
                with self.subTest(seed=seed,limit=limit),patch.object(module,'MAX_PREVIEW_POINTS',limit):
                    preview=module._Preview(segments,lambda:None)
                    actual=module.balanced_visit_improvement._metrics(preview.chains(visits,links),.01,lambda:None)
                    self.assertEqual(actual,expected)
            self.assertEqual((segments,visits,links),before)

    def test_preview_cache_does_not_change_selected_route(self):
        results=[]
        for limit in (0,1,10000):
            with patch.object(module,'MAX_PREVIEW_POINTS',limit):
                results.append(module.propose(self.graph,self.segments,self.visits,self.links,.01,
                                              motion_limits=self.limits,attempts=256,budget=1000))
        self.assertEqual(results[0],results[1])
        self.assertEqual(results[0],results[2])

    def test_preview_mismatch_fails_before_returning_candidate(self):
        original=module.balanced_visit_improvement._metrics;calls=0
        def inconsistent(*args):
            nonlocal calls
            calls+=1;result=original(*args)
            if calls%2==0:result['revisit_connector_m']=-1.
            return result
        before=deepcopy((self.segments,self.visits,self.links))
        with patch.object(module.balanced_visit_improvement,'_metrics',side_effect=inconsistent):
            with self.assertRaisesRegex(ValueError,'预筛与完整重建不一致'):
                module.propose(self.graph,self.segments,self.visits,self.links,.01,
                               motion_limits=self.limits,attempts=256,budget=1000)
        self.assertEqual((self.segments,self.visits,self.links),before)

    def test_record_replay_rechecks_steps_limits_and_selected_links(self):
        chains,links,added,record=module.propose(self.graph,self.segments,self.visits,self.links,.01,
                                                motion_limits=self.limits,attempts=256,budget=1000)
        self.assertTrue(record['adopted'])
        replayed,replayed_added=module.validate_record(record,self.segments,self.visits,self.links,links,.01,
                                                       motion_limits=self.limits,attempts=256,budget=1000)
        self.assertEqual(replayed,chains);self.assertEqual(replayed_added,added)
        altered=[]
        bad=deepcopy(record);bad['steps'][0]['source']=0;altered.append(bad)
        bad=deepcopy(record);bad['visits'][0]=[1,False];altered.append(bad)
        bad=deepcopy(record);bad['selected']['revisit_connector_m']+=1.;altered.append(bad)
        bad=deepcopy(record);bad['tested']=0;altered.append(bad)
        bad=deepcopy(record);bad['motion_limits']['detours']+=1;altered.append(bad)
        bad=deepcopy(record);bad['selected']['detours']=False;altered.append(bad)
        bad=deepcopy(record);bad['selected_objective'][1]=float('nan');altered.append(bad)
        for value in altered:
            with self.subTest(value=value),self.assertRaises(ValueError):
                module.validate_record(value,self.segments,self.visits,self.links,links,.01,
                                       motion_limits=self.limits,attempts=256,budget=1000)
        broken=deepcopy(links);broken[0]['to_segment']=99
        with self.assertRaises(ValueError):
            module.validate_record(record,self.segments,self.visits,self.links,broken,.01,
                                   motion_limits=self.limits,attempts=256,budget=1000)

    def test_record_replay_rejects_changed_fallback_links(self):
        graph=Graph(self.segments,only_baseline=True)
        links=connect(graph,self.segments,self.visits,lambda:None,1000)
        _,selected,_,record=module.propose(graph,self.segments,self.visits,links,.01,motion_limits=self.limits,budget=1000)
        self.assertFalse(record['adopted'])
        module.validate_record(record,self.segments,self.visits,links,selected,.01,motion_limits=self.limits,budget=1000)
        altered=deepcopy(selected);altered[0]['samples'][0]['xyz_m']=[100.,100.,100.]
        with self.assertRaises(ValueError):
            module.validate_record(record,self.segments,self.visits,links,altered,.01,motion_limits=self.limits,budget=1000)

    def test_invalid_inputs_and_cancellation(self):
        for options in (dict(attempts=0),dict(rounds=5),dict(spacing_m=float('nan')),dict(budget=True),
                        dict(motion_limits={'detours':False,'connector_length_m':1.,'revisit_connector_m':0.}),
                        dict(motion_limits={'detours':0,'connector_length_m':float('nan'),'revisit_connector_m':0.})):
            args=dict(spacing_m=.01,motion_limits=self.limits,budget=1000);args.update(options)
            with self.subTest(options=options),self.assertRaises(ValueError):
                module.propose(self.graph,self.segments,self.visits,self.links,**args)
        broken=deepcopy(self.links);broken[0]['to_segment']=99
        with self.assertRaises(ValueError):module.propose(self.graph,self.segments,self.visits,broken,.01,motion_limits=self.limits,budget=1000)
        def cancel():raise RuntimeError('cancelled')
        with self.assertRaisesRegex(RuntimeError,'cancelled'):
            module.propose(self.graph,self.segments,self.visits,self.links,.01,motion_limits=self.limits,check=cancel,budget=1000)


if __name__=='__main__':unittest.main()
