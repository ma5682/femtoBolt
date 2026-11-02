"""Research proposal never bypasses the graph or the scan-source contract."""
from copy import deepcopy
import unittest
from unittest.mock import patch
from test_verified_visit_improvement import Graph,make_segment
from femto.planning import endpoint_preserving_detour as module
from femto.planning.scan_visit_order import anchored_objective,connect,oriented,reconstruct


class PreservedDetourTests(unittest.TestCase):
    def setUp(self):
        self.segments=[make_segment(i,0.) for i in range(4)]
        for segment,row in zip(self.segments,(0,0,1,1)):
            for point in segment.points:point.update(row_index=row,triangle_index=0)
        self.visits=list(enumerate([False]*4));self.graph=Graph(self.segments)
        self.links=connect(self.graph,self.segments,self.visits,lambda:None,1000)

    def test_candidate_preserves_edges_source_and_decreases_detours(self):
        before=deepcopy((self.segments,self.visits,self.links))
        baseline=reconstruct(self.segments,self.visits,self.links)
        chains,links,_,decision=module.propose(self.graph,self.segments,self.visits,self.links,.01,lambda:None,1000)
        self.assertTrue(decision['adopted'])
        self.assertLess(decision['selected']['detours'],decision['baseline']['detours'])
        self.assertLessEqual(decision['selected']['revisit_connector_m'],decision['baseline']['revisit_connector_m']+1e-12)
        self.assertLessEqual(decision['selected_objective'][1],baseline[2][1]*1.04+1e-12)
        self.assertEqual(module._signature(chains,lambda:None),module._signature(baseline[0],lambda:None))
        self.assertEqual((self.segments,self.visits,self.links),before)
        self.assertEqual(module.propose(self.graph,self.segments,self.visits,self.links,.01,lambda:None,1000)[3],decision)
        self.assertEqual(len(links),len(self.links))

    def test_short_candidates_have_distinct_replay_and_preserve_contract(self):
        chains,links,added,record=module.propose(self.graph,self.segments,self.visits,self.links,.01,lambda:None,1000,short_candidates=True)
        self.assertEqual(record['version'],module.SHORT_VERSION)
        baseline=reconstruct(self.segments,self.visits,self.links)
        self.assertEqual(module._signature(chains,lambda:None),module._signature(baseline[0],lambda:None))
        self.assertEqual(module.validate_record(record,self.segments,self.visits,self.links,links,.01,lambda:None,1000,short_candidates=True),(chains,added))
        with self.assertRaises(ValueError):
            module.validate_record(record,self.segments,self.visits,self.links,links,.01,lambda:None,1000)
        if record['adopted']:
            self.assertTrue(all(step['last']-step['first']<=2 for step in record['steps']))

    def test_short_candidate_selection_and_budget_are_bounded(self):
        candidates=[(-1,float(i),1,1+(i%3 if i%4 else 5)) for i in range(20)]
        with patch.object(module,'_ranked',return_value=candidates),patch.object(module,'connect',side_effect=ValueError('连续路线采样点超过预算，请缩小选区')) as connect_mock:
            _,links,_,record=module.propose(self.graph,self.segments,self.visits,self.links,.01,lambda:None,1000,short_candidates=True)
        self.assertFalse(record['adopted']);self.assertEqual(links,self.links)
        self.assertEqual(connect_mock.call_count,8)

    def test_unavailable_graph_links_fall_back_unchanged(self):
        graph=Graph(self.segments,only_baseline=True)
        chains,links,_,decision=module.propose(graph,self.segments,self.visits,self.links,.01,lambda:None,1000)
        self.assertFalse(decision['adopted']);self.assertEqual(links,self.links)
        self.assertEqual(chains,reconstruct(self.segments,self.visits,self.links)[0])

    def test_candidate_that_loses_source_face_is_rejected(self):
        with patch.object(module,'_signature',side_effect=[('old',),('new',),('new',),('new',),('new',)]):
            _,links,_,decision=module.propose(self.graph,self.segments,self.visits,self.links,.01,lambda:None,1000)
        self.assertFalse(decision['adopted']);self.assertEqual(links,self.links)

    def test_candidate_budget_and_cancellation(self):
        with patch.object(module,'_anchored',side_effect=ValueError('连续路线采样点超过预算，请缩小选区')):
            _,links,_,decision=module.propose(self.graph,self.segments,self.visits,self.links,.01,lambda:None,1000)
        self.assertFalse(decision['adopted']);self.assertEqual(links,self.links)
        def cancel():raise RuntimeError('cancelled')
        with self.assertRaisesRegex(RuntimeError,'cancelled'):
            module.propose(self.graph,self.segments,self.visits,self.links,.01,cancel,1000)

    def test_graph_budget_falls_back_without_leaking_partial_result(self):
        with patch.object(module,'connect',side_effect=ValueError('连续路线整边搜索超过预算，请缩小选区')):
            chains,links,added,record=module.propose(self.graph,self.segments,self.visits,self.links,.01,lambda:None,1000)
        self.assertFalse(record['adopted']);self.assertEqual(record['skipped'],'geometry_budget')
        self.assertEqual(links,self.links)
        self.assertEqual(module.validate_record(record,self.segments,self.visits,self.links,links,.01,lambda:None,1000),(chains,added))
        bad=deepcopy(record);bad['steps']=[dict(first=1,last=1)]
        with self.assertRaises(ValueError):
            module.validate_record(bad,self.segments,self.visits,self.links,links,.01,lambda:None,1000)

    def test_anchor_is_counted_in_budget(self):
        links=deepcopy(self.links);links[0]['samples'][-1]['triangle_index']=1
        with self.assertRaisesRegex(ValueError,'超过预算'):
            module._anchored(self.segments,self.visits,links,lambda:None,8)

    def test_single_pass_matches_old_reconstruction_and_keeps_independent_points(self):
        scans=deepcopy(self.segments)
        scans.append(make_segment(4,2.,block=1))
        for point in scans[-1].points:point['triangle_index']=2
        scans[1].points[1]['nested']={'values':[1,2]}
        visits=[(0,False),(1,True),(2,False),(3,True),(4,False)]
        links=connect(Graph(scans),scans,visits,lambda:None,1000)
        links[1]['samples'][-1]['triangle_index']=9
        initial=deepcopy((scans,visits,links))
        _,added,objective=reconstruct(scans,visits,links)
        first=oriented(scans[visits[0][0]],visits[0][1]);old=[(first.block_index,first.points)]
        for (_, _),(right,reverse),link in zip(visits,visits[1:],links):
            following=oriented(scans[right],reverse)
            if link['connected']:
                samples=link['samples'];keep=samples[-1]['triangle_index']!=following.points[0]['triangle_index']
                old[-1][1].extend(deepcopy(samples));old[-1][1].extend(following.points if keep else following.points[1:])
            else:old.append((following.block_index,following.points))
        chains,new_added,new_objective=module._anchored(scans,visits,links,lambda:None,1000)
        self.assertEqual((chains,new_added,new_objective),(old,added,objective))
        self.assertEqual((scans,visits,links),initial)
        chains[0][1][0]['xyz_m'][0]=123
        self.assertEqual((scans,visits,links),initial)

    def test_single_pass_rejects_invalid_link_and_cancel(self):
        bad=deepcopy(self.links);bad[1]['samples'][-1]['xyz_m']=[99.,99.,99.]
        with self.assertRaises(ValueError):module._anchored(self.segments,self.visits,bad,lambda:None,1000)
        def cancel():raise RuntimeError('cancelled')
        with self.assertRaisesRegex(RuntimeError,'cancelled'):module._anchored(self.segments,self.visits,self.links,cancel,1000)

    def test_objective_prefilter_matches_reconstruction_for_orientations_and_gaps(self):
        scans=deepcopy(self.segments)
        scans.append(make_segment(4,2.,block=1))
        for point in scans[-1].points:point['triangle_index']=2
        visits=[(0,False),(2,True),(1,False),(3,True),(4,False)]
        for restricted in (False,True):
            links=connect(Graph(scans,only_baseline=restricted),scans,visits,lambda:None,1000)
            for link in links:
                if link['samples']:
                    link['samples'][-1]['triangle_index']=9
                    break
            before=deepcopy((scans,visits,links))
            for budget in (8,10,1000):
                with self.subTest(restricted=restricted,budget=budget):
                    try:
                        expected=module._anchored(scans,visits,links,lambda:None,budget)[2]
                    except ValueError:
                        with self.assertRaises(ValueError):
                            anchored_objective(scans,visits,links,budget=budget)
                    else:
                        self.assertEqual(anchored_objective(scans,visits,links,budget=budget),expected)
            self.assertEqual((scans,visits,links),before)

    def test_objective_prefilter_rejects_invalid_links(self):
        for field,value in (('from_segment',99),('reason','wrong'),('connected',False),('added_edges',99)):
            bad=deepcopy(self.links);bad[0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                anchored_objective(self.segments,self.visits,bad,budget=1000)
        for field,value in (('xyz_m',[99.,99.,99.]),('role','scan'),('row_index',99)):
            bad=deepcopy(self.links);bad[0]['samples'][-1][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                anchored_objective(self.segments,self.visits,bad,budget=1000)
        def cancel():raise RuntimeError('cancelled')
        with self.assertRaisesRegex(RuntimeError,'cancelled'):
            anchored_objective(self.segments,self.visits,self.links,cancel,budget=1000)

    def test_record_rebuild_and_tamper_rejection(self):
        chains,links,added,record=module.propose(self.graph,self.segments,self.visits,self.links,.01,lambda:None,1000)
        restored,restored_added=module.validate_record(record,self.segments,self.visits,self.links,links,.01,lambda:None,1000)
        self.assertEqual((restored,restored_added),(chains,added))
        for field in ('baseline','selected','visits','steps','selected_objective'):
            bad=deepcopy(record)
            if field in ('baseline','selected'):bad[field]['detours']+=1
            elif field=='visits':bad[field][1][0]=bad[field][0][0]
            elif field=='steps':bad[field][0]['first']=0
            else:bad[field][1]+=1
            with self.subTest(field=field),self.assertRaises(ValueError):
                module.validate_record(bad,self.segments,self.visits,self.links,links,.01,lambda:None,1000)

    def test_unadopted_record_refuses_different_links(self):
        graph=Graph(self.segments,only_baseline=True)
        chains,links,added,record=module.propose(graph,self.segments,self.visits,self.links,.01,lambda:None,1000)
        self.assertEqual(module.validate_record(record,self.segments,self.visits,self.links,links,.01,lambda:None,1000),(chains,added))
        changed=deepcopy(links);changed[0]['reason']='tampered'
        with self.assertRaises(ValueError):
            module.validate_record(record,self.segments,self.visits,self.links,changed,.01,lambda:None,1000)
