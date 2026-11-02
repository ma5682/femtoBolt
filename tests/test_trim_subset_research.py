"""Protected subset proposals retain sources and reject unproven results."""
from contextlib import ExitStack
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from test_verified_visit_improvement import make_segment
from femto.planning import trim_subset_research as module


class TrimSubsetResearchTests(unittest.TestCase):
    def setUp(self):
        self.segments=[make_segment(0,0.),make_segment(1,1.)]
        for segment in self.segments:
            segment.points.insert(1,dict(segment.points[0],xyz_m=[.5,segment.points[0]['xyz_m'][1],0.]))
            for point in segment.points:point['triangle_index']=0
        self.item={'base_segment':1,'side':'start'}

    def test_apply_preserves_inputs_and_rejects_anchor_or_duplicate(self):
        before=deepcopy(self.segments)
        trimmed=module._apply(self.segments,[self.item],[self.item])
        self.assertEqual(len(trimmed[1].points),2)
        self.assertEqual(self.segments,before)
        trimmed[1].points[0]['xyz_m'][0]=99
        self.assertEqual(self.segments,before)
        with self.assertRaisesRegex(ValueError,'起点'):
            item={'base_segment':0,'side':'start'};module._apply(self.segments,[item],[item])
        with self.assertRaisesRegex(ValueError,'重复'):
            module._apply(self.segments,[self.item],[self.item,self.item])
        with self.assertRaisesRegex(ValueError,'受保护'):
            module._apply(self.segments,[],[self.item])
        end={'base_segment':1,'side':'end'}
        with self.assertRaisesRegex(ValueError,'不足两点'):
            module._apply(self.segments,[self.item,end],[self.item,end])
        extended=deepcopy(self.segments)
        extended[1].points.insert(1,dict(extended[1].points[0],xyz_m=[.25,1.,0.]))
        self.assertEqual(len(module._apply(extended,[self.item,end],[self.item,end])[1].points),2)

    def test_invalid_budget_or_prior_proof_stops_before_candidates(self):
        for args in (dict(max_candidates=0),dict(max_proofs=True),dict(budget=0),dict(motion_prefilter=1)):
            with self.subTest(args=args),self.assertRaises(ValueError):
                module.build(None,{}, {},object(),[],[],object(),**args)
        with patch.object(module,'_prepare',side_effect=ValueError('bad proof')), \
             patch.object(module.terminal_overlap_research,'propose') as proposal:
            with self.assertRaisesRegex(ValueError,'bad proof'):
                module.build(None,{}, {},object(),[],[],object())
            proposal.assert_not_called()

    def test_rejects_mismatched_density_sampling(self):
        before={'proximity':[{'radius_m':.001,'near_fraction':.2},{'radius_m':.002,'near_fraction':.3}]}
        after={'proximity':[{'radius_m':.001,'near_fraction':.1}]}
        with self.assertRaisesRegex(ValueError,'配置不一致'):module._density_not_worse(after,before)
        after['proximity'].append({'radius_m':.003,'near_fraction':.1})
        with self.assertRaisesRegex(ValueError,'配置不一致'):module._density_not_worse(after,before)
        after['proximity'][1]['radius_m']=.002
        self.assertTrue(module._density_not_worse(after,before))

    def test_pair_partners_are_ranked_without_admitting_them(self):
        selected=[{'base_segment':i,'side':'end'} for i in range(4)]
        trials=[{'trim':[selected[0]],'connector_lower_bound_m':.9},
                {'trim':[selected[3]],'geometry':{'connector_length_m':1.}},
                {'trim':[selected[2]],'reason':'internal_gap'}]
        pairs=module._pair_subsets(selected,selected[1],trials)
        self.assertEqual([pair[1] for pair in pairs],[selected[0],selected[3],selected[2]])
        self.assertTrue(all(pair[0]==selected[1] for pair in pairs))

    def test_requires_both_retention_proofs_before_returning_recipe(self):
        before={'proximity':[{'radius_m':.001,'near_fraction':.2},{'radius_m':.002,'near_fraction':.3}]}
        after={'proximity':[{'radius_m':.001,'near_fraction':.1},{'radius_m':.002,'near_fraction':.2}]}
        oldstats={'connected_detours':2,'connector_length_m':1.,'revisit_connector_length_m':.1,'total_length_m':2.}
        newstats={'connected_detours':2,'connector_length_m':.9,'revisit_connector_length_m':.09,'total_length_m':1.9}
        params={'quality_evidence':{},'effective_domain':{},'effective_width_mm':5.}
        base=SimpleNamespace(planner={'parameters':params},statistics=oldstats)
        domain=SimpleNamespace(binding={},revision={'sha256':'revision'},areas=[1.],ledger=SimpleNamespace(arrays={'region_faces':[0]}))
        state=dict(domain=domain,base=base,binding={},segments=self.segments,visits=[(0,False),(1,False)],
                   links=[{'connected':False}],mesh=None,prior=base,prior_stats=oldstats,footprint={})
        proof=SimpleNamespace(summary={})
        class FakeGraph:
            def __init__(self,*args):pass
        for baseline_retained,previous_retained,prefilter in ((False,True,True),(True,False,True),(True,True,True),(True,True,False)):
            retained=baseline_retained and previous_retained
            with self.subTest(baseline=baseline_retained,previous=previous_retained,prefilter=prefilter),ExitStack() as stack:
                def mock(obj,name,**kwargs):return stack.enter_context(patch.object(obj,name,**kwargs))
                mock(module,'_prepare',return_value=state)
                mock(module,'_apply',return_value=self.segments)
                mock(module,'replace',return_value=base)
                mock(module.row_proximity_research,'measure',side_effect=[before,before])
                mock(module.terminal_overlap_research,'propose',return_value=([],{'selected':[self.item]}))
                mock(module.effective_domain,'validate',return_value=None)
                lower=mock(module,'_connector_lower_bound',return_value=None)
                mock(module.continuous_quality,'LazyVerifiedSurfaceGraph',new=FakeGraph)
                mock(module.scan_visit_order,'connect',return_value=[{'connected':True}])
                objective=mock(module.scan_visit_order,'anchored_objective',return_value=(0,.9))
                mock(module,'_candidate',return_value=(base,newstats,{'items_sha256':'x'},after))
                mock(module,'scan_endpoints',return_value=[])
                mock(module.coverage,'checked_page_binding',return_value=False)
                mock(module.coverage,'evaluate',return_value=proof)
                mock(module.coverage,'validate')
                mock(module.coverage_partition,'compare_union',return_value={'retained_subset':baseline_retained})
                mock(module.coverage_partition,'compare',return_value={'retained_subset':previous_retained})
                mock(module,'_check_end')
                result,returned=module.build(None,{}, {},object(),[(0,False),(1,False)],[{'connected':False}],object(),motion_prefilter=prefilter)
                self.assertEqual(lower.called,prefilter)
                self.assertEqual(objective.called,prefilter)
            self.assertEqual(result['adopted'],retained)
            self.assertEqual('recipe' in result,retained)
            self.assertIs(returned,proof if retained else None)
            self.assertFalse(result['execution_ready'])

    def test_prefilter_rejects_long_connections_before_geometry_or_proof(self):
        density={'proximity':[{'radius_m':.001,'near_fraction':.2},{'radius_m':.002,'near_fraction':.3}]}
        stats={'connected_detours':2,'connector_length_m':1.,'revisit_connector_length_m':.1}
        params={'quality_evidence':{},'effective_domain':{},'effective_width_mm':5.}
        base=SimpleNamespace(planner={'parameters':params},statistics=stats)
        domain=SimpleNamespace(binding={},revision={'sha256':'revision'},areas=[1.],ledger=SimpleNamespace(arrays={'region_faces':[0]}))
        state=dict(domain=domain,base=base,binding={},segments=self.segments,visits=[(0,False),(1,False)],
                   links=[{'connected':False}],mesh=None,prior=base,prior_stats=stats,footprint={})
        class FakeGraph:
            def __init__(self,*args):pass
        with ExitStack() as stack:
            def mock(obj,name,**kwargs):return stack.enter_context(patch.object(obj,name,**kwargs))
            mock(module,'_prepare',return_value=state);mock(module,'_apply',return_value=self.segments)
            mock(module,'replace',return_value=base)
            mock(module.row_proximity_research,'measure',side_effect=[density,density])
            mock(module.terminal_overlap_research,'propose',return_value=([],{'selected':[self.item]}))
            mock(module.effective_domain,'validate',return_value=None)
            mock(module,'_connector_lower_bound',return_value=None)
            mock(module.continuous_quality,'LazyVerifiedSurfaceGraph',new=FakeGraph)
            mock(module.scan_visit_order,'connect',return_value=[{'connected':True}])
            mock(module.scan_visit_order,'anchored_objective',return_value=(0,1.01))
            geometry=mock(module,'_candidate');proof=mock(module.coverage,'evaluate')
            mock(module,'_check_end')
            result,returned=module.build(None,{}, {},object(),[(0,False),(1,False)],[{'connected':False}],object())
        self.assertFalse(result['adopted']);self.assertIsNone(returned)
        self.assertEqual(result['trials'][0]['reason'],'motion_prefilter')
        geometry.assert_not_called();proof.assert_not_called()

    def test_connector_lower_bound_is_below_checked_path_length(self):
        from test_verified_visit_improvement import Graph
        from femto.planning.scan_visit_order import anchored_objective,connect
        mesh=SimpleNamespace(vertices=np.zeros((3,3)))
        for reverse in (False,True):
            visits=[(0,False),(1,reverse)]
            original=connect(Graph(self.segments),self.segments,visits,lambda:None,1000)
            lengths=module._prior_link_lengths(self.segments,visits,original,lambda:None)
            trimmed=module._apply(self.segments,[self.item],[self.item])
            candidate=connect(Graph(trimmed),trimmed,visits,lambda:None,1000)
            lower=module._connector_lower_bound(self.segments,trimmed,visits,original,lengths,mesh,lambda:None)
            self.assertLessEqual(lower,anchored_objective(trimmed,visits,candidate,budget=1000)[1])
        mesh.vertices[0,0]=11.
        self.assertIsNone(module._connector_lower_bound(self.segments,trimmed,visits,original,lengths,mesh,lambda:None))

    def test_connector_lower_bound_on_multisegment_paths(self):
        from test_verified_visit_improvement import Graph
        from femto.planning.scan_visit_order import anchored_objective,connect
        rng=np.random.default_rng(7123);mesh=SimpleNamespace(vertices=np.zeros((3,3)))
        for _ in range(30):
            segments=deepcopy(self.segments)
            for segment in segments:
                for point in segment.points:point['xyz_m']=rng.uniform(-2.,2.,3).tolist()
            visits=[(0,False),(1,bool(rng.integers(0,2)))]
            original=connect(Graph(segments),segments,visits,lambda:None,1000)
            lengths=module._prior_link_lengths(segments,visits,original,lambda:None)
            trimmed=module._apply(segments,[self.item],[self.item])
            candidate=connect(Graph(trimmed),trimmed,visits,lambda:None,1000)
            middle=dict(candidate[0]['samples'][0],xyz_m=rng.uniform(-2.,2.,3).tolist())
            candidate[0]['samples'].insert(0,middle);candidate[0]['added_edges']+=1
            lower=module._connector_lower_bound(segments,trimmed,visits,original,lengths,mesh,lambda:None)
            self.assertLessEqual(lower,anchored_objective(trimmed,visits,candidate,budget=1000)[1])

    def test_connector_bound_rejects_before_graph_construction(self):
        density={'proximity':[{'radius_m':.001,'near_fraction':.2},{'radius_m':.002,'near_fraction':.3}]}
        stats={'connected_detours':2,'connector_length_m':1.,'revisit_connector_length_m':.1}
        params={'quality_evidence':{},'effective_domain':{},'effective_width_mm':5.}
        base=SimpleNamespace(planner={'parameters':params},statistics=stats)
        domain=SimpleNamespace(binding={},revision={'sha256':'revision'},areas=[1.],ledger=SimpleNamespace(arrays={'region_faces':[0]}))
        state=dict(domain=domain,base=base,binding={},segments=self.segments,visits=[(0,False),(1,False)],
                   links=[{'connected':False}],mesh=None,prior=base,prior_stats=stats,footprint={})
        with ExitStack() as stack:
            def mock(obj,name,**kwargs):return stack.enter_context(patch.object(obj,name,**kwargs))
            mock(module,'_prepare',return_value=state);mock(module,'_apply',return_value=self.segments)
            mock(module.row_proximity_research,'measure',side_effect=[density,density])
            mock(module.terminal_overlap_research,'propose',return_value=([],{'selected':[self.item]}))
            mock(module.effective_domain,'validate',return_value=None)
            mock(module,'_connector_lower_bound',return_value=1.01)
            graph=mock(module.continuous_quality,'LazyVerifiedSurfaceGraph')
            geometry=mock(module,'_candidate');proof=mock(module.coverage,'evaluate');mock(module,'_check_end')
            result,returned=module.build(None,{}, {},object(),[(0,False),(1,False)],[{'connected':False}],object())
        self.assertFalse(result['adopted']);self.assertIsNone(returned)
        self.assertEqual(result['trials'][0]['reason'],'connector_lower_bound')
        graph.assert_not_called();geometry.assert_not_called();proof.assert_not_called()

    def test_input_snapshot_detects_changed_links_before_acceptance(self):
        plan={'treatment_head':{}};visits=[[0,False]];links=[]
        snapshot=module.planning_revision.digest(dict(plan=plan,visits=visits,links=links))
        links.append({'connected':False})
        with patch.object(module.planning_revision,'require_current') as require:
            with self.assertRaisesRegex(ValueError,'连接发生变化'):
                module._check_end(None,{},plan,visits,links,None,(),snapshot,lambda:None)
            require.assert_not_called()

    def test_cancellation_stops_before_replay_or_candidate(self):
        def cancel():raise RuntimeError('cancelled')
        with patch.object(module,'_prepare') as prepare:
            with self.assertRaisesRegex(RuntimeError,'cancelled'):
                module.build(None,{}, {},object(),[],[],object(),check=cancel)
            prepare.assert_not_called()

    def test_replay_rejects_changed_visit_order(self):
        state={'visits':[(0,False),(1,False)]}
        with patch.object(module,'_prepare',return_value=state):
            with self.assertRaisesRegex(ValueError,'访问顺序'):
                module.replay(None,{}, {},object(),[(0,False),(1,False)],[],object(),
                              {'trim':[self.item],'visits':[[1,False],[0,False]],'links':[]},object())

    def test_pair_search_respects_budget_and_selects_verified_pair(self):
        before={'proximity':[{'radius_m':.001,'near_fraction':.2},{'radius_m':.002,'near_fraction':.3}]}
        after={'proximity':[{'radius_m':.001,'near_fraction':.1},{'radius_m':.002,'near_fraction':.2}]}
        oldstats={'connected_detours':2,'connector_length_m':1.,'revisit_connector_length_m':.1,'total_length_m':2.}
        params={'quality_evidence':{},'effective_domain':{},'effective_width_mm':5.}
        base=SimpleNamespace(planner={'parameters':params},statistics=oldstats)
        domain=SimpleNamespace(binding={},revision={'sha256':'revision'},areas=[1.],ledger=SimpleNamespace(arrays={'region_faces':[0]}))
        state=dict(domain=domain,base=base,binding={},segments=self.segments,visits=[(0,False),(1,False)],
                   links=[{'connected':False}],mesh=None,prior=base,prior_stats=oldstats,footprint={})
        selected=[self.item,{'base_segment':1,'side':'end'}];active=[];proof=SimpleNamespace(summary={})
        class FakeGraph:
            def __init__(self,*args):pass
        def applied(segments,all_selected,subset):
            active[:]=subset
            return segments
        def candidate(*args,**kwargs):
            stats=dict(connected_detours=2,connector_length_m=.9,revisit_connector_length_m=.09,
                       total_length_m=1.8 if len(active)==2 else 1.9)
            return base,stats,{'items_sha256':'x','connector_length_m':.9},after
        for budget,fallback in ((2,False),(3,False),(3,True)):
            with self.subTest(budget=budget,fallback=fallback),ExitStack() as stack:
                def mock(obj,name,**kwargs):return stack.enter_context(patch.object(obj,name,**kwargs))
                mock(module,'_prepare',return_value=state);mock(module,'_apply',side_effect=applied)
                mock(module,'replace',return_value=base)
                mock(module.row_proximity_research,'measure',side_effect=[before,before])
                mock(module.terminal_overlap_research,'propose',return_value=([],{'selected':selected}))
                mock(module.effective_domain,'validate',return_value=None);mock(module,'_connector_lower_bound',return_value=None)
                mock(module.continuous_quality,'LazyVerifiedSurfaceGraph',new=FakeGraph)
                mock(module.scan_visit_order,'connect',return_value=[{'connected':True}])
                mock(module.scan_visit_order,'anchored_objective',return_value=(0,.9))
                mock(module,'_candidate',side_effect=candidate);mock(module,'scan_endpoints',return_value=[])
                mock(module.coverage,'checked_page_binding',return_value=False);mock(module.coverage,'evaluate',return_value=proof)
                mock(module.coverage,'validate');mock(module.coverage_partition,'compare_union',return_value={'retained_subset':True})
                retained=mock(module.coverage_partition,'compare',side_effect=[{'retained_subset':False},{'retained_subset':True}]
                              if fallback else None,return_value={'retained_subset':True})
                mock(module,'_check_end')
                result,returned=module.build(None,{}, {},object(),[(0,False),(1,False)],[{'connected':False}],object(),max_candidates=budget)
            self.assertIs(returned,proof)
            self.assertEqual(len(result['trials']),budget)
            self.assertEqual(len(result['recipe']['trim']),1 if budget==2 or fallback else 2)
            self.assertEqual(retained.call_count,2 if fallback else 1)


if __name__=='__main__':unittest.main()
