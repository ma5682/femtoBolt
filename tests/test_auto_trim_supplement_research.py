"""Trim research rejects invalid input and never installs an unchecked result."""
from types import SimpleNamespace
import unittest
from contextlib import ExitStack
from unittest.mock import patch
from femto.planning import auto_trim_supplement_research as module


class AutoTrimResearchTests(unittest.TestCase):
    def setUp(self):
        params={'quality_evidence':{},'effective_domain':{}}
        self.plan={'surface_trajectory':{'planner':{'parameters':params}},'planning_context':{'revision':{}},'treatment_head':{}}
        self.recipe={'edges':[],'visits':[],'links':[]}

    def test_invalid_tolerance_and_prior_proof_fail_before_trim(self):
        for value in (-1,.101,True,float('nan')):
            with self.subTest(value=value),self.assertRaises(ValueError):
                module.propose(None,{},self.plan,object(),self.plan,self.recipe,[],object(),max_connector_increase_m=value)
        with patch.object(module.supplement_coverage_research,'_replay_with_state',side_effect=ValueError('bad proof')), \
             patch.object(module.terminal_overlap_research,'propose') as trim:
            with self.assertRaisesRegex(ValueError,'bad proof'):
                module.propose(None,{},self.plan,object(),self.plan,self.recipe,[],object())
            trim.assert_not_called()

    def test_no_trim_preserves_original_and_rechecks_assets(self):
        domain=SimpleNamespace(ledger=SimpleNamespace(arrays={'region_faces':[0]}))
        state={'base':object(),'domain':domain,'old_domain':domain,'items':[],'original':[],'extra':[]}
        with patch.object(module.supplement_coverage_research,'_replay_with_state',return_value=({},state)), \
             patch.object(module,'replace',return_value=object()), \
             patch.object(module.row_proximity_research,'measure',return_value={'proximity':[]}), \
             patch.object(module.terminal_overlap_research,'propose',return_value=([],{'selected':[]})), \
             patch.object(module.planning_revision,'require_current'), \
             patch.object(module.target_surface,'validate') as target_check, \
             patch.object(module.coverage,'_checked_assets'):
            result,proof=module.propose(None,{},self.plan,object(),self.plan,self.recipe,[],object())
        self.assertFalse(result['adopted']);self.assertEqual(result['reason'],'no_candidates')
        self.assertIsNone(proof);self.assertEqual(target_check.call_count,2)

    def test_admission_requires_both_proximity_and_motion_constraints(self):
        before={'proximity':[{'radius_m':.001,'near_fraction':.2},{'radius_m':.002,'near_fraction':.3}]}
        after={'proximity':[{'radius_m':.001,'near_fraction':.1},{'radius_m':.002,'near_fraction':.25}]}
        old={'connected_detours':2,'connector_length_m':1.,'revisit_connector_length_m':.1}
        new={'connected_detours':2,'connector_length_m':1.01,'revisit_connector_length_m':.09}
        self.assertEqual(module._assess(before,after,old,new,.02),(True,True))
        self.assertEqual(module._assess(before,after,old,new,0.),(True,False))
        new['connected_detours']=3
        self.assertEqual(module._assess(before,after,old,new,.02),(True,False))
        new['connected_detours']=2;after['proximity'][1]['near_fraction']=.31
        self.assertEqual(module._assess(before,after,old,new,.02),(False,True))
        after['proximity'].pop()
        with self.assertRaisesRegex(ValueError,'配置不一致'):module._assess(before,after,old,new,.02)

    def test_unconfirmed_retention_does_not_return_candidate_or_proof(self):
        params={'quality_evidence':{},'effective_domain':{},'effective_width_mm':5.,'row_spacing_mm':4.}
        base=SimpleNamespace(planner={'parameters':params})
        original=SimpleNamespace(block_index=0)
        domain=SimpleNamespace(binding={},revision={'sha256':'revision'},areas=[1.],
                               ledger=SimpleNamespace(arrays={'region_faces':[0]}))
        state={'base':base,'domain':domain,'old_domain':domain,'items':[],
               'original':[original],'extra':[],'current_model':{'effective_width_mm':5.}}
        stats={'connected_detours':0,'connector_length_m':1.,'revisit_connector_length_m':.1}
        prior={'statistics':stats}
        proof=SimpleNamespace(summary={})
        before={'proximity':[{'radius_m':.001,'near_fraction':.2},{'radius_m':.002,'near_fraction':.3}]}
        after={'proximity':[{'radius_m':.001,'near_fraction':.1},{'radius_m':.002,'near_fraction':.2}]}
        class FakeGraph:
            def __init__(self,*args):pass
        with ExitStack() as stack:
            def mock(obj,name,**kwargs):return stack.enter_context(patch.object(obj,name,**kwargs))
            mock(module.supplement_coverage_research,'_replay_with_state',return_value=(prior,state))
            mock(module,'replace',return_value=base)
            mock(module.row_proximity_research,'measure',side_effect=[before,after])
            mock(module.terminal_overlap_research,'propose',return_value=([original],{'selected':[1]}))
            mock(module.surface_workspace,'canonical_region_mesh',return_value=None)
            mock(module.effective_domain,'validate',return_value=None)
            mock(module.continuous_quality,'LazyVerifiedSurfaceGraph',new=FakeGraph)
            mock(module.auto_supplement_route_research,'_source_visits',return_value=[(0,False)])
            mock(module.supplement_insertion_research,'propose',return_value=([],[],0,{'visits':[[0,False]]}))
            mock(module.balanced_visit_improvement,'_metrics',return_value={'detours':0,'revisit_connector_m':.1})
            mock(module.supplement_two_opt_research,'propose',return_value=([],[],0,{'visits':[[0,False]]}))
            mock(module.supplement_route_validation,'reconstruct_and_validate',return_value=([],stats,{}, {'items_sha256':'x'}))
            mock(module.fraction_transfer,'transfer',return_value=([],[],{}))
            mock(module,'scan_endpoints',return_value=[])
            mock(module.coverage_workflow,'route_binding',return_value={})
            mock(module.coverage,'evaluate',return_value=proof)
            mock(module.coverage,'validate')
            mock(module.coverage_partition,'compare',return_value={'retained_subset':False})
            mock(module.coverage_partition,'compare_union',return_value={'retained_subset':True})
            mock(module.planning_revision,'require_current')
            mock(module.target_surface,'validate')
            mock(module.coverage,'_checked_assets')
            model=SimpleNamespace(store=SimpleNamespace(mesh=None))
            result,returned=module.propose(model,{},self.plan,proof,self.plan,self.recipe,[],proof)
        self.assertFalse(result['adopted']);self.assertEqual(result['reason'],'proof_retention_not_confirmed')
        self.assertNotIn('recipe',result);self.assertIsNone(returned)


if __name__=='__main__':unittest.main()
