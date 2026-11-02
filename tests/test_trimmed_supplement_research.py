"""Trim replay cannot bypass the prior route or input budgets."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from femto.planning import trimmed_supplement_research as module


class TrimmedSupplementReplayTests(unittest.TestCase):
    def setUp(self):
        self.recipe={'edges':[[0,0]],'visits':[[0,False]],'links':[]}

    def call(self,prior=None,after=None,**options):
        return module.replay(None,{}, {},object(),{},prior or self.recipe,[],object(),after or self.recipe,[],object(),**options)

    def test_changed_source_and_excess_budget_fail_before_replay(self):
        changed=deepcopy(self.recipe);changed['edges']=[[0,1]]
        with patch.object(module.supplement_coverage_research,'_replay_with_state') as replay:
            with self.assertRaisesRegex(ValueError,'补线来源不一致'):self.call(after=changed)
            with self.assertRaisesRegex(ValueError,'参数无效或超过预算'):self.call(budget=200_001)
            bad=deepcopy(self.recipe);bad['links']=[{'samples':[{},{}]}]
            with self.assertRaisesRegex(ValueError,'连接点超过预算'):self.call(after=bad,budget=1)
            replay.assert_not_called()

    def test_invalid_prior_proof_stops_before_trim(self):
        with patch.object(module.supplement_coverage_research,'_replay_with_state',side_effect=ValueError('invalid prior proof')), \
             patch.object(module.terminal_overlap_research,'propose') as propose:
            with self.assertRaisesRegex(ValueError,'invalid prior proof'):self.call()
            propose.assert_not_called()

    def test_changed_target_asset_is_rejected_after_local_reuse(self):
        item=SimpleNamespace(to_dict=lambda:{'kind':'scan_segment'})
        base=SimpleNamespace(planner={'parameters':{}},ordered_items=[item])
        domain=SimpleNamespace(ledger=SimpleNamespace(arrays={'region_faces':[]}),revision={'sha256':'r'},areas=[])
        prior={'geometry':{'items_sha256':module.planning_revision.digest([item.to_dict()])}}
        state=dict(domain=domain,current_model={'effective_width_mm':5},base=base,original=[],extra=[],items=[item])
        base_plan={'surface_trajectory':{'planner':{'parameters':{'quality_evidence':{},'effective_domain':{}}}},
                   'planning_context':{'revision':{}},'treatment_head':{}}
        old_plan={'planning_context':{'revision':{}},'treatment_head':{}}
        model=SimpleNamespace(store=SimpleNamespace(mesh=None))
        with patch.object(module.supplement_coverage_research,'_replay_with_state',return_value=(prior,state)), \
             patch.object(module,'replace',return_value=base), \
             patch.object(module.terminal_overlap_research,'propose',return_value=([],{})), \
             patch.object(module.surface_workspace,'canonical_region_mesh',return_value=None), \
             patch.object(module.supplement_route_validation,'reconstruct_and_validate',return_value=([item],{}, {}, {})), \
             patch.object(module.scan_fraction_seeds,'propose',return_value=([],{})), \
             patch.object(module,'scan_endpoints',return_value=[]), \
             patch.object(module.coverage_workflow,'route_binding',return_value={}), \
             patch.object(module.coverage,'validate'), \
             patch.object(module.coverage_partition,'compare_union',return_value={}), \
             patch.object(module.coverage_partition,'compare',return_value={}), \
             patch.object(module.planning_revision,'require_current'), \
             patch.object(module.target_surface,'validate',side_effect=ValueError('target changed')):
            with self.assertRaisesRegex(ValueError,'target changed'):
                module.replay(model,{},old_plan,object(),base_plan,self.recipe,[],object(),self.recipe,[],object())


if __name__=='__main__':unittest.main()
