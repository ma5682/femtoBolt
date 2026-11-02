"""Policy and fail-closed checks for the bounded supplement trim search."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from femto.planning import supplement_trim_subset_research as module


def density(first,second):
    return {'proximity':[{'radius_m':.001,'near_fraction':first},
                         {'radius_m':.002,'near_fraction':second}]}


def stats(detours=3,connector=1.,revisit=.2):
    return dict(connected_detours=detours,connector_length_m=connector,
                revisit_connector_length_m=revisit)


class SupplementTrimSubsetResearchTests(unittest.TestCase):
    def test_default_policy_rejects_density_above_original(self):
        prepared=dict(prior_density=density(.03,.12),old_density=density(.01,.13),
                      report={'statistics':stats()},original=SimpleNamespace(statistics=stats()))
        value,flags=module._acceptable(prepared,stats(connector=.9,revisit=.1),density(.02,.11),True)
        self.assertFalse(value);self.assertFalse(flags['baseline_density_passed'])
        value,flags=module._acceptable(prepared,stats(connector=.9,revisit=.1),density(.02,.11),False)
        self.assertTrue(value);self.assertFalse(flags['baseline_density_passed'])
        value,_=module._acceptable(prepared,stats(connector=1.01),density(.02,.11),False)
        self.assertFalse(value)
        value,_=module._acceptable(prepared,stats(),density(.031,.11),False)
        self.assertFalse(value)

    def test_invalid_budget_and_cancel_stop_before_source_read(self):
        for args in ({'max_candidates':0},{'max_candidates':True},{'max_proofs':0},{'max_relocations':-1},{'max_relocations':True},
                     {'require_baseline_density':1},{'budget':0}):
            with self.subTest(args=args),self.assertRaises(ValueError):
                module.build(None,None,None,None,None,None,None,None,**args)
        def cancel():raise RuntimeError('cancelled')
        with self.assertRaisesRegex(RuntimeError,'cancelled'):
            module.build(None,None,None,None,None,None,None,None,check=cancel)

    def test_replay_rejects_unrecognized_recipe_before_source_read(self):
        with self.assertRaisesRegex(ValueError,'配方'):
            module.replay(None,None,None,None,None,None,None,None,{'trim':[]},None)

    def test_replay_rejects_changed_order_without_relocation_record(self):
        prepared={'visits':[(0,False),(1,False)],'segments':[], 'decision':{'selected':[]}}
        recipe={'trim':[{}],'visits':[[0,False],[1,True]],'links':[], 'baseline_links':[],
                'relocation':None,'requests':[]}
        with patch.object(module,'_prepare',return_value=prepared), \
             patch.object(module.trim_subset_research,'_apply',return_value=[]), \
             patch.object(module,'_candidate') as candidate:
            with self.assertRaisesRegex(ValueError,'无换序记录'):
                module.replay(None,{}, {},None,{}, {}, [],None,recipe,None)
            candidate.assert_not_called()


if __name__=='__main__':unittest.main()
