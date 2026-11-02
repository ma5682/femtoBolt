"""Full-segment trim replay fails closed on invalid sources or decisions."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from femto.planning import full_trim_supplement_research as module,planning_revision


class FullTrimSupplementTests(unittest.TestCase):
    def setUp(self):
        self.previous={'edges':[[0,1]],'visits':[],'links':[]}
        self.trimmed={'edges':[[0,1]],'visits':[],'links':[],'trim_decision':{'selected':[]}}

    def invoke(self):
        return module.replay(None,{}, {},object(),{},self.previous,[],object(),self.trimmed,[],object())

    def test_recipe_must_preserve_sources_and_budget(self):
        self.trimmed['edges']=[[0,2]]
        with self.assertRaisesRegex(ValueError,'配方无效'):self.invoke()
        self.trimmed['edges']=[[0,1]];self.trimmed['links']=[{'samples':[{},{}]}]
        with self.assertRaisesRegex(ValueError,'连接点超过预算'):
            module.replay(None,{}, {},object(),{},self.previous,[],object(),self.trimmed,[],object(),budget=1)

    def test_invalid_prior_proof_stops_before_trim(self):
        with patch.object(module.supplement_coverage_research,'_replay_with_state',side_effect=ValueError('bad proof')), \
             patch.object(module.terminal_overlap_research,'propose') as trim:
            with self.assertRaisesRegex(ValueError,'bad proof'):self.invoke()
            trim.assert_not_called()

    def test_tampered_trim_decision_rejected_before_final_geometry(self):
        item=SimpleNamespace(to_dict=lambda:{'kind':'scan_segment'})
        domain=SimpleNamespace(ledger=SimpleNamespace(arrays={'region_faces':[0]}))
        state={'domain':domain,'base':object(),'items':[item],'original':[],'extra':[]}
        prior={'geometry':{'items_sha256':planning_revision.digest([item.to_dict()])}}
        with patch.object(module.supplement_coverage_research,'_replay_with_state',return_value=(prior,state)), \
             patch.object(module,'replace',return_value=object()), \
             patch.object(module.terminal_overlap_research,'propose',return_value=([],{'selected':['different']})), \
             patch.object(module.supplement_route_validation,'reconstruct_and_validate') as geometry:
            with self.assertRaisesRegex(ValueError,'决定无法重现'):self.invoke()
            geometry.assert_not_called()


if __name__=='__main__':unittest.main()
