"""Remaining supplement candidates must follow replay and recheck inputs."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from femto.planning import remaining_supplement_research as module


class RemainingSupplementResearchTests(unittest.TestCase):
    def setUp(self):
        params={'quality_evidence':{},'effective_domain':{}}
        self.old={'treatment_head':{},'planning_context':{'revision':{}},
                  'surface_trajectory':{'planner':{'parameters':params}}}
        self.base={'treatment_head':{},'planning_context':{'revision':{}},
                   'surface_trajectory':{'planner':{'parameters':params}}}
        domain=SimpleNamespace(ledger=SimpleNamespace(arrays={'region_faces':[0]}),areas=[1.],representative_faces=[0])
        self.state={'old_domain':domain,'domain':domain,'old_trajectory':object()}
        self.report={'comparison':{'pending_old_leaf_indices':[2]}}

    def test_replay_precedes_selection_and_combines_original_edges(self):
        with patch.object(module.supplement_coverage_research,'_replay_with_state',return_value=(self.report,self.state)), \
             patch.object(module.witness_scan_supplement,'propose',return_value=([(1,3),(2,4)],{'pending_leaves':1})), \
             patch.object(module.planning_revision,'require_current'),patch.object(module.target_surface,'validate'), \
             patch.object(module.coverage,'_checked_assets'):
            result=module.propose(None,{},self.old,object(),self.base,[[1,3]],[],[],[],object())
        self.assertEqual(result['added_edges'],[[1,3],[2,4]])
        self.assertEqual(result['combined_edges'],[[1,3],[2,4]])
        self.assertFalse(result['installed_route'])

    def test_bad_replay_and_mutation_cannot_return_candidates(self):
        edges=[[1,3]]
        with patch.object(module.supplement_coverage_research,'_replay_with_state',side_effect=ValueError('bad proof')), \
             patch.object(module.witness_scan_supplement,'propose') as selection:
            with self.assertRaisesRegex(ValueError,'bad proof'):
                module.propose(None,{},self.old,object(),self.base,edges,[],[],[],object())
            selection.assert_not_called()
        def mutate(*args,**kwargs):
            edges.append([2,4]);return [],{}
        with patch.object(module.supplement_coverage_research,'_replay_with_state',return_value=(self.report,self.state)), \
             patch.object(module.witness_scan_supplement,'propose',side_effect=mutate):
            with self.assertRaisesRegex(ValueError,'输入发生变化'):
                module.propose(None,{},self.old,object(),self.base,edges,[],[],[],object())


if __name__=='__main__':unittest.main()
