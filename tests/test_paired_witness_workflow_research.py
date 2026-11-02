from contextlib import ExitStack
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from femto.planning import paired_witness_workflow_research as paired


class PairedWitnessWorkflowTests(unittest.TestCase):
    def test_invalid_budgets_reject_before_loading_inputs(self):
        for kwargs in ({'max_pairs':0},{'max_pairs':True},{'max_proofs':5},{'max_proofs':0}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                paired.build(None,None,None,None,None,None,None,**kwargs)

    def test_pair_ranking_excludes_conflicts_and_prefers_more_pending_cells(self):
        def value(seed,segment,leaves,distance):
            return dict(seed=seed,segment=segment,point=1,pending_leaves=leaves,distance_m=distance)
        values=[value(0,0,[1],.001),value(1,1,[2,3],.002),value(2,0,[4],.0001)]
        pairs=paired._pairs(values)
        self.assertEqual([[v['seed'] for v in pair] for pair in pairs],[[1,2],[0,1]])

    def exercise(self,*,retained=True,geometry_fails=False,max_proofs=2,refine=False):
        plan={'surface_trajectory':{},'planning_context':{},'treatment_head':{}}
        proof=SimpleNamespace(metadata={'sha256':'proof'},summary={})
        pending={'retained_subset':False,'pending_old_leaf_indices':[1,2]}
        domain=SimpleNamespace(binding={},areas=[],ledger=SimpleNamespace(arrays={'region_faces':[]}),representative_faces=[])
        state=dict(domain=domain,old_domain=domain,old='old',base='base',segments=['source'],old_density={},
                   nominal='nominal',previous=deepcopy(pending))
        values=[dict(seed=i,segment=i,point=1,pending_leaves=[i+1],distance_m=.001) for i in range(2)]
        with ExitStack() as stack:
            def use(obj,name,**kwargs):return stack.enter_context(patch.object(obj,name,**kwargs))
            use(paired.bend,'_snapshot',return_value='snapshot');use(paired.bend,'_prepare',return_value=state)
            use(paired.bend,'_finish')
            use(paired.fraction_priority,'propose_batched',return_value=([(0,0,1,2)] if refine else [],[],{}))
            use(paired.proposals,'propose',return_value=(values,{}))
            use(paired.proposals,'combine_replacements',return_value={'members':values,'segments':[]})
            candidate=use(paired.bend,'_candidate',side_effect=ValueError('invalid geometry') if geometry_fails else None,
                          return_value=('final',{'total_length_m':1.},{'items_sha256':'items'},{}))
            use(paired.bend,'_acceptable',return_value=True);use(paired.screening,'_acceptable',return_value=True)
            use(paired.bend,'_proof_inputs',return_value=([],'seeds','route',{}))
            use(paired.scan_fraction_seeds,'propose',return_value=([],{}));use(paired,'scan_endpoints',return_value=[])
            use(paired.coverage_workflow,'route_binding',return_value={})
            evaluate=use(paired.coverage,'evaluate',return_value=proof)
            validate=use(paired.coverage,'validate')
            use(paired.coverage_partition,'compare_union',return_value={'retained_subset':retained,'pending_old_leaf_indices':[] if retained else [1]})
            recipe=use(paired.direct,'pair_recipe',return_value={'version':'pair'})
            report,result=paired.build(None,{},plan,proof,plan,[],proof,max_proofs=max_proofs)
            return report,result,evaluate.call_count,validate.call_count,recipe.call_count,candidate.call_count

    def test_combined_geometry_failure_and_unretained_proof_cannot_be_accepted(self):
        report,result,evaluated,validated,recipes,_=self.exercise(geometry_fails=True)
        self.assertFalse(report['adopted']);self.assertIsNone(result)
        self.assertEqual((evaluated,validated,recipes),(0,0,0))
        report,result,evaluated,validated,recipes,_=self.exercise(retained=False,max_proofs=1)
        self.assertFalse(report['adopted']);self.assertIsNone(result)
        self.assertEqual((evaluated,validated,recipes),(1,1,0))

    def test_success_requires_final_replay_and_only_position_refinement_can_keep_route(self):
        report,result,evaluated,validated,recipes,candidates=self.exercise()
        self.assertTrue(report['adopted']);self.assertFalse(report['route_unchanged']);self.assertIsNotNone(result)
        self.assertEqual((evaluated,validated,recipes,candidates),(1,1,1,1))
        self.assertFalse(report['installed_route']);self.assertFalse(report['execution_ready'])
        report,result,evaluated,validated,recipes,candidates=self.exercise(refine=True)
        self.assertTrue(report['adopted']);self.assertTrue(report['route_unchanged'])
        self.assertEqual((evaluated,validated,recipes,candidates),(1,1,0,0))


if __name__=='__main__':unittest.main()
