from contextlib import ExitStack
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from femto.planning import ranked_point_witness_research as ranked


class RankedPointWitnessTests(unittest.TestCase):
    def test_budget_validation(self):
        for kwargs in ({'max_proofs':0},{'max_proofs':True},{'max_ranked_leaves':0},{'max_ranked_leaves':257}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):ranked.build(None,None,None,None,None,**kwargs)

    def test_retry_needs_fresh_final_proof_and_reproducible_recipe(self):
        params={'quality_evidence':{},'effective_domain':{}}
        plan={'surface_trajectory':{'planner':{'parameters':params}},'treatment_head':{},'planning_context':{'revision':{}}}
        ledger=SimpleNamespace(arrays={'region_faces':np.array([0])})
        domain=SimpleNamespace(ledger=ledger,representative_faces=np.array([0]),areas=np.array([1.]),binding={})
        width={'schema':'femto-nominal-fixed-ball-v1','effective_width_mm':10.}
        old=SimpleNamespace(statistics={});base=SimpleNamespace(statistics={},gap_report={'reconnection':{'base_items':[{'kind':'scan_segment'}],'connections':[]}})
        oldproof=SimpleNamespace(metadata={'sha256':'old'});proof=SimpleNamespace(metadata={'sha256':'final'},summary={})
        proposal={'seed':0,'segment':0,'point':1,'distance_m':.0001,'pending_leaves':[4],'segments':[]}
        model=SimpleNamespace(store=SimpleNamespace(mesh='mesh'))
        with ExitStack() as stack:
            def use(obj,name,**kwargs):return stack.enter_context(patch.object(obj,name,**kwargs))
            use(ranked,'from_plan',return_value=(domain,[],width,None))
            from_dict=use(ranked.SurfaceTrajectory,'from_dict',side_effect=[old,base])
            use(ranked.coverage_workflow,'route_binding',return_value={'route':1})
            validate=use(ranked.coverage,'validate');use(ranked,'_same_target')
            use(ranked,'_matched_witnesses',return_value=([],[],np.array([4])))
            fraction=use(ranked.fraction_priority,'propose_batched',side_effect=[([(0,0,1,2)],[],{}),([(0,0,1,4)],[],{})])
            use(ranked.witness_leaf_rank_research,'rank',return_value={'order':[4]})
            use(ranked.ScanSegment,'from_dict',return_value='segment')
            propose=use(ranked.witness_bend_research,'propose',return_value=([proposal],{}))
            use(ranked.auto_supplement_route_research,'_source_visits',return_value=[])
            use(ranked.fraction_footprint,'nominal',return_value='nominal')
            use(ranked.surface_workspace,'canonical_region_mesh',return_value='mesh')
            use(ranked.row_proximity_research,'measure',return_value={})
            candidate=use(ranked.bend,'_candidate',return_value=('final',{}, {'items_sha256':'x'},{}))
            use(ranked.bend,'_acceptable',return_value=True)
            use(ranked.bend,'_proof_inputs',return_value=([(0,0,1,2)],'seeds','route','transfer'))
            use(ranked.scan_fraction_seeds,'propose',return_value=([],{}));use(ranked,'scan_endpoints',return_value=[])
            evaluate=use(ranked.coverage,'evaluate',return_value=proof)
            compare=use(ranked.coverage_partition,'compare_union',side_effect=[
                {'retained_subset':False,'pending_old_leaf_indices':[4]},
                {'retained_subset':True,'pending_old_leaf_indices':[]}])
            recipe=use(ranked.direct,'point_recipe',return_value={'recipe':1})
            use(ranked.planning_revision,'require_current');use(ranked.target_surface,'validate');use(ranked.coverage,'_checked_assets')
            report,result=ranked.build(model,{},plan,oldproof,plan)
            self.assertTrue(report['adopted']);self.assertIs(result,proof)
            self.assertEqual(evaluate.call_count,2);self.assertEqual(validate.call_count,3)
            self.assertEqual(report['trials'][0]['reason'],'not_reconfirmed');self.assertEqual(report['trials'][1]['reason'],'accepted')
            self.assertEqual(fraction.call_args.kwargs['denominator'],16)
            self.assertFalse(propose.call_args.kwargs['edge_insert']);recipe.assert_called_once()
            from_dict.side_effect=[old,base];recipe.reset_mock()
            fraction.side_effect=[([(0,0,1,2)],[],{}),([(0,0,1,4)],[],{})]
            compare.side_effect=[{'retained_subset':False,'pending_old_leaf_indices':[4]}]
            rejected,result=ranked.build(model,{},plan,oldproof,plan,max_proofs=1)
            self.assertFalse(rejected['adopted']);self.assertEqual(rejected['reason'],'proof_budget_exhausted')
            self.assertIsNone(result);recipe.assert_not_called()
            from_dict.side_effect=[old,base];fraction.side_effect=[([(0,0,1,2)],[],{})]
            compare.side_effect=[{'retained_subset':True,'pending_old_leaf_indices':[]}]
            propose.side_effect=[([proposal],{}),([], {})]
            rejected,result=ranked.build(model,{},plan,oldproof,plan)
            self.assertFalse(rejected['adopted']);self.assertIsNone(result)
            self.assertEqual(rejected['trials'][0]['reason'],'recipe_not_rederivable');recipe.assert_not_called()
            from_dict.side_effect=[old,base];fraction.side_effect=[([(0,0,1,2)],[],{})]
            compare.side_effect=[{'retained_subset':False,'pending_old_leaf_indices':[]}]
            propose.side_effect=None;propose.return_value=([proposal,dict(proposal,seed=1)],{})
            candidate.reset_mock();evaluate.reset_mock()
            rejected,result=ranked.build(model,{},plan,oldproof,plan,max_proofs=1)
            self.assertFalse(rejected['adopted']);self.assertIsNone(result)
            self.assertEqual(rejected['reason'],'proof_budget_exhausted')
            self.assertEqual(candidate.call_count,1);self.assertEqual(evaluate.call_count,1)


if __name__=='__main__':unittest.main()
