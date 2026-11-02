from types import SimpleNamespace
from contextlib import ExitStack
import unittest
from unittest.mock import patch
import numpy as np
from femto.planning import single_final_witness_research as fast


class SingleFinalWitnessTests(unittest.TestCase):
    def test_invalid_budget_fails_before_accessing_source(self):
        for value in (0,5,True):
            with self.subTest(value=value),self.assertRaises(ValueError):
                fast.build(None,None,None,None,None,max_proofs=value)

    def test_final_proof_replayed_and_old_set_must_be_retained(self):
        params={'quality_evidence':{},'effective_domain':{}}
        plan={'surface_trajectory':{'planner':{'parameters':params}},'treatment_head':{},'planning_context':{'revision':{}}}
        ledger=SimpleNamespace(arrays={'region_faces':np.array([0])})
        domain=SimpleNamespace(ledger=ledger,representative_faces=np.array([0]),areas=np.array([1.]),binding={})
        width={'schema':'femto-nominal-fixed-ball-v1','effective_width_mm':10.}
        old=SimpleNamespace(statistics={});base=SimpleNamespace(statistics={},gap_report={'reconnection':{'base_items':[{'kind':'scan_segment'}],'connections':[]}})
        old_proof=SimpleNamespace(metadata={'sha256':'old'});final_proof=SimpleNamespace(metadata={'sha256':'final'},summary={})
        proposal={'seed':0,'segment':0,'point':1,'edge':0,'distance_m':.0003,'pending_leaves':[0],'segments':[]}
        model=SimpleNamespace(store=SimpleNamespace(mesh='mesh'))
        with ExitStack() as stack:
            def use(obj,name,**kwargs):return stack.enter_context(patch.object(obj,name,**kwargs))
            use(fast,'from_plan',return_value=(domain,[],width,None))
            from_dict=use(fast.SurfaceTrajectory,'from_dict',side_effect=[old,base])
            use(fast.coverage_workflow,'route_binding',return_value={'route':1})
            validate=use(fast.coverage,'validate');use(fast,'_same_target')
            use(fast,'_matched_witnesses',return_value=([],[],np.array([0])))
            use(fast.fraction_priority,'propose_batched',return_value=([(0,0,1,2)],[],{}))
            use(fast.ScanSegment,'from_dict',return_value='segment')
            propose=use(fast.witness_bend_research,'propose',return_value=([proposal],{}))
            use(fast.auto_supplement_route_research,'_source_visits',return_value=[])
            use(fast.fraction_footprint,'nominal',return_value='nominal')
            use(fast.surface_workspace,'canonical_region_mesh',return_value='mesh')
            use(fast.row_proximity_research,'measure',return_value={})
            candidate=use(fast.bend,'_candidate',return_value=('final',{}, {'items_sha256':'x'},{}))
            use(fast.bend,'_acceptable',return_value=True)
            use(fast.bend,'_proof_inputs',return_value=([(0,0,1,2)],'seeds','route','transfer'))
            evaluate=use(fast.coverage,'evaluate',return_value=final_proof)
            compare=use(fast.coverage_partition,'compare_union',return_value={'retained_subset':True})
            recipe=use(fast.direct,'recipe',return_value={'recipe':1})
            use(fast.planning_revision,'require_current');use(fast.target_surface,'validate');use(fast.coverage,'_checked_assets')
            report,proof=fast.build(model,{},plan,old_proof,plan)
            self.assertTrue(report['adopted']);self.assertIs(proof,final_proof);self.assertEqual(validate.call_count,2)
            self.assertIs(validate.call_args.kwargs['guide'],old_proof);self.assertFalse(report['execution_ready'])
            from_dict.side_effect=[old,base];compare.return_value={'retained_subset':False};recipe.reset_mock()
            rejected,result=fast.build(model,{},plan,old_proof,plan)
            self.assertFalse(rejected['adopted']);self.assertIsNone(result);recipe.assert_not_called()
            from_dict.side_effect=[old,base];compare.return_value={'retained_subset':True}
            propose.side_effect=[([proposal],{}),([], {})]
            rejected,result=fast.build(model,{},plan,old_proof,plan)
            self.assertFalse(rejected['adopted']);self.assertIsNone(result);recipe.assert_not_called()
            self.assertEqual(rejected['trials'][0]['reason'],'recipe_not_rederivable')
            from_dict.side_effect=[old,base];evaluate.reset_mock();propose.reset_mock()
            with patch.object(fast.witness_bend_research,'MAX_PENDING',0):
                rejected,result=fast.build(model,{},plan,old_proof,plan)
            self.assertFalse(rejected['adopted']);self.assertIsNone(result)
            self.assertEqual(rejected['reason'],'witness_budget_exhausted')
            evaluate.assert_not_called();propose.assert_not_called()
            from_dict.side_effect=[old,base];compare.return_value={'retained_subset':False}
            propose.side_effect=None;propose.return_value=([proposal,dict(proposal,seed=1)],{})
            candidate.reset_mock();evaluate.reset_mock()
            rejected,result=fast.build(model,{},plan,old_proof,plan,max_proofs=1)
            self.assertFalse(rejected['adopted']);self.assertIsNone(result)
            self.assertEqual(rejected['reason'],'proof_budget_exhausted')
            self.assertEqual(candidate.call_count,1);self.assertEqual(evaluate.call_count,1)


if __name__=='__main__':unittest.main()
