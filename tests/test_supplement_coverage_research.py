"""Replay must fail closed before exposing a research route or proof."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from femto.planning import supplement_coverage_research as module,footprint_model


class SupplementCoverageResearchTests(unittest.TestCase):
    def setUp(self):
        self.old_plan={'surface_trajectory':{},'treatment_head':{},'planning_context':{'revision':{}}}
        self.base_plan={'surface_trajectory':{'base':True},'treatment_head':{},'planning_context':{'revision':{}}}
        self.old=SimpleNamespace();self.base=SimpleNamespace(gap_report={'reconnection':{'base_items':[
            {'kind':'scan_segment','points':[{'row_index':0}]}]}})
        self.domain=SimpleNamespace(ledger=SimpleNamespace(arrays={'region_faces':[0]}),
                                    revision={'sha256':'revision'},areas=[])
        self.old_model=footprint_model.nominal(5.)

    def invoke(self):
        return module.replay(None,{},self.old_plan,object(),self.base_plan,[],[],[],[],object())

    def test_public_replay_does_not_expose_local_state(self):
        with patch.object(module,'_replay_with_state',return_value=({'status':'checked'},{'private':'data'})):
            self.assertEqual(self.invoke(),{'status':'checked'})

    def test_unbounded_or_malformed_recipes_are_rejected_before_proofs(self):
        with self.assertRaisesRegex(ValueError,'参数无效或超过预算'):
            module.replay(None,{},self.old_plan,object(),self.base_plan,[],[],[],[],object(),budget=200_001)
        with self.assertRaisesRegex(ValueError,'连接点超过预算'):
            module.replay(None,{},self.old_plan,object(),self.base_plan,[],[],[{'samples':[{},{}]}],[],object(),budget=1)
        with self.assertRaisesRegex(ValueError,'连接记录无效'):
            module.replay(None,{},self.old_plan,object(),self.base_plan,[],[],[{}],[],object())

    def test_old_proof_and_target_mismatch_are_rejected_before_route(self):
        def trajectory(value,check):return self.base if value.get('base') else self.old
        with patch.object(module,'from_plan',return_value=(self.domain,[],self.old_model,{})), \
             patch.object(module.SurfaceTrajectory,'from_dict',side_effect=trajectory), \
             patch.object(module.coverage_workflow,'route_binding',return_value={}), \
             patch.object(module.supplement_route_validation,'reconstruct_and_validate') as reconstruct:
            with patch.object(module.coverage,'validate',side_effect=ValueError('old proof invalid')):
                with self.assertRaisesRegex(ValueError,'old proof invalid'):self.invoke()
            reconstruct.assert_not_called()
            with patch.object(module.coverage,'validate'),patch.object(module,'_same_target',side_effect=ValueError('different target')):
                with self.assertRaisesRegex(ValueError,'different target'):self.invoke()
            reconstruct.assert_not_called()
            def domains(model,region,plan,check):
                return (self.domain,[],footprint_model.nominal(6.) if plan is self.base_plan else self.old_model,{})
            with patch.object(module.coverage,'validate'),patch.object(module,'_same_target'),patch.object(module,'from_plan',side_effect=domains):
                with self.assertRaisesRegex(ValueError,'相同固定名义作用宽度'):self.invoke()
            reconstruct.assert_not_called()

    def test_mutated_recipe_cannot_be_returned_after_proof_replay(self):
        edge_indices=[[0,0]];old_proof=object();final_proof=object()
        source_segment=SimpleNamespace(points=[{'row_index':0}],block_index=0)
        item=SimpleNamespace(to_dict=lambda:{'kind':'scan_segment'})
        self.domain.revision={'sha256':'revision'}
        self.base.planner={'parameters':{}}
        calls=[]
        def validate(*args,**kwargs):
            calls.append(args[0])
            if args[0] is final_proof:edge_indices.append([0,1])
        def trajectory(value,check):return self.base if value.get('base') else self.old
        with patch.object(module,'from_plan',return_value=(self.domain,[],self.old_model,{})), \
             patch.object(module.SurfaceTrajectory,'from_dict',side_effect=trajectory), \
             patch.object(module.coverage,'validate',side_effect=validate), \
             patch.object(module.coverage_workflow,'route_binding',return_value={}), \
             patch.object(module,'_same_target'), \
             patch.object(module.ScanSegment,'from_dict',return_value=source_segment), \
             patch.object(module.supplement_scan_segments,'materialize',return_value=([],{})), \
             patch.object(module.surface_workspace,'canonical_region_mesh',return_value=None), \
             patch.object(module.supplement_route_validation,'reconstruct_and_validate',return_value=([item],{}, {}, {})), \
             patch.object(module,'replace',return_value=self.base), \
             patch.object(module.scan_fraction_seeds,'propose',return_value=([],{})), \
             patch.object(module,'scan_endpoints',return_value=[]), \
             patch.object(module.coverage_partition,'compare_union',return_value={}), \
             patch.object(module.planning_revision,'require_current'):
            model=SimpleNamespace(store=SimpleNamespace(mesh=None))
            with self.assertRaisesRegex(ValueError,'输入路线发生变化'):
                module.replay(model,{},self.old_plan,old_proof,self.base_plan,edge_indices,[],[],[],final_proof)
        self.assertEqual(calls,[old_proof,final_proof])

    def test_target_assets_are_rechecked_after_final_proof(self):
        params={'quality_evidence':{},'effective_domain':{}}
        self.old_plan['surface_trajectory']['planner']={'parameters':params}
        self.base_plan['surface_trajectory']['planner']={'parameters':params}
        source=SimpleNamespace(points=[{'row_index':0}],block_index=0)
        item=SimpleNamespace(to_dict=lambda:{'kind':'scan_segment'})
        self.base.planner={'parameters':{}}
        def trajectory(value,check):return self.base if value.get('base') else self.old
        with patch.object(module,'from_plan',return_value=(self.domain,[],self.old_model,{})), \
             patch.object(module.SurfaceTrajectory,'from_dict',side_effect=trajectory), \
             patch.object(module.coverage,'validate'), \
             patch.object(module.coverage_workflow,'route_binding',return_value={}), \
             patch.object(module,'_same_target'), \
             patch.object(module.ScanSegment,'from_dict',return_value=source), \
             patch.object(module.supplement_scan_segments,'materialize',return_value=([],{})), \
             patch.object(module.surface_workspace,'canonical_region_mesh',return_value=None), \
             patch.object(module.supplement_route_validation,'reconstruct_and_validate',return_value=([item],{}, {}, {})), \
             patch.object(module,'replace',return_value=self.base), \
             patch.object(module.scan_fraction_seeds,'propose',return_value=([],{})), \
             patch.object(module,'scan_endpoints',return_value=[]), \
             patch.object(module.coverage_partition,'compare_union',return_value={}), \
             patch.object(module.planning_revision,'require_current'), \
             patch.object(module.target_surface,'validate',side_effect=[None,ValueError('modified target')]) as target_check:
            model=SimpleNamespace(store=SimpleNamespace(mesh=None))
            with self.assertRaisesRegex(ValueError,'modified target'):
                module.replay(model,{},self.old_plan,object(),self.base_plan,[],[],[],[],object())
        self.assertEqual(target_check.call_count,2)
        self.assertTrue(all(call.kwargs['reproduce'] is False for call in target_check.call_args_list))


if __name__=='__main__':unittest.main()
