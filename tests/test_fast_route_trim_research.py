"""Same-call reuse is limited to freshly verified route and proof state."""
from contextlib import ExitStack
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from femto.planning import fast_route_trim_research as module


class FastRouteTrimResearchTests(unittest.TestCase):
    def test_invalid_budgets_or_cancellation_stop_before_source_read(self):
        for args in (dict(relocation_attempts=0),dict(relocation_attempts=True),dict(max_trim_candidates=0),dict(budget=0)):
            with self.subTest(args=args),self.assertRaises(ValueError):
                module.build(None,{}, {},object(),**args)
        def cancel():raise RuntimeError('cancelled')
        with patch.object(module,'from_plan') as source:
            with self.assertRaisesRegex(RuntimeError,'cancelled'):module.build(None,{}, {},object(),check=cancel)
            source.assert_not_called()

    def test_only_reuses_state_after_both_proofs_and_retention(self):
        stats={'connected_detours':1,'connector_length_m':.9,'revisit_connector_length_m':.1}
        params={'effective_width_mm':5.,'quality_evidence':{},'effective_domain':{},'row_spacing_mm':4.}
        base=SimpleNamespace(planner={'parameters':params},statistics=stats,
            gap_report={'reconnection':{'base_items':[{'kind':'scan_segment'}],'connections':[]}})
        domain=SimpleNamespace(binding={},revision={'sha256':'revision'},areas=[1.],ledger=SimpleNamespace(arrays={'region_faces':[0]}))
        footprint={'schema':'femto-nominal-fixed-ball-v1','effective_width_mm':5.}
        prior=SimpleNamespace(summary={});final=SimpleNamespace(summary={});old=object()
        plan={'surface_trajectory':{},'treatment_head':{}}
        class FakeGraph:
            def __init__(self,*args):pass
        for retained,adopted in ((False,False),(True,False),(True,True)):
            with self.subTest(retained=retained,adopted=adopted),ExitStack() as stack:
                def mock(obj,name,**kwargs):return stack.enter_context(patch.object(obj,name,**kwargs))
                mock(module,'from_plan',return_value=(domain,[],footprint,None))
                mock(module.SurfaceTrajectory,'from_dict',return_value=base)
                mock(module.coverage_workflow,'route_binding',return_value={})
                verified=mock(module.coverage,'validate')
                mock(module.ScanSegment,'from_dict',return_value=object())
                mock(module.auto_supplement_route_research,'_source_visits',return_value=[(0,False)])
                mock(module.surface_workspace,'canonical_region_mesh',return_value=None)
                mock(module.effective_domain,'validate',return_value=None)
                mock(module.continuous_quality,'LazyVerifiedSurfaceGraph',new=FakeGraph)
                mock(module,'replace',return_value=base)
                search=mock(module.verified_relocation_research,'propose',return_value=([],[],0,{'visits':[[0,False]],'adopted':True}))
                replay=mock(module.verified_relocation_research,'validate_record',return_value=([],0))
                mock(module.supplement_route_validation,'reconstruct_relocated_and_validate',return_value=([],stats,{}, {'items_sha256':'x'}))
                mock(module.fraction_footprint,'nominal',return_value={})
                mock(module,'scan_endpoints',return_value=[])
                mock(module.coverage,'checked_page_binding',return_value=False)
                mock(module.coverage,'evaluate',return_value=prior)
                mock(module.coverage,'_checked_assets')
                mock(module.coverage_partition,'compare_union',return_value={'retained_subset':retained})
                trim=mock(module.trim_subset_research,'_build_prepared',return_value=({'adopted':adopted},final if adopted else None))
                guard=mock(module.trim_subset_research,'_check_end')
                result,prior_out,final_out=module.build(SimpleNamespace(store=SimpleNamespace(mesh=None)),{},plan,old)
            self.assertEqual(verified.call_count,2)
            self.assertEqual(search.call_args.kwargs['attempts'],32)
            self.assertEqual(replay.call_args.kwargs['attempts'],32)
            self.assertEqual(trim.called,retained)
            if retained:
                self.assertIs(prior_out,prior)
                self.assertIs(final_out,final if adopted else None)
                self.assertEqual(trim.call_args.kwargs['max_candidates'],2)
                self.assertEqual(result['reason'],'trim_candidate_validated' if adopted else 'no_trim_candidate')
            else:
                self.assertIsNone(prior_out);self.assertIsNone(final_out)
                self.assertEqual(result['reason'],'relocation_retention_not_confirmed')
                self.assertNotIn('recipe',result['relocation'])
                guard.assert_called_once()
            self.assertFalse(result['execution_ready']);self.assertFalse(result['installed_route'])

    def test_mutated_source_during_combination_is_rejected(self):
        # The public builder checks the source digest after its internal phase;
        # exercise that rule through the retained-state helper's own guard.
        plan={'value':1};visits=[[0,False]];links=[]
        snapshot=module.planning_revision.digest(dict(plan=plan,visits=visits,links=links))
        plan['value']=2
        with self.assertRaisesRegex(ValueError,'来源在处理前发生变化'):
            module.trim_subset_research._build_prepared(None,{},plan,object(),visits,links,object(),{},snapshot,
                max_candidates=2,max_proofs=1,proof_budget=None,motion_prefilter=True,check=lambda:None,budget=1000)

    def test_invalid_relocation_record_stops_before_geometry_and_new_proof(self):
        stats={'connected_detours':1,'connector_length_m':.9,'revisit_connector_length_m':.1}
        params={'effective_width_mm':5.,'quality_evidence':{},'effective_domain':{},'row_spacing_mm':4.}
        base=SimpleNamespace(planner={'parameters':params},statistics=stats,
            gap_report={'reconnection':{'base_items':[{'kind':'scan_segment'}],'connections':[]}})
        domain=SimpleNamespace();footprint={'schema':'femto-nominal-fixed-ball-v1','effective_width_mm':5.}
        class FakeGraph:
            def __init__(self,*args):pass
        with ExitStack() as stack:
            def mock(obj,name,**kwargs):return stack.enter_context(patch.object(obj,name,**kwargs))
            mock(module,'from_plan',return_value=(domain,[],footprint,None))
            mock(module.SurfaceTrajectory,'from_dict',return_value=base)
            mock(module.coverage_workflow,'route_binding',return_value={});mock(module.coverage,'validate')
            mock(module.ScanSegment,'from_dict',return_value=object())
            mock(module.auto_supplement_route_research,'_source_visits',return_value=[(0,False)])
            mock(module.surface_workspace,'canonical_region_mesh',return_value=None)
            mock(module.effective_domain,'validate',return_value=None)
            mock(module.continuous_quality,'LazyVerifiedSurfaceGraph',new=FakeGraph);mock(module,'replace',return_value=base)
            mock(module.verified_relocation_research,'propose',return_value=([],[],0,{'visits':[[0,False]]}))
            mock(module.verified_relocation_research,'validate_record',side_effect=ValueError('bad record'))
            geometry=mock(module.supplement_route_validation,'reconstruct_relocated_and_validate')
            proof=mock(module.coverage,'evaluate')
            with self.assertRaisesRegex(ValueError,'bad record'):
                module.build(SimpleNamespace(store=SimpleNamespace(mesh=None)),{}, {'surface_trajectory':{}},object())
            geometry.assert_not_called();proof.assert_not_called()


if __name__=='__main__':unittest.main()
