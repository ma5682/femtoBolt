"""Automatic route proposals must preserve recorded scan visits and validation."""
from types import SimpleNamespace
import unittest
from contextlib import ExitStack
from unittest.mock import patch

from femto.planning import auto_supplement_route_research as module
from femto.planning.surface_trajectory import ScanSegment


class AutoSupplementRouteResearchTests(unittest.TestCase):
    def test_adopted_deep_relocation_is_the_source_order(self):
        segments=[ScanSegment(i,i,[],0) for i in range(3)]
        prior=[[0,False],[2,True],[1,True]];deep=[[0,False],[1,True],[2,False]]
        ordering={'visits':[[0,False],[1,False],[2,False]],
                  'relocation':{'adopted':True,'visits':prior},
                  'relocation_deep':{'adopted':True,'visits':deep}}
        base=SimpleNamespace(gap_report={'reconnection':{'ordering':ordering}})
        self.assertEqual(module._source_visits(base,segments),[(0,False),(1,True),(2,False)])
        ordering['relocation_floor']={'adopted':True,'visits':[[0,False],[2,False],[1,False]]}
        self.assertEqual(module._source_visits(base,segments),[(0,False),(2,False),(1,False)])
        ordering['relocation_floor']['adopted']=False
        self.assertEqual(module._source_visits(base,segments),[(0,False),(1,True),(2,False)])
        ordering['relocation_unified']={'adopted':True,'visits':[[0,False],[2,False],[1,True]]}
        self.assertEqual(module._source_visits(base,segments),[(0,False),(2,False),(1,True)])
        ordering['relocation_unified']['adopted']=False
        self.assertEqual(module._source_visits(base,segments),[(0,False),(1,True),(2,False)])
        ordering['relocation_deep']['adopted']=False
        self.assertEqual(module._source_visits(base,segments),[(0,False),(2,True),(1,True)])

    def test_adopted_relocation_is_the_source_order(self):
        segments=[ScanSegment(i,i,[],0) for i in range(3)]
        ordering={'visits':[[0,False],[1,False],[2,False]],
                  'relocation':{'adopted':True,'visits':[[0,False],[2,True],[1,True]]}}
        base=SimpleNamespace(gap_report={'reconnection':{'ordering':ordering}})
        self.assertEqual(module._source_visits(base,segments),[(0,False),(2,True),(1,True)])
        ordering['relocation']['adopted']=False
        self.assertEqual(module._source_visits(base,segments),[(0,False),(1,False),(2,False)])

    def test_recorded_final_order_and_orientation_are_checked(self):
        segments=[ScanSegment(0,0,[],0),ScanSegment(0,1,[],1)]
        base=SimpleNamespace(gap_report={'reconnection':{'ordering':{
            'visits':[[0,False],[1,False]],
            'improvement':{'adopted':True,'visits':[[0,False],[1,True]]},
            'preserved_detour':{'adopted':False},
            'joint_reorder':{'adopted':True,'visits':[[0,False],[1,False]]}}}})
        self.assertEqual(module._source_visits(base,segments),[(0,False),(1,False)])
        base.gap_report['reconnection']['ordering']['joint_reorder']['visits']=[[1,True],[0,False]]
        with self.assertRaises(ValueError):module._source_visits(base,segments)

    def test_invalid_proof_stops_before_graph_or_route_generation(self):
        with patch.object(module.auto_fraction_supplement_research,'_propose_with_state',side_effect=ValueError('bad proof')), \
             patch.object(module.surface_workspace,'canonical_region_mesh') as mesh:
            with self.assertRaisesRegex(ValueError,'bad proof'):
                module.build(None,{}, {},object(),{},object())
            mesh.assert_not_called()

    def test_correction_rounds_are_strictly_bounded(self):
        for value in (-1,4,True,1.5):
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,'校正轮数'):
                module.build(None,{}, {},object(),{},object(),correction_rounds=value)

    def test_correction_rebuilds_and_reproves_without_claiming_unconfirmed_retention(self):
        params={'quality_evidence':{},'effective_domain':{},'effective_width_mm':5.,'row_spacing_mm':4.}
        plan={'treatment_head':{},'planning_context':{'revision':{}},
              'surface_trajectory':{'planner':{'parameters':params}}}
        original=SimpleNamespace(points=[{'row_index':0}],block_index=0)
        base=SimpleNamespace(gap_report={'reconnection':{'base_items':[{'kind':'scan_segment'}]}},
                             planner={'parameters':params})
        domain=SimpleNamespace(binding={},revision={'sha256':'revision'},areas=[1.],representative_faces=[0],
                               ledger=SimpleNamespace(arrays={'region_faces':[0]}))
        state={'old':object(),'base':base,'domain':domain,'old_domain':domain,'current_model':{'effective_width_mm':5.}}
        selection={'selected_edges':[[0,0]],'requests':[]}
        proof=SimpleNamespace(summary={})
        item=SimpleNamespace(to_dict=lambda:{'kind':'scan_segment'})
        class FakeGraph:
            def __init__(self,*args):pass
        with ExitStack() as stack:
            def mock(obj,name,**kwargs):return stack.enter_context(patch.object(obj,name,**kwargs))
            mock(module.auto_fraction_supplement_research,'_propose_with_state',return_value=(selection,proof,state))
            mock(module.ScanSegment,'from_dict',return_value=original)
            mock(module,'_source_visits',return_value=[(0,False)])
            mock(module.supplement_scan_segments,'materialize',return_value=([],{}))
            mock(module.surface_workspace,'canonical_region_mesh',return_value=None)
            mock(module.effective_domain,'validate',return_value=None)
            mock(module.continuous_quality,'LazyVerifiedSurfaceGraph',new=FakeGraph)
            mock(module,'replace',return_value=base)
            mock(module.supplement_insertion_research,'propose',return_value=([],[],0,{'visits':[[0,False]]}))
            mock(module.balanced_visit_improvement,'_metrics',return_value={'detours':0,'revisit_connector_m':0.})
            mock(module.supplement_two_opt_research,'propose',return_value=([],[],0,{'visits':[[0,False]]}))
            mock(module.supplement_route_validation,'reconstruct_and_validate',return_value=([item],{}, {}, {}))
            mock(module.fraction_transfer,'transfer',return_value=([],[],{}))
            mock(module,'scan_endpoints',return_value=[])
            mock(module.coverage_workflow,'route_binding',return_value={})
            mock(module.coverage,'evaluate',return_value=proof)
            mock(module.coverage,'validate')
            comparison=mock(module.coverage_partition,'compare_union',side_effect=[
                {'retained_subset':False,'pending_old_leaf_indices':[1]},
                {'retained_subset':True,'pending_old_leaf_indices':[]}])
            correction=mock(module.witness_scan_supplement,'propose',return_value=([(0,1)],{'selected_edges':1}))
            mock(module.planning_revision,'require_current')
            mock(module.target_surface,'validate')
            mock(module.coverage,'_checked_assets')
            model=SimpleNamespace(store=SimpleNamespace(mesh=None))
            result,_=module.build(model,{},plan,proof,plan,proof,correction_rounds=1)
            self.assertEqual(comparison.call_count,2)
            self.assertEqual(correction.call_count,1)
            comparison.reset_mock(side_effect=True)
            comparison.return_value={'retained_subset':False,'pending_old_leaf_indices':[1]}
            correction.reset_mock()
            incomplete,_=module.build(model,{},plan,proof,plan,proof,correction_rounds=0)
            correction.assert_not_called()
        self.assertEqual(result['recipe']['edges'],[[0,0],[0,1]])
        self.assertTrue(result['comparison']['retained_subset'])
        self.assertFalse(result['installed_route'])
        self.assertFalse(incomplete['comparison']['retained_subset'])
        self.assertEqual(incomplete['comparison']['pending_old_leaf_indices'],[1])


if __name__=='__main__':unittest.main()
