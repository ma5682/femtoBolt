from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from femto.planning import auto_row_shift_screen_research as screen


class AutoRowShiftScreenTests(unittest.TestCase):
    def setUp(self):
        self.stats=dict(connected_detours=2,connector_length_m=.1,revisit_connector_length_m=.02,
                        total_length_m=.3,gap_count=1,inter_block_gap_count=1)
        self.density=dict(proximity=[dict(radius_m=.001,near_fraction=.1,near_sample_m=.01),
                                     dict(radius_m=.002,near_fraction=.2,near_sample_m=.02)])
        self.old=SimpleNamespace(statistics=self.stats,to_dict=lambda:dict(statistics=self.stats))

    def test_does_not_adopt_equal_worse_or_disconnected_routes(self):
        current=deepcopy(self.stats);density=deepcopy(self.density)
        self.assertFalse(screen._acceptable(self.old,SimpleNamespace(statistics=current),self.density,density))
        density['proximity'][0]['near_fraction']-=.01
        density['proximity'][0]['near_sample_m']-=.001
        self.assertTrue(screen._acceptable(self.old,SimpleNamespace(statistics=current),self.density,density))
        for key,value in (('connected_detours',3),('connector_length_m',.11),
                          ('revisit_connector_length_m',.03),('total_length_m',.31),('gap_count',2)):
            changed=deepcopy(current);changed[key]=value
            with self.subTest(key=key):
                self.assertFalse(screen._acceptable(self.old,SimpleNamespace(statistics=changed),self.density,density))
        density['proximity'][1]['near_fraction']+=.01
        self.assertFalse(screen._acceptable(self.old,SimpleNamespace(statistics=current),self.density,density))

    def test_candidate_budget_rejected_before_model_access(self):
        for value in (0,17,True,1.5):
            with self.subTest(value=value),self.assertRaises(ValueError):
                screen.screen(None,None,None,max_candidates=value)

    def test_longer_scans_cannot_dilute_absolute_near_row_length(self):
        density=deepcopy(self.density)
        density['proximity'][0]['near_fraction']-=.01
        current=SimpleNamespace(statistics=deepcopy(self.stats))
        self.assertFalse(screen._acceptable(self.old,current,self.density,density))
        density['proximity'][0]['near_sample_m']+=.001
        self.assertFalse(screen._acceptable(self.old,current,self.density,density))
        density['proximity'][0]['near_sample_m']-=.002
        self.assertTrue(screen._acceptable(self.old,current,self.density,density))
        density['proximity'][1]['near_sample_m']+=.001
        self.assertFalse(screen._acceptable(self.old,current,self.density,density))

    def test_screen_revalidates_each_candidate_and_never_claims_coverage(self):
        plan=dict(treatment_head={'mode':'fixed_width_auto'},planning_context={'revision':{'id':1}})
        model=SimpleNamespace(store=SimpleNamespace(mesh='mesh'))
        first=SimpleNamespace(statistics=deepcopy(self.stats))
        second=SimpleNamespace(statistics=deepcopy(self.stats))
        density1=deepcopy(self.density);density2=deepcopy(self.density)
        density1['proximity'][0]['near_fraction']+=.01
        density2['proximity'][0]['near_fraction']-=.01
        density2['proximity'][0]['near_sample_m']-=.001
        candidates=[dict(row=0,row_shift={'id':0}),dict(row=1,row_shift={'id':1})]
        proposed=dict(proposals=candidates,layout_algorithm='research',routing_algorithm='research')
        plans=[{'surface_trajectory':{'id':0}},{'surface_trajectory':{'id':1}}]
        with patch.object(screen.planning_revision,'require_current'), \
             patch.object(screen.row_shift_proposal_research,'propose',return_value=proposed), \
             patch.object(screen.planning_state,'validated_surface',return_value=self.old), \
             patch.object(screen.planning_state,'validate') as validate, \
             patch.object(screen.surface_workspace,'generate',side_effect=plans), \
             patch.object(screen.SurfaceTrajectory,'from_dict',side_effect=[first,second]), \
             patch.object(screen.row_proximity_research,'measure',side_effect=[self.density,density1,density2]):
            report,selected=screen.screen(model,{},plan,max_candidates=2)
        self.assertEqual(validate.call_count,2)
        self.assertEqual(report['selected_index'],1);self.assertEqual(selected,plans[1])
        self.assertFalse(report['installed_route']);self.assertFalse(report['coverage_verified'])

    def test_geometric_validation_failure_does_not_return_candidate(self):
        plan=dict(treatment_head={},planning_context={'revision':{'id':1}})
        model=SimpleNamespace(store=SimpleNamespace(mesh='mesh'))
        proposed=dict(proposals=[dict(row=0,row_shift={'id':0})],layout_algorithm='x',routing_algorithm='x')
        with patch.object(screen.planning_revision,'require_current'), \
             patch.object(screen.row_shift_proposal_research,'propose',return_value=proposed), \
             patch.object(screen.planning_state,'validated_surface',return_value=self.old), \
             patch.object(screen.planning_state,'validate',side_effect=ValueError('invalid geometry')), \
             patch.object(screen.surface_workspace,'generate',return_value={'surface_trajectory':{}}), \
             patch.object(screen.row_proximity_research,'measure',return_value=self.density):
            with self.assertRaisesRegex(ValueError,'invalid geometry'):
                screen.screen(model,{},plan,max_candidates=1)

    def test_priority_order_only_changes_which_candidates_are_revalidated(self):
        from femto.planning import preconnection_rank_research
        plan=dict(treatment_head={},planning_context={'revision':{'id':1}})
        model=SimpleNamespace(store=SimpleNamespace(mesh='mesh'))
        candidates=[dict(row=0,row_shift={'id':0}),dict(row=1,row_shift={'id':1})]
        proposed=dict(proposals=candidates,layout_algorithm='x',routing_algorithm='x')
        density=deepcopy(self.density);density['proximity'][0]['near_fraction']-=.01;density['proximity'][0]['near_sample_m']-=.001
        with patch.object(screen.planning_revision,'require_current'), \
             patch.object(screen.row_shift_proposal_research,'propose',return_value=proposed), \
             patch.object(screen.planning_state,'validated_surface',return_value=self.old), \
             patch.object(screen.planning_state,'validate') as validate, \
             patch.object(preconnection_rank_research,'rank',return_value={'order':[1,0]}), \
             patch.object(screen.surface_workspace,'generate',return_value={'surface_trajectory':{}}) as generate, \
             patch.object(screen.SurfaceTrajectory,'from_dict',return_value=self.old), \
             patch.object(screen.row_proximity_research,'measure',side_effect=[self.density,density]):
            report,_=screen.screen(model,{},plan,max_candidates=1,rank_strategy='preconnection')
        self.assertEqual(report['selected_index'],1);self.assertEqual(report['trials'][0]['proposal_index'],1)
        self.assertEqual(generate.call_args.kwargs['research_row_shifts'],[{'id':1}])
        self.assertEqual(validate.call_count,1);self.assertFalse(report['coverage_verified'])

    def test_local_preview_falls_back_to_full_ranking(self):
        from femto.planning import local_row_rank_research,preconnection_rank_research
        plan=dict(treatment_head={},planning_context={'revision':{'id':1}})
        model=SimpleNamespace(store=SimpleNamespace(mesh='mesh'))
        proposed=dict(proposals=[dict(row=0,row_shift={'id':0})],layout_algorithm='x',routing_algorithm='x')
        density=deepcopy(self.density);density['proximity'][0]['near_fraction']-=.01;density['proximity'][0]['near_sample_m']-=.001
        with patch.object(screen.planning_revision,'require_current'), \
             patch.object(screen.row_shift_proposal_research,'propose',return_value=proposed), \
             patch.object(screen.planning_state,'validated_surface',return_value=self.old), \
             patch.object(screen.planning_state,'validate'), \
             patch.object(local_row_rank_research,'rank',side_effect=local_row_rank_research.PreviewUnavailable('multirow')), \
             patch.object(preconnection_rank_research,'rank',return_value={'order':[0]}) as fallback, \
             patch.object(screen.surface_workspace,'generate',return_value={'surface_trajectory':{}}), \
             patch.object(screen.SurfaceTrajectory,'from_dict',return_value=self.old), \
             patch.object(screen.row_proximity_research,'measure',side_effect=[self.density,density]):
            report,_=screen.screen(model,{},plan,max_candidates=1,rank_strategy='local')
        self.assertEqual(fallback.call_count,1);self.assertEqual(report['ranking_fallback_reason'],'multirow')
        self.assertFalse(report['coverage_verified'])

    def test_joint_priority_cannot_bypass_candidate_geometry(self):
        from femto.planning import local_row_rank_research
        plan=dict(treatment_head={},planning_context={'revision':{'id':1}})
        model=SimpleNamespace(store=SimpleNamespace(mesh='mesh'))
        proposed=dict(proposals=[dict(row=0,row_shift={'id':0})],layout_algorithm='x',routing_algorithm='x')
        with patch.object(screen.planning_revision,'require_current'), \
             patch.object(screen.row_shift_proposal_research,'propose',return_value=proposed), \
             patch.object(screen.planning_state,'validated_surface',return_value=self.old), \
             patch.object(screen.planning_state,'validate',side_effect=ValueError('invalid geometry')) as validate, \
             patch.object(local_row_rank_research,'rank',return_value={'order':[0]}) as priority, \
             patch.object(screen.surface_workspace,'generate',return_value={'surface_trajectory':{}}), \
             patch.object(screen.row_proximity_research,'measure',return_value=self.density):
            with self.assertRaisesRegex(ValueError,'invalid geometry'):
                screen.screen(model,{},plan,max_candidates=1,rank_strategy='joint_preview')
        self.assertTrue(priority.call_args.kwargs['motion_priority']);self.assertEqual(validate.call_count,1)

    def test_partial_priority_still_requires_complete_candidate_geometry(self):
        from femto.planning import local_row_rank_research
        plan=dict(treatment_head={},planning_context={'revision':{'id':1}})
        model=SimpleNamespace(store=SimpleNamespace(mesh='mesh'))
        proposed=dict(proposals=[dict(row=0,row_shift={'id':0})],layout_algorithm='x',routing_algorithm='x')
        with patch.object(screen.planning_revision,'require_current'), \
             patch.object(screen.row_shift_proposal_research,'propose',return_value=proposed), \
             patch.object(screen.planning_state,'validated_surface',return_value=self.old), \
             patch.object(screen.planning_state,'validate',side_effect=ValueError('invalid geometry')) as validate, \
             patch.object(local_row_rank_research,'rank',return_value={'order':[0]}) as priority, \
             patch.object(screen.surface_workspace,'generate',return_value={'surface_trajectory':{}}), \
             patch.object(screen.row_proximity_research,'measure',return_value=self.density):
            with self.assertRaisesRegex(ValueError,'invalid geometry'):
                screen.screen(model,{},plan,max_candidates=1,rank_strategy='joint_partial')
        self.assertTrue(priority.call_args.kwargs['allow_partial'])
        self.assertTrue(priority.call_args.kwargs['prefer_known_motion'])
        self.assertEqual(validate.call_count,1)

    def test_joint_priority_falls_back_when_preview_is_inconclusive(self):
        from femto.planning import local_row_rank_research,preconnection_rank_research
        plan=dict(treatment_head={},planning_context={'revision':{'id':1}})
        model=SimpleNamespace(store=SimpleNamespace(mesh='mesh'))
        proposed=dict(proposals=[dict(row=0,row_shift={'id':0})],layout_algorithm='x',routing_algorithm='x')
        density=deepcopy(self.density);density['proximity'][0]['near_fraction']-=.01;density['proximity'][0]['near_sample_m']-=.001
        with patch.object(screen.planning_revision,'require_current'), \
             patch.object(screen.row_shift_proposal_research,'propose',return_value=proposed), \
             patch.object(screen.planning_state,'validated_surface',return_value=self.old), \
             patch.object(screen.planning_state,'validate'), \
             patch.object(local_row_rank_research,'rank',side_effect=local_row_rank_research.PreviewUnavailable('ambiguous')) as priority, \
             patch.object(preconnection_rank_research,'rank',return_value={'order':[0]}) as fallback, \
             patch.object(screen.surface_workspace,'generate',return_value={'surface_trajectory':{}}), \
             patch.object(screen.SurfaceTrajectory,'from_dict',return_value=self.old), \
             patch.object(screen.row_proximity_research,'measure',side_effect=[self.density,density]):
            report,_=screen.screen(model,{},plan,max_candidates=1,rank_strategy='joint_preview')
        self.assertTrue(priority.call_args.kwargs['motion_priority']);self.assertEqual(fallback.call_count,1)
        self.assertEqual(report['ranking_fallback_reason'],'ambiguous');self.assertFalse(report['coverage_verified'])


if __name__=='__main__':unittest.main()
