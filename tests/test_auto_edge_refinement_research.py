from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock,patch

from femto.planning import auto_edge_refinement_research as auto


class AutoEdgeRefinementTests(unittest.TestCase):
    def setUp(self):
        self.plan={'treatment_head':{},'planning_context':{'revision':{}},'surface_trajectory':{'id':0}}
        self.old=SimpleNamespace(metadata={'sha256':'old'});self.final=SimpleNamespace(metadata={'sha256':'proof'})
        self.density={'proximity':[{'near_fraction':.1},{'near_fraction':.2}]}
        def trial(index,length):
            return dict(proposal_index=index,accepted=True,proposal={'row_shift':{'id':index}},density=deepcopy(self.density),
                statistics=dict(connected_detours=1,revisit_connector_length_m=.01,connector_length_m=length,total_length_m=1.))
        self.trials=[trial(1,.1),trial(2,.2)]
        self.first={'surface_trajectory':{'id':1}};self.second={'surface_trajectory':{'id':2}}
        self.screen={'trials':self.trials,'selected_index':1}
        self.model=SimpleNamespace(store=SimpleNamespace(mesh='mesh'))
        self.width={'schema':'femto-nominal-fixed-ball-v1','effective_width_mm':10.}
        self.domain=SimpleNamespace(binding={},areas=[])

    def test_empty_screen_preserves_source_and_skips_proof(self):
        screen={'trials':[],'selected_index':None}
        with patch.object(auto.planning_revision,'require_current'), \
             patch.object(auto.auto_row_shift_screen_research,'screen',return_value=(screen,None)), \
             patch.object(auto.coverage,'validate',side_effect=AssertionError('unexpected proof')):
            report,result=auto.run(self.model,{},self.plan,old_certificate=self.old)
        self.assertFalse(report['adopted']);self.assertEqual(report['reason'],'no_screen_candidate');self.assertIsNone(result)

    def test_second_candidate_checked_when_first_fails_retention(self):
        self._check_second_candidate(auto.RELOCATION_ROUTING)

    def test_second_candidate_retains_deep_source_routing(self):
        self.plan['planning_context']['revision']={'inputs':{'algorithms':{'routing':auto.DEEP_RELOCATION_ROUTING}}}
        self._check_second_candidate(auto.DEEP_RELOCATION_ROUTING)

    def _check_second_candidate(self,routing):
        trajectories={0:SimpleNamespace(statistics={}),1:SimpleNamespace(statistics=self.trials[0]['statistics']),
                      2:SimpleNamespace(statistics=self.trials[1]['statistics'])}
        def trajectory(raw,**kwargs):return trajectories[raw['id']]
        session=MagicMock();session.__enter__.return_value=session;session.seeds=[];session.route={}
        with patch.object(auto.planning_revision,'require_current'), \
             patch('femto.planning.old_coverage_session_research._OldCoverageSession',return_value=session), \
             patch.object(auto.auto_row_shift_screen_research,'screen',return_value=(self.screen,self.first)), \
             patch.object(auto,'from_plan',return_value=(self.domain,[],self.width,None)), \
             patch.object(auto.SurfaceTrajectory,'from_dict',side_effect=trajectory), \
             patch.object(auto.coverage_workflow,'route_binding',return_value={'route':'bound'}), \
             patch.object(auto.coverage,'validate'),patch.object(auto,'_same_target'), \
             patch.object(auto.coverage,'evaluate',return_value='normal'), \
             patch.object(auto.auto_fraction_supplement_research,'propose',return_value=({'requests':[]},self.final)) as fraction, \
             patch.object(auto.coverage_partition,'compare_union',side_effect=[{'retained_subset':False},{'retained_subset':True}]) as compare, \
             patch.object(auto.witness_bend_workflow_research,'build',return_value=({'adopted':False,'reason':'no_qualifying_candidate'},None)), \
             patch.object(auto.surface_workspace,'generate',return_value=self.second) as regenerate, \
             patch.object(auto.planning_state,'validate') as validate, \
             patch.object(auto.row_proximity_research,'measure',return_value=self.density), \
             patch.object(auto.coverage,'_checked_assets'):
            report,result=auto.run(self.model,{},self.plan,old_certificate=self.old,max_verified_candidates=2)
            def change_proof(*args,**kwargs):
                self.old.metadata['sha256']='changed'
                return {'requests':[]},self.final
            fraction.side_effect=change_proof;compare.side_effect=[{'retained_subset':True}]
            with self.assertRaisesRegex(ValueError,'原证据发生变化'):
                auto.run(self.model,{},self.plan,old_certificate=self.old,max_verified_candidates=2)
        self.assertTrue(report['adopted']);self.assertEqual(report['selected_index'],2)
        self.assertEqual(result['replay_strategy'],'fraction_reproof');self.assertIs(result['final_certificate'],self.final)
        self.assertEqual(regenerate.call_count,1);self.assertEqual(validate.call_count,1)
        self.assertEqual(regenerate.call_args.kwargs['routing_algorithm'],routing)
        self.assertFalse(report['installed_route']);self.assertFalse(report['execution_ready'])

    def test_input_change_is_detected_even_when_no_candidate(self):
        def mutate(model,region,plan,**kwargs):
            plan['treatment_head']['changed']=True
            return {'trials':[],'selected_index':None},None
        with patch.object(auto.planning_revision,'require_current'), \
             patch.object(auto.auto_row_shift_screen_research,'screen',side_effect=mutate):
            with self.assertRaisesRegex(ValueError,'原路线或选区发生变化'):
                auto.run(self.model,{},self.plan,old_certificate=self.old)

    def test_budget_and_inconsistent_screen_are_rejected(self):
        for value in (0,5,True):
            with self.subTest(value=value),self.assertRaises(ValueError):
                auto.run(None,None,None,max_verified_candidates=value)
        with patch.object(auto.planning_revision,'require_current'), \
             patch.object(auto.auto_row_shift_screen_research,'screen',return_value=(self.screen,None)):
            with self.assertRaisesRegex(ValueError,'排序结果不一致'):
                auto.run(self.model,{},self.plan,old_certificate=self.old)

    def test_existing_coverage_cannot_be_replaced_with_another_certificate(self):
        plan=deepcopy(self.plan);plan['coverage_evidence']={'existing':True}
        existing=SimpleNamespace(metadata={'sha256':'existing'})
        with patch.object(auto.planning_revision,'require_current'), \
             patch.object(auto.auto_row_shift_screen_research,'screen',return_value=(self.screen,self.first)), \
             patch.object(auto.coverage_workflow,'cached',return_value=existing):
            with self.assertRaisesRegex(ValueError,'保留原方案已有覆盖证据'):
                auto.run(self.model,{},plan,old_certificate=self.old)

    def test_direct_strategy_budget_falls_back_without_bypassing_proof(self):
        from femto.planning import direct_fraction_witness_research as direct
        session=MagicMock();session.__enter__.return_value=session;session.seeds=[];session.route={}
        with patch.object(auto.planning_revision,'require_current'), \
             patch('femto.planning.old_coverage_session_research._OldCoverageSession',return_value=session), \
             patch.object(auto.auto_row_shift_screen_research,'screen',return_value=(self.screen,self.first)), \
             patch.object(auto,'from_plan',return_value=(self.domain,[],self.width,None)), \
             patch.object(auto.SurfaceTrajectory,'from_dict',return_value=SimpleNamespace()), \
             patch.object(auto.coverage_workflow,'route_binding',return_value={'route':'bound'}), \
             patch.object(auto,'_same_target'),patch.object(auto.coverage,'evaluate',return_value='normal') as evaluate, \
             patch.object(direct,'propose',side_effect=ValueError('分数位置分批次数超过预算')) as propose, \
             patch.object(auto.auto_fraction_supplement_research,'propose',return_value=({'requests':[]},self.final)) as original, \
             patch.object(auto.coverage_partition,'compare_union',return_value={'retained_subset':True}), \
             patch.object(auto.coverage,'_checked_assets'):
            report,result=auto.run(self.model,{},self.plan,old_certificate=self.old,evidence_strategy='direct_witness')
            propose.side_effect=ValueError('目标资产损坏')
            with self.assertRaisesRegex(ValueError,'目标资产损坏'):
                auto.run(self.model,{},self.plan,old_certificate=self.old,evidence_strategy='direct_witness')
        self.assertTrue(report['adopted']);self.assertIs(result['final_certificate'],self.final)
        self.assertEqual(report['trials'][0]['evidence_strategy'],'intermediate_fallback')
        self.assertEqual(report['trials'][0]['fallback_reason'],'分数位置分批次数超过预算')
        self.assertEqual(evaluate.call_count,1);self.assertEqual(original.call_count,1)

    def test_direct_strategy_without_qualifying_result_retries_existing_method(self):
        from femto.planning import direct_fraction_witness_research as direct
        session=MagicMock();session.__enter__.return_value=session;session.seeds=[];session.route={}
        with patch.object(auto.planning_revision,'require_current'), \
             patch('femto.planning.old_coverage_session_research._OldCoverageSession',return_value=session), \
             patch.object(auto.auto_row_shift_screen_research,'screen',return_value=(self.screen,self.first)), \
             patch.object(auto,'from_plan',return_value=(self.domain,[],self.width,None)), \
             patch.object(auto.SurfaceTrajectory,'from_dict',return_value=SimpleNamespace()), \
             patch.object(auto.coverage_workflow,'route_binding',return_value={'route':'bound'}), \
             patch.object(auto,'_same_target'),patch.object(auto.coverage,'evaluate',return_value='normal'), \
             patch.object(direct,'propose',return_value=({'requests':[]},self.final)), \
             patch.object(auto.auto_fraction_supplement_research,'propose',return_value=({'requests':[]},self.final)) as original, \
             patch.object(auto.coverage_partition,'compare_union',side_effect=[{'retained_subset':False},{'retained_subset':True}]), \
             patch.object(auto.witness_bend_workflow_research,'build',return_value=({'adopted':False,'reason':'no_qualifying_candidate'},None)), \
             patch.object(auto.coverage,'_checked_assets'):
            report,result=auto.run(self.model,{},self.plan,old_certificate=self.old,evidence_strategy='direct_witness')
        entry=report['trials'][0]
        self.assertTrue(report['adopted']);self.assertIs(result['final_certificate'],self.final)
        self.assertEqual(entry['fallback_reason'],'direct_no_qualifying_candidate')
        self.assertEqual(entry['direct_attempt']['reason'],'no_qualifying_candidate');self.assertEqual(original.call_count,1)

    def test_single_final_success_never_uses_intermediate_evidence(self):
        from femto.planning import single_final_witness_research as single
        session=MagicMock();session.__enter__.return_value=session;session.seeds=[];session.route={}
        result={'adopted':True,'reason':'research_candidate_validated','direct_recipe':{'recipe':1}}
        with patch.object(auto.planning_revision,'require_current'), \
             patch('femto.planning.old_coverage_session_research._OldCoverageSession',return_value=session), \
             patch.object(auto.auto_row_shift_screen_research,'screen',return_value=(self.screen,self.first)), \
             patch.object(auto,'from_plan',return_value=(self.domain,[],self.width,None)), \
             patch.object(auto.SurfaceTrajectory,'from_dict',return_value=SimpleNamespace()), \
             patch.object(auto.coverage_workflow,'route_binding',return_value={'route':'bound'}), \
             patch.object(auto,'_same_target'),patch.object(single,'build',return_value=(result,self.final)) as build, \
             patch.object(auto.coverage,'evaluate',side_effect=AssertionError('unexpected intermediate proof')), \
             patch.object(auto.coverage,'_checked_assets'):
            report,selected=auto.run(self.model,{},self.plan,old_certificate=self.old,evidence_strategy='single_final')
        self.assertTrue(report['adopted']);self.assertIs(selected['final_certificate'],self.final)
        self.assertEqual(report['trials'][0]['evidence_strategy'],'single_final')
        self.assertIs(build.call_args.kwargs['_old_proof_session'],session)

    def test_single_final_rejection_falls_back_to_direct_method(self):
        from femto.planning import single_final_witness_research as single,direct_fraction_witness_research as direct
        session=MagicMock();session.__enter__.return_value=session;session.seeds=[];session.route={}
        rejected={'adopted':False,'reason':'no_qualifying_candidate'}
        with patch.object(auto.planning_revision,'require_current'), \
             patch('femto.planning.old_coverage_session_research._OldCoverageSession',return_value=session), \
             patch.object(auto.auto_row_shift_screen_research,'screen',return_value=(self.screen,self.first)), \
             patch.object(auto,'from_plan',return_value=(self.domain,[],self.width,None)), \
             patch.object(auto.SurfaceTrajectory,'from_dict',return_value=SimpleNamespace()), \
             patch.object(auto.coverage_workflow,'route_binding',return_value={'route':'bound'}), \
             patch.object(auto,'_same_target'),patch.object(single,'build',return_value=(rejected,None)), \
             patch.object(direct,'propose',return_value=({'requests':[]},self.final)) as fallback, \
             patch.object(auto.coverage_partition,'compare_union',return_value={'retained_subset':True}), \
             patch.object(auto.coverage,'_checked_assets'):
            report,selected=auto.run(self.model,{},self.plan,old_certificate=self.old,evidence_strategy='single_final')
        self.assertTrue(report['adopted']);self.assertIs(selected['final_certificate'],self.final)
        self.assertEqual(report['trials'][0]['single_fallback_reason'],'no_qualifying_candidate')
        self.assertEqual(fallback.call_count,1)

    def test_ranked_replacement_success_and_budget_fallback(self):
        from femto.planning import ranked_point_witness_research as ranked,direct_fraction_witness_research as direct
        session=MagicMock();session.__enter__.return_value=session;session.seeds=[];session.route={}
        result={'adopted':True,'reason':'research_candidate_validated','direct_recipe':{'recipe':1}}
        with patch.object(auto.planning_revision,'require_current'), \
             patch('femto.planning.old_coverage_session_research._OldCoverageSession',return_value=session), \
             patch.object(auto.auto_row_shift_screen_research,'screen',return_value=(self.screen,self.first)), \
             patch.object(auto,'from_plan',return_value=(self.domain,[],self.width,None)), \
             patch.object(auto.SurfaceTrajectory,'from_dict',return_value=SimpleNamespace()), \
             patch.object(auto.coverage_workflow,'route_binding',return_value={'route':'bound'}), \
             patch.object(auto,'_same_target'),patch.object(ranked,'build',return_value=(result,self.final)) as build, \
             patch.object(direct,'propose',return_value=({'requests':[]},self.final)) as fallback, \
             patch.object(auto.coverage_partition,'compare_union',return_value={'retained_subset':True}), \
             patch.object(auto.coverage,'_checked_assets'):
            report,selected=auto.run(self.model,{},self.plan,old_certificate=self.old,evidence_strategy='ranked_replace')
            self.assertTrue(report['adopted']);self.assertEqual(selected['replay_strategy'],'point_replacement_direct')
            self.assertEqual(report['trials'][0]['evidence_strategy'],'ranked_replace');fallback.assert_not_called()
            build.side_effect=ValueError('见证叶排序超过预算')
            report,selected=auto.run(self.model,{},self.plan,old_certificate=self.old,evidence_strategy='ranked_replace')
            self.assertTrue(report['adopted']);self.assertIs(selected['final_certificate'],self.final)
            self.assertEqual(report['trials'][0]['single_fallback_reason'],'见证叶排序超过预算')
            self.assertEqual(fallback.call_count,1)
            build.side_effect=ValueError('目标资产损坏')
            with self.assertRaisesRegex(ValueError,'目标资产损坏'):
                auto.run(self.model,{},self.plan,old_certificate=self.old,evidence_strategy='ranked_replace')

    def test_paired_strategy_passes_guide_and_preserves_replay_kind(self):
        from femto.planning import paired_witness_workflow_research as paired,direct_fraction_witness_research as direct
        session=MagicMock();session.__enter__.return_value=session;session.seeds=[];session.route={}
        guide=SimpleNamespace(metadata={'sha256':'guide'})
        accepted={'adopted':True,'reason':'research_candidate_validated','route_unchanged':False,'direct_recipe':{'version':'pair'}}
        with patch.object(auto.planning_revision,'require_current'), \
             patch('femto.planning.old_coverage_session_research._OldCoverageSession',return_value=session) as opened, \
             patch.object(auto.auto_row_shift_screen_research,'screen',return_value=(self.screen,self.first)), \
             patch.object(auto,'from_plan',return_value=(self.domain,[],self.width,None)), \
             patch.object(auto.SurfaceTrajectory,'from_dict',return_value=SimpleNamespace()), \
             patch.object(auto.coverage_workflow,'route_binding',return_value={'route':'bound'}), \
             patch.object(auto,'_same_target'), \
             patch.object(direct,'propose',return_value=({'requests':[]},self.final)), \
             patch.object(auto.coverage_partition,'compare_union',return_value={'retained_subset':False}), \
             patch.object(paired,'build',return_value=(accepted,self.final)) as build, \
             patch.object(auto.coverage,'_checked_assets'):
            report,selected=auto.run(self.model,{},self.plan,old_certificate=self.old,old_guide=guide,evidence_strategy='paired_replace')
            self.assertTrue(report['adopted']);self.assertEqual(selected['replay_strategy'],'paired_replacement_direct')
            self.assertIs(selected['old_guide'],guide);self.assertIs(opened.call_args.kwargs['guide'],guide)
            self.assertIs(build.call_args.kwargs['_old_proof_session'],session)
            accepted.update(route_unchanged=True,fraction_requests=[[0,0,1,2]])
            report,selected=auto.run(self.model,{},self.plan,old_certificate=self.old,old_guide=guide,evidence_strategy='paired_replace')
            self.assertEqual(selected['replay_strategy'],'fraction_reproof');self.assertNotIn('direct_recipe',selected)
            def mutate(*args,**kwargs):
                guide.metadata['sha256']='changed'
                return accepted,self.final
            build.side_effect=mutate
            with self.assertRaisesRegex(ValueError,'原引导发生变化'):
                auto.run(self.model,{},self.plan,old_certificate=self.old,old_guide=guide,evidence_strategy='paired_replace')


if __name__=='__main__':unittest.main()
