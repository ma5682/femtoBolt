"""Regression checks for bounded isolated-row replanning decisions."""
from dataclasses import dataclass, field
import unittest
from unittest.mock import patch

from femto.planning.repair_workflow import generate


@dataclass
class Route:
    gaps: int
    length: float = 5.
    unplannable: int = 0
    gap_report: dict = field(default_factory=dict)

    @property
    def statistics(self):
        return dict(gap_count=self.gaps + 1, inter_block_gap_count=1,
                    unplannable_face_count=self.unplannable, total_length_m=self.length)

    def to_dict(self):
        return dict(gaps=self.gaps, length=self.length)


class RepairWorkflowTests(unittest.TestCase):
    def run_workflow(self, routes, proposals):
        with patch('femto.planning.continuous_quality.candidate', side_effect=lambda *a, **kw: kw['fixed_width_layout']) as candidate, \
             patch('femto.planning.continuous_quality.replan', side_effect=routes) as replan, \
             patch('femto.planning.isolated_row_repair.requests', return_value=proposals):
            result=generate('source',None,None,{},5,None,{'version':'trial'})
        return result,candidate.call_count,replan.call_count

    def test_no_isolated_row_keeps_original_without_extra_planning(self):
        result,candidates,replans=self.run_workflow([Route(2)],[])
        self.assertEqual((candidates,replans),(1,1))
        self.assertIsNone(result.gap_report['row_repair']['selected'])
        self.assertEqual(result.gap_report['row_repair']['target_coverage_claim'],'none')

    def test_accepts_improved_checked_trial_and_retains_initial_evidence(self):
        proposal={'trial_shifts':[{'shift_m':.00025},{'shift_m':-.00025}]}
        result,candidates,replans=self.run_workflow([Route(2),Route(0,5.02)],[proposal])
        report=result.gap_report['row_repair']
        self.assertEqual((candidates,replans),(2,2))
        self.assertEqual(report['selected'],proposal['trial_shifts'][0])
        self.assertEqual(report['initial_trajectory']['gaps'],2)
        self.assertFalse(report['failed_search_is_proof_of_disconnection'])

    def test_two_failed_trials_keep_original_even_if_another_proposal_exists(self):
        proposals=[{'trial_shifts':[{'shift_m':.00025},{'shift_m':-.00025}]},
                   {'trial_shifts':[{'shift_m':.00025}]}]
        result,candidates,replans=self.run_workflow([Route(2),Route(2),Route(0,5.2)],proposals)
        self.assertEqual((candidates,replans),(3,3))
        self.assertEqual(result.gaps,2)
        self.assertIsNone(result.gap_report['row_repair']['selected'])
        self.assertEqual(len(result.gap_report['row_repair']['attempts']),2)

    def test_does_not_accept_improvement_with_more_unplannable_faces(self):
        proposal={'trial_shifts':[{'shift_m':.00025},{'shift_m':-.00025}]}
        result,_,_=self.run_workflow([Route(2),Route(0,unplannable=1),Route(2)],[proposal])
        self.assertIsNone(result.gap_report['row_repair']['selected'])

    def test_short_patch_without_neighbor_rows_keeps_original(self):
        proposal={'trial_shifts':[{'shift_m':.00025},{'shift_m':-.00025}]}
        def candidate(*args,**kwargs):
            if 'row_shifts' in kwargs['fixed_width_layout']:raise ValueError('局部行调整缺少相邻扫描行')
            return kwargs['fixed_width_layout']
        with patch('femto.planning.continuous_quality.candidate',side_effect=candidate), \
             patch('femto.planning.continuous_quality.replan',return_value=Route(2)), \
             patch('femto.planning.isolated_row_repair.requests',return_value=[proposal]):
            result=generate('source',None,None,{},5,None,{'version':'trial'})
        report=result.gap_report['row_repair']
        self.assertIsNone(report['selected'])
        self.assertEqual([a['status'] for a in report['attempts']],['infeasible_spacing','infeasible_spacing'])
