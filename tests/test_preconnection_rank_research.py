from types import SimpleNamespace
import unittest
from unittest.mock import patch

from femto.planning import planning_revision,preconnection_rank_research as rank
from femto.planning.fixed_width_layout import SHIFT_PATCH_VERSION


class PreconnectionRankTests(unittest.TestCase):
    def test_order_only_preserves_unranked_candidates_and_does_not_claim_coverage(self):
        plan=dict(treatment_head=dict(effective_treatment_width_mm=5,overlap_percent=20,row_spacing_mm=4),
                  planning_context={'revision':{'id':1}})
        proposal=dict(source_route_sha256=planning_revision.digest(plan),layout_algorithm=SHIFT_PATCH_VERSION,
                      proposals=[dict(row_shift={'id':i}) for i in range(3)])
        model=SimpleNamespace(store=SimpleNamespace(mesh='mesh'))
        trajectory=SimpleNamespace(planner={'parameters':{'quality_evidence':{},'effective_domain':{}}})
        densities=[dict(proximity=[dict(near_fraction=.2),dict(near_fraction=.3)]),
                   dict(proximity=[dict(near_fraction=.1),dict(near_fraction=.2)])]
        with patch.object(rank.planning_revision,'require_current'), \
             patch.object(rank.planning_state,'validated_surface',return_value=trajectory), \
             patch.object(rank.surface_workspace,'snapshot',return_value='manifest'), \
             patch.object(rank.surface_workspace,'canonical_region_mesh',return_value='mesh'), \
             patch.object(rank.continuous_quality,'candidate',return_value='raw') as candidate, \
             patch.object(rank.quality_constrained,'constrain',return_value='constrained'), \
             patch.object(rank.row_proximity_research,'measure',side_effect=densities):
            report=rank.rank(model,{},plan,proposal,max_rank=2)
        self.assertEqual(candidate.call_count,2)
        self.assertEqual(report['order'],[1,0,2]);self.assertEqual(report['unranked_count'],1)
        self.assertFalse(report['coverage_claim']);self.assertTrue(report['candidate_only'])

    def test_rejects_wrong_source_and_bad_budget(self):
        plan=dict(treatment_head={},planning_context={'revision':{}})
        proposal=dict(source_route_sha256='wrong',layout_algorithm=SHIFT_PATCH_VERSION,proposals=[])
        with self.assertRaisesRegex(ValueError,'来源路线'):
            rank.rank(None,None,plan,proposal)
        for value in (0,33,True):
            with self.subTest(value=value),self.assertRaises(ValueError):
                rank.rank(None,None,plan,proposal,max_rank=value)


if __name__=='__main__':unittest.main()
