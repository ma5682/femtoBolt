from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import trimesh

from femto.planning import row_shift_proposal_research as proposals
from femto.planning.fixed_width_layout import OPTIMIZED_PATCH_VERSION
from femto.planning.surface_trajectory import ScanSegment


class RowShiftProposalResearchTests(unittest.TestCase):
    def setUp(self):
        self.mesh=trimesh.Trimesh(vertices=[[0.,0.,.5],[.01,0.,.5],[0.,.01,.5]],faces=[[0,1,2]],process=False)
        self.patch=dict(frame=dict(origin=[0.,0.,.5],e2=[0.,1.,0.]),positions_m=[0.,.003,.006],
                        first_row_index=0,retained_row_count=1,face_ids=[0],seed_face=0)
        self.fragment=dict(block_index=0,face_indices_sha256='a'*64,patches=[self.patch])
        layout=dict(request=dict(version=OPTIMIZED_PATCH_VERSION),row_shifts=[],fragments=[self.fragment])
        points=[dict(xyz_m=[x,.003,.5],row_index=0,triangle_index=0,route_index=i,role='scan')
                for i,x in enumerate((.001,.002,.003))]
        segment=ScanSegment(0,0,points,0)
        self.trajectory=SimpleNamespace(planner={'parameters':{'fixed_width_layout':layout}},
            gap_report={'reconnection':{'base_items':[segment.to_dict()]}})
        self.diagnostics={'rows':[dict(block=0,row=0,near_m=.001)]}

    def derive(self):
        return proposals._derive(self.trajectory,self.mesh,self.diagnostics,width_mm=5.,overlap_percent=20.,
                                 max_rows=8,fraction=.03,check=lambda:None)

    def test_interior_rows_propose_both_bounded_directions_without_mutation(self):
        before=deepcopy(self.trajectory)
        candidates,skipped,matched=self.derive()
        self.assertEqual(len(candidates),2);self.assertFalse(skipped);self.assertEqual(matched,1)
        self.assertEqual([v['row_shift']['shift_m'] for v in candidates],[-.00015,.00015])
        self.assertEqual(candidates[0]['row_shift']['position_m'],.003)
        self.assertEqual(self.trajectory,before)

    def test_ambiguous_or_boundary_rows_do_not_propose(self):
        self.patch['positions_m']=[.003,.006,.009]
        candidates,skipped,_=self.derive();self.assertFalse(candidates)
        self.assertEqual(skipped[0]['reason'],'unmatched_or_boundary_level')
        self.patch['positions_m']=[0.,.003,.006]
        self.fragment['patches'].append(deepcopy(self.patch))
        with self.assertRaisesRegex(ValueError,'不唯一'):self.derive()

    def test_source_budget_and_parameter_gate(self):
        with patch.object(proposals,'MAX_SOURCE_POINTS',0),self.assertRaisesRegex(ValueError,'超过预算'):
            self.derive()
        for fraction in (0,.051,True,float('nan')):
            with self.subTest(fraction=fraction),self.assertRaises(ValueError):
                proposals.propose(None,None,None,fraction=fraction)
        with self.assertRaises(ValueError):proposals.propose(None,None,None,max_rows=17)

    def test_preserves_deep_source_routing_for_shift_candidates(self):
        from femto.planning.continuous_quality import RELOCATION_ROUTING,DEEP_RELOCATION_ROUTING
        model=SimpleNamespace(store=SimpleNamespace(mesh=self.mesh))
        for routing in (RELOCATION_ROUTING,DEEP_RELOCATION_ROUTING):
            plan=dict(treatment_head=dict(effective_treatment_width_mm=5.,overlap_percent=20.),
                      planning_context=dict(revision=dict(inputs=dict(algorithms=dict(routing=routing)))))
            before=deepcopy(plan)
            with self.subTest(routing=routing), \
                 patch.object(proposals.planning_revision,'require_current'), \
                 patch.object(proposals.planning_state,'validated_surface',return_value=self.trajectory), \
                 patch.object(proposals.surface_workspace,'canonical_region_mesh',return_value=self.mesh), \
                 patch.object(proposals.row_overlap_diagnostics,'inspect',return_value=self.diagnostics):
                result=proposals.propose(model,{},plan)
                self.assertEqual(result['routing_algorithm'],routing)
                self.assertEqual(len(result['proposals']),2)
                self.assertFalse(result['coverage_claim']);self.assertFalse(result['execution_ready'])
                self.assertEqual(plan,before)


if __name__=='__main__':unittest.main()
