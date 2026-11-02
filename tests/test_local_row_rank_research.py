import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from femto.planning import local_row_rank_research as local
from femto.planning import preconnection_rank_research as full, planning_revision
from femto.planning.fixed_width_layout import SHIFT_PATCH_VERSION
from femto.planning.surface_trajectory import ScanSegment


class LocalRowRankTests(unittest.TestCase):
    def setUp(self):
        self.key=(0,'a'*64,0)
        part=dict(frame={'origin':[0.,0.,.5],'e2':[0.,1.,0.]},positions_m=[0.,.003,.006],face_ids=[0])
        points=[dict(xyz_m=[x,.003,.5],row_index=0,triangle_index=0,role='scan') for x in (.001,.002,.003)]
        self.state=dict(rows={(0,0):self.key},patches={self.key:part},
                        vertices=np.array([[0.,0.,.5],[.01,0.,.5],[0.,.01,.5]]),
                        faces=np.array([[0,1,2]]),base=[ScanSegment(0,0,points,0)],width=5.)
        self.item=dict(row=0,row_shift=dict(block_index=0,fragment_face_sha256='a'*64,
                        patch_seed_face=0,position_m=.003,shift_m=.00015))

    def test_mismatched_row_or_multilevel_shift_requires_fallback(self):
        wrong=dict(self.item,row=1)
        with self.assertRaises(local.PreviewUnavailable):
            local._preview(self.state,wrong,{'overlap_percent':20},lambda:None)
        with patch.object(local,'shifted',return_value=np.array([.00001,.00315,.006])):
            with self.assertRaisesRegex(local.PreviewUnavailable,'相邻行'):
                local._preview(self.state,self.item,{'overlap_percent':20},lambda:None)

    def test_invalid_budget_and_source_are_rejected(self):
        for budget in (0,33,True):
            with self.subTest(budget=budget),self.assertRaises(ValueError):
                local.rank(None,None,None,{},max_rank=budget)

    def test_empty_proposals_skip_mesh_work_but_validate_source(self):
        plan={'treatment_head':{},'planning_context':{'revision':{}}}
        proposal={'source_route_sha256':planning_revision.digest(plan),
                  'layout_algorithm':SHIFT_PATCH_VERSION,'proposals':[]}
        model=SimpleNamespace(store=SimpleNamespace(mesh='mesh'))
        trajectory=SimpleNamespace(planner={'parameters':{'quality_evidence':{},'effective_domain':{}}})
        for module in (local,full):
            with self.subTest(module=module.__name__), \
                 patch.object(module.planning_revision,'require_current') as revision, \
                 patch.object(module.planning_state,'validated_surface',return_value=trajectory) as validate, \
                 patch.object(module.surface_workspace,'canonical_region_mesh',side_effect=AssertionError('mesh work')), \
                 patch.object(module.surface_workspace,'snapshot',side_effect=AssertionError('snapshot work')):
                result=module.rank(model,{},plan,proposal,max_rank=1)
            self.assertEqual(result['order'],[])
            self.assertEqual(result['ranked_count'],0)
            self.assertEqual(validate.call_count,1)
            self.assertGreaterEqual(revision.call_count,2)
            self.assertFalse(result['coverage_claim'])

    def test_preview_replaces_only_changed_row_and_rechecks_its_edges(self):
        unchanged=[dict(xyz_m=[x,.006,.5],row_index=1,triangle_index=0,role='scan') for x in (.001,.002)]
        self.state['base'].append(ScanSegment(1,1,unchanged,0))
        self.state.update(blocks=[dict(component_id=0)],labels=np.ones((2,2),int),
            components=np.array([0]),params={},intrinsic=np.eye(3),image_size=np.array([2,2]),
            mesh='mesh',record={},allowed_faces=np.array([True]))
        raw=[dict(candidates=[dict(points=[dict(xyz_m=[x,.00315,.5],triangle_index=0) for x in (.001,.002)])])]
        def measure(value,check):
            rows=[item.points[0]['row_index'] for item in value.ordered_items]
            self.assertEqual(rows,[1,0]);self.assertEqual(value.ordered_items[-1].points[0]['xyz_m'][1],.00315)
            return {'proximity':[{'near_fraction':.01}]}
        with patch.object(local,'shifted',return_value=np.array([0.,.00315,.006])), \
             patch.object(local,'_slice_rows',return_value=(raw,[])), \
             patch.object(local.quality_constrained,'edge_checks',return_value=(np.array([True]),['accepted'],np.array([0.]))) as edges, \
             patch.object(local.row_proximity_research,'measure',side_effect=measure):
            value=local._preview(self.state,self.item,{'overlap_percent':20},lambda:None)
        self.assertEqual(value['proximity'][0]['near_fraction'],.01);self.assertEqual(edges.call_count,1)

    def test_partial_preview_keeps_valid_candidates_and_unavailable_last(self):
        plan={'treatment_head':{},'planning_context':{'revision':{}}}
        proposal={'source_route_sha256':planning_revision.digest(plan),'layout_algorithm':SHIFT_PATCH_VERSION,
                  'proposals':[dict(row=i,row_shift={'block_index':0}) for i in range(3)]}
        model=SimpleNamespace(store=SimpleNamespace(mesh='mesh'))
        baseline={'proximity':[{'near_fraction':.02}]}
        better={'proximity':[{'near_fraction':.01}]}
        with patch.object(local.planning_revision,'require_current'), \
             patch.object(local.planning_state,'validated_surface',return_value='trajectory'), \
             patch.object(local.surface_workspace,'canonical_region_mesh',return_value='mesh'), \
             patch.object(local,'_prepare',return_value={'width':5.}), \
             patch.object(local.row_proximity_research,'measure',return_value=baseline), \
             patch.object(local,'_preview',side_effect=[(better,[]),local.PreviewUnavailable('多行联动'),(better,[])]), \
             patch('femto.planning.local_motion_proxy_research.estimate',side_effect=[None,{'predicted_connector_delta_m':0.}]):
            result=local.rank(model,{},plan,proposal,max_rank=3,motion_priority=True,allow_partial=True,prefer_known_motion=True)
        self.assertEqual(result['order'],[2,0,1])
        self.assertEqual(result['ranked_count'],2);self.assertEqual(result['unranked_count'],1)
        self.assertEqual(result['preview_skipped'],[{'index':1,'reason':'多行联动'}])
        self.assertFalse(result['coverage_claim'])

    def test_partial_preview_all_unavailable_requires_full_fallback(self):
        plan={'treatment_head':{},'planning_context':{'revision':{}}}
        proposal={'source_route_sha256':planning_revision.digest(plan),'layout_algorithm':SHIFT_PATCH_VERSION,
                  'proposals':[dict(row=0,row_shift={'block_index':0})]}
        model=SimpleNamespace(store=SimpleNamespace(mesh='mesh'))
        with patch.object(local.planning_revision,'require_current'), \
             patch.object(local.planning_state,'validated_surface',return_value='trajectory'), \
             patch.object(local.surface_workspace,'canonical_region_mesh',return_value='mesh'), \
             patch.object(local,'_prepare',return_value={'width':5.}), \
             patch.object(local.row_proximity_research,'measure',return_value={'proximity':[{'near_fraction':.02}]}), \
             patch.object(local,'_preview',side_effect=local.PreviewUnavailable('不可预览')):
            with self.assertRaisesRegex(local.PreviewUnavailable,'全部局部候选'):
                local.rank(model,{},plan,proposal,max_rank=1,motion_priority=True,allow_partial=True,prefer_known_motion=True)


if __name__=='__main__':unittest.main()
