"""Bounded source references cannot smuggle coordinates or change endpoints."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from femto.planning import planning_revision as revisions,source_anchored_adjustment as adjustment
from femto.planning.surface_trajectory import ScanSegment,SCHEMA
from femto.planning.trajectory_export import current_algorithm_version,product_algorithm_notice


def point(x,row=0):
    return dict(xyz_m=[x,0.,1.],row_index=row,route_index=0,triangle_index=0,role='scan',geometry_support='mesh')


class SourceAnchoredTests(unittest.TestCase):
    def setUp(self):
        self.anchors=[dict(source_segment=0,source_point=1,base_segment=0,base_point=1),
                      dict(source_segment=0,source_point=2,base_segment=0,base_point=2)]
        segment=ScanSegment(0,0,[point(0.),point(.001),point(.002),point(.003)],0)
        self.base=SimpleNamespace(gap_report=dict(reconnection=dict(base_items=[segment.to_dict()])))
        self.source=SimpleNamespace(ordered_items=[ScanSegment(0,0,[point(0.),point(.0011),point(.0021),point(.003)],0)])

    def test_references_only_and_distinct(self):
        for anchors in ([],self.anchors[:1],self.anchors+[self.anchors[0]],
                        [self.anchors[0],self.anchors[0]],
                        [dict(self.anchors[0],xyz_m=[0,0,0]),self.anchors[1]],
                        [dict(self.anchors[0],source_point=True),self.anchors[1]],
                        [dict(self.anchors[0],source_point=-1),self.anchors[1]]):
            with self.subTest(anchors=anchors),self.assertRaises(ValueError):adjustment.checked_anchors(anchors)

    def test_detached_exact_source_points(self):
        before=deepcopy((self.base.gap_report,self.source.ordered_items))
        result=adjustment._segments(self.base,self.source,self.anchors)
        self.assertEqual(result[0].points[1]['xyz_m'],[.0011,0.,1.])
        self.assertEqual(result[0].points[2]['xyz_m'],[.0021,0.,1.])
        self.assertEqual((self.base.gap_report,self.source.ordered_items),before)
        result[0].points[1]['xyz_m'][0]=9
        self.assertEqual(self.source.ordered_items[0].points[1]['xyz_m'][0],.0011)

    def test_endpoints_unknown_indexes_and_source_roles_rejected(self):
        for key,value in (('base_point',0),('base_point',3),('base_segment',9),('source_point',99),('source_segment',3)):
            anchors=deepcopy(self.anchors);anchors[0][key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):adjustment._segments(self.base,self.source,anchors)
        for key,value in (('role','connector'),('row_index',9),('xyz_m',[.01,0,1])):
            source=deepcopy(self.source);source.ordered_items[0].points[1][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):adjustment._segments(self.base,source,self.anchors)
        source=deepcopy(self.source);source.ordered_items[0]=replace(source.ordered_items[0],block_index=1)
        with self.assertRaises(ValueError):adjustment._segments(self.base,source,self.anchors)

    def test_algorithm_registration_is_bounded(self):
        policy=dict(schema=adjustment.POLICY,base_sha256='a'*64,source_sha256='b'*64,anchors=self.anchors)
        algorithms=dict(revisions.RESEARCH_SHIFT_DEEP_RELOCATION_ALGORITHMS,row_shifts_sha256='c'*64,
                        route=adjustment.VERSION,source_adjustment=policy)
        self.assertTrue(revisions.supported_target_algorithms(algorithms))
        for key,value in (('routing','unknown'),('row_shifts_sha256','bad'),('extra',True)):
            self.assertFalse(revisions.supported_target_algorithms(dict(algorithms,**{key:value})))
        broken=deepcopy(algorithms);broken['source_adjustment']['anchors'][0]['source_point']=True
        self.assertFalse(revisions.supported_target_algorithms(broken))

    def test_relocation_excludes_only_display_and_file_paths(self):
        payload=dict(schema=SCHEMA,source=dict(geometry_sha256='a',region_id='b',region_signature='c',region_manifest='old'),
                     ordered_items=[dict(xyz_m=[1,2,3])],data_layers={'path':'old'},route_diagram_status='old')
        relocated=deepcopy(payload);relocated['source']['region_manifest']='new';relocated['data_layers']={'path':'new'}
        self.assertEqual(adjustment.semantic(payload),adjustment.semantic(relocated))
        relocated['ordered_items'][0]['xyz_m'][0]=7
        self.assertNotEqual(adjustment.semantic(payload),adjustment.semantic(relocated))
        nested=dict(schema=SCHEMA,source=payload['source'],gap_report={'base':payload})
        moved=deepcopy(nested);moved['gap_report']['base']['source']['region_manifest']='moved'
        self.assertEqual(adjustment.semantic(nested),adjustment.semantic(moved))

    def test_nested_adjustment_fails_before_loading(self):
        trajectory=SimpleNamespace(gap_report={adjustment.KEY:{'base_trajectory':{'planner':{'algorithm_version':adjustment.VERSION}}}})
        with self.assertRaisesRegex(ValueError,'嵌套'):adjustment.validate(trajectory)

    def test_product_notice_does_not_claim_coverage(self):
        params={'source_adjustment':{},'continuous_routing':'anything'}
        self.assertEqual(current_algorithm_version(params),adjustment.VERSION)
        self.assertIn('独立核验',product_algorithm_notice(params,adjustment.VERSION))

    def test_workflow_creates_separate_plan_without_old_evidence(self):
        from femto.planning import source_adjustment_workflow as workflow,surface_workspace
        base=dict(schema=surface_workspace.SCHEMA,coverage_evidence={'old':'proof'},plan_binding={'old':'binding'},
                  treatment_head={'width':5},planning_context={'old':'context'})
        before=deepcopy(base);trajectory=SimpleNamespace(to_dict=lambda:{'new':'trajectory'})
        projected={'schema':'player-schema','planning_context':{'new':'context'}}
        with patch.object(workflow,'to_player_plan',return_value=projected),patch.object(workflow.planning_state,'binding',return_value={'new':'binding'}):
            result=workflow._plan(base,trajectory,None,None)
        self.assertEqual(result['schema'],surface_workspace.SCHEMA)
        self.assertEqual(result['planning_context'],{'new':'context'})
        self.assertEqual(result['plan_binding'],{'new':'binding'})
        self.assertNotIn('coverage_evidence',result)
        self.assertEqual(base,before)

    def test_workbench_summary_keeps_new_coverage_unknown(self):
        from femto.planning.surface_workspace import continuity_text
        plan={'surface_trajectory':{'planner':{'algorithm_version':adjustment.VERSION},
              'statistics':{'block_count':2,'segment_count':2,'gap_count':1,'inter_block_gap_count':1}}}
        text=continuity_text(plan)
        self.assertIn('块内断点 0 处',text)
        self.assertIn('新路线覆盖尚未核验',text)


if __name__=='__main__':unittest.main()
