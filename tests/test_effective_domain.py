"""Four-stage planning uses explicit derived domains without losing original intent."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import open3d as o3d

from test_fixed_rgb_experience import make_model,K
from test_mesh_slice_planner import grid_mesh
from femto.annotation.fixed_rgb_geometry import FixedProjection
from femto.planning import effective_domain as domain, quality_constrained as quality, surface_workspace as workspace
from femto.planning.treatment_head import head_parameters
from femto.planning.surface_trajectory import ORDERED_SELECTION_ALGORITHM, SurfaceTrajectory, to_player_plan
from femto.planning.trajectory_export import load_trajectory


class EffectiveDomainTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        mesh=grid_mesh(width=.12,height=.08,cell=.005)
        self.model=make_model(self.temp.name,o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),
            o3d.utility.Vector3iVector(mesh.faces)))
        self.model.view_reference['undistorted_intrinsic']=K.tolist()
        self.model.store.origin_reference={'kind':'initial_color_camera','provenance':'synthetic_test'}
        self.region=self.model.region([[30,30],[70,30],[70,70],[30,70]],'synthetic','',
            selection={'mask':self.model.projection.valid.copy(),'method':'work_only'})
        self.mesh=workspace.canonical_region_mesh(self.model.mesh,self.region)
        self.head=head_parameters('direct_row_spacing',row_spacing_mm=7)

    def build(self):
        record=quality.evidence(self.model,self.region,allow_empty=True)
        return domain.build(self.mesh,self.region,record,5),record

    def missing(self):
        depth=self.model.projection.depth.copy();depth[50,50]=0
        self.model.projection=FixedProjection(self.model.mesh,K,depth)

    def test_whole_face_domain_excludes_missing_pixel_without_erasing_intent(self):
        before=copy.deepcopy(self.region)
        clean,_=self.build();self.missing();result,record=self.build()
        self.assertLess(result['eligible_face_count'],clean['eligible_face_count'])
        self.assertEqual(result['pending_pixel_count'],1)
        self.assertEqual(result['missing_regions'][0]['quality_class'],7)
        self.assertEqual(result['missing_regions'][0]['pixel_bounds'],[50,50,50,50])
        self.assertEqual(self.region,before)
        self.assertTrue(domain.validate(result,record,5,self.mesh,self.region).any())

    def test_domain_binding_cannot_be_forged_with_a_new_hash(self):
        result,record=self.build()
        damaged=copy.deepcopy(result);damaged['face_class_runs']=[0,result['face_count']]
        damaged['eligible_face_count']=result['face_count'];damaged['excluded_face_count']=0
        damaged['sha256']=quality._hash(damaged)
        with self.assertRaises(ValueError):domain.validate(damaged,record,5,self.mesh,self.region)
        with self.assertRaises(ValueError):domain.validate(result,record,10,self.mesh,self.region)
        changed=self.mesh.copy();changed.vertices[0,0]+=.001
        with self.assertRaises(ValueError):domain.validate(result,record,5,changed,self.region)

    def test_missing_components_are_separate_by_reason_and_location(self):
        classes=np.zeros((8,8),np.uint8)
        classes[1,1]=7;classes[1,2]=7;classes[5,5]=7;classes[2,2]=5
        items=domain.missing_regions(classes)
        self.assertEqual(len(items),3)
        self.assertEqual(sum(i['pixel_count'] for i in items),4)
        self.assertEqual([i['quality_class'] for i in items],[5,7,7])
        self.assertTrue(all(i['keeps_coverage_requirement'] for i in items))

    def test_all_bad_geometry_leaves_report_and_no_valid_route(self):
        self.model.projection=FixedProjection(self.model.mesh,K,np.zeros((100,100),np.uint16))
        before=copy.deepcopy(self.region)
        with self.assertRaisesRegex(ValueError,'没有通过质量检查'):
            workspace.generate(self.model,self.region,self.head)
        reports=list((self.model.reconstruction/'planning_inputs').rglob('planning_assessment.json'))
        self.assertEqual(len(reports),1)
        report=json.loads(reports[0].read_text(encoding='utf-8'))
        self.assertEqual(report['status'],'blocked_no_supported_faces')
        self.assertEqual(report['effective_domain']['eligible_face_count'],0)
        self.assertFalse(report['execution_ready']);self.assertEqual(self.region,before)

    def test_candidate_failure_records_blocked_state_without_changing_intent(self):
        from femto.planning.mesh_slice_planner import Unplannable
        before=copy.deepcopy(self.region)
        with patch('femto.planning.continuous_quality.candidate',
                side_effect=Unplannable('no_scan_lines','候选路线无法生成')):
            with self.assertRaisesRegex(ValueError,'检查报告'):
                workspace.generate(self.model,self.region,self.head)
        report=json.loads(next((self.model.reconstruction/'planning_inputs').rglob('planning_assessment.json')).read_text(encoding='utf-8'))
        self.assertEqual(report['status'],'blocked_by_candidate_planner')
        self.assertEqual(report['candidate_reason'],'no_scan_lines')
        self.assertFalse(report['execution_ready'])
        self.assertEqual(self.region,before)

    def test_all_width_rejected_still_records_why(self):
        head=head_parameters('direct_row_spacing',row_spacing_mm=7,effective_treatment_width_mm=100)
        with self.assertRaisesRegex(ValueError,'无可保留路线'):workspace.generate(self.model,self.region,head)
        report=json.loads(next((self.model.reconstruction/'planning_inputs').rglob('planning_assessment.json')).read_text(encoding='utf-8'))
        self.assertEqual(report['status'],'blocked_by_route_constraints')
        self.assertEqual(report['route_report']['retained_edges'],0)
        self.assertGreater(report['route_report']['rejected_edges'],0)

    def test_v2_archive_and_domain_reproduce_exactly(self):
        self.missing()
        plan=self.legacy_plan(with_domain=True)
        payload=plan['surface_trajectory'];record=payload['planner']['parameters']['effective_domain']
        self.assertEqual(payload['planner']['algorithm_version'],quality.DOMAIN_ALGORITHM)
        allowed=domain.validate(record,payload['planner']['parameters']['quality_evidence'],5,self.mesh,self.region)
        self.assertTrue(all(allowed[p['triangle_index']] for s in plan['segments'] for p in s))
        folder=workspace.export(self.model,self.region,plan)
        self.assertTrue((folder/'effective_domain.json').is_file())
        reopened=load_trajectory(folder)
        self.assertEqual(to_player_plan(reopened)['segments'],plan['segments'])
        self.assertEqual(reopened.planner['parameters']['effective_domain'],record)
        self.assertEqual(reopened.gap_report['missing_regions'],record['missing_regions'])

    def test_rejected_range_and_missing_report_tampering_rejected_without_replanning(self):
        self.missing();plan=workspace.generate(self.model,self.region,self.head)
        bad=copy.deepcopy(plan['surface_trajectory'])
        bad['gap_report']['quality_constraint']['rejected_ranges']=[]
        with self.assertRaisesRegex(ValueError,'排除范围'):SurfaceTrajectory.from_dict(bad)
        bad=copy.deepcopy(plan['surface_trajectory']);bad['gap_report']['missing_regions']=[]
        with self.assertRaisesRegex(ValueError,'缺失处理'):SurfaceTrajectory.from_dict(bad)

    def legacy_plan(self, *, with_domain=False):
        from femto.planning.mesh_slice_planner import plan_mesh_slice
        from femto.planning.planning_state import binding,region_signature
        manifest=workspace.snapshot(self.model,self.region,lambda:None)
        candidate=plan_mesh_slice(manifest,row_spacing_mm=7,connect_rows=True,block_planning=True,
            selection_blocks=True,selection_algorithm=ORDERED_SELECTION_ALGORITHM)
        record=quality.evidence(self.model,self.region)
        effective=domain.build(self.mesh,self.region,record,5) if with_domain else None
        old=quality.constrain(candidate,self.mesh,self.region,record,5,domain=effective)
        plan=to_player_plan(old)
        plan.update(schema=workspace.SCHEMA,treatment_head=self.head,surface_trajectory=old.to_dict(),
            region_id=self.region['annotation_id'],region_signature=region_signature(self.region),coordinate_frame='world_rdf',unit='metre')
        plan['plan_binding']=binding(self.model.mesh,self.region,plan)
        return plan

    def test_old_v1_still_replays_and_reproduces_without_domain(self):
        plan=self.legacy_plan()
        self.assertEqual(plan['surface_trajectory']['planner']['algorithm_version'],quality.ALGORITHM)
        folder=workspace.export(self.model,self.region,plan)
        reopened=load_trajectory(folder)
        self.assertEqual(reopened.planner['algorithm_version'],quality.ALGORITHM)
        self.assertNotIn('effective_domain',reopened.planner['parameters'])
        self.assertEqual(to_player_plan(reopened)['segments'],plan['segments'])

    def test_current_v3_preserves_domain_and_reproduces_verified_connections(self):
        from femto.planning.continuous_quality import ALGORITHM, LAZY_ROUTING
        self.missing()
        before=copy.deepcopy(self.region)
        plan=workspace.generate(self.model,self.region,self.head)
        payload=plan['surface_trajectory'];params=payload['planner']['parameters']
        self.assertEqual(payload['planner']['algorithm_version'],ALGORITHM)
        self.assertEqual(params['continuous_routing'],LAZY_ROUTING)
        self.assertTrue(params['preserve_face_crossings'])
        record=params['effective_domain']
        allowed=domain.validate(record,params['quality_evidence'],5,self.mesh,self.region)
        self.assertTrue(all(allowed[p['triangle_index']] for s in plan['segments'] for p in s))
        reconnection=payload['gap_report']['reconnection']
        self.assertGreater(reconnection['before_chains'],reconnection['after_chains'])
        self.assertGreater(reconnection['added_connector_edges'],0)
        self.assertFalse(reconnection['failed_search_is_proof_of_disconnection'])
        folder=workspace.export(self.model,self.region,plan)
        self.assertTrue((folder/'effective_domain.json').is_file())
        reopened=load_trajectory(folder)
        self.assertEqual(reopened.to_dict(),load_trajectory(folder,reproduce=False).to_dict())
        # Export deliberately creates a fresh immutable source snapshot and
        # records diagram availability; all route/evidence fields stay exact.
        expected=copy.deepcopy(payload)
        for key in ('region_manifest','region_export_dir'):
            self.assertNotEqual(reopened.source[key],payload['source'][key])
            self.assertTrue(Path(reopened.source[key]).exists())
            expected['source'][key]=reopened.source[key]
        self.assertEqual(Path(reopened.source['region_manifest']).parent,Path(reopened.source['region_export_dir']))
        expected['route_diagram_status']={'reason':"'source_dataset'",'status':'unavailable'}
        self.assertEqual(reopened.to_dict(),expected)
        self.assertEqual(to_player_plan(reopened)['segments'],plan['segments'])
        self.assertEqual(reopened.gap_report['missing_regions'],record['missing_regions'])
        self.assertEqual(self.region,before)

    def test_original_selection_gap_is_preserved_and_cannot_be_dropped(self):
        mask=self.model.projection.valid.copy();mask[:,48:53]=False
        self.region=self.model.region([[30,30],[70,30],[70,70],[30,70]],'split','',
            selection={'mask':mask,'method':'work_only'})
        self.mesh=workspace.canonical_region_mesh(self.model.mesh,self.region)
        plan=workspace.generate(self.model,self.region,self.head)
        original=plan['surface_trajectory']['gap_report']['candidate_gaps']
        self.assertEqual(len(original),1);self.assertEqual(original[0]['reason'],'selection_break')
        bad=copy.deepcopy(plan['surface_trajectory']);bad['gap_report']['candidate_gaps']=[]
        with self.assertRaisesRegex(ValueError,'原断点记录'):SurfaceTrajectory.from_dict(bad)

    def test_evidence_semantics_cannot_be_changed_by_rehashing(self):
        record=quality.evidence(self.model,self.region)
        for key,value in [('depth_gate_mm',100),('residual_semantics','calibrated_error_bound'),('intent_sha256','wrong')]:
            bad=copy.deepcopy(record);bad[key]=value;bad['sha256']=quality._hash(bad)
            with self.assertRaises(ValueError):quality.checked_evidence(bad)


if __name__=='__main__':unittest.main()
