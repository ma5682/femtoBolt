import unittest
import json
from pathlib import Path
import tempfile
from unittest import mock
import numpy as np
import open3d as o3d
from femto.local.local_pose_refinement import approve,stable_shape,solve,metrics,refine


class RefinementTests(unittest.TestCase):
    def scores(self,d):
        return {k:dict(median_m=d,p90_m=d*2,overlap=.9,color_error=.03) for k in ('forward','reverse')}
    def test_reject_large_correction_workspace_and_color_regression(self):
        initial=np.eye(4);candidate=np.eye(4);candidate[0,3]=.006
        reasons=approve(initial,candidate,np.linalg.inv(candidate),self.scores(.004),self.scores(.002))
        self.assertIn('correction_exceeds_5mm_or_1deg',reasons)
        candidate[0,3]=.002
        after=self.scores(.002);after['reverse']['color_error']=.2
        reasons=approve(initial,candidate,np.linalg.inv(candidate),self.scores(.004),after,.001)
        self.assertIn('outside_workspace',reasons);self.assertIn('color_alignment_degraded',reasons)
    def test_no_improvement_and_nonfinite_rejected(self):
        self.assertIn('insufficient_heldout_improvement',approve(np.eye(4),np.eye(4),np.eye(4),self.scores(.002),self.scores(.002)))
        after=self.scores(.001);after['forward']['median_m']=float('nan')
        self.assertIn('missing_validation',approve(np.eye(4),np.eye(4),np.eye(4),self.scores(.002),after))
    def test_plane_is_not_sufficient_structure(self):
        rng=np.random.default_rng(1);points=rng.random((300,3));points[:,2]=.5
        c=o3d.geometry.PointCloud();c.points=o3d.utility.Vector3dVector(points)
        self.assertFalse(stable_shape(c))
    def test_real_icp_recovers_small_error_without_mutating_input(self):
        rng=np.random.default_rng(12)
        points=rng.uniform(-.1,.1,(1800,3));points[:,2]+=.6
        target=o3d.geometry.PointCloud();target.points=o3d.utility.Vector3dVector(points)
        target.colors=o3d.utility.Vector3dVector(np.tile([.3,.5,.7],(len(points),1)))
        source=o3d.geometry.PointCloud(target);source.translate([.002,0,0])
        before=np.asarray(source.points).copy();t=solve(source,target,np.eye(4))
        np.testing.assert_allclose(t[:3,3],[-.002,0,0],atol=.0005)
        np.testing.assert_array_equal(before,np.asarray(source.points))
        self.assertLess(metrics(source,target,t)['median_m'],.0005)

    def test_derived_export_preserves_reference_and_raw_and_fallback_publishes_no_dataset(self):
        from femto.pipeline.synthetic_scene_demo import generate_dataset
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);source=generate_dataset(root/'raw',frames=4,width=128,height=96)
            rows=[json.loads(s) for s in (source/'frames.jsonl').read_text().splitlines()]
            origin=np.asarray(json.loads((source/rows[0]['pose']).read_text())['T_WC'])
            for row in rows:
                path=source/row['pose'];p=json.loads(path.read_text());p['T_WC']=(np.linalg.inv(origin)@p['T_WC']).tolist()
                path.write_text(json.dumps(p))
            path=source/'manifest.json';m=json.loads(path.read_text());m['local_capture']={'measured_only':True};path.write_text(json.dumps(m))
            (source/'pose_quality_report.json').write_text('{}')
            before={p:p.read_bytes() for p in source.rglob('*') if p.is_file()}
            with mock.patch('femto.local.local_pose_refinement.stable_shape',return_value=True),mock.patch('femto.local.local_pose_refinement.approve',return_value=[]):
                output,report=refine(source,root/'accepted')
            self.assertTrue(report['accepted'])
            self.assertEqual(report['adoption_policy'],'all_views_must_pass; one rejected frame keeps the entire online pose set')
            self.assertIn('noise_floor',report['frames'][0])
            self.assertIn('correction_mm',report['frames'][0])
            self.assertIn('only_correction_limit_rejected',report['frames'][0])
            poses=[json.loads((output/r['pose']).read_text()) for r in [json.loads(s) for s in (output/'frames.jsonl').read_text().splitlines()]]
            np.testing.assert_allclose(poses[0]['T_WC'],np.eye(4),atol=1e-9)
            self.assertEqual(poses[1]['pose_source'],'offline_robust_icp_research_grade')
            self.assertEqual(before,{p:p.read_bytes() for p in source.rglob('*') if p.is_file()})
            with mock.patch('femto.local.local_pose_refinement.stable_shape',return_value=False):
                output,report=refine(source,root/'rejected')
            self.assertFalse(report['accepted']);self.assertEqual(output,source.resolve())
            self.assertFalse((root/'rejected/dataset').exists())
