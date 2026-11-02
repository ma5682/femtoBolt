"""Sequential RGB-D reference proof, rejection paths and portable history."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import open3d as o3d

from femto.annotation.odometry_reference import sequential_reference, POSE_SOURCE, PROVENANCE
from femto.annotation.fixed_rgb_geometry import FixedRGBModel
from femto.annotation.scene_annotations import AnnotationStore
from femto.dataset.local_reference import CALIBRATION_FILES
from femto.geometry.rgbd_frames import distortion_maps
from femto.io_utils import imwrite_unicode
from femto.planning.region_project import save_project, open_project


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def fixture(root):
    raw = root/'raw'; dataset = raw/'derived'/'posed'; rec = root/'result'/'reconstruction'
    rec.mkdir(parents=True)
    source = rec/'scene_tsdf_mesh_m.ply'
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(
        [[-.3,-.3,1],[.3,-.3,1],[.3,.3,1],[-.3,.3,1]]),
        o3d.utility.Vector3iVector([[0,1,2],[0,2,3]]))
    o3d.io.write_triangle_mesh(str(source), mesh)
    intrinsic = dict(width=100,height=100,fx=100.,fy=100.,cx=50.,cy=50.)
    distortion = dict(rgb=dict(model='BROWN_CONRADY_K6',k1=0,k2=0,k3=0,k4=0,k5=0,k6=0,p1=0,p2=0))
    for base in (raw,dataset):
        for name in CALIBRATION_FILES:
            write(base/'calibration'/name, intrinsic if name=='color_intrinsic.json' else distortion if name=='distortion.json' else {})
    original = dict(schema='femto-full-scene-rgbd-v1',status='captured_pose_pending',frame_count=3,
                    parameters=dict(fixed_camera=False))
    write(raw/'manifest.json', original)
    write(dataset/'manifest.json',dict(original,status='derived_pose_estimated',source_dataset=str(raw),
                                      pose=dict(valid=True,source=POSE_SOURCE)))
    raw_rows=[]; selected_rows=[]; poses=[]
    for ordinal in range(3):
        (raw/'rgb').mkdir(exist_ok=True);(raw/'depth').mkdir(exist_ok=True)
        imwrite_unicode(raw/f'rgb/{ordinal}.png',np.full((100,100,3),120,np.uint8))
        imwrite_unicode(raw/f'depth/{ordinal}.png',np.full((100,100),1000,np.uint16))
        row=dict(frame=ordinal+5,rgb=f'rgb/{ordinal}.png',depth=f'depth/{ordinal}.png',pose=f'pose/{ordinal}.json',
                 pose_valid=False,camera_timestamp_us=1000+ordinal*100)
        raw_rows.append(row)
        write(raw/row['pose'],dict(pose_valid=False,T_WC=None))
        selected_rows.append(dict(row,rgb='../../'+row['rgb'],depth='../../'+row['depth'],pose_valid=True))
        pose=np.eye(4);pose[0,3]=.02*ordinal;poses.append(pose)
        write(dataset/row['pose'],dict(schema='femto-scene-frame-pose-v1',frame=row['frame'],pose_valid=True,
              T_WC=pose.tolist(),pose_source=POSE_SOURCE,coordinate_system='+X right, +Y down, +Z forward; metres',source_dataset=str(raw)))
    for base, frames in ((raw,raw_rows),(dataset,selected_rows)):
        (base/'frames.jsonl').write_text('\n'.join(json.dumps(r) for r in frames)+'\n',encoding='utf-8')
    pairs=[]
    for i in range(2):
        pairs.append(dict(source_frame=raw_rows[i]['frame'],target_frame=raw_rows[i+1]['frame'],
                     source_to_target=(np.linalg.inv(poses[i+1])@poses[i]).tolist(),
                     cycle_consistency=dict(passed=True),fitness=.8,reverse_fitness=.8,inlier_rmse_m=.001))
    write(dataset/'pose_quality_report.json',dict(schema='femto-rgbd-odometry-report-v1',
          status='estimated_research_grade',source_dataset=str(raw),output_dataset=str(dataset),
          source_frame_count=3,frame_count=3,accepted_frame_ordinals=[0,1,2],pairs=pairs,
          parameters=dict(min_fitness=.15,max_rmse_m=.06),
          keyframe_selection=dict(tracked_frame_ordinals=[0,1,2],selected_frame_ordinals=[0,1,2]),
          frame_quality=[dict(frame_ordinal=i,usable=True) for i in range(3)]))
    write(dataset/'quality/validation_report.json',dict(schema='femto-scene-validation-v1',dataset=str(dataset),
          ready_for_multiview_reconstruction=True,errors=[],metrics=dict(valid_pose_count=3)))
    write(rec/'reconstruction_report.json',dict(schema='femto-scene-reconstruction-v1',
          status='complete_geometry_gaussian_init_untrained',source_dataset=str(dataset),frame_count=3,
          outputs=dict(mesh=dict(ply=str(source))),parameters=dict(depth_min_m=.5,depth_max_m=3.86,
          distortion_remap=dict(new_K=distortion_maps(dataset)[2].tolist()))))
    write(rec.parent/'demo_summary.json',dict(schema='femto-complete-demo-v1',
          status='pass_geometry_gaussian_init_untrained',data_kind='real_femto',pose_stage=POSE_SOURCE,
          fixed_camera=False,raw_dataset=str(raw),source_dataset=str(dataset),reconstruction=str(rec),frame_count=3))
    return raw,dataset,rec,mesh


class SequentialReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.raw,self.dataset,self.rec,self.mesh=fixture(self.root)
        self.source=self.rec/'scene_tsdf_mesh_m.ply'

    def snapshot(self):
        return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in self.root.rglob('*') if p.is_file()}

    def mutate(self,path,change):
        record=json.loads(path.read_text(encoding='utf-8'));original=deepcopy(record)
        change(record);write(path,record)
        return original

    def test_load_uses_verified_first_tracked_frame_without_rebasing_or_local_labels(self):
        before=self.snapshot()
        model=FixedRGBModel(self.rec)
        self.assertEqual(model.store.origin_reference['provenance'],PROVENANCE)
        self.assertEqual(model.view_reference['reference_frame_id'],5)
        self.assertIsNone(model.store.local_context)
        self.assertTrue(model.projection.valid.any())
        np.testing.assert_array_equal(np.asarray(model.mesh.vertices),np.asarray(self.mesh.vertices))
        self.assertEqual(model.max_radius_mm,40.)
        self.assertEqual(before,self.snapshot())
        # Standalone old annotations retain model-origin semantics.
        self.assertEqual(AnnotationStore(self.mesh,self.source).origin_reference['provenance'],'unknown')

    def test_changed_pose_and_chain_are_rejected(self):
        path=self.dataset/'pose/1.json'
        original=self.mutate(path,lambda d:d['T_WC'][0].__setitem__(3,.2))
        with self.assertRaisesRegex(ValueError,'位姿'):sequential_reference(self.source)
        write(path,original)
        path=self.dataset/'pose_quality_report.json'
        for change in (lambda d:d['pairs'][0]['cycle_consistency'].update(passed=False),
                       lambda d:d['pairs'][0].update(target_frame=99),
                       lambda d:d['pairs'][0].update(inlier_rmse_m=float('nan')),
                       lambda d:d['keyframe_selection'].update(selected_frame_ordinals=[False,1,2])):
            original=self.mutate(path,change)
            with self.assertRaises(ValueError):sequential_reference(self.source)
            write(path,original)

    def test_calibration_mismatch_or_missing_is_not_accepted(self):
        path=self.dataset/'calibration/color_intrinsic.json'
        original=self.mutate(path,lambda d:d.update(fx=120))
        with self.assertRaisesRegex(ValueError,'标定'):sequential_reference(self.source)
        write(path,original)
        (self.raw/'calibration/depth_to_color.json').unlink()
        with self.assertRaises(ValueError):FixedRGBModel(self.rec)

    def test_changed_both_calibrations_still_must_match_reconstruction(self):
        for base in (self.raw,self.dataset):
            self.mutate(base/'calibration/color_intrinsic.json',lambda d:d.update(fx=120))
        with self.assertRaisesRegex(ValueError,'生成模型时的标定'):FixedRGBModel(self.rec)

    def test_source_report_failure_and_reference_asset_mismatch_are_rejected(self):
        path=self.dataset/'quality/validation_report.json'
        original=self.mutate(path,lambda d:d.update(ready_for_multiview_reconstruction=False))
        with self.assertRaises(ValueError):sequential_reference(self.source)
        write(path,original)
        path=self.rec.parent/'demo_summary.json'
        original=self.mutate(path,lambda d:d.update(raw_dataset=str(self.root/'other')))
        with self.assertRaises(ValueError):sequential_reference(self.source)
        write(path,original)
        path=self.dataset/'frames.jsonl'
        frames=[json.loads(line) for line in path.read_text().splitlines()]
        frames[0]['rgb']='../../rgb/1.png'
        path.write_text('\n'.join(json.dumps(row) for row in frames),encoding='utf-8')
        with self.assertRaises(ValueError):sequential_reference(self.source)

    def test_damaged_local_capture_never_falls_back_to_sequential_proof(self):
        write(self.rec/'local_geometry.json',{})
        self.assertIsNone(sequential_reference(self.source))
        with self.assertRaisesRegex(ValueError,'origin cannot be verified'):FixedRGBModel(self.rec)

    def test_source_changed_during_model_load_is_rejected(self):
        from femto.annotation.fixed_rgb_geometry import FixedProjection
        def changing_projection(*args,**kwargs):
            result=FixedProjection(*args,**kwargs)
            path=self.dataset/'pose/1.json'
            path.write_bytes(path.read_bytes()+b' ')
            return result
        with patch('femto.annotation.fixed_rgb_geometry.FixedProjection',side_effect=changing_projection):
            with self.assertRaisesRegex(ValueError,'读取期间'):FixedRGBModel(self.rec)

    def test_portable_region_restores_source_binding_and_preserves_original(self):
        model=FixedRGBModel(self.rec)
        region=model.store.add_region(dict(triangle_indices=[0,1],screen_polygon=[[20,20],[80,20],[80,80],[20,80]],
                                          view_reference=model.view_reference,name='物品测试夹具'))
        before=self.snapshot()
        archive=save_project(model,region,self.root/'object.region.zip')
        archive_hash=hashlib.sha256(archive.read_bytes()).hexdigest()
        loaded=open_project(archive,self.root/'replay')
        self.assertEqual(loaded.store.origin_reference,model.store.origin_reference)
        self.assertEqual(loaded.view_reference,model.view_reference)
        self.assertEqual(loaded.store.regions[0]['name'],'物品测试夹具')
        self.assertIsNone(loaded.store.local_context)
        np.testing.assert_array_equal(np.asarray(loaded.mesh.vertices),np.asarray(model.mesh.vertices))
        self.assertEqual(archive_hash,hashlib.sha256(archive.read_bytes()).hexdigest())
        after=self.snapshot()
        self.assertTrue(all(after[path]==value for path,value in before.items()))


if __name__=='__main__':unittest.main()
