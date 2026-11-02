"""Same-revision product roundtrips and hostile coverage-package regressions."""
from copy import copy, deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import open3d as o3d

from test_fixed_rgb_experience import make_model, K
from test_mesh_slice_planner import grid_mesh
from test_coverage_certificate import reseal
from femto.planning import coverage_workflow as workflow, coverage_certificate as coverage, coverage_display
from femto.planning import surface_workspace, planning_state, region_project
from femto.planning.surface_trajectory import SurfaceTrajectory
from femto.planning.treatment_head import head_parameters
from femto.planning.trajectory_export import load_trajectory


def fixture(folder):
    mesh = grid_mesh(width=.12, height=.08, cell=.01)
    geometry = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices), o3d.utility.Vector3iVector(mesh.faces))
    model = make_model(folder, geometry)
    model.view_reference['undistorted_intrinsic'] = K.tolist()
    model.store.origin_reference = dict(kind='initial_color_camera', provenance='synthetic_test')
    o3d.io.write_triangle_mesh(model.store.source, model.mesh)
    region = model.region([[30,30],[70,30],[70,70],[30,70]], '覆盖测试', '',
        selection=dict(mask=model.projection.valid.copy(), method='work_only'))
    plan = surface_workspace.generate(model, region, head_parameters('direct_row_spacing', row_spacing_mm=7))
    certificate, ledger = workflow.calculate(model, region, plan, budget=coverage.CoverageBudget(max_depth=1))
    return model, region, plan, certificate, ledger


class CoverageWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.base = fixture(Path(cls.temp.name)/'source')

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def setUp(self):
        self.case = tempfile.TemporaryDirectory();self.addCleanup(self.case.cleanup)
        self.root = Path(self.case.name)
        model,region,plan,self.certificate,self.ledger = self.base
        self.model = copy(model);self.model.store = copy(model.store)
        self.region,self.original = deepcopy(region),deepcopy(plan)
        self.model.store.regions = [self.region]
        self.plan = workflow.attach(self.model,self.region,self.original,self.certificate,self.ledger)

    def save(self):
        return region_project.save_project(self.model,self.region,self.root/'plan.zip',self.plan)

    def read(self, path):return region_project.open_project(path,workspace=self.root/'opened')

    def rewrite(self, path, update):
        with zipfile.ZipFile(path) as archive:files={name:archive.read(name) for name in archive.namelist()}
        data=json.loads(files['project.json'])
        update(data,files)
        data['files']={name:hashlib.sha256(raw).hexdigest() for name,raw in files.items() if name!='project.json'}
        files['project.json']=json.dumps(data,ensure_ascii=False).encode()
        with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as archive:
            for name,raw in files.items():archive.writestr(name,raw)

    def test_attachment_preserves_route_p1_contract_and_execution_flags(self):
        self.assertEqual(self.original['surface_trajectory'],self.plan['surface_trajectory'])
        self.assertEqual(self.original['segments'],self.plan['segments'])
        self.assertEqual(set(self.plan['planning_context']),{'schema','revision','target'})
        self.assertIsNone(self.plan['planning_context']['target']['coverage_certificate'])
        self.assertEqual(self.plan['planning_context']['target']['target_completion'],'unverified')
        self.assertFalse(self.plan['coverage_evidence']['execution_ready'])
        self.assertFalse(self.plan['coverage_evidence']['physical_accuracy_verified'])
        planning_state.validate(self.plan,self.model.mesh,self.region)

    def test_old_record_stays_explicitly_unverified(self):
        self.assertIn('未进行此项核验',workflow.summaries(self.original)[2])
        region_project.save_project(self.model,self.region,self.root/'old.zip',self.original)
        model=self.read(self.root/'old.zip')
        self.assertNotIn(workflow.KEY,model.saved_plan)
        self.assertIsNone(model.coverage_certificate)

    def test_failed_display_preparation_preserves_previous_evidence_caches(self):
        previous=(self.model.coverage_certificate,self.model.target_ledger,self.model.coverage_display)
        with patch('femto.planning.coverage_display.build',side_effect=RuntimeError('display cancelled')):
            with self.assertRaisesRegex(RuntimeError,'display cancelled'):
                workflow.attach(self.model,self.region,self.original,copy(self.certificate),copy(self.ledger))
        for before,after in zip(previous,(self.model.coverage_certificate,self.model.target_ledger,self.model.coverage_display)):
            self.assertIs(before,after)
        self.assertIs(workflow.cached(self.model,self.plan),previous[0])

    def test_portable_roundtrip_replays_certificate_and_preserves_every_array(self):
        model=self.read(self.save())
        self.assertEqual(model.saved_plan['coverage_evidence'],self.certificate.metadata)
        self.assertEqual(model.saved_plan['surface_trajectory'],self.plan['surface_trajectory'])
        self.assertEqual(model.coverage_display.tasks.to_dict(),self.model.coverage_display.tasks.to_dict())
        self.assertEqual(model.target_ledger.metadata,self.ledger.metadata)
        for name,value in self.certificate.arrays.items():
            np.testing.assert_array_equal(model.coverage_certificate.arrays[name],value)
        self.assertEqual(model.coverage_display.certificate_sha256,self.certificate.metadata['sha256'])

    def test_source_relocation_and_display_metadata_do_not_change_numerical_route_binding(self):
        trajectory=SurfaceTrajectory.from_dict(self.plan['surface_trajectory'])
        from dataclasses import replace
        moved=replace(trajectory,source=dict(trajectory.source,region_manifest='elsewhere/region.json'),
                      route_diagram_status={'display_only':'new layout'})
        self.assertEqual(workflow.route_binding(trajectory),workflow.route_binding(moved))
        changed=deepcopy(self.plan);changed['surface_trajectory']['ordered_items'][0]['points'][0]['xyz_m'][0]+=.001
        with self.assertRaises(ValueError):workflow.check_binding(changed)

    def test_native_export_reloads_same_proof_after_source_snapshot_changes(self):
        self.model.reconstruction=self.root/'export';self.model.reconstruction.mkdir()
        folder=surface_workspace.export(self.model,self.region,self.plan)
        certificates=[]
        trajectory=load_trajectory(folder,reproduce=False,coverage_sink=certificates.append)
        self.assertEqual(len(certificates),1)
        self.assertEqual(certificates[0].metadata,self.certificate.metadata)
        self.assertNotEqual(trajectory.source,self.plan['surface_trajectory']['source'])
        self.assertEqual(trajectory.to_dict()['ordered_items'],self.plan['surface_trajectory']['ordered_items'])
        self.assertTrue((folder/'model_coverage_report.txt').is_file())
        self.assertEqual(load_trajectory(folder).to_dict(),trajectory.to_dict())

    def test_name_and_note_change_preserves_coverage_and_new_save_version(self):
        first=self.save();a=self.read(first)
        region=a.store.regions[0]
        region.update(name='名称修改',note='不改变原选区')
        second=region_project.save_project(a,region,self.root/'second.zip',a.saved_plan)
        b=self.read(second)
        self.assertEqual(b.saved_plan[workflow.KEY],self.certificate.metadata)
        self.assertNotEqual(a.parent_revision_id,b.parent_revision_id)
        with zipfile.ZipFile(first) as z:first_id=json.loads(z.read('project.json'))['revision_id']
        self.assertEqual(b.parent_revision_id,first_id)

    def test_portable_reexport_strictly_reproduces_route_with_relocated_source(self):
        model=self.read(self.save())
        saved=deepcopy(model.saved_plan)
        folder=surface_workspace.export(model,model.store.regions[0],model.saved_plan)
        certificates=[]
        trajectory=load_trajectory(folder,coverage_sink=certificates.append)
        manifest=json.loads(Path(trajectory.source['region_manifest']).read_text(encoding='utf-8'))
        self.assertEqual(trajectory.data_layers,manifest['data_layers'])
        self.assertNotEqual(trajectory.data_layers['source_mesh'],saved['surface_trajectory']['data_layers']['source_mesh'])
        self.assertEqual(trajectory.to_dict()['ordered_items'],saved['surface_trajectory']['ordered_items'])
        self.assertEqual(certificates[0].metadata,self.certificate.metadata)
        self.assertEqual(model.saved_plan,saved)

    def test_width_or_seed_role_change_cannot_reuse_coverage(self):
        for mutate in (
            lambda p:p['treatment_head'].update(effective_treatment_width_mm=10),
            lambda p:p['surface_trajectory']['ordered_items'][0]['points'][1].update(role='connector')):
            wrong=deepcopy(self.plan);mutate(wrong)
            with self.assertRaises(ValueError):workflow.check_binding(wrong)

    def test_missing_cache_cannot_publish_metadata_only(self):
        self.model.coverage_certificate=None
        with self.assertRaisesRegex(ValueError,'资产缺失'):self.save()
        self.assertFalse((self.root/'plan.zip').exists())

    def test_missing_asset_is_rejected_even_after_outer_hashes_are_rebuilt(self):
        path=self.save()
        self.rewrite(path,lambda data,files:files.pop('coverage_certificate.npz'))
        with self.assertRaisesRegex(ValueError,'覆盖证据'):self.read(path)

    def test_orphan_coverage_assets_are_not_silently_treated_as_legacy(self):
        path=self.save()
        self.rewrite(path,lambda data,files:data['plan'].pop(workflow.KEY))
        with self.assertRaisesRegex(ValueError,'覆盖证据'):self.read(path)

    def test_public_hash_resealing_cannot_forge_coverage(self):
        path=self.save()
        values={name:value.copy() for name,value in self.certificate.arrays.items()}
        index=int(np.flatnonzero(values['status'] != 1)[0]);values['status'][index]=1
        forged=reseal(self.certificate,arrays=values)
        stream=io.BytesIO();np.savez_compressed(stream,**values)
        def update(data,files):
            data['plan'][workflow.KEY]=forged.metadata
            data['plan']['plan_binding']=planning_state.binding(self.model.mesh,self.region,data['plan'])
            files['coverage_certificate.json']=json.dumps(forged.metadata).encode()
            files['coverage_certificate.npz']=stream.getvalue()
        self.rewrite(path,update)
        with self.assertRaisesRegex(ValueError,'无法重现'):self.read(path)

    def test_cancel_during_readback_does_not_replace_previous_saved_version(self):
        from femto.annotation.fixed_rgb_geometry import GeometryTaskCancelled
        path=self.save();before=path.read_bytes();revision=self.model.revision_id
        with patch('femto.planning.coverage_archive.read',side_effect=GeometryTaskCancelled('cancelled')):
            with self.assertRaises(GeometryTaskCancelled):
                region_project.save_project(self.model,self.region,path,self.plan)
        self.assertEqual(path.read_bytes(),before)
        self.assertEqual(self.model.revision_id,revision)
        self.assertFalse(any(p.name.startswith('.pending') for p in self.root.iterdir()))

    def test_display_preserves_leaf_count_and_partial_pixels_never_turn_green(self):
        display=self.model.coverage_display
        self.assertEqual(len(display.triangles),len(self.certificate.arrays['root']))
        self.assertFalse(display.triangles.flags.writeable)
        green=display.pixel_state==1
        self.assertTrue(np.all(display.pixel_bits[green]==(1<<1)))
        self.assertTrue(np.all(self.ledger.arrays['pixel_status'][green]==1))
        unknown=self.ledger.arrays['mask'] & (self.ledger.arrays['pixel_status']!=1)
        self.assertTrue(np.all(display.pixel_state[unknown]==8))
        for reason in range(1,9):
            rgba=display.rgba(reason)
            self.assertTrue(np.array_equal(rgba[:,:,3]>0,display.mask(reason)))

    def test_display_coordinates_are_derived_and_not_saved_as_new_geometry(self):
        display=coverage_display.build(self.ledger,self.certificate)
        self.assertEqual(display.triangles.shape,(self.certificate.summary['leaf_count'],3,3))
        self.assertTrue(np.isfinite(display.triangles).all())
        self.assertTrue(np.all(display.source_faces==self.ledger.arrays['source_face'][self.certificate.arrays['root']]))
        path=self.save()
        with zipfile.ZipFile(path) as archive:
            self.assertNotIn('coverage_mesh.ply',archive.namelist())
            self.assertNotIn('coverage_display.npz',archive.namelist())
