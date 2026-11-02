"""Request-scoped target reuse must never authorize changed or later saves."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from test_coverage_workflow import fixture
from femto.app.qt_planning_io import clone_model
from femto.planning import coverage_workflow, region_project, surface_workspace, target_surface
from femto.planning.save_validation import SaveValidationSession


class SaveValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.base = fixture(Path(cls.temp.name)/'source')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.temp_case = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_case.cleanup)
        self.root = Path(self.temp_case.name)
        model, region, plan, certificate, ledger = self.base
        self.model = clone_model(model, freeze_inputs=True)
        self.region, self.plan = deepcopy(region), deepcopy(plan)
        self.model.store.regions = [self.region]
        self.model.target_ledger = target_surface.TargetLedger(deepcopy(ledger.metadata),
            {name: array.copy() for name, array in ledger.arrays.items()})
        self.certificate = certificate

    def native(self, session):
        return surface_workspace.export(self.model, self.region, self.plan, target_session=session)

    def portable(self, session, **kwargs):
        return region_project.save_project(self.model, self.region, self.root/'plan.zip',
                                          self.plan, target_session=session, **kwargs)

    def test_single_save_validates_route_once_and_independent_open_checks_again(self):
        from femto.planning import planning_state
        original=planning_state.validated_surface
        with patch.object(planning_state,'validated_surface',wraps=original) as validated, \
             patch.object(planning_state,'validate',wraps=planning_state.validate) as validate:
            with SaveValidationSession() as session:
                self.native(session)
                path=self.portable(session)
            self.assertEqual(validated.call_count,1)
            self.assertEqual(validate.call_count,0)
            region_project.open_project(path,workspace=self.root/'reopened-route')
            self.assertGreaterEqual(validate.call_count,1)

    def test_plain_coverage_replays_once_per_save_and_on_independent_open(self):
        from femto.planning import coverage_certificate
        self.plan = coverage_workflow.attach(self.model,self.region,self.plan,self.certificate,self.model.target_ledger)
        with patch.object(coverage_certificate,'validate',wraps=coverage_certificate.validate) as replay:
            with SaveValidationSession() as session:
                self.native(session)
                count=replay.call_count
                self.assertGreater(count,0)
                path=self.portable(session)
                self.assertEqual(replay.call_count,count)
            region_project.open_project(path,workspace=self.root/'reopened-coverage')
            self.assertGreater(replay.call_count,count)

    def test_changed_coverage_array_cannot_reuse_native_proof(self):
        from femto.planning import coverage_certificate
        self.plan = coverage_workflow.attach(self.model,self.region,self.plan,self.certificate,self.model.target_ledger)
        with SaveValidationSession() as session:
            self.native(session)
            old=self.model.coverage_certificate
            arrays={key:value.copy() for key,value in old.arrays.items()}
            arrays['status'][0] = (int(arrays['status'][0])+1) % 8
            self.model.coverage_certificate=coverage_certificate.Certificate(deepcopy(old.metadata),arrays)
            with self.assertRaises(ValueError):self.portable(session)
        self.assertFalse((self.root/'plan.zip').exists())

    def test_coverage_asset_corruption_during_portable_write_is_rejected(self):
        import zipfile
        from femto.planning import coverage_archive
        self.plan = coverage_workflow.attach(self.model,self.region,self.plan,self.certificate,self.model.target_ledger)
        original=coverage_archive.write_assets
        with SaveValidationSession() as session:
            self.native(session)
            def corrupt(folder,certificate,check=lambda:None,**kwargs):
                original(folder,certificate,check,**kwargs)
                path=Path(folder)/coverage_archive.ASSET
                path.write_bytes(b'corrupt')
            with patch.object(coverage_archive,'write_assets',side_effect=corrupt):
                with self.assertRaises(zipfile.BadZipFile):self.portable(session)
        self.assertFalse((self.root/'plan.zip').exists())

    def test_route_mutation_between_validation_and_receipt_is_rejected(self):
        from femto.planning import planning_state
        original=planning_state.validated_surface
        def mutate_after_check(plan, mesh, region, check=lambda:None):
            trajectory=original(plan,mesh,region,check=check)
            plan['save_test_marker'] = 'changed after route validation'
            return trajectory
        with SaveValidationSession() as session:
            with patch.object(planning_state,'validated_surface',side_effect=mutate_after_check):
                with self.assertRaisesRegex(ValueError,'核验期间输入已变化'):
                    self.native(session)
            with self.assertRaises(ValueError):self.portable(session)
        self.assertFalse((self.root/'plan.zip').exists())

    def test_mutation_between_receipt_check_and_normalization_is_rejected(self):
        original=self.model.store.normalized_region
        def changed(region):
            result=original(region)
            result['note']='changed after receipt'
            return result
        with SaveValidationSession() as session:
            self.native(session)
            with patch.object(self.model.store,'normalized_region',side_effect=changed):
                with self.assertRaisesRegex(ValueError,'输入已变化'):
                    self.portable(session)
        self.assertFalse((self.root/'plan.zip').exists())

    def test_combined_save_replays_once_and_independent_open_replays_again(self):
        self.plan = coverage_workflow.attach(self.model, self.region, self.plan,
                                             self.certificate, self.model.target_ledger)
        with patch.object(target_surface, '_construct', wraps=target_surface._construct) as replay:
            with SaveValidationSession() as session:
                self.native(session)
                path = self.portable(session)
            self.assertEqual(replay.call_count, 1)
            opened = region_project.open_project(path, workspace=self.root/'opened')
            self.assertEqual(replay.call_count, 2)
        # Portable open rebinds source paths/data layers to its new workspace.
        for name in ('segments', 'planning_context', 'coverage_evidence', 'treatment_head'):
            self.assertEqual(opened.saved_plan[name], self.plan[name], name)
        for name, value in self.model.target_ledger.arrays.items():
            np.testing.assert_array_equal(opened.target_ledger.arrays[name], value)
        with self.assertRaises(ValueError):
            self.portable(session)

    def test_standalone_portable_keeps_full_target_replay(self):
        with patch.object(target_surface, '_construct', wraps=target_surface._construct) as replay:
            region_project.save_project(self.model, self.region, self.root/'standalone.zip', self.plan)
        self.assertEqual(replay.call_count, 1)

    def test_unpublished_and_closed_sessions_cannot_authorize_save(self):
        for close in (False, True):
            with SaveValidationSession() as session:
                if close: session.close()
                with self.assertRaises(ValueError):
                    self.portable(session)
            self.assertFalse((self.root/'plan.zip').exists())

    def test_cross_thread_use_invalidates_receipt(self):
        from concurrent.futures import ThreadPoolExecutor
        with SaveValidationSession() as session:
            self.native(session)
            with ThreadPoolExecutor(max_workers=1) as pool:
                with self.assertRaises(ValueError): pool.submit(self.portable, session).result()
            with self.assertRaises(ValueError): self.portable(session)

    def test_session_retains_only_digests_and_releases_them_on_exit(self):
        with SaveValidationSession() as session:
            self.native(session)
            self.assertIsInstance(session._binding, str)
            self.assertEqual(len(session._binding),64)
        self.assertIsNone(session._binding)
        with self.assertRaises(ValueError): session.__enter__()

    def test_failed_native_does_not_authorize_portable(self):
        with SaveValidationSession() as session:
            with patch('femto.planning.trajectory_export.export_trajectory', side_effect=OSError('disk failed')):
                with self.assertRaises(OSError): self.native(session)
            with self.assertRaises(ValueError): self.portable(session)

    def test_cancel_after_native_publication_retains_path_and_expires_receipt(self):
        from femto.annotation.fixed_rgb_geometry import GeometryTaskCancelled
        from femto.operation_diagnostics import record_operation
        from femto.planning import trajectory_export
        original_publish=trajectory_export.publish
        published=[]
        def publish(action, check, commit):
            path=original_publish(action,check,commit)
            published.append(path)
            return path
        def check():
            if published: raise GeometryTaskCancelled('cancel after native publication')
        with SaveValidationSession() as session:
            with record_operation('save_session_test') as trace:
                with patch.object(trajectory_export,'publish',side_effect=publish):
                    with self.assertRaisesRegex(GeometryTaskCancelled,'cancel after native publication'):
                        surface_workspace.export(self.model,self.region,self.plan,check,target_session=session)
            self.assertEqual(len(published),1)
            self.assertTrue((published[0]/'trajectory.json').is_file())
            self.assertTrue(any(Path(item.path)==published[0].resolve() for item in trace.summary().retained))
            with self.assertRaises(ValueError):self.portable(session)

    def test_failed_portable_publication_closes_receipt_and_preserves_native(self):
        with SaveValidationSession() as session:
            native=self.native(session)
            def failed_commit(action):raise OSError('injected publication failure')
            with self.assertRaisesRegex(OSError,'injected publication failure'):
                self.portable(session,commit=failed_commit)
            with self.assertRaises(ValueError):self.portable(session)
        self.assertTrue((native/'trajectory.json').is_file())
        self.assertFalse((self.root/'plan.zip').exists())

    def test_target_array_changed_without_metadata_is_rejected(self):
        with SaveValidationSession() as session:
            self.native(session)
            self.model.target_ledger.arrays['area_m2'][0] += .001
            with self.assertRaises(ValueError): self.portable(session)
        self.assertFalse((self.root/'plan.zip').exists())

    def test_resealed_target_with_new_matching_hashes_cannot_reuse_old_proof(self):
        with SaveValidationSession() as session:
            self.native(session)
            old=self.model.target_ledger
            arrays={name:value.copy() for name,value in old.arrays.items()}
            arrays['area_m2'][0]*=2
            forged=target_surface._seal(arrays,self.plan['planning_context']['revision'],
                target_surface.TargetBudget(**old.metadata['budget']),deepcopy(old.metadata['metrics']),
                old.metadata['quality_evidence_sha256'],old.metadata['effective_domain_sha256'])
            self.model.target_ledger=forged
            self.plan['planning_context']['target']=deepcopy(forged.metadata)
            # A fresh set of internally matching checksums is not the result
            # that the native publication actually reconstructed and proved.
            self.assertIs(surface_workspace.target_for(self.model,self.plan,reproduce=False),forged)
            with self.assertRaisesRegex(ValueError,'输入已变化'):
                self.portable(session)
        self.assertFalse((self.root/'plan.zip').exists())

    def test_changed_live_inputs_are_rejected(self):
        # Each mutation changes a genuine dependency while saved IDs stay intact.
        for name in ('rgb', 'depth', 'intrinsic', 'mesh', 'region', 'plan'):
            with self.subTest(name=name):
                model = clone_model(self.model, freeze_inputs=True)
                region, plan = deepcopy(self.region), deepcopy(self.plan)
                with SaveValidationSession() as session:
                    surface_workspace.export(model, region, plan, target_session=session)
                    if name == 'rgb':
                        model.rgb = model.rgb.copy(); model.rgb.flat[0] ^= 1
                    elif name == 'depth':
                        model.projection.depth = model.projection.depth.copy(); model.projection.depth.flat[0] += 1
                    elif name == 'intrinsic':
                        model.projection.k = model.projection.k.copy(); model.projection.k[0, 0] += .01
                    elif name == 'mesh': np.asarray(model.mesh.vertices)[0, 0] += .001
                    elif name == 'region': region['note'] = 'changed during save'
                    else: plan['segments'][0][0]['xyz_m'][0] += .001
                    with self.assertRaises(ValueError):
                        region_project.save_project(model, region, self.root/'plan.zip', plan, target_session=session)
                self.assertFalse((self.root/'plan.zip').exists())

    def test_changed_source_file_is_rejected(self):
        source = Path(self.model.store.source); raw = source.read_bytes()
        try:
            with SaveValidationSession() as session:
                self.native(session)
                source.write_bytes(raw+b'changed')
                with self.assertRaises(ValueError): self.portable(session)
        finally: source.write_bytes(raw)

    def test_mutation_during_archive_write_is_rejected_before_publication(self):
        with SaveValidationSession() as session:
            self.native(session)
            def progress(stage):
                if stage == '写入方案清单与压缩包': self.plan['segments'][0][0]['xyz_m'][0] += .001
            with self.assertRaises(ValueError): self.portable(session, progress=progress)
        self.assertFalse((self.root/'plan.zip').exists())

    def test_cancel_closes_receipt_and_fresh_request_can_retry(self):
        with SaveValidationSession() as session:
            native = self.native(session)
            def cancelled(): raise RuntimeError('injected cancellation')
            with self.assertRaisesRegex(RuntimeError, 'injected cancellation'):
                self.portable(session, check=cancelled)
            with self.assertRaises(ValueError): self.portable(session)
        self.assertTrue((native/'trajectory.json').is_file())
        with SaveValidationSession() as retry:
            self.native(retry)
            self.assertTrue(self.portable(retry).is_file())


if __name__ == '__main__': unittest.main()
