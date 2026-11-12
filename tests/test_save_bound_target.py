"""Portable target reuse stays bound to one verified object and save request."""
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import open3d as o3d

import test_surface_workspace as fixtures
from femto.operation_diagnostics import record_operation
from femto.planning import region_project, surface_workspace, target_surface
from femto.planning.save_validation import SaveValidationSession
from femto.planning.trajectory_export import load_trajectory


class BoundSaveTargetTests(unittest.TestCase):
    def setUp(self):
        case = fixtures.SurfaceWorkspaceTests()
        case.setUp()
        self.addCleanup(case.doCleanups)
        self.model = case.model
        o3d.io.write_triangle_mesh(self.model.store.source, self.model.mesh)
        self.region = case.region(True)
        self.model.store.regions = [self.region]
        self.plan = surface_workspace.generate(self.model, self.region, case.head, defer_target=True)
        # Generation writes the target asset; this fixture exercises a model
        # retaining that verified object, including mutation after entry checks.
        self.model.target_ledger = surface_workspace.target_for(self.model, self.plan)
        self.root = Path(case.temp.name)
        self.path = self.root/'bound.region.zip'

    def save(self, session, **kwargs):
        return region_project.save_project(self.model, self.region, self.path,
                                          self.plan, target_session=session, **kwargs)

    def test_one_portable_target_check_keeps_independent_asset_readbacks(self):
        expected = deepcopy(self.plan)
        ledger = self.model.target_ledger
        with record_operation('save_bound_target_test', root=self.root/'diagnostics') as trace:
            with SaveValidationSession() as session:
                native = surface_workspace.export(self.model, self.region, self.plan, target_session=session)
                with patch.object(surface_workspace, 'target_for', wraps=surface_workspace.target_for) as prepare, \
                        patch.object(target_surface, 'read', wraps=target_surface.read) as readback:
                    self.save(session)
                self.assertEqual(prepare.call_count, 1)
                self.assertEqual(readback.call_count, 1, 'Portable staged target must be read independently')
            self.assertIsNone(session._portable_target_id)
        saved = load_trajectory(native, reproduce=True)
        reopened = region_project.open_project(self.path, workspace=self.root/'independent_reopen')
        self.assertEqual(saved.to_dict()['ordered_items'], expected['surface_trajectory']['ordered_items'])
        self.assertEqual(reopened.saved_plan['segments'], expected['segments'])
        self.assertEqual(reopened.target_ledger.metadata, ledger.metadata)
        for name, value in ledger.arrays.items():
            np.testing.assert_array_equal(reopened.target_ledger.arrays[name], value)
        self.assertEqual(self.plan, expected)
        stages = trace.totals
        for name in ('save.bind_and_rehash_inputs', 'portable.reference_assets',
                     'portable.target_write_readback', 'portable.asset_hashes', 'portable.publish'):
            self.assertGreater(stages[name]['calls'], 0, name)

    def test_equivalent_replacement_object_cannot_consume_another_target_binding(self):
        original = region_project._save_project
        def replace(*args, **kwargs):
            ledger = kwargs['bound_target']
            kwargs['bound_target'] = target_surface.TargetLedger(deepcopy(ledger.metadata),
                {key: value.copy() for key, value in ledger.arrays.items()})
            return original(*args, **kwargs)
        with SaveValidationSession() as session:
            surface_workspace.export(self.model, self.region, self.plan, target_session=session)
            with patch.object(region_project, '_save_project', side_effect=replace):
                with self.assertRaisesRegex(ValueError, '不属于当前验证会话'):
                    self.save(session)
            self.assertEqual(session._state, 'closed')
            self.assertIsNone(session._portable_target_id)
        self.assertFalse(self.path.exists())

    def test_array_mutation_after_entry_check_is_detected_before_writing(self):
        original = self.model.store.normalized_region
        def mutate(region):
            value = original(region)
            ledger = self.model.target_ledger
            # The saved IDs remain unchanged; only actual target bytes change.
            array = ledger.arrays['pixel_status'].copy()
            array.flat[0] ^= 1
            ledger.arrays['pixel_status'] = array
            return value
        with SaveValidationSession() as session:
            surface_workspace.export(self.model, self.region, self.plan, target_session=session)
            with patch.object(self.model.store, 'normalized_region', side_effect=mutate):
                with self.assertRaisesRegex(ValueError, '原目标数组已改变'):
                    self.save(session)
            self.assertEqual(session._state, 'closed')
        self.assertFalse(self.path.exists())

    def test_cancel_between_entry_and_internal_writer_expires_binding(self):
        original = region_project._save_project
        def cancelled():
            raise RuntimeError('cancelled after target check')
        def cancel(*args, **kwargs):
            values = list(args)
            values[4] = cancelled
            return original(*values, **kwargs)
        with SaveValidationSession() as session:
            surface_workspace.export(self.model, self.region, self.plan, target_session=session)
            with patch.object(region_project, '_save_project', side_effect=cancel):
                with self.assertRaisesRegex(RuntimeError, 'cancelled after target check'):
                    self.save(session)
            self.assertEqual(session._state, 'closed')
            self.assertIsNone(session._portable_target_id)
        self.assertFalse(self.path.exists())

    def test_corrupt_portable_target_still_fails_independent_readback(self):
        original = target_surface.write
        def corrupt(folder, ledger, check=lambda: None):
            original(folder, ledger, check)
            (Path(folder)/target_surface.ASSET).write_bytes(b'corrupt target')
        with SaveValidationSession() as session:
            native = surface_workspace.export(self.model, self.region, self.plan, target_session=session)
            with patch.object(target_surface, 'write', side_effect=corrupt):
                with self.assertRaises(zipfile.BadZipFile):
                    self.save(session)
            self.assertEqual(session._state, 'closed')
        self.assertTrue((native/'trajectory.json').exists())
        self.assertFalse(self.path.exists())


if __name__ == '__main__':
    unittest.main()
