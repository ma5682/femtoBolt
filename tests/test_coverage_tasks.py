"""Evidence-derived tasks preserve original geometry and never grant approval."""
from copy import deepcopy
from dataclasses import FrozenInstanceError
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from femto.planning import coverage_tasks as tasks, coverage_workflow as workflow, coverage_certificate as coverage
from femto.planning.planning_revision import digest
from test_coverage_workflow import fixture
from test_coverage_certificate import reseal


class CoverageTaskTests(unittest.TestCase):
    def test_unknown_guidance_separates_capture_loss_from_unprocessed_targets(self):
        task = tasks.CoverageTask(8, 0, 100, None,
            (('no_mesh_surface', 2), ('budget_unprocessed', 90), ('numerically_unconfirmed', 8)))
        self.assertIn('无网格表面 2 像素', task.guidance)
        self.assertIn('目标建账预算未处理 90 像素', task.guidance)
        self.assertIn('不直接作为补采缺失', task.guidance)
        self.assertIn('数值与部分定位范围', task.guidance)
        budget_only = tasks.CoverageTask(8, 0, 90, None, (('budget_unprocessed', 90),))
        self.assertNotIn('确认实测缺失后补采', budget_only.guidance)
        self.assertEqual(tasks.CoverageTask(7, 1, 1, .001).guidance, tasks.GUIDANCE[7][2])

    def test_export_guidance_matches_the_display_guidance(self):
        view = tasks.CoverageTasks('target', 'target-sha', 'revision', 'certificate', 'route', False,
            (tasks.CoverageTask(8, 0, 2, None, (('no_mesh_surface', 2),)),))
        row = view.to_dict()['tasks'][0]
        self.assertEqual(row['guidance'], view.items[0].guidance)
        self.assertEqual(row['next_action'], 'inspect_reference')

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.model, cls.region, cls.original, cls.certificate, cls.ledger = fixture(Path(cls.temp.name)/'source')
        cls.plan = workflow.attach(cls.model, cls.region, cls.original, cls.certificate, cls.ledger)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def setUp(self):
        self.view = tasks.build(self.ledger, self.certificate)

    def test_area_partition_and_pixel_counts_are_derived_from_original_arrays(self):
        self.assertAlmostEqual(math.fsum(row.nominal_area_m2 for row in self.view.items[:7]),
                               self.certificate.summary['located_nominal_area_m2'], places=15)
        self.assertAlmostEqual(self.view.task(1).nominal_area_m2,
                               self.certificate.summary['confirmed_nominal_area_m2'], places=15)
        self.assertEqual(sum(row.leaf_count for row in self.view.items[:7]), len(self.certificate.arrays['root']))
        for row in self.view.items[:7]:
            mask = self.certificate.arrays['status'] == row.reason
            pixels = self.ledger.arrays['pixel_index'][self.certificate.arrays['root'][mask]]
            self.assertEqual(row.pixel_count, len(np.unique(pixels)))
        self.assertIsNone(self.view.task(8).nominal_area_m2)
        self.assertEqual(self.view.task(8).pixel_count, self.certificate.summary['unlocated_pixel_count'])
        self.assertEqual(sum(count for _, count in self.view.task(8).unknown_by_reason), self.view.task(8).pixel_count)

    def test_reason_mapping_keeps_refinement_separate_from_unknown_geometry(self):
        for reason in range(1,9):
            task = tasks.CoverageTask(reason, int(reason != 8), 1, None if reason == 8 else .001)
            action, _ = tasks.operation(task, tasks.FIXED_BASES[0])
            expected = {1:None, 2:'inspect_geometry', 3:'refine', 4:'recalculate', 5:'recalculate',
                        6:'inspect_target', 7:'recalculate', 8:'inspect_reference'}[reason]
            self.assertEqual(action, expected)
        self.assertIsNone(tasks.operation(tasks.CoverageTask(3,0,0,0.), tasks.FIXED_BASES[0])[0])
        self.assertIsNone(tasks.operation(None, tasks.FIXED_BASES[0])[0])

    def test_refined_and_variable_routes_resolve_to_explicit_restoration(self):
        for reason in (3,4,5,7):
            task = tasks.CoverageTask(reason, 1, 1, .001)
            for algorithm in tasks.REFINED:self.assertEqual(tasks.operation(task, algorithm)[0], 'restore')
        task = tasks.CoverageTask(3,1,1,.001)
        self.assertEqual(tasks.operation(task,tasks.VARIABLE)[0], 'refine')
        self.assertEqual(tasks.operation(tasks.CoverageTask(7,1,1,.001),tasks.VARIABLE)[0], 'recalculate')

    def test_budget_stop_remains_actionable_even_without_unprocessed_leaves(self):
        item=tasks.CoverageTask(7,0,0,0.,calculation_limits_reached=('proofs_budget',))
        self.assertTrue(item.present)
        self.assertEqual(tasks.operation(item,tasks.FIXED_BASES[0])[0],'recalculate')

    def test_report_is_deterministic_bounded_guidance_with_unknown_area_and_stable_ids(self):
        result = self.view.to_dict()
        self.assertEqual(result, tasks.build(self.ledger,self.certificate).to_dict())
        self.assertLess(len(json.dumps(result)), 20000)
        self.assertFalse(result['execution_ready']);self.assertFalse(result['physical_accuracy_verified'])
        self.assertFalse(result['imported_report_is_authority'])
        self.assertEqual(result['sha256'], digest({k:v for k,v in result.items() if k != 'sha256'}))
        self.assertEqual(len({row['task_id'] for row in result['tasks']}),len(result['tasks']))
        for row in result['tasks']:
            if row['reason'] == 8:
                self.assertIsNone(row['nominal_area_m2'])
                self.assertEqual(row['locator']['kind'],'original_target_pixels')
        with self.assertRaises(FrozenInstanceError):self.view.task(3).leaf_count=0

    def test_changed_route_or_certificate_cannot_reuse_tasks(self):
        self.view.require_plan(self.plan)
        with self.assertRaises(ValueError):self.view.require_plan(self.original)
        changed = deepcopy(self.plan)
        changed['surface_trajectory']['source']['geometry_sha256']='f'*64
        with self.assertRaises(ValueError):self.view.require_plan(changed)

    def test_cross_target_binding_and_resealed_partition_corruption_are_rejected(self):
        wrong = deepcopy(self.ledger)
        wrong.metadata['sha256']='e'*64
        with self.assertRaisesRegex(ValueError,'同一修订'):tasks.build(wrong,self.certificate)
        broken = deepcopy(self.certificate)
        arrays = dict(broken.arrays);arrays['code']=arrays['code'].copy()
        arrays['code'][1]=4**int(arrays['depth'][1])
        broken = reseal(broken, arrays=arrays)
        with self.assertRaises(ValueError):tasks.build(self.ledger,broken)

    def test_export_is_atomic_cancel_preserves_previous_file_and_plan(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)/'tasks.json';target.write_bytes(b'previous')
            before = deepcopy(self.plan)
            def cancel():raise RuntimeError('cancel')
            with self.assertRaisesRegex(RuntimeError,'cancel'):tasks.write(target,self.view,self.plan,cancel)
            self.assertEqual(target.read_bytes(),b'previous')
            calls=0
            def cancel_after_write():
                nonlocal calls
                calls+=1
                if calls==2:raise RuntimeError('cancel after write')
            with self.assertRaisesRegex(RuntimeError,'cancel after write'):
                tasks.write(target,self.view,self.plan,cancel_after_write)
            self.assertEqual(target.read_bytes(),b'previous')
            self.assertEqual(list(Path(folder).iterdir()),[target])
            with patch('femto.planning.coverage_tasks.os.replace', side_effect=PermissionError('locked')):
                with self.assertRaises(PermissionError):tasks.write(target,self.view,self.plan)
            self.assertEqual(target.read_bytes(),b'previous')
            self.assertEqual(list(Path(folder).iterdir()),[target])
            tasks.write(target,self.view,self.plan)
            self.assertEqual(json.loads(target.read_bytes()),self.view.to_dict())
            self.assertEqual(self.plan,before)

    def test_cancelled_aggregation_keeps_old_display_and_model_assets(self):
        previous = self.model.coverage_display
        with self.assertRaisesRegex(RuntimeError,'cancel'):
            tasks.build(self.ledger,self.certificate,lambda:(_ for _ in ()).throw(RuntimeError('cancel')))
        self.assertIs(self.model.coverage_display,previous)
        self.assertIs(self.model.coverage_certificate,self.certificate)


if __name__ == '__main__':unittest.main()
