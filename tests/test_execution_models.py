"""Snapshot integrity and native inventory isolation; real-plan integration is separate."""
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from femto.execution.models import AssetReference, FrozenPlan, canonical, gap_summary, load_saved_plan, synthetic_plan
from femto.io_utils import sha256_file
from femto.planning.trajectory_export import FILES_SCHEMA


class ExecutionModelTests(unittest.TestCase):
    def test_display_summary_keeps_unknown_conclusions_without_copying_proof_arrays(self):
        report=dict(counts={'selection_break':1},quality=dict(pending_pixels=52,registration_uncertainty_mm=None,
                    target_completion='partial',edge_decision_runs=[{'edge':i} for i in range(2000)]))
        before=canonical(report);summary=gap_summary(report)
        self.assertEqual(summary['counts']['selection_break'],1)
        self.assertEqual(summary['quality']['pending_pixels'],52)
        self.assertIsNone(summary['quality']['registration_uncertainty_mm'])
        self.assertEqual(summary['quality']['edge_decision_runs_count'],2000)
        self.assertNotIn('edge_decision_runs',summary['quality'])
        self.assertEqual(canonical(report),before)
    def test_snapshot_rejects_mismatched_steps_route_digest_and_mutable_assets(self):
        base=synthetic_plan()
        for steps in (base.steps[::-1],(replace(base.steps[0],index=7),*base.steps[1:]),
                      (replace(base.steps[0],reference_sha256='0'*64),*base.steps[1:])):
            with self.assertRaises(ValueError):FrozenPlan(base.data_json,steps,(),base.source_kind)
        changed=base.public_plan;changed['ordered_items'][0]['points'][0]['xyz_m'][0]=1
        with self.assertRaises(ValueError):FrozenPlan(canonical(changed),base.steps,(),base.source_kind)
        with self.assertRaises(ValueError):FrozenPlan(base.data_json,base.steps,({'path':'mutable'},),base.source_kind)
        with self.assertRaises(ValueError):AssetReference('relative.json','0'*64)

    def test_cancelled_initial_hash_does_not_start_source_verification(self):
        cancel=threading.Event();cancel.set()
        with patch('femto.execution.models.load_source') as load:
            with self.assertRaises(RuntimeError):load_saved_plan('missing.region.zip',cancel=cancel)
        load.assert_not_called()

    def test_native_snapshot_binds_registered_outputs_and_excludes_unrelated_draft(self):
        with TemporaryDirectory() as root:
            root=Path(root);base=synthetic_plan()
            selected=root/'trajectory.json';selected.write_text(base.data_json,encoding='utf8')
            registered=root/'proof.json';registered.write_text('{}',encoding='utf8')
            unrelated=root/'draft.json';unrelated.write_text('{}',encoding='utf8')
            mesh=root/'source.ply';mesh.write_bytes(b'synthetic source metadata fixture')
            inventory=dict(schema=FILES_SCHEMA,result_sha256=base.public_plan['result_sha256'],
                           files={p.name:sha256_file(p) for p in (selected,registered)})
            (root/'trajectory_files.json').write_text(json.dumps(inventory),encoding='utf8')
            trajectory=base.public_plan;trajectory['source']={}
            model=SimpleNamespace(saved_plan=dict(surface_trajectory=trajectory,treatment_head=base.public_plan['treatment_head']),
                                  store=SimpleNamespace(digest='0'*64,source=str(mesh),regions=[dict(annotation_id='synthetic')]))
            with patch('femto.execution.models.load_source',return_value=model):plan=load_saved_plan(selected)
            self.assertNotIn(str(unrelated),{a.path for a in plan.assets})
            self.assertIn(str(registered),{a.path for a in plan.assets})
            unrelated.write_text('{"edited":true}',encoding='utf8');plan.verify_assets()
            registered.write_text('{"edited":true}',encoding='utf8')
            with self.assertRaises(ValueError):plan.verify_assets()


if __name__=='__main__':unittest.main()
