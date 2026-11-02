"""Permanent deletion stays restricted to hidden, unreferenced captures."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from femto.data_access import DataLease
from femto.history_store import HistoryStore
from femto.pipeline.history_purge import preview_purge,execute_purge


class HistoryPurgeTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.target=self.root/'datasets'/'scene_20261022_local_example'
        self.target.mkdir(parents=True);(self.root/'outputs').mkdir()
        (self.target/'depth.bin').write_bytes(b'original-data')
        self.row=dict(key=str(self.target),name='example')
        self.store=HistoryStore(self.root/'outputs');self.store.set_hidden([self.row],True)

    def test_preview_preserves_files_and_delete_clears_hidden_record(self):
        plan=preview_purge(self.root,self.target)
        self.assertEqual(plan['files'],1);self.assertFalse(plan['blockers'])
        self.assertTrue((self.target/'depth.bin').exists())
        execute_purge(self.root,self.target,plan['digest'])
        self.assertFalse(self.target.exists());self.assertEqual(self.store.deletions(),{})

    def test_active_workstation_lease_prevents_physical_deletion(self):
        plan=preview_purge(self.root,self.target)
        with DataLease(self.root):
            with self.assertRaises(RuntimeError):execute_purge(self.root,self.target,plan['digest'])
        self.assertTrue(self.target.exists())

    def test_idle_owning_workstation_can_delete_and_regains_shared_lease(self):
        with DataLease(self.root) as current:
            execute_purge(self.root,self.target,workstation_lease=current)
            self.assertIsNotNone(current.stream)
        self.assertFalse(self.target.exists())

    def test_another_workstation_blocks_upgrade_and_owner_lease_is_restored(self):
        with DataLease(self.root) as current,DataLease(self.root):
            with self.assertRaises(RuntimeError):execute_purge(self.root,self.target,workstation_lease=current)
            self.assertIsNotNone(current.stream)
        self.assertTrue(self.target.exists())

    def test_shared_source_reference_blocks_delete(self):
        result=self.root/'outputs'/'complete_demo_other'/'reconstruction';result.mkdir(parents=True)
        (result/'reconstruction_report.json').write_text(json.dumps(dict(source_dataset=str(self.target))))
        plan=preview_purge(self.root,self.target)
        self.assertIn(str(self.target),plan['blockers'])
        with self.assertRaises(ValueError):execute_purge(self.root,self.target,plan['digest'])
        self.assertTrue(self.target.exists())

    def test_new_reference_after_preview_is_rechecked(self):
        plan=preview_purge(self.root,self.target)
        other=self.root/'datasets'/'scene_other_local_capture';other.mkdir()
        (other/'manifest.json').write_text(json.dumps(dict(source_dataset=str(self.target))))
        with self.assertRaises(ValueError):execute_purge(self.root,self.target,plan['digest'])
        self.assertTrue(self.target.exists())

    def test_changed_directory_requires_new_confirmation(self):
        plan=preview_purge(self.root,self.target)
        (self.target/'new.bin').write_bytes(b'new')
        with self.assertRaises(ValueError):execute_purge(self.root,self.target,plan['digest'])
        self.assertTrue(self.target.exists())

    def test_restored_record_cannot_be_deleted(self):
        plan=preview_purge(self.root,self.target);self.store.set_hidden([self.row],False)
        with self.assertRaises(ValueError):execute_purge(self.root,self.target,plan['digest'])

    def test_workspace_root_and_external_paths_are_rejected(self):
        for path in (self.root,self.root/'datasets',self.root.parent):
            with self.subTest(path=path),self.assertRaises(ValueError):preview_purge(self.root,path)

    def test_result_delete_does_not_follow_its_source_and_does_not_self_block(self):
        result=self.root/'outputs'/'complete_demo_example';rec=result/'reconstruction';rec.mkdir(parents=True)
        (rec/'reconstruction_report.json').write_text(json.dumps(dict(source_dataset=str(self.target))))
        row=dict(key=str(result),result=str(result),source=str(self.target),name='result',time=1)
        self.store.set_hidden([row],True);self.store.replace('captures',[row],search_text=lambda row:row['name'])
        plan=preview_purge(self.root,result)
        self.assertFalse(plan['blockers'])
        execute_purge(self.root,result,plan['digest'])
        self.assertFalse(result.exists());self.assertTrue(self.target.exists())


if __name__=='__main__':unittest.main()
