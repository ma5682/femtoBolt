from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from femto import data_backup as bundle


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve()
        self.outputs=self.root/'outputs'
        self.datasets=self.root/'datasets'
        self.outputs.mkdir();self.datasets.mkdir()
        (self.datasets/'原始帧.bin').write_bytes(b'original-rgb-depth')
        self.database=self.outputs/'history.sqlite3'
        self.source=sqlite3.connect(self.database)
        self.addCleanup(self.source.close)
        self.source.execute('PRAGMA journal_mode=WAL')
        self.source.execute('CREATE TABLE entries (value TEXT)')
        self.source.execute("INSERT INTO entries VALUES ('hidden-and-history')")
        self.source.commit()
        self.roots=dict(outputs=self.outputs,datasets=self.datasets)
        self.destination=self.root/'backup'

    def backup(self,**kwargs):
        return bundle.backup(self.roots,self.destination,database_files=[('outputs','history.sqlite3')],
                             references=[self.datasets/'原始帧.bin'],workspace=self.root,reserve=0,**kwargs)

    def test_wal_snapshot_assets_and_independent_restore(self):
        self.backup()
        manifest=bundle.verify(self.destination)
        self.assertNotIn('history.sqlite3-wal',manifest['roots']['outputs']['files'])
        targets=dict(outputs=self.root/'restored_outputs',datasets=self.root/'restored_datasets')
        result=bundle.restore_new(self.destination,targets,workspace=self.root,reserve=0)
        self.assertFalse(result['original_locations'])
        self.assertEqual((targets['datasets']/'原始帧.bin').read_bytes(),b'original-rgb-depth')
        with closing(sqlite3.connect(targets['outputs']/'history.sqlite3')) as database:
            self.assertEqual(database.execute('SELECT value FROM entries').fetchall(),[('hidden-and-history',)])

    def test_original_directories_unavailable_restore_database_and_assets_to_original_locations(self):
        self.backup();self.source.close()
        preserved=self.root/'preserved_originals';preserved.mkdir()
        for name,path in self.roots.items():
            self.assertTrue(path.resolve().is_relative_to(self.root))
            target=preserved/name;self.assertTrue(target.resolve().is_relative_to(self.root))
            path.rename(target)
        self.assertFalse(self.outputs.exists());self.assertFalse(self.datasets.exists())
        # Verify and restore using only the backup, while original paths are gone.
        bundle.verify(self.destination)
        restored=bundle.restore_new(self.destination,self.roots,workspace=self.root,reserve=0)
        self.assertTrue(restored['original_locations']);self.assertFalse(restored['paths_rebased'])
        self.assertEqual((self.datasets/'原始帧.bin').read_bytes(),b'original-rgb-depth')
        with closing(sqlite3.connect(self.database)) as database:
            self.assertEqual(database.execute('SELECT value FROM entries').fetchall(),[('hidden-and-history',)])
        self.assertEqual((preserved/'datasets/原始帧.bin').read_bytes(),b'original-rgb-depth')

    def test_deep_windows_paths_are_enumerated_copied_and_restored(self):
        if os.name=='nt':
            # Cleanup only this test's generated TemporaryDirectory, using an
            # extended path too. Otherwise stdlib cleanup misses deep children.
            resolved=Path(self.temp.name).resolve()
            self.assertEqual(resolved.parent,Path(tempfile.gettempdir()).resolve())
            self.assertTrue(resolved.name.startswith('tmp'))
            self.temp.name='\\\\?\\'+str(resolved)
        key='/'.join(['deep_level_'+str(i)+'_'+'x'*25 for i in range(7)])+'/recorded.bin'
        source=bundle.relative(self.datasets,key)
        source.parent.mkdir(parents=True);source.write_bytes(b'deep original')
        self.assertGreater(len(str(source)),260)
        self.backup()
        manifest=bundle.verify(self.destination)
        self.assertIn(key,manifest['roots']['datasets']['files'])
        targets=dict(outputs=self.root/'new_outputs',datasets=self.root/'new_datasets')
        bundle.restore_new(self.destination,targets,workspace=self.root,reserve=0)
        self.assertEqual(bundle.relative(targets['datasets'],key).read_bytes(),b'deep original')

    def test_corruption_rejected_before_restore_destination_created(self):
        self.backup()
        (self.destination/'contents/datasets/原始帧.bin').write_bytes(b'corrupt')
        targets=dict(outputs=self.root/'restore_o',datasets=self.root/'restore_d')
        with self.assertRaises(ValueError):bundle.restore_new(self.destination,targets,workspace=self.root,reserve=0)
        self.assertTrue(all(not p.exists() for p in targets.values()))

    def test_manifest_changed_after_preview_requires_a_new_review(self):
        self.backup();reviewed=bundle.verify(self.destination)['_verified_sha256']
        path=self.destination/'manifest.json';document=json.loads(path.read_text('utf-8'))
        document['finished_unix']+=1
        path.write_text(json.dumps(document),encoding='utf-8')
        (self.destination/'manifest.sha256').write_text(bundle.digest(path),encoding='ascii')
        targets=dict(outputs=self.root/'restore_o',datasets=self.root/'restore_d')
        with self.assertRaisesRegex(ValueError,'预览后已改变'):
            bundle.restore_new(self.destination,targets,workspace=self.root,expected_manifest_sha=reviewed)
        self.assertEqual(bundle.pending_restores(self.root),[])
        self.assertTrue(all(not p.exists() for p in targets.values()))

    def test_existing_target_never_overwritten(self):
        self.backup()
        sentinel=self.outputs/'keep.txt';sentinel.write_text('keep')
        with self.assertRaises(FileExistsError):bundle.restore_new(self.destination,self.roots,workspace=self.root,reserve=0)
        self.assertEqual(sentinel.read_text(),'keep')

    def test_invalid_restore_mapping_never_leaves_a_blocking_transaction(self):
        self.backup()
        with self.assertRaisesRegex(ValueError,'every data root'):
            bundle.restore_new(self.destination,dict(outputs=self.root/'new_o'),workspace=self.root,reserve=0)
        self.assertEqual(bundle.pending_restores(self.root),[])
        targets=dict(outputs=self.destination/'nested',datasets=self.root/'new_d')
        with self.assertRaisesRegex(ValueError,'overlaps bundle'):
            bundle.restore_new(self.destination,targets,workspace=self.root,reserve=0)
        self.assertEqual(bundle.pending_restores(self.root),[])

    def test_cancel_keeps_source_and_does_not_publish(self):
        def cancel():raise RuntimeError('cancelled')
        with self.assertRaisesRegex(RuntimeError,'cancelled'):self.backup(check=cancel)
        self.assertFalse(self.destination.exists())
        self.assertEqual((self.datasets/'原始帧.bin').read_bytes(),b'original-rgb-depth')
        self.assertEqual(self.source.execute('SELECT count(*) FROM entries').fetchone()[0],1)

    def test_low_disk_space_does_not_publish(self):
        usage=type('Usage',(),dict(free=0))()
        with patch.object(bundle.shutil,'disk_usage',return_value=usage):
            with self.assertRaisesRegex(OSError,'磁盘空间不足'):self.backup()
        self.assertFalse(self.destination.exists())

    def test_external_missing_and_escaping_references_refused(self):
        with self.assertRaises(FileNotFoundError):
            bundle.backup(self.roots,self.destination,workspace=self.root,references=[self.root/'missing'])
        external=self.root/'external';external.write_text('not-covered')
        with self.assertRaisesRegex(ValueError,'备份数据根'):
            bundle.backup(self.roots,self.destination,workspace=self.root,references=[external])
        for path in ('../escape','C:/escape','/escape','a\\..\\escape','a:stream','CON','a/../b'):
            with self.subTest(path=path),self.assertRaises(ValueError):bundle.relative(self.root,path)

    def test_both_databases_locked_before_any_copy(self):
        second=self.outputs/'execution.sqlite3'
        with closing(sqlite3.connect(second)) as connection:
            connection.execute('CREATE TABLE events (value TEXT)')
        observed=[]
        original=bundle.inventory
        def inspect(root,**kwargs):
            if root==self.outputs and not observed:
                for path in (self.database,second):
                    with closing(sqlite3.connect(path,timeout=0)) as writer:
                        with self.assertRaises(sqlite3.OperationalError):writer.execute('BEGIN IMMEDIATE')
                observed.append(True)
            return original(root,**kwargs)
        with patch.object(bundle,'inventory',side_effect=inspect):
            bundle.backup(self.roots,self.destination,workspace=self.root,
                          database_files=[('outputs','history.sqlite3'),('outputs','execution.sqlite3')],reserve=0)
        self.assertEqual(observed,[True])

    def test_asset_mutation_refuses_publication_and_allows_new_retry(self):
        original=bundle.inventory
        called=0
        def mutate(root,**kwargs):
            nonlocal called
            result=original(root,**kwargs)
            if root==self.datasets:
                called+=1
                if called==1:(root/'原始帧.bin').write_bytes(b'changed')
            return result
        with patch.object(bundle,'inventory',side_effect=mutate):
            with self.assertRaisesRegex(ValueError,'changed'):self.backup()
        self.assertFalse(self.destination.exists())
        self.backup()
        self.assertEqual(bundle.verify(self.destination)['state'],'complete')

    def crash_restore(self,phase):
        self.backup()
        targets=dict(outputs=self.root/'restored_o',datasets=self.root/'restored_d')
        script=r'''
import os,sys,json
from pathlib import Path
from femto import data_backup as b
workspace=Path(sys.argv[1]); phase=sys.argv[2]
rename=Path.rename; unlink=Path.unlink
def replace(self,target):
    value=rename(self,target)
    if phase=='publish' and '.restore.' in self.name:os._exit(37)
    return value
def remove(self,*args,**kwargs):
    value=unlink(self,*args,**kwargs)
    if phase=='finalize' and self.name=='.femto_restore.json':os._exit(38)
    return value
Path.rename=replace;Path.unlink=remove
b.restore_new(workspace/'backup',dict(outputs=workspace/'restored_o',datasets=workspace/'restored_d'),workspace=workspace,reserve=0)
'''
        result=subprocess.run([sys.executable,'-c',script,str(self.root),phase],capture_output=True,text=True)
        self.assertEqual(result.returncode,37 if phase=='publish' else 38,result.stderr)
        journal,=bundle.pending_restores(self.root)
        with self.assertRaisesRegex(RuntimeError,'未完成'):bundle.assert_no_pending_restore(self.root)
        return targets,journal

    def test_real_process_crash_after_first_root_can_resume(self):
        targets,journal=self.crash_restore('publish')
        self.assertEqual(bundle.recover_restore(journal,targets,workspace=self.root)['state'],'restored')
        bundle.assert_no_pending_restore(self.root)
        self.assertEqual((targets['datasets']/'原始帧.bin').read_bytes(),b'original-rgb-depth')

    def test_ready_staging_can_recover_after_original_bundle_is_corrupted(self):
        targets,journal=self.crash_restore('publish')
        (self.destination/'manifest.json').write_text('corrupt')
        self.assertEqual(bundle.recover_restore(journal,targets,workspace=self.root)['state'],'restored')
        self.assertEqual((targets['datasets']/'原始帧.bin').read_bytes(),b'original-rgb-depth')

    def test_corrupt_restore_receipt_still_blocks_start(self):
        folder=self.root/'.data_restores';folder.mkdir()
        (folder/'broken.json').write_text('{')
        with self.assertRaisesRegex(RuntimeError,'未完成'):bundle.assert_no_pending_restore(self.root)

    def test_real_process_crash_while_clearing_markers_can_resume(self):
        targets,journal=self.crash_restore('finalize')
        self.assertEqual(bundle.recover_restore(journal,targets,workspace=self.root)['state'],'restored')
        bundle.assert_no_pending_restore(self.root)
        self.assertFalse(any((p/'.femto_restore.json').exists() for p in targets.values()))

    def test_recovery_refuses_changed_published_root(self):
        targets,journal=self.crash_restore('publish')
        published=next(p for p in targets.values() if p.exists())
        (published/'foreign.txt').write_text('keep')
        with self.assertRaisesRegex(ValueError,'改变'):
            bundle.recover_restore(journal,targets,workspace=self.root)
        self.assertEqual((published/'foreign.txt').read_text(),'keep')

    def test_partial_preparation_can_be_aborted_without_deleting_assets(self):
        self.backup()
        targets=dict(outputs=self.root/'restore_o',datasets=self.root/'restore_d')
        original=bundle.space
        def fail(parent,*args,**kwargs):
            if '.restore.' in parent.name:raise OSError('disk failure')
            return original(parent,*args,**kwargs)
        with patch.object(bundle,'space',side_effect=fail):
            with self.assertRaisesRegex(OSError,'disk failure'):
                bundle.restore_new(self.destination,targets,workspace=self.root,reserve=0)
        journal,=bundle.pending_restores(self.root)
        result=bundle.abort_restore(journal,targets,workspace=self.root)
        self.assertEqual(result['state'],'aborted')
        self.assertTrue(all(Path(p).exists() for p in result['retained_stages'].values()))
        bundle.assert_no_pending_restore(self.root)
        self.assertEqual(bundle.restore_new(self.destination,targets,workspace=self.root,reserve=0)['state'],'restored')
