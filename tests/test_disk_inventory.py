import json
import os
from pathlib import Path
import shutil
import tempfile
import subprocess
import unittest
from unittest.mock import patch

from femto.pipeline.disk_inventory import inspect_disk


class DiskInventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def write(self,name,data=b'original'):
        p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);return p

    def test_preview_retains_evidence_drafts_and_recovery_stages(self):
        paths=[self.write(name) for name in ('datasets/capture/depth.bin',
            'outputs/planning_draft.json','outputs/.draft_ab.tmp',
            'outputs/.publications/abc.stage','outputs/.restored.partial/__pycache__/x.pyc',
            'outputs/__pycache__/x.pyc')]
        before={p:p.read_bytes() for p in paths}
        report=inspect_disk({'raw':self.root/'datasets','results':self.root/'outputs'})
        self.assertTrue(report['scan_complete']);self.assertTrue(report['read_only'])
        self.assertEqual(report['logical_bytes'],sum(map(len,before.values())))
        self.assertEqual(report['cleanup']['recovery_review']['files'],3)
        self.assertEqual(report['cleanup']['regenerable_cache']['files'],1)
        self.assertEqual({p:p.read_bytes() for p in paths},before)

    def test_volume_warning_and_no_duplicate_volume_accounting(self):
        for name in ('a/data','b/data'):self.write(name)
        usage=shutil._ntuple_diskusage(total=1000,used=950,free=50)
        with patch('femto.pipeline.disk_inventory.shutil.disk_usage',return_value=usage) as query:
            report=inspect_disk({'a':self.root/'a','b':self.root/'b'},low_bytes=0)
        self.assertEqual(query.call_count,1)
        self.assertEqual(len(report['volumes']),1);self.assertTrue(report['volumes'][0]['low_space'])

    def test_incomplete_scan_surfaces_error_and_keeps_valid_sizes(self):
        self.write('valid/file',b'123')
        report=inspect_disk({'valid':self.root/'valid','missing':self.root/'missing'})
        self.assertFalse(report['scan_complete']);self.assertEqual(report['error_count'],1)
        self.assertEqual(report['logical_bytes'],3)

    def test_overlap_rejected_and_cancellation_propagates(self):
        self.write('data/file')
        with self.assertRaises(ValueError):inspect_disk({'a':self.root,'b':self.root/'data'})
        def cancel():raise InterruptedError('cancel')
        with self.assertRaises(InterruptedError):inspect_disk({'a':self.root},check=cancel)

    def test_links_never_traversed_including_selected_root(self):
        self.write('external/data',b'secret')
        scan=self.root/'scan';scan.mkdir()
        link=scan/'linked'
        try:link.symlink_to(self.root/'external',target_is_directory=True)
        except OSError as error:
            if os.name!='nt':self.skipTest('Symlink privilege unavailable: '+str(error))
            created=subprocess.run(['cmd','/c','mklink','/J',str(link),str(self.root/'external')],capture_output=True)
            if created.returncode:self.skipTest('Junction creation unavailable')
            self.addCleanup(link.rmdir)
        report=inspect_disk({'scan':scan})
        self.assertEqual(report['logical_bytes'],0);self.assertEqual(report['skipped_link_count'],1)
        with self.assertRaises(ValueError):inspect_disk({'link':link})

    def test_empty_roots_and_normalized_overlap_rejected(self):
        with self.assertRaises(ValueError):inspect_disk({})
        self.write('a/data')
        with self.assertRaises(ValueError):inspect_disk({'a':self.root/'a','b':self.root/'a/../a'})

    def test_unreadable_directory_is_reported_instead_of_silently_skipped(self):
        self.write('data/file')
        with patch('femto.pipeline.disk_inventory.os.scandir',side_effect=PermissionError('denied')):
            report=inspect_disk({'data':self.root/'data'})
        self.assertFalse(report['scan_complete']);self.assertEqual(report['error_count'],1)
        self.assertIn('denied',report['errors'][0]['error'])

    def test_bounded_samples_keep_complete_totals(self):
        for i in range(4):self.write('outputs/.draft_%d.tmp'%i,b'xx')
        report=inspect_disk({'outputs':self.root/'outputs'},sample_limit=1)
        self.assertEqual(report['cleanup']['recovery_review']['files'],4)
        self.assertEqual(len(report['cleanup']['recovery_review']['samples']),1)
        json.dumps(report,allow_nan=False)

    def test_selected_recovery_root_remains_protected_and_mid_scan_cancel_preserves_files(self):
        for i in range(5):self.write('.restore.partial/__pycache__/%d.pyc'%i,b'keep')
        root=self.root/'.restore.partial'
        report=inspect_disk({'recovery':root})
        self.assertEqual(report['cleanup']['recovery_review']['files'],5)
        self.assertEqual(report['cleanup']['regenerable_cache']['files'],0)
        calls=0
        def cancel():
            nonlocal calls
            calls+=1
            if calls==6:raise InterruptedError('cancel in scan')
        with self.assertRaises(InterruptedError):inspect_disk({'recovery':root},check=cancel)
        self.assertEqual(len(list(root.rglob('*.pyc'))),5)
