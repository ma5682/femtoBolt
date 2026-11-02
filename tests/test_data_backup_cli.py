import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class DataBackupCLITests(unittest.TestCase):
    def test_inspect_backup_verify_and_explicit_new_restore(self):
        repo=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'outputs').mkdir();(root/'datasets').mkdir()
            (root/'datasets/raw.bin').write_bytes(b'original')
            base=[sys.executable,'-X','utf8',str(repo/'tools/data_backup.py'),'--workspace',str(root)]
            def run(args,expected=0):
                result=subprocess.run(base+args,cwd=repo,capture_output=True,encoding='utf-8')
                self.assertEqual(result.returncode,expected,result.stderr)
                return json.loads(result.stdout) if not expected else result
            self.assertTrue(run(['inspect'])['ready'])
            preview=run(['disk-preview'])
            self.assertTrue(preview['read_only']);self.assertTrue(preview['scan_complete'])
            self.assertEqual(preview['logical_bytes'],len(b'original'))
            incomplete=subprocess.run(base+['--root','missing='+str(root/'missing'),'disk-preview'],cwd=repo,
                                      capture_output=True,encoding='utf-8')
            self.assertEqual(incomplete.returncode,2,incomplete.stderr)
            self.assertFalse(json.loads(incomplete.stdout)['scan_complete'])
            self.assertEqual(run(['backup',str(root/'backup')])['state'],'backed_up')
            self.assertTrue(run(['verify',str(root/'backup')])['source_assets_complete'])
            run(['restore',str(root/'backup')],1)
            result=run(['--to','outputs='+str(root/'new_outputs'),'--to','datasets='+str(root/'new_datasets'),
                        'restore',str(root/'backup')])
            self.assertEqual(result['state'],'restored')
            self.assertFalse(result['original_locations'])
            self.assertEqual((root/'new_datasets/raw.bin').read_bytes(),b'original')
