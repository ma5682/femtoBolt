"""Both publication adapters preserve byte manifests under one/two hash workers."""
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from femto.resource_policy import MemorySnapshot, GiB, MiB
from femto.checksum_resources import CpuCapacity, checksum_resources
from femto.planning import trajectory_export, region_project
from test_surface_trajectory import trajectory_fixture


class ChecksumAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        block=bytes(range(256))*4096
        for index in range(4):
            with (self.root/f'资产_{index}.bin').open('wb') as stream:
                for _ in range(4):stream.write(block)

    def policy(self,workers):
        def select(count,size):
            memory=MemorySnapshot(32*GiB,24*GiB,'simulated-only') if workers==2 else MemorySnapshot(32*GiB,1*GiB,'simulated-only')
            return checksum_resources(count,size,memory=memory,cpu=CpuCapacity(8,'simulated-only'))
        return select

    def test_native_manifest_bytes_and_portable_asset_order_are_identical(self):
        trajectory=trajectory_fixture();owner=threading.get_ident()
        def check():self.assertEqual(threading.get_ident(),owner)
        native=[];portable=[];decisions=[]
        for workers in (1,2):
            def select(count,size):
                selected=self.policy(workers)(count,size);decisions.append(selected.workers);return selected
            with patch('femto.checksum_batch.checksum_resources',side_effect=select):
                path=trajectory_export.write_files_manifest(self.root,trajectory,check=check)
                native.append(path.read_bytes())
                # The native JSON is now an ordinary portable asset, too.
                portable.append(region_project._project_asset_hashes(self.root,check))
        self.assertEqual(decisions,[1,1,2,2])
        self.assertEqual(native[0],native[1])
        self.assertEqual(portable[0],portable[1])
        self.assertEqual(list(portable[0]),list(portable[1]))
        files=json.loads(native[0])['files']
        for name,digest in files.items():self.assertEqual(digest,hashlib.sha256((self.root/name).read_bytes()).hexdigest())

    def test_hash_failure_cannot_replace_existing_manifest_and_joins_workers(self):
        class UserCancelled(RuntimeError):pass
        original=UserCancelled('injected checksum cancellation')
        path=self.root/'trajectory_files.json';path.write_bytes(b'previous manifest')
        def fail(file,check):raise original
        with patch('femto.checksum_batch.checksum_resources',side_effect=self.policy(2)), patch.object(trajectory_export,'_sha256',side_effect=fail):
            with self.assertRaises(UserCancelled) as caught:trajectory_export.write_files_manifest(self.root,trajectory_fixture())
        self.assertIs(caught.exception,original)
        self.assertEqual(path.read_bytes(),b'previous manifest')
        self.assertFalse(any(t.name.startswith('FemtoChecksum') and t.is_alive() for t in threading.enumerate()))


if __name__=='__main__':unittest.main()
