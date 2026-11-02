"""Full-byte checksum, bounded allocation and cancellation at file boundaries."""
import hashlib
from pathlib import Path
import tempfile
import tracemalloc
import unittest
from unittest.mock import patch

from femto.io_utils import sha256_file
from femto.planning import trajectory_export,region_project,region_library,mesh_slice_planner
from test_surface_trajectory import trajectory_fixture


class Cancelled(RuntimeError):pass


class StreamingChecksumTests(unittest.TestCase):
    def test_unicode_empty_and_chunk_boundaries_match_full_byte_sha256(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'中文校验.bin'
            for size in (0,1,1024*1024-1,1024*1024,1024*1024+17):
                data=(bytes(range(256))*((size+255)//256))[:size];path.write_bytes(data)
                expected=hashlib.sha256(data).hexdigest()
                with patch.object(Path,'read_bytes',side_effect=AssertionError('whole-file read')):
                    for fn in (sha256_file,trajectory_export._sha256,region_project.digest,region_library.digest,mesh_slice_planner._sha256):
                        self.assertEqual(fn(path),expected)

    def test_large_file_checksum_has_bounded_python_allocation(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'large.bin';block=bytes(range(256))*4096;expected=hashlib.sha256()
            with path.open('wb') as stream:
                for _ in range(32):stream.write(block);expected.update(block)
            tracemalloc.start()
            try:
                actual=sha256_file(path);_,peak=tracemalloc.get_traced_memory()
            finally:tracemalloc.stop()
            self.assertEqual(actual,expected.hexdigest())
            self.assertLess(peak,3*1024*1024)

    def test_pre_and_mid_read_cancel_return_no_digest_and_close_source(self):
        def stop():raise Cancelled('cancel')
        with self.assertRaises(Cancelled):sha256_file(Path('does-not-exist'),stop)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'cancel.bin';data=b'a'*(3*1024*1024);path.write_bytes(data);calls=0
            def during():
                nonlocal calls
                calls+=1
                if calls==3:stop()
            for fn in (sha256_file,trajectory_export._sha256,region_project.digest,region_library.digest):
                calls=0
                with self.assertRaises(Cancelled):fn(path,during)
                moved=path.with_suffix('.moved');path.rename(moved);moved.rename(path)
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),hashlib.sha256(data).hexdigest())

    def test_native_publication_cancelled_during_hash_leaves_no_output(self):
        with tempfile.TemporaryDirectory() as temp:
            original=trajectory_export._sha256
            def cancelling(path,check=None):
                def stop():raise Cancelled('cancel checksum')
                return original(path,stop)
            with patch.object(trajectory_export,'_sha256',side_effect=cancelling):
                with self.assertRaises(Cancelled):
                    trajectory_export.export_trajectory(trajectory_fixture(),Path(temp),include_visuals=False)
            self.assertEqual(list(Path(temp).iterdir()),[])

    def test_native_file_verification_honours_cancel_and_rejects_changed_asset(self):
        with tempfile.TemporaryDirectory() as temp:
            trajectory=trajectory_fixture();folder=trajectory_export.export_trajectory(trajectory,Path(temp),include_visuals=False)
            expected=trajectory.to_dict()['result_sha256'];calls=0
            def during():
                nonlocal calls
                calls+=1
                if calls==3:raise Cancelled('cancel checksum')
            with self.assertRaises(Cancelled):trajectory_export._verify_files(folder,expected,during)
            with (folder/'trajectory.csv').open('ab') as stream:stream.write(b'changed')
            with self.assertRaisesRegex(ValueError,'file hash mismatch'):
                trajectory_export._verify_files(folder,expected)


if __name__=='__main__':unittest.main()
