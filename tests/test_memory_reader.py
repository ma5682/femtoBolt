"""Fresh Windows observations must not retain a new ctypes type per query."""
import ctypes
import gc
import unittest
from unittest.mock import patch

from femto import resource_policy as policy


class MemoryReaderTests(unittest.TestCase):
    def test_windows_type_cache_is_bounded_across_fresh_queries(self):
        if policy.os.name != 'nt':self.skipTest('Real Windows API observation')
        policy.memory_snapshot()
        before=len(ctypes._pointer_type_cache)
        for _ in range(40):policy.memory_snapshot()
        gc.collect()
        self.assertEqual(len(ctypes._pointer_type_cache),before)

    def test_cached_binding_reads_fresh_status_instead_of_caching_memory_values(self):
        count=0
        def query(pointer):
            nonlocal count
            count+=1
            status=ctypes.cast(pointer,ctypes.POINTER(policy._WindowsMemoryStatus)).contents
            self.assertEqual(status.length,ctypes.sizeof(policy._WindowsMemoryStatus))
            status.total=16*policy.GiB
            status.available=count*policy.MiB
            return 1
        with patch.object(policy,'_windows_memory_function',return_value=query), patch.object(policy.os,'name','nt'):
            first,second=policy.memory_snapshot(),policy.memory_snapshot()
        self.assertEqual(first.available_bytes,policy.MiB)
        self.assertEqual(second.available_bytes,2*policy.MiB)

    def test_failed_binding_keeps_unknown_memory_fallback(self):
        with patch.object(policy,'_windows_memory_snapshot',side_effect=OSError('unavailable')), patch.object(policy.os,'name','nt'):
            observed=policy.memory_snapshot()
        self.assertIsNone(observed.total_bytes)
        self.assertIsNone(observed.available_bytes)
        self.assertEqual(observed.source,'unavailable')


if __name__=='__main__':unittest.main()
