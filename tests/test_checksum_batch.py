"""Simulated CPU/RAM decisions and real bounded checksum worker behavior."""
from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from femto.resource_policy import MemorySnapshot, GiB, MiB
from femto import checksum_batch as batch
from femto.checksum_resources import CpuCapacity, checksum_resources, cpu_capacity


class ChecksumWorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.paths = tuple(Path(self.temp.name)/f'中文_{i}.bin' for i in range(4))
        for i,p in enumerate(self.paths):p.write_bytes(bytes([i])*8193)
        self.policy = checksum_resources(4,16*MiB,
            memory=MemorySnapshot(32*GiB,24*GiB,'simulated'),cpu=CpuCapacity(8,'simulated'))

    def test_cpu_and_actual_available_memory_both_gate_parallel_readers(self):
        for total,available,cpu,expected in ((16,1,18,1),(32,1,18,1),(64,1,18,1),(32,24,2,1),(32,24,4,2)):
            selected = checksum_resources(4,16*MiB,memory=MemorySnapshot(total*GiB,available*GiB,'simulated'),
                cpu=CpuCapacity(cpu,'simulated'))
            self.assertEqual(selected.workers,expected)
        for count,size in ((3,16*MiB),(4,16*MiB-1)):
            self.assertEqual(checksum_resources(count,size,memory=self.policy.memory,cpu=self.policy.cpu).workers,1)
        self.assertEqual(checksum_resources(4,16*MiB,memory=MemorySnapshot(None,None,'unavailable'),cpu=self.policy.cpu).workers,1)

    def test_actual_affinity_observation_has_a_safe_fallback(self):
        actual=cpu_capacity()
        self.assertGreaterEqual(actual.available_logical_cpus,1)
        self.assertNotEqual(actual.source,'simulated')

    def test_workers_one_and_two_hash_full_bytes_keep_order_and_read_only_results(self):
        expected={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in self.paths}
        for workers in (1,2):
            with patch.object(batch,'checksum_resources',return_value=replace(self.policy,workers=workers)):
                result=batch.sha256_files(self.paths)
            self.assertEqual(dict(result.hashes),expected)
            self.assertEqual(tuple(result.hashes),self.paths)
            with self.assertRaises(TypeError):result.hashes[self.paths[0]]='changed'

    def test_two_workers_run_concurrently_with_only_caller_thread_budget_checks(self):
        owner=threading.get_ident();barrier=threading.Barrier(2);lock=threading.Lock()
        active=peak=0;worker_threads=set()
        def check():self.assertEqual(threading.get_ident(),owner)
        def hasher(path,stop):
            nonlocal active,peak
            with lock:
                active+=1;peak=max(peak,active);worker_threads.add(threading.get_ident())
            try:
                barrier.wait(timeout=2);stop()
                return hashlib.sha256(path.read_bytes()).hexdigest()
            finally:
                with lock:active-=1
        with patch.object(batch,'checksum_resources',return_value=self.policy):
            result=batch.sha256_files(self.paths,check,hasher=hasher)
        self.assertEqual(peak,2);self.assertEqual(len(worker_threads),2);self.assertEqual(active,0)
        self.assertEqual(tuple(result.hashes),self.paths)

    def test_caller_cancellation_stops_and_joins_workers_and_preserves_original_error(self):
        class UserCancelled(RuntimeError):pass
        original=UserCancelled('requested');started=threading.Event();finished=threading.Event()
        before={p:p.read_bytes() for p in self.paths}
        owner=threading.get_ident()
        def check():
            self.assertEqual(threading.get_ident(),owner)
            if started.is_set():raise original
        def hasher(path,stop):
            try:
                with path.open('rb') as stream:
                    started.set()
                    while True:stop();stream.read(1);time.sleep(.001)
            finally:finished.set()
        with patch.object(batch,'checksum_resources',return_value=self.policy):
            with self.assertRaises(UserCancelled) as caught:batch.sha256_files(self.paths,check,hasher=hasher)
        self.assertIs(caught.exception,original);self.assertTrue(finished.is_set())
        self.assertFalse(any(t.name.startswith('FemtoChecksum') and t.is_alive() for t in threading.enumerate()))
        self.assertEqual({p:p.read_bytes() for p in self.paths},before)

    def test_failed_workers_keep_input_error_order_and_timeout_exception_identity(self):
        later=threading.Event();first_error=ValueError('first input');second_error=RuntimeError('second input')
        def hasher(path,stop):
            if path==self.paths[0]:later.wait(timeout=2);raise first_error
            later.set();raise second_error
        with patch.object(batch,'checksum_resources',return_value=self.policy):
            with self.assertRaises(ValueError) as caught:batch.sha256_files(self.paths,hasher=hasher)
        self.assertIs(caught.exception,first_error)
        original=TimeoutError('worker timeout')
        def timed_out(path,stop):raise original
        with patch.object(batch,'checksum_resources',return_value=self.policy):
            with self.assertRaises(TimeoutError) as caught:batch.sha256_files(self.paths,hasher=timed_out)
        self.assertIs(caught.exception,original)


if __name__=='__main__':unittest.main()
