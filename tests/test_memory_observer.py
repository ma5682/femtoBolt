"""Observer lifetime checks must not need real geometry or forced collection."""
import gc
import unittest
import weakref
from unittest.mock import patch

from tools.soak_workstation import WeakObjects
from tools.runtime_memory import MemoryDiagnostics, performance_environment
import tracemalloc


class ObserverLifetimeTests(unittest.TestCase):
    def test_environment_observation_does_not_start_allocation_tracing(self):
        before = tracemalloc.is_tracing()
        data = performance_environment()
        self.assertEqual(tracemalloc.is_tracing(), before)
        self.assertEqual(data['allocation_tracer_active'], before)
        self.assertIn('system_cpu_times', data)
        self.assertLessEqual(len(data['other_python_processes']), 16)

    def test_unobservable_native_objects_are_explicitly_reported(self):
        tracker = WeakObjects()
        tracker.watch('not_weakrefable', 1)
        self.assertEqual(tracker.snapshot()['unsupported_types']['builtins.int'], 1)
        self.assertEqual(tracker.snapshot()['alive'], 0)

    def test_survivor_descriptions_do_not_retain_referred_objects(self):
        class Value:
            pass
        tracker = WeakObjects()
        value = Value()
        reference = weakref.ref(value)
        tracker.watch('fixture', value)
        tracker.cycle = 1
        rows = tracker.survivor_details()
        self.assertEqual(rows[0]['age'], 1)
        self.assertTrue(all(isinstance(item, str) for item in rows[0]['referrer_types']))
        del value
        self.assertIsNone(reference())
        self.assertEqual(tracker.survivor_details(), [])

    def test_minimal_mode_does_not_traverse_or_watch_objects(self):
        tracker = WeakObjects(enabled=False)
        value = unittest.mock.Mock()
        tracker.model('fixture', value)
        tracker.watch('fixture', value)
        self.assertFalse(tracker.snapshot()['enabled'])
        self.assertEqual(tracker.observed, 0)

    def test_allocation_diagnostics_are_bounded_and_restore_tracer(self):
        with MemoryDiagnostics('allocations') as observer:
            first = observer.snapshot(allocation_detail=True)
            values = [bytearray(1024) for _ in range(32)]
            second = observer.snapshot(allocation_detail=True)
            self.assertTrue(tracemalloc.is_tracing())
            self.assertEqual(len(values), 32)
            self.assertGreater(second['traced_current_bytes'], first['traced_current_bytes'])
            self.assertLessEqual(len(second['allocation_changes']), 12)
            self.assertIn('numpy_domain_bytes', second)
            self.assertIn('tracer_metadata_bytes', second)
        self.assertFalse(tracemalloc.is_tracing())
        self.assertIsNone(observer.previous)

    def test_existing_tracer_is_not_stopped_or_reconfigured(self):
        tracemalloc.start(1)
        try:
            with self.assertRaises(RuntimeError):
                with MemoryDiagnostics('allocations'):
                    pass
            self.assertTrue(tracemalloc.is_tracing())
            self.assertEqual(tracemalloc.get_traceback_limit(), 1)
        finally:
            tracemalloc.stop()

    def test_invalid_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            MemoryDiagnostics('unknown')

    def test_visited_set_releases_without_waiting_for_cyclic_gc(self):
        references = []
        def observed_set():
            value = set()
            references.append(weakref.ref(value))
            return value
        enabled = gc.isenabled()
        gc.disable()
        try:
            tracker = WeakObjects()
            with patch('tools.soak_workstation.set', observed_set, create=True):
                for _ in range(4):
                    tracker.model('fixture', {'nested': [dict(value=i) for i in range(200)]})
                    self.assertIsNone(references[-1](), 'Observer visited IDs survive the traversal')
        finally:
            if enabled:
                gc.enable()


if __name__ == '__main__':
    unittest.main()
