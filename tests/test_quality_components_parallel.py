"""Parallel native component arithmetic preserves ordered reports and errors."""
import threading
from unittest import TestCase
from unittest.mock import patch
import numpy as np
from femto.planning import effective_domain, quality_components as components


class QualityComponentsParallelTests(TestCase):
    def test_identical_reports_for_all_classes_empty_and_noncontiguous_inputs(self):
        rng = np.random.default_rng(78)
        for classes in (np.ones((50,70), np.int32), rng.integers(0,8,(60,45),dtype=np.int32),
                        rng.integers(0,8,(45,60),dtype=np.uint8).T):
            before = classes.copy()
            with patch.object(components, 'MIN_PARALLEL_PIXELS', 10**9):
                expected = effective_domain.missing_regions(classes)
            for workers in (2,4):
                with patch.object(components, 'MIN_PARALLEL_PIXELS', 0), patch.object(components, 'MAX_WORKERS', workers):
                    self.assertEqual(effective_domain.missing_regions(classes), expected)
            np.testing.assert_array_equal(classes, before)

    def test_fragmentation_rejection_is_identical(self):
        classes = np.ones((150,150), np.int32)
        classes[::2,::2] = 7
        for threshold in (0,10**9):
            with patch.object(components, 'MIN_PARALLEL_PIXELS', threshold):
                with self.assertRaisesRegex(ValueError, '超过报告预算'):
                    effective_domain.missing_regions(classes)

    def test_checks_stay_on_caller_and_cancellation_joins_all_workers(self):
        caller = threading.get_ident()
        calls = []
        def check():
            calls.append(threading.get_ident())
            if len(calls) == 3: raise RuntimeError('cancelled')
        with patch.object(components, 'MIN_PARALLEL_PIXELS', 0):
            with self.assertRaisesRegex(RuntimeError, 'cancelled'):
                effective_domain.missing_regions(np.ones((100,100), np.int32), check)
        self.assertEqual(set(calls), {caller})
        self.assertFalse(any(t.name.startswith('quality-components') for t in threading.enumerate()))

    def test_worker_errors_are_consumed_in_class_order(self):
        original = components._component
        def fail(classes, code):
            if code == 2: raise ValueError('first class')
            if code == 3: raise RuntimeError('later class')
            return original(classes, code)
        with patch.object(components, 'MIN_PARALLEL_PIXELS', 0), patch.object(components, '_component', side_effect=fail):
            with self.assertRaisesRegex(ValueError, 'first class'):
                effective_domain.missing_regions(np.ones((100,100), np.int32))

    def test_small_inputs_do_not_start_threads(self):
        with patch.object(components, 'ThreadPoolExecutor', side_effect=AssertionError('unneeded pool')):
            self.assertEqual(effective_domain.missing_regions(np.ones((10,10), np.int32)), [])
