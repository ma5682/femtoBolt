"""Opt-in timing cannot change geometry values, exceptions or evidence."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from femto import performance_detail as detail
from femto.operation_diagnostics import record_operation


class PerformanceDetailTests(unittest.TestCase):
    def test_nested_contexts_are_isolated_and_persist_only_to_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            with detail.collect() as outer:
                with record_operation('test',root=directory) as operation:
                    detail.count('batches',one=3)
                    with detail.collect() as inner:detail.count('batches',one=5)
                    self.assertEqual(inner.snapshot()['rows']['batches']['counts']['one'],5)
            payload=json.loads((operation.directory/'operation.json').read_text(encoding='utf-8'))
            self.assertEqual(payload['performance_detail'],outer.snapshot())
            self.assertEqual(outer.snapshot()['rows']['batches']['counts']['one'],3)
            self.assertIsNone(detail.current())

    def test_disabled_and_failed_recording_keep_result_and_original_exception(self):
        @detail.observed('measured')
        def function(value):
            if isinstance(value,Exception):raise value
            return value
        expected=object();self.assertIs(function(expected),expected)
        error=ValueError('original')
        with detail.collect() as active,patch.object(active,'add',side_effect=OSError('diagnostic failure')):
            self.assertIs(function(expected),expected)
            with self.assertRaises(ValueError) as result:function(error)
            self.assertIs(result.exception,error)
            self.assertEqual(active.omitted,2)

    def test_aggregate_memory_is_bounded(self):
        with detail.collect() as active:
            for index in range(1000):detail.count(str(index),queries=1)
        self.assertEqual(len(active.rows),detail.MAX_ROWS)
        self.assertEqual(active.omitted,1000-detail.MAX_ROWS)


if __name__=='__main__':unittest.main()
