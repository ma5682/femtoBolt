import gc
import sys
import unittest
import weakref
from unittest.mock import patch

from tools.arena_object_census import arena_index, census, type_name, shallow_size


class CensusTests(unittest.TestCase):
    def test_extension_style_module_descriptor_and_custom_size_are_safe(self):
        class Value:
            __module__ = property(lambda self: 'not-a-module-string')
            def __sizeof__(self):
                raise AssertionError('Application method must not be called')
        value=Value()
        self.assertTrue(type_name(value).startswith('<non_string_module>.'))
        self.assertGreater(shallow_size(value),0)
        result=census([(id(value),sys.getsizeof(object()))],sample_limit=1)
        self.assertEqual(result['tracked_objects_in_known_arenas'],1)

    def test_range_edges_and_gaps(self):
        ranges = [(100, 20), (200, 30)]
        self.assertEqual(arena_index(100, ranges, [100, 200]), 0)
        self.assertEqual(arena_index(229, ranges, [100, 200]), 1)
        self.assertIsNone(arena_index(120, ranges, [100, 200]))
        self.assertIsNone(arena_index(99, ranges, [100, 200]))

    def test_sampling_focus_does_not_hide_total_census(self):
        value=[]
        result=census([(id(value),sys.getsizeof(value))],focus_bases=set())
        self.assertEqual(result['tracked_objects_in_known_arenas'],1)
        self.assertEqual(result['samples'],[])

    def test_result_does_not_retain_sampled_object_or_force_collection(self):
        class Value:
            pass
        value = Value()
        reference = weakref.ref(value)
        with patch('gc.collect', side_effect=AssertionError('No forced GC')):
            result = census([(id(value), sys.getsizeof(value))], sample_limit=2)
        self.assertGreaterEqual(result['tracked_objects_in_known_arenas'], 1)
        self.assertEqual(result['samples'][0]['type'], type(value).__module__+'.'+type(value).__qualname__)
        del value
        self.assertIsNone(reference())

    def test_overlap_or_unsafe_bounds_rejected(self):
        with self.assertRaises(ValueError):
            census([(100, 100), (150, 100)])
        with self.assertRaises(ValueError):
            census([], sample_limit=100)

    def test_module_root_and_values_are_not_serialized(self):
        value = []
        globals()['_census_fixture'] = value
        try:
            result = census([(id(value), sys.getsizeof(value))], sample_limit=1)
            self.assertTrue(any(__name__+'.__dict__' in path
                                for path in result['samples'][0]['module_paths']))
            self.assertTrue(gc.isenabled())
        finally:
            globals().pop('_census_fixture')
