"""Cooperative JSON output preserves bytes and unpublished failure boundaries."""
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from femto.io_utils import JsonSizeLimitError, write_json, write_json_strict


class CheckedJsonTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)/'中文';self.root.mkdir()
        self.target = self.root/'原清单.json';self.target.write_bytes(b'previous')
        self.value = {'中文键': ['重复值🙂' for _ in range(24000)], 'number': 1e-300}

    def preserved(self):
        self.assertEqual(self.target.read_bytes(), b'previous')
        self.assertEqual(list(self.root.iterdir()), [self.target])

    def test_all_legacy_serialization_options_keep_exact_bytes(self):
        value={'文字':'中🙂\r\n\t', 'numbers':[0.,-0.,1e-300,1e300], 'object':{'z':None,'a':True}}
        expected=self.root/'expected.json'
        for indent in (None,0,2,'\t'):
            for ascii_only in (True,False):
                with self.subTest(indent=indent,ensure_ascii=ascii_only):
                    write_json(expected,value,indent=indent,ensure_ascii=ascii_only)
                    write_json(self.target,value,indent=indent,ensure_ascii=ascii_only,check=lambda:None)
                    self.assertEqual(self.target.read_bytes(),expected.read_bytes())

    def test_limit_counts_utf8_and_platform_newlines_at_exact_boundary(self):
        value={'文字':'中🙂\n'*1000}
        expected=self.root/'expected.json';write_json(expected,value)
        raw=expected.read_bytes();expected.unlink()
        with self.assertRaises(JsonSizeLimitError):write_json(self.target,value,max_bytes=len(raw)-1)
        self.preserved()
        write_json(self.target,value,max_bytes=len(raw))
        self.assertEqual(self.target.read_bytes(),raw)

    def test_cancellation_after_partial_write_preserves_original(self):
        def check():
            if any(p.stat().st_size>0 for p in self.root.glob('*.tmp')):
                raise RuntimeError('cancelled during output')
        with self.assertRaisesRegex(RuntimeError,'cancelled during output'):
            write_json_strict(self.target,self.value,check=check)
        self.preserved()

    def test_deadline_after_close_still_prevents_publication(self):
        def check():
            for path in self.root.glob('*.tmp'):
                try:complete=json.loads(path.read_bytes())==self.value
                except (ValueError,OSError):complete=False
                if complete:raise TimeoutError('deadline before publication')
        with self.assertRaisesRegex(TimeoutError,'deadline before publication'):
            write_json_strict(self.target,self.value,check=check)
        self.preserved()

    def test_partial_disk_write_failure_is_not_published(self):
        original=tempfile.NamedTemporaryFile
        def faulty(*args,**kwargs):
            stream=original(*args,**kwargs);write=stream.write
            def fail(data):write(data[:len(data)//2]);raise OSError('disk full')
            stream.write=fail
            return stream
        with patch('femto.io_utils.tempfile.NamedTemporaryFile',side_effect=faulty):
            with self.assertRaisesRegex(OSError,'disk full'):
                write_json_strict(self.target,self.value,check=lambda:None)
        self.preserved()

    def test_durable_flush_failure_preserves_previous_file(self):
        with patch('femto.io_utils.os.fsync',side_effect=OSError('flush failed')):
            with self.assertRaisesRegex(OSError,'flush failed'):
                write_json_strict(self.target,{'n':1},check=lambda:None,durable=True)
        self.preserved()

    def test_durable_streaming_uses_small_writes_and_syncs_before_replace(self):
        sizes=[];synced=[];original=tempfile.NamedTemporaryFile;replace=os.replace
        def observed(*args,**kwargs):
            stream=original(*args,**kwargs);write=stream.write
            def record(data):sizes.append(len(data));return write(data)
            stream.write=record
            return stream
        def publish(source,target):
            self.assertEqual(synced,[True]);self.assertEqual(json.loads(Path(source).read_bytes()),self.value)
            return replace(source,target)
        with patch('femto.io_utils.tempfile.NamedTemporaryFile',side_effect=observed), \
             patch('femto.io_utils.os.fsync',side_effect=lambda _:synced.append(True)), \
             patch('femto.io_utils.os.replace',side_effect=publish):
            write_json_strict(self.target,self.value,check=lambda:None,durable=True)
        self.assertGreater(len(sizes),1);self.assertLessEqual(max(sizes),128*1024)
        self.assertEqual(json.loads(self.target.read_bytes()),self.value)

    def test_invalid_or_nonfinite_input_never_replaces_previous_file(self):
        circular=[];circular.append(circular)
        for value in ({'n':math.nan},{'n':math.inf},circular,{'raw':b'bytes'}):
            with self.assertRaises((ValueError,TypeError)):
                write_json_strict(self.target,value,check=lambda:None)
            self.preserved()

    def test_invalid_byte_limits_are_rejected(self):
        for limit in (-1,True,1.5):
            with self.assertRaises(ValueError):write_json(self.target,{},max_bytes=limit)
            self.preserved()

    def test_batched_unicode_and_exact_limit_keep_original_bytes(self):
        value={'rows':[{'label':'中文🙂'*7,'v':i} for i in range(2500)],'large':'文🙂'*12000}
        expected=self.root/'expected.json';write_json(expected,value,indent=None)
        raw=expected.read_bytes();expected.unlink()
        with self.assertRaises(JsonSizeLimitError):write_json(self.target,value,indent=None,check=lambda:None,max_bytes=len(raw)-1)
        self.preserved()
        write_json(self.target,value,indent=None,check=lambda:None,max_bytes=len(raw))
        self.assertEqual(self.target.read_bytes(),raw)

    def test_cancellation_inside_one_large_string_preserves_original(self):
        def check():
            if any(p.stat().st_size>0 for p in self.root.glob('*.tmp')):
                raise RuntimeError('cancelled large string')
        with self.assertRaisesRegex(RuntimeError,'cancelled large string'):
            write_json_strict(self.target,{'text':'中文🙂'*200000},check=check)
        self.preserved()


if __name__=='__main__':unittest.main()
