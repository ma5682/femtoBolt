"""Corruption, allocation bounds and publication semantics of B1-only assets."""
from copy import deepcopy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile

import numpy as np

from femto.planning import coverage_archive as archive, coverage_certificate as coverage, footprint_model
from femto.planning.coverage_domain import CoverageDomain
from femto.planning.planning_io import TaskControl
from femto.planning.planning_revision import digest
from test_coverage_certificate import triangle_domain, seed, run, reseal


class CoverageArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.folder = self.root/'proof'
        self.domain = triangle_domain(); self.seeds = [seed()]; self.model = footprint_model.nominal(500)
        self.result = run()

    def save(self, certificate=None, **kwargs):
        return archive.save(self.folder, certificate or self.result, self.domain, self.seeds, self.model, **kwargs)

    def read(self, **kwargs):return archive.read(self.folder, self.domain, self.seeds, self.model, **kwargs)

    def rewrite_zip(self, replace=None, *, extra=None, omit=None):
        with zipfile.ZipFile(self.folder/archive.ASSET) as source:
            items = {i.filename: source.read(i) for i in source.infolist()}
        if replace:items.update(replace)
        if omit:items.pop(omit)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            with zipfile.ZipFile(self.folder/archive.ASSET, 'w', zipfile.ZIP_DEFLATED) as target:
                for name, data in items.items():target.writestr(name, data)
                if extra:target.writestr(*extra)

    def rewrite_metadata(self, metadata):
        metadata['sha256'] = digest({k: v for k, v in metadata.items() if k != 'sha256'})
        (self.folder/archive.MANIFEST).write_text(json.dumps(metadata), encoding='utf-8')

    def test_roundtrip_recomputes_and_keeps_readonly_arrays(self):
        self.save(); restored = self.read()
        self.assertEqual(restored.metadata, self.result.metadata)
        self.assertEqual(set(p.name for p in self.folder.iterdir()), {archive.MANIFEST, archive.ASSET})
        for key in coverage.DTYPES:
            np.testing.assert_array_equal(restored.arrays[key], self.result.arrays[key])
            self.assertFalse(restored.arrays[key].flags.writeable)

    def test_existing_destination_is_never_replaced(self):
        self.folder.mkdir(); marker = self.folder/'keep'; marker.write_bytes(b'original')
        with self.assertRaises(FileExistsError):self.save()
        self.assertEqual(marker.read_bytes(), b'original')

    def test_cancel_at_commit_cleans_staging_and_installs_nothing(self):
        control = TaskControl()
        def commit(action):
            control.set(); return control.publish(action)
        with self.assertRaisesRegex(RuntimeError, '取消'):self.save(check=control.check, commit=commit)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_invalid_proof_cannot_be_published_after_rehash(self):
        values = {k: v.copy() for k, v in self.result.arrays.items()}; values['proof_parent'][0] = 0
        with self.assertRaises(ValueError):self.save(reseal(self.result, arrays=values))
        self.assertEqual(list(self.root.iterdir()), [])

    def test_budget_stop_roundtrip_preserves_incomplete_status(self):
        self.save(run(seconds=0))
        self.assertEqual(self.read().summary['status'], 'calculation_incomplete')

    def test_other_expected_input_is_rejected(self):
        self.save()
        with self.assertRaises(ValueError):
            archive.read(self.folder, triangle_domain(.25), self.seeds, self.model)

    def test_missing_or_invalid_zip_is_rejected(self):
        self.save(); path = self.folder/archive.ASSET
        path.write_bytes(b'not a zip')
        with self.assertRaises(zipfile.BadZipFile):self.read()
        path.unlink()
        with self.assertRaises(ValueError):self.read()

    def test_missing_duplicate_and_unknown_entries_are_rejected(self):
        self.save()
        original = (self.folder/archive.ASSET).read_bytes()
        for kwargs in (dict(omit='root.npy'), dict(extra=('root.npy', b'bad')), dict(extra=('../escape.npy', b'bad'))):
            with self.subTest(kwargs=kwargs):
                (self.folder/archive.ASSET).write_bytes(original)
                self.rewrite_zip(**kwargs)
                with self.assertRaises(ValueError):self.read()
        self.assertFalse((self.root/'escape.npy').exists())

    def test_object_array_is_rejected_without_pickle_loading(self):
        self.save(); output = io.BytesIO()
        np.save(output, np.array([dict(payload='object')], dtype=object))
        self.rewrite_zip({'root.npy': output.getvalue()})
        with self.assertRaises(ValueError):self.read()

    def test_forged_huge_shape_is_rejected_before_allocation(self):
        self.save(); metadata = deepcopy(self.result.metadata)
        for key in ('root', 'depth', 'code', 'status', 'witness'):metadata['arrays'][key]['shape'] = [2**40]
        self.rewrite_metadata(metadata)
        with self.assertRaisesRegex(ValueError, '尺寸'):self.read()

    def test_array_header_shape_and_exact_byte_length_are_checked(self):
        self.save(); original = (self.folder/archive.ASSET).read_bytes()
        for shape, payload in (((2**40,), b''), ((1,), b'\0'*9), ((1,), b'\0'*7)):
            with self.subTest(shape=shape, length=len(payload)):
                (self.folder/archive.ASSET).write_bytes(original)
                output = io.BytesIO()
                np.lib.format.write_array_header_1_0(output, dict(descr='<i8', fortran_order=False, shape=shape))
                self.rewrite_zip({'root.npy': output.getvalue()+payload})
                with self.assertRaises(ValueError):self.read()

    def test_compressed_size_cannot_hide_decompression_over_budget(self):
        self.save(); self.rewrite_zip({'root.npy': b'\0'*(128*1024)})
        self.assertLess((self.folder/archive.ASSET).stat().st_size, 64*1024)
        with patch('femto.planning.coverage_certificate.MAX_ARRAY_BYTES', 64*1024):
            with self.assertRaisesRegex(ValueError, '解压'):self.read()

    def test_coordinate_and_execution_claims_cannot_be_forged_by_rehashing(self):
        self.save()
        for key, value in (('coordinate_frame', 'camera'), ('unit', 'mm'), ('execution_ready', True)):
            with self.subTest(key=key):
                metadata = deepcopy(self.result.metadata); metadata[key] = value
                self.rewrite_metadata(metadata)
                with self.assertRaises(ValueError):self.read()

    def test_oversize_or_duplicate_json_fields_are_rejected(self):
        self.save(); manifest = self.folder/archive.MANIFEST
        manifest.write_bytes(b' '* (archive.MAX_MANIFEST_BYTES+1))
        with self.assertRaises(ValueError):self.read()
        manifest.write_text('{"schema":"a","schema":"b"}', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, '重复'):self.read()

    def test_new_array_hash_and_outer_hash_do_not_hide_fake_area(self):
        self.save(); metadata = deepcopy(self.result.metadata)
        metadata['summary']['confirmed_nominal_area_m2'] *= 2
        self.rewrite_metadata(metadata)
        with self.assertRaisesRegex(ValueError, '无法重现'):self.read()

    def test_cli_evaluate_and_verify_require_the_same_declared_case(self):
        from tools import check_nominal_coverage as cli
        case = dict(schema=cli.CASE_SCHEMA, coordinate_frame='world_rdf', unit='metre',
                    vertices=self.domain.vertices.tolist(), faces=self.domain.faces.tolist(),
                    seeds=self.seeds, footprint=self.model, execution_ready=False, physical_accuracy_verified=False)
        source = self.root/'case.json'; source.write_text(json.dumps(case), encoding='utf-8')
        with tempfile.TemporaryDirectory(dir=cli.ROOT/'outputs', prefix='coverage_cli_test_') as temp:
            destination = Path(temp)/'certificate'
            with redirect_stdout(io.StringIO()) as output:
                self.assertEqual(cli.main(['evaluate', '--case', str(source), '--output', str(destination)]), 0)
            self.assertEqual(json.loads(output.getvalue())['summary']['status'], 'model_coverage_complete')
            with redirect_stdout(io.StringIO()) as output:
                self.assertEqual(cli.main(['verify', '--case', str(source), '--archive', str(destination)]), 0)
            self.assertEqual(json.loads(output.getvalue())['summary']['confirmed_nominal_area_m2'], .125**2/2)

    def test_cli_refuses_experiment_publication_outside_outputs(self):
        from tools import check_nominal_coverage as cli
        from contextlib import redirect_stderr
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            cli.main(['evaluate', '--case', str(self.root/'missing.json'), '--output', str(self.root/'outside')])
        self.assertEqual(error.exception.code, 2)
        self.assertFalse((self.root/'outside').exists())


if __name__ == '__main__':unittest.main()
