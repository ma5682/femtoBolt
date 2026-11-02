"""One full target reconstruction per open, also for a plan without coverage."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from test_coverage_workflow import fixture
from femto.planning import coverage_workflow, region_project, target_surface


class SingleTargetReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name)
        model,region,plan,certificate,ledger=fixture(cls.root/'source')
        model.store.regions=[region]
        verified=coverage_workflow.attach(model,region,plan,certificate,ledger)
        cls.p2=region_project.save_project(model,region,cls.root/'covered.zip',verified)
        cls.p1=region_project.save_project(model,region,cls.root/'target_only.zip',plan)
        cls.ledger=ledger

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_target_is_fully_reconstructed_once_and_shared_with_coverage_reader(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(target_surface,'read',wraps=target_surface.read) as reader, \
                 patch.object(target_surface,'validate',wraps=target_surface.validate) as validator, \
                 patch.object(coverage_workflow,'read',wraps=coverage_workflow.read) as coverage:
                model=region_project.open_project(self.p2,workspace=Path(folder))
            self.assertEqual(reader.call_count,1);self.assertEqual(validator.call_count,1)
            self.assertIs(coverage.call_args.args[3],model.target_ledger)
            for name,value in self.ledger.arrays.items():np.testing.assert_array_equal(value,model.target_ledger.arrays[name])

    def test_target_without_coverage_still_requires_full_reconstruction(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(target_surface,'validate',wraps=target_surface.validate) as validator:
                model=region_project.open_project(self.p1,workspace=Path(folder)/'opened')
            self.assertEqual(validator.call_count,1);self.assertIsNotNone(model.target_ledger)
            self.assertIsNone(model.coverage_certificate)
            with patch.object(target_surface,'validate',side_effect=ValueError('target reconstruction failed')):
                with self.assertRaisesRegex(ValueError,'target reconstruction failed'):
                    region_project.open_project(self.p1,workspace=Path(folder)/'rejected')

    def test_rehashed_corrupt_target_is_rejected_before_coverage_replay(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'corrupt.zip'
            with zipfile.ZipFile(self.p2) as archive:files={name:archive.read(name) for name in archive.namelist()}
            files[target_surface.ASSET]=b'not a target array archive'
            data=json.loads(files['project.json'])
            data['files']={name:hashlib.sha256(raw).hexdigest() for name,raw in files.items() if name!='project.json'}
            files['project.json']=json.dumps(data,ensure_ascii=False).encode('utf-8')
            with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as archive:
                for name,raw in files.items():archive.writestr(name,raw)
            with patch.object(coverage_workflow,'read',side_effect=AssertionError('corrupt target reached coverage')) as coverage:
                with self.assertRaises((ValueError,zipfile.BadZipFile)):
                    region_project.open_project(path,workspace=Path(folder)/'opened')
            self.assertFalse(coverage.called)


if __name__=='__main__':unittest.main()
