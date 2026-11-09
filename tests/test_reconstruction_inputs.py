"""Real temporary RGB-D file evidence, including non-reference inputs."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import unittest

import numpy as np

from femto.dataset.reconstruction_inputs import snapshot,verify,InputSourceChanged
from femto.geometry.local_region import load_reference,save_request
from femto.io_utils import load_jsonl,sha256_file
from test_region_first_reconstruction import dataset_fixture


class ReconstructionInputTests(unittest.TestCase):
    def setUp(self):
        temporary=TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name)
        self.capture,self.dataset=dataset_fixture(self.root,3,width=80,height=60)
        self.rows=load_jsonl(self.dataset/'frames.jsonl')

    def test_records_all_admitted_files_and_does_not_change_sources(self):
        before={str(p):sha256_file(p) for p in self.capture.session.rglob('*') if p.is_file()}
        record=snapshot(self.dataset);verify(record)
        self.assertEqual(record['frame_ids'],[0,1,2])
        for row in self.rows:
            for key in ('rgb','depth','pose'):
                path=(self.dataset/row[key]).resolve()
                self.assertEqual(record['files'][str(path)],sha256_file(path))
        self.assertTrue(all(sha256_file(p)==value for p,value in before.items()))

    def test_nonreference_rgb_depth_pose_calibration_and_index_changes_are_detected(self):
        record=snapshot(self.dataset)
        paths=[self.dataset/self.rows[-1][key] for key in ('rgb','depth','pose')]
        paths.extend([self.dataset/'calibration/depth_to_color.json',self.dataset/'frames.jsonl'])
        for path in paths:
            with self.subTest(path=path):
                original=path.read_bytes();path.write_bytes(original+b'\n')
                try:
                    with self.assertRaises(InputSourceChanged):verify(record)
                finally:path.write_bytes(original)
        verify(record)

    def test_missing_input_is_not_treated_as_a_hash_match(self):
        record=snapshot(self.dataset)
        path=(self.dataset/self.rows[-1]['depth']).resolve()
        hidden=path.with_suffix('.temporarily_unavailable')
        self.assertTrue(path.is_relative_to(self.root));self.assertTrue(hidden.is_relative_to(self.root))
        path.rename(hidden)
        try:
            with self.assertRaises(FileNotFoundError):verify(record)
        finally:hidden.rename(path)
        verify(record)

    def test_changed_record_and_cancelled_verification_are_rejected(self):
        from femto.annotation.fixed_rgb_geometry import GeometryBudget,GeometryTaskCancelled
        record=snapshot(self.dataset)
        altered=deepcopy(record);altered['files'].pop(next(iter(altered['files'])))
        with self.assertRaisesRegex(ValueError,'摘要'):verify(altered)
        cancelled=threading.Event();cancelled.set()
        with self.assertRaises(GeometryTaskCancelled):verify(record,GeometryBudget(cancelled).check)

    def test_changed_input_during_reconstruction_cannot_publish_a_complete_report(self):
        from femto.pipeline.scene_reconstruction import reconstruct_scene
        ref=load_reference(self.dataset)
        mask=np.zeros(ref.depth.shape,np.uint8);mask[20:40,30:50]=1
        request=save_request(ref,mask,self.root/'request')
        path=self.dataset/self.rows[-1]['depth'];original=path.read_bytes()
        changed=[]
        def observer(stage,mesh):
            if stage=='final_components_filtered':
                path.write_bytes(original+b'\n');changed.append(stage)
        output=self.root/'reconstruction'
        try:
            with self.assertRaises(InputSourceChanged):
                reconstruct_scene(self.dataset,output,depth_min_m=.25,depth_max_m=2.21,
                                  region_request=request,geometry_observer=observer)
        finally:path.write_bytes(original)
        self.assertEqual(changed,['final_components_filtered'])
        self.assertFalse((output/'reconstruction_report.json').exists())


if __name__=='__main__':unittest.main()
