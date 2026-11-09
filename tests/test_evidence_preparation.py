"""Decoded data reuse must not accept changed records or outlive generation."""
from copy import deepcopy
from contextvars import copy_context
from threading import Thread
from unittest import TestCase,main
from unittest.mock import patch
import numpy as np
from femto.planning import quality_constrained as quality,evidence_preparation as preparation
from femto.annotation import selection_intent
from femto.performance_detail import collect
from test_refinement_candidates_v2 import evidence


class EvidencePreparationTests(TestCase):
    def record(self):
        classes=np.ones((100,100),np.uint8);classes[8:10,8:10]=7
        return evidence(classes)

    def test_unchanged_record_reuses_only_decode_and_returns_owned_arrays(self):
        record=self.record();expected=quality.checked_evidence(record)
        with preparation.operation(),collect() as detail,patch.object(quality,'decode_runs',wraps=quality.decode_runs) as decode:
            first=quality.checked_evidence(record);first[0][:]=6;first[1][:]=999
            for _ in range(3):
                actual=quality.checked_evidence(deepcopy(record))
                for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b);self.assertTrue(a.flags.writeable)
            self.assertEqual(decode.call_count,2)
            self.assertEqual(detail.snapshot()['rows']['route.evidence_preparation']['counts']['hits'],3)
        self.assertIsNone(preparation._state())
        with patch.object(quality,'decode_runs',wraps=quality.decode_runs) as decode:
            quality.checked_evidence(record);self.assertEqual(decode.call_count,2)

    def test_same_json_container_and_integer_subclasses_cannot_bypass_decoder(self):
        class IntSubclass(int):pass
        record=self.record()
        variants=[]
        for key in ('classes_runs','residual_um_runs'):
            bad=deepcopy(record);bad[key]=tuple(bad[key]);variants.append(bad)
            bad=deepcopy(record);bad[key][0]=IntSubclass(bad[key][0]);variants.append(bad)
        bad=deepcopy(record);bad['image_size'][0]=IntSubclass(100);variants.append(bad)
        with preparation.operation():
            quality.checked_evidence(record)
            for bad in variants:
                self.assertEqual(quality._hash(bad),record['sha256'])
                with self.assertRaises(ValueError):quality.checked_evidence(bad)

    def test_changed_sha_residual_unknown_support_and_calibration_reject_or_redecode(self):
        record=self.record()
        with preparation.operation():
            quality.checked_evidence(record)
            bad=deepcopy(record);bad['residual_um_runs'][0]=1
            with self.assertRaisesRegex(ValueError,'质量约束证据'):quality.checked_evidence(bad)
            bad['sha256']=quality._hash(bad)
            with self.assertRaisesRegex(ValueError,'伪造残差'):quality.checked_evidence(bad)
            changed=deepcopy(record);values=np.zeros((100,100),np.int32);values[50,50]=17
            changed['residual_um_runs']=selection_intent.encode_runs(values);changed['sha256']=quality._hash(changed)
            self.assertEqual(quality.checked_evidence(changed)[1][50,50],17)
            self.assertEqual(quality.checked_evidence(record)[1][50,50],0)
            bad=deepcopy(record);bad['view_reference']['undistorted_intrinsic'][0][0]=0;bad['sha256']=quality._hash(bad)
            with self.assertRaisesRegex(ValueError,'参考标定'):quality.checked_evidence(bad)

    def test_decoder_limits_and_optional_memory_failure_keep_original_rejection(self):
        record=self.record()
        with preparation.operation():
            quality.checked_evidence(record)
            with patch.object(selection_intent,'MAX_PIXELS',1),self.assertRaisesRegex(ValueError,'image size'):
                quality.checked_evidence(record)
        for target in ('_snapshot','_restore'):
            with preparation.operation():
                if target=='_restore':quality.checked_evidence(record)
                with patch.object(preparation,target,side_effect=MemoryError):
                    actual=quality.checked_evidence(record)
                    self.assertEqual(actual[0][8,8],7)
        with preparation.operation(),patch.object(preparation,'MAX_BYTES',0),patch.object(quality,'decode_runs',wraps=quality.decode_runs) as decode:
            quality.checked_evidence(record);quality.checked_evidence(record);self.assertEqual(decode.call_count,4)

    def test_failure_nested_context_and_copied_context_release_cache(self):
        record=self.record()
        with self.assertRaisesRegex(RuntimeError,'cancel'):
            with preparation.operation():
                quality.checked_evidence(record);state=preparation._state();context=copy_context()
                with preparation.operation():self.assertIs(preparation._state(),state)
                raise RuntimeError('cancel')
        self.assertFalse(state['active']);self.assertIsNone(state['entry']);self.assertIsNone(preparation._state())
        self.assertIsNone(context.run(preparation._state))
        with preparation.operation():
            quality.checked_evidence(record);context=copy_context();seen=[]
            thread=Thread(target=lambda:seen.append(context.run(preparation._state)))
            thread.start();thread.join();self.assertEqual(seen,[None])
        quality.checked_evidence(record)

    def test_record_digest_reuses_only_identical_content_and_detects_every_change(self):
        record=self.record()
        with preparation.operation(),collect() as detail,patch.object(quality,'_full_hash',wraps=quality._full_hash) as full:
            self.assertEqual(quality._hash(record),record['sha256'])
            self.assertEqual(quality._hash(record),record['sha256'])
            self.assertEqual(full.call_count,1)
            self.assertEqual(detail.snapshot()['rows']['route.evidence_digest']['counts']['hits'],1)
            changes=[]
            bad=deepcopy(record);changes.append(bad)
            bad=deepcopy(record);bad['residual_um_runs'][0]+=1;changes.append(bad)
            bad=deepcopy(record);bad['classes_runs'].append(1);changes.append(bad)
            bad=deepcopy(record);bad['depth_gate_mm']=10;changes.append(bad)
            bad=deepcopy(record);bad['view_reference']['image_size']=[1,1];changes.append(bad)
            for bad in changes:
                self.assertEqual(quality._hash(bad),quality._full_hash(bad))
            calls=full.call_count
            # In-place mutation of the hashed lists, including the JSON-distinct
            # but ==-equal True/1 substitution.
            quality._hash(record)
            index=record['classes_runs'].index(1)
            record['classes_runs'][index]=True
            self.assertNotEqual(quality._hash(record),record['sha256'])
            self.assertEqual(quality._hash(record),quality._full_hash(record))
            record['classes_runs'][index]=1
            self.assertEqual(quality._hash(record),record['sha256'])
            record['residual_um_runs'][0]+=1
            self.assertEqual(quality._hash(record),quality._full_hash(record))
            self.assertGreater(full.call_count,calls)
        self.assertIsNone(preparation._state())
        with patch.object(quality,'_full_hash',wraps=quality._full_hash) as full:
            quality._hash(record);quality._hash(record);self.assertEqual(full.call_count,2)


if __name__=='__main__':main()
