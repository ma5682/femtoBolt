"""One-task preparation equivalence, mutation, isolation and budget contracts."""
from copy import deepcopy
from contextvars import copy_context
from threading import Thread
from unittest import TestCase, main
from unittest.mock import patch
import numpy as np
from femto.annotation import selection_intent
from femto.planning import evidence_preparation as operation_state
from femto.planning import task_preparation as preparation
from femto.planning import selection_blocks as selection
from femto.planning.quality_constrained import ReferenceEnvelope
from femto.performance_detail import collect
from test_refinement_candidates_v2 import evidence


class TaskPreparationTests(TestCase):
    def record(self):
        classes=np.ones((100,100),np.uint8);classes[4:8,4:8]=7
        record=evidence(classes)
        residual=np.zeros(classes.shape,np.int32);residual[20:80,20:80]=123
        record['residual_um_runs']=selection_intent.encode_runs(residual)
        from femto.planning.quality_constrained import _hash
        record['sha256']=_hash(record)
        return record

    def region(self):
        mask=np.ones((100,100),bool);mask[5:10,5:10]=False
        region=dict(view_reference=dict(image_size=[100,100],undistorted_intrinsic=
            [[100.,0.,50.],[0.,100.,50.],[0.,0.,1.]]),
            selection_intent=selection_intent.build_selection_intent(mask,method='work_only'))
        vertices=np.array([[-.01,-.01,1.],[.01,-.01,1.],[0.,.01,1.]])
        faces=np.array([[0,1,2]],np.int64)
        return region,vertices,faces

    def test_integral_and_tiles_shared_but_every_envelope_charges_original_work(self):
        record=self.record()
        expected=[ReferenceEnvelope(record,10.) for _ in range(3)]
        with operation_state.operation(),collect() as detail:
            actual=[ReferenceEnvelope(deepcopy(record),10.) for _ in range(3)]
            for new,old in zip(actual,expected):
                for rect in [(0,0,99,99),(14,15,82,81),(-40,-3,20,40)]:
                    self.assertEqual(new._residual_max(rect),old._residual_max(rect))
                    self.assertEqual(new.sampled_pixels,old.sampled_pixels)
                np.testing.assert_array_equal(new.integral,old.integral)
                with self.assertRaises(ValueError):new.integral.setflags(write=True)
            self.assertIs(actual[0].integral,actual[1].integral)
            self.assertIs(actual[0]._shared_residual_tiles,actual[1]._shared_residual_tiles)
            counts=detail.snapshot()['rows']['route.task_preparation']['counts']
            self.assertEqual(counts['reference_integral_builds'],1)
            self.assertEqual(counts['reference_integral_hits'],2)
            self.assertEqual(counts['residual_tiles_builds'],1)
            # An already prepared tile cannot make the first logical charge free.
            failed=ReferenceEnvelope(record,10.);failed.sampled_pixels=100000000
            with self.assertRaisesRegex(ValueError,'超过预算'):failed._residual_max((0,0,99,99))

    def test_array_mutation_dtype_and_prepared_storage_are_bound(self):
        classes=np.ones((20,20),np.int32)
        with operation_state.operation():
            first=preparation.integral_image(classes)
            self.assertIs(first,preparation.integral_image(classes.copy()))
            classes[5,5]=7
            changed=preparation.integral_image(classes)
            self.assertEqual(first[-1,-1],0);self.assertEqual(changed[-1,-1],1)
            converted=preparation.integral_image(classes.astype(np.int64))
            self.assertIsNot(converted,changed)
            np.testing.assert_array_equal(converted,changed)
            with self.assertRaises(ValueError):converted.view().setflags(write=True)

    def test_selection_reuses_preparation_and_returns_owned_values(self):
        region,vertices,faces=self.region()
        expected=selection.selection_face_assignment(region,vertices,faces)
        with operation_state.operation(),patch.object(selection,'validate_selection_intent',
                wraps=selection.validate_selection_intent) as validate:
            first=selection.selection_face_assignment(region,vertices,faces)
            second=selection.selection_face_assignment(deepcopy(region),vertices.copy(),faces.copy())
            second[0]['components'][0]['pixel_count']=1;second[2][:]=0;second[3][:]=99
            third=selection.selection_face_assignment(region,vertices,faces)
            self.assertEqual(validate.call_count,1)
            self.assertEqual(third[:2],expected[:2])
            for new,old in zip(third[2:],expected[2:]):
                np.testing.assert_array_equal(new,old);self.assertTrue(new.flags.writeable)
            self.assertIsNot(first[2],third[2])

    def test_selection_detects_changed_geometry_intent_types_and_limits(self):
        region,vertices,faces=self.region()
        with operation_state.operation():
            selection.selection_face_assignment(region,vertices,faces)
            moved=vertices.copy();moved[:,0]=10.
            with self.assertRaisesRegex(ValueError,'outside the selection'):
                selection.selection_face_assignment(region,moved,faces)
            selection.selection_face_assignment(region,vertices,faces)
            bad=deepcopy(region);bad['selection_intent']['components'][0]['pixel_count']+=1
            with self.assertRaisesRegex(ValueError,'components/holes'):
                selection.selection_face_assignment(bad,vertices,faces)
            class IntSubclass(int):pass
            for value in (True,IntSubclass(1)):
                bad=deepcopy(region);bad['selection_intent']['selected_mask_runs'][0]=value
                with self.assertRaises(ValueError):selection.selection_face_assignment(bad,vertices,faces)
            selection.selection_face_assignment(region,vertices,faces)
            with patch.object(selection_intent,'MAX_PIXELS',1),self.assertRaisesRegex(ValueError,'image size'):
                selection.selection_face_assignment(region,vertices,faces)
            with patch.object(selection_intent,'MISSING_GEOMETRY_POLICY','changed'),self.assertRaisesRegex(ValueError,'missing geometry policy'):
                selection.selection_face_assignment(region,vertices,faces)

    def test_selection_binds_support_and_intrinsic_and_memory_failure_recomputes(self):
        from types import SimpleNamespace
        region,vertices,faces=self.region()
        mask=selection_intent.intent_mask(region['selection_intent'])
        projection=SimpleNamespace(reasons=np.zeros(mask.shape,np.uint8),depth=np.full(mask.shape,1000),
                                   minimum=.2,maximum=2.)
        region['selection_support']=selection_intent.build_selection_support(mask,projection)
        with operation_state.operation():
            expected=selection.selection_face_assignment(region,vertices,faces)
            changed=deepcopy(region);changed['selection_support']['geometry_supported_pixel_count']+=1
            with self.assertRaisesRegex(ValueError,'counts'):
                selection.selection_face_assignment(changed,vertices,faces)
            selection.selection_face_assignment(region,vertices,faces)
            changed=deepcopy(region);changed['view_reference']['undistorted_intrinsic'][0][2]=10000.
            with self.assertRaisesRegex(ValueError,'outside the selection'):
                selection.selection_face_assignment(changed,vertices,faces)
        with operation_state.operation(),patch.object(preparation,'frozen',side_effect=MemoryError):
            actual=selection.selection_face_assignment(region,vertices,faces)
            self.assertEqual(actual[:2],expected[:2])
            for new,old in zip(actual[2:],expected[2:]):np.testing.assert_array_equal(new,old)

    def test_context_thread_and_optional_memory_limits(self):
        classes=np.ones((20,20),np.int32)
        with operation_state.operation():
            first=preparation.integral_image(classes);state=operation_state._state();context=copy_context()
            with operation_state.operation():self.assertIs(first,preparation.integral_image(classes))
            seen=[]
            thread=Thread(target=lambda:seen.append(context.run(preparation.integral_image,classes)))
            thread.start();thread.join();self.assertIsNot(first,seen[0])
        self.assertFalse(state['active']);self.assertEqual(state['prepared'],{})
        self.assertIsNot(first,context.run(preparation.integral_image,classes))
        with operation_state.operation(),patch.object(preparation,'MAX_BYTES',0):
            self.assertIsNot(preparation.integral_image(classes),preparation.integral_image(classes))
        with operation_state.operation(),patch.object(preparation,'frozen',side_effect=MemoryError):
            np.testing.assert_array_equal(preparation.integral_image(classes),np.zeros((21,21),np.int64))

    def test_nonordinary_and_recursive_records_fall_back_without_a_snapshot(self):
        with operation_state.operation():
            self.assertIsNone(preparation.record_key({'runs':(1,2)}))
            class IntSubclass(int):pass
            self.assertIsNone(preparation.record_key({'runs':[IntSubclass(1),2]}))
            recursive=[];recursive.append(recursive)
            self.assertIsNone(preparation.record_key(recursive))

    def test_custom_metaclass_cannot_impersonate_builtin_types(self):
        from femto.planning.quality_constrained import checked_evidence, _hash
        class PretendInt(type):
            def __hash__(cls):return hash(int)
            def __eq__(cls,other):return other is int or other is cls
        class IntSubclass(int,metaclass=PretendInt):pass
        class PretendList(type):
            def __hash__(cls):return hash(list)
            def __eq__(cls,other):return other is list or other is cls
        class ListSubclass(list,metaclass=PretendList):pass
        # Include int first: a set of type objects would discard the impostor.
        self.assertFalse(operation_state._plain_ints([1,IntSubclass(1),2]))
        record=self.record()
        with operation_state.operation():
            checked_evidence(record)
            for key in ('classes_runs','residual_um_runs'):
                damaged=deepcopy(record);damaged[key][0]=IntSubclass(damaged[key][0])
                self.assertEqual(_hash(damaged),record['sha256'])
                with self.assertRaises(ValueError):checked_evidence(damaged)
            for damaged in ([1,IntSubclass(1)],{'value':IntSubclass(1)},
                            [[],ListSubclass([1,2])],{'value':ListSubclass([1,2])}):
                self.assertIsNone(preparation.record_key(damaged))
            region,vertices,faces=self.region()
            selection.selection_face_assignment(region,vertices,faces)
            region['selection_intent']['selected_mask_runs'][0]=IntSubclass(
                region['selection_intent']['selected_mask_runs'][0])
            with self.assertRaises(ValueError):selection.selection_face_assignment(region,vertices,faces)

    def test_optional_fingerprint_allocation_failure_uses_original_integral(self):
        classes=np.ones((20,20),np.int32);classes[5,5]=7
        expected=np.pad((classes!=1).astype(np.int64),((1,0),(1,0))).cumsum(0).cumsum(1)
        with operation_state.operation(),patch.object(preparation.hashlib,'sha256',side_effect=MemoryError):
            self.assertIsNone(preparation.array_key(classes))
            self.assertIsNone(preparation.record_key({'runs':[0,20,1,10]}))
            np.testing.assert_array_equal(preparation.integral_image(classes),expected)

    def test_selection_policy_binds_nested_pixel_projection_and_topology_helpers(self):
        region,vertices,faces=self.region()
        with operation_state.operation():
            for name in ('pixel_indices','_as_mask','_bbox'):
                selection.selection_face_assignment(region,vertices,faces)
                with patch.object(selection_intent,name,side_effect=RuntimeError('changed helper')):
                    with self.assertRaisesRegex(RuntimeError,'changed helper'):
                        selection.selection_face_assignment(region,vertices,faces)


if __name__=='__main__':main()
