"""Pure metadata tests: no SDK, transport, hardware, UI or capture installation."""
from dataclasses import FrozenInstanceError, replace
import unittest

from femto.hardware.interface_contract import (
    Capabilities, DeviceBinding, DeviceFeedback, DeviceKind, DeviceRequest,
    FeedbackPhase, Intent, Readiness, SourceKind, validate_feedback, validate_request,
)


class HardwareInterfaceContractTests(unittest.TestCase):
    def setUp(self):
        self.binding = DeviceBinding('device-a', DeviceKind.ARM, 3, SourceKind.REAL)
        self.request = DeviceRequest('request-a', self.binding, Intent.QUERY_STATUS)
        self.feedback = DeviceFeedback('request-a', self.binding, Intent.QUERY_STATUS, 1,
                                       FeedbackPhase.RECEIVED)

    def test_unknown_capability_allows_only_metadata_queries(self):
        unknown = Capabilities(self.binding)
        self.assertIsNone(unknown.protocol_id)
        self.assertEqual(unknown.supported_intents, frozenset())
        self.assertEqual(validate_request(self.request, unknown), self.request)
        for intent in (Intent.PREPARE, Intent.REQUEST_STOP, Intent.CANCEL_REQUEST):
            request = replace(self.request, intent=intent,
                              related_request_id='previous' if intent is Intent.CANCEL_REQUEST else None)
            with self.subTest(intent=intent), self.assertRaises(ValueError):
                validate_request(request, unknown)

    def test_intent_requires_protocol_and_its_exact_declared_capability(self):
        request = replace(self.request, intent=Intent.PREPARE)
        for capabilities in (
                Capabilities(self.binding, supported_intents=frozenset({Intent.PREPARE})),
                Capabilities(self.binding, protocol_id='vendor-reference-v1')):
            with self.assertRaises(ValueError):
                validate_request(request, capabilities)
        declaration = Capabilities(self.binding, 'vendor-reference-v1', frozenset({Intent.PREPARE}))
        self.assertEqual(validate_request(request, declaration), request)
        with self.assertRaises(ValueError):
            validate_request(request, replace(declaration, binding=replace(self.binding, connection_epoch=2)))

    def test_received_is_not_completed_and_unknown_is_not_ready_or_stopped(self):
        checked = validate_feedback(self.request, self.feedback, current_binding=self.binding)
        self.assertFalse(checked.request_completed)
        self.assertIs(checked.readiness, Readiness.UNKNOWN)
        self.assertIsNone(checked.motion_stopped)
        completed = replace(checked, phase=FeedbackPhase.COMPLETED)
        self.assertTrue(completed.request_completed)
        self.assertIs(completed.readiness, Readiness.UNKNOWN)
        self.assertIsNone(completed.motion_stopped)

    def test_device_kind_identity_epoch_and_source_mismatches_are_rejected(self):
        for binding in (
                replace(self.binding, device_id='device-b'),
                replace(self.binding, kind=DeviceKind.BASE),
                replace(self.binding, connection_epoch=2),
                replace(self.binding, source=SourceKind.REPLAY),
                replace(self.binding, source=SourceKind.SIMULATION)):
            with self.subTest(binding=binding), self.assertRaises(ValueError):
                validate_feedback(self.request, replace(self.feedback, binding=binding),
                                  current_binding=self.binding)
        with self.assertRaises(ValueError):
            validate_feedback(self.request, self.feedback,
                              current_binding=replace(self.binding, connection_epoch=4))

    def test_wrong_request_intent_duplicate_and_late_sequence_are_rejected(self):
        for feedback in (replace(self.feedback, request_id='other'),
                         replace(self.feedback, intent=Intent.QUERY_CAPABILITIES)):
            with self.assertRaises(ValueError):
                validate_feedback(self.request, feedback, current_binding=self.binding)
        for previous in (1, 2):
            with self.assertRaises(ValueError):
                validate_feedback(self.request, self.feedback, current_binding=self.binding,
                                  last_sequence=previous)
        self.assertEqual(validate_feedback(self.request, self.feedback,
                         current_binding=self.binding, last_sequence=0), self.feedback)

    def test_replay_and_simulation_completion_remain_in_their_own_source(self):
        for source in (SourceKind.REPLAY, SourceKind.SIMULATION):
            binding = replace(self.binding, source=source)
            request = replace(self.request, binding=binding)
            feedback = replace(self.feedback, binding=binding, phase=FeedbackPhase.COMPLETED)
            checked = validate_feedback(request, feedback, current_binding=binding)
            self.assertIs(checked.binding.source, source)
            self.assertFalse(hasattr(checked, 'accepted_for_fusion'))
            self.assertFalse(hasattr(checked, 'execution_ready'))
            with self.assertRaises(ValueError):
                validate_feedback(self.request, checked, current_binding=self.binding)

    def test_requests_have_no_motion_or_treatment_payload_and_cancellation_is_correlated(self):
        with self.assertRaises(ValueError):
            DeviceRequest('cancel', self.binding, Intent.CANCEL_REQUEST)
        with self.assertRaises(ValueError):
            DeviceRequest('same', self.binding, Intent.CANCEL_REQUEST, 'same')
        request = DeviceRequest('cancel', self.binding, Intent.CANCEL_REQUEST, 'original')
        self.assertEqual(request.related_request_id, 'original')
        with self.assertRaises(TypeError):
            DeviceRequest('move', self.binding, Intent.PREPARE, joints=[0, 0, 0])

    def test_records_are_immutable_and_reject_ambiguous_types(self):
        with self.assertRaises(FrozenInstanceError):
            self.feedback.phase = FeedbackPhase.COMPLETED
        for values in (
                dict(connection_epoch=True), dict(device_id=' '), dict(source='simulation')):
            with self.assertRaises(ValueError):
                replace(self.binding, **values)
        for values in (
                dict(sequence=True), dict(motion_stopped=1), dict(readiness='unknown')):
            with self.assertRaises(ValueError):
                replace(self.feedback, **values)
        with self.assertRaises(ValueError):
            Capabilities(self.binding, supported_intents=frozenset({'move_joint'}))


if __name__ == '__main__':
    unittest.main()
