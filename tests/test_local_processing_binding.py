"""Result/report lineage tests using real temporary manifests and W0 proofs.

These are software provenance tests, not camera/physical accuracy validation.
No SDK, reconstruction worker, application state, or real latest index is used.
"""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from local_processing import SCHEMA, empty_report, format_quality, result_quality, temporal_summary
from local_reference import CALIBRATION_FILES, calibration_identity, reference_provenance


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


class ProcessingBindingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.online = self.dataset('online')

    def dataset(self, name, *, session='session-a', reference=0,
                pose_source='online_rgbd_research_grade', refinement_source=None, masked=()):
        path = self.root/name
        for filename in CALIBRATION_FILES:
            write_json(path/'calibration'/filename,
                       dict(test_fixture=True, file=filename, width=8, height=6))
        calibration = calibration_identity(path)
        proof = reference_provenance(session,
            dict(frame_id=reference, reference_frame_id=reference, state='tracking',
                 T_WC=np.eye(4).tolist(), capture_timestamp_us=1000000,
                 color_timestamp_us=1000000, depth_timestamp_us=1000000), calibration)
        frames = []
        for index, frame_id in enumerate((reference, reference+10, reference+20)):
            timestamp = 1000000+index*500000
            transform = np.eye(4)
            transform[0, 3] = index*.005
            frame = dict(frame=frame_id, rgb=f'rgb/{frame_id:06}.png',
                         depth=f'depth/{frame_id:06}.png', pose=f'pose/{frame_id:06}.json',
                         camera_timestamp_us=timestamp, color_timestamp_us=timestamp,
                         depth_timestamp_us=timestamp)
            for folder in ('rgb', 'depth'):
                (path/folder).mkdir(exist_ok=True)
            Image.new('RGB', (8, 6), (40, 80, 120)).save(path/frame['rgb'])
            Image.fromarray(np.full((6, 8), 500, dtype=np.uint16)).save(path/frame['depth'])
            write_json(path/frame['pose'], dict(T_WC=transform.tolist(), pose_valid=True,
                       session_id=session, reference_frame_id=reference, pose_source=pose_source,
                       reference_provenance=proof))
            if frame_id in masked:
                mask = path/f'mask_{frame_id:06}.png'
                Image.new('L', (8, 6), 255).save(mask)
                frame['local_temporal_support'] = dict(mask=str(mask),
                    sha256=hashlib.sha256(mask.read_bytes()).hexdigest(),
                    source_frames=list(range(frame_id, frame_id+5)),
                    rule='4_of_5_within_10mm_of_selected_raw_depth')
            frames.append(frame)
        (path/'frames.jsonl').write_text(''.join(json.dumps(f)+'\n' for f in frames), encoding='utf-8')
        manifest = dict(schema='femto-full-scene-rgbd-v1', data_kind='synthetic',
                        pose=dict(valid=True, source=pose_source),
                        local_capture=dict(measured_only=True, session_id=session,
                            reference_frame_id=reference, reference_provenance=proof,
                            calibration=calibration, local_workspace_radius_m=.02))
        if refinement_source is not None:
            manifest['refinement_source_dataset'] = str(refinement_source)
        write_json(path/'manifest.json', manifest)
        return path

    def model(self, actual, *, name='model'):
        root = self.root/name
        local = read_json(actual/'manifest.json')['local_capture']
        write_json(root/'reconstruction/reconstruction_report.json',
                   dict(source_dataset=str(actual), data_kind='synthetic'))
        write_json(root/'reconstruction/local_geometry.json',
                   dict(schema='femto-local-geometry-v1', session_id=local['session_id'],
                        reference_frame_id=local['reference_frame_id'],
                        reference_provenance=local['reference_provenance'],
                        coordinate_frame='world_rdf', unit='metre', transform_direction='T_W0_Ct'))
        return root

    def attach_refinement(self, actual, source, *, accepted):
        report = dict(schema='femto-local-refinement-v1', source_dataset=str(source),
                      accepted=accepted, reference_fixed=True,
                      frames=[dict(frame=10, accepted=accepted,
                          reasons=[] if accepted else ['insufficient_heldout_improvement'])])
        path = actual.parent/(actual.name+'_refinement.json')
        write_json(path, report)
        manifest = read_json(actual/'manifest.json')
        manifest['local_capture']['refinement_report'] = str(path)
        write_json(actual/'manifest.json', manifest)
        return report

    def linked(self, model, actual, **updates):
        manifest = read_json(actual/'manifest.json')
        local = manifest['local_capture']
        value = dict(empty_report(), state='complete', actual_dataset=str(actual),
                     actual_pose_source=manifest['pose']['source'],
                     source_session_id=local['session_id'], reference_frame_id=local['reference_frame_id'],
                     coordinate_frame='world_rdf', unit='metre', transform_direction='T_W0_Ct')
        value.update(updates)
        write_json(model/'processing_report.json', value)

    def assert_untrusted(self, value):
        self.assertEqual(value['refinement']['status'], 'not_recorded')
        self.assertEqual(value['temporal']['status'], 'not_recorded')
        self.assertTrue(value.get('read_error'))

    def test_legacy_foreign_report_cannot_upgrade_online_pose(self):
        foreign = self.dataset('foreign', session='session-b')
        self.attach_refinement(self.online, foreign, accepted=True)
        self.assert_untrusted(result_quality(self.model(self.online)))

    def test_legacy_refined_source_must_have_matching_session_and_reference(self):
        for name, session, reference in [('other_session', 'session-b', 0),
                                         ('other_reference', 'session-a', 7)]:
            with self.subTest(name=name):
                foreign = self.dataset(name, session=session, reference=reference)
                actual = self.dataset('refined_'+name, pose_source='offline_robust_icp_research_grade',
                                      refinement_source=foreign)
                self.attach_refinement(actual, foreign, accepted=True)
                self.assert_untrusted(result_quality(self.model(actual, name='model_'+name)))

    def test_legacy_rejected_report_for_another_input_is_not_this_models_attempt(self):
        same_session_other_input = self.dataset('other_input')
        self.attach_refinement(self.online, same_session_other_input, accepted=False)
        self.assert_untrusted(result_quality(self.model(self.online)))

    def test_valid_legacy_rejection_remains_visible_and_read_only(self):
        self.attach_refinement(self.online, self.online, accepted=False)
        model = self.model(self.online)
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in self.root.rglob('*') if p.is_file()}
        quality = result_quality(model)
        self.assertEqual(quality['refinement']['status'], 'not_adopted')
        self.assertEqual(quality['actual_pose_source'], 'online_rgbd_research_grade')
        self.assertNotIn('read_error', quality)
        self.assertIn('验证采样改善不足', format_quality(quality))
        self.assertIn('Held-out improvement below', format_quality(quality, 'en'))
        self.assertEqual(before, {p: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in self.root.rglob('*') if p.is_file()})

    def test_valid_legacy_adoption_preserves_its_original_w0_proof(self):
        actual = self.dataset('refined', pose_source='offline_robust_icp_research_grade',
                              refinement_source=self.online)
        self.attach_refinement(actual, self.online, accepted=True)
        quality = result_quality(self.model(actual))
        self.assertEqual(quality['refinement']['status'], 'adopted')
        self.assertEqual(quality['actual_pose_source'], 'offline_robust_icp_research_grade')
        self.assertNotIn('read_error', quality)
        self.assertEqual(read_json(actual/'manifest.json')['local_capture']['reference_provenance'],
                         read_json(self.online/'manifest.json')['local_capture']['reference_provenance'])

    def test_linked_report_pose_source_claim_must_match_actual_manifest(self):
        model = self.model(self.online)
        self.linked(model, self.online, actual_pose_source='offline_robust_icp_research_grade',
                    refinement=dict(status='adopted'))
        self.assert_untrusted(result_quality(model))

    def test_linked_adoption_cannot_override_actual_online_pose_source(self):
        details = self.attach_refinement(self.online, self.online, accepted=True)
        model = self.model(self.online)
        self.linked(model, self.online, refinement=dict(status='adopted', details=details))
        self.assert_untrusted(result_quality(model))

    def test_linked_adopted_label_cannot_override_rejected_refinement_details(self):
        actual = self.dataset('refined', pose_source='offline_robust_icp_research_grade',
                              refinement_source=self.online)
        details = self.attach_refinement(actual, self.online, accepted=False)
        model = self.model(actual)
        self.linked(model, actual, refinement=dict(status='adopted', details=details))
        self.assert_untrusted(result_quality(model))

    def test_empty_or_partial_temporal_report_does_not_invent_zero_metrics(self):
        for report in ({}, {'frames': []}, {'frames': None}, {'frames': [{}]}):
            with self.subTest(report=report):
                summary = temporal_summary(report)
                self.assertEqual(summary['status'], 'not_recorded')
                for key in ('accepted_views', 'windows_found', 'applied_views',
                            'removed_pixels', 'removed_ratio'):
                    self.assertIsNone(summary[key])
                rendered = format_quality(dict(empty_report(), temporal=summary))
                self.assertNotIn('0.0%', rendered)

    def test_missing_applied_pixel_metrics_remain_unknown(self):
        summary = temporal_summary(dict(applied_frames=1,
            frames=[dict(frame=0, applied=True, source_frames=[0, 1, 2, 3, 4])]))
        self.assertEqual(summary['applied_views'], 1)
        self.assertIsNone(summary['removed_pixels'])
        self.assertIsNone(summary['removed_ratio'])

    def test_declared_temporal_applied_count_must_match_frame_evidence(self):
        with self.assertRaisesRegex(ValueError, 'applied count'):
            temporal_summary(dict(applied_frames=0, frames=[dict(frame=0, applied=True)]))

    def test_linked_temporal_applied_count_and_frame_ids_match_actual_masks(self):
        actual = self.dataset('masked', masked=(0,))
        model = self.model(actual)
        valid = temporal_summary(dict(applied_frames=1, frames=[dict(frame=0, applied=True,
            source_frames=[0, 1, 2, 3, 4], valid_pixels=48, removed_pixels=0)]))
        self.linked(model, actual, temporal=valid)
        self.assertEqual(result_quality(model)['temporal']['status'], 'adopted')
        for mismatch in ('count', 'frame'):
            with self.subTest(mismatch=mismatch):
                invalid = copy.deepcopy(valid)
                if mismatch == 'count':
                    invalid['applied_views'] = 2
                else:
                    invalid['frames'][0]['frame'] = 10
                self.linked(model, actual, temporal=invalid)
                self.assert_untrusted(result_quality(model))

    def test_linked_applied_temporal_report_without_actual_mask_is_rejected(self):
        model = self.model(self.online)
        temporal = temporal_summary(dict(applied_frames=1, frames=[dict(frame=0, applied=True,
            source_frames=[0, 1, 2, 3, 4], valid_pixels=48, removed_pixels=1)]))
        self.linked(model, self.online, temporal=temporal)
        self.assert_untrusted(result_quality(model))

    def test_linked_not_applied_claim_cannot_hide_actual_mask_inputs(self):
        actual = self.dataset('masked', masked=(0,))
        model = self.model(actual)
        temporal = temporal_summary(dict(applied_frames=0, frames=[dict(frame=0, applied=False,
            reason='no_stationary_five_frame_window')]))
        self.linked(model, actual, temporal=temporal)
        self.assert_untrusted(result_quality(model))


if __name__ == '__main__':
    unittest.main()
