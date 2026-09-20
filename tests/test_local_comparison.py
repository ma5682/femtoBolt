"""Software-only A/B/C provenance, independent replay and process-budget checks."""
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

import numpy as np
import open3d as o3d

from femto.pipeline.local_comparison import (StageStopped, changed_files, copy_unchanged_model, fingerprints,
    load_json_object, read_rows, run_comparison, sha256, supervise, validate_comparison_input,
    validate_variant_input, write_json_strict)
from femto.pipeline.local_comparison_viewer import sampled_depth_metrics, _mesh_scene, render_fixed_surface, write_comparison_html


def make_fixture(root, width=128, height=96):
    """New synthetic evidence only; original generator output is not rewritten.

    Five copies of the first synthetic pose exercise stationary-window lineage.
    They are simulated records, explicitly marked synthetic, not online camera
    tracking or a physical acceptance recording.
    """
    from femto.pipeline.synthetic_scene_demo import generate_dataset
    from femto.dataset.local_reference import calibration_identity, reference_provenance
    from femto.geometry.fusion import read_geometry_safe, export_mesh_formats
    generated = generate_dataset(root/'generated', frames=4, width=width, height=height)
    original_frames = read_rows(generated/'frames.jsonl')
    original_poses = [load_json_object(generated/f['pose']) for f in original_frames]
    world0 = np.asarray(original_poses[0]['T_WC'])
    inverse = np.linalg.inv(world0)
    raw = root/'raw'
    for child in ('rgb', 'depth', 'pose', 'derived'):
        (raw/child).mkdir(parents=True, exist_ok=False)
    shutil.copytree(generated/'calibration', raw/'calibration')
    sequence = [0,0,0,0,0,1,2,3]
    accepted = [0,5,6,7]
    records, frames = [], []
    session = 'synthetic-comparison-fixture'
    for index, source_index in enumerate(sequence):
        original = original_frames[source_index]
        pose = inverse @ np.asarray(original_poses[source_index]['T_WC'])
        if index == 0:
            pose = np.eye(4)
        timestamp = 1_000_000+index*100_000
        frame = dict(frame=index, camera_timestamp_us=timestamp, color_timestamp_us=timestamp,
            depth_timestamp_us=timestamp, rgb_depth_timestamp_delta_us=0, pose_valid=False,
            rgb=f'rgb/{index:06}.jpg', depth=f'depth/{index:06}.png', pose=f'pose/{index:06}.json')
        for key in ('rgb', 'depth'):
            shutil.copyfile(generated/original[key], raw/frame[key])
        write_json_strict(raw/frame['pose'], dict(frame=index, pose_valid=True, pose_source='synthetic_ground_truth',
                                          T_WC=original_poses[source_index]['T_WC']))
        frames.append(frame)
        records.append(dict(frame_id=index, session_id=session, reference_frame_id=0, T_WC=pose.tolist(),
            capture_timestamp_us=timestamp, color_timestamp_us=timestamp, depth_timestamp_us=timestamp,
            state='tracking', latency_ms=10., quality=dict(usable=True), selected_for_fusion=index in accepted))
    manifest = load_json_object(generated/'manifest.json')
    manifest.update(frame_count=len(frames), purpose='Synthetic same-source replay fixture, not camera evidence',
                    parameters=dict(depth_min_mm=250., depth_max_mm=2210., fixed_camera=False))
    write_json_strict(raw/'manifest.json', manifest)
    (raw/'frames.jsonl').write_text(''.join(json.dumps(f)+'\n' for f in frames), encoding='utf-8')
    (raw/'derived/live_tracking.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records), encoding='utf-8')
    proof = reference_provenance(session, records[0], calibration_identity(raw))
    local = dict(measured_only=True, session_id=session, reference_frame_id=0,
        reference_provenance=proof, tracking_report=str(raw/'derived/local_capture_report.json'))
    write_json_strict(raw/'derived/local_capture_report.json', dict(schema='femto-local-tracking-v1', state='stopped',
        session_id=session, reference_frame_id=0, reference_provenance=proof, accepted_frames=accepted,
        coordinate_frame='world_rdf', unit='metre', transform_direction='T_W0_Ct',
        pose_source='synthetic_ground_truth', frame_count=len(records), tracking_lost_events=0))
    posed = raw/'derived/local_posed'
    (posed/'pose').mkdir(parents=True)
    shutil.copytree(raw/'calibration', posed/'calibration')
    selected = []
    for ordinal, index in enumerate(accepted):
        frame = copy.deepcopy(frames[index])
        frame.update(pose=f'pose/{ordinal:06}.json', pose_valid=True)
        for kind in ('rgb', 'depth'):
            frame[kind] = str(raw/frames[index][kind])
        write_json_strict(posed/frame['pose'], dict(frame=index, pose_valid=True,
            pose_source='synthetic_ground_truth', T_WC=records[index]['T_WC'], session_id=session,
            reference_frame_id=0, reference_provenance=proof, capture_timestamp_us=records[index]['capture_timestamp_us']))
        selected.append(frame)
    truth = read_geometry_safe(generated/'ground_truth/synthetic_scene_ground_truth_m.ply', mesh=True)
    truth.transform(inverse)
    truth_files = export_mesh_formats(truth, root/'truth', stem='synthetic_ground_truth_W0_m',
        note='Synthetic W0 truth fixture only; no physical accuracy claim')
    truth_path = Path(truth_files['ply'])
    posed_manifest = copy.deepcopy(manifest)
    posed_manifest.update(frame_count=len(selected), source_dataset=str(raw), local_capture=local,
        pose=dict(valid=True, source='synthetic_ground_truth'), comparison_ground_truth=dict(mesh=str(truth_path),
            coordinate_frame='world_rdf', unit='metre', reference_provenance=proof, sha256=sha256(truth_path)))
    write_json_strict(posed/'manifest.json', posed_manifest)
    (posed/'frames.jsonl').write_text(''.join(json.dumps(f)+'\n' for f in selected), encoding='utf-8')
    write_json_strict(posed/'pose_quality_report.json', dict(source_frame_count=len(records), frame_count=len(selected),
        accepted_frame_ordinals=accepted, rejected_frames=[]))
    return posed


class SourceValidationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dataset = make_fixture(self.root)

    def test_original_online_binding_and_no_validation_side_effect(self):
        before = fingerprints(p for p in self.root.rglob('*') if p.is_file())
        result = validate_comparison_input(self.dataset, allow_synthetic=True)
        self.assertEqual(result['accepted_frame_ids'], [0,5,6,7])
        self.assertEqual(result['source_frame_count'], 8)
        self.assertEqual(result['reference_frame_id'], 0)
        self.assertFalse(result['physical_accuracy_verified'])
        self.assertEqual(changed_files(before), [])
        self.assertFalse((self.dataset/'quality/validation_report.json').exists())
        with self.assertRaisesRegex(ValueError, 'synthetic'):
            validate_comparison_input(self.dataset)

    def test_session_reference_nonfinite_stale_dynamic_and_workspace_rejection(self):
        rows = read_rows(self.dataset/'frames.jsonl')
        pose_path = self.dataset/rows[1]['pose']
        original = load_json_object(pose_path)
        for key, value in [('session_id', 'another'), ('reference_frame_id', 9), ('T_WC', [[float('nan')]*4]*4)]:
            changed = copy.deepcopy(original)
            changed[key] = value
            # Deliberately malformed fixture bytes; production writers reject NaN.
            pose_path.write_text(json.dumps(changed), encoding='utf-8')
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_comparison_input(self.dataset, allow_synthetic=True)
            write_json_strict(pose_path, original)
        tracking_path = self.dataset.parent/'live_tracking.jsonl'
        original_records = read_rows(tracking_path)
        for alteration in ('lost', 'stale', 'dynamic', 'sync'):
            records = copy.deepcopy(original_records)
            selected = records[5]
            if alteration == 'lost':selected['state']='lost'
            if alteration == 'stale':selected['latency_ms']=2001.
            if alteration == 'dynamic':selected['quality']['usable']=False
            if alteration == 'sync':selected['color_timestamp_us']+=6000
            tracking_path.write_text(''.join(json.dumps(r)+'\n' for r in records), encoding='utf-8')
            with self.subTest(alteration=alteration), self.assertRaises(ValueError):
                validate_comparison_input(self.dataset, allow_synthetic=True)
        tracking_path.write_text(''.join(json.dumps(r)+'\n' for r in original_records), encoding='utf-8')
        manifest = load_json_object(self.dataset/'manifest.json')
        manifest['local_capture']['local_workspace_radius_m']=.02
        write_json_strict(self.dataset/'manifest.json', manifest)
        report_path = self.dataset.parent/'local_capture_report.json'
        report = load_json_object(report_path)
        report['local_workspace_radius_m']=.02
        write_json_strict(report_path, report)
        with self.assertRaisesRegex(ValueError, 'workspace'):
            validate_comparison_input(self.dataset, allow_synthetic=True)

    def test_changed_source_rgb_and_masked_baseline_rejected(self):
        index_path = self.dataset/'frames.jsonl'
        frames = read_rows(index_path)
        original = copy.deepcopy(frames)
        frames[1]['rgb'] = frames[0]['rgb']
        index_path.write_text(''.join(json.dumps(f)+'\n' for f in frames), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'differs'):
            validate_comparison_input(self.dataset, allow_synthetic=True)
        original[0]['local_temporal_support'] = {'mask': 'not-a-baseline-mask.png'}
        index_path.write_text(''.join(json.dumps(f)+'\n' for f in original), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'mask'):
            validate_comparison_input(self.dataset, allow_synthetic=True)

    def test_mask_provenance_and_changed_mask_rejected(self):
        from femto.local.local_temporal_support import prepare
        baseline = validate_comparison_input(self.dataset, allow_synthetic=True)
        # Controlled image-audit result exercises file contracts; this is not an actual motion pass.
        with mock.patch('femto.local.local_temporal_support.audit_temporal_motion', return_value=(np.zeros((96,128),bool), dict(passed=True))):
            derived, report = prepare(self.dataset, self.root/'temporal')
        self.assertEqual(report['applied_frames'], 1)
        binding = validate_variant_input(baseline, derived, allow_masks=True)
        self.assertEqual(binding['masks_applied'], 1)
        with self.assertRaisesRegex(ValueError, 'Unexpected temporal mask'):
            validate_variant_input(baseline, derived, allow_masks=False)
        frames = read_rows(derived/'frames.jsonl')
        mask = Path(frames[0]['local_temporal_support']['mask'])
        mask.write_bytes(b'changed mask')
        with self.assertRaisesRegex(ValueError, 'hash'):
            validate_variant_input(baseline, derived, allow_masks=True)

    def test_output_inside_source_existing_and_over_budget_are_rejected(self):
        for destination in (self.dataset, self.dataset/'new_output', self.dataset.parents[1]/'new_output'):
            with self.subTest(destination=destination), self.assertRaises((ValueError, FileExistsError)):
                run_comparison(self.dataset, destination, allow_synthetic=True)
        with self.assertRaisesRegex(ValueError, 'at most 20'):
            validate_comparison_input(self.dataset, max_views=3, allow_synthetic=True)
        with self.assertRaises(ValueError):
            run_comparison(self.dataset, self.root/'outside', voxel_m=float('nan'))


class ComparisonRuntimeTests(unittest.TestCase):
    def test_actual_synthetic_abc_same_source_hash_and_latest_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dataset = make_fixture(root)
            latest = root/'latest_complete_demo.txt'
            latest.write_text('original-business-result', encoding='utf-8')
            before = sha256(latest)
            report = run_comparison(dataset, root/'comparison', allow_synthetic=True,
                                    render_width=128, latest_paths=[latest], stage_timeout_s=180)
            self.assertEqual(report['state'], 'complete', report.get('error'))
            self.assertTrue(report['source_unchanged'])
            self.assertTrue(report['latest_unchanged'])
            self.assertEqual(sha256(latest), before)
            self.assertEqual(len(report['variants']), 3)
            views = [v['metrics']['fixed_view'] for v in report['variants']]
            self.assertTrue(all(v == views[0] for v in views))
            self.assertEqual(len({v['initial_cloud_sha256'] for v in report['variants']}), 1)
            for index, variant in enumerate(report['variants']):
                self.assertEqual(variant['frames'], [0,5,6,7])
                self.assertFalse(variant['physical_accuracy_verified'])
                self.assertIn('internal_geometric_residual', variant['metrics'])
                self.assertIn('depth_consistency', variant['metrics'])
                self.assertIn('valid_pixels', variant['metrics'])
                self.assertIn('synthetic_ground_truth', variant['metrics'])
                self.assertIsNone(variant['metrics']['physical_accuracy']['accepted'])
                if variant['same_as']:
                    self.assertEqual(variant['mesh_sha256'], report['variants'][index-1]['mesh_sha256'])
            self.assertTrue((root/'comparison/comparison_report.html').is_file())

    def test_real_process_timeout_and_cancel_cannot_write_late_result(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            marker = root/'late-result.json'
            script = root/'delayed.py'
            script.write_text('import time\nfrom pathlib import Path\ntime.sleep(1.5)\nPath(__file__).with_name("late-result.json").write_text("late")\n')
            start = time.monotonic()
            with self.assertRaises(StageStopped) as stopped:
                supervise([sys.executable, str(script)], root/'timeout.log', timeout_s=.15, memory_mb=1024)
            self.assertEqual(stopped.exception.status, 'failed')
            self.assertLess(time.monotonic()-start, 1.5)
            with self.assertRaises(StageStopped) as stopped:
                supervise([sys.executable, str(script)], root/'cancel.log', timeout_s=5, memory_mb=1024, cancelled=lambda:True)
            self.assertEqual(stopped.exception.status, 'cancelled')
            # The Windows venv redirector has a child interpreter. Verify it did
            # not survive termination and publish after the supervisor returned.
            time.sleep(1.6)
            self.assertFalse(marker.exists())

    def test_real_process_memory_budget_stops_child(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            script = root/'allocate.py'
            script.write_text('import time\ndata=bytearray(64*1024*1024)\ntime.sleep(5)\n')
            with self.assertRaises(StageStopped) as stopped:
                supervise([sys.executable, str(script)], root/'memory.log', timeout_s=5, memory_mb=16)
            self.assertEqual(stopped.exception.status, 'failed')
            self.assertEqual(stopped.exception.stats['memory_limit_mb'], 16)
            if sys.platform == 'win32':
                # The kernel rejects the allocation before it crosses the hard
                # limit; the child then exits with a real MemoryError.
                self.assertIn('MemoryError', (root/'memory.log').read_text(errors='replace'))
                self.assertGreater(stopped.exception.stats['peak_job_committed_mb'], 0)
            else:
                self.assertIn('memory', str(stopped.exception))
                self.assertGreater(stopped.exception.stats['peak_rss_sampled_mb'], 16)

    def test_cancel_before_validation_stays_separate_and_no_source_changed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dataset = make_fixture(root)
            before = fingerprints(p for p in dataset.parents[1].rglob('*') if p.is_file())
            cancel = root/'CANCEL'
            cancel.touch()
            report = run_comparison(dataset, root/'cancelled', allow_synthetic=True, cancel_file=cancel, latest_paths=[])
            self.assertEqual(report['state'], 'cancelled')
            self.assertEqual(report['variants'], [])
            self.assertEqual(changed_files(before), [])

    def test_event_cancel_after_validation_does_not_publish_a_model(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dataset = make_fixture(root)
            event = threading.Event()
            def progress(report):
                if report['stages'] and report['stages'][-1]['state']=='complete':
                    event.set()
            report = run_comparison(dataset, root/'event_cancel', allow_synthetic=True,
                                    cancelled=event.is_set, progress=progress, latest_paths=[])
            self.assertEqual(report['state'], 'cancelled')
            self.assertTrue(report['source_unchanged'])
            self.assertEqual(report['variants'], [])
            self.assertFalse((root/'event_cancel/CANCEL').exists())
            self.assertFalse(list((root/'event_cancel').rglob('scene_tsdf_mesh_m.ply')))

    def test_byte_identical_reuse_rebinds_geometry_report(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root/'A/reconstruction'
            source.mkdir(parents=True)
            mesh = source/'scene_tsdf_mesh_m.ply'
            mesh.write_bytes(b'unchanged fixture geometry')
            write_json_strict(source/'reconstruction_report.json', dict(outputs=dict(mesh=dict(ply=str(mesh))), source_dataset='original'))
            output = root/'B/reconstruction'
            report = copy_unchanged_model(source, output)
            self.assertEqual(sha256(mesh), sha256(output/mesh.name))
            self.assertEqual(report['outputs']['mesh']['ply'], str(output/mesh.name))
            self.assertEqual(report['source_dataset'], 'original')


class ComparisonProjectionTests(unittest.TestCase):
    @staticmethod
    def plane(z=1):
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices=o3d.utility.Vector3dVector([[-2,-2,z],[2,-2,z],[2,2,z],[-2,2,z]])
        mesh.triangles=o3d.utility.Vector3iVector([[0,1,2],[0,2,3]])
        return mesh

    def test_depth_units_fixed_w0_and_missing_hits_are_explicit(self):
        k = np.array([[4.,0,2],[0,4.,2],[0,0,1.]])
        depth = np.full((5,5),1000,np.uint16)
        mesh = self.plane()
        before = np.asarray(mesh.vertices).copy()
        metrics = sampled_depth_metrics(_mesh_scene(mesh), depth, k, np.eye(4))
        # Raycasting returns float32 hit parameters; allow its ~1e-7 m arithmetic roundoff.
        self.assertAlmostEqual(metrics['error_missing_as_zero']['rmse_m'], 0, delta=1e-6)
        self.assertEqual(metrics['sampled_pixels'],25)
        pose = np.eye(4)
        pose[0,3]=20
        missing = sampled_depth_metrics(_mesh_scene(mesh), depth, k, pose)
        self.assertEqual(missing['missing_hit_pixels'],25)
        self.assertAlmostEqual(missing['error_missing_as_zero']['rmse_m'],1.)
        np.testing.assert_array_equal(np.asarray(mesh.vertices), before)

    def test_fixed_projection_checks_frontmost_depth_and_does_not_mutate_mesh(self):
        k = np.array([[4.,0,2],[0,4.,2],[0,0,1.]])
        mesh = self.plane()
        before = np.asarray(mesh.vertices).copy()
        _, matched = render_fixed_surface(mesh, k, np.full((5,5),1000,np.uint16), width=128)
        _, mismatched = render_fixed_surface(mesh, k, np.full((5,5),500,np.uint16), width=256)
        self.assertEqual(matched['visible_supported_pixels'],25)
        self.assertEqual(mismatched['visible_supported_pixels'],0)
        np.testing.assert_array_equal(np.asarray(mesh.vertices), before)

    def test_report_is_bilingual_and_unknown_values_not_zero_success(self):
        with tempfile.TemporaryDirectory() as td:
            path = write_comparison_html(Path(td), dict(state='failed',
                variants=[dict(name='A_online_raw', optimization=dict(status='failed'))],
                parameters={}, error='source rejected'))
            text = path.read_text(encoding='utf-8')
            self.assertIn('data-zh=', text)
            self.assertIn('data-en=', text)
            self.assertIn('未记录', text)
            self.assertIn('Not recorded', text)
            self.assertIn('source rejected', text)
            self.assertNotIn('0.000 mm', text)
            self.assertNotIn('None', text)
            self.assertIn('data-zh="失败" data-en="Failed"', text)
            self.assertNotIn('data-zh="已采用"', text)


if __name__ == '__main__':
    unittest.main()
