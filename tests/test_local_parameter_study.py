"""File-contract tests only: these fixtures are not reconstructed camera evidence."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import struct
import tempfile
import threading
import unittest
from unittest import mock
import zlib

import femto.local.local_parameter_study as study


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(dx=0.):
    return [[1., 0., 0., dx], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]


def fixture_png(width=128, height=72):
    def chunk(tag, value):
        return struct.pack('>I', len(value)) + tag + value + struct.pack('>I', zlib.crc32(tag+value))
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)) +
            chunk(b'IDAT', zlib.compress((b'\0'+b'\x10\x20\x30'*width)*height)) + chunk(b'IEND', b''))


class ParameterStudyFixture:
    def __init__(self, root, *, kind='real_femto'):
        self.root, self.kind = Path(root), kind
        self.raw = self.root/'raw'
        self.dataset = self.raw/'derived/local_posed'
        self.dataset.mkdir(parents=True)
        self.latest = self.root/'latest_complete_demo.txt'
        self.latest.write_text('existing business result', encoding='utf-8')
        self.proof = dict(schema='femto-local-reference-v1', origin='first_valid_color_camera',
            session_id='fixture_session', reference_frame_id=0, coordinate_frame='world_rdf', unit='metre',
            transform_direction='T_W0_Ct', calibration={'sha256': 'a'*64})
        self.pose_source = 'synthetic_ground_truth' if kind == 'synthetic' else 'online_rgbd_research_grade'
        self.frames = []
        for number in range(3):
            frame = dict(frame=number)
            for name in ('rgb', 'depth'):
                path = self.raw/name/f'{number:06}.png'
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f'raw fixture {name} {number}'.encode())
                frame[name] = str(path)
            pose = dict(frame=number, session_id='fixture_session', reference_frame_id=0,
                        T_WC=identity(number*.002), pose_source=self.pose_source)
            write_json(self.raw/'pose'/f'{number:06}.json', pose)
            path = self.dataset/'pose'/f'{number:06}.json'
            write_json(path, pose)
            frame['pose'] = str(path)
            self.frames.append(frame)
        for folder in (self.raw, self.dataset):
            (folder/'frames.jsonl').write_text(''.join(json.dumps(f)+'\n' for f in self.frames), encoding='utf-8')
            write_json(folder/'calibration/color_intrinsic.json', {'fixture': True})
        write_json(self.raw/'manifest.json', dict(schema='femto-full-scene-rgbd-v1', data_kind=kind,
            camera={'device': 'synthetic_test_camera' if kind == 'synthetic' else 'Orbbec Femto Bolt'}))
        write_json(self.dataset/'manifest.json', dict(schema='femto-full-scene-rgbd-v1', source_dataset=str(self.raw),
            pose={'source': self.pose_source}, local_capture=dict(measured_only=True, session_id='fixture_session',
                reference_frame_id=0, reference_provenance=self.proof)))
        write_json(self.dataset/'pose_quality_report.json', {'fixture': True})
        write_json(self.raw/'derived/local_capture_report.json', {'fixture': True})
        (self.raw/'derived/live_tracking.jsonl').write_text('{}\n', encoding='utf-8')
        self.source_hashes = {str(p): digest(p) for p in self.raw.rglob('*') if p.is_file()}
        self.evidence = dict(dataset=str(self.dataset), raw_source=str(self.raw), data_kind=kind,
            session_id='fixture_session', reference_frame_id=0, reference_provenance=self.proof,
            accepted_frame_ids=[0, 1, 2], source_frame_count=3, accepted_frame_count=3,
            pose_source=self.pose_source, workspace_radius_m=.02, workspace_policy='recorded',
            rejected_tracking=[], physical_accuracy_verified=False, synthetic_ground_truth=None,
            source_sha256=self.source_hashes,
            implementation_sha256={str((study.ROOT/name).resolve()): digest(study.ROOT/name) for name in study.IMPLEMENTATION_FILES})

    def make_case(self, path, voxel=.005, trunc=.02):
        path = Path(path).resolve()
        path.mkdir(parents=True)
        write_json(path/'source_evidence.json', self.evidence)
        params = dict(voxel_m=voxel, sdf_trunc_m=trunc, image_scale=1., point_stride=1, depth_min_m=.25, depth_max_m=2.21)
        stats = dict(elapsed_s=5., peak_job_committed_mb=512., peak_rss_sampled_mb=None, memory_limit_mb=3072., returncode=0)
        report = dict(schema='femto-local-comparison-v1', state='complete', output=str(path),
            source_dataset=str(self.dataset), source_evidence=str(path/'source_evidence.json'),
            session_id='fixture_session', reference_frame_id=0, accepted_frame_ids=[0, 1, 2], source_frame_count=3,
            data_kind=self.kind, coordinate_frame='world_rdf', unit='metre', transform_direction='T_W0_Ct', parameters=params,
            budgets=dict(max_views=20, stage_timeout_s=300., memory_mb=3072., worker_count=1, log_storage='streamed to disk', render_width=128),
            stages=[dict(name='fixture_stage', state='complete', **stats)], variants=[], elapsed_s=20.,
            physical_accuracy_verified=False, latest_before={str(self.latest):digest(self.latest)},
            source_unchanged=True, source_changed=[], latest_unchanged=True, latest_changed=[])
        view = dict(reference_frame_id=0, session_id='fixture_session', coordinate_frame='world_rdf', unit='metre',
            view='initial_color_camera_locked', camera_to_world=identity(), undistorted_intrinsic=[[120., 0., 64.], [0., 120., 36.], [0., 0., 1.]],
            source_image_size=[128, 72], display_image_size=[128, 72], rgb_sha256='a'*64, depth_sha256='b'*64,
            near_m=.25, far_m=2.21, same_lighting=True)
        metric = dict(schema='femto-local-comparison-metrics-v1', coordinate_frame='world_rdf', unit='metre', fixed_view=view,
            internal_geometric_residual={'frames': [dict(frame=i, sample_count=n, rmse_m=r) for i,n,r in ((0,2,.003),(1,3,.004),(2,5,.005))]},
            depth_consistency={'frames': [dict(frame=i, sampled_pixels=n, hit_pixels=n-1, missing_hit_pixels=1,
                error_on_hits=dict(sample_count=n-1, rmse_m=r), error_missing_as_zero=dict(sample_count=n, rmse_m=r*10))
                for i,n,r in ((0,2,.003),(1,3,.004),(2,5,.005))]},
            valid_pixels={'frames': [dict(frame=i, raw_valid_pixels=100, actual_valid_pixels=100, masked_pixels=0,
                raw_valid_pixel_ratio=.5, actual_valid_pixel_ratio=.5) for i in range(3)]},
            fixed_projection=dict(unsupported_pixel_ratio=.2), topology=dict(boundary_edges=3, nonmanifold_edges=0),
            axis_aligned_extent_m=[1.+voxel, 2., 0.], mesh_surface_area_m2=1.)
        for index, name in enumerate(study.VARIANTS):
            reconstruction = path/name/'reconstruction'
            reconstruction.mkdir(parents=True)
            mesh = reconstruction/'scene_tsdf_mesh_m.ply'
            mesh.write_bytes(f'geometry fixture {voxel} {trunc}'.encode())
            cloud = reconstruction/'initial_observed_points_m.ply'
            cloud.write_bytes(b'same original W0 cloud fixture')
            for kind in ('surface', 'reference_rgb', 'initial_raw'):
                (reconstruction/f'comparison_{kind}.png').write_bytes(fixture_png())
            report['variants'].append(dict(name=name, state='complete', same_as=study.VARIANTS[index-1] if index else None,
                geometry_reused=bool(index), actual_dataset=str(self.dataset), actual_pose_source=self.pose_source,
                frames=[0, 1, 2], masks_applied=0, reconstruction=str(reconstruction),
                reconstruction_metrics=dict(fused_point_count=40, mesh_vertices=30, mesh_triangles=12),
                metrics=copy.deepcopy(metric), mesh_sha256=digest(mesh), initial_cloud_sha256=digest(cloud),
                resource_usage=stats, source_frame_count=3, accepted_frame_count=3, physical_accuracy_verified=False,
                optimization=dict(status='not_adopted' if index else 'baseline', reason='fixture only')))
        self.write_case(path, report)
        return path

    @staticmethod
    def write_case(path, report):
        write_json(Path(path)/'comparison_report.json', report)
        for variant in report['variants']:
            write_json(Path(path)/variant['name']/'variant.json', variant)
            write_json(Path(variant['reconstruction'])/'comparison_metrics.json', variant['metrics'])
            write_json(Path(variant['reconstruction'])/'comparison_view.json', variant['metrics']['fixed_view'])


class ParameterStudyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.fixture = ParameterStudyFixture(self.root)
        self.base = self.fixture.make_case(self.root/'base')
        self.candidate = self.fixture.make_case(self.root/'candidate', .003, .012)

    def read(self, path):
        return json.loads(Path(path).read_text(encoding='utf-8'))

    def summarize(self, *paths, **kwargs):
        return study.summarize_existing(paths or (self.base, self.candidate), self.root/'summary', **kwargs)

    def test_same_source_reuse_preserves_hashes_and_business_latest(self):
        original = {str(p): digest(p) for p in self.root.rglob('*') if p.is_file()}
        result = self.summarize()
        self.assertEqual(result['state'], 'complete', result.get('error'))
        self.assertEqual(result['completed_cases'], 2)
        self.assertTrue(result['latest_unchanged'])
        self.assertTrue(result['artifacts_unchanged'])
        self.assertFalse(result['default_changed'])
        self.assertFalse(result['physical_accuracy_verified'])
        for path, expected in original.items():
            self.assertEqual(digest(path), expected)
        self.assertTrue((self.root/'summary/parameter_study_report.html').is_file())
        self.assertEqual(result['reference_frame_id'], 0)
        self.assertEqual(result['accepted_frame_ids'], [0, 1, 2])
        self.assertEqual(result['fixed_view']['camera_to_world'], identity())

    def test_pooled_rms_and_metric_categories_are_separate(self):
        result = self.summarize()
        variant = result['cases'][0]['variants'][0]
        expected = math.sqrt((2*.003**2+3*.004**2+5*.005**2)/10)
        self.assertAlmostEqual(variant['internal_geometry']['pooled_rmse_m'], expected)
        self.assertAlmostEqual(variant['depth_consistency']['missing_as_zero_rmse_m'], expected*10)
        self.assertEqual(variant['depth_consistency']['missing_hits'], 3)
        self.assertEqual(variant['depth_consistency']['sampled_pixels'], 10)
        self.assertEqual(variant['valid_pixels']['mean_actual_valid_ratio'], .5)
        self.assertIsNone(variant['physical_accuracy']['accepted'])
        self.assertNotIn('p95_m', variant['internal_geometry'])

    def test_extent_delta_mm_negative_and_zero_reference(self):
        delta = self.summarize()['cases'][1]['variants'][0]['extent_change_vs_5mm']
        self.assertAlmostEqual(delta['absolute_mm'][0], -2.)
        self.assertEqual(delta['absolute_mm'][1:], [0., 0.])
        self.assertIsNone(delta['relative_percent'][2])

    def test_memory_peak_is_maximum_of_stages_not_sum_or_rss(self):
        report = self.read(self.base/'comparison_report.json')
        report['stages'].append(dict(peak_job_committed_mb=700., peak_rss_sampled_mb=900.))
        self.fixture.write_case(self.base, report)
        resource = self.summarize(self.base)['cases'][0]['resource_usage']
        self.assertEqual(resource['peak_job_committed_mb'], 700.)
        self.assertIsNone(resource['peak_rss_sampled_mb'])
        self.assertEqual(resource['comparison_elapsed_s'], 20.)

    def test_not_adopted_keeps_actual_original_source_and_same_model(self):
        variants = self.summarize()['cases'][0]['variants']
        for index in (1, 2):
            self.assertEqual(variants[index]['optimization']['status'], 'not_adopted')
            self.assertEqual(variants[index]['mesh_sha256'], variants[0]['mesh_sha256'])
            self.assertEqual(variants[index]['actual_dataset'], str(self.fixture.dataset))

    def test_failed_candidate_retained_without_partial_metrics_or_fake_success(self):
        report = self.read(self.candidate/'comparison_report.json')
        report.update(state='failed', error='Stage task-group memory limit', variants=[])
        self.fixture.write_case(self.candidate, report)
        result = self.summarize()
        self.assertEqual(result['state'], 'incomplete')
        self.assertEqual(result['completed_cases'], 1)
        self.assertEqual(result['cases'][1]['state'], 'failed')
        self.assertEqual(result['cases'][1]['variants'], [])
        self.assertIn('memory', result['cases'][1]['error'])

    def test_cancelled_candidate_is_not_counted_as_passed(self):
        report = self.read(self.candidate/'comparison_report.json')
        report.update(state='cancelled', variants=[])
        self.fixture.write_case(self.candidate, report)
        result = self.summarize()
        self.assertEqual(result['state'], 'incomplete')
        self.assertEqual(result['completed_cases'], 1)

    def test_missing_metrics_remain_none(self):
        report = self.read(self.base/'comparison_report.json')
        for variant in report['variants']:
            for key in ('internal_geometric_residual','depth_consistency','valid_pixels','topology','axis_aligned_extent_m'):
                variant['metrics'].pop(key)
        self.fixture.write_case(self.base, report)
        variant = self.summarize(self.base)['cases'][0]['variants'][0]
        self.assertIsNone(variant['internal_geometry']['pooled_rmse_m'])
        self.assertIsNone(variant['depth_consistency']['sampled_pixels'])
        self.assertIsNone(variant['valid_pixels']['mean_actual_valid_ratio'])
        self.assertIsNone(variant['extent_change_vs_5mm']['absolute_mm'])

    def test_changed_original_depth_is_rejected(self):
        Path(self.fixture.frames[1]['depth']).write_bytes(b'changed original')
        with self.assertRaisesRegex(ValueError, 'fingerprint|SHA-256'):
            self.summarize()

    def test_changed_frame_index_cannot_replace_the_original_fingerprint(self):
        path = self.fixture.dataset/'frames.jsonl'
        rows = copy.deepcopy(self.fixture.frames)
        rows[1]['depth'] = rows[0]['depth']
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'fingerprint|SHA-256'):
            self.summarize()

    def test_mask_hash_mismatch_rejected_before_any_summary_result(self):
        report = self.read(self.candidate/'comparison_report.json')
        derived = self.candidate/'derived/temporal/dataset'
        derived.mkdir(parents=True)
        mask = self.candidate/'derived/temporal/support_000000.png'
        mask.write_bytes(b'mask changed after original report')
        frames = copy.deepcopy(self.fixture.frames)
        frames[0]['local_temporal_support'] = dict(mask=str(mask), sha256='0'*64)
        (derived/'frames.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in frames), encoding='utf-8')
        variant = report['variants'][2]
        variant.update(actual_dataset=str(derived), masks_applied=1, same_as=None, geometry_reused=False,
                       optimization={'status':'adopted'})
        self.fixture.write_case(self.candidate, report)
        with self.assertRaisesRegex(ValueError, 'SHA-256'):
            self.summarize()

    def test_same_geometry_input_digest_across_parameter_changes(self):
        result = self.summarize()
        first, second = result['cases']
        self.assertNotEqual(first['parameters']['voxel_m'], second['parameters']['voxel_m'])
        self.assertEqual([v['geometry_input_sha256'] for v in first['variants']],
                         [v['geometry_input_sha256'] for v in second['variants']])

    def test_changed_original_pose_is_rejected_without_rebuilding_or_applying_twice(self):
        path = Path(self.fixture.frames[1]['pose'])
        pose = self.read(path)
        pose['T_WC'][0][3] += 1.
        write_json(path, pose)
        with self.assertRaisesRegex(ValueError, 'fingerprint|SHA-256'):
            self.summarize()

    def test_changed_actual_mesh_is_rejected(self):
        (self.candidate/'B_refined_raw/reconstruction/scene_tsdf_mesh_m.ply').write_bytes(b'wrong model')
        with self.assertRaisesRegex(ValueError, 'SHA-256'):
            self.summarize()

    def test_unadopted_different_actual_input_is_rejected(self):
        report = self.read(self.candidate/'comparison_report.json')
        report['variants'][1]['actual_dataset'] = str(self.root/'unrelated')
        self.fixture.write_case(self.candidate, report)
        with self.assertRaisesRegex(ValueError, 'Not-adopted'):
            self.summarize()

    def test_changed_implementation_and_missing_hash_are_rejected(self):
        path = self.candidate/'source_evidence.json'
        evidence = self.read(path)
        name = next(iter(evidence['implementation_sha256']))
        for altered in ('0'*64, None):
            changed = copy.deepcopy(evidence)
            if altered is None:
                changed['implementation_sha256'].pop(name)
            else:
                changed['implementation_sha256'][name] = altered
            write_json(path, changed)
            with self.assertRaises(ValueError):
                self.summarize()

    def test_missing_source_fingerprint_is_rejected(self):
        path = self.base/'source_evidence.json'
        evidence = self.read(path)
        evidence['source_sha256'].pop(str(self.fixture.raw/'manifest.json'))
        write_json(path, evidence)
        with self.assertRaisesRegex(ValueError, 'omits'):
            self.summarize()

    def test_source_session_mismatch_is_rejected(self):
        report = self.read(self.candidate/'comparison_report.json')
        report['session_id'] = 'another_session'
        self.fixture.write_case(self.candidate, report)
        with self.assertRaisesRegex(ValueError, 'session'):
            self.summarize()

    def test_frame_selection_mismatch_is_rejected(self):
        report = self.read(self.candidate/'comparison_report.json')
        report['variants'][0]['frames'] = [0, 2, 1]
        self.fixture.write_case(self.candidate, report)
        with self.assertRaisesRegex(ValueError, 'frame'):
            self.summarize()

    def test_coordinate_unit_or_reference_transform_mismatch_is_rejected(self):
        original = self.read(self.candidate/'comparison_report.json')
        for change in ('unit', 'reference'):
            report = copy.deepcopy(original)
            if change == 'unit':
                report['unit'] = 'mm'
            else:
                for variant in report['variants']:
                    variant['metrics']['fixed_view']['camera_to_world'] = identity(.01)
            self.fixture.write_case(self.candidate, report)
            with self.assertRaises(ValueError):
                self.summarize()

    def test_same_view_scale_and_non_tsdf_parameters_required(self):
        original = self.read(self.candidate/'comparison_report.json')
        for change in ('projection', 'depth_range'):
            report = copy.deepcopy(original)
            if change == 'projection':
                for variant in report['variants']:
                    variant['metrics']['fixed_view']['undistorted_intrinsic'][0][0] += 5.
            else:
                report['parameters']['depth_min_m'] = .1
            self.fixture.write_case(self.candidate, report)
            with self.assertRaisesRegex(ValueError, 'projection|non-TSDF'):
                self.summarize()

    def test_metric_file_mismatch_is_rejected(self):
        write_json(self.candidate/'A_online_raw/reconstruction/comparison_metrics.json', {'wrong':True})
        with self.assertRaisesRegex(ValueError, 'metric file'):
            self.summarize()

    def test_nonfinite_report_rejected(self):
        report = self.read(self.candidate/'comparison_report.json')
        for number in (float('nan'), float('inf'), -float('inf')):
            report['elapsed_s'] = number
            write_json(self.candidate/'comparison_report.json', report)
            with self.assertRaises(ValueError):
                self.summarize()

    def test_synthetic_truth_cannot_be_attached_to_real_report(self):
        report = self.read(self.candidate/'comparison_report.json')
        report['variants'][0]['metrics']['synthetic_ground_truth'] = dict(source={'fixture': True}, rmse_m=.001)
        self.fixture.write_case(self.candidate, report)
        with self.assertRaisesRegex(ValueError, 'Synthetic truth'):
            self.summarize()

    def test_synthetic_report_requires_explicit_opt_in_and_stays_nonphysical(self):
        fixture = ParameterStudyFixture(self.root/'synthetic_fixture', kind='synthetic')
        case = fixture.make_case(self.root/'synthetic_case')
        with self.assertRaisesRegex(ValueError, 'Synthetic replay'):
            study.summarize_existing([case], self.root/'synthetic_summary')
        result = study.summarize_existing([case], self.root/'synthetic_summary', allow_synthetic=True)
        self.assertEqual(result['state'], 'complete')
        self.assertEqual(result['data_kind'], 'synthetic')
        self.assertFalse(result['physical_accuracy_verified'])

    def test_source_data_kind_cannot_be_promoted_by_report_only(self):
        source = self.read(self.fixture.raw/'manifest.json')
        source['camera']['device'] = 'unknown camera'
        write_json(self.fixture.raw/'manifest.json', source)
        with self.assertRaisesRegex(ValueError, 'camera metadata'):
            self.summarize()

    def test_existing_output_and_output_inside_source_are_rejected_without_overwrite(self):
        marker = self.fixture.latest.read_bytes()
        for output in (self.base, self.fixture.raw/'new-study', self.fixture.dataset/'new-study'):
            with self.subTest(output=output), self.assertRaises((ValueError, FileExistsError)):
                study.summarize_existing([self.base], output)
        self.assertEqual(self.fixture.latest.read_bytes(), marker)

    def test_missing_duplicate_or_failed_baseline_is_rejected(self):
        for paths in ([self.candidate], [self.base, self.base]):
            with self.assertRaises(ValueError):
                study.summarize_existing(paths, self.root/'summary')
        report = self.read(self.base/'comparison_report.json')
        report.update(state='failed', variants=[])
        self.fixture.write_case(self.base, report)
        with self.assertRaisesRegex(ValueError, 'complete 5 mm'):
            self.summarize()

    def test_source_mutation_during_summary_is_not_published_as_complete(self):
        def progress(report):
            if report['completed_cases'] == 1:
                Path(self.fixture.frames[0]['depth']).write_bytes(b'concurrent change')
        result = self.summarize(progress=progress)
        self.assertEqual(result['state'], 'failed')
        self.assertFalse((self.root/'summary/parameter_study_report.html').exists())

    def test_cancel_preflight_and_after_first_case_isolated(self):
        event = threading.Event()
        event.set()
        with self.assertRaises(study.StudyStopped):
            self.summarize(cancelled=event.is_set)
        self.assertFalse((self.root/'summary').exists())
        event.clear()
        def progress(report):
            if report['completed_cases'] == 1:
                event.set()
        result = self.summarize(cancelled=event.is_set, progress=progress)
        self.assertEqual(result['state'], 'cancelled')
        self.assertEqual(result['completed_cases'], 1)
        self.assertFalse((self.root/'summary/parameter_study_report.html').exists())

    def test_wrong_projection_image_size_fails_without_success_page(self):
        path = self.base/'A_online_raw/reconstruction/comparison_surface.png'
        path.write_bytes(fixture_png(64, 36))
        result = self.summarize()
        self.assertEqual(result['state'], 'failed')
        self.assertIn('PNG', result['error'])

    def test_historical_latest_value_can_advance_before_new_readonly_audit(self):
        self.fixture.latest.write_text('new legitimate business result', encoding='utf-8')
        result = self.summarize()
        self.assertTrue(result['latest_unchanged'])
        self.assertEqual(self.fixture.latest.read_text(encoding='utf-8'), 'new legitimate business result')

    def test_run_reuses_baseline_and_calls_existing_comparison_only_for_missing_candidate(self):
        calls = []
        def run(dataset, output, **kwargs):
            calls.append((dataset, output, kwargs))
            self.fixture.make_case(output, kwargs['voxel_m'], kwargs['sdf_trunc_m'])
            return self.read(Path(output)/'comparison_report.json')
        with mock.patch('femto.local.local_comparison.run_comparison', side_effect=run):
            result = study.run_study(self.fixture.dataset, self.root/'job', candidates=(study.BASELINE, (.003, .02)),
                reuse_paths=[self.base], render_width=128)
        self.assertEqual(result['state'], 'complete', result.get('error'))
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][2]['voxel_m'], .003)
        self.assertEqual(calls[0][2]['sdf_trunc_m'], .02)
        self.assertTrue(callable(calls[0][2]['cancelled']))
        self.assertEqual([r['execution'] for r in result['cases']], ['reused_verified_report','executed_now'])

    def test_run_cancel_does_not_start_later_candidate(self):
        event = threading.Event()
        def progress(job):
            if len(job['cases']) == 1:
                event.set()
        with mock.patch('femto.local.local_comparison.run_comparison') as run:
            result = study.run_study(self.fixture.dataset, self.root/'job', candidates=(study.BASELINE, (.003, .02)),
                reuse_paths=[self.base], render_width=128, cancelled=event.is_set, progress=progress)
        self.assertEqual(result['state'], 'cancelled')
        run.assert_not_called()

    def test_reuse_with_different_requested_budget_does_not_start_comparison(self):
        with mock.patch('femto.local.local_comparison.run_comparison') as run:
            with self.assertRaisesRegex(ValueError, 'budgets'):
                study.run_study(self.fixture.dataset, self.root/'job', candidates=(study.BASELINE,),
                    reuse_paths=[self.base], memory_mb=6144, render_width=128)
        run.assert_not_called()

    def test_parameter_and_audit_budgets_are_finite_and_bounded(self):
        for pair in ((float('nan'), .02), (.003, float('inf')), (0., .02), (.003, .003), (True, .02)):
            with self.subTest(pair=pair), self.assertRaises(ValueError):
                study.parameter_key(*pair)
        budget = study.AuditBudget(1.)
        budget.started -= 2.
        with self.assertRaises(study.StudyStopped) as stopped:
            budget.check()
        self.assertEqual(stopped.exception.state, 'failed')
        budget = study.AuditBudget(hash_limit=1)
        with self.assertRaisesRegex(study.StudyStopped, 'hash-read'):
            budget.digest(self.fixture.latest)
        budget = study.AuditBudget(json_limit=1)
        with self.assertRaisesRegex(ValueError, 'memory budget'):
            budget.read(self.base/'comparison_report.json')
        budget = study.AuditBudget(json_total_limit=1)
        with self.assertRaisesRegex(ValueError, 'Cumulative JSON'):
            budget.read(self.base/'comparison_report.json')


if __name__ == '__main__':
    unittest.main()
