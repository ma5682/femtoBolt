"""Software-only coordinate, provenance, persistence and viewer-event regressions."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import open3d as o3d

import scene_demo_viewer as viewer
from scene_annotations import (
    AnnotationStore, distance_m, distance_to_origin_m, point_readout_mm,
)
from scene_annotation_ui import surface_readout_text


def triangle(z=0.):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector([[-2., -2., z], [2., -2., z], [0., 3., z]])
    mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2]])
    return mesh


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')


def local_evidence(root, source, session='session-test', reference=7):
    """Synthetic metadata fixture matching LocalCapture._export_poses' contract."""
    dataset = root / 'raw' / 'derived' / 'local_posed'
    tracking_path = root / 'raw' / 'derived' / 'local_capture_report.json'
    binding = dict(session_id=session, reference_frame_id=reference)
    context = dict(**binding, coordinate_frame='world_rdf', unit='metre',
                   transform_direction='T_W0_Ct', measured_only=True,
                   inferred_repairs_integrated=False, tracking_report=str(tracking_path),
                   support=dict(rule='all_vertices_and_centroid_supported_in_one_raw_view',
                                tolerance_m=.01, triangles_after=1))
    write_json(source.parent / 'local_geometry.json', context)
    write_json(source.parent / 'reconstruction_report.json', dict(
        source_dataset=str(dataset), local_capture=binding, outputs=dict(mesh=dict(ply=str(source)))))
    write_json(dataset / 'manifest.json', dict(local_capture=binding, source_kind='synthetic'))
    pose_path = dataset / 'pose' / '000000.json'
    write_json(pose_path, dict(**binding, frame=reference, pose_valid=True,
                              pose_source='online_rgbd_research_grade', T_WC=np.eye(4).tolist()))
    (dataset / 'frames.jsonl').write_text(json.dumps(
        dict(frame=reference, pose_valid=True, pose='pose/000000.json')) + '\n', encoding='utf-8')
    write_json(tracking_path, dict(**binding, schema='femto-local-tracking-v1',
        coordinate_frame='world_rdf', unit='metre', transform_direction='T_W0_Ct',
        pose_source='online_rgbd_research_grade', accepted_frames=[reference, reference + 1]))
    return dataset, pose_path, tracking_path


class OriginMathTests(unittest.TestCase):
    def test_known_xyz_negative_and_zero_distance(self):
        for xyz, expected in [
            ([.3, .4, 0.], .5), ([-.3, -.4, 0.], .5),
            ([0., 0., 0.], 0.), ([-.2, .3, -.6], .7),
        ]:
            with self.subTest(xyz=xyz):
                self.assertAlmostEqual(distance_to_origin_m(xyz), expected)
                values, distance = point_readout_mm(xyz)
                np.testing.assert_allclose(values, np.array(xyz) * 1000.)
                self.assertAlmostEqual(distance, expected * 1000.)

    def test_nonfinite_malformed_and_overflow_coordinates_are_rejected(self):
        for xyz in ([np.nan, 0, 0], [0, np.inf, 0], [0, 0, -np.inf], [1, 2], [[1, 2, 3]]):
            with self.subTest(xyz=xyz), self.assertRaises(ValueError):
                distance_to_origin_m(xyz)
        with self.assertRaises(ValueError):
            point_readout_mm([1e308, 0, 0])

    def test_two_point_distance_is_distinct_and_still_metric(self):
        a, b = dict(point_world_m=[.3, .4, 0]), dict(point_world_m=[.3, .4, .012])
        self.assertAlmostEqual(distance_to_origin_m(a['point_world_m']), .5)
        self.assertAlmostEqual(distance_m(a, b), .012)
        with self.assertRaises(ValueError):
            distance_m(a, dict(point_world_m=[np.nan, 0, 0]))


class OriginPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'model' / 'mesh.ply'
        self.mesh = triangle()
        self.store = AnnotationStore(self.mesh, self.source)

    def add_points(self):
        self.store.add_point(dict(point_world_m=[.3, .4, 0], triangle_index=0),
                             name='测试标记 A', note='研发估计')
        self.store.add_point(dict(point_world_m=[-.3, .4, 0], triangle_index=0),
                             name='Marker B')

    def test_save_reload_recompute_stale_origin_and_pair_distances(self):
        self.add_points()
        path = self.store.save(self.root / 'annotations')
        data = json.loads(path.read_text(encoding='utf-8'))
        self.assertEqual(data['points'][0]['distance_to_origin_m'], .5)
        self.assertAlmostEqual(data['measurements'][0]['distance_m'], .6)
        data['points'][0]['distance_to_origin_m'] = 12345
        data['measurements'][0]['distance_m'] = 6789
        write_json(path, data)
        loaded = AnnotationStore(self.mesh, self.source)
        result = loaded.load(path)
        self.assertEqual(loaded.points[0]['distance_to_origin_m'], .5)
        self.assertAlmostEqual(result['measurements'][0]['distance_m'], .6)
        self.assertEqual(loaded.points[0]['name'], '测试标记 A')
        self.assertEqual(loaded.points[0]['note'], '研发估计')
        # Load does not silently rewrite the user's original annotation file.
        self.assertEqual(json.loads(path.read_text(encoding='utf-8'))['points'][0]['distance_to_origin_m'], 12345)

    def test_legacy_missing_distance_and_origin_metadata_remains_readable(self):
        self.add_points()
        path = self.store.save(self.root / 'annotations')
        data = json.loads(path.read_text(encoding='utf-8'))
        del data['origin_reference']
        for point in data['points']:
            for key in ('distance_to_origin_m', 'coordinate_frame', 'unit', 'scale_validation'):
                del point[key]
        path.write_text(json.dumps(data), encoding='utf-8-sig')
        self.store.load(path)
        self.assertEqual([p['distance_to_origin_m'] for p in self.store.points], [.5, .5])
        self.assertEqual(self.store.origin_reference['provenance'], 'unknown')

    def test_wrong_geometry_and_same_geometry_other_model_are_rejected(self):
        self.add_points()
        path = self.store.save(self.root / 'annotations')
        for mesh, source in ((triangle(.1), self.source), (self.mesh, self.root / 'another_model.ply')):
            with self.subTest(source=source), self.assertRaises(ValueError):
                AnnotationStore(mesh, source).load(path)

    def test_nonfinite_off_surface_and_wrong_point_frame_fail_atomically(self):
        self.add_points()
        path = self.store.save(self.root / 'annotations')
        saved = json.loads(path.read_text(encoding='utf-8'))
        original_points = list(self.store.points)
        for change in (dict(point_world_m=[np.nan, 0, 0]),
                       dict(point_world_m=[0, 0, .1]), dict(point_world_m=[10, 0, 0]),
                       dict(triangle_index=-1), dict(coordinate_frame='camera_rdf'), dict(unit='mm')):
            with self.subTest(change=change):
                data = json.loads(json.dumps(saved))
                data['points'][0].update(change)
                write_json(path, data)
                with self.assertRaises(ValueError):
                    self.store.load(path)
                self.assertEqual(self.store.points, original_points)
        before = len(self.store.points)
        with self.assertRaises(ValueError):
            self.store.add_point(dict(point_world_m=[0, np.inf, 0], triangle_index=0))
        self.assertEqual(len(self.store.points), before)

    def test_top_level_coordinate_unit_and_origin_binding_mismatch(self):
        self.add_points()
        path = self.store.save(self.root / 'annotations')
        saved = json.loads(path.read_text(encoding='utf-8'))
        for change in (dict(coordinate_frame='camera_rdf'), dict(unit='mm'),
                       dict(origin_reference=dict(kind='initial_color_camera'))):
            with self.subTest(change=change):
                write_json(path, dict(saved, **change))
                with self.assertRaises(ValueError):
                    self.store.load(path)

    def test_confirmed_local_reference_is_bound_and_source_evidence_is_unchanged(self):
        local_evidence(self.root, self.source)
        evidence = {path: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in self.root.rglob('*') if path.is_file()}
        self.store = AnnotationStore(self.mesh, self.source)
        self.assertEqual(self.store.origin_reference['kind'], 'initial_color_camera')
        self.assertEqual(self.store.origin_reference['reference_frame_id'], 7)
        self.add_points()
        path = self.store.save(self.root / 'annotations')
        loaded = AnnotationStore(self.mesh, self.source)
        loaded.load(path)
        self.assertEqual(loaded.origin_reference, self.store.origin_reference)
        self.assertEqual(loaded.local_context['session_id'], 'session-test')
        self.assertEqual(loaded.points[0]['distance_to_origin_m'], .5)
        self.assertEqual(evidence, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in evidence})
        text = surface_readout_text(loaded, loaded.points[0], 'hit', 'zh')
        self.assertIn('距初始原点的直线距离', text)
        self.assertIn('500.0 mm', text)

    def test_same_mesh_different_session_or_reference_is_rejected(self):
        local_evidence(self.root, self.source)
        self.store = AnnotationStore(self.mesh, self.source)
        self.add_points()
        path = self.store.save(self.root / 'annotations')
        for session, frame in (('another-session', 7), ('session-test', 8)):
            local_evidence(self.root, self.source, session, frame)
            with self.subTest(session=session, frame=frame), self.assertRaises(ValueError):
                AnnotationStore(self.mesh, self.source).load(path)

    def test_rdf_or_identity_alone_never_proves_first_camera_origin(self):
        dataset, pose_path, tracking_path = local_evidence(self.root, self.source)
        pose = json.loads(pose_path.read_text(encoding='utf-8'))
        for change in (dict(pose_source='synthetic_ground_truth'), dict(session_id='wrong'),
                       dict(reference_frame_id=0),
                       dict(T_WC=[[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])):
            with self.subTest(change=change):
                write_json(pose_path, dict(pose, **change))
                self.assertEqual(AnnotationStore(self.mesh, self.source).origin_reference['provenance'], 'unknown')
        write_json(pose_path, pose)
        tracking_path.unlink()
        self.assertEqual(AnnotationStore(self.mesh, self.source).origin_reference['provenance'], 'unknown')

    def test_malformed_reference_evidence_is_unknown_and_boolean_frame_cannot_match(self):
        _, _, tracking = local_evidence(self.root, self.source, reference=0)
        self.store = AnnotationStore(self.mesh, self.source)
        saved = self.store.save(self.root / 'annotations')
        data = json.loads(saved.read_text(encoding='utf-8'))
        data['local_context']['reference_frame_id'] = False
        write_json(saved, data)
        with self.assertRaises(ValueError):
            self.store.load(saved)
        write_json(tracking, [])
        self.assertEqual(AnnotationStore(self.mesh, self.source).origin_reference['provenance'], 'unknown')

    def test_unknown_legacy_model_labels_model_origin_in_both_languages(self):
        self.add_points()
        zh = surface_readout_text(self.store, self.store.points[0], 'hit', 'zh')
        en = surface_readout_text(self.store, self.store.points[0], 'hit', 'en')
        self.assertIn('距模型坐标原点', zh)
        self.assertIn('来源未知', zh)
        self.assertIn('model origin', en)
        self.assertIn('source unknown', en)
        for text in (zh, en):
            self.assertIn('X: 300.0 mm', text)
            self.assertIn('Y: 400.0 mm', text)
            self.assertIn('Z: 0.0 mm', text)
            self.assertIn('500.0 mm', text)
            self.assertNotIn('Straight distance to initial origin', text)
            self.assertIn('mesh.ply', text)


class FakeCamera:
    def __init__(self):
        self.view = np.eye(4)
        self.projection = np.eye(4)

    def get_view_matrix(self):
        return self.view

    def get_projection_matrix(self):
        return self.projection

    def unproject(self, x, y, depth, width, height):
        return np.array([(x - width / 2) / 20., (y - height / 2) / 20., depth * 3])


class FakeScene:
    def __init__(self):
        self.camera = FakeCamera()
        self.geometries = {}

    def has_geometry(self, name):
        return name in self.geometries

    def add_geometry(self, name, geometry, material):
        self.geometries[name] = geometry

    def remove_geometry(self, name):
        del self.geometries[name]


class FakeCombo:
    def __init__(self):
        self.items = []
        self.selected_index = 0

    def clear_items(self):
        self.items.clear()

    def add_item(self, item):
        self.items.append(item)


class InlineWorker:
    def __init__(self, *, target, **kwargs):
        self.target = target

    def start(self):
        self.target()


class ViewerSelectionTests(unittest.TestCase):
    """Run real pick/annotation callbacks and ray casting with a queued GUI adapter."""
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        v = viewer.FemtoSceneViewer.__new__(viewer.FemtoSceneViewer)
        v.reconstruction = self.root
        v.mesh = triangle(1.)
        v.language = 'en'
        v._pick_generation, v._pick_busy, v._closing = 0, False, False
        v._pick_status = 'ready'
        v._init_annotations(self.root / 'mesh.ply')
        v.scene_visible = dict(mesh=True)
        v.window = SimpleNamespace(set_needs_layout=mock.Mock(), post_redraw=mock.Mock())
        v.scene_widget = SimpleNamespace(scene=FakeScene(), frame=SimpleNamespace(width=100, height=100),
                                         force_redraw=mock.Mock())
        v.pick_result_label = SimpleNamespace(text='', tooltip='')
        v.annotation_label = SimpleNamespace(text='')
        v.annotation_mode = FakeCombo()
        v.annotation_selection = FakeCombo()
        v.annotation_name = SimpleNamespace(text_value='Point A')
        v.annotation_note = SimpleNamespace(text_value='software fixture')
        self.v = v
        self.callbacks = []
        fake_app = SimpleNamespace(post_to_main_thread=lambda window, fn: self.callbacks.append(fn))
        self.thread_patch = mock.patch.object(viewer.threading, 'Thread', InlineWorker)
        self.gui_patch = mock.patch.object(viewer.gui, 'Application', SimpleNamespace(instance=fake_app))
        self.thread_patch.start()
        self.gui_patch.start()
        self.addCleanup(self.thread_patch.stop)
        self.addCleanup(self.gui_patch.stop)

    def pick(self, x=50, y=50):
        self.v.pick_surface_pixel(x, y)
        if self.callbacks:
            self.callbacks.pop(0)()

    def assert_no_current_reading(self):
        self.assertIsNone(self.v.selected_surface)
        self.assertIsNone(self.v._selected_annotation_index)
        self.assertNotIn('X:', self.v.pick_result_label.text)
        self.assertNotIn('Straight distance to model origin:', self.v.pick_result_label.text)
        self.assertFalse(self.v.scene_widget.scene.has_geometry('selected_surface_marker'))
        self.assertEqual(self.v.annotation_selection.selected_index, 0)

    def test_valid_hit_displays_name_mm_source_and_validation_state(self):
        self.pick()
        self.assertEqual(len(self.v.annotations.points), 1)
        self.assertEqual(self.v.selected_surface['name'], 'Point A')
        self.assertIn('X: 0.0 mm', self.v.pick_result_label.text)
        self.assertIn('Z: 1000.0 mm', self.v.pick_result_label.text)
        self.assertIn('1000.0 mm', self.v.pick_result_label.text)
        self.assertIn('Physical accuracy not validated', self.v.pick_result_label.text)
        self.assertEqual(self.v.annotation_selection.selected_index, 1)

    def test_background_click_clears_previous_reading_without_appending(self):
        self.pick()
        self.pick(99, 99)
        self.assertEqual(len(self.v.annotations.points), 1)
        self.assertEqual(self.v._pick_status, 'miss')
        self.assert_no_current_reading()

    def test_real_mesh_hole_is_not_snapped_to_nearby_geometry(self):
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector([
            [-1,-1,1], [1,-1,1], [1,1,1], [-1,1,1],
            [-.2,-.2,1], [.2,-.2,1], [.2,.2,1], [-.2,.2,1]])
        mesh.triangles = o3d.utility.Vector3iVector([
            [0,1,5], [0,5,4], [1,2,6], [1,6,5],
            [2,3,7], [2,7,6], [3,0,4], [3,4,7]])
        self.v.mesh = mesh
        self.v._init_annotations(self.root / 'hole_mesh.ply')
        self.pick(62, 50)
        self.assertEqual(len(self.v.annotations.points), 1)
        self.pick()
        self.assertEqual(len(self.v.annotations.points), 1)
        self.assertEqual(self.v._pick_status, 'miss')
        self.assert_no_current_reading()

    def test_nonfinite_pick_and_hidden_mesh_never_reuse_old_reading(self):
        self.pick()
        self.v._surface_picker = SimpleNamespace(pick=lambda near, far: dict(
            point_world_m=[np.nan, 0., 1.], triangle_index=0))
        self.pick()
        self.assertEqual(len(self.v.annotations.points), 1)
        self.assertEqual(self.v._pick_status, 'error')
        self.assert_no_current_reading()
        self.v.scene_visible['mesh'] = False
        self.pick()
        self.assertEqual(self.v._pick_status, 'show_mesh')
        self.assert_no_current_reading()

    def test_clear_selection_and_clear_all_cancel_pending_pick_results(self):
        for clear in (self.v.clear_surface_pick, self.v._clear_annotations):
            with self.subTest(clear=clear.__name__):
                self.v.pick_surface_pixel(50, 50)
                clear()
                self.callbacks.pop(0)()
                self.assertEqual(self.v.annotations.points, [])
                self.assert_no_current_reading()

    def test_busy_and_outside_click_invalidate_older_async_result(self):
        for x in (50, -1):
            with self.subTest(x=x):
                self.v.pick_surface_pixel(50, 50)
                self.v.pick_surface_pixel(x, 50)
                self.callbacks.pop(0)()
                self.assertEqual(self.v.annotations.points, [])
                self.assert_no_current_reading()
        self.pick()
        self.assertEqual(len(self.v.annotations.points), 1)

    def test_undo_selects_remaining_point_and_final_undo_clears_readout(self):
        self.pick()
        self.v.annotation_name.text_value = 'Point B'
        self.pick(60, 50)
        self.assertIn('500.000 mm', self.v.annotation_label.text)
        self.v._undo_annotation()
        self.assertEqual(len(self.v.annotations.points), 1)
        self.assertEqual(self.v.selected_surface['name'], 'Point A')
        self.assertIn('X: 0.0 mm', self.v.pick_result_label.text)
        self.assertNotIn('distance between last two', self.v.annotation_label.text)
        self.v._undo_annotation()
        self.assert_no_current_reading()
        self.assertEqual(len(self.v.annotation_selection.items), 1)

    def test_load_selects_last_marker_and_selector_can_read_prior_marker(self):
        self.pick()
        self.v.annotation_name.text_value = 'Point B'
        self.pick(60, 50)
        saved = self.v.annotations.save(self.root / 'annotations')
        self.v._clear_annotations()
        self.assert_no_current_reading()
        self.assertTrue(self.v._load_annotations(saved))
        self.assertEqual(self.v.selected_surface['name'], 'Point B')
        self.v._annotation_selected('1. Point A', 1)
        self.assertEqual(self.v.selected_surface['name'], 'Point A')
        self.assertIn('X: 0.0 mm', self.v.pick_result_label.text)
        self.v.clear_surface_pick()
        self.assertEqual(len(self.v.annotations.points), 2)
        self.assert_no_current_reading()

    def test_failed_load_keeps_annotations_but_clears_current_readout(self):
        self.pick()
        saved = self.v.annotations.save(self.root / 'annotations')
        data = json.loads(saved.read_text(encoding='utf-8'))
        data['geometry_sha256'] = 'wrong-model'
        write_json(saved, data)
        self.assertFalse(self.v._load_annotations(saved))
        self.assertEqual(len(self.v.annotations.points), 1)
        self.assert_no_current_reading()

    def test_model_switch_clears_readout_markers_and_pending_result(self):
        self.pick()
        self.v.pick_surface_pixel(50, 50)
        self.v._init_annotations(self.root / 'another_mesh.ply')
        self.callbacks.pop(0)()
        self.assertEqual(self.v.annotations.points, [])
        self.assertEqual(self.v.scene_widget.scene.geometries, {})
        self.assert_no_current_reading()

    def test_view_rotation_zoom_and_capture_pose_do_not_transform_model_point_again(self):
        self.v.capture_poses = [np.eye(4), np.array([
            [0., -1., 0., 8.], [1., 0., 0., -3.], [0., 0., 1., 2.], [0., 0., 0., 1.]])]
        self.pick(60, 50)
        before = dict(self.v.selected_surface)
        np.testing.assert_allclose(before['point_world_m'], [.5, 0., 1.], atol=1e-6)
        self.assertAlmostEqual(before['distance_to_origin_m'], np.sqrt(1.25), places=6)
        self.v.scene_widget.scene.camera.view = self.v.capture_poses[1].copy()
        self.v.scene_widget.scene.camera.projection *= 2.
        self.v._update_pick_label()
        self.assertEqual(self.v.selected_surface, before)
        saved = self.v.annotations.save(self.root / 'annotations')
        self.assertTrue(self.v._load_annotations(saved))
        self.assertEqual(self.v.selected_surface, before)

    def test_camera_change_during_pending_pick_discards_old_hit(self):
        self.v.pick_surface_pixel(50, 50)
        self.v.scene_widget.scene.camera.view[0, 3] = 1.
        self.callbacks.pop(0)()
        self.assertEqual(self.v.annotations.points, [])
        self.assert_no_current_reading()


if __name__ == '__main__':
    unittest.main()
