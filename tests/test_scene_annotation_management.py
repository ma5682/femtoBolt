"""Point/region editing, undo, lineage and asynchronous GUI regression tests."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import open3d as o3d

import test_scene_annotation_origin as origin
from scene_annotations import AnnotationStore, region_boundary


class AnnotationManagementTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.mesh = origin.triangle()
        self.store = AnnotationStore(self.mesh, self.root / 'mesh.ply')

    def add_items(self):
        self.store.add_point(dict(point_world_m=[0., 0., 0.], triangle_index=0), name='A')
        self.store.add_point(dict(point_world_m=[.3, .4, 0.], triangle_index=0), name='B')
        self.store.add_region(dict(triangle_indices=[0], name='Region', note='Initial note'))

    def test_edit_preserves_geometry_identity_and_creation_time(self):
        self.add_items()
        before = copy.deepcopy(self.store.points[1])
        changed = self.store.update_annotation('point', 1, name='测量 B', note='Reviewed')
        for key in ('point_world_m', 'distance_to_origin_m', 'annotation_id', 'created_at'):
            self.assertEqual(changed[key], before[key])
        self.assertTrue(changed['created_at'])
        self.assertEqual(changed['note'], 'Reviewed')
        self.assertEqual(self.store.undo(), ('point', before['annotation_id']))
        self.assertEqual(self.store.points[1], before)
        before_region = copy.deepcopy(self.store.regions[0])
        self.store.update_annotation('region', 0, name='区域 A', note='R&D only')
        self.assertEqual(self.store.regions[0]['boundary_world_m'], before_region['boundary_world_m'])
        self.assertEqual(self.store.regions[0]['annotation_id'], before_region['annotation_id'])
        saved = self.store.save(self.root / 'saved')
        other = AnnotationStore(self.mesh, self.store.source)
        other.load(saved)
        self.assertEqual(other.regions[0], self.store.regions[0] | {
            'mesh_file': 'region_1_mesh_m.ply', 'point_cloud_file': 'region_1_vertices_m.ply',
            'glb_file': 'region_1_mesh_m.glb', 'boundary_file': 'region_1_boundary_m.csv',
            'region_manifest_file': 'region_1.json'})
        self.assertAlmostEqual(other.points[1]['distance_to_origin_m'], .5)

    def test_delete_and_clear_undo_restore_exact_items_and_pair_distance(self):
        self.add_items()
        before = copy.deepcopy((self.store.points, self.store.regions))
        self.store.delete_annotation('point', 0)
        self.assertEqual(self.store.points[0]['name'], 'B')
        saved = self.store.save(self.root / 'saved')
        self.assertEqual(json.loads(saved.read_text(encoding='utf-8'))['measurements'], [])
        self.store.undo()
        self.assertEqual((self.store.points, self.store.regions), before)
        self.store.clear_annotations(('region', before[1][0]['annotation_id']))
        self.assertEqual(self.store.points + self.store.regions, [])
        self.assertEqual(self.store.undo(), ('region', before[1][0]['annotation_id']))
        self.assertEqual((self.store.points, self.store.regions), before)
        saved = self.store.save(self.root / 'saved')
        self.assertAlmostEqual(json.loads(saved.read_text(encoding='utf-8'))['measurements'][0]['distance_m'], .5)

    def test_legacy_ids_are_assigned_once_without_inventing_creation_time(self):
        self.add_items()
        saved = self.store.save(self.root / 'saved')
        data = json.loads(saved.read_text(encoding='utf-8'))
        for item in data['points'] + data['regions']:
            for key in ('annotation_id', 'created_at', 'updated_at', 'boundary_world_m'):
                item.pop(key, None)
        origin.write_json(saved, data)
        raw = saved.read_bytes()
        self.store.load(saved)
        ids = [item['annotation_id'] for item in self.store.points + self.store.regions]
        self.assertEqual(len(set(ids)), 3)
        self.assertTrue(all(item['created_at'] is None for item in self.store.points + self.store.regions))
        again = self.store.save(self.root / 'saved')
        self.store.load(again)
        self.assertEqual(ids, [item['annotation_id'] for item in self.store.points + self.store.regions])
        self.assertEqual(raw, saved.read_bytes())
        self.assertFalse(self.store.can_undo)

    def test_region_boundaries_recomputed_from_mesh_with_hole_preserved(self):
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector([
            [-1,-1,0], [1,-1,0], [1,1,0], [-1,1,0],
            [-.2,-.2,0], [.2,-.2,0], [.2,.2,0], [-.2,.2,0]])
        mesh.triangles = o3d.utility.Vector3iVector([
            [0,1,5], [0,5,4], [1,2,6], [1,6,5],
            [2,3,7], [2,7,6], [3,0,4], [3,4,7]])
        store = AnnotationStore(mesh, self.root / 'ring.ply')
        ids = list(range(8))
        store.add_region(dict(triangle_indices=ids, boundary_world_m=[[[99,99,99], [100,100,100]]]))
        self.assertEqual(len(store.regions[0]['boundary_world_m']), 8)
        expected = region_boundary(mesh, ids)
        saved = store.save(self.root / 'saved')
        data = json.loads(saved.read_text(encoding='utf-8'))
        data['regions'][0]['boundary_world_m'] = [[[9,9,9], [10,10,10]]]
        origin.write_json(saved, data)
        store.load(saved)
        self.assertEqual(store.regions[0]['boundary_world_m'], expected)
        self.assertEqual(len(mesh.triangles), 8)

    def test_bad_region_metadata_and_duplicate_ids_do_not_partially_load(self):
        self.add_items()
        saved = self.store.save(self.root / 'saved')
        good = json.loads(saved.read_text(encoding='utf-8'))
        before = copy.deepcopy((self.store.points, self.store.regions))
        for changes in (dict(triangle_indices=[True]), dict(triangle_indices=[0,0]),
                        dict(unit='millimetre'), dict(source_mesh='different.ply'),
                        dict(boundary_world_m=[[[np.nan,0,0], [0,0,0]]]),
                        dict(annotation_id=good['points'][0]['annotation_id']),
                        dict(name=123), dict(created_at='not-a-date')):
            with self.subTest(changes=changes):
                data = copy.deepcopy(good)
                data['regions'][0].update(changes)
                origin.write_json(saved, data)
                with self.assertRaises((ValueError, TypeError)):
                    self.store.load(saved)
                self.assertEqual((self.store.points, self.store.regions), before)


class ViewerManagementTests(unittest.TestCase):
    setUp = origin.ViewerSelectionTests.setUp
    pick = origin.ViewerSelectionTests.pick
    assert_no_current_reading = origin.ViewerSelectionTests.assert_no_current_reading

    def polygon(self, finish=True):
        self.v.pick_checkbox = SimpleNamespace(checked=True)
        self.v.annotation_mode.selected_index = 2
        for x, y in ((5,5), (95,5), (95,95), (5,95)):
            self.v._polygon_click(x, y)
        if finish:
            self.v._finish_polygon()

    def complete_polygon(self):
        self.polygon()
        self.callbacks.pop(0)()
        self.assertEqual(len(self.v.annotations.regions), 1)

    def test_region_selection_edits_and_delete_have_no_stale_point_reading(self):
        self.pick()
        self.complete_polygon()
        self.assertIsNone(self.v.selected_surface)
        self.assertNotIn('X:', self.v.pick_result_label.text)
        self.assertEqual(self.v._selected_region_index, 0)
        self.assertEqual(self.v.annotation_selection.selected_index, 2)
        xyz = copy.deepcopy(self.v.annotations.points[0]['point_world_m'])
        self.v.annotation_name.text_value = '区域 R / Region R'
        self.v.annotation_note.text_value = 'Reviewed surface'
        self.v._apply_annotation_edits()
        self.assertEqual(self.v.annotations.regions[0]['name'], '区域 R / Region R')
        self.assertIn('Reviewed surface', self.v.annotation_label.text)
        self.assertEqual(self.v.annotations.points[0]['point_world_m'], xyz)
        self.v._delete_annotation()
        self.assertEqual(self.v.annotations.regions, [])
        self.assert_no_current_reading()
        self.v._undo_annotation()
        self.assertEqual(self.v._selected_region_index, 0)
        self.assertIsNone(self.v.selected_surface)
        self.assertNotIn('X:', self.v.pick_result_label.text)

    def test_delete_middle_point_and_undo_restore_identity_and_readout(self):
        self.pick()
        self.pick(55, 50)
        self.pick(60, 50)
        self.v._select_annotation(1)
        identifier = self.v.selected_surface['annotation_id']
        self.v._delete_annotation()
        self.assert_no_current_reading()
        self.assertEqual(len(self.v.annotations.points), 2)
        self.assertIn('500.000 mm', self.v.annotation_label.text)
        self.v._undo_annotation()
        self.assertEqual(self.v.selected_surface['annotation_id'], identifier)
        self.assertIn('X: 250.0 mm', self.v.pick_result_label.text)
        self.assertIn('250.000 mm', self.v.annotation_label.text)

    def test_clear_undo_and_region_only_load_select_correct_item(self):
        self.complete_polygon()
        identifier = self.v.annotations.regions[0]['annotation_id']
        saved = self.v.annotations.save(self.root / 'saved')
        self.v._clear_annotations()
        self.assert_no_current_reading()
        self.v._undo_annotation()
        self.assertEqual(self.v.annotations.regions[0]['annotation_id'], identifier)
        self.assertEqual(self.v._selected_region_index, 0)
        self.assertTrue(self.v._load_annotations(saved))
        self.assertEqual(self.v._selected_region_index, 0)
        self.assertFalse(self.v.annotations.can_undo)
        self.assertNotIn('X:', self.v.pick_result_label.text)
        for language in ('zh', 'en'):
            self.v.language = language
            self.v._update_pick_label()
            self.assertNotIn('X:', self.v.pick_result_label.text)
            self.assertIn('1', self.v.annotation_label.text)

    def test_pending_point_keeps_name_and_note_from_click_time(self):
        self.v.pick_surface_pixel(50, 50)
        self.v.annotation_name.text_value = 'Later draft'
        self.v.annotation_note.text_value = 'Later note'
        self.callbacks.pop(0)()
        self.assertEqual(self.v.selected_surface['name'], 'Point A')
        self.assertEqual(self.v.selected_surface['note'], 'software fixture')

    def test_pending_region_keeps_input_and_is_cancelled_by_clear_or_model_change(self):
        self.polygon()
        self.v.annotation_name.text_value = 'Later draft'
        self.callbacks.pop(0)()
        self.assertEqual(self.v.annotations.regions[0]['name'], 'Point A')
        self.v._clear_annotations()
        self.polygon()
        self.v._clear_annotations()
        self.callbacks.pop(0)()
        self.assertEqual(self.v.annotations.regions, [])
        self.polygon()
        self.v._init_annotations(self.root / 'different.ply')
        self.polygon()
        self.assertTrue(self.v._region_busy)
        self.callbacks.pop(0)()
        self.assertTrue(self.v._region_busy)  # An old model's callback cannot finish the new job.
        self.callbacks.pop(0)()
        self.assertEqual(len(self.v.annotations.regions), 1)

    def test_invalid_polygon_input_and_view_change_do_not_create_region_or_keep_reading(self):
        self.pick()
        self.v._polygon_click(float('nan'), 1)
        self.assertEqual(self.v._polygon, [])
        self.assert_no_current_reading()
        self.polygon()
        self.v.scene_widget.scene.camera.view[0, 3] = 1.
        self.callbacks.pop(0)()
        self.assertEqual(self.v.annotations.regions, [])
        self.assert_no_current_reading()

    def test_unsaved_editor_changes_require_apply_and_saved_item_is_current(self):
        self.pick()
        self.v.annotation_note.text_value = 'New note'
        self.v._save_annotations()
        self.assertFalse((self.root / 'annotations').exists())
        self.assertIn('Apply', self.v._annotation_message)
        self.v._apply_annotation_edits()
        self.v._save_annotations()
        saved = next((self.root / 'annotations').glob('*/annotations.json'))
        self.assertEqual(json.loads(saved.read_text(encoding='utf-8'))['points'][0]['note'], 'New note')
        self.assertIs(self.v.selected_surface, self.v.annotations.points[0])


if __name__ == '__main__':
    unittest.main()
