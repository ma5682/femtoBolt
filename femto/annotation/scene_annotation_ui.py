"""Surface measurement and polygon controls for the desktop viewer."""
from __future__ import annotations
from typing import Any

import threading
import unicodedata
from pathlib import Path
import numpy as np
import open3d as o3d
from open3d.visualization import gui, rendering
from femto.annotation.scene_annotations import AnnotationStore, distance_m, point_readout_mm, visible_region, region_mesh, region_boundary
from femto.ui_i18n import tr


def wrap_readout(text: Any, columns: int=38) -> Any:
    """Wrap long model/annotation names in the narrow Open3D side panel."""
    lines, line, width = [], '', 0
    for char in str(text):
        size = 2 if unicodedata.east_asian_width(char) in ('W', 'F') else 1
        if char == '\n':
            lines.append(line)
            line, width = '', 0
            continue
        if line and width + size > columns:
            split = line.rfind(' ')
            if split > len(line) // 2:
                lines.append(line[:split])
                line = line[split + 1:]
                width = sum(2 if unicodedata.east_asian_width(c) in ('W', 'F') else 1 for c in line)
            else:
                lines.append(line)
                line, width = '', 0
        line += char
        width += size
    lines.append(line)
    return '\n'.join(lines)


def surface_readout_text(store: Any, point: Any, status: str, language: str, columns: int=38) -> Any:
    """Format stored W/model coordinates only; the viewing camera is irrelevant."""
    def text(key, **values):
        return tr(language, 'viewer.pick.' + key, **values)

    lines = [text(status)]
    if point is not None:
        xyz, distance = point_readout_mm(point['point_world_m'])
        lines.append(text('name', name=point.get('name') or text('unnamed')))
        for axis, value in zip('XYZ', xyz):
            lines.append(f'{axis}: {0.0 if abs(value) < .05 else value:.1f} mm')
        key = 'origin_distance' if store.origin_reference['kind'] == 'initial_color_camera' else 'model_distance'
        lines.extend((text(key), f'{distance:.1f} mm'))
    if store.origin_reference['kind'] == 'initial_color_camera':
        lines.append(text('initial_frame', frame=store.origin_reference['reference_frame_id']))
    else:
        lines.append(text('unknown_frame'))
    lines.append(text('frame'))
    source = Path(store.source)
    model_name = source.parent.parent.name if source.parent.name == 'reconstruction' else source.parent.name
    lines.extend((text('source'), f'{model_name}/{source.name}',
                  text('estimate'), text('precision'), text('research')))
    return wrap_readout('\n'.join(lines), columns)


class AnnotationUI:
    def _init_annotations(self, mesh_path):
        annotations = AnnotationStore(self.mesh, mesh_path)
        if hasattr(self, 'scene_widget'):
            self.clear_surface_pick()
            for name in getattr(self, '_annotation_names', []):
                if self.scene_widget.scene.has_geometry(name):
                    self.scene_widget.scene.remove_geometry(name)
        self.annotations = annotations
        self.selected_surface = None
        self._selected_annotation_index = None
        self._selected_region_index = None
        self._surface_picker = None
        self._pick_generation = getattr(self, '_pick_generation', 0) + 1
        self._polygon = []
        self._region_busy = False
        self._region_job = None
        self._annotation_names = []
        self._annotation_message = ''
        self._updating_annotation_choices = False
        if hasattr(self, 'pick_result_label'):
            self._update_pick_label()

    def _annotation_text(self, zh, en):
        return zh if self.language == 'zh' else en

    def _build_annotations(self):
        self.annotation_name = gui.TextEdit()
        self.annotation_name.text_value = self._annotation_text('标记 01', 'Marker 01')
        self.annotation_note = gui.TextEdit()
        self.annotation_selection_label = gui.Label('')
        self.panel.add_child(self.annotation_selection_label)
        self.annotation_selection = gui.Combobox()
        self.annotation_selection.set_on_selection_changed(self._annotation_selected)
        self.panel.add_child(self.annotation_selection)
        self.annotation_name_label = gui.Label('')
        self.panel.add_child(self.annotation_name_label)
        self.panel.add_child(self.annotation_name)
        self.panel.add_child(self.annotation_note)
        self.annotation_actions = {}
        edit_row = gui.Horiz(4)
        for key, callback in [('apply', self._apply_annotation_edits), ('delete', self._delete_annotation)]:
            button = gui.Button('')
            button.set_on_clicked(callback)
            edit_row.add_child(button)
            self.annotation_actions[key] = button
        self.panel.add_child(edit_row)
        self.annotation_mode = gui.Combobox()
        self.annotation_mode.set_on_selection_changed(self._annotation_mode_changed)
        self.panel.add_child(self.annotation_mode)
        self.annotation_label = gui.Label('')
        self.panel.add_child(self.annotation_label)
        actions = [
            ('finish', self._finish_polygon),
            ('undo', self._undo_annotation),
            ('clear', self._clear_annotations),
            ('save', self._save_annotations),
            ('load', self._load_annotations_dialog),
        ]
        for key, callback in actions:
            button = gui.Button('')
            button.set_on_clicked(callback)
            self.panel.add_child(button)
            self.annotation_actions[key] = button
        self._apply_annotation_language()
        self._update_annotation_label()

    def _apply_annotation_language(self):
        if hasattr(self, 'annotation_selection_label'):
            self.annotation_selection_label.text = self._t('viewer.pick.saved_points')
            self.annotation_name_label.text = self._t('viewer.pick.name_note')
            self.annotation_note.tooltip = self._t('viewer.annotation.note')
            for key, button in self.annotation_actions.items():
                button.text = self._t('viewer.annotation.' + key)
            index = max(0, self.annotation_mode.selected_index)
            self._updating_annotation_choices = True
            try:
                self.annotation_mode.clear_items()
                for key in ('point', 'distance', 'polygon'):
                    self.annotation_mode.add_item(self._t('viewer.annotation.mode_' + key))
                self.annotation_mode.selected_index = index
            finally:
                self._updating_annotation_choices = False
            self._refresh_annotation_choices()

    def _layout_annotation_readouts(self, context, available_width):
        # Font fallback and Windows DPI change glyph widths. Measure with the
        # actual Open3D font instead of assuming every Latin glyph has equal size.
        for columns in range(38, 15, -1):
            self.pick_result_label.text = surface_readout_text(
                self.annotations, self.selected_surface, self._pick_status, self.language, columns)
            self.annotation_label.text = wrap_readout(self._annotation_label_raw, columns)
            widths = [label.calc_preferred_size(context, gui.Widget.Constraints()).width
                      for label in (self.pick_result_label, self.annotation_label)]
            if max(widths) <= available_width:
                break
        self._readout_columns = columns

    def _refresh_annotation_choices(self):
        if not hasattr(self, 'annotation_selection'):
            return
        self._updating_annotation_choices = True
        try:
            self.annotation_selection.clear_items()
            self.annotation_selection.add_item(self._t('viewer.pick.no_selection'))
            for kind, items in [('point', self.annotations.points), ('region', self.annotations.regions)]:
                for i, item in enumerate(items):
                    name = item.get('name') or self._t('viewer.pick.unnamed')
                    lines = wrap_readout(name, 24).splitlines()
                    short = lines[0] + ('…' if len(lines) > 1 else '')
                    self.annotation_selection.add_item(self._t('viewer.annotation.' + kind, number=i + 1) + ' · ' + short)
            index = self._selected_annotation_index
            region_index = self._selected_region_index
            self.annotation_selection.selected_index = (index + 1 if index is not None else
                len(self.annotations.points) + region_index + 1 if region_index is not None else 0)
            for key in ('apply', 'delete'):
                if key in getattr(self, 'annotation_actions', {}):
                    self.annotation_actions[key].enabled = index is not None or region_index is not None
            if 'undo' in getattr(self, 'annotation_actions', {}):
                self.annotation_actions['undo'].enabled = bool(self._polygon) or self.annotations.can_undo
        finally:
            self._updating_annotation_choices = False

    def _annotation_selected(self, _text, index):
        if self._updating_annotation_choices:
            return
        if index > len(self.annotations.points):
            self._select_region(index - len(self.annotations.points) - 1)
        else:
            self._select_annotation(index - 1 if index > 0 else None)

    def _selected_item(self):
        if self._selected_annotation_index is not None:
            return 'point', self._selected_annotation_index, self.annotations.points[self._selected_annotation_index]
        if self._selected_region_index is not None:
            return 'region', self._selected_region_index, self.annotations.regions[self._selected_region_index]
        return None

    def _selection_focus(self):
        selected = self._selected_item()
        return (selected[0], selected[2]['annotation_id']) if selected else None

    def _focus_annotation(self, focus):
        if focus is not None:
            kind, identifier = focus
            items = self.annotations.points if kind == 'point' else self.annotations.regions
            for i, item in enumerate(items):
                if item['annotation_id'] == identifier:
                    (self._select_annotation if kind == 'point' else self._select_region)(i)
                    return
        self.clear_surface_pick()

    def _fill_annotation_editor(self, item):
        self.annotation_name.text_value = item.get('name', '')
        self.annotation_note.text_value = item.get('note', '')

    def _select_annotation(self, index):
        self.clear_surface_pick()
        if index is not None and 0 <= index < len(self.annotations.points):
            self._selected_annotation_index = index
            self.selected_surface = self.annotations.points[index]
            self._fill_annotation_editor(self.selected_surface)
            self._pick_status = 'hit'
            marker = o3d.geometry.TriangleMesh.create_sphere(radius=.003)
            marker.translate(self.selected_surface['point_world_m'])
            material = rendering.MaterialRecord()
            material.shader = 'defaultUnlit'
            material.base_color = [1., .18, .05, 1.]
            self.scene_widget.scene.add_geometry('selected_surface_marker', marker, material)
        self._update_pick_label()
        self.scene_widget.force_redraw()

    def _select_region(self, index):
        self.clear_surface_pick()
        if index is not None and 0 <= index < len(self.annotations.regions):
            self._selected_region_index = index
            region = self.annotations.regions[index]
            self._fill_annotation_editor(region)
            self._pick_status = 'region'
            lines = o3d.geometry.LineSet.create_from_triangle_mesh(
                region_mesh(self.mesh, region['triangle_indices']))
            material = rendering.MaterialRecord()
            material.shader = 'unlitLine'
            material.base_color = [1., .18, .05, 1.]
            material.line_width = 5.
            self.scene_widget.scene.add_geometry('selected_surface_marker', lines, material)
        self._update_pick_label()
        self.scene_widget.force_redraw()

    def _annotation_mode_changed(self, text, index):
        if self._updating_annotation_choices:
            return
        self.clear_surface_pick()
        self.pick_checkbox.checked = True
        self._polygon = []
        self._render_annotations()

    def _update_annotation_label(self):
        if not hasattr(self, 'annotation_label'):
            return
        points = self.annotations.points
        text = self._annotation_text('未经过实物尺度验证', 'Not physically scale-validated')
        text += self._annotation_text(f'\n选点 {len(points)} · 区域 {len(self.annotations.regions)}', f'\nPoints {len(points)} · Regions {len(self.annotations.regions)}')
        selected = self._selected_item()
        if selected:
            kind, _, item = selected
            if kind == 'region':
                text += '\n' + self._t('viewer.annotation.region_info',
                    name=item.get('name') or self._t('viewer.pick.unnamed'),
                    count=len(item['triangle_indices']))
            text += '\n' + self._t('viewer.annotation.note') + ': ' + (item.get('note') or '—')
            if item.get('created_at') is None:
                text += '\n' + self._t('viewer.annotation.legacy_time')
        if len(points) >= 2:
            distance = distance_m(points[-2], points[-1])
            text += self._annotation_text('\n末两标记之间的直线距离', '\nStraight distance between last two markers')
            text += f'\n{distance:.6f} m / {distance*1000:.3f} mm'
        text += self._annotation_text('\n圈选：依次点击边界，再完成\n按三角面中心选取可见表面', '\nPolygon: click corners, then Finish\nVisible triangle centers define selection')
        if self._polygon:
            text += self._annotation_text(f'\n边界点：{len(self._polygon)}', f'\nCorners: {len(self._polygon)}')
        if self._annotation_message:
            text += '\n' + self._annotation_message
        self._annotation_label_raw = text
        self.annotation_label.text = wrap_readout(text, getattr(self, '_readout_columns', 38))
        self.window.set_needs_layout()

    def _accept_annotation_point(self, hit, *, name=None, note=None):
        self.annotations.add_point(hit,
            name=(self.annotation_name.text_value.strip() if name is None else name) or self._t('viewer.pick.unnamed'),
            note=self.annotation_note.text_value.strip() if note is None else note)
        self._select_annotation(len(self.annotations.points) - 1)
        self._annotation_message = ''
        self._render_annotations()

    def _polygon_click(self, x, y):
        self.clear_surface_pick()
        frame = self.scene_widget.frame
        if not (np.isfinite([x, y]).all() and 0 <= x < frame.width and 0 <= y < frame.height):
            self._pick_status = 'miss'
            self._update_pick_label()
            return
        if self._region_busy or not self.scene_visible['mesh']:
            self._pick_status = 'retry' if self._region_busy else 'show_mesh'
            self._update_pick_label()
            return
        camera, frame = self.scene_widget.scene.camera, self.scene_widget.frame
        state = (np.asarray(camera.get_view_matrix()).copy(), np.asarray(camera.get_projection_matrix()).copy(), frame.width, frame.height)
        if self._polygon and not self._same_polygon_camera():
            self._polygon = []
        if not self._polygon:
            self._polygon_camera = state
        self._polygon.append([float(x), float(y)])
        self._render_annotations()

    def _region_boundary(self, ids):
        return region_boundary(self.mesh, ids)

    def _same_polygon_camera(self, state=None):
        view, projection, width, height = self._polygon_camera if state is None else state
        camera, frame = self.scene_widget.scene.camera, self.scene_widget.frame
        return (width == frame.width and height == frame.height and
                np.allclose(view, camera.get_view_matrix()) and np.allclose(projection, camera.get_projection_matrix()))

    def _finish_polygon(self):
        if self._region_busy:
            return
        if len(self._polygon) < 3 or not self.scene_visible['mesh']:
            self._annotation_message = self._annotation_text('请显示网格并点击至少三个边界点', 'Show mesh and click at least three corners')
            self._update_annotation_label()
            return
        if not self._same_polygon_camera():
            self._polygon = []
            self._annotation_message = self._annotation_text('视角已变，请重新圈选', 'Camera changed; redraw polygon')
            self._render_annotations()
            return
        self.clear_surface_pick()
        polygon = [corner[:] for corner in self._polygon]
        camera_state = self._polygon_camera
        view, projection, width, height = camera_state
        mesh, annotations = self.mesh, self.annotations
        name, note = self.annotation_name.text_value.strip(), self.annotation_note.text_value.strip()
        generation = self._pick_generation
        job = self._region_job = object()
        self._region_busy = True
        self._annotation_message = self._annotation_text('正在提取可见区域…', 'Extracting visible region…')
        self._update_annotation_label()
        def worker():
            ids, error = [], None
            try:
                ids = visible_region(mesh, polygon, view, projection, width, height)
            except Exception as exc:
                error = str(exc)
            def finish():
                nonlocal error
                if self._closing or self._region_job is not job:
                    return
                self._region_busy = False
                self._region_job = None
                if (generation != self._pick_generation or self.mesh is not mesh or self.annotations is not annotations
                        or not self._same_polygon_camera(camera_state) or not self.scene_visible['mesh']):
                    self._annotation_message = self._annotation_text('操作已改变，区域结果已取消', 'Selection changed; region discarded')
                    self._update_annotation_label()
                    return
                if ids and not error:
                    try:
                        self.annotations.add_region(dict(triangle_indices=ids, screen_polygon=polygon,
                            name=name or self._t('viewer.pick.unnamed'), note=note,
                            view_matrix=view.tolist(), projection_matrix=projection.tolist(), viewport=[width,height],
                            selection_rule='visible_triangle_centroid_inside_polygon'))
                        self._polygon = []
                        self._select_region(len(self.annotations.regions) - 1)
                    except (ValueError, TypeError, KeyError) as exc:
                        error = str(exc)
                self._annotation_message = error or self._annotation_text(f'选中 {len(ids)} 个三角面', f'Selected {len(ids)} triangles')
                self._render_annotations()
            if not self._closing:
                gui.Application.instance.post_to_main_thread(self.window, finish)
        threading.Thread(target=worker, daemon=True, name='surface-region').start()

    def _render_annotations(self):
        scene = self.scene_widget.scene
        for name in self._annotation_names:
            if scene.has_geometry(name):
                scene.remove_geometry(name)
        self._annotation_names = []
        def add(geometry, color, lines=False):
            name = f'annotation_{len(self._annotation_names)}'
            material = rendering.MaterialRecord()
            material.shader = 'unlitLine' if lines else 'defaultUnlit'
            material.base_color = color
            material.line_width = 3.
            scene.add_geometry(name, geometry, material)
            self._annotation_names.append(name)
        points = self.annotations.points
        for p in points:
            sphere = o3d.geometry.TriangleMesh.create_sphere(.0025)
            sphere.translate(p['point_world_m'])
            add(sphere, [1.,.35,.02,1.])
        if len(points) > 1:
            line = o3d.geometry.LineSet()
            line.points = o3d.utility.Vector3dVector([p['point_world_m'] for p in points[-2:]])
            line.lines = o3d.utility.Vector2iVector([[0,1]])
            add(line, [1.,.9,.05,1.], True)
        for region in self.annotations.regions:
            mesh = region_mesh(self.mesh, region['triangle_indices'])
            lines = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
            add(lines, [.1,1.,.45,1.], True)
        if len(self._polygon) > 1 and self._same_polygon_camera():
            camera, frame = self.scene_widget.scene.camera, self.scene_widget.frame
            line = o3d.geometry.LineSet()
            line.points = o3d.utility.Vector3dVector([camera.unproject(x,y,.01,frame.width,frame.height) for x,y in self._polygon])
            line.lines = o3d.utility.Vector2iVector([[i,(i+1)%len(self._polygon)] for i in range(len(self._polygon))])
            add(line, [1.,.7,.05,1.], True)
        self._refresh_annotation_choices()
        self._update_annotation_label()
        self.scene_widget.force_redraw()

    def _undo_annotation(self):
        self.clear_surface_pick()
        focus = None
        if self._polygon:
            self._polygon.pop()
        elif self.annotations.can_undo:
            focus = self.annotations.undo()
        self._annotation_message = ''
        self._render_annotations()
        self._focus_annotation(focus)

    def _apply_annotation_edits(self):
        selected = self._selected_item()
        if selected is None:
            return
        kind, index, _ = selected
        focus = self._selection_focus()
        self.annotations.update_annotation(kind, index,
            name=self.annotation_name.text_value.strip() or self._t('viewer.pick.unnamed'),
            note=self.annotation_note.text_value.strip())
        self._focus_annotation(focus)
        self._annotation_message = self._t('viewer.annotation.applied')
        self._render_annotations()

    def _delete_annotation(self):
        selected = self._selected_item()
        if selected is None:
            return
        kind, index, _ = selected
        self.clear_surface_pick()
        self.annotations.delete_annotation(kind, index)
        self._annotation_message = self._t('viewer.annotation.deleted')
        self._render_annotations()

    def _clear_annotations(self):
        focus = self._selection_focus()
        self.clear_surface_pick()
        self.annotations.clear_annotations(focus)
        self._polygon = []
        self._annotation_message = ''
        self._render_annotations()
        self._update_pick_label()

    def _save_annotations(self):
        selected = self._selected_item()
        if self._pick_busy or self._region_busy or self._polygon:
            self._annotation_message = self._annotation_text('请先完成选点／圈选', 'Finish picking / polygon first')
        elif selected and (self.annotation_name.text_value.strip() != selected[2].get('name', '')
                           or self.annotation_note.text_value.strip() != selected[2].get('note', '')):
            self._annotation_message = self._t('viewer.annotation.apply_first')
        else:
            try:
                focus = self._selection_focus()
                path = self.annotations.save(self.reconstruction / 'annotations')
                self._focus_annotation(focus)
                self._annotation_message = self._annotation_text('已保存：', 'Saved: ') + path.parent.name
            except Exception as exc:
                self._annotation_message = self._annotation_text('保存失败：', 'Save failed: ') + str(exc)
        self._update_annotation_label()

    def _load_annotations_dialog(self):
        dialog = gui.FileDialog(gui.FileDialog.OPEN, self._annotation_text('回读标注', 'Load annotations'), self.window.theme)
        dialog.add_filter('.json', 'annotations.json')
        dialog.set_path(str(self.reconstruction))
        dialog.set_on_cancel(self.window.close_dialog)
        def done(path):
            self.window.close_dialog()
            self._load_annotations(path)
        dialog.set_on_done(done)
        self.window.show_dialog(dialog)

    def _load_annotations(self, path):
        # Cancel pending picks before I/O; failed loads must not leave a current reading.
        self.clear_surface_pick()
        self._polygon = []
        try:
            self.annotations.load(path)
            if self.annotations.points:
                self._select_annotation(len(self.annotations.points) - 1)
            elif self.annotations.regions:
                self._select_region(len(self.annotations.regions) - 1)
            self._annotation_message = self._annotation_text('已回读，模型和参考系匹配', 'Loaded; model and reference match')
            success = True
        except Exception as exc:
            self._annotation_message = self._annotation_text('回读失败：', 'Load failed: ') + str(exc)
            success = False
        self._render_annotations()
        self._update_pick_label()
        return success
