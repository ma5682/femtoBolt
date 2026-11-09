"""History migration keeps source gates, portable records and evidence intact."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import zipfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMessageBox, QTreeWidgetItem

from femto.app.qt_history import HistoryDialog, AdvancedToolsDialog, record_target, reference_image
from femto.planning.export_history import register_project


class QtHistoryTests(unittest.TestCase):
    def test_inline_view_uses_clicked_row_instead_of_previous_selection(self):
        from PySide6.QtTest import QTest
        tree=self.dialog.session_tree
        first=tree.topLevelItem(0)
        second=QTreeWidgetItem(['2026-10-22','原始采集','1','second'])
        second.setData(0,Qt.UserRole,dict(first.data(0,Qt.UserRole),name='second'))
        tree.addTopLevelItem(second);tree.setCurrentItem(first)
        rectangle=tree.visualRect(tree.indexFromItem(second,4))
        with patch.object(self.dialog,'open_selected') as opened:
            QTest.mouseClick(tree.viewport(),Qt.LeftButton,pos=rectangle.center())
        self.assertIs(tree.currentItem(),second)
        opened.assert_called_once()

    def test_permanent_delete_is_on_removed_row_and_needs_only_one_confirmation(self):
        self.assertTrue(self.dialog.purge_button.isHidden())
        tree=self.dialog.session_tree
        menu=self.dialog.row_menu(tree,tree.currentItem())
        self.assertNotIn('永久删除',[a.text() for a in menu.actions()])
        from femto.pipeline.local_history import set_records_deleted
        set_records_deleted([self.dialog.sessions[0]],self.dialog.datasets,self.dialog.outputs,True)
        self.dialog.removed.setChecked(True)
        self.wait(lambda:not self.dialog.busy and self.dialog.current() is not None)
        menu=self.dialog.row_menu(tree,tree.currentItem())
        delete=next(a for a in menu.actions() if a.text()=='永久删除')
        self.assertIn('恢复记录',[a.text() for a in menu.actions()])
        with patch('femto.app.qt_history.QMessageBox.question',return_value=QMessageBox.No) as confirm:
            delete.trigger()
        confirm.assert_called_once()
        self.assertTrue(Path(self.dialog.current()['key']).exists())
        with patch('femto.app.qt_history.QMessageBox.question',return_value=QMessageBox.Yes) as confirm, \
             patch('femto.pipeline.history_purge.execute_purge') as execute:
            delete.trigger();self.wait(lambda:not self.dialog.busy)
        confirm.assert_called_once();execute.assert_called_once()
        self.assertEqual(execute.call_args.args[1],Path(self.dialog.current()['key']))

    def test_visible_data_maintenance_entry_uses_this_workspace(self):
        from PySide6.QtTest import QTest
        self.assertTrue(self.dialog.maintenance_button.isVisible())
        with patch('subprocess.Popen') as launch:
            QTest.mouseClick(self.dialog.maintenance_button,Qt.LeftButton)
        args=launch.call_args.args[0]
        self.assertEqual(args[-2:],['--workspace',str(self.root.resolve())])
        self.assertIn('femto.app.qt_data_maintenance',args)

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        from tests import test_local_history as fixture
        self.fixture = fixture.HistoryTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.rec = self.fixture.result / "reconstruction"
        self.dialog = HistoryDialog(root=self.root)
        self.dialog.show()
        self.addCleanup(self.close_dialog)
        self.wait(lambda: bool(self.dialog.sessions) and not self.dialog.busy)

    def close_dialog(self):
        self.dialog.close()
        self.wait(lambda: not self.dialog.runner.busy)
        self.dialog.close()
        self.app.processEvents()

    def wait(self, condition, timeout=10):
        deadline = time.monotonic() + timeout
        while not condition() and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(.005)
        self.app.processEvents()
        self.assertTrue(condition(), "Qt history operation did not finish")

    def snapshot(self):
        # The metadata index is writable; every original asset/marker remains
        # part of the byte-for-byte evidence assertion.
        metadata = self.root / 'outputs' / 'history_metadata'
        return {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob("*")
                if p.is_file() and not p.is_relative_to(metadata)}

    def test_reconstruction_focus_uses_its_project_root(self):
        focused = HistoryDialog(root=self.rec)
        try:
            self.assertEqual(focused.root, self.root.resolve())
            self.assertEqual(focused.datasets, self.root.resolve() / "datasets")
            self.assertEqual(focused.outputs, self.root.resolve() / "outputs")
        finally:
            focused.close()
            self.wait(lambda: not focused.runner.busy)

    def test_session_open_and_refresh_are_read_only(self):
        before = self.snapshot()
        self.assertEqual(len(self.dialog.sessions), 1)
        self.assertTrue(self.dialog.open_button.isEnabled())
        opened = []
        self.dialog.planning_requested.connect(opened.append)
        self.dialog.open_selected()
        self.assertEqual(opened, [str(self.rec)])
        self.dialog.refresh()
        self.wait(lambda: not self.dialog.busy)
        self.assertEqual(before, self.snapshot())
        self.dialog.search.setText("不存在的记录")
        self.assertFalse(self.dialog.open_button.isEnabled())

    def test_exports_tab_follows_selected_capture_record(self):
        export = dict(time="2026-09-26 04:30:00", kind="区域工程 / 方案",
                      name="默认工作区", status="可打开继续规划",
                      folder=self.rec, source=self.rec / "region_project_history/index.json",
                      project=self.rec / "region.zip", file_status="工程资产指纹一致")
        with patch("femto.app.qt_history.scan_capture_exports", return_value=[export]) as scan:
            self.dialog.tabs.setCurrentIndex(1)
            self.wait(lambda: not self.dialog.busy and self.dialog.tabs.currentIndex() == 1
                      and self.dialog.records)
            scan.assert_called_once_with(self.rec.resolve(), check=unittest.mock.ANY)
            self.assertEqual(self.dialog.reconstruction, self.rec.resolve())
            self.assertEqual(self.dialog.current(), export)
            self.assertEqual(self.dialog.record_tree.currentItem().text(2), "默认工作区")

    def test_switching_capture_clears_previous_exports_while_loading(self):
        first = dict(time="2026-09-26 04:30:00", kind="区域工程 / 方案",
                     name="第一条工作区", status="可打开继续规划",
                     folder=self.rec, source=self.rec / "region.json",
                     project=self.rec / "region.zip", file_status="工程资产指纹一致")
        self.dialog.reconstruction = self.rec.resolve()
        self.dialog.records = [first]
        first_item = QTreeWidgetItem([first["time"], first["kind"], first["name"], first["status"]])
        first_item.setData(0, Qt.ItemDataRole.UserRole, first)
        self.dialog.record_tree.addTopLevelItem(first_item)
        self.dialog.record_tree.setCurrentItem(first_item)

        second_reconstruction = self.root / "outputs" / "complete_demo_b" / "reconstruction"
        second_reconstruction.mkdir(parents=True)
        second_session = dict(self.dialog.sessions[0],
                              key=str(second_reconstruction.parent.resolve()),
                              name="complete_demo_b",
                              result=str(second_reconstruction.parent.resolve()))
        second_item = QTreeWidgetItem(["2026-09-25 04:00:00", "模型已生成", "1", second_session["name"]])
        second_item.setData(0, Qt.ItemDataRole.UserRole, second_session)
        self.dialog.tabs.setCurrentIndex(0)
        self.dialog.session_tree.addTopLevelItem(second_item)
        self.dialog.session_tree.setCurrentItem(second_item)
        second_export = dict(first, name="第二条工作区", folder=second_reconstruction,
                             source=second_reconstruction / "region.json",
                             project=second_reconstruction / "region.zip")

        def slow_scan(*_args, **_kwargs):
            time.sleep(.15)
            return [second_export]

        with patch("femto.app.qt_history.scan_capture_exports", side_effect=slow_scan):
            self.dialog.tabs.setCurrentIndex(1)
            self.assertEqual(self.dialog.record_tree.topLevelItemCount(), 0)
            self.wait(lambda: not self.dialog.busy and self.dialog.records)
        self.assertEqual(self.dialog.record_tree.currentItem().text(2), "第二条工作区")

    def test_changed_source_is_blocked_when_opened(self):
        opened = []
        self.dialog.planning_requested.connect(opened.append)
        (self.fixture.source / "rgb/0").write_bytes(b"changed evidence")
        self.dialog.open_selected()
        self.assertEqual(opened, [])
        self.assertFalse(self.dialog.open_button.isEnabled())
        self.assertIn("变化", self.dialog.status.text())

    def test_filter_selects_visible_match_and_refresh_keeps_it(self):
        row = dict(self.dialog.sessions[0], key='second', name='唯一匹配名称')
        with patch('femto.app.qt_history.scan_history', return_value=[self.dialog.sessions[0], row]):
            self.dialog.refresh()
            self.wait(lambda: not self.dialog.busy)
            self.dialog.search.setText('唯一匹配名称')
            self.assertEqual(self.dialog.current()['key'], 'second')
            self.dialog.refresh()
            self.wait(lambda: not self.dialog.busy)
            self.assertEqual(self.dialog.current()['key'], 'second')

    def test_plan_name_search_survives_refresh_with_capture_context(self):
        session = self.dialog.current()
        row = dict(time='2026-10-20', kind='区域工程 / 方案', name='仅方案匹配名称',
                   status='可打开', folder=self.rec, project=self.rec/'region.zip')
        with patch('femto.app.qt_history.scan_capture_exports', return_value=[row]) as scan:
            self.dialog.tabs.setCurrentIndex(1)
            self.wait(lambda: not self.dialog.busy)
            self.dialog.search.setText(row['name'])
            self.assertEqual(self.dialog.current(), row)
            self.assertEqual(self.dialog._selected_session(), session)
            self.dialog.refresh()
            self.wait(lambda: not self.dialog.busy)
            self.assertEqual(self.dialog.current(), row)
            self.assertEqual(self.dialog.records, [row])
            self.assertEqual(self.dialog._selected_session(), session)
            self.assertEqual(scan.call_count, 2)

    def test_search_queries_and_selection_follow_each_tab(self):
        session = self.dialog.current()
        row = dict(time='2026-10-20', kind='区域工程 / 方案', name='独立方案搜索',
                   status='可打开', folder=self.rec, project=self.rec/'region.zip')
        self.dialog.search.setText(session['name'])
        with patch('femto.app.qt_history.scan_capture_exports', return_value=[row]) as scan:
            self.dialog.tabs.setCurrentIndex(1)
            self.wait(lambda: not self.dialog.busy)
            self.assertEqual(self.dialog.search.text(), '')
            self.assertEqual(self.dialog.current(), row)
            self.dialog.search.setText(row['name'])
            self.dialog.tabs.setCurrentIndex(0)
            self.assertEqual(self.dialog.search.text(), session['name'])
            self.assertEqual(self.dialog.current(), session)
            self.dialog.tabs.setCurrentIndex(1)
            self.assertEqual(self.dialog.search.text(), row['name'])
            self.assertEqual(self.dialog.current(), row)
            self.assertEqual(scan.call_count, 1)

    def _second_capture(self):
        reconstruction = self.root / 'outputs' / 'complete_demo_second' / 'reconstruction'
        reconstruction.mkdir(parents=True)
        row = dict(self.dialog.sessions[0], key=str(reconstruction.parent.resolve()),
                   result=str(reconstruction.parent.resolve()), name='第二次采集')
        with patch('femto.app.qt_history.scan_history', return_value=[self.dialog.sessions[0], row]):
            self.dialog.refresh()
            self.wait(lambda: not self.dialog.busy)
        return reconstruction, row

    def test_capture_switch_during_export_read_discards_stale_rows_without_tab_change(self):
        second_rec, second_session = self._second_capture()
        first = dict(time='2026-10-20', kind='区域工程 / 方案', name='旧方案',
                     status='可打开', folder=self.rec, project=self.rec/'region.zip')
        second = dict(first, name='新方案', folder=second_rec, project=second_rec/'region.zip')
        release = threading.Event()
        entered = threading.Event()
        def scan(path, **kwargs):
            if path == self.rec.resolve():
                entered.set()
                release.wait(5)
                return [first]
            return [second]
        with patch('femto.app.qt_history.scan_capture_exports', side_effect=scan) as scanned:
            try:
                self.dialog.tabs.setCurrentIndex(1)
                self.wait(entered.is_set)
                self.dialog.search.setText('新方案')
                self.dialog.tabs.setCurrentIndex(0)
                self.dialog.search.setText('第二次采集')
                self.assertEqual(self.dialog.current(), second_session)
                release.set()
                self.wait(lambda: not self.dialog.busy)
                self.assertEqual(self.dialog.tabs.currentIndex(), 0)
                self.assertEqual(self.dialog.current(), second_session)
                self.assertEqual(self.dialog.records, [])
                self.assertEqual(scanned.call_count, 1)
                self.dialog.tabs.setCurrentIndex(1)
                self.wait(lambda: not self.dialog.busy)
                self.assertEqual(self.dialog.search.text(), '新方案')
                self.assertEqual(self.dialog.current(), second)
                self.assertEqual(self.dialog.reconstruction, second_rec.resolve())
                self.assertEqual(scanned.call_count, 2)
            finally:
                release.set()

    def test_capture_switch_while_reading_queues_current_capture_on_exports_tab(self):
        second_rec, second_session = self._second_capture()
        first = dict(time='2026-10-20', kind='区域工程 / 方案', name='旧方案',
                     status='可打开', folder=self.rec, project=self.rec/'region.zip')
        second = dict(first, name='新方案', folder=second_rec, project=second_rec/'region.zip')
        release = threading.Event()
        entered = threading.Event()
        def scan(path, **kwargs):
            if path == self.rec.resolve():
                entered.set()
                release.wait(5)
                return [first]
            return [second]
        with patch('femto.app.qt_history.scan_capture_exports', side_effect=scan) as scanned:
            try:
                self.dialog.tabs.setCurrentIndex(1)
                self.wait(entered.is_set)
                self.dialog.search.setText('新方案')
                self.dialog.tabs.setCurrentIndex(0)
                self.dialog.search.setText('第二次采集')
                self.assertEqual(self.dialog.current(), second_session)
                self.dialog.tabs.setCurrentIndex(1)
                self.assertEqual(self.dialog.records, [])
                release.set()
                self.wait(lambda: not self.dialog.busy and bool(self.dialog.records))
                self.assertEqual(self.dialog.tabs.currentIndex(), 1)
                self.assertEqual(self.dialog.current(), second)
                self.assertEqual(self.dialog.records, [second])
                self.assertEqual(self.dialog._selected_session(), second_session)
                self.assertEqual(self.dialog.reconstruction, second_rec.resolve())
                self.assertEqual(scanned.call_count, 2)
            finally:
                release.set()

    def test_export_completion_does_not_switch_back_to_exports(self):
        release = threading.Event()
        def scan(*args, **kwargs):
            release.wait(5)
            return []
        with patch('femto.app.qt_history.scan_capture_exports', side_effect=scan):
            try:
                self.dialog.tabs.setCurrentIndex(1)
                self.assertTrue(self.dialog.busy)
                self.dialog.tabs.setCurrentIndex(0)
                release.set()
                self.wait(lambda: not self.dialog.busy)
                self.assertEqual(self.dialog.tabs.currentIndex(), 0)
            finally:
                release.set()

    def test_export_refresh_failure_preserves_rows(self):
        row = dict(time='2026-10-20', kind='区域工程 / 方案', name='保留记录',
                   status='可打开', folder=self.rec, project=self.rec/'region.zip')
        with patch('femto.app.qt_history.scan_capture_exports', return_value=[row]):
            self.dialog.tabs.setCurrentIndex(1)
            self.wait(lambda: not self.dialog.busy)
        with patch('femto.app.qt_history.scan_capture_exports', side_effect=ValueError('索引损坏')):
            self.dialog.refresh()
            self.wait(lambda: not self.dialog.busy)
        self.assertEqual(self.dialog.current(), row)
        self.assertIn('索引损坏', self.dialog.status.text())

    def test_source_details_are_optional_and_cancel_is_hidden_when_idle(self):
        self.assertNotIn('模型：', self.dialog.details.text())
        self.assertTrue(self.dialog.cancel_button.isHidden())
        self.dialog.source_toggle.click()
        self.assertIn('模型：', self.dialog.details.text())
        self.dialog.source_toggle.click()
        self.assertNotIn('模型：', self.dialog.details.text())

    def test_export_refresh_preserves_selected_version(self):
        first = dict(time='2026-10-20', kind='区域工程 / 方案', name='第一版',
                     status='可打开', folder=self.rec, project=self.rec/'first.zip')
        second = dict(first, name='第二版', project=self.rec/'second.zip')
        with patch('femto.app.qt_history.scan_capture_exports', return_value=[first, second]):
            self.dialog.tabs.setCurrentIndex(1)
            self.wait(lambda: not self.dialog.busy)
            self.dialog.record_tree.setCurrentItem(self.dialog.record_tree.topLevelItem(1))
            self.dialog.refresh()
            self.wait(lambda: not self.dialog.busy)
            self.assertEqual(self.dialog.current(), second)

    def test_invalid_record_target_is_fail_closed(self):
        self.assertIsNone(record_target({"kind": "区域工程 / 方案", "project": object()}))
        self.assertIsNone(record_target({"kind": "区域＋路径", "source": "\\0"}))

    def test_portable_and_native_records_are_visible_with_corrupt_gate(self):
        portable = self.root / "saved.region.zip"
        assets = {name: b"fixture" for name in ("scene_tsdf_mesh_m.ply", "rgb.png", "depth.npy", "image_valid.npy")}
        metadata = dict(schema="femto-region-project-v1", project_id="p", revision_id="v",
            region=dict(annotation_id="area", name="已保存区域"),
            geometry_sha256="geometry", origin_reference={"origin": "W0"},
            files={name: hashlib.sha256(data).hexdigest() for name, data in assets.items()})
        with zipfile.ZipFile(portable, "w") as archive:
            archive.writestr("project.json", json.dumps(metadata))
            for name, data in assets.items():
                archive.writestr(name, data)
        register_project(self.rec, portable)
        native = self.rec / "surface_paths" / "trajectory_bad" / "trajectory.json"
        native.parent.mkdir(parents=True)
        native.write_text(json.dumps(dict(schema="femto-surface-trajectory-v1", source={"region_id": "broken"})))
        before = self.snapshot()
        self.dialog.show_exports()
        self.wait(lambda: not self.dialog.busy and self.dialog.tabs.currentIndex() == 1)
        self.assertEqual(len(self.dialog.records), 2)
        rows = self.dialog.records
        project = next(row for row in rows if row["kind"] == "区域工程 / 方案")
        self.assertEqual(record_target(project), portable)
        bad = next(row for row in rows if row["source"] == native)
        self.assertEqual(bad["file_status"], "损坏")
        opened = []
        self.dialog.planning_requested.connect(opened.append)
        for index in range(self.dialog.record_tree.topLevelItemCount()):
            item = self.dialog.record_tree.topLevelItem(index)
            if item.data(0, Qt.ItemDataRole.UserRole)["source"] == native:
                self.dialog.record_tree.setCurrentItem(item)
                self.dialog.open_selected()
                self.assertFalse(self.dialog.open_button.isEnabled())
            else:
                self.dialog.record_tree.setCurrentItem(item)
                self.dialog.open_selected()
        self.assertEqual(opened, [str(portable)])
        self.assertEqual(before, self.snapshot())

    def test_failed_refresh_keeps_current_rows(self):
        previous = list(self.dialog.sessions)
        with patch("femto.app.qt_history.scan_history", side_effect=ValueError("索引损坏")):
            self.dialog.refresh()
            self.wait(lambda: not self.dialog.busy)
        self.assertEqual(self.dialog.sessions, previous)
        self.assertIn("当前列表保留", self.dialog.status.text())
        self.assertIn("索引损坏", self.dialog.status.text())

    def test_cancel_catalogue_waits_preserves_rows_and_allows_retry(self):
        previous = list(self.dialog.sessions)
        release = threading.Event()
        entered = threading.Event()

        def slow_scan(*args, **kwargs):
            entered.set()
            release.wait(5)
            return []

        with patch("femto.app.qt_history.scan_history", side_effect=slow_scan):
            try:
                self.dialog.refresh()
                self.wait(entered.is_set)
                self.assertIn("等待", self.dialog.open_button.toolTip())
                self.dialog.cancel_button.click()
                self.assertFalse(self.dialog.cancel_button.isEnabled())
                self.assertEqual(self.dialog.cancel_button.text(), "正在取消…")
                self.assertTrue(self.dialog.busy)
                self.assertFalse(self.dialog.refresh_button.isEnabled())
            finally:
                release.set()
                self.wait(lambda: not self.dialog.busy)
        self.assertEqual(self.dialog.sessions, previous)
        self.assertIn("取消", self.dialog.status.text())
        self.dialog.refresh()
        self.wait(lambda: not self.dialog.busy)
        self.assertEqual(self.dialog.sessions, previous)
        self.assertTrue(self.dialog.open_button.isEnabled())

    def test_open_hint_distinguishes_empty_selection_from_invalid_record(self):
        self.dialog.search.setText("不存在的记录")
        self.assertIn("选择", self.dialog.open_button.toolTip())
        self.assertNotIn("损坏", self.dialog.open_button.toolTip())

    def test_selected_record_cancel_emits_once_and_waits(self):
        requests = []
        self.dialog.cancel_open_requested.connect(lambda: requests.append(True))
        self.dialog.set_opening(True)
        self.dialog.cancel_read()
        self.dialog.cancel_read()
        self.assertEqual(requests, [True])
        self.assertEqual(self.dialog.cancel_button.text(), "正在取消…")
        self.assertFalse(self.dialog.cancel_button.isEnabled())
        self.dialog.set_opening(False)
        self.assertTrue(self.dialog.open_button.isEnabled())

    def test_hide_and_undo_only_change_catalogue(self):
        before = self.snapshot()
        with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes):
            self.dialog.hide_selected()
        self.wait(lambda: not self.dialog.busy)
        self.assertEqual(self.dialog.sessions, [])
        after = self.snapshot()
        self.assertEqual(before, {p: data for p, data in after.items() if p != Path("outputs/local_history_deleted.json")})
        self.dialog.undo_hide()
        self.wait(lambda: not self.dialog.busy)
        self.assertEqual(len(self.dialog.sessions), 1)

    def test_reference_image_does_not_escape_dataset(self):
        outside = self.root / "outside.png"
        outside.write_bytes(b"fixture")
        (self.fixture.source / "frames.jsonl").write_text(json.dumps({"rgb": str(outside)}), encoding="utf-8")
        self.assertIsNone(reference_image(self.dialog.sessions[0]))

    def test_removed_record_can_be_restored_after_reopening_window(self):
        before = self.snapshot()
        with patch('femto.app.qt_history.QMessageBox.question', return_value=QMessageBox.StandardButton.Yes):
            self.dialog.hide_selected()
            self.wait(lambda: not self.dialog.busy)
        self.close_dialog()
        self.dialog = HistoryDialog(root=self.root)
        self.dialog.show()
        self.wait(lambda: self.dialog.status.text().startswith('共') and not self.dialog.busy)
        self.assertEqual(self.dialog.sessions, [])
        self.assertEqual(self.dialog.deleted_stack, [])
        self.dialog.removed.setChecked(True)
        self.wait(lambda: not self.dialog.busy)
        self.assertEqual(len(self.dialog.sessions), 1)
        self.assertEqual(self.dialog.hide_button.text(), '恢复记录')
        self.dialog.hide_selected()
        self.wait(lambda: not self.dialog.busy)
        self.assertEqual(self.dialog.sessions, [])
        self.dialog.removed.setChecked(False)
        self.wait(lambda: not self.dialog.busy)
        self.assertEqual(len(self.dialog.sessions), 1)
        self.assertEqual(before, self.snapshot())

    def test_database_write_failure_preserves_ui_and_previous_index(self):
        before = self.dialog.store.query('captures')['rows']
        with patch.object(self.dialog.store, 'replace', side_effect=OSError('磁盘不可写')):
            self.dialog.refresh()
            self.wait(lambda: not self.dialog.busy)
        self.assertEqual(self.dialog.sessions, before)
        self.assertEqual(self.dialog.store.query('captures')['rows'], before)
        self.assertIn('磁盘不可写', self.dialog.status.text())

    def test_search_uses_full_version_identifier(self):
        row = dict(time='2026-10-20', kind='区域工程 / 方案', name='已保存',
                   status='可打开', folder=self.rec, project=self.rec/'region.zip',
                   project_id='project', revision_id='abcdefgh-match-at-end', parent_revision_id='parent')
        with patch('femto.app.qt_history.scan_capture_exports', return_value=[row]):
            self.dialog.tabs.setCurrentIndex(1)
            self.wait(lambda: not self.dialog.busy)
        self.dialog.search.setText('match-at-end')
        self.assertEqual(self.dialog.current(), row)
        self.assertEqual(self.dialog.store.versions('project')[0]['parent_revision_id'], 'parent')

    def test_advanced_diagnostics_read_only(self):
        before = self.snapshot()
        dialog = AdvancedToolsDialog(root=self.root)
        try:
            self.assertIn("PySide6", dialog.report.toPlainText())
            self.assertEqual(before, self.snapshot())
        finally:
            dialog.close()


if __name__ == "__main__":
    unittest.main()
