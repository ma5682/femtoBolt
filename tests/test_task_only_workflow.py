"""Task-only acquisition: schema, persistence, navigation and scan admission."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PySide6.QtWidgets import QApplication
from femto.dataset.task_context import new_task, checked_task, save_task, read_task
from femto.app.qt_capture import CapturePage
from femto.app.task_records import verify_record_context


class TaskOnlyWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_new_task_and_disk_roundtrip_have_no_anatomical_fields(self):
        task = new_task()
        self.assertEqual(set(task), {'schema', 'task_id', 'name', 'created_at'})
        self.assertEqual(task['schema'], 'femto-ui-task-v2')
        with TemporaryDirectory() as folder:
            self.assertEqual(save_task(folder, task), task)
            self.assertEqual(read_task(folder), task)
            record = json.loads((Path(folder)/'task_context.json').read_text(encoding='utf-8'))
            self.assertEqual(record['task'], task)
            before = (Path(folder)/'task_context.json').read_bytes()
            self.assertEqual(save_task(folder, new_task()), task)
            self.assertEqual((Path(folder)/'task_context.json').read_bytes(), before)

    def test_removed_fields_and_retired_schema_are_rejected(self):
        task = new_task()
        for invalid in (dict(task, body_region='胸部'), dict(task, body_selection={}),
                        dict(task, schema='femto-ui-task-v1'), dict(task, task_id=''),
                        dict(task, name=' '), dict(task, patient_side='left')):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                checked_task(invalid)

    def test_task_link_cannot_be_copied_to_another_capture(self):
        with TemporaryDirectory() as folder:
            a, b = Path(folder)/'a', Path(folder)/'b'
            a.mkdir(); b.mkdir(); save_task(a, new_task())
            (b/'task_context.json').write_bytes((a/'task_context.json').read_bytes())
            with self.assertRaises(ValueError):
                read_task(b)

    def test_record_identity_does_not_match_by_display_name(self):
        first, other = new_task(), new_task()
        other['name'] = first['name']
        expected = dict(task=first)
        verify_record_context(SimpleNamespace(task_context=first), expected)
        with self.assertRaises(ValueError):
            verify_record_context(SimpleNamespace(task_context=other), expected)

    def page(self):
        page = CapturePage(connect_camera=False)
        self.addCleanup(page.close)
        self.addCleanup(page.shutdown)
        return page

    def test_initial_page_has_five_steps_and_no_body_picker(self):
        page = self.page()
        self.assertEqual([button.text() for button in page.workflow.buttons],
                         ['01  局部采集', '02  检查模型', '03  标记区域', '04  生成路线', '05  核对保存'])
        self.assertEqual(page.workflow.current_step, 1)
        self.assertEqual(page.prepare_action.text(), '连接相机')
        for name in ('task_region', 'body_site_button', 'choose_body_site', 'apply_body_selection'):
            self.assertFalse(hasattr(page, name), name)

    def test_ready_frame_still_requires_explicit_scan_roi_before_start(self):
        page = self.page()
        with patch.object(page, 'live_frame_valid', return_value=True), patch.object(page, 'send') as send:
            page.start()
            send.assert_not_called()
            self.assertFalse(page.session.active)
            self.assertIn('扫描范围', page.roi_message)
        with patch.object(page, 'live_frame_valid', return_value=True), patch.object(page, 'choose_scan_roi') as choose:
            page.prepare_next()
            choose.assert_called_once()

    def test_rescan_preserves_task_identity_and_clears_geometry_without_picker(self):
        page = self.page()
        task = page.task_context.copy()
        page.session.result = Path('completed-capture')
        page.clouds = {'old': 'old-geometry'}
        page.scan_roi_rect = [.1, .1, .9, .9]
        page.rescan_current_region()
        self.assertEqual(page.task_context, task)
        self.assertEqual(page.clouds, {})
        self.assertIsNone(page.scan_roi_rect)
        self.assertIsNone(page.session.result)
        self.assertFalse(page.session.active)


if __name__ == '__main__':
    unittest.main()
