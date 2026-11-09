"""Task-only acquisition: schema, persistence, navigation and scan admission."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import numpy as np
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

    def test_initial_page_has_region_first_steps_and_no_body_picker(self):
        page = self.page()
        self.assertEqual([button.text() for button in page.workflow.stage_buttons],
                         ['01  采集', '02  涂抹选区', '03  模型与路线', '04  保存'])
        self.assertEqual(page.workflow.current_step, 1)
        self.assertEqual(page.prepare_action.text(), '连接相机')
        for name in ('task_region', 'body_site_button', 'choose_body_site', 'apply_body_selection'):
            self.assertFalse(hasattr(page, name), name)

    def test_ready_frame_can_start_without_a_second_region_selection(self):
        page = self.page()
        from test_segmented_capture import make_frame
        frame=make_frame([500]*19200,1)
        frame.depth_mm=frame.depth_mm.reshape(120,160)
        frame.color_bgr=np.zeros((120,160,3),np.uint8)
        page.host.latest_frame=frame
        page.device_info={'depth_range_m':[.25,2.21]}
        with patch.object(page, 'live_frame_valid', return_value=True), patch.object(page, 'send',return_value=True) as send:
            page.start()
            self.assertTrue(page.session.active)
            self.assertEqual(send.call_args.kwargs['scan_roi']['rect_norm'],[0.,0.,1.,1.])
            self.assertTrue(send.call_args.kwargs['region_first'])
        page.session.active=False

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
