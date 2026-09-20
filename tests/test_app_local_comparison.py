import json
from pathlib import Path
import tempfile
import threading
import time
import tkinter as tk
import unittest
from unittest import mock

from femto.app.app_local_comparison import ComparisonWindow, comparison_source, progress_text


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


class ComparisonControlsTests(unittest.TestCase):
    def test_ancestry_uses_original_recorded_input_and_rejects_cross_session(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original, refined, filtered, result = [root / s for s in ('original', 'refined', 'filtered', 'result')]
            local = dict(measured_only=True, session_id='same', reference_frame_id=0)
            write(original / 'manifest.json', dict(local_capture=local, pose=dict(source='online_rgbd_research_grade')))
            write(refined / 'manifest.json', dict(local_capture=local, refinement_source_dataset=str(original)))
            write(root / 'temporal.json', dict(source_dataset=str(refined)))
            write(filtered / 'manifest.json', dict(local_capture=dict(local, temporal_support_report=str(root / 'temporal.json'))))
            write(result / 'reconstruction/reconstruction_report.json', dict(source_dataset=str(filtered)))
            write(result / 'reconstruction/local_geometry.json', dict(session_id='same', reference_frame_id=0))
            self.assertEqual(comparison_source(result), original)
            write(original / 'manifest.json', dict(local_capture=dict(local, session_id='other'), pose=dict(source='online_rgbd_research_grade')))
            with self.assertRaisesRegex(ValueError, 'crosses'):
                comparison_source(result)
            write(result / 'publication.json', dict(status='cancelled'))
            with self.assertRaisesRegex(ValueError, 'not published'):
                comparison_source(result)

    def window(self, language='zh'):
        root = tk.Tk()
        root.withdraw()
        window = ComparisonWindow(root, Path('model'), Path(tempfile.gettempdir()) / 'femto_comparison_ui_tests', language)
        self.addCleanup(lambda: root.destroy() if root.winfo_exists() else None)
        return root, window

    def test_close_parent_cancels_worker_and_late_event_stays_in_its_dialog(self):
        root, window = self.window()
        started, stopped = threading.Event(), threading.Event()
        def compare(source, output, *, cancelled, progress):
            started.set()
            progress(dict(state='running', stages=[]))
            deadline = time.monotonic() + 3
            while not cancelled() and time.monotonic() < deadline:
                time.sleep(.01)
            stopped.set()
            return dict(state='cancelled', stages=[])
        with mock.patch('femto.app.app_local_comparison.comparison_source', return_value=Path('fixture')), \
             mock.patch('femto.pipeline.local_comparison.run_comparison', side_effect=compare):
            window.start()
            self.assertTrue(started.wait(2))
            poll_timer = window.timer
            self.assertIn(poll_timer, root.tk.call('after', 'info'))
            root.destroy()
            self.assertTrue(window.closed)
            self.assertTrue(window.cancel_event.is_set())
            self.assertIsNone(window.timer)
            self.assertNotIn(poll_timer, root.tk.call('after', 'info'))
            self.assertTrue(stopped.wait(2))
            window.thread.join(2)
        # Remove the now-invalid Tcl root cleanup; no worker reaches Tk.
        self._cleanups.clear()

    def test_progress_is_not_terminal_until_report_writer_returns_and_stale_generation_ignored(self):
        root, window = self.window()
        window.running = True
        window.generation = 2
        window._put(1, dict(state='complete'), terminal=True)
        window.poll()
        self.assertTrue(window.running)
        self.assertIsNone(window.report)
        window._put(2, dict(state='complete'), terminal=False)
        window.poll()
        self.assertTrue(window.running)
        window._put(2, dict(state='complete'), terminal=True)
        window.poll()
        self.assertFalse(window.running)
        window.close()

    def test_repeated_poll_keeps_one_timer_and_close_preserves_other_callbacks(self):
        root, window = self.window()
        unrelated = root.after(60_000, lambda: None)
        try:
            for _ in range(4):
                previous = window.timer
                self.assertIn(previous, root.tk.call('after', 'info'))
                window.poll()
                pending = set(root.tk.call('after', 'info'))
                self.assertNotIn(previous, pending)
                self.assertIsNotNone(window.timer)
                self.assertEqual(pending, {unrelated, window.timer})
            window.close()
            self.assertIsNone(window.timer)
            self.assertEqual(set(root.tk.call('after', 'info')), {unrelated})
            # Closing and polling an already closed dialog remain idempotent.
            window.close()
            window.poll()
            self.assertEqual(set(root.tk.call('after', 'info')), {unrelated})
        finally:
            root.after_cancel(unrelated)
            window.close()

    def test_small_window_controls_available_in_both_languages(self):
        for language in ('zh', 'en'):
            with self.subTest(language=language):
                root = tk.Tk()
                root.withdraw()
                window = ComparisonWindow(root, Path('model'), Path('outputs'), language)
                try:
                    window.window.geometry('600x400')
                    root.update()
                    for button in window.buttons.values():
                        self.assertGreaterEqual(button.winfo_height(), button.winfo_reqheight())
                        self.assertLessEqual(button.winfo_rootx()+button.winfo_width(), window.window.winfo_rootx()+window.window.winfo_width())
                        self.assertLessEqual(button.winfo_rooty()+button.winfo_height(), window.window.winfo_rooty()+window.window.winfo_height())
                finally:
                    window.close()
                    root.destroy()

    def test_status_reports_reused_geometry_and_missing_evidence_without_fabricated_zero(self):
        report = dict(state='complete', variants=[dict(name='B', optimization=dict(status='not_adopted'), same_as='A')])
        self.assertIn('几何与 A', progress_text(report))
        self.assertIn('Not recorded', progress_text(report, 'en'))
        self.assertNotIn('0.0', progress_text(report))


if __name__ == '__main__':
    unittest.main()
