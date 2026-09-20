import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch, Mock

import femto.pipeline.reconstruction_sources as sources


class SourceTests(unittest.TestCase):
    def test_gui_mode_switch_keeps_both_sources_and_missing_source_disables(self):
        import tkinter as tk
        import femto.app.femto_control_app as control
        with patch.object(control.FemtoControlApp, '_start_camera'), patch.object(control.FemtoControlApp, '_load_latest_paths'):
            root = tk.Tk()
            root.withdraw()
            try:
                app = control.FemtoControlApp(root)
                app.latest_keyframe_result = Path('fixed')
                app.latest_moving_result = Path('moving')
                app.latest_single_gs = Path('single_model')
                for mode in ('keyframe', 'moving', 'keyframe'):
                    app.scan_mode_var.set(mode)
                    app.latest_result = None
                    app._set_action_state()
                    self.assertEqual(str(app.single_train_button['state']), 'normal')
                    self.assertEqual(str(app.train_3dgs_button['state']), 'normal')
                    self.assertEqual(str(app.single_view_button['state']), 'normal')
                    self.assertEqual(app.latest_single_gs, Path('single_model'))
                    expected = Path('fixed' if mode == 'keyframe' else 'moving')
                    self.assertEqual(str(app.view_button['state']), 'normal')
                    self.assertEqual(app.view_button['text'], app._t(f'app.view_model.{mode}'))
                    with patch.object(control.subprocess, 'Popen') as viewer:
                        app.view_model()
                        self.assertEqual(viewer.call_args.args[0][3], str(expected / 'reconstruction'))
                app.latest_keyframe_result = None
                app._set_action_state()
                self.assertEqual(str(app.single_train_button['state']), 'disabled')
                self.assertEqual(str(app.train_3dgs_button['state']), 'normal')
                self.assertEqual(str(app.view_button['state']), 'disabled')
                with patch.object(control.subprocess, 'Popen') as viewer:
                    app.view_model()
                    viewer.assert_not_called()
                app.scan_mode_display_var.set(app._t('app.scan.moving'))
                app._scan_mode_combo_changed()
                self.assertEqual(str(app.view_button['state']), 'normal')
                self.assertEqual(app.view_button['text'], app._t('app.view_model.moving'))
                self.assertIs(app.three_dgs_profile_combo.master, app.train_3dgs_button.master)
            finally:
                for timer in root.tk.call('after', 'info'):
                    root.after_cancel(timer)
                root.destroy()

    def make(self, root, name, fixed, date, real=True):
        folder = root / name
        (folder / 'reconstruction').mkdir(parents=True)
        (folder / 'reconstruction' / 'reconstruction_report.json').write_text('{}')
        (folder / 'nerfstudio').mkdir()
        (folder / 'nerfstudio' / 'transforms.json').write_text('{}')
        (folder / 'demo_summary.json').write_text(json.dumps(dict(
            data_kind='real_femto' if real else 'synthetic', fixed_camera=fixed,
            frame_count=1 if fixed else 90, created_local=date)))
        return folder.resolve()

    @patch('femto.gsplat.scene_single_gs.resolve_source')
    def test_alternating_and_restart_preserve_each_mode(self, resolve):
        with TemporaryDirectory() as td:
            root = Path(td)
            fixed = self.make(root, 'fixed', True, '2026-09-10 10:00')
            moving = self.make(root, 'moving', False, '2026-09-10 11:00')
            sources.remember(root, fixed)
            sources.remember(root, moving)
            newer = self.make(root, 'new_fixed', True, '2026-09-10 12:00')
            sources.remember(root, newer)
            self.assertEqual(sources.restore(root), dict(keyframe=newer, moving=moving))
            self.assertEqual((root / sources.MARKERS['moving']).read_text(), str(moving))

    @patch('femto.gsplat.scene_single_gs.resolve_source')
    def test_legacy_discovery_rejects_synthetic_and_incomplete(self, resolve):
        with TemporaryDirectory() as td:
            root = Path(td)
            good = self.make(root, 'old', False, '2026-09-08')
            self.make(root, 'synthetic', False, '2026-09-11', real=False)
            incomplete = self.make(root, 'incomplete', True, '2026-09-12')
            (incomplete / 'reconstruction' / 'reconstruction_report.json').unlink()
            (root / sources.MARKERS['moving']).write_text(str(incomplete))
            self.assertEqual(sources.restore(root), dict(keyframe=None, moving=good))

    def test_training_dispatch_ignores_shared_result_and_selector(self):
        import femto.app.femto_control_app as control
        app = control.FemtoControlApp.__new__(control.FemtoControlApp)
        app.processing = app.capturing = app.photo_saving = False
        app.latest_result = Path('unrelated')
        app.latest_keyframe_result = Path('fixed')
        app.latest_moving_result = Path('moving')
        app.three_dgs_profile_var = Mock(get=lambda: next(iter(control.PROFILES)))
        for name in ('_set_status', '_set_detail', '_set_raw_detail', '_set_action_state', 'progress_var', 'event_queue'):
            setattr(app, name, Mock())
        with patch.object(control.threading, 'Thread') as thread:
            app.train_3dgs()
            self.assertEqual(thread.call_args.kwargs['args'][1], Path('moving'))
        app.processing = False
        with patch.object(control.threading, 'Thread') as thread, patch('femto.gsplat.scene_single_gs.launch', return_value=Path('model')) as launch:
            app.train_single_gs()
            thread.call_args.kwargs['target']()
            self.assertEqual(launch.call_args.args[0], Path('fixed'))
        app.processing = False
        app.latest_moving_result = None
        with patch.object(control.threading, 'Thread') as thread:
            app.train_3dgs()
            thread.assert_not_called()

    def test_local_moving_result_without_nerfstudio_is_registered(self):
        with TemporaryDirectory() as td:
            root=Path(td);result=root/'local';(result/'reconstruction').mkdir(parents=True)
            (result/'reconstruction/reconstruction_report.json').write_text('{}')
            (result/'demo_summary.json').write_text(json.dumps(dict(data_kind='real_femto',fixed_camera=False,
                frame_count=5,local_measured=True,reconstruction_source_mode='moving',
                nerfstudio=None,nerfstudio_skipped_reason='local_capture_has_fewer_than_12_views')))
            self.assertEqual(sources.mode_of(result),'moving')
            self.assertEqual(sources.remember(root,result),'moving')
            self.assertEqual((root/sources.MARKERS['moving']).read_text(),str(result.resolve()))
