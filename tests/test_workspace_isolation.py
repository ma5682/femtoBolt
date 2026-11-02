"""Checkout identity, runtime selection and mutable-cache isolation."""
import os
import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from femto import paths
from femto.gsplat import gsplat_backend, scene_gsplat_trainer, scene_gsplat_metrics
from femto.ui_i18n import preference_path


class WorkspaceIsolationTests(unittest.TestCase):
    def test_status_uses_owned_weights_and_clean_environment(self):
        root=paths.ASCII_GSPLAT_RUNTIME_ROOT
        payload={'prefix':str(root),'torch_path':str(root/'Lib/site-packages/torch/__init__.py'),
                 'gsplat_path':str(root/'Lib/site-packages/gsplat/__init__.py'),'cuda_ready':True}
        completed=subprocess.CompletedProcess([],0,json.dumps(payload),'')
        with mock.patch.dict(os.environ,{'PYTHONHOME':'foreign','PYTHONPATH':'foreign','TORCH_HOME':'foreign'}), \
                mock.patch.object(Path,'is_file',return_value=True), \
                mock.patch.object(scene_gsplat_trainer,'lpips_backbone_status',return_value={'ready':True}) as weights, \
                mock.patch.object(scene_gsplat_trainer.subprocess,'run',return_value=completed) as run:
            result=scene_gsplat_trainer.runtime_status()
        self.assertTrue(result['ready'])
        self.assertNotIn('PYTHONHOME',run.call_args.kwargs['env'])
        self.assertNotIn('PYTHONPATH',run.call_args.kwargs['env'])
        self.assertIn('-I',run.call_args.args[0])
        self.assertEqual(weights.call_args.kwargs['checkpoint_dir'],paths.WORKSPACE_CACHE_ROOT/'torch/hub/checkpoints')

    def test_status_rejects_foreign_packages_even_if_cuda_is_available(self):
        for payload in ({'prefix':str(paths.ASCII_GSPLAT_RUNTIME_ROOT),'torch_path':'foreign/torch.py',
                         'gsplat_path':'foreign/gsplat.py','cuda_ready':True},[]):
            with mock.patch.object(Path,'is_file',return_value=True), \
                    mock.patch.object(scene_gsplat_trainer,'lpips_backbone_status',return_value={'ready':False}), \
                    mock.patch.object(scene_gsplat_trainer.subprocess,'run',return_value=subprocess.CompletedProcess([],0,json.dumps(payload),'')):
                result=scene_gsplat_trainer.runtime_status()
            self.assertFalse(result['ready']);self.assertTrue(result['error'])

    def test_siblings_have_distinct_stable_ids(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            self.assertEqual(paths.workspace_id(root / "ui"), paths.workspace_id(root / "ui" / "."))
            self.assertNotEqual(paths.workspace_id(root / "ui"), paths.workspace_id(root / "main"))
            self.assertTrue(paths.workspace_id(root / "中文").isascii())

    def test_missing_owned_runtime_never_falls_back_to_shared_runtime(self):
        with mock.patch.object(Path, "is_file", return_value=False):
            selected = scene_gsplat_trainer.runtime_python()
        self.assertEqual(selected.parent.parent, paths.ASCII_GSPLAT_RUNTIME_ROOT)
        self.assertNotIn("FemtoGsplatRuntime", str(selected))

    def test_subprocess_overrides_inherited_shared_caches(self):
        with mock.patch.dict(os.environ, {"TORCH_HOME": "shared", "HF_HOME": "shared",
                "PYTHONHOME": "other-python", "PYTHONPATH": "other-checkout"}), \
                mock.patch.object(gsplat_backend, "_visual_studio_environment", return_value={}):
            environment = gsplat_backend.runtime_environment()
        self.assertNotIn('PYTHONHOME',environment)
        self.assertNotIn('PYTHONPATH',environment)
        self.assertEqual(environment['PYTHONNOUSERSITE'],'1')
        self.assertEqual(environment['VIRTUAL_ENV'],str(paths.ASCII_GSPLAT_RUNTIME_ROOT))
        for name, value in paths.workspace_runtime_environment().items():
            self.assertEqual(environment[name], value)
            self.assertTrue(Path(value).is_relative_to(paths.WORKSPACE_CACHE_ROOT))

    def test_preferences_and_default_weights_are_workspace_local(self):
        self.assertTrue(preference_path().is_relative_to(paths.WORKSPACE_CACHE_ROOT))
        with mock.patch.dict(os.environ, {}, clear=True):
            weight_dir = scene_gsplat_metrics.torch_hub_checkpoint_dir()
        self.assertEqual(weight_dir, paths.WORKSPACE_CACHE_ROOT / "torch" / "hub" / "checkpoints")


if __name__ == "__main__":
    unittest.main()
