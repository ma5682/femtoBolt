"""Runtime diagnostics reject cross-checkout imports and explain probe failures."""
import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from unittest import TestCase,main,mock

from tools import check_workspace_runtime as check


class RuntimeCheckTests(TestCase):
    def test_foreign_prefix_and_package_are_rejected(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);env=root/'env'
            payload={'prefix':str(env),'base_prefix':str(root/'python'),
                     'source':str(root/'femto/paths.py'),
                     'packages':{'numpy':{'path':str(env/'Lib/site-packages/numpy/__init__.py')}}}
            self.assertEqual(check.evaluate(payload,env,root),[])
            payload['prefix']=str(root/'other-env')
            payload['source']=str(root/'other-repo/femto/paths.py')
            payload['packages']['numpy']['path']=str(root/'other-env/numpy/__init__.py')
            self.assertEqual(len(check.evaluate(payload,env,root)),3)
            payload['site_paths']=[str(root/'system/Lib/site-packages')]
            self.assertEqual(len(check.evaluate(payload,env,root)),4)

    def test_missing_environment_has_actionable_result(self):
        with TemporaryDirectory() as folder:
            with mock.patch.object(check.subprocess,'run') as run:
                result=check.check_runtime('capture',Path(folder)/'missing')
            run.assert_not_called();self.assertFalse(result['ready'])
            self.assertIn('setup_environment.cmd',result['errors'][0])

    def test_probe_failure_timeout_and_missing_modules_are_not_success(self):
        with TemporaryDirectory() as folder:
            env=Path(folder)/'env';exe=env/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
            exe.parent.mkdir(parents=True);exe.touch()
            with mock.patch.object(check.subprocess,'run',side_effect=subprocess.TimeoutExpired('probe',2)):
                self.assertIn('超过',check.check_runtime('capture',env,2)['errors'][0])
            payload={'prefix':str(env),'base_prefix':str(env.parent),
                     'source':str(check.ROOT/'femto/paths.py'),'packages':{}}
            output=subprocess.CompletedProcess([],0,'noise\nFEMTO_RUNTIME_RESULT='+json.dumps(payload),'')
            with mock.patch.object(check.subprocess,'run',return_value=output) as run:
                self.assertFalse(check.check_runtime('capture',env)['ready'])
                self.assertIn('-I',run.call_args.args[0])
                self.assertIn('-B',run.call_args.args[0])
            for malformed in ([],None,dict(payload,packages={'numpy':None}),dict(payload,site_paths='wrong')):
                output=subprocess.CompletedProcess([],0,'FEMTO_RUNTIME_RESULT='+json.dumps(malformed),'')
                with mock.patch.object(check.subprocess,'run',return_value=output):
                    result=check.check_runtime('capture',env)
                self.assertFalse(result['ready'])
                self.assertIn('格式无效',result['errors'][0])

    def test_capture_resolver_ignores_inherited_python_location(self):
        if os.name!='nt' or not (check.ROOT/'.venv/Scripts/python.exe').is_file():
            self.skipTest('Windows workspace interpreter required')
        environment=dict(os.environ,PYTHONHOME='X:\\nonexistent-python',PYTHONPATH='X:\\other-checkout')
        result=subprocess.run(['cmd.exe','/d','/c',str(check.ROOT/'_resolve_femto_python.cmd')],
            env=environment,cwd=check.ROOT,capture_output=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stderr.decode(errors='replace'))


if __name__=='__main__':main()
