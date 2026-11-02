"""Read-only checkout runtime diagnostics. No camera, UI or training startup."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from femto.paths import ASCII_GSPLAT_RUNTIME_ROOT, workspace_runtime_environment

PACKAGES = {
    'capture': {'numpy':'numpy','open3d':'open3d','cv2':'opencv-contrib-python',
                'PySide6':'PySide6','vtkmodules':'vtk','pyorbbecsdk':'pyorbbecsdk2'},
    'gsplat': {'torch':'torch','gsplat':'gsplat','spz':'spz','PySide6':'PySide6','vtkmodules':'vtk'},
}
PROBE = r'''
import sys,json,importlib,importlib.metadata as metadata
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import femto.paths
result={'prefix':sys.prefix,'base_prefix':sys.base_prefix,'source':femto.paths.__file__,'packages':{}}
for name,distribution in json.loads(sys.argv[2]).items():
    try:
        module=importlib.import_module(name)
        result['packages'][name]={'path':module.__file__,'version':metadata.version(distribution)}
    except Exception as error:
        result['packages'][name]={'error':type(error).__name__+': '+str(error)}
result['site_paths']=[p for p in sys.path if 'site-packages' in [part.lower() for part in Path(p).parts]]
print('FEMTO_RUNTIME_RESULT='+json.dumps(result,ensure_ascii=True))
'''


def evaluate(payload, expected, root=ROOT):
    """Validate reported ownership; lexical target must not resolve elsewhere."""
    if (not isinstance(payload,dict) or
            any(not isinstance(payload.get(key),str) or not payload[key] for key in ('prefix','base_prefix','source')) or
            not isinstance(payload.get('packages'),dict) or
            any(not isinstance(record,dict) for record in payload['packages'].values()) or
            not isinstance(payload.get('site_paths',[]),list) or
            any(not isinstance(path,str) for path in payload.get('site_paths',[]))):
        raise ValueError('环境探测结果格式无效')
    expected=Path(expected).absolute()
    errors=[]
    if expected.resolve()!=expected:
        errors.append('环境目录通过联接或符号链接指向其他位置')
    if Path(payload.get('prefix','')).resolve()!=expected:
        errors.append('解释器 sys.prefix 不属于目标环境')
    if payload.get('prefix')==payload.get('base_prefix'):
        errors.append('解释器未运行于独立虚拟环境')
    if Path(payload.get('source','')).resolve()!=(Path(root)/'femto/paths.py').resolve():
        errors.append('femto 源码来自其他工作区')
    if any(not Path(path).resolve().is_relative_to(expected) for path in payload.get('site_paths',[])):
        errors.append('搜索路径混入其他环境的 site-packages')
    for name,record in payload.get('packages',{}).items():
        if record.get('error'):
            errors.append(name+'：'+record['error'])
        elif not record.get('path') or not Path(record['path']).resolve().is_relative_to(expected):
            errors.append(name+' 加载自目标环境之外')
    return errors


def check_runtime(name, expected, timeout=90):
    expected=Path(expected).absolute()
    executable=expected/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    report={'name':name,'expected':str(expected),'ready':False,'errors':[]}
    if not executable.is_file():
        report['errors']=['解释器不存在，请运行 '+('setup_environment.cmd' if name=='capture' else 'setup_direct_gsplat_environment.cmd')]
        return report
    if expected.resolve()!=expected:
        report['errors']=['环境目录指向其他位置；未启动探测子进程']
        return report
    environment=dict(os.environ)
    environment.pop('PYTHONHOME',None);environment.pop('PYTHONPATH',None)
    environment.update(workspace_runtime_environment())
    try:
        result=subprocess.run([str(executable),'-I','-B','-c',PROBE,str(ROOT),json.dumps(PACKAGES[name])],
            cwd=ROOT,env=environment,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=timeout)
        records=[line.split('=',1)[1] for line in result.stdout.splitlines() if line.startswith('FEMTO_RUNTIME_RESULT=')]
        if result.returncode or not records:
            report['errors']=[f'环境探测进程未正常完成（退出码 {result.returncode}）']
            return report
        payload=json.loads(records[-1]);report['observed']=payload
        report['errors']=evaluate(payload,expected)
        missing=PACKAGES[name].keys()-payload.get('packages',{}).keys()
        if missing:report['errors'].append('探测结果缺少模块：'+', '.join(sorted(missing)))
        report['ready']=not report['errors']
    except subprocess.TimeoutExpired:
        report['errors']=[f'环境检查超过 {timeout} 秒；进程已停止']
    except (OSError,ValueError,TypeError) as error:
        report['errors']=['环境检查失败：'+str(error)]
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description='只读检查本工作区解释器、源码及核心依赖归属，不启动设备')
    parser.add_argument('--json',action='store_true',help='输出结构化结果')
    parser.add_argument('--capture-only',action='store_true',help='只检查采集/UI 环境')
    args=parser.parse_args(argv)
    checks=[check_runtime('capture',ROOT/'.venv')]
    if not args.capture_only:checks.append(check_runtime('gsplat',ASCII_GSPLAT_RUNTIME_ROOT))
    result={'workspace':str(ROOT),'ready':all(item['ready'] for item in checks),'runtimes':checks}
    if args.json:print(json.dumps(result,ensure_ascii=False,indent=2))
    else:
        print('工作区：',ROOT)
        for item in checks:
            print(('通过' if item['ready'] else '未通过')+' · '+item['name']+' · '+item['expected'])
            for error in item['errors']:print('  '+error)
        print('检查范围：环境归属与核心模块导入；未启动相机、界面或训练任务。')
    return 0 if result['ready'] else 1


if __name__=='__main__':raise SystemExit(main())
