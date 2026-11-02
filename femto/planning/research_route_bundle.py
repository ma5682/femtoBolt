"""研究用路线归档：保存来源并在打开时重新验证；不是产品方案格式。"""
from pathlib import Path
from unittest.mock import patch
import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from femto.planning import coverage_archive, target_surface
from femto.planning import witness_edge_direct_replay_research as direct
from femto.planning.region_project import open_project

SCHEMA = 'research-route-bundle-v1'
FILES = {
    'old.region.zip': 1024**3, 'base.region.zip': 1024**3,
    'recipe.json': 10*1024**2, 'research_route.json': 10*1024**2,
    'old_proof/coverage_certificate.json': 1024**2,
    'old_proof/coverage_certificate.npz': 256*1024**2,
    'final_proof/coverage_certificate.json': 1024**2,
    'final_proof/coverage_certificate.npz': 256*1024**2,
}
FLAGS = dict(installed_route=False, full_target_coverage_claim=False,
             execution_ready=False, physical_accuracy_verified=False)

def unique(pairs):
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError('重复 JSON 字段')
    return result

def read_json(data):
    return json.loads(data, object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('非法数字')))

def digest(data):
    return hashlib.sha256(data).hexdigest()

def verify(bundle):
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='research-route-') as tmp:
        work = Path(tmp)
        with zipfile.ZipFile(bundle) as archive:
            infos = archive.infolist()
            names = [x.filename for x in infos]
            if len(names) != len(set(names)) or set(names) != set(FILES) | {'manifest.json'}:
                raise ValueError('研究包文件清单不匹配')
            if any(x.flag_bits & 1 or x.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
                   for x in infos):
                raise ValueError('不支持的压缩或加密方式')
            if archive.getinfo('manifest.json').file_size > 1024**2:
                raise ValueError('研究包清单过大')
            manifest = read_json(archive.read('manifest.json'))
            if (not isinstance(manifest, dict) or manifest.get('schema') != SCHEMA
                or any(manifest.get(k) is not v for k, v in FLAGS.items())
                or not isinstance(manifest.get('files'), dict) or set(manifest['files']) != set(FILES)):
                raise ValueError('研究包清单无效')
            for name, limit in FILES.items():
                info = archive.getinfo(name)
                record = manifest['files'][name]
                if (info.file_size > limit or not isinstance(record, dict)
                    or type(record.get('size')) is not int or record['size'] != info.file_size
                    or not isinstance(record.get('sha256'), str)):
                    raise ValueError('研究包条目大小或清单无效：'+name)
                dest = work/name
                dest.parent.mkdir(parents=True, exist_ok=True)
                hasher = hashlib.sha256()
                with archive.open(info) as src, dest.open('xb') as dst:
                    while chunk := src.read(1024*1024):
                        hasher.update(chunk); dst.write(chunk)
                if hasher.hexdigest() != record['sha256']:
                    raise ValueError('研究包条目摘要不匹配：'+name)
        recipe = read_json((work/'recipe.json').read_bytes())
        snapshot = read_json((work/'research_route.json').read_bytes())
        models = []
        for name in ('old', 'base'):
            workspace = work/(name+'_workspace')
            model = open_project(work/(name+'.region.zip'), workspace=workspace)
            if not Path(model.reconstruction).resolve().is_relative_to(workspace.resolve()):
                raise ValueError('工程重建路径不在独立工作区')
            if model.target_ledger is None or model.target_ledger.metadata != model.saved_plan['planning_context']['target']:
                raise ValueError('工程未包含对应的目标表面证据')
            models.append(model)
        old, base = models
        old_proof = coverage_archive.load_assets(work/'old_proof')
        proof = coverage_archive.load_assets(work/'final_proof')
        with patch.object(target_surface, 'read', side_effect=AssertionError('禁止读取外部目标文件')):
            result = direct.replay(base, base.store.regions[0], old.saved_plan, old_proof,
                                   base.saved_plan, recipe, proof, old_model=old,
                                   include_research_route=True)
        if result.pop('research_route') != snapshot:
            raise ValueError('路线与独立重放结果不一致')
        if not result['comparison']['retained_subset']:
            raise ValueError('未保留原已确认范围')
        return dict(verified=True, replay_seconds=time.perf_counter()-started,
                    retained_confirmed_area=True, **FLAGS)

def pack(source, destination):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError('目标已存在，不会覆盖')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.research-bundle-', dir=destination.parent) as tmp:
        stage = Path(tmp)/'bundle.zip'
        records = {}
        with zipfile.ZipFile(stage, 'x', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for name, limit in FILES.items():
                path = source/name
                size = path.stat().st_size
                if size > limit:
                    raise ValueError('输入过大：'+name)
                hasher = hashlib.sha256()
                with path.open('rb') as src, archive.open(name, 'w', force_zip64=True) as dst:
                    while chunk := src.read(1024*1024):
                        hasher.update(chunk); dst.write(chunk)
                records[name] = dict(size=size, sha256=hasher.hexdigest())
            manifest = dict(schema=SCHEMA, files=records, **FLAGS)
            archive.writestr('manifest.json', json.dumps(manifest, ensure_ascii=False))
        report = verify(stage)
        # 原子新建链接，已存在时失败，不覆盖任何文件。
        os.link(stage, destination)
    return report

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    make = commands.add_parser('pack'); make.add_argument('source'); make.add_argument('destination')
    check = commands.add_parser('verify'); check.add_argument('bundle')
    args = parser.parse_args()
    result = pack(args.source, args.destination) if args.command == 'pack' else verify(args.bundle)
    print(json.dumps(result, ensure_ascii=False, indent=2))
